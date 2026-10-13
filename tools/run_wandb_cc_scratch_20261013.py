"""Serial fresh-G1 N20 comparison of 657, K8, and K8+Map64.

The supervising process uses the standard library only. Isolated workers reuse
the existing scratch runner and cached Torch loader; no old search artifacts are
modified. Only complete, native, hash-verified stages enter comparisons.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "output/reports/wandb_cc_20261013/scratch"
RUNS = ROOT / "logs/wandb_cc_scratch_20261013"
OLD_ART = ROOT / "output/reports/init_diffusion_657_edges_20261013"
PYTHON = "/home/ke/miniconda3/envs/sim/bin/python"
PREFIX = "model.model_config.decoder.init_diffusion."
DEADLINE = "2026-10-13T12:48:11+08:00"
STEPS, SCENES, PARAMS = 2048, 4096, 8083978
SEEDS, GENS = [4817, 5817], [2817, 3817]
ARMS = ["fresh657", "freshK8", "freshK8Map64"]
METRICS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
           "width_jsd", "speed_jsd", "collision_rate")
EVAL_MANIFEST = ROOT / "tmp/control657_paired_20261012/eval4096_files.json"
TRAIN_MANIFEST = ROOT / "src/waymo_data/full/training_map2_sd_files.pkl"
NORMALIZER = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json"
MAP = ROOT / "src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt"
SOURCE_FILES = [
    "src/smart/diffusion/denoiser.py", "src/smart/diffusion/initial_diffusion.py",
    "src/smart/diffusion/scale_flow.py", "src/smart/diffusion/diffusion_utils.py",
    "src/smart/diffusion/component_loss.py", "src/smart/diffusion/distribution_loss.py",
    "src/smart/model/smart.py", "src/smart/model/smart_gail.py",
    "src/smart/tokens/token_processor.py", "src/smart/metrics/metric_core.py",
    "src/smart/modules/edge_encoder.py", "src/smart/utils/edge_utils.py",
    "src/smart/datamodules/scalable_datamodule.py", "src/smart/datasets/scalable_dataset.py",
    "configs/run.yaml", "configs/model/smart.yaml", "configs/model/smart_fast.yaml",
    "configs/experiment/init_bc.yaml", "configs/experiment/init_diffusion_lane_conditioned.yaml",
    "configs/experiment/init_diffusion_lane_conditioned_mixed_loss.yaml",
    "configs/optimizer/default.yaml", "tools/overnight_init_diffusion_sweep.py",
    "tools/scratch_component_loss_sweep.py", "tools/runtime_cached_torch_loader_sweep.py",
    "tools/torch_pickle_class_cache.py", "tools/quick_component_loss_sweep.py",
    "tools/run_init_diffusion_657_iteration.py", "tools/run_wandb_cc_scratch_20261013.py",
]
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def deadline_timestamp():
    return datetime.fromisoformat(DEADLINE).timestamp()


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def gpu_pids():
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, check=True)
    return sorted({int(value) for value in query.stdout.splitlines()
                   if value.strip().isdigit() and alive(int(value))})


def hashes():
    return {path: sha(ROOT / path) for path in SOURCE_FILES}


def defaults_proof():
    tree = ast.parse((ROOT / "src/smart/diffusion/initial_diffusion.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InitDiffusion")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    values = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    result = {key: ast.literal_eval(values[key]) for key in
              ("sampling_exact_x0", "warm_start_from_ema", "collision_guidance")}
    if result != {"sampling_exact_x0": False, "warm_start_from_ema": False, "collision_guidance": None}:
        raise RuntimeError("Sampler/warm-start constructor defaults are not OFF")
    return result


def make_variants():
    previous = read(OLD_ART / "variants.json")
    variants = {"fresh657": copy.deepcopy(previous["control657"]),
                "freshK8": copy.deepcopy(previous["a2a_k8"]),
                "freshK8Map64": copy.deepcopy(previous["a2a_k8"])}
    variants["freshK8Map64"]["__config"][PREFIX + "map2agent_max_num_neighbors"] = 64
    for name, variant in variants.items():
        cfg = variant["__config"]
        cfg[PREFIX + "state_feature_normalization"] = "none" if name == "fresh657" else "train"
        cfg[PREFIX + "time_conditioning"] = "input" if name == "fresh657" else "every_layer"
        assert cfg[PREFIX + "sampling_steps"] == 20 and cfg[PREFIX + "denoiser_num_layers"] == 3
        assert cfg[PREFIX + "denoiser_hidden_dim"] == 256 and variant["__lr"] == .0005
        assert set(variant) == {"__config", "__lr"}
    return variants


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    variants, recipe = make_variants(), read(OLD_ART / "recipe.json")
    for name, payload in (("variants.json", variants), ("recipe.json", recipe)):
        path = ART / name
        if path.exists() and read(path) != payload:
            raise RuntimeError("Existing registration differs: " + name)
        if not path.exists():
            save(path, payload)
    with TRAIN_MANIFEST.open("rb") as handle:
        train = pickle.load(handle)["files"]
    evaluation = read(EVAL_MANIFEST)
    if isinstance(evaluation, dict):
        evaluation = evaluation["files"]
    assert len(train) == 493542 and len(set(train)) == len(train)
    assert len(evaluation) == SCENES and len(set(evaluation)) == SCENES
    assert all(Path(name).name == name and (ROOT / "src/waymo_data/full/scenario_dreamer_val" / name).is_file()
               for name in evaluation)
    exposure = STEPS // 964 * len(train) + STEPS % 964 * 512
    assert exposure == 1048524
    source_defaults = defaults_proof()
    plan = {"deadline": DEADLINE, "TRAIN": SEEDS, "GEN": GENS, "arms": ARMS,
            "order": [{"label": f"train{seed}_{arm}", "seed": seed, "arm": arm}
                      for seed in SEEDS for arm in ARMS],
            "steps": STEPS, "scenes": SCENES, "batch_size": 512, "eval_batch_size": 128,
            "checkpoint_steps": [], "sampler_nfe": 20, "guidance": "OFF", "sampling_exact_x0": False,
            "warm_start_from_ema": False, "fresh_G1_and_optimizer_and_EMA": True,
            "actual_expected_generator_parameters": PARAMS, "parameter_cap": 8100000,
            "precision": "32-true", "matmul": "medium", "evaluation_weights": "ema",
            "expected_training_examples": exposure, "full_train_scenes": len(train),
            "equivalent_epochs": exposure / len(train), "no_128epoch_convergence_claim": True,
            "inputs": {"eval_manifest": {"path": str(EVAL_MANIFEST), "sha256": sha(EVAL_MANIFEST),
                                         "ordered_names_sha256": hashlib.sha256(("\n".join(evaluation) + "\n").encode()).hexdigest()},
                       "train_manifest": {"path": str(TRAIN_MANIFEST), "sha256": sha(TRAIN_MANIFEST)},
                       "normalizer": {"path": str(NORMALIZER), "sha256": sha(NORMALIZER)},
                       "fixed_map": {"path": str(MAP), "sha256": sha(MAP)},
                       "recipe": {"path": str(ART / "recipe.json"), "sha256": sha(ART / "recipe.json")},
                       "variants": {"path": str(ART / "variants.json"), "sha256": sha(ART / "variants.json")}},
            "source_defaults_proof": source_defaults,
            "initial_estimate": {"seconds_per_step": .32, "eval_seconds_per_gen": 90, "startup_seconds": 90},
            "selection": "Primary R_jsd=mean of six JSD ratios to fresh657, equal GEN then TRAIN means; report R_geometry(first3) and R_attr(last3), every JSD and per-TRAIN results. Collision is auxiliary, excluded from ranking. Require matched coverage/count and report per-component regressions."}
    assert plan["inputs"]["fixed_map"]["sha256"] == "610c1f94a0f573fd078e251af37c38c73255b969e52154e66546588996ef99fb"
    assert plan["inputs"]["normalizer"]["sha256"] == "adcc5e5c232b018919b269ec06b7e8b065a49fd9f0219f3d0f432f7caf4cea23"
    path = ART / "plan.json"
    if path.exists() and read(path) != plan:
        raise RuntimeError("Existing plan changed")
    if not path.exists():
        save(path, plan)
    return plan


def validate(rows, spec, protocol):
    import yaml
    expected = {(spec["arm"], STEPS, gen) for gen in GENS}
    assert len(rows) == 2 and {(r["variant"], r["budget"], r["eval_seed"]) for r in rows} == expected
    arm_cfg = read(ART / "variants.json")[spec["arm"]]["__config"]
    metadata, resources = [], []
    folder = RUNS / spec["label"] / spec["arm"]
    config = yaml.safe_load((folder / "config.yaml").read_text())
    mc = config["model"]["model_config"]
    decoder = mc["decoder"]["init_diffusion"]
    assert config["ckpt_path"] is None and mc["initial_map_checkpoint"] == str(MAP)
    assert mc["optimizer"] == "default" and mc["lr"] == .0005 and mc["lr_min_ratio"] == .05
    assert decoder.get("sampling_exact_x0", False) is False
    assert decoder.get("warm_start_from_ema", False) is False
    assert not decoder.get("collision_guidance") or decoder["collision_guidance"].get("scale", 0) == 0
    assert decoder["component_loss"] == read(ART / "recipe.json")
    initialization = read(folder / "initialization.json")
    assert initialization["resumed_from"] is None and initialization["optimizer_continued"] is False
    assert initialization["actual_train_offset"] == 0 and initialization["resume_global_step"] == 0
    assert initialization["initialization_scope"] == "frozen_pretrained_map_fresh_G1"
    assert read(folder / "train_summary.json")["steps"] == STEPS
    with (folder / "csv/version_0/metrics.csv").open(newline="") as handle:
        telemetry = list(csv.DictReader(handle))
    assert telemetry and max(int(row["step"]) for row in telemetry) == STEPS - 1
    for row in rows:
        report = read(row["metrics_path"])
        assert row["seed"] == spec["seed"] and row["batch_size"] == 512
        assert row["generator_num_parameters"] == row["generator_trainable_parameters"] == PARAMS
        assert row["frozen_map_num_parameters"] == 437312 and row["map_frozen"] is True
        assert row["seen_training_examples"] == protocol["plan"]["expected_training_examples"]
        assert row["evaluation_only"] is False and row["evaluation_weights"] == "ema"
        assert row["calibration_sha256"] == protocol["plan"]["inputs"]["normalizer"]["sha256"]
        assert row["map_checkpoint_sha256"] == protocol["plan"]["inputs"]["fixed_map"]["sha256"]
        assert row["initial_map_sha256"] == "09b7f55a4c78816794dd5ffcdce36760af5aa5afe038a1bf8cd34f168c51f720"
        assert row["eval_manifest_sha256"] == protocol["plan"]["inputs"]["eval_manifest"]["ordered_names_sha256"]
        assert report["num_samples"] == report["num_gt_samples"] == SCENES
        assert report["reference_mode"] == "matched" and report["map_source"] == "reference"
        assert report["unverified_generation_timestep_count"] == 0
        assert report["runtime"]["trainer_precision"] == "32-true"
        assert report["runtime"]["float32_matmul_precision"] == "medium"
        assert report["runtime"]["torch_initial_seed"] == row["eval_seed"]
        actual = report["decoder"]
        assert actual["sampling_steps"] == 20 and actual["use_ema"] is True and actual["ema_num_updates"] == STEPS
        assert actual["init_map_crop"] == "square" and actual["init_map_half_extent_m"] == 50.
        for field in ("position", "heading", "shape", "type"):
            assert actual["fix_ego_" + field] is True
        assert actual["fix_ego_velocity"] is False and actual["generate_type"] is True
        assert actual["type_process"] == "separate" and actual["velocity_representation"] == "speed"
        assert actual["heading_noise"] == "circular" and actual["heading_objective"] == "x0"
        registered_loss = read(ART / "recipe.json")
        actual_loss = copy.deepcopy(actual["component_loss"])
        extra = actual_loss["components"].pop("speed_magnitude", None)
        assert extra is None or extra == {"kind": "mse", "time_power": 0, "weight": 0.0, "scale": "data"}
        assert actual_loss == registered_loss
        for key, value in arm_cfg.items():
            option = key[len(PREFIX):]
            if key.startswith(PREFIX) and option in actual["denoiser_options"]:
                assert actual["denoiser_options"][option] == value, option
        for key in METRICS:
            assert math.isfinite(row[key]) and row[key] == report["agent_metrics"][key]
        metadata.append({"path": row["metrics_path"], "sha256": sha(row["metrics_path"])})
    for path in (folder / "config.yaml", folder / "initialization.json", folder / "train_summary.json",
                 folder / "csv/version_0/metrics.csv", folder / "last.ckpt"):
        resources.append({"path": str(path), "sha256": sha(path)})
    return {"native_reports": metadata, "training_evidence": resources}


def worker(spec_path):
    spec = read(spec_path)
    from tools import run_init_diffusion_657_iteration as legacy
    legacy.ART, legacy.RUNS = ART, RUNS
    rows = legacy.stage(spec["label"], [spec["arm"]], STEPS, SCENES,
                        seed=spec["seed"], checkpoints=(), gens=tuple(GENS))
    validate(rows, spec, read(ART / "protocol.json"))
    print("WAND_B_CC_SCRATCH_WORKER_SUCCESS", flush=True)


def refresh(rows):
    from tools.run_init_diffusion_657_iteration import score
    controls = [row for row in rows if row["variant"] == "fresh657"]
    results = []
    for arm in ARMS[1:]:
        candidate = [row for row in rows if row["variant"] == arm]
        available = [row for row in candidate if any(base["seed"] == row["seed"]
                     and base["eval_seed"] == row["eval_seed"] for base in controls)]
        if not available:
            continue
        # Paired model initialization is required in addition to score's data/seed checks.
        for row in available:
            base = next(base for base in controls if base["seed"] == row["seed"] and base["eval_seed"] == row["eval_seed"])
            assert row["initial_generator_sha256"] == base["initial_generator_sha256"]
        result = {"arm": arm, "complete_two_TRAIN_grid": len(available) == 4,
                  "observed_cells": len(available), **score(available, controls)}
        for item in [result, *result.get("per_TRAIN", {}).values()]:
            item["R_jsd"] = sum(item["ratios"][key] for key in METRICS[:6]) / 6
            item["R_geometry"] = sum(item["ratios"][key] for key in METRICS[:3]) / 3
        result["each_TRAIN_jsd_improves"] = all(item["R_jsd"] < 1 for item in result.get("per_TRAIN", {}).values())
        results.append(result)
    save(ART / "comparisons.json", {"generated_utc": now(), "comparisons": results,
          "native_128epoch_verified": False, "guidance": "OFF", "sampling_steps": 20,
          "primary_metric": "R_jsd", "collision_in_primary_metric": False})


def on_signal(_number, _frame):
    global STOP
    STOP = True


def execute(plan):
    lock = (ART / "driver.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = ART / "protocol.json"
    expected = {"plan": plan, "plan_sha256": sha(ART / "plan.json"), "sources": hashes()}
    if protocol_path.exists():
        protocol = read(protocol_path)
        assert all(protocol[key] == value for key, value in expected.items()), "Frozen registration changed"
    else:
        protocol = dict(expected, registered_utc=now(), driver_pid=os.getpid(),
                        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        save(protocol_path, protocol)
    rows, durations, failures = [], [], []
    def status(state, **detail):
        save(ART / "status.json", {"status": state, "driver_pid": os.getpid(), "updated_utc": now(),
             "deadline": DEADLINE, "remaining_seconds": max(0, deadline_timestamp() - time.time()),
             "observations": len(rows), **detail})
    def frozen():
        assert hashes() == protocol["sources"], "Source changed during registered execution"
        for value in plan["inputs"].values():
            assert sha(value["path"]) == value["sha256"], "Input changed during registered execution"
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    for spec in plan["order"]:
        frozen()
        folder = RUNS / spec["label"]
        marker = folder / "scratch_verified_completion.json"
        if marker.exists():
            record = read(marker)
            assert record["protocol_sha256"] == sha(protocol_path)
            assert record["summary_sha256"] == sha(folder / "summary.json")
            values = read(folder / "summary.json")
            assert record["verification"] == validate(values, spec, protocol)
            rows.extend(values)
            save(ART / "observations.json", rows)
            refresh(rows)
            continue
        if folder.exists() and any(folder.iterdir()):
            status("INCOMPLETE_STAGE_REQUIRES_FRESH_REGISTRATION", active_stage=spec["label"], partial_usable=False)
            return
        estimate = max(durations or [.32 * STEPS + 90 * 2 + 90]) * 1.1
        if STOP or time.time() + estimate > deadline_timestamp():
            status("DEADLINE_NO_NEW_STAGE", next_stage=spec, estimated_seconds=estimate)
            return
        if gpu_pids():
            status("WAITING_FOR_UNRELATED_GPU", active_stage=spec["label"], gpu_pids=gpu_pids())
            while gpu_pids() and not STOP and time.time() + estimate < deadline_timestamp():
                time.sleep(5)
            if STOP or time.time() + estimate > deadline_timestamp():
                status("DEADLINE_AFTER_GPU_WAIT", next_stage=spec)
                return
        folder.mkdir(parents=True)
        spec_path = ART / (spec["label"] + ".spec.json")
        save(spec_path, spec)
        save(folder / "scratch_contract.json", {"spec": spec, "protocol_sha256": sha(protocol_path),
                                                "source_hashes": protocol["sources"]})
        command = [PYTHON, str(Path(__file__).resolve()), "--worker", str(spec_path)]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONFAULTHANDLER="1")
        started = time.monotonic()
        with (folder / "driver_console.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            status("RUNNING", active_stage=spec["label"], child_pid=child.pid, command=command)
            last_heartbeat = time.monotonic()
            while child.poll() is None:
                if STOP or time.time() >= deadline_timestamp():
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    status("DEADLINE_PARTIAL_STAGE_EXCLUDED", active_stage=spec["label"], partial_usable=False)
                    return
                if time.monotonic() - last_heartbeat >= 30:
                    status("RUNNING", active_stage=spec["label"], child_pid=child.pid,
                           elapsed_seconds=time.monotonic() - started)
                    last_heartbeat = time.monotonic()
                time.sleep(2)
        elapsed = time.monotonic() - started
        if child.returncode != 0 or "WAND_B_CC_SCRATCH_WORKER_SUCCESS" not in (folder / "driver_console.log").read_text():
            failures.append({"stage": spec["label"], "returncode": child.returncode, "partial_usable": False,
                             "driver_console": str(folder / "driver_console.log")})
            save(ART / "failures.json", failures)
            status("FAILED_NATIVE_STAGE", active_stage=spec["label"], returncode=child.returncode)
            return
        native_marker = read(folder / "verified_completion.json")
        assert native_marker["status"] == "VERIFIED_COMPLETE" and native_marker["returncode"] == 0
        assert native_marker["summary_sha256"] == sha(folder / "summary.json")
        log = (folder / "console.log").read_text()
        assert "RUNTIME_LOADER_SUCCESS" in log and f"DONE {folder}" in log
        frozen()
        values = read(folder / "summary.json")
        record = {"status": "VERIFIED_COMPLETE", "finished_utc": now(), "returncode": 0,
                  "protocol_sha256": sha(protocol_path), "summary_sha256": sha(folder / "summary.json"),
                  "verification": validate(values, spec, protocol), "elapsed_seconds": elapsed,
                  "worker_console_sha256": sha(folder / "driver_console.log"),
                  "native_console_sha256": sha(folder / "console.log")}
        save(marker, record)
        rows.extend(values)
        durations.append(elapsed)
        save(ART / "observations.json", rows)
        save(ART / "durations.json", durations)
        refresh(rows)
        status("STAGE_COMPLETE", active_stage=spec["label"], elapsed_seconds=elapsed)
    assert len(rows) == 12
    status("COMPLETE", complete_stages=6, native_cells=12,
           warning="2048-update from-scratch evidence; does not prove128epoch convergence")
    print("ALL_WANDB_CC_SCRATCH_STAGES_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    plan = prepare()
    if not args.execute:
        print(json.dumps({"status": "PREPARED_NOT_LAUNCHED", "plan": str(ART / "plan.json"),
                          "plan_sha256": sha(ART / "plan.json"), "stages": len(plan["order"]),
                          "torch_imported": "torch" in sys.modules, "deadline": DEADLINE}, indent=2))
        return
    execute(plan)


if __name__ == "__main__":
    main()
