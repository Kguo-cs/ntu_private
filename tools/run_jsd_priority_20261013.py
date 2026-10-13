"""JSD-first continuation of the authorized two-hour, N20 scratch search.

Preserves the original registration. Only unstarted work is redirected; the
first completed 657 control is imported with its original verification marker.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_wandb_cc_scratch_20261013 as original
from tools import run_init_diffusion_657_iteration as legacy

read, save, sha, now = original.read, original.save, original.sha, original.now
ART = ROOT / "output/reports/wandb_cc_20261013/jsd_priority"
RUNS = ROOT / "logs/wandb_cc_jsd_priority_20261013"
PREFIX, METRICS = original.PREFIX, original.METRICS
BASE_CONTROL = original.RUNS / "train4817_fresh657"
STOP = False
ORDER = [
    {"arm": "fresh657", "seed": 5817, "steps": 4096, "checkpoints": [2048]},
    {"arm": "lane_agentmax05", "seed": 5817, "steps": 4096, "checkpoints": [2048]},
    {"arm": "freshK8", "seed": 4817, "steps": 2048, "checkpoints": []},
    {"arm": "lane_only", "seed": 4817, "steps": 2048, "checkpoints": []},
    {"arm": "freshK8Map64", "seed": 4817, "steps": 2048, "checkpoints": []},
]
for item in ORDER:
    item["label"] = f'train{item["seed"]}_{item["arm"]}'


def source_hashes():
    files = original.SOURCE_FILES + ["src/smart/modules/map_decoder.py",
                                   "tools/run_jsd_priority_20261013.py"]
    return {p: sha(ROOT / p) for p in files}


def effective_recipe(arm):
    result = read(ART / "recipe.json")
    if arm == "lane_agentmax05":
        result["components"]["collision"]["weight"] = .5
    return result


def prepare():
    variants = original.make_variants()
    variants["lane_only"] = copy.deepcopy(variants["freshK8"])
    variants["lane_only"]["__config"][PREFIX + "map_condition_types"] = "lanes_and_edges"
    variants["lane_agentmax05"] = copy.deepcopy(variants["lane_only"])
    variants["lane_agentmax05"]["__config"][PREFIX + "collision_reduction"] = "agent_max"
    variants["lane_agentmax05"]["collision"] = {"weight": .5}
    plan = {
        "registered_utc": now(), "deadline": original.DEADLINE, "order": ORDER,
        "primary": "R_jsd: equal-GEN then equal-TRAIN metric means; mean six candidate/control JSD ratios",
        "collision_in_ranking": False, "report_each_jsd": True,
        "report_R_geometry_and_R_attr": True,
        "require_onroad_coverage_and_vehicle_count": True,
        "reason": "User explicitly prioritizes JSD; historical lane_agentmax05 beats K8 on six-JSD mean but loses when collision enters R7. Separate lane-only input, then replicate combined candidate at another TRAIN seed.",
        "historical_selection_source": {"path": "output/reports/init_diffusion_657_compact_20261013/live_report.json",
                                        "sha256": sha(ROOT / "output/reports/init_diffusion_657_compact_20261013/live_report.json")},
        "historical_selection_is_posthoc": True,
        "fresh_G1_optimizer_EMA": True, "guidance": "OFF", "sampling_steps": 20,
        "parameter_cap": 8100000, "generator_parameters": 8083978,
        "GEN": original.GENS, "eval_scenes": 4096, "batch_size": 512,
        "precision": "32-true", "matmul": "medium", "evaluation_weights": "ema",
        "imported_control": str(BASE_CONTROL), "original_protocol_sha256": sha(original.ART / "protocol.json"),
        "source_defaults": original.defaults_proof(),
        "sources": source_hashes(),
        "inputs": read(original.ART / "plan.json")["inputs"],
        "no_full_convergence_claim": True,
        "stop_policy": "Preserve complete native stages; deadline guard may leave lower-priority ablations unrun; no unpaired result promoted.",
    }
    for name, value in (("variants.json", variants), ("recipe.json", read(original.ART / "recipe.json"))):
        path = ART / name
        if path.exists():
            assert read(path) == value
        else:
            save(path, value)
    plan["local_inputs"] = {name: sha(ART / name) for name in ("variants.json", "recipe.json")}
    path = ART / "protocol.json"
    if path.exists():
        previous = read(path)
        plan["registered_utc"] = previous["registered_utc"]
        assert previous == plan, "Registration changed"
    else:
        save(path, plan)
    return plan


def frozen(plan):
    assert source_hashes() == plan["sources"], "Registered source changed"
    for value in plan["inputs"].values():
        assert sha(value["path"]) == value["sha256"]
    for name, digest in plan["local_inputs"].items():
        assert sha(ART / name) == digest


def verify(rows, spec):
    import yaml
    budgets = spec["checkpoints"] + [spec["steps"]]
    assert len(rows) == len(budgets) * 2
    assert {(r["budget"], r["eval_seed"]) for r in rows} == {(b, g) for b in budgets for g in original.GENS}
    folder = RUNS / spec["label"] / spec["arm"]
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    mc = cfg["model"]["model_config"]
    decoder = mc["decoder"]["init_diffusion"]
    assert cfg["ckpt_path"] is None and mc["initial_map_checkpoint"] == str(original.MAP)
    assert decoder["component_loss"] == effective_recipe(spec["arm"])
    assert mc["optimizer"] == "default" and mc["lr"] == .0005 and mc["lr_min_ratio"] == .05
    assert not decoder.get("sampling_exact_x0", False) and not decoder.get("warm_start_from_ema", False)
    assert not decoder.get("collision_guidance") or decoder["collision_guidance"].get("scale", 0) == 0
    init = read(folder / "initialization.json")
    assert init["resumed_from"] is None and not init["optimizer_continued"]
    assert init["actual_train_offset"] == init["resume_global_step"] == 0
    assert read(folder / "train_summary.json")["steps"] == spec["steps"]
    with (folder / "csv/version_0/metrics.csv").open() as stream:
        assert max(int(r["step"]) for r in csv.DictReader(stream)) == spec["steps"] - 1
    reports = []
    expected_cfg = read(ART / "variants.json")[spec["arm"]]["__config"]
    for row in rows:
        b = row["budget"]
        assert row["variant"] == spec["arm"] and row["seed"] == spec["seed"]
        assert row["generator_num_parameters"] == row["generator_trainable_parameters"] == 8083978
        assert row["batch_size"] == 512 and row["map_frozen"]
        assert row["seen_training_examples"] == b // 964 * 493542 + b % 964 * 512
        assert not row["evaluation_only"] and row["evaluation_weights"] == "ema"
        assert row["calibration_sha256"] == sha(original.NORMALIZER)
        assert row["map_checkpoint_sha256"] == read(original.ART / "plan.json")["inputs"]["fixed_map"]["sha256"]
        assert row["eval_manifest_sha256"] == read(original.ART / "plan.json")["inputs"]["eval_manifest"]["ordered_names_sha256"]
        report = read(row["metrics_path"])
        assert report["num_samples"] == report["num_gt_samples"] == 4096
        assert report["map_source"] == "reference" and report["reference_mode"] == "matched"
        assert report["unverified_generation_timestep_count"] == 0
        assert report["runtime"]["trainer_precision"] == "32-true"
        assert report["runtime"]["float32_matmul_precision"] == "medium"
        assert report["runtime"]["torch_initial_seed"] == row["eval_seed"]
        actual = report["decoder"]
        assert actual["sampling_steps"] == 20 and actual["ema_num_updates"] == b and actual["use_ema"]
        assert actual["init_map_half_extent_m"] == 50 and actual["init_map_crop"] == "square"
        for field in ("position", "heading", "shape", "type"):
            assert actual["fix_ego_" + field]
        assert not actual["fix_ego_velocity"] and actual["generate_type"]
        assert actual["type_process"] == "separate" and actual["velocity_representation"] == "speed"
        loss = copy.deepcopy(actual["component_loss"])
        extra = loss["components"].pop("speed_magnitude", None)
        assert extra is None or extra == {"kind": "mse", "time_power": 0, "weight": 0.0, "scale": "data"}
        assert loss == effective_recipe(spec["arm"])
        for key, value in expected_cfg.items():
            if key.startswith(PREFIX):
                option = key[len(PREFIX):]
                assert decoder[option] == value
                if option in actual.get("denoiser_options", {}):
                    assert actual["denoiser_options"][option] == value
        for key in METRICS:
            assert math.isfinite(row[key]) and row[key] == report["agent_metrics"][key]
        reports.append({"path": row["metrics_path"], "sha256": sha(row["metrics_path"])})
    return {"native_reports": reports, "training": {p: sha(folder / p) for p in
            ("config.yaml", "initialization.json", "train_summary.json", "csv/version_0/metrics.csv", "last.ckpt")}}


def worker(spec_path):
    spec = read(spec_path)
    legacy.ART, legacy.RUNS = ART, RUNS
    rows = legacy.stage(spec["label"], [spec["arm"]], spec["steps"], 4096,
                        seed=spec["seed"], checkpoints=tuple(spec["checkpoints"]), gens=tuple(original.GENS))
    verify(rows, spec)
    print("JSD_PRIORITY_WORKER_SUCCESS", flush=True)


def refresh(rows):
    results = []
    controls = [r for r in rows if r["variant"] == "fresh657"]
    for arm in sorted({r["variant"] for r in rows} - {"fresh657"}):
        for budget in sorted({r["budget"] for r in rows if r["variant"] == arm}):
            candidates = [r for r in rows if r["variant"] == arm and r["budget"] == budget]
            candidates = [r for r in candidates if any(c["seed"] == r["seed"] and c["budget"] == budget
                           and c["eval_seed"] == r["eval_seed"] for c in controls)]
            if not candidates:
                continue
            for row in candidates:
                control = next(c for c in controls if (c["seed"], c["budget"], c["eval_seed"]) == (row["seed"], budget, row["eval_seed"]))
                assert row["initial_generator_sha256"] == control["initial_generator_sha256"]
            score = legacy.score(candidates, controls)
            for value in [score, *score["per_TRAIN"].values()]:
                value["R_jsd"] = sum(value["ratios"][k] for k in METRICS[:6]) / 6
                value["R_geometry"] = sum(value["ratios"][k] for k in METRICS[:3]) / 3
            results.append({"arm": arm, "budget": budget, **score})
    save(ART / "comparisons.json", {"primary_metric": "R_jsd", "collision_in_ranking": False,
                                   "generated_utc": now(), "comparisons": results})


def on_signal(_number, _frame):
    global STOP
    STOP = True


def execute(plan):
    lock = (ART / "driver.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    def status(state, **kw):
        save(ART / "status.json", {"status": state, "driver_pid": os.getpid(), "updated_utc": now(), **kw})
    marker = BASE_CONTROL / "scratch_verified_completion.json"
    while not marker.exists() and not STOP:
        state = read(original.ART / "status.json")
        if state["status"] in ("FAILED_NATIVE_STAGE", "DEADLINE_PARTIAL_STAGE_EXCLUDED"):
            raise RuntimeError("Imported control failed")
        status("WAITING_FOR_IMPORTED_CONTROL")
        time.sleep(5)
    assert not STOP
    record = read(marker)
    assert record["protocol_sha256"] == plan["original_protocol_sha256"]
    rows = read(BASE_CONTROL / "summary.json")
    assert record["summary_sha256"] == sha(BASE_CONTROL / "summary.json")
    assert record["verification"] == original.validate(rows, {"arm": "fresh657", "seed": 4817, "label": "train4817_fresh657"}, read(original.ART / "protocol.json"))
    save(ART / "imported_control.json", {"marker": str(marker), "sha256": sha(marker)})
    save(ART / "observations.json", rows)
    for spec in plan["order"]:
        frozen(plan)
        folder = RUNS / spec["label"]
        marker = folder / "jsd_verified_completion.json"
        if marker.exists():
            rec = read(marker)
            values = read(folder / "summary.json")
            assert rec["protocol_sha256"] == sha(ART / "protocol.json")
            assert rec["summary_sha256"] == sha(folder / "summary.json")
            assert rec["verification"] == verify(values, spec)
            rows.extend(values)
            refresh(rows)
            continue
        assert not folder.exists(), "Inspect incomplete stage before reusing output"
        estimate = .43 * spec["steps"] + 150 + 50 * len(spec["checkpoints"])
        if STOP or time.time() + estimate > original.deadline_timestamp():
            status("NO_NEW_STAGE_BEFORE_DEADLINE", next_stage=spec, estimated_seconds=estimate)
            return
        while original.gpu_pids() and not STOP:
            status("WAITING_FOR_GPU", gpu_pids=original.gpu_pids())
            if time.time() + estimate > original.deadline_timestamp():
                status("NO_TIME_AFTER_GPU_WAIT", next_stage=spec)
                return
            time.sleep(5)
        assert not STOP
        spec_path = ART / (spec["label"] + ".spec.json")
        save(spec_path, spec)
        folder.mkdir(parents=True)
        command = [original.PYTHON, str(Path(__file__).resolve()), "--worker", str(spec_path)]
        started = time.monotonic()
        with (folder / "driver_console.log").open("w") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            heartbeat = 0
            while child.poll() is None:
                if STOP or time.time() >= original.deadline_timestamp():
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    status("PARTIAL_STAGE_EXCLUDED", active_stage=spec, returncode=child.returncode)
                    return
                if time.monotonic() - heartbeat > 30:
                    status("RUNNING", active_stage=spec, child_pid=child.pid, elapsed=time.monotonic()-started)
                    heartbeat = time.monotonic()
                time.sleep(2)
        if child.returncode != 0 or "JSD_PRIORITY_WORKER_SUCCESS" not in (folder / "driver_console.log").read_text():
            status("FAILED_STAGE", active_stage=spec, returncode=child.returncode)
            return
        frozen(plan)
        native = read(folder / "verified_completion.json")
        assert native["status"] == "VERIFIED_COMPLETE" and native["returncode"] == 0
        assert native["summary_sha256"] == sha(folder / "summary.json")
        values = read(folder / "summary.json")
        save(marker, {"status": "VERIFIED_COMPLETE", "returncode": 0, "finished_utc": now(),
                      "protocol_sha256": sha(ART / "protocol.json"), "summary_sha256": sha(folder / "summary.json"),
                      "elapsed_seconds": time.monotonic()-started, "verification": verify(values, spec)})
        rows.extend(values)
        save(ART / "observations.json", rows)
        refresh(rows)
        status("STAGE_COMPLETE", active_stage=spec, observations=len(rows))
    status("COMPLETE", observations=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        plan = prepare()
        if args.execute:
            execute(plan)
        else:
            print(json.dumps({"status": "REGISTERED_NOT_EXECUTED", "protocol": str(ART / "protocol.json"),
                              "sha256": sha(ART / "protocol.json"), "torch_imported": "torch" in sys.modules}))
