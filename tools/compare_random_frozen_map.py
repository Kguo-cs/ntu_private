"""Prepare or explicitly run a separate current-code map-policy comparison.

Default is CPU audit and command materialization only.  Missing checkpoints
fail immediately; this program never waits for or modifies the search queue.
Only --run executes six serialized evaluator invocations.  Each invocation
uses the old heldout4096 and two GEN seeds, for all three TRAIN seeds.  The
old warm_winner is deliberately excluded because it has a different loss.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from datetime import datetime

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_frozen_map_optional_scratch_comparison import canonical_recipe, leaf_differences
from tools.analyze_frozen_map_search import MULTIPLIERS, SCALES, stats, training_seed_means

PYTHON = "/home/ke/miniconda3/envs/sim/bin/python"
ART = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
OLD_ART = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
OLD_RUNS = ROOT / "logs/component_loss_scratch"
NEW_RUNS = ROOT / "logs/component_loss_frozen_map"
OLD_GEO = "geo_mse32_col20_heading_l1_shape2"
SEEDS = (817, 1817, 2817)
GENERATIONS = (2817, 3817)
MAP = ROOT / "src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt"
HELDOUT_JSON_SHA256 = "312c2bbbe12b645eb429519db39b5ca5a6155aa58eb7f99a707072d07da04a00"
HELDOUT_ORDER_SHA256 = "e8f8048f505301fde1522ad4826c2997372635daf63aaa91957d68b5fe77e7e1"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state, prefix, torch):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith(prefix) and torch.is_tensor(tensor):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_paths(seed):
    old = OLD_RUNS / f"batch512_seed{seed}" / OLD_GEO / "step_2048.ckpt"
    stage = "long_seed817_new_control" if seed == 817 else f"confirm_seed{seed}_new_control"
    return old, NEW_RUNS / stage / "new_control/step_2048.ckpt"


def canonical_model_config(config):
    value = copy.deepcopy(config["model"]["model_config"])
    value.pop("initial_map_checkpoint", None)
    value.pop("finetune", None)
    value["decoder"]["init_diffusion"]["component_loss"] = canonical_recipe(value["decoder"]["init_diffusion"]["component_loss"])
    return value


def canonical_training_config(config):
    """Ignore output destinations and declared map policy, preserve semantics."""
    value = copy.deepcopy(config)
    value["model"]["model_config"] = canonical_model_config(config)
    for key in ("log_dir", "output_dir"):
        value.get("paths", {}).pop(key, None)
    value.get("trainer", {}).pop("default_root_dir", None)
    value.get("logger", {}).get("wandb", {}).pop("save_dir", None)
    for key in ("scenario_dreamer_eval_set", "scenario_dreamer_train_sample_list", "num_workers"):
        value.get("data", {}).pop(key, None)
    return value


def ordered_files(path):
    with Path(path).open("rb") as handle:
        files = pickle.load(handle)
    files = files["files"] if isinstance(files, dict) else files
    return files, hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()


def metadata(path):
    folder = path.parent
    config = yaml.safe_load((folder / "config.yaml").read_text())
    init = read(folder / "initialization.json")
    protocol = read(folder.parent / "protocol.json")
    train_order, order_hash = ordered_files(folder.parent / "train_order.pkl")
    if order_hash != init["train_order_sha256"] or len(train_order) != 493542:
        raise ValueError(f"Actual TRAIN order differs from recorded provenance: {folder}")
    if int(init.get("resume_global_step") or 0) != 0 or init.get("optimizer_continued") or init.get("resumed_from"):
        raise ValueError(f"Optional comparison requires actual fresh-training arms: {folder}")
    arm_files, arm_hash = ordered_files(folder / "train_files.pkl")
    if arm_hash != init["train_manifest_sha256"] or arm_files != train_order:
        raise ValueError(f"Actual consumed TRAIN manifest differs from full fresh order: {folder}")
    return {"folder": str(folder), "checkpoint": str(path), "config": config, "initialization": init,
            "protocol": protocol, "train_examples": len(train_order), "actual_train_order_sha256": order_hash}


def verify_metadata_pair(old, new, seed, calibration_hash, map_hash):
    a, b = old["initialization"], new["initialization"]
    checks = {key: a.get(key) == b.get(key) for key in (
        "seed", "batch_size", "lr", "train_order_sha256", "train_manifest_sha256",
        "calibration_sha256", "initial_generator_sha256", "base_train_offset", "actual_train_offset")}
    checks.update({"expected_seed": a["seed"] == b["seed"] == seed,
                   "expected_B512": a["batch_size"] == b["batch_size"] == 512,
                   "expected_lr5e4": a["lr"] == b["lr"] == .0005,
                   "expected_calibration_file": a["calibration_sha256"] == b["calibration_sha256"] == calibration_hash,
                   "random_old_map_policy": a["initialization_scope"] == "map_encoder_and_G1_from_scratch",
                   "fixed_new_map_policy": b["initialization_scope"] == "frozen_pretrained_map_fresh_G1" and b.get("map_frozen") is True,
                   "new_source_map_file_hash": b.get("map_checkpoint_sha256") == map_hash,
                   "effective_model_config": canonical_model_config(old["config"]) == canonical_model_config(new["config"]),
                   "effective_training_config": canonical_training_config(old["config"]) == canonical_training_config(new["config"]),
                   "old_config_map_unfrozen": old["config"]["model"]["model_config"]["finetune"] is False and not old["config"]["model"]["model_config"].get("initial_map_checkpoint"),
                   "new_config_source_map": new["config"]["model"]["model_config"]["finetune"] is True and Path(new["config"]["model"]["model_config"].get("initial_map_checkpoint", "")).resolve() == MAP.resolve(),
                   "same_clip": old["config"]["trainer"]["gradient_clip_val"] == new["config"]["trainer"]["gradient_clip_val"] == .5,
                   "same_precision": old["protocol"]["precision"] == new["protocol"]["precision"] == "32-true",
                   "same_matmul_precision": old["protocol"]["float32_matmul_precision"] == new["protocol"]["float32_matmul_precision"] == "medium",
                   "same_square50": old["protocol"]["map_crop"] == new["protocol"]["map_crop"] == "square" and old["protocol"]["map_half_extent"] == new["protocol"]["map_half_extent"] == 50.})
    return {"seed": seed, "checks": checks, "passed": all(checks.values()),
            "effective_model_config_differences": leaf_differences(canonical_model_config(old["config"]), canonical_model_config(new["config"])),
            "effective_training_config_differences": leaf_differences(canonical_training_config(old["config"]), canonical_training_config(new["config"])),
            "old_num_workers": old["config"]["data"]["num_workers"], "new_num_workers": new["config"]["data"]["num_workers"],
            "evaluation_workers": 0,
            "batch": 512, "TRAIN_order_sha256": a["train_order_sha256"],
            "initial_G1_sha256": a["initial_generator_sha256"], "calibration_sha256": a["calibration_sha256"]}


def verify_checkpoint(item, *, policy, calibration, torch):
    path = Path(item["checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    extra = state["encoder.init_decoder._extra_state"]
    ema = extra.get("ema") or {}
    init, saved = item["initialization"], checkpoint.get("scratch_sweep") or {}
    metadata_keys = ("seed", "batch_size", "base_train_offset", "train_order_sha256", "calibration_sha256", "initial_generator_sha256", "initial_map_sha256")
    checks = {"budget2048": checkpoint["global_step"] == 2048,
              "epoch2": checkpoint["epoch"] == 2,
              "ema2048": ema.get("num_updates") == 2048,
              "ema_decay9999": ema.get("decay") == .9999,
              "saved_provenance": all(saved.get(key) == init.get(key) for key in metadata_keys),
              "canonical_loss": canonical_recipe(extra["component_loss"]) == canonical_recipe(item["config"]["model"]["model_config"]["decoder"]["init_diffusion"]["component_loss"])}
    checks["finite_map_and_generator_parameters"] = all(bool(torch.isfinite(value).all()) for name, value in state.items()
                                                        if name.startswith(("encoder.map_encoder.", "encoder.init_decoder.G1.")) and torch.is_tensor(value))
    for key in ("normal_mean", "normal_scale"):
        actual = state[f"encoder.init_decoder.G1.model.{key}"]
        expected = torch.as_tensor(calibration[key], dtype=actual.dtype)
        checks[key] = actual.shape == expected.shape and torch.equal(actual, expected) and bool(torch.isfinite(actual).all())
    map_hash = state_hash(state, "encoder.map_encoder.", torch)
    checks["map_policy_checkpoint_state"] = map_hash == init["initial_map_sha256"] if policy == "frozen" else map_hash != init["initial_map_sha256"]
    optimizers = checkpoint.get("optimizer_states") or []
    if len(optimizers) != 1:
        raise ValueError(f"Expected one actual AdamW optimizer: {path}")
    optimizer = optimizers[0]
    updates = sorted({int(value["step"]) for value in optimizer["state"].values()})
    groups = [{key: value for key, value in group.items() if key != "params"} for group in optimizer["param_groups"]]
    checks["adam_all_steps2048"] = updates == [2048]
    checks["adam_defaults"] = all(group.get("betas") == (.9, .999) and group.get("eps") == 1e-8 and group.get("weight_decay") == .01 for group in groups)
    schedulers = checkpoint.get("lr_schedulers") or []
    checks["epoch_scheduler2"] = len(schedulers) == 1 and schedulers[0].get("last_epoch") == 2
    summary_path = Path(item["folder"]).parent / "summary.json"
    summaries = read(summary_path) if summary_path.is_file() else []
    exposure_rows = [row for row in summaries if row.get("variant") == Path(item["folder"]).name and row.get("budget") == 2048]
    seen = sorted({row.get("seen_training_examples") for row in exposure_rows})
    checks["actual_checkpoint_exposure_recorded"] = seen == [1048524]
    checks["finite_ema_parameters"] = bool(ema.get("shadow_params")) and all(bool(torch.isfinite(value).all()) for value in ema.get("shadow_params", []))
    result = {"checkpoint": str(path), "policy": policy, "checks": checks, "passed": all(checks.values()),
              "checkpoint_file_sha256": hash_file(path),
              "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
              "ema_updates": ema.get("num_updates"), "ema_decay": ema.get("decay"),
              "optimizer_state_tensor_count": len(optimizer["state"]), "optimizer_update_values": updates,
              "optimizer_param_group_settings": groups, "actual_exposure": seen,
              "saved_map_state_sha256": map_hash, "saved_G1_state_sha256": state_hash(state, "encoder.init_decoder.G1.", torch),
              "normalizer_exact_match": checks["normal_mean"] and checks["normal_scale"]}
    del checkpoint, state, extra, ema, optimizer
    gc.collect()
    return result


def checkpoint_metric_metadata(item):
    """Inspect saved evaluation provenance; historical metrics are never reused."""
    summaries = read(Path(item["folder"]).parent / "summary.json")
    rows = [row for row in summaries if row.get("variant") == Path(item["folder"]).name
            and row.get("budget") == 2048 and row.get("evaluation_weights") == "ema"]
    if not rows:
        raise ValueError(f"No actual2048 EMA metric provenance: {item['folder']}")
    report = read(rows[0]["metrics_path"])
    keys = ("metric_scope", "generation_task", "map_source", "reference_mode", "official_source_sha256", "reference_source_manifest_sha256")
    result = {key: report[key] for key in keys}
    result["decoder"] = copy.deepcopy(report["decoder"])
    result["decoder"]["component_loss"] = canonical_recipe(result["decoder"]["component_loss"])
    return result


def verify_metric_pair(old, new):
    left, right = checkpoint_metric_metadata(old), checkpoint_metric_metadata(new)
    return {"passed": left == right, "metadata": left, "differences": leaf_differences(left, right)}


def verify_old_heldout():
    path = OLD_ART / "heldout4096_files.json"
    value = read(path)
    files = value["files"] if isinstance(value, dict) else value
    ordered_hash = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
    if len(files) != 4096 or len(set(files)) != 4096 or hash_file(path) != HELDOUT_JSON_SHA256 or ordered_hash != HELDOUT_ORDER_SHA256:
        raise ValueError("Old heldout4096 changed from the separately audited fixed manifest")
    return {"json_sha256": hash_file(path), "ordered_lines_sha256": ordered_hash, "num_scenes": 4096}


def commands_for(output, seed, old_path, new_path):
    base = [PYTHON, "tools/scratch_component_loss_sweep.py", "--steps", "1", "--checkpoint-steps", "", "--variants", "",
            "--seed", str(seed), "--batch-size", "512", "--scenes", "4096", "--eval-batch-size", "128", "--workers", "0",
            "--eval-seed", "2817", "--extra-eval-seed", "3817", "--evaluation-weights", "ema",
            "--normalizer-json", str(OLD_ART / "train_normalizer.json"), "--eval-manifest", str(OLD_ART / "heldout4096_files.json")]
    commands = []
    for policy, checkpoint in (("random", old_path), ("frozen", new_path)):
        checkpoint_map = output / "design" / f"{policy}_seed{seed}.json"
        save(checkpoint_map, {policy + "_map_geo_recipe": str(checkpoint)})
        command = base + ["--evaluation-checkpoints", str(checkpoint_map), "--output", str(output / (policy + f"_seed{seed}"))]
        if policy == "frozen":
            command += ["--map-checkpoint", str(MAP)]
        commands.append({"policy": policy, "TRAIN_seed": seed, "GEN_seeds": list(GENERATIONS), "command": command})
    return commands


def make_comparison_summary(output):
    by_policy = {"random": [], "frozen": []}
    for policy in by_policy:
        for seed in SEEDS:
            rows = read(output / (policy + f"_seed{seed}") / "summary.json")
            if len(rows) != 2 or {row["eval_seed"] for row in rows} != set(GENERATIONS):
                raise ValueError(f"Incomplete optional evaluation: {policy} seed{seed}")
            for row in rows:
                if row["budget"] != 2048 or row["evaluation_weights"] != "ema" or row["num_samples"] != 4096:
                    raise ValueError("Optional metric scope changed")
                row["S"] = statistics_mean(row[key] / SCALES[key] for key in SCALES)
                row["vehicle_count_ratio"] = row["num_generated_vehicles"] / row["num_gt_vehicles"]
            by_policy[policy] += rows
    seeds = {policy: training_seed_means(rows) for policy, rows in by_policy.items()}
    values = (*SCALES, "S", "generated_onroad_fraction", "vehicle_count_ratio")
    aggregate = {policy: {key: stats(row[key] for row in items.values()) for key in values} for policy, items in seeds.items()}
    confirmation = {policy: {key: stats(items[seed][key] for seed in (1817, 2817)) for key in values} for policy, items in seeds.items()}
    ratios = {key: aggregate["frozen"][key]["mean"] / aggregate["random"][key]["mean"] if aggregate["random"][key]["mean"] > 0 else None for key in SCALES}
    result = {"status": "COMPLETE_OPTIONAL_AUDIT", "selection_role": "NONE",
              "TRAIN_N_all": 3, "independent_confirmation_TRAIN_N": 2, "GEN_repeats_per_TRAIN": 2,
              "TRAIN_seed_roles": {"817": "adaptive discovery", "1817": "held until frozen-pair confirmation", "2817": "held until frozen-pair confirmation"},
              "all_three_TRAIN_statistics_native": aggregate, "independent_confirmation_only_statistics_native": confirmation,
              "frozen_random_ratio_of_all_three_arm_means": ratios,
              "native_divisors_for_raw_JSD": MULTIPLIERS, "per_training_seed": seeds,
              "comparison_definition": "Pretrained frozen map versus randomly initialized jointly trained map; exact same geo loss, not a pretraining-only causal ablation."}
    save(output / "comparison_summary.json", result)
    return result


def statistics_mean(values):
    values = list(values)
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute the prepared evaluator invocations serially only after every CPU audit passes")
    parser.add_argument("--output", type=Path, default=NEW_RUNS / ("map_policy_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a fresh --output; audit/evaluation provenance is never overwritten")
    output.mkdir(parents=True)
    missing = [str(path) for seed in SEEDS for path in checkpoint_paths(seed) if not path.is_file()]
    if missing:
        save(output / "eligibility.json", {"status": "MISSING_CHECKPOINTS", "missing": missing, "commands_executed": False,
                                         "selection_role": "NONE", "no_wait_no_queue_change": True})
        print(json.dumps({"status": "MISSING_CHECKPOINTS", "missing": missing, "output": str(output)}, indent=2))
        return 2
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU eligibility audit must not initialize CUDA")
    calibration_path = OLD_ART / "train_normalizer.json"
    calibration, calibration_hash, map_hash = read(calibration_path), hash_file(calibration_path), hash_file(MAP)
    audit = {"status": "CPU_AUDIT_RUNNING", "selection_role": "NONE", "model_pairs": [], "commands_executed": False,
             "old_heldout_manifest_sha256": hash_file(OLD_ART / "heldout4096_files.json"),
             "training_seed_roles": {"all_distinct_TRAIN": list(SEEDS), "independent_confirmation": [1817, 2817]},
             "history_code_parity_strategy": "Both saved policies are reevaluated using the same current runner/evaluator; no historical metric reuse."}
    commands = []
    try:
        audit["old_heldout_manifest"] = verify_old_heldout()
        for seed in SEEDS:
            old_path, new_path = checkpoint_paths(seed)
            old, new = metadata(old_path), metadata(new_path)
            pair = verify_metadata_pair(old, new, seed, calibration_hash, map_hash)
            if not pair["passed"]:
                audit["model_pairs"].append(pair)
                raise ValueError(f"Historical/current configs are not comparable for seed{seed}")
            pair["old_checkpoint"] = verify_checkpoint(old, policy="random", calibration=calibration, torch=torch)
            pair["new_checkpoint"] = verify_checkpoint(new, policy="frozen", calibration=calibration, torch=torch)
            pair["historical_metric_metadata"] = verify_metric_pair(old, new)
            pair["optimizer_settings_equal"] = pair["old_checkpoint"]["optimizer_param_group_settings"] == pair["new_checkpoint"]["optimizer_param_group_settings"]
            pair["passed"] = pair["old_checkpoint"]["passed"] and pair["new_checkpoint"]["passed"] and pair["optimizer_settings_equal"] and pair["historical_metric_metadata"]["passed"]
            audit["model_pairs"].append(pair)
            if not pair["passed"]:
                raise ValueError(f"Actual2048 checkpoint/EMA/normalizer/optimizer is not comparable for seed{seed}")
            commands += commands_for(output, seed, old_path, new_path)
        assert not torch.cuda.is_initialized()
        audit["current_runner_sha256"] = hash_file(ROOT / "tools/scratch_component_loss_sweep.py")
        audit["current_model_source_sha256"] = {str(path.relative_to(ROOT)): hash_file(path) for path in (
            ROOT / "src/smart/model/smart_gail.py", ROOT / "src/smart/diffusion/initial_diffusion.py", ROOT / "src/smart/diffusion/denoiser.py") if path.is_file()}
        audit["limitation"] = "Historical full training-source hashes are unavailable; this audits compatible actual configs/checkpoints and reevaluates both under identical current code. It does not prove historical training-code identity or pretraining-only causality."
        audit.update(status="PASS_CPU_READY", cuda_initialized=False, commands=commands,
                     side_effects="Only optional output files created; source checkpoints/queue unchanged.")
    except (ValueError, KeyError, OSError) as error:
        audit.update(status="FAILED_ELIGIBILITY", error=str(error), cuda_initialized=torch.cuda.is_initialized())
        save(output / "eligibility.json", audit)
        print(json.dumps({"status": audit["status"], "reason": str(error), "output": str(output)}, indent=2))
        return 2
    save(output / "eligibility.json", audit)
    (output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" + "\n".join(shlex.join(item["command"]) for item in commands) + "\n")
    if not args.run:
        print(json.dumps({"status": audit["status"], "commands": len(commands), "GEN_cells": 12,
                          "executed": False, "output": str(output)}, indent=2))
        return 0
    for item in commands:
        label = item["policy"] + f"_seed{item['TRAIN_seed']}"
        with (output / (label + ".log")).open("w") as log:
            process = subprocess.run(item["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            audit.update(status="OPTIONAL_EVALUATION_FAILED", commands_executed=True, failed_stage=label, returncode=process.returncode)
            save(output / "eligibility.json", audit)
            return process.returncode
    make_comparison_summary(output)
    audit.update(status="COMPLETE_OPTIONAL_AUDIT", commands_executed=True)
    save(output / "eligibility.json", audit)
    print(json.dumps({"status": audit["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
