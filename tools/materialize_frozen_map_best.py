"""Materialize native fixed-map configs only after the actual N=3 decision.

This CPU-only utility never trains or selects a candidate. It refuses to write
anything until this search's final_selection.json and complete metric matrix
exist. Native fit loads the AIRL map only and starts G1/Adam/scheduler/EMA fresh;
native eval requires a complete initializer checkpoint and restores its saved
map/G1/normalization/EMA state. The evidence checkpoint is never a fit resume.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
TRAIN_NAME = "init_diffusion_lane_conditioned_frozen_map_best"
EVAL_NAME = TRAIN_NAME + "_eval"
TRAIN_SEEDS = (817, 1817, 2817)
GEN_SEEDS = (2817, 3817)
METRICS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
           "width_jsd", "speed_jsd", "collision_rate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def repository_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def portable_path(value):
    path = repository_path(value)
    try:
        return "${sim_root:}/" + path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def aggregate(rows):
    """Mean GEN within TRAIN, then equally mean independent TRAIN seeds."""
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    keys = METRICS + ("generated_onroad_fraction", "num_generated_vehicles", "num_gt_vehicles")
    return {key: sum(sum(row[key] for row in group) / len(group)
                     for group in by_seed.values()) / len(by_seed) for key in keys}


def compare(values, control):
    ratios = {key: values[key] / max(control[key], 1e-6) for key in METRICS}
    eligible = (values["generated_onroad_fraction"] >= control["generated_onroad_fraction"] - .02
                and .9 <= values["num_generated_vehicles"] / values["num_gt_vehicles"] <= 1.1
                and all(ratios[key] <= 1.15 for key in ("length_jsd", "width_jsd", "speed_jsd")))
    return sum(ratios.values()) / len(METRICS), eligible


def validate_decision(decision, frozen):
    """Validate the complete frozen pair and recompute the declared decision."""
    selected = decision.get("selected")
    arm = decision.get("selected_arm", {})
    require(isinstance(selected, str) and selected and arm.get("name") == selected,
            "Selected name and selected_arm disagree")
    require(frozen.get("train_seeds") == list(TRAIN_SEEDS)
            and frozen.get("generation_seeds") == list(GEN_SEEDS),
            "Frozen confirmation must contain the three-by-two seed protocol")
    arm_names = [item.get("name") for item in frozen.get("arms", [])]
    require(len(arm_names) == 2 and len(set(arm_names)) == 2 and "new_control" in arm_names,
            "Frozen confirmation must contain control and one distinct candidate")
    require(selected in arm_names and frozen.get("full50000_no_reselection") is True,
            "Decision does not follow the frozen pair / no-reselection rule")
    rows = decision.get("rows", [])
    require(len(rows) == 12, "Expected two recipes by three TRAIN by two GEN completed rows")
    expected_cells = {(name, seed, gen) for name in arm_names for seed in TRAIN_SEEDS for gen in GEN_SEEDS}
    cells = {(row.get("variant"), row.get("seed"), row.get("eval_seed")) for row in rows}
    require(cells == expected_cells, "Decision has missing, duplicate, or unexpected confirmation cells")
    for row in rows:
        require(row.get("evaluation_weights") == "ema" and row.get("num_samples") == 4096,
                "Decision row is not the completed EMA / confirmation4096 endpoint")
        batch = row.get("batch_size")
        require(batch in (256, 512) and row.get("budget") == (8192 if batch == 256 else 4096),
                "Decision endpoint differs from the frozen equal-exposure forecast budget")
        for key in METRICS + ("generated_onroad_fraction", "num_generated_vehicles", "num_gt_vehicles"):
            require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]),
                    f"Missing or nonfinite actual metric: {key}")
        require(row["num_gt_vehicles"] > 0, "Decision has no matched GT vehicles")
    means = {name: aggregate([row for row in rows if row["variant"] == name]) for name in arm_names}
    scores = {name: compare(values, means["new_control"]) for name, values in means.items()}
    improving = sorted((name for name, (score, eligible) in scores.items() if eligible and score < 1.),
                       key=lambda name: (scores[name][0], name))
    expected = improving[0] if improving else "new_control"
    require(selected == expected, "Decision disagrees with recomputed eligible relative-R selection")
    declared = {row["variant"]: row for row in decision.get("all", [])}
    require(set(declared) == set(arm_names), "Final ranking does not contain both confirmed arms")
    for name, (score, eligible) in scores.items():
        require(math.isclose(declared[name].get("R", -1), score, rel_tol=1e-12, abs_tol=1e-12)
                and declared[name].get("eligible") is eligible,
                "Declared relative-R/gates differ from actual confirmation rows")
    selected_rows = [row for row in rows if row["variant"] == selected]
    return selected, arm, selected_rows, scores


def build_configs(source, loss, normalizer, map_checkpoint):
    """Pure builder: no decision fabrication, checkpoint load, or filesystem write."""
    model = copy.deepcopy(source["model"]["model_config"])
    model.update(finetune=True, initial_map_checkpoint=portable_path(map_checkpoint),
                 sd_reference_mode="matched", sd_require_full_set=False, sd_expected_scenes=50000)
    init = model["decoder"]["init_diffusion"]
    init.update(component_loss=copy.deepcopy(loss), normalizer_path=portable_path(normalizer))
    batch = int(source["data"]["train_batch_size"])
    clip = float(source["trainer"]["gradient_clip_val"])
    train = {
        "defaults": ["init_diffusion_lane_conditioned_frozen_map", "_self_"],
        "task_name": TRAIN_NAME, "action": "fit", "ckpt_path": None, "seed": 817,
        "float32_matmul_precision": source.get("float32_matmul_precision", "medium"),
        "paths": {"root_dir": "${sim_root:}/src"},
        "model": {"model_config": model},
        "data": {"train_batch_size": batch, "shuffle": True, "num_workers": 0,
                 "pin_memory": False, "persistent_workers": False},
        "trainer": {"max_epochs": 128, "max_steps": -1, "gradient_clip_val": clip,
                    "accumulate_grad_batches": 1, "precision": "32-true"},
        "callbacks": {
            "model_checkpoint": {"filename": "init-diffusion-frozen-map-best-{epoch}-{step}"},
            "latest_checkpoint": {"filename": "init-diffusion-frozen-map-best-latest-{epoch}-{step}"}},
    }
    evaluation_model = copy.deepcopy(model)
    evaluation_model.update(initial_map_checkpoint=None, sd_require_full_set=True)
    evaluation_model["decoder"]["init_diffusion"]["normalizer_path"] = None
    evaluation = {
        "defaults": ["init_diffusion_lane_conditioned_frozen_map_eval", "_self_"],
        "task_name": EVAL_NAME, "action": "test",
        "ckpt_path": "${oc.env:INIT_DIFFUSION_FROZEN_MAP_BEST_CKPT}", "seed": 2817,
        "float32_matmul_precision": source.get("float32_matmul_precision", "medium"),
        "model": {"model_config": evaluation_model},
        "data": {"train_batch_size": batch, "scenario_dreamer_eval_set": None,
                 "num_workers": 0, "pin_memory": False, "persistent_workers": False},
        "trainer": {"limit_test_batches": 1.0, "logger": False},
        "callbacks": None, "logger": False,
    }
    return train, evaluation


def yaml_text(value, comments):
    return "# @package _global_\n" + "".join("# " + text + "\n" for text in comments) + yaml.safe_dump(
        value, sort_keys=False, allow_unicode=True)


def write_outputs(outputs, *, overwrite=False):
    for path, content in outputs.items():
        require(not path.exists() or path.read_text() == content or overwrite,
                f"Refusing to overwrite changed materialization without --overwrite: {path}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() == content:
            continue
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def verify_actual_evidence(decision, frozen, protocol):
    """Read actual reports/configs/checkpoint on CPU; reject provisional evidence."""
    selected, arm, rows, scores = validate_decision(decision, frozen)
    for row in decision["rows"]:
        report_path = repository_path(row["metrics_path"])
        require(report_path.is_file(), f"Completed metric report missing: {report_path}")
        report = read(report_path)
        require(report.get("num_samples") == 4096 and report.get("num_gt_samples") == 4096,
                "Actual metric report does not contain matched confirmation4096")
        for key in METRICS:
            require(math.isclose(report["agent_metrics"][key], row[key], rel_tol=1e-12, abs_tol=1e-12),
                    "Final decision metrics differ from their actual saved reports")
    evidence = [row for row in rows if row["seed"] == 817]
    paths = {repository_path(row["checkpoint"]) for row in evidence}
    require(len(paths) == 1, "Seed817 generation repeats do not reference the same complete checkpoint")
    checkpoint_path = paths.pop()
    require(checkpoint_path.is_file(), "Selected seed817 endpoint checkpoint is missing")
    source_path = checkpoint_path.parent / "config.yaml"
    require(source_path.is_file(), "Selected endpoint's resolved config is missing")
    source = OmegaConf.to_container(OmegaConf.load(source_path), resolve=True)
    model = source["model"]["model_config"]
    init = model["decoder"]["init_diffusion"]
    require(model.get("finetune") is True and model["decoder"].get("sep_map") is False
            and model["decoder"].get("initial_scene_only") is True
            and model["decoder"].get("init_decoder") == "flow",
            "Selected endpoint is not the fixed-map supervised InitDiffusion path")
    require(source.get("ckpt_path") is None and source.get("seed") == 817,
            "Selected forecast did not train a fresh seed817 G1")
    require(int(source["data"]["train_batch_size"]) == arm["batch"]
            and math.isclose(float(model["lr"]), arm["lr"], rel_tol=1e-12)
            and float(source["trainer"]["gradient_clip_val"]) == arm["clip"],
            "Selected effective batch/LR/clip disagrees with the completed endpoint")
    for key, value in arm["config"].items():
        require(OmegaConf.select(OmegaConf.create(source), key) == value,
                f"Selected effective config differs from the actual endpoint: {key}")
    require(init.get("component_loss") == arm["recipe"], "Selected loss recipe differs from actual endpoint")
    require(init.get("use_ema") is True, "Selected native recipe must retain the evaluated G1 EMA")
    for key, expected in {"fix_ego_position": True, "fix_ego_heading": True, "fix_ego_shape": True,
                          "fix_ego_velocity": False, "fix_ego_type": True, "generate_type": True,
                          "type_process": "separate", "heading_noise": "circular", "heading_objective": "x0"}.items():
        require(init.get(key) == expected, f"Selected endpoint changed the frozen input/target conditions: {key}")
    normalizer_path = repository_path(arm["normalizer"])
    require(normalizer_path.is_file(), "Selected TRAIN normalizer is missing")
    normalizer = read(normalizer_path)
    map_checkpoint = repository_path(protocol["source_map_checkpoint"])
    require(file_hash(map_checkpoint) == protocol["source_map_sha256"], "Fixed source map file changed")
    require(repository_path(model["initial_map_checkpoint"]) == map_checkpoint,
            "Selected endpoint used a different fixed source map")
    import torch
    spec = importlib.util.spec_from_file_location("frozen_materializer_component_loss", ROOT / "src/smart/diffusion/component_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = module.ComponentLoss(init["component_loss"], type_kind="ce", type_weight=init["type_loss_weight"],
                                speed_weight=init["speed_loss_weight"],
                                speed_scale="data" if init["speed_loss_scale"] is None else init["speed_loss_scale"]).config
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("scratch_sweep", {})
    require(checkpoint.get("global_step") == evidence[0]["budget"], "Evidence checkpoint is not the selected endpoint")
    require(metadata.get("initialization") == "frozen pretrained map + random G1"
            and metadata.get("seed") == 817 and metadata.get("batch_size") == arm["batch"]
            and metadata.get("base_train_offset") == 0
            and metadata.get("saved_global_step") == evidence[0]["budget"]
            and metadata.get("map_frozen") is True
            and metadata.get("map_checkpoint_sha256") == protocol["source_map_sha256"]
            and metadata.get("calibration_sha256") == file_hash(normalizer_path),
            "Selected checkpoint does not retain fresh-G1 / fixed-map / TRAIN normalization provenance")
    state = checkpoint["state_dict"]
    extra = state.get("encoder.init_decoder._extra_state", {})
    require(extra.get("component_loss") == loss, "Saved checkpoint loss differs from resolved recipe")
    require((extra.get("ema") or {}).get("num_updates") == evidence[0]["budget"]
            and bool((extra.get("ema") or {}).get("shadow_params")), "Saved G1 EMA is incomplete")
    dims = 7 if init["velocity_representation"] == "speed" else 8
    if init["size_representation"] == "log":
        require(normalizer.get("size_representation") == "log", "Log-size recipe needs the log TRAIN calibration")
    for key in ("normal_mean", "normal_scale"):
        current = state[f"encoder.init_decoder.G1.model.{key}"]
        value = torch.as_tensor(normalizer[key], dtype=current.dtype).reshape(1, -1)
        require(value.shape == (1, dims) and torch.isfinite(value).all().item() and torch.equal(current, value),
                f"Selected checkpoint {key} differs from the TRAIN calibration")
        require((value > 0).all().item() if key == "normal_scale" else (value != 0).any().item(),
                f"Invalid TRAIN {key}")
    require(not torch.cuda.is_initialized(), "Materialization unexpectedly initialized CUDA")
    return selected, arm, source, loss, normalizer_path, map_checkpoint, checkpoint_path, source_path, metadata, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this utility's three output files")
    args = parser.parse_args()
    decision_path = ART / "final_selection.json"
    if not decision_path.is_file():
        parser.error("Actual final_selection.json is not present; no best config or decision was written")
    frozen_path = ART / "frozen_confirmation_protocol.json"
    protocol_path = ART / "search_protocol.json"
    require(frozen_path.is_file() and protocol_path.is_file(), "Actual frozen confirmation/search protocol is missing")
    decision, frozen, protocol = read(decision_path), read(frozen_path), read(protocol_path)
    verified = verify_actual_evidence(decision, frozen, protocol)
    selected, arm, source, loss, normalizer, map_checkpoint, checkpoint, source_path, metadata, scores = verified
    train, evaluation = build_configs(source, loss, normalizer, map_checkpoint)
    train_path = ROOT / "configs/experiment" / (TRAIN_NAME + ".yaml")
    eval_path = ROOT / "configs/experiment" / (EVAL_NAME + ".yaml")
    train_text = yaml_text(train, [
        f"Actual frozen-map N=3 confirmation selected: {selected}.",
        "Native fit loads only the AIRL map, kept frozen/eval; G1/Adam/scheduler/EMA start fresh.",
        "Evidence initializer checkpoint is never resumed. TRAIN JSON initializes fresh normalization.",
        "Full 128-epoch training is a recommended forecast, not already measured convergence.",
        "Native fit reshuffles the full TRAIN set each epoch; quick search repeated its seeded order.",
    ])
    eval_text = yaml_text(evaluation, [
        f"Full matched 50000-scene evaluation for the selected {selected} recipe.",
        "Pass ckpt_path=/path/to/full_initializer.ckpt or INIT_DIFFUSION_FROZEN_MAP_BEST_CKPT.",
        "Checkpoint restores map/G1/normalizer/EMA; do not reload AIRL or TRAIN statistics.",
        "Choose a declared saved milestone. Larger overlapping evaluation does not reselect hyperparameters.",
    ])
    artifact = {
        "materialized_utc": datetime.now(timezone.utc).isoformat(), "selected": selected,
        "decision": {"path": str(decision_path), "sha256": file_hash(decision_path),
                     "frozen_protocol_sha256": file_hash(frozen_path), "scores": scores,
                     "TRAIN_seeds": list(TRAIN_SEEDS), "GEN_seeds": list(GEN_SEEDS)},
        "source_map": {"path": str(map_checkpoint), "sha256": file_hash(map_checkpoint),
                       "native_fit_loaded_scope": "strict encoder.map_encoder tensors only; frozen and eval"},
        "evidence_initializer_checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint),
                                            "config": str(source_path), "config_sha256": file_hash(source_path),
                                            "provenance": metadata, "used_to_initialize_native_fit": False},
        "effective_recipe": {"loss": loss, "loss_sha256": object_hash(loss), "config": arm["config"],
                             "lr": arm["lr"], "clip": arm["clip"], "batch": arm["batch"]},
        "normalizer": {"path": str(normalizer), "sha256": file_hash(normalizer),
                       "fit": "TRAIN JSON initializes fresh buffers", "eval": "saved checkpoint buffers restore; path=null"},
        "native_fit": {"action": "fit", "ckpt_path": None, "finetune": True,
                       "G1_optimizer_scheduler_EMA": "fresh", "map": "fixed pretrained AIRL map",
                       "epochs": 128, "full_training_completed": False,
                       "role": "recommended future full training; not an already completed 128-epoch result",
                       "shuffle_each_epoch": True, "workers": 0,
                       "quick_workers": source["data"].get("num_workers"),
                       "quick_order": "same seeded full order repeats; native full fit reshuffles"},
        "native_eval": {"action": "test", "requires_complete_initializer_checkpoint": True,
                        "initial_map_checkpoint": None, "normalizer_path": None,
                        "restored_scope": "map + G1 + normalization buffers + G1 EMA", "matched_scenes": 50000},
        "preserved_conditions": {key: source["model"]["model_config"]["decoder"]["init_diffusion"].get(key)
                                 for key in ("fix_ego_position", "fix_ego_heading", "fix_ego_shape", "fix_ego_velocity",
                                             "fix_ego_type", "generate_type", "type_process", "heading_noise",
                                             "heading_objective", "velocity_representation", "size_representation",
                                             "pos_source", "shape_source", "velocity_source")},
        "global128epoch_optimum_proven": False,
        "configs": {"fit": {"path": str(train_path), "sha256": hashlib.sha256(train_text.encode()).hexdigest()},
                    "eval": {"path": str(eval_path), "sha256": hashlib.sha256(eval_text.encode()).hexdigest()}},
    }
    artifact_path = ART / "native_best_materialization.json"
    write_outputs({train_path: train_text, eval_path: eval_text,
                   artifact_path: json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n"},
                  overwrite=args.overwrite)
    print(json.dumps({"selected": selected, "fit_config": str(train_path), "eval_config": str(eval_path),
                      "artifact": str(artifact_path)}, indent=2))


if __name__ == "__main__":
    main()
