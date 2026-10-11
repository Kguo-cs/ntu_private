"""Materialize native full-training configs after the frozen scratch decision.

This CPU-only utility does not train, evaluate, or select a recipe. It refuses
an incomplete heldout decision and uses only the selected seed817 checkpoint
to verify the exact resolved loss, TRAIN normalization, and scratch provenance.
The resulting fit experiment starts from random weights; the source checkpoint
is evidence for the recipe, never its initialization.

Run after paired_finalists/paired_summary.json has a complete EMA2048 decision:
  python tools/materialize_scratch_best_config.py

The repository's run.yaml default experiment is intentionally unchanged.
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
import pickle
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
TRAIN_NAME = "init_diffusion_lane_conditioned_scratch_best"
EVAL_NAME = TRAIN_NAME + "_eval"
TRAIN_SEEDS = (817, 1817, 2817)
GENERATION_SEEDS = (2817, 3817)
SCALES = {"nearest_dist_jsd": 1., "lat_dev_jsd": 1., "ang_dev_jsd": 10.,
          "length_jsd": 20., "width_jsd": 15., "speed_jsd": 20., "collision_rate": 5.}


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_hash(value):
    # Same representation as analyze_scratch_loss_search.py's recipe identity.
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def repository_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def portable_path(path):
    try:
        return "${sim_root:}/" + str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_decision(summary):
    """Reject provisional decisions and return the complete seed817 evidence."""
    selected = summary.get("selected_variant")
    require(isinstance(selected, str) and selected, "No selected_variant in the heldout decision")
    require(summary.get("primary_weights") == "ema" and summary.get("endpoint_budget") == 2048,
            "Native materialization requires the frozen EMA2048 endpoint")
    require(summary.get("training_seeds") == list(TRAIN_SEEDS), "Expected three fixed independent training seeds")
    require(summary.get("generation_seeds") == list(GENERATION_SEEDS), "Expected two fixed generation seeds")
    require(summary.get("primary_score_native_scales") == SCALES, "Decision score units differ from the frozen protocol")
    mode = summary.get("modes", {}).get("ema", {})
    require(mode.get("complete") is True and mode.get("missing_cells") == [], "Heldout EMA endpoint is incomplete")
    require(mode.get("budget") == 2048 and mode.get("primary_deployment_endpoint") is True,
            "EMA mode is not the primary 2048-step deployment endpoint")
    require(mode.get("num_paired_training_seeds") == 3 and mode.get("generation_seeds_per_training_seed") == 2,
            "Decision lacks the complete three-by-two paired matrix")
    selection = summary.get("deployment_selection", {})
    require(selection.get("performed") is True and selection.get("selected_variant") == selected,
            "Deployment selection was not performed or disagrees with selected_variant")
    require(mode.get("selection") == selection, "Primary mode and deployment decision disagree")
    require(selection.get("eligibility", {}).get(selected, {}).get("eligible") is True,
            "Selected recipe did not pass the frozen coverage/count guard")
    require(selected in (summary.get("baseline"), summary.get("candidate")), "Selected recipe is not a confirmed finalist")
    arm = mode.get("arms", {}).get(selected, {})
    require(arm.get("num_independent_training_seeds") == 3, "Selected arm lacks three independent training seeds")
    cohort = summary.get("cohort", {})
    require(cohort.get("num_samples") == 4096 and cohort.get("train_batch_size") == 512,
            "Decision did not use the heldout4096 / batch512 cohort")
    require(math.isclose(cohort.get("learning_rate", -1), 5e-4, rel_tol=1e-12), "Decision LR is not 5e-4")
    require(cohort.get("reference_mode") == "matched" and cohort.get("precision") == "32-true",
            "Decision data reference or precision differs from the native recipe")
    require(cohort.get("initialization_scope") == "map_encoder_and_G1_from_scratch",
            "Decision was not trained with both map encoder and G1 from scratch")
    observations = [row for row in summary.get("source_observation_identities", [])
                    if row.get("variant") == selected and row.get("budget") == 2048
                    and row.get("evaluation_weights") == "ema"]
    cells = {(row.get("seed"), row.get("eval_seed")) for row in observations}
    require(cells == {(seed, generation) for seed in TRAIN_SEEDS for generation in GENERATION_SEEDS},
            "Selected recipe's actual EMA2048 observation identities are incomplete")
    require(len({row.get("recipe_sha256") for row in observations}) == 1,
            "Selected recipe changed across the confirmation matrix")
    for row in observations:
        require(row.get("report_source") and repository_path(row["report_source"]).is_file(),
                "A completed EMA2048 metric report is missing")
    return selected, [row for row in observations if row["seed"] == 817]


def validate_source_config(source):
    model = source["model"]["model_config"]
    decoder = model["decoder"]
    init = decoder["init_diffusion"]
    require(source.get("seed") == 817 and source.get("ckpt_path") is None,
            "Source config is not the seed817 scratch training run")
    require(source["data"].get("train_batch_size") == 512 and model.get("optimizer") == "default",
            "Source batch size or optimizer differs from the confirmed production cohort")
    require(math.isclose(model.get("lr", -1), 5e-4, rel_tol=1e-12), "Source LR is not 5e-4")
    require(model.get("finetune") is False and decoder.get("sep_map") is False
            and decoder.get("initial_scene_only") is True and decoder.get("init_decoder") == "flow",
            "Source must train the shared map encoder and InitDiffusion supervised initial-scene path")
    require(init.get("use_ema") is True and math.isclose(init.get("ema_decay", -1), .9999),
            "Source must use G1 EMA with decay .9999")
    for field, expected in {"fix_ego_position": True, "fix_ego_heading": True,
                            "fix_ego_shape": True, "fix_ego_velocity": False,
                            "fix_ego_type": True, "generate_type": True,
                            "type_process": "separate", "type_noise_stats": "standard",
                            "velocity_representation": "speed", "size_representation": "linear",
                            "heading_noise": "circular", "heading_objective": "x0"}.items():
        require(init.get(field) == expected, f"Source conditioning/representation changed: {field}")
    tokens = model["token_processor"]
    require(tokens.get("init_map_crop") == "square" and tokens.get("init_map_half_extent") == 50.,
            "Source map crop differs from the confirmed square50 condition")
    require(all(tokens.get(field) is True for field in ("pred_init", "learn_init", "scenario_dreamer_init")),
            "Source token processor is not the supervised instantaneous initialization path")
    return model


def verify_checkpoint(path, source_model, normalizer, calibration_hash, expected_recipe_hash):
    """Read the existing trusted local checkpoint on CPU, without model creation."""
    import torch

    spec = importlib.util.spec_from_file_location("scratch_materializer_component_loss",
                                               ROOT / "src/smart/diffusion/component_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    init = source_model["decoder"]["init_diffusion"]
    loss = module.ComponentLoss(init["component_loss"], type_kind="ce",
                               type_weight=init["type_loss_weight"], speed_weight=init["speed_loss_weight"],
                               speed_scale="data" if init["speed_loss_scale"] is None else init["speed_loss_scale"]).config
    require(recipe_hash(loss) == expected_recipe_hash, "Source resolved loss differs from the heldout recipe identity")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint.get("global_step") == 2048, "Seed817 source checkpoint is not the completed 2048-step endpoint")
    metadata = checkpoint.get("scratch_sweep", {})
    expected = {"initialization": "random map encoder + G1", "seed": 817,
                "batch_size": 512, "base_train_offset": 0, "saved_global_step": 2048,
                "calibration_sha256": calibration_hash}
    require(all(metadata.get(key) == value for key, value in expected.items()),
            "Checkpoint lacks the matching scratch / TRAIN calibration provenance")
    state = checkpoint["state_dict"]
    extra = state.get("encoder.init_decoder._extra_state", {})
    require(extra.get("component_loss") == loss, "Saved checkpoint loss differs from resolved source config")
    ema = extra.get("ema") or {}
    require(ema.get("num_updates") == 2048 and bool(ema.get("shadow_params")),
            "Selected checkpoint does not contain the completed G1 EMA state")
    require(state.get("encoder._extra_state", {}).get("initial_map_ema") is None,
            "Source unexpectedly applied EMA to its shared map encoder")
    for field in ("normal_mean", "normal_scale"):
        current = state[f"encoder.init_decoder.G1.model.{field}"]
        value = torch.as_tensor(normalizer[field], dtype=current.dtype).reshape(1, -1)
        require(current.shape == (1, 7) and torch.isfinite(value).all().item()
                and torch.equal(current, value), f"Checkpoint {field} differs from the TRAIN normalization file")
        require((value > 0).all().item() if field == "normal_scale" else (value != 0).any().item(),
                f"Invalid TRAIN {field}")
    return loss, metadata


def verify_manifest(json_path, pickle_path, expected_hash):
    names = read_json(json_path)["files"]
    require(isinstance(names, list) and len(names) == 4096 and len(set(names)) == 4096
            and all(isinstance(name, str) and Path(name).name == name for name in names),
            "Heldout JSON must contain 4096 unique cache basenames")
    with Path(pickle_path).open("rb") as handle:
        native_names = pickle.load(handle)["files"]
    require(native_names == names, "Native heldout pickle differs from the confirmed JSON manifest")
    actual_hash = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    require(actual_hash == expected_hash, "Heldout manifest differs from the completed confirmation cohort")
    return actual_hash


def build_configs(source_model, loss, normalizer_path, heldout_path, matmul_precision):
    """Build plain mappings; no writes, checkpoint loading, or Hydra side effects."""
    decoder = copy.deepcopy(source_model["decoder"])
    decoder["sep_map"] = False
    decoder["initial_scene_only"] = True
    decoder["init_diffusion"].update(component_loss=copy.deepcopy(loss),
                                      normalizer_path=portable_path(normalizer_path), use_ema=True, ema_decay=.9999)
    train = {
        "defaults": ["init_diffusion_lane_conditioned_scratch", "_self_"],
        "task_name": TRAIN_NAME, "action": "fit", "ckpt_path": None, "seed": 817,
        "float32_matmul_precision": matmul_precision,
        "paths": {"root_dir": "${sim_root:}/src"},
        "model": {"model_config": {
            "finetune": False, "optimizer": "default", "lr": .0005,
            "lr_total_steps": 128, "lr_warmup_steps": 0, "lr_min_ratio": .05,
            "token_processor": copy.deepcopy(source_model["token_processor"]), "decoder": decoder,
            "sd_reference_mode": "matched", "sd_require_full_set": False,
            "sd_expected_scenes": 50000, "sd_use_cached_evaluator": True,
            "sd_eval_set": "${paths.root_dir}/waymo_data/waymo_eval_set.pkl",
            "sd_real_cache": "${paths.root_dir}/waymo_data/sd_real_metric_cache.sqlite",
            "sd_gen_timestep": 0, "sd_prediction_frame": "world", "sd_require_generation_timestep": True,
            "n_vis_batch": 0}},
        "data": {
            "train_batch_size": 512, "val_batch_size": 256, "test_batch_size": 128,
            "train_raw_dir": "${paths.cache_root}/training_map2_sd",
            "val_raw_dir": "${paths.cache_root}/scenario_dreamer_val",
            "test_raw_dir": "${paths.cache_root}/scenario_dreamer_val", "val_tfrecords_splitted": None,
            "num_workers": 0, "pin_memory": False, "persistent_workers": False, "shuffle": True,
            "scenario_dreamer_train_sample_list": "${paths.cache_root}/training_map2_sd_files.pkl",
            "scenario_dreamer_eval_set": portable_path(heldout_path),
            "scenario_dreamer_preprocessed": False, "scenario_dreamer_train_non_partitioned_only": False,
            "scenario_dreamer_train_latent_cache": None, "scenario_dreamer_val_latent_cache": None,
            "scenario_dreamer_test_latent_cache": None},
        "trainer": {
            "accelerator": "gpu", "devices": 1, "precision": "32-true", "max_epochs": 128, "max_steps": -1,
            "gradient_clip_val": .5, "accumulate_grad_batches": 1, "num_sanity_val_steps": 0,
            "limit_train_batches": 1.0, "limit_val_batches": 1.0, "limit_test_batches": 1.0,
            "val_check_interval": 1.0, "check_val_every_n_epoch": 4},
        "callbacks": {"model_checkpoint": {
            "monitor": None, "save_top_k": -1, "save_last": True,
            "every_n_epochs": 4, "every_n_train_steps": None, "train_time_interval": None,
            "save_on_train_epoch_end": True, "save_weights_only": False,
            "auto_insert_metric_name": False, "filename": "init-diffusion-scratch-best-{epoch}-{step}"}},
    }
    evaluation = {
        "defaults": [TRAIN_NAME, "_self_"], "task_name": EVAL_NAME,
        # A literal ??? cannot replace the parent's existing null during
        # OmegaConf merging. An env interpolation without a default raises
        # unless ckpt_path is explicitly provided on the CLI (or in the env).
        "action": "test", "ckpt_path": "${oc.env:INIT_DIFFUSION_SCRATCH_BEST_CKPT}", "seed": 2817,
        "model": {"model_config": {"sd_reference_mode": "matched", "sd_require_full_set": True,
                                      "sd_expected_scenes": 50000,
                                      "decoder": {"init_diffusion": {"normalizer_path": None}}}},
        "data": {"scenario_dreamer_eval_set": None},
        "trainer": {"limit_test_batches": 1.0, "logger": False}, "callbacks": None,
    }
    return train, evaluation


def yaml_text(config, comments):
    return "# @package _global_\n" + "".join("# " + line + "\n" for line in comments) + yaml.safe_dump(
        config, sort_keys=False, allow_unicode=True)


def write_outputs(outputs, overwrite=False):
    """Preflight all destinations, then replace each file atomically."""
    for path, text in outputs.items():
        if path.exists() and path.read_text() != text and not overwrite:
            raise FileExistsError(f"Refusing to replace existing content without --overwrite: {path}")
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() == text:
            continue
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-summary", type=Path, default=ARTIFACTS / "paired_finalists/paired_summary.json")
    parser.add_argument("--checkpoint-map", type=Path,
                        default=ROOT / "logs/component_loss_scratch/design/heldout_seed817_checkpoints.json")
    parser.add_argument("--normalizer", type=Path, default=ARTIFACTS / "train_normalizer.json")
    parser.add_argument("--heldout-json", type=Path, default=ARTIFACTS / "heldout4096_files.json")
    parser.add_argument("--heldout-pickle", type=Path, default=ARTIFACTS / "heldout4096_files.pkl")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this utility's three named outputs")
    args = parser.parse_args()
    for name in ("paired_summary", "checkpoint_map", "normalizer", "heldout_json", "heldout_pickle"):
        setattr(args, name, repository_path(getattr(args, name)))
    summary = read_json(args.paired_summary)
    selected, observations = validate_decision(summary)
    checkpoint_map = read_json(args.checkpoint_map)
    require(selected in checkpoint_map, "Selected recipe is absent from the seed817 checkpoint map")
    checkpoint = repository_path(checkpoint_map[selected])
    require(checkpoint.is_file(), f"Selected seed817 checkpoint is missing: {checkpoint}")
    require(all(repository_path(row["checkpoint"]) == checkpoint for row in observations),
            "Seed817 checkpoint map differs from the selected EMA2048 observation identities")
    source_path = checkpoint.parent / "config.yaml"
    source = yaml.safe_load(source_path.read_text())
    source_model = validate_source_config(source)
    normalizer_hash = file_hash(args.normalizer)
    require(summary["cohort"].get("calibration_sha256") == normalizer_hash,
            "Native TRAIN normalization file differs from the confirmation calibration")
    manifest_hash = verify_manifest(args.heldout_json, args.heldout_pickle,
                                    summary["cohort"].get("eval_manifest_sha256"))
    train_manifest = ROOT / "src/waymo_data/full/training_map2_sd_files.pkl"
    require(train_manifest.is_file(), "Full TRAIN filename manifest is missing")
    loss, checkpoint_metadata = verify_checkpoint(checkpoint, source_model, read_json(args.normalizer),
                                                   normalizer_hash, observations[0]["recipe_sha256"])
    train, evaluation = build_configs(source_model, loss, args.normalizer, args.heldout_pickle,
                                     source["float32_matmul_precision"])
    train_path = ROOT / "configs/experiment" / (TRAIN_NAME + ".yaml")
    eval_path = ROOT / "configs/experiment" / (EVAL_NAME + ".yaml")
    train_text = yaml_text(train, [
        f"Frozen heldout EMA2048 selection: {selected}.",
        "Full native training starts from random map encoder and G1 weights; no checkpoint is loaded.",
        "Default AdamW + existing cosine scheduler counts 128 real data epochs.",
        "G1 uses adaptive torch_ema; shared map encoder stays online (sep_map=false).",
        "Keep every fourth epoch checkpoint; last.ckpt updates every training epoch.",
        "Validation uses all 4096 confirmation scenes every four epochs; no metric chooses checkpoints.",
    ])
    eval_text = yaml_text(evaluation, [
        f"Evaluate the {selected} native recipe on all 50000 matched-reference scenes.",
        "Pass ckpt_path=/path/to/checkpoint.ckpt (required); saved normalizer buffers override initialization.",
        "A required interpolation prevents inherited null from silently evaluating random weights.",
        "Use a declared checkpoint milestone; the larger overlapping set does not reselect the loss recipe.",
    ])
    artifact = {
        "materialized_utc": datetime.now(timezone.utc).isoformat(), "selected_variant": selected,
        "decision": {"summary": str(repository_path(args.paired_summary)), "sha256": file_hash(args.paired_summary),
                     "endpoint_budget": 2048, "evaluation_weights": "ema", "training_seeds": list(TRAIN_SEEDS),
                     "generation_seeds": list(GENERATION_SEEDS), "selection": summary["deployment_selection"]},
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": file_hash(checkpoint),
        "source_config": str(source_path), "source_config_sha256": file_hash(source_path),
        "source_checkpoint_provenance": checkpoint_metadata,
        "training_initialization": "random map encoder and G1; source checkpoint is recipe evidence only",
        "selected_component_loss": loss, "loss_sha256": recipe_hash(loss),
        "normalizer_file": str(repository_path(args.normalizer)), "normalizer_sha256": normalizer_hash,
        "normalizer_precedence": "TRAIN JSON initializes buffers for fresh fit; checkpoint buffers take precedence on load",
        "validation_manifest": str(repository_path(args.heldout_pickle)), "validation_manifest_sha256": manifest_hash,
        "validation_scenes": 4096, "native_training_shuffle": True,
        "quick_confirmation_order": "seeded full TRAIN order repeated unchanged; native fit reshuffles every epoch",
        "native_optimizer": {"name": "default AdamW", "lr": .0005, "scheduler": "existing epoch cosine",
                             "total_real_epochs": 128, "warmup_epochs": 0, "minimum_lr_ratio": .05},
        "ema": {"scope": "G1 only; shared map stays online", "decay_cap": .9999,
                "adaptive_num_updates": True},
        "checkpoint_policy": {"historical_every_n_epochs": 4, "preserve_all_historical": True,
                              "metric_monitor": None, "last_update": "every training epoch"},
        "configs": {"train": {"path": str(train_path), "sha256": hashlib.sha256(train_text.encode()).hexdigest()},
                    "eval": {"path": str(eval_path), "sha256": hashlib.sha256(eval_text.encode()).hexdigest()}},
    }
    artifact_path = ARTIFACTS / "native_best_materialization.json"
    write_outputs({train_path: train_text, eval_path: eval_text,
                   artifact_path: json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n"}, args.overwrite)
    print(json.dumps({"selected_variant": selected, "train_config": str(train_path),
                      "eval_config": str(eval_path), "artifact": str(artifact_path)}, indent=2))


if __name__ == "__main__":
    main()
