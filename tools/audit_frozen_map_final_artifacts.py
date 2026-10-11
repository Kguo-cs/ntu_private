"""Independent CPU audit and final facts for the completed fixed-map search.

The final facts are written only after queue_complete.json verifies completion.
Reads protocol-root summaries, actual reports, stage traces, final decisions,
and all six final-pair checkpoints. It never trains, changes a decision, or
modifies any existing report/PDF. Seed817 is adaptive discovery; only the fresh
1817/2817 runs are post-freeze confirmation. Full128-epoch training is future.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import tempfile
import os

import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
RUNS = ROOT / "logs/component_loss_frozen_map"
METRICS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
           "width_jsd", "speed_jsd", "collision_rate")
DIVISORS = (10., 10., 100., 100., 100., 100., 1.)
SCALES = (1., 1., 10., 20., 15., 20., 5.)
TRAIN_SEEDS = (817, 1817, 2817)
GEN_SEEDS = (2817, 3817)
PILOTS = {"worker2_benchmark"}
OPTIONAL_STAGE_PREFIXES = ("map_policy_comparison_",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def path_from(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def exposure(updates, batch, count):
    require(all(type(value) is int for value in (updates, batch, count))
            and updates >= 0 and batch > 0 and count > 0, "Invalid exposure arguments")
    epochs, remaining = divmod(updates, (count + batch - 1) // batch)
    return epochs * count + remaining * batch


def stats(values):
    values = list(values)
    require(values and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
            "Statistics need finite observations")
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.,
            "minimum": min(values), "maximum": max(values)}


def gen_means(rows):
    """Equal GEN means within each independent TRAIN seed."""
    by_train = defaultdict(list)
    for row in rows:
        by_train[row["seed"]].append(row)
    keys = METRICS + ("generated_onroad_fraction", "num_generated_vehicles", "num_gt_vehicles")
    return {seed: {key: statistics.fmean(row[key] for row in group) for key in keys}
            for seed, group in by_train.items()}


def mean_training(rows):
    values = gen_means(rows)
    return {key: statistics.fmean(value[key] for value in values.values())
            for key in next(iter(values.values()))}


def relative_result(values, control):
    ratios = {key: values[key] / max(control[key], 1e-6) for key in METRICS}
    eligible = (values["generated_onroad_fraction"] >= control["generated_onroad_fraction"] - .02
                and .9 <= values["num_generated_vehicles"] / values["num_gt_vehicles"] <= 1.1
                and all(ratios[key] <= 1.15 for key in METRICS[3:6]))
    return {"R": statistics.fmean(ratios.values()), "R_attr": statistics.fmean(ratios[key] for key in METRICS[3:6]),
            "S": statistics.fmean(values[key] / scale for key, scale in zip(METRICS, SCALES)),
            "ratios": ratios, "eligible": eligible}


def verify_metric_report(row):
    report_path = path_from(row["metrics_path"])
    require(report_path.is_file(), f"Actual metric report missing: {report_path}")
    report = read(report_path)
    require(report.get("num_samples") == row["num_samples"]
            and report.get("num_gt_samples") == row["num_samples"], "Reported generated/matched GT scene count differs")
    require(report.get("reference_mode") == "matched", "Actual metric reference is not matched")
    for key in METRICS:
        require(math.isclose(row[key], report["agent_metrics"][key], abs_tol=1e-12, rel_tol=1e-12),
                f"Summary and actual report differ: {key}")
        require(math.isfinite(row[key]) and row[key] >= 0, "Metric is nonfinite/negative")
    for category in ("generated", "gt"):
        count = row[f"num_{category}_vehicles"]
        onroad = row[f"num_{category}_onroad_vehicles"]
        require(count > 0 and 0 <= onroad <= count, "Vehicle/on-road counts invalid")
        require(math.isclose(onroad / count, row[f"{category}_onroad_fraction"], abs_tol=1e-12),
                "Coverage fraction differs from actual counts")
    return report


def observation_identity(row):
    return (str(path_from(row["checkpoint"])), row["variant"], row["budget"], row["seed"], row["eval_seed"],
            row["evaluation_weights"], row["eval_manifest_sha256"], row["num_samples"])


def collect_inventory(root, train_count):
    """Count stages once, including true continuation deltas and partial batches."""
    stages, pilot_stages, main_rows, pilot_rows = [], [], [], []
    optional_protocol_roots, optional_reported_observations = 0, 0
    hashes, seen = {}, {}
    root = Path(root).resolve()
    for protocol_path in sorted(root.rglob("protocol.json")):
        folder = protocol_path.parent
        # Optional current-policy comparisons can use the same fixed-map
        # initialization scope as the search. Classify by directory ancestry
        # before enforcing any main-stage completion/provenance requirements;
        # nested evaluator protocols and unfinished diagnostics stay optional.
        optional_root = next((part for part in folder.relative_to(root).parts
                              if part.startswith(OPTIONAL_STAGE_PREFIXES)), None)
        if optional_root is not None:
            summary_path = folder / "summary.json"
            status, reported = "no_summary", 0
            if summary_path.is_file():
                try:
                    optional_rows = read(summary_path)
                    if isinstance(optional_rows, list):
                        reported = sum(isinstance(row, dict) for row in optional_rows)
                        status = "summary_present_not_audited_as_search"
                    else:
                        status = "non_list_summary_not_audited_as_search"
                except (OSError, ValueError):
                    status = "unreadable_summary_not_audited_as_search"
                hashes[str(summary_path.resolve())] = file_hash(summary_path)
            optional_protocol_roots += 1
            optional_reported_observations += reported
            pilot_stages.append({"stage": folder.relative_to(root).as_posix(),
                                 "exclusion_kind": "optional_map_policy_diagnostic",
                                 "optional_root": optional_root, "status": status,
                                 "protocol_source": str(protocol_path.resolve()),
                                 "summary_source": str(summary_path.resolve()) if summary_path.is_file() else None,
                                 "reported_metric_observations_not_main_audited": reported,
                                 "excluded_from_main_metrics_training_recipes_and_cost": True})
            continue
        protocol = read(protocol_path)
        if protocol.get("initialization_scope") != "frozen_pretrained_map_fresh_G1":
            continue
        summary_path = folder / "summary.json"
        if folder.name in PILOTS and not summary_path.is_file():
            pilot_stages.append({"stage": folder.name, "status": "partial_or_failed_auxiliary_pilot"})
            continue
        require(summary_path.is_file(), f"A main stage is incomplete: {folder}")
        rows = read(summary_path)
        require(isinstance(rows, list) and rows, f"Stage summary has no actual completed observations: {folder}")
        is_pilot = folder.name in PILOTS
        if not is_pilot:
            log = ROOT / ("logs/component_loss_frozen_map_" + folder.name + ".log")
            require(log.is_file(), f"Main stage terminal log is missing: {log}")
            terminal = log.read_text().splitlines()[-1].strip()
            require(terminal.startswith("DONE ") and path_from(terminal[5:]) == folder.resolve(),
                    f"Main stage has no verified terminal DONE: {folder}")
        by_variant = defaultdict(list)
        for row in rows:
            verify_metric_report(row)
            require(row.get("map_frozen") is True and row.get("initialization_scope") == "frozen_pretrained_map_fresh_G1",
                    "Observation lacks the fixed-map provenance")
            expected = exposure(row["budget"], row["batch_size"], train_count)
            require(row.get("seen_training_examples") == expected, "Reported exposure ignores partial end-of-epoch batches")
            by_variant[row["variant"]].append(row)
            if is_pilot:
                pilot_rows.append(row)
                continue
            identity = observation_identity(row)
            if identity in seen:
                require(all(seen[identity][key] == row[key] for key in METRICS), "Duplicate observation contradicts its source")
                continue
            seen[identity] = row
            main_rows.append(row)
        for variant, variant_rows in by_variant.items():
            init_path = folder / variant / "initialization.json"
            config_path = folder / variant / "config.yaml"
            require(init_path.is_file() and config_path.is_file(), "Stage lacks resolved config / initialization proof")
            init, config = read(init_path), yaml.safe_load(config_path.read_text())
            maximum = max(row["budget"] for row in variant_rows)
            begin = int(init["resume_global_step"])
            evaluation_only = all(row["evaluation_only"] for row in variant_rows)
            require(evaluation_only or maximum == protocol["steps"], "Training stage did not reach its requested endpoint")
            updates = 0 if evaluation_only else maximum - begin
            require(updates >= 0, "Continuation has a negative update delta")
            consumed = 0 if evaluation_only else exposure(maximum, init["batch_size"], train_count) - exposure(begin, init["batch_size"], train_count)
            fit_times = {row["train_seconds"] for row in variant_rows}
            require(len(fit_times) == 1, "Stage-level fit duration changed across repeated evaluation rows")
            if not evaluation_only:
                trace_path = folder / variant / "train_trace.json"
                require(trace_path.is_file(), "Actual completed training trace is missing")
                trace = read(trace_path)
                require(len(trace) == updates and [item["step"] for item in trace] == list(range(begin, maximum)),
                        "Actual recorded updates do not match stage start/end")
                require(all(item.get("map_grad_norm_before_clip") == 0. for item in trace),
                        "A fixed map received a recorded training gradient")
                if init["optimizer_continued"]:
                    require(begin * init["batch_size"] < train_count
                            and init["actual_train_offset"] == init["base_train_offset"] + begin * init["batch_size"],
                            "Continuation changed the no-replay first-epoch slice rule")
                    files = pickle.loads((folder / variant / "train_files.pkl").read_bytes())["files"]
                    full = pickle.loads((folder / "train_order.pkl").read_bytes())["files"]
                    require(files == full[init["actual_train_offset"]:], "Continued manifest replays consumed samples")
            model = config["model"]["model_config"]
            effective = {"decoder": model["decoder"], "optimizer": model["optimizer"], "lr": model["lr"],
                         "lr_warmup_steps": model["lr_warmup_steps"], "lr_total_steps": model["lr_total_steps"],
                         "lr_min_ratio": model["lr_min_ratio"], "batch": init["batch_size"],
                         "clip": config["trainer"]["gradient_clip_val"], "calibration_sha256": init["calibration_sha256"]}
            stage = {"stage": folder.name, "variant": variant, "seed": init["seed"], "from_step": begin,
                     "to_step": maximum, "successful_updates": updates, "training_cache_exposures": consumed,
                     "batch_size": init["batch_size"], "fresh_model": not evaluation_only and not init["optimizer_continued"],
                     "continuation": not evaluation_only and init["optimizer_continued"], "evaluation_only": evaluation_only,
                     "fit_seconds": next(iter(fit_times)), "observation_count": len(variant_rows),
                     "effective_recipe_sha256": digest(effective), "summary_source": str(summary_path.resolve())}
            (pilot_stages if is_pilot else stages).append(stage)
        hashes[str(summary_path.resolve())] = file_hash(summary_path)
    fit_seconds = sum(stage["fit_seconds"] for stage in stages if not stage["evaluation_only"])
    eval_by_report = {str(path_from(row["metrics_path"])): row["eval_seconds"] for row in main_rows}
    counts = {"main_metric_observations": len(main_rows), "auxiliary_pilot_metric_observations": len(pilot_rows),
              "excluded_optional_protocol_roots": optional_protocol_roots,
              "excluded_optional_reported_metric_observations": optional_reported_observations,
              "successful_fresh_models": sum(stage["fresh_model"] for stage in stages),
              "successful_continuation_stages": sum(stage["continuation"] for stage in stages),
              "evaluation_only_stages": sum(stage["evaluation_only"] for stage in stages),
              "named_variants": len({row["variant"] for row in main_rows}),
              "effective_recipes": len({stage["effective_recipe_sha256"] for stage in stages}),
              "successful_optimizer_updates": sum(stage["successful_updates"] for stage in stages),
              "training_cache_exposures": sum(stage["training_cache_exposures"] for stage in stages),
              "generated_evaluation_scene_exposures": sum(row["num_samples"] for row in main_rows),
              "known_fit_seconds_deduplicated_by_stage": fit_seconds,
              "known_eval_seconds_deduplicated_by_report": sum(eval_by_report.values()),
              "cost_scope": "Recorded main-search fit/eval durations only; excludes pilots, optional map-policy diagnostics, init/calibration/CPU/PDF/failed-stage time. Exposures repeat scenes; they are not independent unique scenes."}
    return counts, stages, pilot_stages, main_rows, hashes


def verify_phase_decisions():
    checks = []
    for filename, ranking_key, control_key in (("phase1_selection.json", "all_ranked", "control"),
                                                ("phase2_selection.json", "ranked", "reference")):
        artifact = read(ART / filename)
        for row in artifact[ranking_key]:
            calculated = relative_result(row, artifact[control_key])
            require(math.isclose(calculated["R"], row["R"], abs_tol=1e-12, rel_tol=1e-12)
                    and calculated["eligible"] is row["eligible"], f"Adaptive phase decision mismatch: {filename}")
        checks.append({"path": str(ART / filename), "sha256": file_hash(ART / filename), "ranking_rows": len(artifact[ranking_key])})
    anchor = read(ART / "phase2_anchor_selection.json")
    control = next(row for row in anchor["ranking"] if row["variant"] == "new_control")
    for row in anchor["ranking"]:
        calculated = relative_result(row, control)
        require(math.isclose(calculated["R"], row["R"], abs_tol=1e-12, rel_tol=1e-12), "Extended phase relative-R mismatch")
    checks.append({"path": str(ART / "phase2_anchor_selection.json"), "sha256": file_hash(ART / "phase2_anchor_selection.json"),
                   "ranking_rows": len(anchor["ranking"])})
    return checks


def audit_final_matrix(decision, frozen, all_rows):
    arm_names = [arm["name"] for arm in frozen["arms"]]
    require(len(arm_names) == 2 and len(set(arm_names)) == 2 and "new_control" in arm_names, "Final pair invalid")
    require(frozen["train_seeds"] == list(TRAIN_SEEDS) and frozen["generation_seeds"] == list(GEN_SEEDS), "Final seed protocol changed")
    require(frozen.get("full50000_no_reselection") is True, "Full-set no-reselection rule was not frozen")
    rows = decision["rows"]
    require(len(rows) == 12 and {(row["variant"], row["seed"], row["eval_seed"]) for row in rows}
            == {(arm, seed, gen) for arm in arm_names for seed in TRAIN_SEEDS for gen in GEN_SEEDS},
            "Final 3TRAIN x 2GEN matrix incomplete or duplicated")
    inventory = {observation_identity(row): row for row in all_rows}
    for row in rows:
        require(row["evaluation_weights"] == "ema" and row["num_samples"] == 4096
                and row["batch_size"] in (256, 512)
                and row["budget"] == (8192 if row["batch_size"] == 256 else 4096), "Final scope/budget changed")
        require(observation_identity(row) in inventory and all(inventory[observation_identity(row)][key] == row[key] for key in METRICS),
                "Final decision row is absent from actual complete summary roots")
    means = {arm: mean_training([row for row in rows if row["variant"] == arm]) for arm in arm_names}
    relative = {arm: relative_result(mean, means["new_control"]) for arm, mean in means.items()}
    improving = sorted((arm for arm in arm_names if relative[arm]["eligible"] and relative[arm]["R"] < 1.),
                       key=lambda arm: (relative[arm]["R"], arm))
    selected = improving[0] if improving else "new_control"
    require(decision["selected"] == selected and decision["selected_arm"]["name"] == selected, "Final selected recipe differs from independent recomputation")
    declared = {row["variant"]: row for row in decision["all"]}
    require(set(declared) == set(arm_names), "Declared final ranking lacks an arm")
    for arm in arm_names:
        require(math.isclose(relative[arm]["R"], declared[arm]["R"], abs_tol=1e-12, rel_tol=1e-12)
                and relative[arm]["eligible"] is declared[arm]["eligible"], "Declared final R/gates mismatch")
    candidate = next(arm for arm in arm_names if arm != "new_control")
    per_seed = {arm: gen_means([row for row in rows if row["variant"] == arm]) for arm in arm_names}
    paired = []
    for seed in TRAIN_SEEDS:
        control, changed = per_seed["new_control"][seed], per_seed[candidate][seed]
        paired.append({"TRAIN_seed": seed, "role": "adaptive discovery" if seed == 817 else "post-freeze fresh confirmation",
                       "R": relative_result(changed, control)["R"],
                       "metric_percent_change": {key: 100. * (changed[key] / max(control[key], 1e-6) - 1.) for key in METRICS},
                       "onroad_change_percentage_points": 100. * (changed["generated_onroad_fraction"] - control["generated_onroad_fraction"])})
    metric_stats = {arm: {key: stats(values[key] for values in per_seed[arm].values()) for key in METRICS}
                    for arm in arm_names}
    confirmation_means = {arm: mean_training([row for row in rows if row["variant"] == arm and row["seed"] != 817]) for arm in arm_names}
    confirmation_relative = {arm: relative_result(value, confirmation_means["new_control"]) for arm, value in confirmation_means.items()}
    return {"selected": selected, "control": "new_control", "candidate": candidate, "actual_cells": 12,
            "actual_independent_training_initializations": 3, "fully_post_freeze_confirmation_training_N": 2,
            "adaptive_discovery_training_seeds": [817], "post_freeze_training_seeds": [1817, 2817],
            "aggregation": "Mean GEN inside TRAIN, then TRAIN equal means; SD is sample SD between TRAIN seeds.",
            "arm_means_native": means, "arm_metric_stats_native": metric_stats, "relative_R_and_gates": relative,
            "paired_training_seed_results": paired, "paired_R_sample_stats_all3": stats(row["R"] for row in paired),
            "metric_improving_TRAIN_seeds_all3": {key: sum(row["metric_percent_change"][key] < 0 for row in paired) for key in METRICS},
            "metric_improving_post_freeze_TRAIN_seeds_of2": {key: sum(row["metric_percent_change"][key] < 0 for row in paired if row["TRAIN_seed"] != 817) for key in METRICS},
            "post_freeze_only_arm_means_native": confirmation_means, "post_freeze_only_relative_R_and_gates": confirmation_relative,
            "post_freeze_only_is_diagnostic_no_reselection": True,
            "seven_metric_raw_jsd_divisors": dict(zip(METRICS, DIVISORS))}


def audit_checkpoints(final_rows, source_map_path, source_map_sha):
    import torch
    require(file_hash(source_map_path) == source_map_sha, "Original map file hash changed")
    source = torch.load(source_map_path, map_location="cpu", weights_only=False)
    prefix = "encoder.map_encoder."
    original = {key: value for key, value in source["state_dict"].items() if key.startswith(prefix)}
    require(len(original) == 66, "Expected actual original fixed map's 66 tensor states")
    map_hash = hashlib.sha256()
    for key, value in sorted(original.items()):
        map_hash.update(key.encode()); map_hash.update(value.contiguous().numpy().tobytes())
    source_tensor_hash = map_hash.hexdigest()
    unique = {}
    for row in final_rows:
        unique.setdefault(str(path_from(row["checkpoint"])), []).append(row)
    require(len(unique) == 6, "Final pair must reference exactly six separately trained checkpoints")
    results = []
    initial_hashes = defaultdict(set)
    order_hashes = defaultdict(set)
    final_configs = defaultdict(set)
    for filename, rows in sorted(unique.items()):
        row = rows[0]
        require(len(rows) == 2 and {item["eval_seed"] for item in rows} == set(GEN_SEEDS), "GEN repeats do not share one checkpoint")
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        state, metadata = checkpoint["state_dict"], checkpoint.get("scratch_sweep", {})
        require(checkpoint["global_step"] == row["budget"] and metadata.get("saved_global_step") == row["budget"], "Checkpoint/global metadata update count differs")
        require(metadata.get("seed") == row["seed"] and metadata.get("batch_size") == row["batch_size"]
                and metadata.get("map_frozen") is True and metadata.get("map_checkpoint_sha256") == source_map_sha
                and metadata.get("base_train_offset") == 0 and metadata.get("initial_map_sha256") == source_tensor_hash
                and metadata.get("initialization") == "frozen pretrained map + random G1", "Final checkpoint fresh fixed-map provenance invalid")
        require({key for key in state if key.startswith(prefix)} == set(original)
                and all(state[key].shape == value.shape and state[key].dtype == value.dtype
                        and torch.equal(state[key], value) for key, value in original.items()), "Final checkpoint changed a source map tensor")
        require(isinstance(checkpoint.get("epoch"), int) and 0 <= checkpoint["epoch"] < 128,
                "Forecast checkpoint does not support the stated incomplete128-epoch scope")
        extra = state["encoder.init_decoder._extra_state"]
        ema = extra.get("ema") or {}
        names = extra.get("ema_parameter_names")
        require(ema.get("num_updates") == row["budget"] and isinstance(names, list)
                and len(names) == len(ema.get("shadow_params", [])) > 0, "Final checkpoint G1 EMA incomplete")
        for name, shadow in zip(names, ema["shadow_params"]):
            parameter = state["encoder.init_decoder.G1." + name]
            require(shadow.shape == parameter.shape and torch.isfinite(shadow).all().item(), "EMA shadow shape/value invalid")
        protocol = read(Path(filename).parent.parent / "protocol.json")
        config = yaml.safe_load((Path(filename).parent / "config.yaml").read_text())
        actual_model = config["model"]["model_config"]
        actual_init = actual_model["decoder"]["init_diffusion"]
        require(extra["component_loss"] == actual_init["component_loss"], "Checkpoint loss differs from its resolved source config")
        configuration = {"decoder": actual_model["decoder"], "optimizer": actual_model["optimizer"], "lr": actual_model["lr"],
                         "batch_size": row["batch_size"], "clip": config["trainer"]["gradient_clip_val"],
                         "calibration_sha256": metadata["calibration_sha256"]}
        final_configs[row["variant"]].add(digest(configuration))
        initial_hashes[row["variant"]].add(metadata["initial_generator_sha256"])
        order_hashes[row["variant"]].add(metadata["train_order_sha256"])
        normalizer_path = path_from(protocol["normalizer_json"])
        normalizer = read(normalizer_path)
        require(file_hash(normalizer_path) == metadata["calibration_sha256"], "Final normalization provenance changed")
        for name in ("normal_mean", "normal_scale"):
            saved = state["encoder.init_decoder.G1.model." + name]
            expected = torch.as_tensor(normalizer[name], dtype=saved.dtype).reshape(1, -1)
            require(torch.equal(saved, expected), "Final normalization buffers differ from TRAIN calibration")
        optimizer = checkpoint["optimizer_states"]
        require(len(optimizer) == 1, "Final supervised training checkpoint has unexpected optimizer count")
        parameter_count = sum(len(group["params"]) for group in optimizer[0]["param_groups"])
        require(parameter_count == len(names), "Optimizer contains parameter slots outside the generator")
        steps = []
        for entry in optimizer[0]["state"].values():
            steps.append(int(entry["step"]))
            require(all(torch.isfinite(value).all().item() for value in entry.values() if torch.is_tensor(value)), "Optimizer has nonfinite state")
        require(steps and min(steps) > 0 and max(steps) <= row["budget"], "Optimizer update counters invalid")
        results.append({"checkpoint": filename, "sha256": file_hash(filename), "variant": row["variant"], "TRAIN_seed": row["seed"],
                        "TRAIN_role": "adaptive discovery" if row["seed"] == 817 else "post-freeze fresh confirmation",
                        "global_step": checkpoint["global_step"], "epoch": checkpoint.get("epoch"),
                        "all66_map_tensors_exact": True, "EMA_num_updates": ema["num_updates"], "EMA_shadow_count": len(names),
                        "optimizer_parameter_slots_generator_only": parameter_count, "optimizer_state_entries": len(steps),
                        "optimizer_step_min": min(steps), "optimizer_step_max": max(steps), "normalizer_exact": True,
                        "initial_generator_sha256": metadata["initial_generator_sha256"],
                        "actual_train_order_sha256": metadata["train_order_sha256"]})
        del checkpoint, state, optimizer
    require(all(len(values) == 3 for values in initial_hashes.values())
            and all(len(values) == 3 for values in order_hashes.values()), "Final TRAIN seeds did not have distinct initial G1/order provenance")
    require(all(len(values) == 1 for values in final_configs.values()), "Final effective recipe changed between TRAIN seeds")
    require(not torch.cuda.is_initialized(), "CPU checkpoint audit unexpectedly initialized CUDA")
    return {"status": "PASS", "source_map_file_sha256": source_map_sha, "source_map_tensor_sha256": source_tensor_hash,
            "original_map_tensor_count": 66, "six_final_checkpoints": results, "cuda_initialized": False}


def render_facts(value):
    count, result = value["counts"], value["final_matrix"]
    lines = ["# Fixed-map search final audited facts", "", f"Selected: `{result['selected']}`. Audit: PASS.", "",
             "Actual independent training starts N=3; seed817 selected the candidate adaptively. Fresh post-freeze confirmation N=2 (1817/2817). Each TRAIN has two GEN repetitions; GEN repeats do not increase training N.", "",
             "| Count | Actual |", "|---|---:|"]
    for key in ("main_metric_observations", "successful_fresh_models", "successful_continuation_stages", "evaluation_only_stages",
                "named_variants", "effective_recipes", "successful_optimizer_updates", "training_cache_exposures",
                "generated_evaluation_scene_exposures", "excluded_optional_protocol_roots",
                "excluded_optional_reported_metric_observations"):
        lines.append(f"| {key} | {count[key]} |")
    lines += ["", "| Arm | R (ratios of arm means) | Eligible |", "|---|---:|---|"]
    for arm, row in result["relative_R_and_gates"].items():
        lines.append(f"| {arm} | {row['R']:.6f} | {row['eligible']} |")
    lines += ["", "| TRAIN seed | Role | Paired R |", "|---:|---|---:|"]
    for row in result["paired_training_seed_results"]:
        lines.append(f"| {row['TRAIN_seed']} | {row['role']} | {row['R']:.6f} |")
    lines += ["", "The six JSD metrics use the saved native helper units; collision is percent. Raw JSD divides nearest/lateral by10 and angle/length/width/speed by100. R/S are search scores, not paper scores.", "",
              "Each final B5124096 or B2568192 model consumes the same actual cache exposure, including partial end-of-epoch batches. Full50000 audits overlap validation and do not reselect.", "",
              "Full128-epoch native training is a recommended future command, not a completed result or proven global optimum. Logged main-search durations exclude pilots, optional map-policy diagnostics, initialization, CPU calibration/audits, PDF and failed-stage time.", ""]
    return "\n".join(lines)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text); temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    complete_path = ART / "queue_complete.json"
    if not complete_path.is_file():
        print(json.dumps({"status": "NOT_READY", "reason": "Actual queue_complete.json is absent; final facts not written."}))
        raise SystemExit(2)
    complete = read(complete_path)
    require(complete.get("status") == "COMPLETE_ACTUAL_SEARCH" and complete.get("full_set_audits") == 2,
            "Queue completion does not prove the actual final pair plus two full-set audits")
    train_manifest = ROOT / "src/waymo_data/full/training_map2_sd_files.pkl"
    train_files = pickle.loads(train_manifest.read_bytes())["files"]
    require(len(train_files) == 493542, "Full TRAIN manifest changed")
    counts, stages, pilots, rows, summary_hashes = collect_inventory(RUNS, len(train_files))
    phase_checks = verify_phase_decisions()
    decision_path = ART / "final_selection.json"
    decision = read(decision_path)
    frozen = read(ART / "frozen_confirmation_protocol.json")
    require(complete["selected"] == decision["selected"], "Queue and final decision disagree")
    matrix = audit_final_matrix(decision, frozen, rows)
    protocol = read(ART / "search_protocol.json")
    checkpoint_audit = audit_checkpoints(decision["rows"], path_from(protocol["source_map_checkpoint"]), protocol["source_map_sha256"])
    full_rows = [row for row in rows if row["num_samples"] == 50000]
    require(len(full_rows) == 2 and {row["variant"] for row in full_rows} == {matrix["control"], matrix["candidate"]},
            "Expected exactly two actual full50000 generated/matched audits")
    require(all(row["evaluation_only"] and row["seed"] == 817 and row["eval_seed"] == 2817 for row in full_rows),
            "Full50000 scope differs from the declared discovery-seed audit")
    old_analyzer = read(ART / "search_summary.json") if (ART / "search_summary.json").is_file() else None
    reported_counts = {"search_summary_present": old_analyzer is not None,
                       "search_summary_num_observations": old_analyzer.get("num_observations") if old_analyzer else None,
                       "matches_independent_actual_count": old_analyzer is not None and old_analyzer.get("num_observations") == counts["main_metric_observations"],
                       "note": "A false match identifies a stale/alternate-scope analyzer snapshot; these facts come directly from actual completed summaries."}
    facts = {"status": "PASS", "completed_actual_search": True, "audited_utc": datetime.now(timezone.utc).isoformat(),
             "source_tool_sha256": file_hash(Path(__file__)), "counts": counts, "training_stages": stages,
             "excluded_auxiliary_pilots": [stage for stage in pilots
                                            if stage.get("exclusion_kind") != "optional_map_policy_diagnostic"],
             "excluded_optional_diagnostics": [stage for stage in pilots
                                                if stage.get("exclusion_kind") == "optional_map_policy_diagnostic"],
             "summary_source_sha256": summary_hashes,
             "phase_decision_audits": phase_checks, "decision_sha256": file_hash(decision_path),
             "frozen_pair_protocol_sha256": file_hash(ART / "frozen_confirmation_protocol.json"),
             "final_matrix": matrix, "final_checkpoints": checkpoint_audit,
             "full50000_actual_rows": full_rows, "full50000_role": "overlapping larger-distribution audit; no reselection",
             "reported_analyzer_counts": reported_counts,
             "normalizer_scope": "TRAIN-only; representation-specific modes share the same8192 selected TRAIN manifest",
             "native_full128_training_completed": False, "global128epoch_optimum_proven": False,
             "native_fit_order": "future full TRAIN fit reshuffles each epoch", "quick_fit_order": "same seeded full TRAIN order repeated",
             "cuda_initialized": False}
    atomic_write(ART / "final_facts.json", json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(ART / "final_facts.md", render_facts(facts))
    print(json.dumps({"status": "PASS", "selected": matrix["selected"], "counts": counts,
                      "facts_json": str(ART / "final_facts.json"), "facts_md": str(ART / "final_facts.md")}, indent=2))


if __name__ == "__main__":
    main()
