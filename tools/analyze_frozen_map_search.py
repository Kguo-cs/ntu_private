"""CPU-only, cohort-preserving summaries of the frozen-map InitDiffusion search.

Only protocol-root summary.json files create observations.  Nested evaluator
reports supply missing metadata and never create extra observations.  Averages
are taken within generation seed, then across generation seeds within TRAIN
seed; sample SD and N are across independent TRAIN seeds.  An explicit external
control may bridge different update/batch budgets at exactly equal observed
sample exposure, but may never bridge evaluation/conditioning cohorts.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import pickle
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCALES = {"nearest_dist_jsd": 1., "lat_dev_jsd": 1., "ang_dev_jsd": 10.,
          "length_jsd": 20., "width_jsd": 15., "speed_jsd": 20., "collision_rate": 5.}
METRICS = tuple(SCALES)
ATTR = ("length_jsd", "width_jsd", "speed_jsd")
MULTIPLIERS = {key: 10. if key in METRICS[:2] else 100. for key in METRICS[:-1]}
MULTIPLIERS["collision_rate"] = 1.
COHORT_FIELDS = ("eval_manifest_sha256", "num_samples", "reference_mode", "initialization_scope",
                 "map_checkpoint_sha256", "calibration_train_manifest_sha256", "map_crop", "map_half_extent",
                 "precision", "float32_matmul_precision", "sampling_steps", "base_train_offset",
                 "conditioning", "metric_backend_sha256", "reference_source_manifest_sha256")
GATES = {"onroad_drop_fraction": .02, "vehicle_ratio_min": .9, "vehicle_ratio_max": 1.1,
         "length_width_speed_max_control_ratio": 1.15}
EXCLUDED_PILOT_STAGES = ("worker2_benchmark",)
EXCLUDED_OPTIONAL_STAGES = ("map_policy_comparison_*",)
EXCLUDED_NON_MAIN_STAGES = EXCLUDED_PILOT_STAGES + EXCLUDED_OPTIONAL_STAGES
ADAPTIVE_DISCOVERY_SEEDS = (817,)
FROZEN_PAIR_CONFIRMATION_SEEDS = (1817, 2817)


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def first(*values):
    return next((value for value in values if value is not None), None)


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def read_json(path, default=None, warnings=None):
    if not Path(path).is_file():
        return default
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        if warnings is not None:
            warnings.append(f"Partial/unreadable JSON skipped: {path}: {error}")
        return default


def rows_from_summary(value):
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("rows", "results", "summary"):
            if isinstance(value.get(key), list):
                return rows_from_summary(value[key])
        if all(key in value.get("agent_metrics", value) for key in METRICS):
            return [value]
    return []


def stats(values):
    values = [float(value) for value in values if finite(value) is not None]
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.,
            "minimum": min(values), "maximum": max(values)}


def exposure(updates, batch, active_count):
    """Count actual partial end-of-epoch batches, rather than steps * B."""
    if any(type(value) is not int for value in (updates, batch, active_count)) or updates < 0 or batch <= 0 or active_count <= 0:
        return None
    batches = (active_count + batch - 1) // batch
    epochs, remaining = divmod(updates, batches)
    return epochs * active_count + remaining * batch


def score(values):
    return statistics.fmean(values[key] / SCALES[key] for key in METRICS)


def ratio(candidate, control):
    if finite(candidate) is None or finite(control) is None or control <= 0:
        return None
    return candidate / control


def mean_complete(values):
    values = list(values)
    return statistics.fmean(values) if values and all(finite(value) is not None for value in values) else None


def report_for_row(original, run, warnings):
    value = original.get("metrics_path") or original.get("report_path")
    if not value:
        return {}, None
    path = Path(value)
    if not path.is_absolute():
        # Harness currently writes absolute report paths.  Permit repo-relative
        # or run-relative paths in fixtures/exported stages.
        path = ROOT / path if (ROOT / path).is_file() else run / path
    return read_json(path, {}, warnings), str(path.resolve())


def enrich(original, protocol, run, source, warnings):
    variant = original.get("variant")
    if not variant:
        warnings.append(f"Summary row without variant skipped: {source}")
        return None
    folder = run / str(variant)
    init = read_json(folder / "initialization.json", {}, warnings)
    config_path = folder / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    config = config if isinstance(config, dict) else {}
    report, report_path = report_for_row(original, run, warnings)
    values = original.get("agent_metrics", original)
    metrics = {key: finite(first(values.get(key), nested(report, "agent_metrics", key))) for key in METRICS}
    if any(value is None or value < 0 for value in metrics.values()):
        warnings.append(f"Incomplete/nonfinite metric observation skipped: {source} / {variant}")
        return None
    decoder = nested(config, "model", "model_config", "decoder", "init_diffusion") or {}
    loss = first(nested(report, "decoder", "component_loss"), decoder.get("component_loss"))
    normalizer = protocol.get("normalizer") or {}
    if not normalizer.get("train_manifest_sha256") and protocol.get("normalizer_json"):
        normalizer_path = Path(protocol["normalizer_json"])
        if not normalizer_path.is_absolute():
            normalizer_path = ROOT / normalizer_path
        normalizer = read_json(normalizer_path, {}, warnings)
    calibration_sha256 = first(original.get("calibration_sha256"), protocol.get("calibration_sha256"))
    calibration_train_manifest = normalizer.get("train_manifest_sha256")
    if not calibration_train_manifest:
        # Without proof of the calibration scene selection, only an identical
        # calibration artifact may match; never pool unknown selections merely
        # because their metadata fields are missing.
        calibration_train_manifest = "uncertified_selection:" + str(calibration_sha256)
    seed = first(original.get("seed"), original.get("train_seed"), protocol.get("seed"))
    generation = first(original.get("eval_seed"), nested(report, "runtime", "torch_initial_seed"), protocol.get("eval_seed"))
    budget = first(original.get("budget"), original.get("global_step"))
    batch = first(original.get("batch_size"), original.get("train_batch_size"), init.get("batch_size"), protocol.get("batch_size"))
    seen = original.get("seen_training_examples")
    if any(type(value) is not int for value in (seed, generation, budget, batch, seen)):
        warnings.append(f"Missing actual budget/B/exposure or seed; observation skipped rather than guessed: {source} / {variant}")
        return None
    if budget <= 0 or batch <= 0 or seen <= 0:
        warnings.append(f"Nonpositive actual budget/B/exposure skipped: {source} / {variant}")
        return None
    observation = {
        "run_path": str(run.resolve()), "summary_source": str(source.resolve()), "report_source": report_path,
        "variant": str(variant), "seed": seed, "eval_seed": generation,
        "budget": budget, "train_batch_size": batch, "seen_training_examples": seen,
        "evaluation_weights": first(original.get("evaluation_weights"),
            "ema" if nested(report, "decoder", "use_ema") else None, "online"),
        "reference_mode": first(report.get("reference_mode"), protocol.get("reference_mode"), "matched"),
        "num_samples": first(original.get("num_samples"), report.get("num_samples"), protocol.get("scenes")),
        "num_gt_samples": first(original.get("num_gt_samples"), report.get("num_gt_samples")),
        "eval_manifest_sha256": first(original.get("eval_manifest_sha256"), protocol.get("eval_manifest_sha256")),
        "initialization_scope": first(original.get("initialization_scope"), protocol.get("initialization_scope")),
        "map_checkpoint_sha256": first(original.get("map_checkpoint_sha256"), protocol.get("map_checkpoint_sha256")),
        "calibration_sha256": calibration_sha256,
        "calibration_train_manifest_sha256": calibration_train_manifest,
        "calibration_feature_dimensions": normalizer.get("dimensions"),
        "size_representation": first(nested(report, "decoder", "size_representation"), decoder.get("size_representation")),
        "velocity_representation": first(nested(report, "decoder", "velocity_representation"), decoder.get("velocity_representation")),
        "map_crop": first(nested(report, "decoder", "init_map_crop"), original.get("map_crop"), protocol.get("map_crop")),
        "map_half_extent": first(nested(report, "decoder", "init_map_half_extent_m"), original.get("map_half_extent"), protocol.get("map_half_extent")),
        "precision": first(nested(report, "runtime", "trainer_precision"), original.get("precision"), protocol.get("precision")),
        "float32_matmul_precision": first(nested(report, "runtime", "float32_matmul_precision"), original.get("float32_matmul_precision"), protocol.get("float32_matmul_precision")),
        "sampling_steps": first(original.get("sampling_steps"), nested(report, "decoder", "sampling_steps"), decoder.get("sampling_steps")),
        "base_train_offset": first(original.get("base_train_offset"), protocol.get("base_train_offset"), 0),
        "conditioning": nested(report, "decoder", "conditioning"),
        "metric_backend_sha256": digest(report["official_source_sha256"]) if report.get("official_source_sha256") else None,
        "reference_source_manifest_sha256": report.get("reference_source_manifest_sha256"),
        "learning_rate": finite(first(original.get("lr"), init.get("lr"), nested(config, "model", "model_config", "lr"), protocol.get("lr"))),
        "gradient_clip_val": finite(first(original.get("gradient_clip_val"), init.get("gradient_clip_val"), nested(config, "trainer", "gradient_clip_val"), protocol.get("gradient_clip"))),
        "evaluation_only": bool(original.get("evaluation_only")),
        "optimizer_continued": bool(first(original.get("optimizer_continued"), init.get("optimizer_continued"), False)),
        "resume_global_step": first(original.get("resume_global_step"), init.get("resume_global_step"), 0),
        "checkpoint": original.get("checkpoint"), "component_loss": loss,
        "recipe_sha256": digest(loss), "train_seconds": finite(original.get("train_seconds")),
        "eval_seconds": finite(original.get("eval_seconds")), **metrics,
    }
    for key in ("num_generated_vehicles", "num_gt_vehicles", "num_generated_onroad_vehicles", "num_gt_onroad_vehicles", "generated_onroad_fraction", "gt_onroad_fraction"):
        observation[key] = finite(first(original.get(key), report.get(key)))
    for label in ("generated", "gt"):
        vehicles, onroad = observation[f"num_{label}_vehicles"], observation[f"num_{label}_onroad_vehicles"]
        if vehicles and onroad is not None:
            derived = onroad / vehicles
            recorded = observation[f"{label}_onroad_fraction"]
            if recorded is not None and not math.isclose(derived, recorded, abs_tol=1e-10):
                warnings.append(f"Coverage count/fraction mismatch: {source} / {variant}")
            observation[f"{label}_onroad_fraction"] = derived
    observation["vehicle_count_ratio"] = ratio(observation["num_generated_vehicles"], observation["num_gt_vehicles"])
    observation["S"] = score(metrics)
    observation["cohort"] = {key: observation[key] for key in COHORT_FIELDS}
    observation["cohort_sha256"] = digest(observation["cohort"])
    # LR, clip, EMA configuration and architecture/representation changes are
    # search arms, not independent observations or silently merged recipes.
    model_config = nested(config, "model", "model_config") or {}
    optimizer_context = {key: value for key, value in model_config.items()
                         if key.startswith("lr_") or key in ("optimizer", "weight_decay", "optimizer_name", "betas", "eps")}
    relevant_config = {"decoder": nested(config, "model", "model_config", "decoder") or decoder,
                       "lr": observation["learning_rate"], "clip": observation["gradient_clip_val"],
                       "optimizer_context": optimizer_context, "calibration_sha256": calibration_sha256,
                       "size_representation": observation["size_representation"],
                       "velocity_representation": observation["velocity_representation"]}
    observation["arm_config_sha256"] = digest(relevant_config)
    observation["arm_id"] = observation["variant"] + "@" + observation["arm_config_sha256"][:12]
    return observation


def stage_in_scope(path, exclude_stage_names=EXCLUDED_NON_MAIN_STAGES, only_stage_names=None):
    # Optional audits own nested protocol roots; exclusion must cover their
    # descendants, not just the immediate protocol-root basename.
    parts = Path(path).parts
    matches = lambda patterns: any(fnmatch.fnmatchcase(part, pattern) for part in parts for pattern in patterns)
    return not matches(exclude_stage_names) and (only_stage_names is None or matches(only_stage_names))


def discover(runs, explicit_summaries=(), *, exclude_stage_names=EXCLUDED_NON_MAIN_STAGES, only_stage_names=None):
    observations, warnings, protocols = [], [], {}
    # Only roots that own protocol.json count.  In particular, evaluator JSON
    # and nested files with summary-like names never become extra records.
    summaries = {path.resolve() for run in runs for path in Path(run).rglob("summary.json") if (path.parent / "protocol.json").is_file()}
    summaries.update(Path(path).resolve() for path in explicit_summaries)
    for source in sorted(summaries):
        if not stage_in_scope(source.parent, exclude_stage_names, only_stage_names):
            continue
        protocol_path = source.parent / "protocol.json"
        if not protocol_path.is_file():
            raise ValueError(f"Control summary must belong to an exact protocol stage: {source}")
        protocol = read_json(protocol_path, {}, warnings)
        protocols[str(source.parent)] = protocol
        for original in rows_from_summary(read_json(source, [], warnings)):
            observation = enrich(original, protocol, source.parent, source, warnings)
            if observation is not None:
                observations.append(observation)
    unique = {}
    for row in observations:
        identity = tuple(row[key] for key in ("cohort_sha256", "budget", "train_batch_size", "seen_training_examples", "evaluation_weights", "arm_id", "seed", "eval_seed"))
        identity += tuple(row[key] for key in (*METRICS, "num_generated_vehicles", "num_generated_onroad_vehicles"))
        if identity in unique:
            warnings.append(f"Duplicate identical summary observation omitted: {row['summary_source']} / {row['variant']}")
        else:
            unique[identity] = row
    return list(unique.values()), warnings, protocols


VALUES = (*METRICS, "S", "generated_onroad_fraction", "gt_onroad_fraction", "vehicle_count_ratio", "num_generated_vehicles", "num_gt_vehicles")


def training_seed_means(rows):
    """Never count multiple GEN repetitions as independent training N."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["seed"], row["eval_seed"])].append(row)
    by_seed = defaultdict(dict)
    for (seed, generation), values in cells.items():
        by_seed[seed][generation] = {key: mean_complete(row.get(key) for row in values) for key in VALUES}
    result = {}
    for seed, generations in by_seed.items():
        result[seed] = {key: mean_complete(value[key] for value in generations.values()) for key in VALUES}
        result[seed]["generation_seeds"] = sorted(generations)
        result[seed]["num_generation_cells"] = len(generations)
    return result


def arm_statistics(rows):
    per_seed = training_seed_means(rows)
    native = {key: stats(value[key] for value in per_seed.values()) for key in VALUES}
    raw = {key: ({name: value if name == "n" else value / MULTIPLIERS[key] for name, value in native[key].items()} if native[key] else None) for key in METRICS}
    return {"arm_id": rows[0]["arm_id"], "variant": rows[0]["variant"],
            "arm_config_sha256": rows[0]["arm_config_sha256"], "recipe_sha256": rows[0]["recipe_sha256"],
            "component_loss": rows[0]["component_loss"], "learning_rate": rows[0]["learning_rate"],
            "gradient_clip_val": rows[0]["gradient_clip_val"], "training_seed_ids": sorted(per_seed),
            "num_independent_training_seeds": len(per_seed), "num_actual_observations": len(rows),
            "generation_seeds_by_training_seed": {str(seed): value["generation_seeds"] for seed, value in per_seed.items()},
            "statistics_native": native, "metrics_raw_jsd_collision_percent": raw,
            "per_training_seed": {str(seed): value for seed, value in per_seed.items()}}


def compare(candidate_rows, control_rows, external=False, *, include_confirmation_subset=True):
    candidate, control = training_seed_means(candidate_rows), training_seed_means(control_rows)
    paired = sorted(set(candidate) & set(control))
    paired_ratios, per_seed, components = [], [], {}
    for seed in paired:
        # GEN has already been averaged within this independent training seed.
        c, b = candidate[seed], control[seed]
        ratios = {key: ratio(c[key], b[key]) for key in METRICS}
        r, r_attr = mean_complete(ratios.values()), mean_complete(ratios[key] for key in ATTR)
        paired_ratios.append({**ratios, "R": r, "R_attr": r_attr, "S_delta": c["S"] - b["S"]})
        seed_coverage_known = all(finite(value) is not None for value in (c["generated_onroad_fraction"], b["generated_onroad_fraction"], c["vehicle_count_ratio"]))
        seed_checks = {
            "onroad": seed_coverage_known and c["generated_onroad_fraction"] >= b["generated_onroad_fraction"] - GATES["onroad_drop_fraction"],
            "vehicle_ratio": seed_coverage_known and GATES["vehicle_ratio_min"] <= c["vehicle_count_ratio"] <= GATES["vehicle_ratio_max"],
            **{key: ratios[key] is not None and ratios[key] <= GATES["length_width_speed_max_control_ratio"] for key in ATTR},
        }
        per_seed.append({"training_seed": seed, "candidate": c, "control": b, "component_ratios": ratios,
                         "R": r, "R_attr": r_attr, "S_delta": c["S"] - b["S"],
                         "generation_seed_sets_equal": c["generation_seeds"] == b["generation_seeds"],
                         "gate_checks": seed_checks, "gates_pass": all(seed_checks.values())})
    candidate_means = {key: mean_complete(candidate[seed][key] for seed in paired) for key in VALUES}
    control_means = {key: mean_complete(control[seed][key] for seed in paired) for key in VALUES}
    component_ratios = {key: ratio(candidate_means[key], control_means[key]) for key in METRICS}
    for key in METRICS:
        diffs = [candidate[seed][key] - control[seed][key] for seed in paired]
        components[key] = {"ratio_of_arm_means": component_ratios[key],
                           "relative_change_percent_of_arm_means": 100 * (component_ratios[key] - 1) if component_ratios[key] is not None else None,
                           "paired_ratio_statistics": stats(item[key] for item in paired_ratios),
                           "paired_delta_native": stats(diffs), "candidate_lower_training_seed_count": sum(value < 0 for value in diffs),
                           "candidate_higher_training_seed_count": sum(value > 0 for value in diffs)}
    complete = bool(paired) and set(candidate) == set(control) and all(item["generation_seed_sets_equal"] for item in per_seed)
    road, base_road, vehicles = candidate_means["generated_onroad_fraction"], control_means["generated_onroad_fraction"], candidate_means["vehicle_count_ratio"]
    coverage_known = (all(finite(value) is not None for value in (road, base_road, vehicles))
                      and 0 <= road <= 1 and 0 <= base_road <= 1 and vehicles >= 0)
    checks = {
        "onroad": coverage_known and road >= base_road - GATES["onroad_drop_fraction"],
        "vehicle_ratio": coverage_known and GATES["vehicle_ratio_min"] <= vehicles <= GATES["vehicle_ratio_max"],
        **{key: component_ratios[key] is not None and component_ratios[key] <= GATES["length_width_speed_max_control_ratio"] for key in ATTR},
    }
    result = {"control_variant": "new_control", "control_arm_ids": sorted({row["arm_id"] for row in control_rows}),
            "external_control": external, "control_summary_sources": sorted({row["summary_source"] for row in control_rows}),
            "comparison_basis": ("EXACT same observed exposure and evaluation cohort; different B/update reference allowed, never described as same update" if external else "Same update/B/exposure/evaluation cohort"),
            "control_budgets": sorted({row["budget"] for row in control_rows}),
            "control_batch_sizes": sorted({row["train_batch_size"] for row in control_rows}),
            "num_paired_independent_training_seeds": len(paired), "paired_training_seed_ids": paired,
            "num_paired_adaptive_discovery_training_seeds": sum(seed in ADAPTIVE_DISCOVERY_SEEDS for seed in paired),
            "num_paired_frozen_pair_confirmation_training_seeds": sum(seed in FROZEN_PAIR_CONFIRMATION_SEEDS for seed in paired),
            "training_seed_roles": {str(seed): "adaptive discovery / long-stage selection" if seed in ADAPTIVE_DISCOVERY_SEEDS else "held until frozen-pair confirmation" if seed in FROZEN_PAIR_CONFIRMATION_SEEDS else "unspecified" for seed in paired},
            "generation_seed_pairing_complete": complete, "provisional": len(paired) < 3 or not complete,
            "components": components, "R": stats(item["R"] for item in paired_ratios),
            "R_attr": stats(item["R_attr"] for item in paired_ratios),
            "R_ratio_of_arm_means": mean_complete(component_ratios.values()),
            "R_attr_ratio_of_arm_means": mean_complete(component_ratios[key] for key in ATTR),
            "paired_S_delta": stats(item["S_delta"] for item in paired_ratios),
            "coverage_gate_inputs": {"candidate_onroad": road, "control_onroad": base_road,
                                     "candidate_vehicle_ratio": vehicles},
            "gate_checks": checks, "gates_pass": all(checks.values()),
            "training_seed_gate_pass_count": sum(item["gates_pass"] for item in per_seed),
            "ranking_eligible": complete and all(checks.values()) and bool(paired) and mean_complete(component_ratios.values()) is not None,
            "per_training_seed": per_seed}
    if include_confirmation_subset:
        confirmation_candidates = [row for row in candidate_rows if row["seed"] in FROZEN_PAIR_CONFIRMATION_SEEDS]
        confirmation_controls = [row for row in control_rows if row["seed"] in FROZEN_PAIR_CONFIRMATION_SEEDS]
        result["independent_confirmation_only_sensitivity"] = (
            compare(confirmation_candidates, confirmation_controls, external, include_confirmation_subset=False)
            if confirmation_candidates and confirmation_controls else None)
        if result["independent_confirmation_only_sensitivity"]:
            result["independent_confirmation_only_sensitivity"]["selection_role"] = "Additional sensitivity summary; declared primary remains all-three-training-seed arm-mean R."
    return result


def pareto(arms):
    eligible = [arm for arm in arms if arm.get("comparison") and arm["comparison"].get("ranking_eligible")]
    vectors = {arm["arm_id"]: [arm["comparison"]["components"][key]["ratio_of_arm_means"] for key in METRICS] for arm in eligible}
    output = []
    for name, vector in vectors.items():
        dominated = any(other != name and all(a <= b for a, b in zip(values, vector)) and any(a < b for a, b in zip(values, vector)) for other, values in vectors.items())
        if not dominated:
            output.append(name)
    return sorted(output)


def choose_controls(group, external_controls, policy="fill_missing"):
    """A supplied complete external reference can replace a partial local one.

    Default fills only missing TRAIN/GEN cells.  It refuses heterogeneous B/
    update baselines across seeds: a new_control method must remain one method.
    prefer_external instead uses the explicit coherent reference as a whole.
    """
    first_row = group[0]
    local = [row for row in group if row["variant"] == "new_control"]
    matching = [row for row in external_controls if row["variant"] == "new_control"
                and row["cohort_sha256"] == first_row["cohort_sha256"]
                and row["seen_training_examples"] == first_row["seen_training_examples"]
                and row["evaluation_weights"] == first_row["evaluation_weights"]]
    if policy == "prefer_external" and matching:
        return matching, True, "explicit_external_replaces_local"
    if not local:
        return matching, bool(matching), "explicit_external" if matching else "none"
    local_cells = {(row["seed"], row["eval_seed"]) for row in local}
    local_budgets = {(row["budget"], row["train_batch_size"]) for row in local}
    missing = [row for row in matching if (row["seed"], row["eval_seed"]) not in local_cells]
    compatible = [row for row in missing if (row["budget"], row["train_batch_size"]) in local_budgets]
    incompatible = [row for row in missing if (row["budget"], row["train_batch_size"]) not in local_budgets]
    note = "local_plus_explicit_missing_cells" if compatible else "local"
    if incompatible:
        note += "; rejected_missing_cells_with_different_control_B_update_use_prefer_external_for_coherent_reference"
    return local + compatible, bool(compatible), note


def build_groups(rows, external_controls=(), *, control_policy="fill_missing"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cohort_sha256"], row["budget"], row["train_batch_size"], row["seen_training_examples"], row["evaluation_weights"])].append(row)
    output = []
    for signature, group in sorted(grouped.items()):
        cohort, budget, batch, seen, weights = signature
        control, external, control_source_policy = choose_controls(group, external_controls, control_policy)
        controls_by_config = {row["arm_config_sha256"] for row in control}
        if len(controls_by_config) > 1:
            raise ValueError("Ambiguous new_control recipes at the same cohort/exposure. Supply one exact --control-summary stage.")
        control_cell_budgets = defaultdict(set)
        for row in control:
            control_cell_budgets[(row["seed"], row["eval_seed"])].add((row["budget"], row["train_batch_size"]))
        if any(len(budgets) > 1 for budgets in control_cell_budgets.values()):
            raise ValueError("Multiple external B/update references for the same TRAIN/GEN cell; select one exact control stage instead of averaging them.")
        if len({(row["budget"], row["train_batch_size"]) for row in control}) > 1:
            raise ValueError("Control B/update method differs across TRAIN seeds; supply one coherent exact-exposure baseline method.")
        control_pairs = {(row["seed"], row["eval_seed"]) for row in control}
        arm_rows = defaultdict(list)
        for row in group:
            arm_rows[row["arm_id"]].append(row)
        arms = []
        for _, values in sorted(arm_rows.items()):
            arm = arm_statistics(values)
            arm["comparison"] = compare(values, control, external) if control else None
            arms.append(arm)
        eligible = [arm for arm in arms if arm.get("comparison") and arm["comparison"]["ranking_eligible"]]
        ranking = sorted(eligible, key=lambda arm: (arm["comparison"]["R_ratio_of_arm_means"], arm["variant"], arm["arm_id"]))
        attr_ranking = sorted(eligible, key=lambda arm: (arm["comparison"]["R_attr_ratio_of_arm_means"], arm["variant"], arm["arm_id"]))
        output.append({"group_id": digest(signature), "cohort_sha256": cohort, "cohort": group[0]["cohort"],
                       "budget": budget, "train_batch_size": batch, "seen_training_examples_per_training_seed": seen,
                       "evaluation_weights": weights, "primary_ema": weights == "ema", "has_control": bool(control),
                       "external_control": external, "control_source_policy": control_source_policy,
                       "control_generation_cells": len(control_pairs),
                       "arm_statistics": arms, "ranking_eligible_by_R": [arm["arm_id"] for arm in ranking],
                       "ranking_eligible_by_R_attr": [arm["arm_id"] for arm in attr_ranking],
                       "pareto_eligible_seven_metrics": pareto(arms),
                       "best_available_arm_id": ranking[0]["arm_id"] if ranking else None,
                       "final_training_optimum_proven": False})
    return output


def annotate_observation_comparisons(rows, external_controls=(), *, control_policy="fill_missing"):
    """Save GEN-cell diagnostics in CSV; group arm-mean R remains primary."""
    groups = defaultdict(list)
    for row in rows:
        signature = tuple(row[key] for key in ("cohort_sha256", "budget", "train_batch_size", "seen_training_examples", "evaluation_weights"))
        groups[signature].append(row)
    for group in groups.values():
        references, external, note = choose_controls(group, external_controls, control_policy)
        for row in group:
            controls = [control for control in references if (control["seed"], control["eval_seed"]) == (row["seed"], row["eval_seed"])]
            ratios = {key: ratio(row[key], mean_complete(control[key] for control in controls)) for key in METRICS}
            row.update({"R_generation_cell_diagnostic": mean_complete(ratios.values()),
                        "R_attr_generation_cell_diagnostic": mean_complete(ratios[key] for key in ATTR),
                        "relative_control_component_ratios_generation_cell": ratios,
                        "control_generation_cell_external_group": external,
                        "control_source_policy": note,
                        "control_generation_cell_budgets": sorted({control["budget"] for control in controls}),
                        "control_generation_cell_batch_sizes": sorted({control["train_batch_size"] for control in controls})})


def cost_inventory(runs, rows, warnings, *, exclude_stage_names=EXCLUDED_NON_MAIN_STAGES, only_stage_names=None):
    stages, full_counts = [], {}
    by_folder = defaultdict(list)
    for row in rows:
        by_folder[str(Path(row["run_path"]) / row["variant"])].append(row)
    protocols = sorted({path.resolve() for run in runs for path in Path(run).rglob("protocol.json")})
    for protocol_path in protocols:
        run = protocol_path.parent
        if not stage_in_scope(run, exclude_stage_names, only_stage_names):
            continue
        protocol = read_json(protocol_path, {}, warnings)
        order = run / "train_order.pkl"
        full_count = None
        if order.is_file():
            try:
                with order.open("rb") as handle:
                    manifest = pickle.load(handle)
                full_count = len(manifest["files"] if isinstance(manifest, dict) else manifest)
            except (OSError, ValueError, KeyError, pickle.UnpicklingError) as error:
                warnings.append(f"Cannot read stage TRAIN order count: {order}: {error}")
        full_counts[str(run)] = full_count
        for init_path in sorted(run.glob("*/initialization.json")):
            folder = init_path.parent
            init = read_json(init_path, {}, warnings)
            records = by_folder[str(folder)]
            eval_only = bool(records and all(row["evaluation_only"] for row in records))
            if not records and init.get("actual_train_offset") is None and init.get("resumed_from"):
                eval_only = True
            continued = bool(init.get("optimizer_continued")) and not eval_only
            kind = "evaluation_only" if eval_only else "continuation" if continued else "fresh"
            trace = read_json(folder / "train_trace.json", None, warnings)
            trace_steps = sorted({int(item["step"]) for item in trace if isinstance(item, dict) and type(item.get("step")) is int}) if isinstance(trace, list) else []
            start = int(init.get("resume_global_step") or 0)
            updates = 0 if eval_only else len(trace_steps) if trace_steps else None
            contiguous = bool(trace_steps) and trace_steps == list(range(start, start + len(trace_steps)))
            if updates is not None and not eval_only and not contiguous:
                warnings.append(f"Noncontiguous actual trace, cost marked unknown: {folder}")
                updates = None
            attained = start if eval_only else start + updates if updates is not None else None
            planned = start if eval_only else protocol.get("steps")
            fit_complete = eval_only or attained == planned
            batch = init.get("batch_size", protocol.get("batch_size"))
            active_count = full_count - init["actual_train_offset"] if full_count and type(init.get("actual_train_offset")) is int else None
            examples = 0 if eval_only else exposure(updates, batch, active_count) if updates is not None else None
            timings = [row["train_seconds"] for row in records if row["train_seconds"] is not None]
            fit_seconds = 0. if eval_only else max(timings) if timings else None
            if len(set(timings)) > 1:
                warnings.append(f"Repeated fit timing differs; max counted once: {folder}")
            # A stage is complete only if both fit and every planned evaluator
            # endpoint/GEN/weight cell are present; running stages remain visible.
            milestones = [int(value) for value in str(protocol.get("checkpoint_steps", "")).split(",") if value.strip()]
            budgets = [start] if eval_only else sorted({planned, *milestones} - {None})
            budgets = [value for value in budgets if eval_only or value > start]
            modes = protocol.get("evaluation_weight_modes") or ["ema" if protocol.get("evaluate_ema") else "online"]
            generations = [protocol.get("eval_seed")]
            if protocol.get("extra_eval_seed") is not None:
                generations.append(protocol["extra_eval_seed"])
            expected = {(budget, generation, mode) for budget in budgets for generation in generations for mode in modes}
            actual = {(row["budget"], row["eval_seed"], row["evaluation_weights"]) for row in records}
            complete = fit_complete and expected <= actual
            stages.append({"stage_path": str(folder), "variant": folder.name, "training_seed": init.get("seed"),
                           "stage_kind": kind, "status": "complete" if complete else "trained_evaluation_pending" if fit_complete else "partial_or_running",
                           "resume_global_step": start, "planned_global_endpoint": planned, "attained_actual_global_step": attained,
                           "exact_trace_optimizer_updates": updates, "exact_trace_sample_exposures": examples,
                           "batch_size": batch, "active_train_loader_examples": active_count,
                           "unique_metric_records": len(records), "expected_generation_weight_cells": len(expected),
                           "fit_wall_seconds_known_once": fit_seconds,
                           "eval_wall_seconds_known": sum(row["eval_seconds"] or 0. for row in records),
                           "trace_contiguous": contiguous if not eval_only else None,
                           "counted_source": "Actual contiguous saved trace; zero for eval-only. Unknown without trace, never planned steps * B.",
                           "resumed_from": init.get("resumed_from"), "initial_generator_sha256": init.get("initial_generator_sha256")})
    totals = {
        "observed_stage_count": len(stages),
        "completed_fresh_models": sum(item["stage_kind"] == "fresh" and item["status"] == "complete" for item in stages),
        "completed_continuation_stages": sum(item["stage_kind"] == "continuation" and item["status"] == "complete" for item in stages),
        "evaluation_only_stages": sum(item["stage_kind"] == "evaluation_only" for item in stages),
        "partial_stages": sum(item["status"] != "complete" for item in stages),
        "known_actual_trace_optimizer_updates": sum(item["exact_trace_optimizer_updates"] or 0 for item in stages),
        "known_actual_trace_sample_exposures": sum(item["exact_trace_sample_exposures"] or 0 for item in stages),
        "stages_without_exact_update_cost": sum(item["exact_trace_optimizer_updates"] is None for item in stages),
        "fit_wall_seconds_known_once_per_stage": sum(item["fit_wall_seconds_known_once"] or 0. for item in stages),
        "fit_stages_without_wall_time": sum(item["stage_kind"] != "evaluation_only" and item["fit_wall_seconds_known_once"] is None for item in stages),
        "evaluation_wall_seconds_known_unique_records": sum(item["eval_wall_seconds_known"] for item in stages),
        "unique_successful_observations": len(rows),
        "scope": "Known recorded stages only; excludes setup, preprocessing, unrecorded failure time and CPU analysis. These are fit/test wall times, not measured GPU busy time.",
    }
    return {"totals": totals, "stages": stages}


def format_value(value, digits=4):
    return "—" if finite(value) is None else f"{value:.{digits}f}"


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["variant", "budget", "seed", "eval_seed"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def markdown(summary):
    lines = ["# 固定 map encoder 的 InitDiffusion 搜索汇总", "",
             f"实际有效记录：{summary['num_observations']}；分层组：{len(summary['groups'])}。自动读取当前已完成 summary，因此运行中报告可刷新。", "",
             "主评价为 EMA。GEN 先在 TRAIN seed 内平均，N 和 sample SD(ddof=1)只跨独立 TRAIN seeds；N=1 的 SD=0 是描述值，不是无不确定性。", "",
             "S=mean(native metric/[1,1,10,20,15,20,5])；R=七项相对 new_control 的 ratio 平均；R_attr=length/width/speed 三项 ratio 平均，均越低越好。R 的排序使用 paired TRAIN seed 后的 arm-mean ratios，另完整保存逐 seed ratio/SD。", "",
             "筛选门槛：onroad≥同组 control−0.02；V/GT∈[0.9,1.1]；L/W/speed 各项≤control×1.15。无匹配 control/覆盖 metadata/GEN seed pairing 时不排名。不同 evaluation cohort 不混排名。", "",
             "单位：nearest/lat 为 JSD×10；angle/length/width/speed 为 JSD×100；collision 为百分比。JSON 同时保存 raw JSD；S/R 都是搜索辅助分数，不是论文指标。", ""]
    if not summary["groups"]:
        lines += ["尚无完整评价 summary；已列出训练 stage 的当前状态和未知成本。", ""]
    for group in sorted(summary["groups"], key=lambda value: (not value["primary_ema"], value["cohort_sha256"], value["seen_training_examples_per_training_seed"], value["budget"], value["train_batch_size"])):
        lines += [f"## {group['evaluation_weights']} / B{group['train_batch_size']} / updates {group['budget']} / exposure {group['seen_training_examples_per_training_seed']:,}", "",
                  f"cohort `{group['cohort_sha256'][:12]}`，eval manifest `{str(group['cohort']['eval_manifest_sha256'])[:12]}`，{group['cohort']['num_samples']} 场景。",
                  "", "| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for arm in sorted(group["arm_statistics"], key=lambda value: (value["comparison"]["R_ratio_of_arm_means"] if value["comparison"] and value["comparison"]["R_ratio_of_arm_means"] is not None else math.inf, value["variant"])):
            native, comparison = arm["statistics_native"], arm["comparison"]
            s = native["S"]
            road, vehicles = native["generated_onroad_fraction"], native["vehicle_count_ratio"]
            gate = "no control" if not comparison else "PASS" if comparison["gates_pass"] else "FAIL"
            if comparison and not comparison["generation_seed_pairing_complete"]:
                gate += " / incomplete pairing"
            if comparison and comparison["provisional"]:
                gate += " / provisional"
            lines.append(f"| {arm['variant']} `{arm['arm_config_sha256'][:6]}` | {arm['num_independent_training_seeds']} | {format_value(s['mean'])}±{format_value(s['sd'])} | {format_value(comparison['R_ratio_of_arm_means'] if comparison else None)} | {format_value(comparison['R_attr_ratio_of_arm_means'] if comparison else None)} | {format_value(100*road['mean'] if road else None,2)}% | {format_value(vehicles['mean'] if vehicles else None,3)} | {gate} |")
        lines += ["", "**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：", "",
                  "| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in group["arm_statistics"]:
            comparison = arm["comparison"]
            values = [format_value(comparison["components"][key]["relative_change_percent_of_arm_means"], 2) + "%" if comparison else "—" for key in METRICS]
            lines.append("| " + " | ".join([arm["variant"], *values, str(comparison["num_paired_independent_training_seeds"] if comparison else 0)]) + " |")
        if group["external_control"]:
            lines += ["", "本组使用明确 --control-summary 的 **相同 actual exposure** 基线；B/update 可以不同，绝不称为同 update 比较。"]
        lines += ["", "通过 gate 的 R 排名：" + (" → ".join(group["ranking_eligible_by_R"]) or "无"),
                  "", "通过 gate 的七指标 Pareto：" + (", ".join(group["pareto_eligible_seven_metrics"]) or "无"), ""]
    totals = summary["cost_inventory"]["totals"]
    lines += ["## 实际成本与当前状态", "",
              f"完成 fresh {totals['completed_fresh_models']}，continuation {totals['completed_continuation_stages']}，eval-only {totals['evaluation_only_stages']}；partial {totals['partial_stages']}。", "",
              f"已保存连续 trace 确认的实际 updates {totals['known_actual_trace_optimizer_updates']:,}；实际 TRAIN cache exposures {totals['known_actual_trace_sample_exposures']:,}。{totals['stages_without_exact_update_cost']} stages 仍无精确更新记录，不用 planned budget 冒充。", "",
              f"已知去重 fit {totals['fit_wall_seconds_known_once_per_stage']/60:.2f} 分钟，test {totals['evaluation_wall_seconds_known_unique_records']/60:.2f} 分钟；不包含 setup/preprocess/未知失败耗时，不是 GPU busy 时间。", "",
              "| stage | kind | status | actual updates | actual samples | metric rows |", "|---|---|---|---:|---:|---:|"]
    for stage in summary["cost_inventory"]["stages"]:
        lines.append(f"| {Path(stage['stage_path']).parent.name}/{stage['variant']} | {stage['stage_kind']} | {stage['status']} | {stage['exact_trace_optimizer_updates']} | {stage['exact_trace_sample_exposures']} | {stage['unique_metric_records']} |")
    excluded = summary["excluded_pilot_diagnostics"]
    pilot_totals = excluded["cost_inventory"]["totals"]
    lines += ["", "## 排除的 worker benchmark 诊断", "",
              "`worker2_benchmark` 是 DataLoader workers=2 性能 pilot，默认不计入主模型数、主 metric rows 或任何模型配方排名。实际记录和成本保留如下，不是丢弃实验。", "",
              f"排除 pilot 有效 metric rows {excluded['num_observations']}，stage {pilot_totals['observed_stage_count']}；已知实际 updates {pilot_totals['known_actual_trace_optimizer_updates']:,} / sample exposures {pilot_totals['known_actual_trace_sample_exposures']:,}；fit {pilot_totals['fit_wall_seconds_known_once_per_stage']/60:.2f} 分钟、test {pilot_totals['evaluation_wall_seconds_known_unique_records']/60:.2f} 分钟。未保存 trace 的成本仍标 unknown。", "",
              "| pilot | status | actual updates | actual samples | metric rows |", "|---|---|---:|---:|---:|"]
    for stage in excluded["cost_inventory"]["stages"]:
        lines.append(f"| {Path(stage['stage_path']).parent.name}/{stage['variant']} | {stage['status']} | {stage['exact_trace_optimizer_updates']} | {stage['exact_trace_sample_exposures']} | {stage['unique_metric_records']} |")
    optional = summary["excluded_optional_comparison_diagnostics"]
    optional_totals = optional["cost_inventory"]["totals"]
    lines += ["", "## 独立保留的 map-policy comparison", "",
              "`map_policy_comparison_*` 及其所有 nested evaluator stages 是额外的随机训练地图/预训练固定地图对比，绝不用于主搜索排名、候选选择或主模型成本计数。独立成本仍记录，结果由各 comparison 目录的 comparison_summary.json 汇总。", "",
              f"有效 metric rows {optional['num_observations']}，eval-only stages {optional_totals['evaluation_only_stages']}；实际新增 TRAIN updates {optional_totals['known_actual_trace_optimizer_updates']:,} / exposures {optional_totals['known_actual_trace_sample_exposures']:,}，已知 test {optional_totals['evaluation_wall_seconds_known_unique_records']/60:.2f} 分钟。", ""]
    lines += ["", "## 解释范围", "", "所有排名只描述已测试 budget/cohort。quick trend 无法证明 128 epoch 最终最优；新4096确认集仍位于之前审计过的50k池中，不是 pristine unseen test。", "",
              "817 是自适应搜索及long阶段选择 seed；1817/2817 两个 TRAIN seeds 在冻结候选对之后才用于确认。最终可合法报告总 TRAIN N=3，但只有 **2 个独立确认 seeds**，不能写成3次独立确认。主分数仍按全部3个TRAIN等权计算，同时 JSON 保留仅1817/2817的敏感性汇总。每 TRAIN 有2个 GEN 重复时也不是 N=6。", "",
              "paired summary 已保存在各 group 的 comparison/per_training_seed，independent_confirmation_only_sensitivity 是额外核验，不替代事先声明的 all-three primary。", ""]
    if summary["warnings"]:
        lines += ["## 记录警告", "", *["- " + value for value in summary["warnings"]], ""]
    return "\n".join(lines)


def run_analysis(runs, output, control_summaries=(), *, control_policy="fill_missing"):
    rows, warnings, _ = discover(runs)
    controls, control_warnings, _ = discover([], control_summaries) if control_summaries else ([], [], {})
    warnings += control_warnings
    groups = build_groups(rows, controls, control_policy=control_policy)
    annotate_observation_comparisons(rows, controls, control_policy=control_policy)
    inventory = cost_inventory(runs, rows, warnings)
    pilot_rows, pilot_warnings, _ = discover(runs, exclude_stage_names=(), only_stage_names=EXCLUDED_PILOT_STAGES)
    pilot_inventory = cost_inventory(runs, pilot_rows, pilot_warnings, exclude_stage_names=(), only_stage_names=EXCLUDED_PILOT_STAGES)
    optional_rows, optional_warnings, _ = discover(runs, exclude_stage_names=(), only_stage_names=EXCLUDED_OPTIONAL_STAGES)
    optional_inventory = cost_inventory(runs, optional_rows, optional_warnings, exclude_stage_names=(), only_stage_names=EXCLUDED_OPTIONAL_STAGES)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_version": 1,
               "input_roots": [str(Path(path).resolve()) for path in runs],
               "explicit_control_summaries": [str(Path(path).resolve()) for path in control_summaries],
               "control_policy": control_policy,
               "num_observations": len(rows), "primary_evaluation_weights": "ema",
               "metric_native_divisors_for_raw_jsd": MULTIPLIERS, "S_native_scales": SCALES,
               "R_definition": "Mean of seven candidate/control metric ratios; sort by paired-training arm-mean ratios.",
               "R_attr_definition": "Mean of length/width/speed candidate/control ratios.",
               "gates": GATES, "training_N_rule": "Mean within GEN cell; mean GEN inside TRAIN; independent TRAIN mean/sample SD ddof=1.",
               "training_seed_roles": {"adaptive_discovery_and_long_selection": ADAPTIVE_DISCOVERY_SEEDS,
                    "held_until_frozen_pair_confirmation": FROZEN_PAIR_CONFIRMATION_SEEDS,
                    "total_distinct_training_seeds_when_complete": 3, "independent_confirmation_training_seeds_when_complete": 2,
                    "primary": "All-three-training-seed arm-mean R; confirmation-only two-seed summary is sensitivity, not a changed selection rule."},
               "groups": groups, "cost_inventory": inventory, "warnings": warnings,
               "excluded_pilot_diagnostics": {"stage_names": EXCLUDED_PILOT_STAGES,
                    "reason": "DataLoader worker performance pilot, not a main model-selection trial.",
                    "num_observations": len(pilot_rows), "actual_observations": pilot_rows,
                    "cost_inventory": pilot_inventory, "warnings": pilot_warnings},
               "excluded_optional_comparison_diagnostics": {"stage_patterns": EXCLUDED_OPTIONAL_STAGES,
                    "reason": "Optional map-policy audit is excluded from primary selection and primary training/metric counts.",
                    "num_observations": len(optional_rows), "actual_observations": optional_rows,
                    "cost_inventory": optional_inventory, "warnings": optional_warnings},
               "no_global_cross_cohort_ranking": True, "full_convergence_optimum_proven": False,
               "cuda_or_model_imported": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_observations.csv", rows)
    (output / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "search_summary.md").write_text(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "logs/component_loss_frozen_map"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011")
    parser.add_argument("--control-summary", type=Path, action="append", default=[], help="Exact protocol-root new_control summary; equal actual exposure only, never cross evaluation cohorts")
    parser.add_argument("--control-policy", choices=("fill_missing", "prefer_external"), default="fill_missing",
                        help="Fill missing local TRAIN/GEN controls at the same B/update, or replace partial local controls with a coherent explicit exact-exposure reference")
    args = parser.parse_args()
    summary = run_analysis(args.runs, args.output, args.control_summary, control_policy=args.control_policy)
    print(json.dumps({"observations": summary["num_observations"], "groups": len(summary["groups"]),
                      "warnings": len(summary["warnings"]), "cost": summary["cost_inventory"]["totals"]}, indent=2))


if __name__ == "__main__":
    main()
