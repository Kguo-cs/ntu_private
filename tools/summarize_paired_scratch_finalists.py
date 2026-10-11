"""CPU-only paired finalist summary with a frozen EMA endpoint decision.

Input is analyze_scratch_loss_search.py's observations.json, never lossy
aggregate records. Repeated observations are averaged inside generation seed;
generation seeds are averaged inside independent training seed. All reported
SDs across training seeds are sample SD (ddof=1, or 0 for N=1), not CIs.

Example (after all heldout evaluations have completed):
  python tools/summarize_paired_scratch_finalists.py \
    --observations path/to/heldout/observations.json \
    --protocol docs/experiments/init_diffusion_scratch_loss_search_20261011/final_deployment_selection_protocol.json \
    --candidate geo_mse32_col20_heading_l1_shape2 \
    --output path/to/paired_finalists

EMA2048 is the only deployment selection endpoint. Online rows and other
budgets are diagnostics; larger overlapping full-50k audits cannot reselect.
Incomplete input is rejected unless --allow-incomplete is supplied; a
provisional primary summary never selects a deployment recipe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCALES = {
    "nearest_dist_jsd": 1., "lat_dev_jsd": 1., "ang_dev_jsd": 10.,
    "length_jsd": 20., "width_jsd": 15., "speed_jsd": 20., "collision_rate": 5.,
}
MULTIPLIERS = {key: 10 if key in tuple(SCALES)[:2] else 100 for key in tuple(SCALES)[:-1]}
MULTIPLIERS["collision_rate"] = 1
VALUES = (*SCALES, "score", "generated_onroad_fraction", "vehicle_count_ratio",
          "num_generated_vehicles", "num_gt_vehicles")
COHORT_KEYS = (
    "train_batch_size", "learning_rate", "reference_mode", "num_samples",
    "eval_manifest_sha256", "eval_set_sha256", "initialization_scope",
    "calibration_sha256", "sampling_steps", "map_crop", "map_half_extent",
    "precision", "float32_matmul_precision",
    "base_train_offset",
)


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stats(values):
    values = [float(value) for value in values if finite(value) is not None]
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.,
            "minimum": min(values), "maximum": max(values)}


def percent(candidate, baseline):
    return 100 * (candidate / baseline - 1) if baseline != 0 else None


def scaled_stats(value, divisor):
    return {key: number if key == "n" else number / divisor
            for key, number in value.items()} if value else None


def read_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get("frozen_before_heldout_runs") is not True:
        raise ValueError("Protocol must have been frozen before heldout runs")
    if protocol.get("primary_weights") != "ema" or protocol.get("endpoint_budget") != 2048:
        raise ValueError("This confirmation utility requires the frozen EMA2048 endpoint")
    if protocol.get("primary_score_scales") != SCALES:
        raise ValueError("Frozen score scales differ from actual helper metric units")
    for key in ("training_seeds", "generation_seeds"):
        seeds = protocol.get(key)
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"Protocol needs explicit integer {key}")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Protocol {key} must not contain duplicates")
    # These machine-readable gates implement the rule frozen in the report's
    # protocol. If later protocols expose gates explicitly they must agree.
    gates = {"onroad_drop_fraction": .02, "vehicle_ratio_min": .9, "vehicle_ratio_max": 1.1}
    if protocol.get("coverage_gates", gates) != gates:
        raise ValueError("Coverage gates do not match the frozen confirmation rule")
    rule = str(protocol.get("selection_rule", "")).replace(" ", "")
    if not all(token in rule for token in ("0.02", "0.9", "1.1")):
        raise ValueError("Frozen textual rule must explicitly state the coverage gates")
    return protocol, gates


def validate_row(row):
    if isinstance(row.get("score"), dict):
        raise ValueError("Use observations.json; aggregate means lose paired generation-seed evidence")
    if type(row.get("budget")) is not int or row["budget"] <= 0:
        raise ValueError("Each observation needs its actual positive optimizer-step budget")
    for key in VALUES:
        value = finite(row.get(key))
        if value is None or value < 0:
            raise ValueError(f"Nonfinite/missing/negative {key} for {row.get('variant')}")
    if row["generated_onroad_fraction"] > 1 or row["num_gt_vehicles"] <= 0:
        raise ValueError("Invalid on-road fraction or GT vehicle denominator")
    if row.get("num_gt_samples") is not None and row["num_gt_samples"] != row.get("num_samples"):
        raise ValueError("Generated and matched-GT evaluation scene counts differ")
    computed_score = statistics.fmean(row[key] / SCALES[key] for key in SCALES)
    if not math.isclose(row["score"], computed_score, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Stored score does not match the frozen seven-metric scales")
    ratio = row["num_generated_vehicles"] / row["num_gt_vehicles"]
    if not math.isclose(row["vehicle_count_ratio"], ratio, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Stored vehicle ratio does not match generated/GT counts")
    if not row.get("recipe_sha256") or not row.get("cohort_sha256") or not row.get("eval_manifest_sha256"):
        raise ValueError("Each observation needs recipe, cohort and evaluation-manifest identities")


def select_rows(rows, variants, protocol, samples, batch_size, learning_rate, cohorts=()):
    """Keep endpoint plus trend rows; never silently combine evaluation cohorts."""
    selected, exclusions = [], defaultdict(int)
    for row in rows:
        if row.get("variant") not in variants:
            exclusions["other_variant"] += 1
        elif row.get("num_samples") != samples:
            exclusions["other_evaluation_sample_count"] += 1
        elif cohorts and not any(str(row.get("cohort_sha256", "")).startswith(key) for key in cohorts):
            exclusions["other_requested_cohort"] += 1
        elif row.get("seed") not in protocol["training_seeds"] or row.get("eval_seed") not in protocol["generation_seeds"]:
            exclusions["other_training_or_generation_seed"] += 1
        elif row.get("evaluation_weights") not in ("ema", "online"):
            exclusions["other_weights"] += 1
        else:
            validate_row(row)
            if row.get("train_batch_size") != batch_size or not math.isclose(
                    finite(row.get("learning_rate")) or -1, learning_rate, rel_tol=1e-12):
                raise ValueError("Input confirmation batch size/LR differ from the requested production cohort")
            selected.append(row)
    if not selected:
        raise ValueError("No actual matching finalist observations")
    signatures = {json.dumps({key: row.get(key) for key in COHORT_KEYS}, sort_keys=True)
                  for row in selected}
    if len(signatures) != 1:
        raise ValueError("Mixed data/model/optimizer/precision cohorts; select exactly one heldout cohort")
    for variant in variants:
        hashes = {row["recipe_sha256"] for row in selected if row["variant"] == variant}
        if len(hashes) > 1:
            raise ValueError(f"Recipe changed across training seeds/budgets: {variant}")
    for weights in ("ema", "online"):
        hashes = {row["cohort_sha256"] for row in selected if row["evaluation_weights"] == weights}
        if len(hashes) > 1:
            raise ValueError(f"Mixed analyzer cohort identities for {weights}; do not pool them")
    return selected, dict(exclusions), json.loads(next(iter(signatures)))


def make_cells(rows, budget, weights, variants, protocol, warnings):
    cells = defaultdict(list)
    for row in rows:
        if row["budget"] == budget and row["evaluation_weights"] == weights:
            cells[(row["variant"], row["seed"], row["eval_seed"])].append(row)
    expected = [(variant, seed, generation) for variant in variants
                for seed in protocol["training_seeds"] for generation in protocol["generation_seeds"]]
    missing = [list(cell) for cell in expected if cell not in cells]
    output = {}
    for cell, observations in cells.items():
        # Identical copies from summary/report recovery are not new repeats.
        unique = {tuple(float(row[key]) for key in VALUES): row for row in observations}
        if len(unique) < len(observations):
            warnings.append(f"Identical copies deduplicated in cell {cell} ({len(observations)} -> {len(unique)})")
        if len(unique) > 1:
            warnings.append(f"Nonidentical repeats averaged within generation cell {cell}: {len(unique)}")
        output[cell] = {key: statistics.fmean(row[key] for row in unique.values()) for key in VALUES}
    return output, missing, len(cells), sum(len(value) for value in cells.values())


def training_seed_means(cells, variants, protocol):
    result = {variant: {} for variant in variants}
    for variant in variants:
        for seed in protocol["training_seeds"]:
            keys = [(variant, seed, generation) for generation in protocol["generation_seeds"]]
            if not all(key in cells for key in keys):
                continue
            result[variant][seed] = {key: statistics.fmean(cells[cell][key] for cell in keys) for key in VALUES}
    return result


def summarize_mode(cells, missing, weights, variants, protocol, gates):
    seed_values = training_seed_means(cells, variants, protocol)
    baseline, candidate = variants
    paired_seeds = [seed for seed in protocol["training_seeds"]
                    if all(seed in seed_values[variant] for variant in variants)]
    arms, comparisons, per_seed = {}, {}, []
    for variant in variants:
        values = seed_values[variant]
        arms[variant] = {"num_independent_training_seeds": len(values),
                         "training_seed_ids": list(values),
                         "statistics_native": {key: stats(item[key] for item in values.values()) for key in VALUES}}
        arms[variant]["metrics_raw_jsd_collision_percent"] = {
            key: scaled_stats(arms[variant]["statistics_native"][key], MULTIPLIERS[key]) for key in SCALES}
    for key in VALUES:
        deltas = [seed_values[candidate][seed][key] - seed_values[baseline][seed][key] for seed in paired_seeds]
        percent_changes = [percent(seed_values[candidate][seed][key], seed_values[baseline][seed][key])
                           for seed in paired_seeds]
        item = {"absolute_delta_native": stats(deltas),
                "candidate_lower_count": sum(delta < 0 for delta in deltas),
                "candidate_equal_count": sum(delta == 0 for delta in deltas),
                "candidate_higher_count": sum(delta > 0 for delta in deltas)}
        if key == "generated_onroad_fraction":
            item["absolute_delta_percentage_points"] = stats(100 * delta for delta in deltas)
            item["direction"] = "higher coverage is preferable; compare absolute percentage points"
        else:
            item["paired_percent_change"] = stats(percent_changes)
            item["zero_baseline_denominator_count"] = sum(value is None for value in percent_changes)
            base_mean, candidate_mean = arms[baseline]["statistics_native"][key], arms[candidate]["statistics_native"][key]
            item["percent_change_of_arm_means"] = (percent(candidate_mean["mean"], base_mean["mean"])
                                                   if base_mean and candidate_mean else None)
        if key in SCALES:
            item["absolute_delta_raw_jsd_collision_percent"] = scaled_stats(item["absolute_delta_native"], MULTIPLIERS[key])
        comparisons[key] = item
    for seed in paired_seeds:
        item = {"seed": seed, "generation_seeds": protocol["generation_seeds"],
                "baseline": seed_values[baseline][seed], "candidate": seed_values[candidate][seed],
                "generation_ranges": {}, "paired": {}}
        for key in VALUES:
            base, chosen = item["baseline"][key], item["candidate"][key]
            item["paired"][key] = {"absolute_delta_native": chosen - base,
                "absolute_delta_percentage_points": 100 * (chosen - base) if key == "generated_onroad_fraction" else None,
                "percent_change": None if key == "generated_onroad_fraction" else percent(chosen, base)}
            generated = {variant: [cells[(variant, seed, generation)][key] for generation in protocol["generation_seeds"]]
                         for variant in variants}
            item["generation_ranges"][key] = {
                "baseline": {"minimum": min(generated[baseline]), "maximum": max(generated[baseline])},
                "candidate": {"minimum": min(generated[candidate]), "maximum": max(generated[candidate])},
                "paired_delta": {"minimum": min(chosen - base for base, chosen in zip(generated[baseline], generated[candidate])),
                                 "maximum": max(chosen - base for base, chosen in zip(generated[baseline], generated[candidate]))}}
        per_seed.append(item)
    complete = not missing
    primary = weights == protocol["primary_weights"]
    selection = {"performed": complete and primary, "selected_variant": None, "eligibility": {},
                 "rule": "Minimum mean S among eligible arms; tie broken by variant name"}
    if complete:
        base_onroad = arms[baseline]["statistics_native"]["generated_onroad_fraction"]["mean"]
        for variant in variants:
            summary = arms[variant]["statistics_native"]
            road, ratio = summary["generated_onroad_fraction"]["mean"], summary["vehicle_count_ratio"]["mean"]
            passes_road = road >= base_onroad - gates["onroad_drop_fraction"]
            passes_ratio = gates["vehicle_ratio_min"] <= ratio <= gates["vehicle_ratio_max"]
            selection["eligibility"][variant] = {"eligible": passes_road and passes_ratio,
                "passes_onroad_gate": passes_road, "passes_vehicle_ratio_gate": passes_ratio,
                "mean_onroad_fraction": road, "minimum_onroad_fraction": base_onroad - gates["onroad_drop_fraction"],
                "mean_vehicle_count_ratio": ratio, "mean_score": summary["score"]["mean"]}
        eligible = [variant for variant in variants if selection["eligibility"][variant]["eligible"]]
        if primary and eligible:
            selection["selected_variant"] = min(eligible, key=lambda variant: (arms[variant]["statistics_native"]["score"]["mean"], variant))
    return {"evaluation_weights": weights, "budget": protocol["endpoint_budget"], "primary_deployment_endpoint": primary,
            "complete": complete, "missing_cells": missing, "num_paired_training_seeds": len(paired_seeds),
            "generation_seeds_per_training_seed": len(protocol["generation_seeds"]),
            "arms": arms, "paired_comparisons": comparisons, "per_training_seed": per_seed, "selection": selection}


def training_costs(rows):
    """Deduplicate training stages; never sum repeated checkpoint/mode rows."""
    stages = defaultdict(list)
    for row in rows:
        if not row.get("evaluation_only") and finite(row.get("train_seconds")) is not None:
            stages[(row.get("run_path"), row["variant"])].append(row)
    records = [{"run_path": run, "variant": variant, "train_seconds": max(row["train_seconds"] for row in stage),
                "minimum_recorded_train_seconds": min(row["train_seconds"] for row in stage),
                "observation_rows": len(stage), "max_recorded_budget": max(row["budget"] for row in stage)}
               for (run, variant), stage in sorted(stages.items(), key=lambda item: str(item[0]))]
    return {"scope": "Only input finalist stages; not total search GPU cost. Maximum recorded elapsed train time per (run_path,variant).",
            "num_training_stages": len(records), "train_seconds": sum(row["train_seconds"] for row in records), "stages": records}


def build_summary(rows, protocol, gates, candidate, baseline="warm_winner", samples=4096,
                  batch_size=512, learning_rate=5e-4, cohorts=(), allow_incomplete=False):
    if candidate == baseline:
        raise ValueError("Candidate and baseline must be different recipe variants")
    variants = (baseline, candidate)
    selected, exclusions, cohort = select_rows(rows, variants, protocol, samples, batch_size, learning_rate, cohorts)
    modes, warnings = {}, []
    for weights in (protocol["primary_weights"], "online"):
        cells, missing, cell_count, observation_count = make_cells(selected, protocol["endpoint_budget"], weights, variants, protocol, warnings)
        if weights != protocol["primary_weights"] and not cells:
            continue
        if missing and not allow_incomplete:
            raise ValueError(f"Incomplete {weights}{protocol['endpoint_budget']} paired endpoint: {len(missing)} missing cells: {missing}")
        mode = summarize_mode(cells, missing, weights, variants, protocol, gates)
        mode.update(num_generation_cells=cell_count, num_input_observation_rows=observation_count)
        modes[weights] = mode
    result = {"baseline": baseline, "candidate": candidate, "primary_weights": protocol["primary_weights"],
              "endpoint_budget": protocol["endpoint_budget"], "training_seeds": protocol["training_seeds"],
              "generation_seeds": protocol["generation_seeds"], "cohort": cohort, "coverage_gates": gates,
              "primary_score_native_scales": SCALES, "metric_output_multipliers": MULTIPLIERS,
              "jsd_log_base": "natural", "raw_jsd_theoretical_upper_bound": math.log(2),
              "training_seed_sd_ddof": 1, "single_training_seed_sd": 0,
              "aggregation": "Equal generation means inside training seed, then equal independent training-seed means",
              "paired_percent_change": "Percent change of each paired training-seed mean, then equal mean and sample SD across training seeds; ratio of overall means is separate",
              "uncertainty": "Descriptive sample SD across independent training seeds, not CI; generation ranges separate, not extra training N",
              "exclusions": exclusions, "nonendpoint_budgets_diagnostic_only": sorted({row["budget"] for row in selected if row["budget"] != protocol["endpoint_budget"]}),
              "modes": modes, "warnings": sorted(set(warnings)), "input_training_cost": training_costs(selected)}
    result["source_observation_identities"] = [
        {key: row.get(key) for key in ("variant", "budget", "seed", "eval_seed", "evaluation_weights",
            "checkpoint", "report_source", "summary_source", "recipe_sha256", "cohort_sha256",
            "train_manifest_sha256", "initial_generator_sha256", "initial_map_sha256", "evaluation_only")}
        for row in selected]
    # Incomplete diagnostic online input must not suppress a complete primary
    # result, but any provisional primary input cannot make a decision.
    result["deployment_selection"] = modes[protocol["primary_weights"]]["selection"]
    result["selected_variant"] = result["deployment_selection"]["selected_variant"]
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(output, summary):
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    paired, arms = [], []
    for weights, mode in summary["modes"].items():
        for seed in mode["per_training_seed"]:
            for key in VALUES:
                paired.append({"weights": weights, "budget": mode["budget"], "training_seed": seed["seed"],
                    "feature": key, "baseline_native": seed["baseline"][key], "candidate_native": seed["candidate"][key],
                    **seed["paired"][key],
                    "baseline_generation_minimum": seed["generation_ranges"][key]["baseline"]["minimum"],
                    "baseline_generation_maximum": seed["generation_ranges"][key]["baseline"]["maximum"],
                    "candidate_generation_minimum": seed["generation_ranges"][key]["candidate"]["minimum"],
                    "candidate_generation_maximum": seed["generation_ranges"][key]["candidate"]["maximum"]})
        for variant, arm in mode["arms"].items():
            for key, value in arm["statistics_native"].items():
                if value:
                    arms.append({"weights": weights, "budget": mode["budget"], "variant": variant, "feature": key,
                                 "display_divisor": MULTIPLIERS.get(key, 1), **value})
    write_csv(output / "per_training_seed.csv", paired)
    write_csv(output / "arm_statistics.csv", arms)
    lines = ["# Paired scratch finalist confirmation", "",
             f"Primary endpoint: **{summary['primary_weights']} / {summary['endpoint_budget']} steps**. "
             f"Selected variant: **{summary['selected_variant'] or 'no selection'}**.", "",
             "Independent training seeds are the sample units. Generation seeds are averaged inside each training seed. "
             "SD uses ddof=1 (N=1: 0), is descriptive, and is not a confidence interval. "
             "Each training seed's generation range is recorded separately in JSON/CSV.", ""]
    for weights, mode in summary["modes"].items():
        lines += [f"## {weights}: {'primary' if mode['primary_deployment_endpoint'] else 'diagnostic only'}", "",
                  f"Complete endpoint: {mode['complete']}; paired training N={mode['num_paired_training_seeds']}; "
                  f"generation repetitions per training seed={mode['generation_seeds_per_training_seed']}.", "",
                  "JSD values below use natural logarithms, with theoretical maximum ln2 (helper nearest/lateral divided by10, other JSD by100); collision is percent.", "",
                  "| Metric | Control mean ± sample SD | Candidate mean ± sample SD | Paired change mean ± sample SD |", "|---|---:|---:|---:|"]
        for key in (*SCALES, "score", "generated_onroad_fraction", "vehicle_count_ratio"):
            divisor = MULTIPLIERS.get(key, 1)
            base = mode["arms"][summary["baseline"]]["statistics_native"][key]
            chosen = mode["arms"][summary["candidate"]]["statistics_native"][key]
            if not base or not chosen:
                continue
            paired = mode["paired_comparisons"][key]
            change = paired.get("absolute_delta_percentage_points") if key == "generated_onroad_fraction" else paired.get("paired_percent_change")
            delta_text = f"{change['mean']:+.2f} ± {change['sd']:.2f} {'pp' if key == 'generated_onroad_fraction' else '%'}" if change else "N/A (zero control denominator)"
            if key == "generated_onroad_fraction":
                divisor = .01
            lines.append(f"| {key} | {base['mean']/divisor:.5f} ± {base['sd']/divisor:.5f} | "
                         f"{chosen['mean']/divisor:.5f} ± {chosen['sd']/divisor:.5f} | {delta_text} |")
        if mode["missing_cells"]:
            lines += ["", f"Missing generation cells: `{mode['missing_cells']}`. No provisional primary decision."]
        lines.append("")
    lines += ["Only the complete frozen EMA endpoint may select deployment. Online results and other budgets cannot reselect. "
              "The larger full-50k audit is separate from this decision.", "",
              "The reported training cost deduplicates each (run_path, variant) stage using its maximum elapsed time; "
              "it covers only stages present in this input, not the whole search.", ""]
    (output / "paired_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="warm_winner")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--cohort", nargs="*", default=[])
    parser.add_argument("--allow-incomplete", action="store_true", help="Write explicitly provisional descriptive results; missing primary cells disable selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        protocol, gates = read_protocol(args.protocol)
        rows = json.loads(args.observations.read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Input must be a list of actual analyzer observations")
        summary = build_summary(rows, protocol, gates, args.candidate, args.baseline, args.samples,
                                args.batch_size, args.learning_rate, args.cohort, args.allow_incomplete)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    summary["provenance"] = {"observations": str(args.observations.resolve()),
        "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_protocol": protocol}
    save_outputs(args.output.resolve(), summary)
    print(json.dumps({"output": str(args.output.resolve()), "selected_variant": summary["selected_variant"],
                      "primary_complete": summary["modes"][protocol["primary_weights"]]["complete"],
                      "paired_training_N": summary["modes"][protocol["primary_weights"]]["num_paired_training_seeds"]}))


if __name__ == "__main__":
    main()
