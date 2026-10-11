"""Aggregate paired quick-sweep evaluations without importing PyTorch.

Examples:
    python tools/summarize_component_loss_confirmation.py \
        logs/component_loss_quick/confirmation_seed817 \
        logs/component_loss_quick/confirmation_seed1817 \
        --output logs/component_loss_quick/confirmation_aggregate

Generation seeds are repeated measurements nested within training seeds. All
standard deviations and ranges are descriptive, not confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
    "width_jsd", "speed_jsd", "collision_rate",
)


def read_json(path):
    return json.loads(path.read_text())


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std_sample": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "std_sample": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def metric_values(value, path):
    if value is None:
        return None
    missing = set(METRICS) - set(value)
    if missing:
        raise ValueError(f"{path}: missing metrics {sorted(missing)}")
    result = {key: float(value[key]) for key in METRICS}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError(f"{path}: metrics must be finite and nonnegative")
    return result


def checkpoint_steps(row, protocol):
    if row["variant"] == "warm_start":
        return 0
    match = re.search(r"(?:step[_=-]?)(\d+)", Path(row.get("checkpoint", "")).name)
    return int(match.group(1)) if match else protocol.get("steps")


def collect(root):
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path) if protocol_path.exists() else {}
    summary = read_json(root / "summary.json")
    observations = []
    for row in summary:
        folder = root / row["variant"]
        primary_path = folder / "sd_agent_metrics.json"
        primary = read_json(primary_path) if primary_path.exists() else {}
        seeds = {int(protocol.get("eval_seed", primary.get("runtime", {}).get("torch_initial_seed", 0))):
                 (primary_path, primary, {k: row[k] for k in METRICS if k in row})}
        # Prefer full JSON reports for counts and secondary full-reference metrics.
        for extra_path in sorted(folder.glob("eval_seed_*/sd_agent_metrics.json")):
            seed = int(extra_path.parent.name.removeprefix("eval_seed_"))
            seeds[seed] = (extra_path, read_json(extra_path), None)
        if row.get("extra_eval_seed") is not None:
            seed = int(row["extra_eval_seed"])
            if seed not in seeds:
                seeds[seed] = (root / "summary.json", {}, row.get("extra_agent_metrics"))
        for generation_seed, (source, report, fallback) in seeds.items():
            primary_metrics = metric_values(report.get("agent_metrics", fallback), source)
            if primary_metrics is None:
                continue
            metadata = {
                "input_directory": str(root), "variant": row["variant"],
                "training_seed": protocol.get("seed"), "generation_seed": generation_seed,
                "training_steps": checkpoint_steps(row, protocol),
                "checkpoint_path": row.get("checkpoint"),
                "initial_generator_sha256": protocol.get("initial_generator_sha256"),
                "inference_weights": protocol.get("evaluation_weights", "unknown"),
                "eval_manifest_sha256": protocol.get("eval_manifest_sha256", report.get("eval_set_sha256")),
                "num_samples": report.get("num_samples", row.get("num_samples")),
                "num_generated_vehicles": report.get("num_generated_vehicles",
                                                       row.get("num_generated_vehicles") if source == primary_path else None),
                "num_gt_vehicles": report.get("num_gt_vehicles"),
                "report_path": str(source),
            }
            observations.append({**metadata, "reference_mode": report.get("reference_mode", protocol.get("reference_mode", "matched")),
                                 "metrics": primary_metrics})
            secondary = metric_values(report.get("full_reference_agent_metrics"), source)
            if secondary is not None:
                # The source report's GT count belongs to the matched subset;
                # the secondary full-reference report supplies metrics only.
                observations.append({**metadata, "num_gt_vehicles": None, "reference_mode": "full_reference",
                                     "metrics": secondary})
    return observations


def pairing_key(row, include_directory=True):
    fields = ("training_seed", "generation_seed", "inference_weights", "initial_generator_sha256",
              "reference_mode", "eval_manifest_sha256", "num_samples")
    return tuple(row.get(k) for k in (("input_directory",) + fields if include_directory else fields))


def aggregate(observations, baseline_name="warm_start"):
    baseline_local, baseline_shared = {}, defaultdict(list)
    for row in observations:
        if row["variant"] == baseline_name:
            baseline_local[pairing_key(row)] = row
            baseline_shared[pairing_key(row, False)].append(row)
    paired, warnings, seen = [], [], set()
    for row in observations:
        baseline = baseline_local.get(pairing_key(row))
        if baseline is None:
            choices = baseline_shared.get(pairing_key(row, False), [])
            # Evaluation-only directories inherit a checkpoint from another
            # training run. Use that run's warm start even when other runs have
            # fresh warm evaluations with slightly different numerical values.
            # Never average warm metrics across training runs.
            checkpoint = row.get("checkpoint_path")
            checkpoint_run = Path(checkpoint).parent.parent.resolve() if checkpoint else None
            inherited = [choice for choice in choices
                         if checkpoint_run == Path(choice["input_directory"]).resolve()]
            if len(inherited) == 1:
                baseline = inherited[0]
            elif len(choices) == 1:
                baseline = choices[0]
            elif len(choices) > 1:
                warnings.append(f"Ambiguous warm start for {row['report_path']}; observation omitted")
                continue
        if baseline is None:
            warnings.append(f"Missing paired warm start for {row['report_path']} ({row['reference_mode']}); observation omitted")
            continue
        if any(baseline["metrics"][key] == 0 for key in METRICS):
            warnings.append(f"Zero warm-start metric in {baseline['report_path']}; observation omitted")
            continue
        repeated_key = (row["variant"], row["training_steps"], pairing_key(row, False))
        if repeated_key in seen:
            warnings.append(f"Repeated variant/seed/scope at {row['report_path']}; only first observation retained")
            continue
        seen.add(repeated_key)
        ratios = {key: row["metrics"][key] / baseline["metrics"][key] for key in METRICS}
        count, base_count = row["num_generated_vehicles"], baseline["num_generated_vehicles"]
        paired.append({**row, "warm_start_report_path": baseline["report_path"],
                       "warm_start_metrics": baseline["metrics"], "ratios": ratios,
                       "balanced_score": statistics.mean(ratios.values()),
                       "vehicle_count_ratio": count / base_count if count is not None and base_count else None})
    buckets = defaultdict(list)
    group_fields = ("variant", "reference_mode", "inference_weights", "initial_generator_sha256", "eval_manifest_sha256", "num_samples", "training_steps")
    for row in paired:
        buckets[tuple(row[k] for k in group_fields)].append(row)
    groups = []
    for key, rows in buckets.items():
        # Never give a training seed greater weight just because it has more
        # generation repeats. Retain both pooled and training-seed summaries.
        training_buckets = defaultdict(list)
        for row in rows:
            training_buckets[row["training_seed"]].append(row)
        by_training_seed = []
        for seed, training_rows in sorted(training_buckets.items(), key=lambda item: str(item[0])):
            by_training_seed.append({
                "training_seed": seed,
                "generation_seeds": sorted({r["generation_seed"] for r in training_rows}),
                "balanced_score": stats([r["balanced_score"] for r in training_rows]),
                "metrics_mean": {m: statistics.mean(r["metrics"][m] for r in training_rows) for m in METRICS},
                "ratios_mean": {m: statistics.mean(r["ratios"][m] for r in training_rows) for m in METRICS},
            })
        groups.append({
            **dict(zip(group_fields, key)), "n_observations": len(rows),
            "training_seeds": [r["training_seed"] for r in by_training_seed],
            "generation_seeds": sorted({r["generation_seed"] for r in rows}),
            "balanced_score": stats([r["balanced_score"] for r in rows]),
            "training_seed_mean_scores": stats([r["balanced_score"]["mean"] for r in by_training_seed]),
            "metrics": {m: stats([r["metrics"][m] for r in rows]) for m in METRICS},
            "ratios": {m: stats([r["ratios"][m] for r in rows]) for m in METRICS},
            "num_generated_vehicles": stats([r["num_generated_vehicles"] for r in rows]),
            "num_gt_vehicles": stats([r["num_gt_vehicles"] for r in rows]),
            "vehicle_count_ratio": stats([r["vehicle_count_ratio"] for r in rows]),
            "by_training_seed": by_training_seed,
        })
        repeat_counts = {len(r["generation_seeds"]) for r in by_training_seed}
        if len(repeat_counts) > 1:
            warnings.append(f"Unequal generation repeats for {key}; ranking uses the mean of training-seed means")
    groups.sort(key=lambda g: (g["reference_mode"], g["inference_weights"], str(g["eval_manifest_sha256"]),
                               g["training_seed_mean_scores"]["mean"], g["variant"]))
    return paired, groups, warnings


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def flatten_observation(row):
    result = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for mapping, prefix in (("metrics", ""), ("ratios", "ratio_"), ("warm_start_metrics", "warm_start_")):
        result.update({prefix + k: v for k, v in row[mapping].items()})
    return result


def flatten_group(group):
    result = {k: v for k, v in group.items() if not isinstance(v, (dict, list))}
    result["training_seeds"] = ",".join(map(str, group["training_seeds"]))
    result["generation_seeds"] = ",".join(map(str, group["generation_seeds"]))
    for label in ("balanced_score", "training_seed_mean_scores", "num_generated_vehicles", "num_gt_vehicles", "vehicle_count_ratio"):
        result.update({f"{label}_{k}": v for k, v in group[label].items()})
    for label, prefix in (("metrics", ""), ("ratios", "ratio_")):
        for metric, values in group[label].items():
            result.update({f"{prefix}{metric}_{k}": v for k, v in values.items()})
    return result


def markdown(groups, warnings):
    lines = ["# Component loss confirmation", "",
             "Scores are means of the seven metric ratios to the paired warm start; lower is better. "
             "Warm starts are paired by training seed, generation seed, weights, initial generator hash, scene manifest and reference scope. "
             "Evaluation-only runs use the warm start from their checkpoint's training directory; warm metrics are never averaged across runs.", "",
             "Generation seeds are nested repeats within training seeds. Standard deviations and ranges are descriptive; "
             "they are not confidence intervals. Ranking uses the mean of training-seed means.", ""]
    scopes = defaultdict(list)
    for group in groups:
        scopes[(group["reference_mode"], group["inference_weights"], group["eval_manifest_sha256"], group["num_samples"])].append(group)
    for (scope, weights, manifest, scenes), rows in scopes.items():
        lines += [f"## {scope}; {weights}; {scenes} scenes", "", f"Scene manifest: `{manifest}`", "",
                  "| Variant | Steps | Train seeds / evals | Score mean ± SD [range] | Nearest | Lat | Ang | Length | Width | Speed | Collision % | Vehicles |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in rows:
            score = group["balanced_score"]
            sd = f"{score['std_sample']:.4f}" if score["std_sample"] is not None else "—"
            cells = [group["variant"], str(group["training_steps"]),
                     f"{len(group['training_seeds'])} / {group['n_observations']}",
                     f"{group['training_seed_mean_scores']['mean']:.4f} ± {sd} [{score['min']:.4f}, {score['max']:.4f}]"]
            cells += [f"{group['metrics'][m]['mean']:.4f}" for m in METRICS]
            count = group["num_generated_vehicles"]["mean"]
            cells.append(f"{count:.1f}" if count is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Paired relative metric changes (negative is better):", "",
                  "| Variant | Nearest | Lat | Ang | Length | Width | Speed | Collision |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for group in rows:
            cells = [group["variant"]] + [f"{(group['ratios'][m]['mean'] - 1) * 100:+.1f}%" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if warnings:
        lines += ["## Incomplete or unmatched observations", ""]
        lines += ["- " + warning for warning in warnings]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Sweep directories or summary.json files")
    parser.add_argument("--output", type=Path, required=True, help="Directory for aggregate JSON, CSVs and Markdown")
    parser.add_argument("--baseline", default="warm_start")
    args = parser.parse_args()
    roots = [p.resolve().parent if p.name == "summary.json" else p.resolve() for p in args.inputs]
    if len(roots) != len(set(roots)):
        parser.error("Inputs contain duplicate directories")
    observations = [row for root in roots for row in collect(root)]
    paired, groups, warnings = aggregate(observations, args.baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "baseline": args.baseline, "inputs": list(map(str, roots)),
               "statistics_scope": "Descriptive; generation seeds nested within training seeds; no confidence intervals",
               "warnings": warnings, "paired_observations": paired, "groups": groups}
    (args.output / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output / "paired_observations.csv", [flatten_observation(row) for row in paired])
    write_csv(args.output / "aggregate.csv", [flatten_group(group) for group in groups])
    (args.output / "aggregate.md").write_text(markdown(groups, warnings))
    print(f"Aggregated {len(paired)} paired observations into {len(groups)} groups: {args.output.resolve()}")
    for warning in warnings:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()
