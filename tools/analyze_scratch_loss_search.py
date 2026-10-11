"""CPU-only analysis of InitDiffusion training runs initialized from scratch.

Discover completed summary rows recursively; enrich them from their actual
per-checkpoint metric reports and training diagnostics. Never extrapolate a
short-run score into a claimed final-convergence result.

Example:
    python tools/analyze_scratch_loss_search.py \
      --runs logs/component_loss_scratch \
      --output docs/experiments/init_diffusion_scratch_loss_search_20261011 \
      --plot-variants legacy,warm_winner,geometry_strong

Plot selection never filters the saved observations, aggregate tables, report
data, rankings or late-budget trends.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim_scratch_loss_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCALES = {
    # The helper reports distance JSD x10 and angle/size/speed JSD x100.
    # Convert the chosen raw-JSD reference scales to those same report units.
    "nearest_dist_jsd": 1.0,
    "lat_dev_jsd": 1.0,
    "ang_dev_jsd": 10.0,
    "length_jsd": 20.0,
    "width_jsd": 15.0,
    "speed_jsd": 20.0,
    "collision_rate": 5.0,
}
METRICS = tuple(SCALES)
LABELS = ("Nearest JSD ×10", "Lateral JSD ×10", "Angle JSD ×100",
          "Length JSD ×100", "Width JSD ×100", "Speed JSD ×100", "Collision %")
SCORE_LABEL = "Score = mean(metric / [1, 1, 10, 20, 15, 20, 5])"


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))[:160]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def first(*values):
    return next((value for value in values if value is not None), None)


def seed_sort(value):
    numeric = number(value)
    return (numeric is None, numeric if numeric is not None else 0, str(value))


def metric_values(row):
    values = row.get("agent_metrics", row)
    if not isinstance(values, dict):
        return None
    result = {name: number(values.get(name)) for name in METRICS}
    if any(value is None or value < 0 for value in result.values()):
        return None
    return result


def summary_rows(value):
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("results", "rows", "summary"):
            if isinstance(value.get(key), list):
                return summary_rows(value[key])
        if metric_values(value) is not None:
            return [value]
    return []


def nearest_file(folder, name, boundary):
    folder, boundary = Path(folder).resolve(), Path(boundary).resolve()
    while folder == boundary or boundary in folder.parents:
        candidate = folder / name
        if candidate.is_file():
            return candidate
        if folder == boundary:
            break
        folder = folder.parent
    return None


def nested(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def infer_budget(path):
    for part in reversed(Path(path).parts):
        match = re.fullmatch(r"(?:eval_step_|step_|budget_)(\d+)(?:\.ckpt)?", part)
        if match:
            return int(match[1])
    return None


def find_report(run, original, protocol, cache):
    """Match report identity and metrics; do not attach a different timestep."""
    for key in ("metrics_path", "metric_report", "report_path"):
        if original.get(key):
            path = Path(original[key])
            if not path.is_absolute():
                path = run / path
            if path.is_file():
                return path, read_json(path)
    variant = str(original.get("variant", ""))
    folder = run / variant
    budget = first(original.get("budget"), original.get("steps"), protocol.get("steps"))
    if budget is not None and (folder / f"eval_step_{budget}").is_dir():
        folder = folder / f"eval_step_{budget}"
    paths = sorted(folder.rglob("sd_agent_metrics.json")) if folder.is_dir() else []
    wanted = metric_values(original)
    wanted_seed = first(original.get("eval_seed"), protocol.get("eval_seed"))
    wanted_mode = original.get("evaluation_weights", protocol.get("evaluation_weights", "online"))
    matches = []
    for path in paths:
        if path not in cache:
            cache[path] = read_json(path)
        report = cache[path]
        values = metric_values(report)
        if values is None or wanted is None:
            continue
        if any(not math.isclose(values[key], wanted[key], rel_tol=1e-10, abs_tol=1e-12)
               for key in METRICS):
            continue
        report_seed = nested(report, "runtime", "torch_initial_seed")
        report_mode = "ema" if nested(report, "decoder", "use_ema") else "online"
        if wanted_seed is not None and report_seed is not None and int(wanted_seed) != int(report_seed):
            continue
        if report_mode != wanted_mode:
            continue
        report_budget = infer_budget(path)
        if budget is not None and report_budget is not None and int(budget) != report_budget:
            continue
        matches.append((path, report))
    if len(matches) > 1:
        # Multiple identical reports carry identical metrics and coverage; keep
        # one source and leave observation deduplication to the identity checks.
        matches.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    return matches[0] if matches else (None, {})


def enrich_row(run, original, protocol, report_path, report, source, boundary):
    metrics = metric_values(original) or metric_values(report)
    if metrics is None:
        return None
    folder = report_path.parent if report_path else run / str(original.get("variant", ""))
    config_path = nearest_file(folder, "config.yaml", boundary)
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    config = config if isinstance(config, dict) else {}
    loss = first(nested(report, "decoder", "component_loss"),
                 nested(config, "model", "model_config", "decoder", "init_diffusion", "component_loss"))
    train_summary_path = nearest_file(folder, "train_summary.json", boundary)
    train = read_json(train_summary_path) if train_summary_path else {}
    trace_path = nearest_file(folder, "train_trace.json", boundary)
    budget = first(original.get("budget"), original.get("global_step"), infer_budget(folder),
                   original.get("steps"), protocol.get("steps"), train.get("steps"))
    row = {
        "run": run.name, "run_path": str(run.resolve()), "summary_source": str(source.resolve()),
        "report_source": str(report_path.resolve()) if report_path else None,
        "variant": first(original.get("variant"), folder.name),
        "budget": int(budget) if budget is not None else None,
        "seed": first(original.get("seed"), original.get("train_seed"), protocol.get("seed"), config.get("seed")),
        "eval_seed": first(original.get("eval_seed"), nested(report, "runtime", "torch_initial_seed"), protocol.get("eval_seed")),
        "learning_rate": number(first(original.get("learning_rate"), original.get("lr"), protocol.get("lr"),
                                      nested(config, "model", "model_config", "lr"))),
        "evaluation_weights": first(original.get("evaluation_weights"),
            "ema" if nested(report, "decoder", "use_ema") else None, protocol.get("evaluation_weights"), "online"),
        "reference_mode": first(report.get("reference_mode"), protocol.get("reference_mode"), "matched"),
        "num_samples": first(original.get("num_samples"), report.get("num_samples"), protocol.get("scenes")),
        "num_gt_samples": first(original.get("num_gt_samples"), report.get("num_gt_samples")),
        "eval_manifest_sha256": first(original.get("eval_manifest_sha256"), protocol.get("eval_manifest_sha256"),
                                       report.get("eval_set_sha256")),
        "eval_set_sha256": report.get("eval_set_sha256"),
        "train_manifest_sha256": first(original.get("train_manifest_sha256"), protocol.get("train_manifest_sha256")),
        "calibration_sha256": first(original.get("calibration_sha256"), protocol.get("calibration_sha256")),
        "initial_generator_sha256": first(original.get("initial_generator_sha256"), protocol.get("initial_generator_sha256")),
        "initial_map_sha256": first(original.get("initial_map_sha256"), protocol.get("initial_map_sha256")),
        "initialization_scope": first(original.get("initialization_scope"), protocol.get("initialization_scope")),
        "train_batch_size": first(original.get("batch_size"), protocol.get("batch_size"), nested(config, "data", "train_batch_size")),
        "base_train_offset": first(original.get("base_train_offset"), protocol.get("base_train_offset"), 0),
        "checkpoint": original.get("checkpoint"), "evaluation_only": original.get("evaluation_only", False),
        "train_seconds": number(original.get("train_seconds")), "eval_seconds": number(original.get("eval_seconds")),
        "clip_fraction": number(first(original.get("clip_fraction"), train.get("clip_fraction"))),
        "train_summary_source": str(train_summary_path.resolve()) if train_summary_path else None,
        "train_trace_source": str(trace_path.resolve()) if trace_path else None,
        "component_loss": loss, "recipe_sha256": stable_hash(loss), **metrics,
    }
    row.update({key: first(original.get(key), protocol.get(key), nested(report, "decoder", key))
                for key in ("sampling_steps", "map_crop", "map_half_extent", "precision", "float32_matmul_precision")})
    row["full_reference_agent_metrics"] = first(original.get("full_reference_agent_metrics"),
                                                report.get("full_reference_agent_metrics"))
    full_metrics = metric_values(row["full_reference_agent_metrics"]) if isinstance(row["full_reference_agent_metrics"], dict) else None
    row["score_full_reference"] = (sum(full_metrics[key] / SCALES[key] for key in METRICS) / len(METRICS)
                                    if full_metrics is not None else None)
    for key in ("num_generated_vehicles", "num_gt_vehicles", "num_generated_onroad_vehicles",
                "num_gt_onroad_vehicles", "generated_onroad_fraction", "gt_onroad_fraction", "num_colliding_vehicles"):
        row[key] = number(first(original.get(key), report.get(key)))
    generated_counts, gt_counts = report.get("feature_counts_generated", []), report.get("feature_counts_gt", [])
    histogram_path = folder / "feature_histograms.json"
    if histogram_path.is_file():
        histograms = read_json(histogram_path)
        row["histogram_counts_generated"] = {key: sum(value.get("generated", [])) for key, value in histograms.items()}
        row["histogram_counts_gt"] = {key: sum(value.get("matched_gt", [])) for key, value in histograms.items()}
        for label in ("generated", "gt"):
            counts = row[f"histogram_counts_{label}"]
            row[f"num_{label}_onroad_vehicles"] = first(row[f"num_{label}_onroad_vehicles"], counts.get("lat_dev_jsd"))
            row[f"num_{label}_vehicles"] = first(row[f"num_{label}_vehicles"], counts.get("length_jsd"))
    for label, counts in (("generated", generated_counts), ("gt", gt_counts)):
        if len(counts) >= 2:
            row[f"num_{label}_onroad_vehicles"] = first(row[f"num_{label}_onroad_vehicles"], counts[1])
        vehicles = row[f"num_{label}_vehicles"]
        onroad = row[f"num_{label}_onroad_vehicles"]
        if vehicles is not None and vehicles > 0 and onroad is not None:
            row[f"{label}_onroad_fraction"] = onroad / vehicles
    generated, gt = row["num_generated_vehicles"], row["num_gt_vehicles"]
    row["vehicle_count_ratio"] = generated / gt if generated is not None and gt else None
    row["generated_onroad_gap_pp"] = (100 * (row["generated_onroad_fraction"] - row["gt_onroad_fraction"])
        if row["generated_onroad_fraction"] is not None and row["gt_onroad_fraction"] is not None else None)
    row["coverage_flags"] = []
    if not generated:
        row["coverage_flags"].append("no_generated_vehicles")
    if row["vehicle_count_ratio"] is not None and not 0.8 <= row["vehicle_count_ratio"] <= 1.2:
        row["coverage_flags"].append("vehicle_count_differs_more_than_20pct")
    if row["generated_onroad_fraction"] is not None and row["generated_onroad_fraction"] < 0.8:
        row["coverage_flags"].append("generated_onroad_below_80pct")
    row["score"] = sum(row[key] / SCALES[key] for key in METRICS) / len(METRICS)
    # These flags are descriptive, not hidden changes to metric definitions or
    # automatic exclusion criteria; report all rows and scores transparently.
    for window in ("first_window", "last_window"):
        for key in ("optimized_loss", "grad_norm_before_clip", "train/type_accuracy"):
            row[f"{window}_{key.replace('/', '_')}"] = number(train.get(window, {}).get(key))
    cohort = {key: row[key] for key in ("learning_rate", "evaluation_weights", "reference_mode", "num_samples",
        "eval_manifest_sha256", "initialization_scope", "train_batch_size", "sampling_steps", "map_crop",
        "map_half_extent", "precision", "float32_matmul_precision", "base_train_offset")}
    row["cohort_sha256"] = stable_hash(cohort)
    return row


def discover(roots):
    rows, warnings, used_reports, cache = [], [], set(), {}
    summary_paths = sorted({path.resolve() for root in roots for path in root.rglob("summary.json")})
    all_reports = sorted({path.resolve() for root in roots for path in root.rglob("sd_agent_metrics.json")})
    for path in summary_paths:
        run = path.parent
        boundary = next(root for root in roots if root.resolve() == run or root.resolve() in run.parents)
        protocol_path = nearest_file(run, "protocol.json", boundary)
        protocol = read_json(protocol_path) if protocol_path else {}
        for original in summary_rows(read_json(path)):
            report_path, report = find_report(run, original, protocol, cache)
            row = enrich_row(run, original, protocol, report_path, report, path, boundary)
            if row is None:
                warnings.append(f"Incomplete/nonfinite metric row omitted from {path}: {original.get('variant')}")
                continue
            if report_path:
                used_reports.add(report_path.resolve())
            else:
                warnings.append(f"No matching per-checkpoint report for {path}: {row['variant']} / {row['budget']}")
            rows.append(row)
    for path in all_reports:
        if path in used_reports:
            continue
        boundary = next(root for root in roots if root.resolve() == path or root.resolve() in path.parents)
        protocol_path = nearest_file(path.parent, "protocol.json", boundary)
        protocol = read_json(protocol_path) if protocol_path else {}
        run = protocol_path.parent if protocol_path else boundary
        relative = path.relative_to(run)
        variant = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
        report = cache.get(path) or read_json(path)
        original = {"variant": variant, "budget": infer_budget(path), "agent_metrics": report.get("agent_metrics", {})}
        row = enrich_row(run, original, protocol, path, report, path, boundary)
        if row is not None:
            rows.append(row)
            warnings.append(f"Report-only observation recovered: {path}")
    unique, identities = [], set()
    for row in rows:
        identity = tuple(row[key] for key in ("variant", "budget", "seed", "eval_seed", "cohort_sha256", "recipe_sha256"))
        identity += tuple(row[key] for key in METRICS)
        if identity in identities:
            warnings.append(f"Identical repeated observation deduplicated: {row['report_source'] or row['summary_source']}")
            continue
        identities.add(identity)
        unique.append(row)
    for row in unique:
        if row["budget"] is None:
            warnings.append(f"Unknown training budget: {row['report_source'] or row['summary_source']}")
        if row["seed"] is None or row["eval_seed"] is None:
            warnings.append(f"Unknown training/evaluation seed: {row['report_source'] or row['summary_source']}")
    return unique, warnings


def mean_stats(values):
    values = [float(value) for value in values if number(value) is not None]
    # These are descriptive sample statistics across independent training
    # seeds. Sampler repetitions are averaged before reaching this function.
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
            "minimum": min(values), "maximum": max(values)} if values else None


def paired_anchor(rows, anchor):
    anchors = defaultdict(list)
    for row in rows:
        if row["variant"] == anchor:
            anchors[(row["cohort_sha256"], row["budget"], row["seed"], row["eval_seed"])].append(row)
    for row in rows:
        reference = anchors.get((row["cohort_sha256"], row["budget"], row["seed"], row["eval_seed"]), [])
        row["anchor_variant"] = anchor if reference else None
        row["anchor_score_ratio"] = None
        for key in METRICS:
            row[f"{key}_delta_anchor_pct"] = None
        if reference:
            base_score = float(np.mean([base["score"] for base in reference]))
            row["anchor_score_ratio"] = row["score"] / base_score if base_score > 0 else None
            for key in METRICS:
                base = float(np.mean([item[key] for item in reference]))
                row[f"{key}_delta_anchor_pct"] = 100 * (row[key] / base - 1) if base > 0 else None


def aggregate(rows, warnings):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort_sha256"], row["variant"], row["budget"], row["recipe_sha256"])].append(row)
    results = []
    value_keys = list(METRICS) + ["score", "score_full_reference", "anchor_score_ratio", "vehicle_count_ratio", "generated_onroad_fraction",
        "gt_onroad_fraction", "generated_onroad_gap_pp", "num_generated_vehicles", "num_gt_vehicles", "clip_fraction"]
    value_keys += [f"{key}_delta_anchor_pct" for key in METRICS]
    for (cohort, variant, budget, recipe), observations in groups.items():
        seed_cells = defaultdict(lambda: defaultdict(list))
        for row in observations:
            seed_cells[row["seed"]][row["eval_seed"]].append(row)
        seed_means = []
        for seed, cells in seed_cells.items():
            result = {"seed": seed, "eval_seeds": sorted(cells, key=seed_sort), "num_observations": sum(map(len, cells.values()))}
            for key in value_keys:
                # Equal average over generation seeds inside a training seed;
                # repeated stage evaluations do not become extra independent N.
                cell_means = []
                for eval_seed, cell in cells.items():
                    values = [item[key] for item in cell if number(item[key]) is not None]
                    if values:
                        cell_means.append(float(np.mean(values)))
                    if key == "score" and len(cell) > 1:
                        warnings.append(f"Repeated nonidentical eval cell averaged: {variant}, budget{budget}, trainseed{seed}, evalseed{eval_seed}")
                result[key] = float(np.mean(cell_means)) if cell_means else None
            seed_means.append(result)
        first_row = observations[0]
        result = {key: first_row.get(key) for key in ("variant", "budget", "learning_rate", "evaluation_weights", "reference_mode",
            "num_samples", "eval_manifest_sha256", "initialization_scope", "train_batch_size", "component_loss", "recipe_sha256",
            "cohort_sha256", "anchor_variant")}
        result.update(num_training_seeds=len(seed_cells), num_observations=len(observations),
            training_seed_means=seed_means, report_sources=[row["report_source"] for row in observations],
            anchor_variant=first(*(row["anchor_variant"] for row in observations)),
            coverage_flag_counts={flag: sum(flag in row["coverage_flags"] for row in observations)
                                 for flag in sorted({flag for row in observations for flag in row["coverage_flags"]})})
        for key in value_keys:
            result[key] = mean_stats([item[key] for item in seed_means])
        results.append(result)
    ranked = defaultdict(list)
    for result in results:
        ranked[(result["cohort_sha256"], result["budget"])].append(result)
    for group in ranked.values():
        for rank, result in enumerate(sorted(group, key=lambda item: item["score"]["mean"]), 1):
            result["rank_same_budget"] = rank
    return sorted(results, key=lambda row: (row["cohort_sha256"], row["budget"] or -1, row["score"]["mean"]))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def flatten_aggregates(rows):
    result = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, dict) and "mean" in value:
                item.update({f"{key}_{stat}": number for stat, number in value.items()})
            else:
                item[key] = value
        result.append(item)
    return result


def late_budget_trends(aggregates):
    series_groups = defaultdict(list)
    for row in aggregates:
        if row["budget"] is not None:
            series_groups[(row["cohort_sha256"], row["variant"], row["recipe_sha256"])].append(row)
    trends = []
    for (cohort, variant, recipe), series in sorted(series_groups.items()):
        series.sort(key=lambda row: row["budget"])
        if len(series) < 2:
            continue
        start, end = series[-2:]
        difference = end["budget"] - start["budget"]
        if difference <= 0:
            continue
        before = {item["seed"]: item for item in start["training_seed_means"]}
        after = {item["seed"]: item for item in end["training_seed_means"]}
        common = sorted(set(before).intersection(after), key=seed_sort)
        changes = [after[seed]["score"] - before[seed]["score"] for seed in common]
        change = float(np.mean(changes)) if changes else None
        trends.append({"cohort_sha256": cohort, "variant": variant, "recipe_sha256": recipe,
            "start_budget": start["budget"], "end_budget": end["budget"], "score_change": change,
            "score_slope_per_1000_steps": 1000 * change / difference if change is not None else None,
            "paired_training_seeds": common,
            "aggregate_score_change_unpaired": end["score"]["mean"] - start["score"]["mean"],
            "training_seeds_start": start["num_training_seeds"], "training_seeds_end": end["num_training_seeds"]})
    return trends


def plot_learning_curves(aggregates, output):
    cohorts = defaultdict(list)
    for row in aggregates:
        if row["budget"] is not None:
            cohorts[row["cohort_sha256"]].append(row)
    figures = []
    for cohort, rows in cohorts.items():
        variants = defaultdict(list)
        for row in rows:
            variants[(row["variant"], row["recipe_sha256"])].append(row)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        colors = plt.get_cmap("tab20")
        for index, ((variant, recipe), series) in enumerate(sorted(variants.items())):
            series.sort(key=lambda row: row["budget"])
            x = [row["budget"] for row in series]
            for ax, key in zip(axes, ("score", *METRICS)):
                y = [row[key]["mean"] for row in series]
                ax.plot(x, y, "o-", color=colors(index % 20), label=variant, linewidth=1.3, markersize=3)
                if any(row["num_training_seeds"] > 1 for row in series):
                    sd = np.asarray([row[key]["sd"] for row in series])
                    ax.fill_between(x, np.maximum(0, np.asarray(y) - sd), np.asarray(y) + sd,
                                    color=colors(index % 20), alpha=.1)
        for ax, label in zip(axes, ("Fixed-scale score", *LABELS)):
            ax.set_title(label)
            ax.set_xlabel("Optimizer steps from random initialization")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.005), ncol=min(5, len(labels)), fontsize=7)
        example = rows[0]
        fig.suptitle(f"lr={example['learning_rate']}, {example['evaluation_weights']}, n={example['num_samples']}; "
                     "bands = training-seed SD, not confidence intervals\n" + SCORE_LABEL, fontsize=10)
        fig.tight_layout(rect=(0, .11, 1, .92))
        path = output / f"learning_curves_{cohort[:12]}.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        figures.append(path.name)
    return figures, late_budget_trends(aggregates)


def plot_coverage_curves(aggregates, output):
    cohorts = defaultdict(list)
    for row in aggregates:
        if row["budget"] is not None and (row["generated_onroad_fraction"] or row["vehicle_count_ratio"]):
            cohorts[row["cohort_sha256"]].append(row)
    figures = []
    for cohort, rows in cohorts.items():
        variants = defaultdict(list)
        for row in rows:
            variants[(row["variant"], row["recipe_sha256"])].append(row)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = plt.get_cmap("tab20")
        for index, ((variant, _), series) in enumerate(sorted(variants.items())):
            series.sort(key=lambda row: row["budget"])
            for ax, key in zip(axes, ("generated_onroad_fraction", "vehicle_count_ratio")):
                active = [row for row in series if row[key]]
                if not active:
                    continue
                x = [row["budget"] for row in active]
                y = np.asarray([row[key]["mean"] for row in active]) * 100
                ax.plot(x, y, "o-", color=colors(index % 20), label=variant, linewidth=1.3, markersize=3)
                if any(row["num_training_seeds"] > 1 for row in active):
                    sd = np.asarray([row[key]["sd"] for row in active]) * 100
                    ax.fill_between(x, np.maximum(0, y - sd), y + sd, color=colors(index % 20), alpha=.1)
        gt_onroad = [row["gt_onroad_fraction"]["mean"] * 100 for row in rows if row["gt_onroad_fraction"]]
        if gt_onroad:
            axes[0].axhline(float(np.mean(gt_onroad)), color="black", linestyle="--", linewidth=.9, label="Matched GT")
        axes[1].axhline(100, color="black", linestyle="--", linewidth=.9, label="Matched GT count")
        axes[0].set_title("Generated vehicles classified on-road (%)")
        axes[0].set_ylim(0, 101)
        axes[1].set_title("Generated / matched-GT vehicle count (%)")
        for ax in axes:
            ax.set_xlabel("Optimizer steps from random initialization")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=7)
        example = rows[0]
        fig.suptitle(f"lr={example['learning_rate']}, {example['evaluation_weights']}, n={example['num_samples']}; "
                     "bands = training-seed SD; lateral/angle metrics use on-road vehicles", fontsize=9)
        fig.tight_layout(rect=(0, .15, 1, .93))
        path = output / f"coverage_curves_{cohort[:12]}.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        figures.append(path.name)
    return figures


def plot_heatmaps(aggregates, output):
    groups = defaultdict(list)
    for row in aggregates:
        if row["anchor_variant"]:
            groups[(row["cohort_sha256"], row["budget"])].append(row)
    figures = []
    for (cohort, budget), rows in sorted(groups.items(), key=str):
        rows.sort(key=lambda row: row["score"]["mean"])
        values = np.asarray([[row[f"{metric}_delta_anchor_pct"]["mean"]
            if row[f"{metric}_delta_anchor_pct"] else np.nan for metric in METRICS] for row in rows])
        if not np.isfinite(values).any():
            continue
        fig, ax = plt.subplots(figsize=(12, max(3, .42 * len(rows) + 1.6)))
        picture = ax.imshow(values, cmap="RdYlGn_r", vmin=-100, vmax=100, aspect="auto")
        ax.set_xticks(range(len(METRICS)), LABELS)
        ax.set_yticks(range(len(rows)), [f"{row['variant']} | S={row['score']['mean']:.3f} | Nseed={row['num_training_seeds']}" for row in rows])
        for i in range(len(rows)):
            for j in range(len(METRICS)):
                value = values[i, j]
                ax.text(j, i, f"{value:+.1f}%" if np.isfinite(value) else "N/A", ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > 80 else "black")
        ax.set_title(f"{budget} steps: paired metric change vs {rows[0]['anchor_variant']}\n"
                     "Same LR, split, train/evaluation seed and weights; negative = improvement")
        fig.colorbar(picture, ax=ax, pad=.025, fraction=.025, label="Percent change; colors clipped ±100%")
        fig.tight_layout()
        path = output / f"metric_heatmap_{cohort[:12]}_step_{budget}.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        figures.append(path.name)
    return figures


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_gradient_traces(rows, output):
    sources = {}
    for row in rows:
        if row["train_trace_source"]:
            sources[row["train_trace_source"]] = row
    if not sources:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plotted = 0
    for path, row in sorted(sources.items()):
        records = read_json(path)
        if isinstance(records, dict):
            records = records.get("records", records.get("trace", []))
        if not isinstance(records, list) or not records:
            continue
        label = f"{row['run']}/{row['variant']} (seed{row['seed']}, lr{row['learning_rate']})"
        for ax, key in zip(axes, ("optimized_loss", "grad_norm_before_clip", "train/type_accuracy")):
            active = [record for record in records if number(record.get(key)) is not None and number(record.get("step")) is not None]
            if not active:
                continue
            x = np.asarray([record["step"] for record in active])
            y = [record[key] for record in active]
            window = min(64, max(1, len(y) // 16))
            y = smooth(y, window)
            x = x[len(x) - len(y):]
            ax.plot(x, y, label=label, linewidth=.9, alpha=.85)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_ylabel("Optimized objective\n(not comparable across recipes)")
    axes[1].set_ylabel("Gradient norm before clipping")
    axes[2].set_ylabel("Training type accuracy")
    axes[2].set_xlabel("Optimizer step")
    for ax in axes:
        ax.grid(alpha=.2)
    if all(line.get_ydata().min() > 0 for line in axes[1].lines):
        axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6)
    fig.suptitle("Training diagnostics: smoothed within each run; full trace retained in raw artifacts")
    fig.tight_layout(rect=(0, min(.35, .027 * math.ceil(len(labels) / 2)), 1, .96))
    path = output / "gradient_training_traces.png"
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.name


def markdown_report(output, rows, aggregates, warnings, figures, anchor, plot_variants=None):
    lines = ["# 从头训练 loss 搜索：自动汇总", "",
        "以下只包含已经完成并保存了七项生成指标的实际评价。固定综合分数为七项 `metric / scale` 的等权平均，越低越好；"
        "这些 scale 只是数值尺度，不来自模型权重，也不是论文的官方综合指标。", "",
        "| 指标 | 固定 scale |", "|---|---:|"]
    lines.extend(f"| {name} | {scale:g} |" for name, scale in SCALES.items())
    lines.extend(["", "指标使用官方 helper 输出单位：nearest/lateral JSD ×10，angle/length/width/speed JSD ×100，"
        "collision 为百分数；上述 scale 已转换为相同单位。", "",
        "生成 seed 在训练 seed 内平均，再跨训练 seed 等权平均。误差条仅为描述性训练 seed 样本标准差"
        "（N≥2 时 ddof=1，N=1 时显示 0），不是置信区间；"
        "重复评价不会增加独立训练样本量。不同学习率、评价子集、权重模式、初始化范围分别汇总，不混合排名。", "",
        "lat/angle 只统计 on-road 车辆，因此同时检查车辆数量和 on-road 覆盖。coverage flags 仅提示问题，不修改官方指标或隐藏排除结果。"
        "短期训练的趋势不能证明最终完整训练已经收敛或获得全局最优。", "",
        f"实际评价记录：**{len(rows)}**；分组：**{len(aggregates)}**。配对 heatmap anchor 为 `{anchor}`。", ""])
    if plot_variants is not None:
        lines.extend(["图表仅显示指定方案：`" + ", ".join(sorted(plot_variants)) + "`；下表、全部原始记录、聚合统计、"
            "排序和后期趋势仍包含所有实际方案。", ""])
    cohorts = defaultdict(list)
    for row in aggregates:
        cohorts[row["cohort_sha256"]].append(row)
    for cohort, group in cohorts.items():
        example = group[0]
        lines.extend([f"## 比较组 {cohort[:12]}", "",
            f"LR={example['learning_rate']}，weights={example['evaluation_weights']}，samples={example['num_samples']}，"
            f"initialization={example['initialization_scope']}。", "",
            "| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in sorted(group, key=lambda item: (item["budget"] or -1, item["score"]["mean"])):
            values = " | ".join(f"{row[key]['mean']:.4f}" for key in METRICS)
            coverage = row["vehicle_count_ratio"]
            onroad = row["generated_onroad_fraction"]
            coverage_text = f"{coverage['mean']:.3f}" if coverage else "N/A"
            onroad_text = f"{onroad['mean']:.1%}" if onroad else "N/A"
            lines.append(f"| {row['budget']} | {row['variant']} | {row['num_training_seeds']} | "
                f"{row['score']['mean']:.3f} ± {row['score']['sd']:.3f} | {values} | {coverage_text} | {onroad_text} |")
    if figures:
        lines.extend(["", "## 图表", ""])
        for figure in figures:
            lines.extend([f"![{figure}]({figure})", ""])
    if warnings:
        lines.extend(["## 数据审计提示", ""])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
        lines.append("")
    lines.extend(["逐次记录见 [observations.csv](observations.csv)，分组统计见 [aggregate.csv](aggregate.csv)，"
        "完整协议和原始报告位置记录在 JSON 中。学习趋势仅使用同一比较组最后两个预算的实际点，未拟合所谓最终性能预测。", ""])
    (output / "analysis.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, default=[ROOT / "logs/component_loss_scratch"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011")
    parser.add_argument("--anchor", default="anchor")
    parser.add_argument("--plot-variants", default=None,
                        help="Optional comma-separated variant names for figures only; tables and rankings retain all actual arms")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.runs]
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        parser.error(f"Run directories do not exist: {missing}")
    rows, warnings = discover(roots)
    if not rows:
        parser.error("No completed seven-metric observations found; no result report was created")
    paired_anchor(rows, args.anchor)
    aggregates = aggregate(rows, warnings)
    plot_variants = ({name.strip() for name in args.plot_variants.split(",") if name.strip()}
                     if args.plot_variants is not None else None)
    if plot_variants is not None:
        unknown = plot_variants - {row["variant"] for row in rows}
        if unknown:
            warnings.append("Requested plot variants have no completed observations: " + ", ".join(sorted(unknown)))
    plotted_rows = [row for row in rows if plot_variants is None or row["variant"] in plot_variants]
    plotted_aggregates = [row for row in aggregates if plot_variants is None or row["variant"] in plot_variants]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "observations.json", rows)
    write_csv(output / "observations.csv", rows)
    save_json(output / "aggregate.json", aggregates)
    write_csv(output / "aggregate.csv", flatten_aggregates(aggregates))
    figures, _ = plot_learning_curves(plotted_aggregates, output)
    trends = late_budget_trends(aggregates)
    figures.extend(plot_coverage_curves(plotted_aggregates, output))
    figures.extend(plot_heatmaps(plotted_aggregates, output))
    gradients = plot_gradient_traces(plotted_rows, output)
    if gradients:
        figures.append(gradients)
    save_json(output / "late_budget_trends.json", trends)
    write_csv(output / "late_budget_trends.csv", trends)
    save_json(output / "analysis_protocol.json", {"scales": SCALES, "score": "mean(metric/scale)",
        "scale_units": "official helper output units; collision percentage points",
        "metric_output_multipliers": {name: 10 if name in METRICS[:2] else 100 for name in METRICS[:-1]},
        "roots": [str(path) for path in roots], "anchor": args.anchor, "num_observations": len(rows),
        "num_groups": len(aggregates), "warnings": sorted(set(warnings)),
        "plot_variants": sorted(plot_variants) if plot_variants is not None else None,
        "actual_plotted_variants": sorted({row["variant"] for row in plotted_rows}),
        "aggregation": "repeats within evaluation seed, then evaluation seeds within training seed, then training seeds",
        "training_seed_sd_ddof": 1, "single_training_seed_sd": 0,
        "uncertainty": "Descriptive sample SD across independent training seeds, not confidence intervals",
        "figures": figures, "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    markdown_report(output, rows, aggregates, warnings, figures, args.anchor, plot_variants)
    print(json.dumps({"output": str(output), "observations": len(rows), "groups": len(aggregates),
                      "warnings": len(set(warnings)), "figures": len(figures)}))


if __name__ == "__main__":
    main()
