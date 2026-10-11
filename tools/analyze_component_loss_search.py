"""Analyze local InitDiffusion loss sweeps without loading models or using a GPU.

Example (use the sim Python):
  python tools/analyze_component_loss_search.py \
    --runs logs/component_loss_quick/search_round1_stable \
           logs/component_loss_quick/search_round2 \
    --output logs/component_loss_quick/analysis

Scores are the arithmetic mean of seven metric / warm-start metric ratios;
lower is better and the warm start scores 1. Heatmaps instead compare each
variant with its own run's mixed control, falling back to that warm start.
These are descriptive rankings of short runs, not statistical significance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    "nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
    "width_jsd", "speed_jsd", "collision_rate",
)
LABELS = ("Nearest", "Lateral", "Angle", "Length", "Width", "Speed", "Collision")


def read_json(path: Path):
    return json.loads(path.read_text())


def metric_values(row: dict) -> dict[str, float]:
    metrics = row.get("agent_metrics", row)
    result = {name: float(metrics[name]) for name in METRICS}
    if not all(math.isfinite(value) and value >= 0 for value in result.values()):
        raise ValueError("Metrics must be finite and nonnegative")
    return result


def load_baseline(path: Path) -> tuple[dict, str]:
    """Allow a sweep directory, summary.json, or individual metric report."""
    if path.is_dir():
        path = path / "summary.json"
    data = read_json(path)
    if isinstance(data, list):
        matches = [row for row in data if row.get("variant") == "warm_start"]
        if len(matches) != 1:
            raise ValueError(f"{path} must contain exactly one warm_start baseline")
        data = matches[0]
    return metric_values(data), str(path.resolve())


def ratio(value: float, reference: float, description: str) -> float:
    if reference <= 0:
        raise ValueError(f"{description} is zero; ratio scores are undefined")
    return value / reference


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_heatmap(rows: list[dict], output: Path, run_name: str, anchor: str):
    # Keep the reference row visible: it verifies the zero-delta convention.
    rows = sorted(rows, key=lambda row: row["score"])
    values = np.asarray([[row[f"{metric}_delta_anchor_pct"] for metric in METRICS]
                         for row in rows])
    fig, ax = plt.subplots(figsize=(12.5, max(3.2, .49 * len(rows) + 1.8)))
    picture = ax.imshow(values, cmap="RdYlGn_r", vmin=-100, vmax=100,
                        aspect="auto", interpolation="nearest")
    labels = []
    for row in rows:
        count = row.get("num_generated_vehicles")
        clip = row.get("clip_fraction")
        vehicle_text = f"{count:,}" if isinstance(count, int) else "—"
        clip_text = f"{clip:.0%}" if isinstance(clip, (int, float)) else "—"
        labels.append(f"{row['variant']}  |  score {row['score']:.3f}  |  "
                      f"vehicles {vehicle_text}  |  clipped {clip_text}")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks(range(len(METRICS)), LABELS)
    ax.tick_params(length=0)
    for row_index in range(len(rows)):
        for column_index in range(len(METRICS)):
            value = values[row_index, column_index]
            ax.text(column_index, row_index, f"{value:+.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if abs(value) >= 85 else "black")
    ax.set_title(f"{run_name}: metric change vs {anchor}\n"
                 "Negative = improvement; colors capped at ±100%; numbers uncapped")
    fig.colorbar(picture, ax=ax, fraction=.022, pad=.025, label="Metric change (%)")
    fig.tight_layout()
    figure_path = output / f"heatmap_{slug(run_name)}.png"
    fig.savefig(figure_path, dpi=170)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)
    return figure_path.name


def analyze_run(run: Path, fallback: Path | None, anchor_name: str):
    path = run / "summary.json"
    summary = read_json(path)
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"{path} must be a nonempty list of completed results")
    names = [row["variant"] for row in summary]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate variant names in {path}")
    warm = next((row for row in summary if row["variant"] == "warm_start"), None)
    if warm is not None:
        baseline = metric_values(warm)
        baseline_source = str(path.resolve()) + "#warm_start"
    elif fallback is not None:
        baseline, baseline_source = load_baseline(fallback)
    else:
        raise ValueError(f"{path} has no warm_start; provide --baseline PATH explicitly")
    anchor_row = next((row for row in summary if row["variant"] == anchor_name), warm)
    if anchor_row is None:
        anchor = baseline
        anchor_variant = "external_warm_start"
    else:
        anchor = metric_values(anchor_row)
        anchor_variant = anchor_row["variant"]
    protocol_path = run / "protocol.json"
    protocol = read_json(protocol_path) if protocol_path.exists() else {}
    rows = []
    for original in summary:
        variant = original["variant"]
        metrics = metric_values(original)
        row = {
            "run": run.name, "run_path": str(run.resolve()), "variant": variant,
            "baseline_source": baseline_source, "anchor_variant": anchor_variant,
            "num_samples": original.get("num_samples"),
            "num_generated_vehicles": original.get("num_generated_vehicles"),
            "clip_fraction": original.get("clip_fraction"),
            "train_seconds": original.get("train_seconds"),
            "eval_seconds": original.get("eval_seconds"),
            "steps": protocol.get("steps"), "learning_rate": protocol.get("lr"),
            "seed": protocol.get("seed"), "eval_seed": protocol.get("eval_seed"),
            "eval_offset": protocol.get("eval_offset"),
            "train_manifest_sha256": protocol.get("train_manifest_sha256"),
            "eval_manifest_sha256": protocol.get("eval_manifest_sha256"),
        }
        scores = []
        for name in METRICS:
            metric_ratio = ratio(metrics[name], baseline[name], f"warm_start {name}")
            row[name] = metrics[name]
            row[f"{name}_ratio_baseline"] = metric_ratio
            row[f"{name}_delta_baseline"] = metrics[name] - baseline[name]
            row[f"{name}_delta_baseline_pct"] = (metric_ratio - 1) * 100
            row[f"{name}_delta_anchor_pct"] = (
                ratio(metrics[name], anchor[name], f"anchor {name}") - 1) * 100
            scores.append(metric_ratio)
        row["score"] = sum(scores) / len(scores)
        row["score_delta_baseline_pct"] = (row["score"] - 1) * 100
        train_summary = run / variant / "train_summary.json"
        if train_summary.exists():
            train = read_json(train_summary)
            for window in ("first_window", "last_window"):
                for name in ("optimized_loss", "grad_norm_before_clip", "train/type_accuracy"):
                    row[f"{window}_{name.replace('/', '_')}"] = train.get(window, {}).get(name)
        metrics_path = run / variant / "sd_agent_metrics.json"
        if metrics_path.exists():
            report = read_json(metrics_path)
            row["num_gt_vehicles"] = report.get("num_gt_vehicles")
            row["component_loss"] = report.get("decoder", {}).get("component_loss")
        rows.append(row)
    for rank, row in enumerate(sorted(rows, key=lambda row: row["score"]), 1):
        row["rank_within_run"] = rank
    return rows, protocol, anchor_variant


def write_report(output: Path, grouped: list, warnings: list[str]):
    lines = ["# Component loss search analysis", "",
             "Score = mean of the seven metric / own warm-start metric ratios. Lower is better; "
             "warm start = 1. Percent changes below compare each run with its own warm start. "
             "These are short training results, not confidence intervals or proof of convergence.", ""]
    for warning in warnings:
        lines.extend([f"Caution: {warning}", ""])
    for run_name, rows, anchor, figure_name in grouped:
        lines.extend([f"## {run_name}", "",
                      "| Variant | Score | Nearest Δ% | Lateral Δ% | Angle Δ% | Length Δ% | Width Δ% "
                      "| Speed Δ% | Collision Δ% | Vehicles | Clipped |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in sorted(rows, key=lambda row: row["score"]):
            changes = " | ".join(f"{row[f'{name}_delta_baseline_pct']:+.1f}" for name in METRICS)
            count = row.get("num_generated_vehicles")
            clip = row.get("clip_fraction")
            clipped = f"{clip:.0%}" if isinstance(clip, (int, float)) else "—"
            lines.append(f"| {row['variant']} | {row['score']:.3f} | {changes} | "
                         f"{count if count is not None else '—'} | {clipped} |")
        lines.extend(["", f"Heatmap compares with `{anchor}`; its colors are capped but values are not.",
                      "", f"![Metric changes for {run_name}]({figure_name})", ""])
    (output / "analysis.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, type=Path,
                        help="Sweep directories containing summary.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="Explicit fallback baseline for runs without their own warm_start")
    parser.add_argument("--anchor", default="mixed",
                        help="Heatmap comparison variant within each run; default mixed, fallback warm_start")
    args = parser.parse_args()
    if len({path.resolve() for path in args.runs}) != len(args.runs):
        parser.error("--runs contains duplicate directories")
    if len({path.name for path in args.runs}) != len(args.runs):
        parser.error("Run directory names must be unique for unambiguous figure names")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    grouped = []
    protocols = {}
    warnings = []
    try:
        for run in args.runs:
            rows, protocol, anchor = analyze_run(run, args.baseline, args.anchor)
            figure_name = make_heatmap(rows, args.output, run.name, anchor)
            grouped.append((run.name, rows, anchor, figure_name))
            protocols[run.name] = protocol
            results.extend(sorted(rows, key=lambda row: row["score"]))
        comparison_keys = ("steps", "lr", "seed", "eval_seed", "eval_manifest_sha256", "train_manifest_sha256")
        varying = [key for key in comparison_keys
                   if len({json.dumps(protocol.get(key)) for protocol in protocols.values()}) > 1]
        if varying:
            warnings.append("Runs differ in " + ", ".join(varying) +
                            "; compare variants within each run. Scores normalized to warm starts "
                            "do not remove those experimental differences.")
        save_json(args.output / "all_results.json", results)
        fields = list(dict.fromkeys(key for row in results for key in row))
        with (args.output / "all_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                             for key, value in row.items()} for row in results)
        save_json(args.output / "analysis_protocol.json", {
            "score": "arithmetic mean of seven metric / own warm-start metric ratios",
            "heatmap_anchor_preference": args.anchor, "metric_order": METRICS,
            "source_runs": [str(path.resolve()) for path in args.runs],
            "fallback_baseline": str(args.baseline.resolve()) if args.baseline else None,
            "protocols": protocols, "warnings": warnings,
        })
        write_report(args.output, grouped, warnings)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Analysis error: {error}\n")
    print(f"Wrote {len(results)} result rows and {len(grouped)} heatmaps to {args.output.resolve()}")
    for run_name, rows, _, _ in grouped:
        trained = [row for row in rows if row["variant"] != "warm_start"]
        if trained:
            best = min(trained, key=lambda row: row["score"])
            print(f"{run_name}: best trained variant {best['variant']}, score={best['score']:.4f}; "
                  f"warm-start-relative change={best['score_delta_baseline_pct']:+.1f}%")
    for warning in warnings:
        print(f"Caution: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
