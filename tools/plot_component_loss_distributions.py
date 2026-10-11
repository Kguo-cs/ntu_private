"""Plot saved official-bin vehicle histograms without loading models or tensors.

Example:
  python tools/plot_component_loss_distributions.py \
    --reference RUN/warm_start/feature_histograms.json \
    --candidates recipe_A=RUN/recipe_A/feature_histograms.json \
                 recipe_B=RUN/recipe_B/feature_histograms.json \
    --output REPORT/distributions --full-range

The y axis is probability per bin, not probability density. Saved counts use
official clipping bounds. Statistics use bin midpoints and are approximations
to the clipped distributions, not statistics of raw generated physical values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURES = {
    "speed_jsd": {"label": "Speed", "unit": "m/s", "zoom": None,
                  "thresholds": (("below_1_m_per_s", "below", 1.),
                                 ("at_least_10_m_per_s", "above", 10.),
                                 ("at_least_20_m_per_s", "above", 20.))},
    "length_jsd": {"label": "Length", "unit": "m", "zoom": (1., 9.),
                   "thresholds": (("below_3_m", "below", 3.),
                                  ("at_least_6_m", "above", 6.),
                                  ("at_least_10_m", "above", 10.))},
    "width_jsd": {"label": "Width", "unit": "m", "zoom": (.5, 3.5),
                  "thresholds": (("below_1_5_m", "below", 1.5),
                                 ("at_least_2_5_m", "above", 2.5),
                                 ("at_least_3_m", "above", 3.))},
}


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_counts(spec: dict, field: str, description: str):
    lo, hi, step = (float(spec[key]) for key in ("minimum", "maximum", "bin_width"))
    if not all(math.isfinite(value) for value in (lo, hi, step)) or hi <= lo or step <= 0:
        raise ValueError(f"Invalid histogram bounds for {description}")
    count = int(round((hi - lo) / step))
    edges = lo + np.arange(count + 1) * step
    if not math.isclose(edges[-1], hi, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"Bounds are not a whole number of bins for {description}")
    counts = np.asarray(spec[field], dtype=np.float64)
    if counts.ndim != 1 or len(counts) != count:
        raise ValueError(f"Histogram length differs from bin bounds for {description}")
    if not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError(f"Counts must be finite, nonnegative, and nonempty for {description}")
    if not np.allclose(counts, np.round(counts)):
        raise ValueError(f"Expected integer histogram counts for {description}")
    return edges, counts, counts / counts.sum()


def tail_probability(edges: np.ndarray, probabilities: np.ndarray, direction: str, threshold: float):
    matches = np.flatnonzero(np.isclose(edges, threshold, rtol=0, atol=1e-8))
    if not len(matches):
        raise ValueError(f"Tail threshold {threshold} must be an exact histogram bin edge")
    boundary = int(matches[0])
    return float(probabilities[:boundary].sum() if direction == "below"
                 else probabilities[boundary:].sum())


def describe(feature: str, label: str, source: Path, edges, counts, probabilities):
    centers = (edges[:-1] + edges[1:]) / 2
    mean = float(np.dot(probabilities, centers))
    variance = float(np.dot(probabilities, (centers - mean) ** 2))
    row = {"feature": feature, "series": label, "source_path": str(source.resolve()),
           "unit": FEATURES[feature]["unit"], "count": int(counts.sum()),
           "clip_minimum": float(edges[0]), "clip_maximum": float(edges[-1]),
           "bin_width": float(edges[1] - edges[0]),
           "clipped_bin_midpoint_mean_approx": mean,
           "clipped_bin_midpoint_std_approx": math.sqrt(max(variance, 0)),
           "lowest_bin_fraction": float(probabilities[0]),
           "highest_bin_fraction": float(probabilities[-1])}
    for name, direction, threshold in FEATURES[feature]["thresholds"]:
        row[f"fraction_{name}"] = tail_probability(edges, probabilities, direction, threshold)
    return row


def plot(loaded: dict, output: Path, *, full_range: bool):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))
    colors = ["black", "#777777", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for ax, (feature, series) in zip(axes, loaded.items()):
        feature_info = FEATURES[feature]
        for index, item in enumerate(series):
            ax.stairs(item["probabilities"], item["edges"], label=item["label"],
                      color=colors[index % len(colors)], linewidth=2.1 if index == 0 else 1.6,
                      linestyle="-" if index != 1 else "--")
        edges = series[0]["edges"]
        limits = (edges[0], edges[-1])
        if not full_range and feature_info["zoom"] is not None:
            limits = feature_info["zoom"]
        ax.set_xlim(*limits)
        visible_peaks = []
        for item in series:
            visible = (item["edges"][:-1] < limits[1]) & (item["edges"][1:] > limits[0])
            visible_peaks.append(float(item["probabilities"][visible].max()))
        ax.set_ylim(0, max(visible_peaks) * 1.1)
        ax.set_xlabel(f"{feature_info['label']} ({feature_info['unit']})")
        ax.set_ylabel("Probability per bin")
        ax.set_title(f"{feature_info['label']} · bin width {edges[1]-edges[0]:g} {feature_info['unit']}")
        ax.grid(alpha=.2, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.015),
               ncol=min(4, len(labels)), frameon=False)
    footer = "Official fixed bins; values clipped to histogram bounds; each curve sums to 1 over all bins."
    if not full_range:
        footer += " Length and width axes are zoomed; probabilities are not renormalized after zoom."
    fig.text(.5, .015, footer, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .055, 1, .925))
    stem = "vehicle_distributions_full_range" if full_range else "vehicle_distributions"
    fig.savefig(output / f"{stem}.png", dpi=200)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def write_report(output: Path, rows: list[dict], gt_field: str, full_range: bool, warnings: list[str]):
    lines = ["# Speed and size histogram comparison", "",
             f"GT source: `{gt_field}` from the warm-start/reference histogram file. "
             "Plots show probability per official bin, not density. "
             "Length and width axes are zoomed without renormalizing probabilities.", "",
             "Mean and standard deviation below use bin midpoints after official clipping. "
             "They approximate the clipped distributions and are not raw physical-value statistics. "
             "The first/last bins combine their in-range values with values clipped from outside the bounds; "
             "the histogram cannot distinguish actual underflow or overflow. "
             "Speed below 1 m/s is a first-bin proxy and does not measure the fraction exactly at rest.", "",
             "![Vehicle histogram probabilities](vehicle_distributions.png)", ""]
    for warning in warnings:
        lines.extend([f"Caution: {warning}", ""])
    if full_range:
        lines.extend(["![Full-range vehicle histogram probabilities](vehicle_distributions_full_range.png)", ""])
    for feature, info in FEATURES.items():
        threshold_labels = [name.removeprefix("fraction_") for name, _, _ in info["thresholds"]]
        lines.extend([f"## {info['label']}", "",
                      f"| Series | Count | Approx. mean ({info['unit']}) | Approx. std ({info['unit']}) | "
                      + " | ".join(f"{label} (%)" for label in threshold_labels)
                      + " | First bin (%) | Last bin (%) |",
                      "|---|---:|---:|---:|" + "---:|" * (len(threshold_labels) + 2)])
        for row in (row for row in rows if row["feature"] == feature):
            fractions = " | ".join(f"{row[f'fraction_{name}']*100:.2f}"
                                   for name, _, _ in info["thresholds"])
            lines.append(f"| {row['series']} | {row['count']:,} | "
                         f"{row['clipped_bin_midpoint_mean_approx']:.3f} | "
                         f"{row['clipped_bin_midpoint_std_approx']:.3f} | {fractions} | "
                         f"{row['lowest_bin_fraction']*100:.2f} | {row['highest_bin_fraction']*100:.2f} |")
        lines.append("")
    (output / "distributions.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt", choices=("matched_gt", "full_gt"), default="matched_gt")
    parser.add_argument("--warm-label", default="Warm start")
    parser.add_argument("--full-range", action="store_true", help="Also save plots showing all clipping bounds")
    args = parser.parse_args()
    try:
        candidates = []
        for entry in args.candidates:
            if "=" not in entry:
                raise ValueError("Candidate arguments must be LABEL=PATH")
            label, path = entry.split("=", 1)
            if not label or not path:
                raise ValueError("Candidate LABEL and PATH must both be nonempty")
            candidates.append((label, Path(path), read_json(Path(path))))
        labels = ["GT", args.warm_label] + [item[0] for item in candidates]
        if len(labels) != len(set(labels)):
            raise ValueError("GT, warm, and candidate labels must be unique")
        reference = read_json(args.reference)
        loaded = {}
        rows = []
        warnings = []
        for feature in FEATURES:
            reference_spec = reference[feature]
            series = []
            for label, source, spec, field in (
                ("GT", args.reference, reference_spec, args.gt),
                (args.warm_label, args.reference, reference_spec, "generated"),
                *((label, path, histogram[feature], "generated") for label, path, histogram in candidates),
            ):
                edges, counts, probabilities = validate_counts(spec, field, f"{label} {feature}")
                if series and not np.allclose(edges, series[0]["edges"], rtol=0, atol=1e-8):
                    raise ValueError(f"Official bin bounds differ for {label} {feature}")
                series.append({"label": label, "edges": edges, "probabilities": probabilities})
                rows.append(describe(feature, label, source, edges, counts, probabilities))
            for label, path, histogram in candidates:
                candidate_gt = np.asarray(histogram[feature][args.gt], dtype=np.float64)
                reference_gt = np.asarray(reference_spec[args.gt], dtype=np.float64)
                if not np.array_equal(candidate_gt, reference_gt):
                    warnings.append(f"{label} {feature}: candidate {args.gt} counts differ from reference; "
                                    "plot uses reference GT only. Verify scene scopes before interpreting differences.")
            loaded[feature] = series
        args.output.mkdir(parents=True, exist_ok=True)
        plot(loaded, args.output, full_range=False)
        if args.full_range:
            plot(loaded, args.output, full_range=True)
        summary = {
            "statistics_scope": "Approximate moments of official-clipped histograms using bin midpoints; not raw moments.",
            "probability_scope": "Count in bin divided by total feature count; not density. Zoom does not renormalize.",
            "tail_scope": "Fractions at exact bin-edge thresholds; boundary-bin fractions do not isolate actual overflow.",
            "reference": str(args.reference.resolve()), "gt_field": args.gt,
            "warnings": warnings, "series": rows,
        }
        (args.output / "distribution_stats.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output / "distribution_stats.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_report(args.output, rows, args.gt, args.full_range, warnings)
        for warning in warnings:
            print(f"Caution: {warning}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Distribution plotting error: {error}\n")
    print(f"Wrote {len(rows)} feature-series statistics and PNG/PDF plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
