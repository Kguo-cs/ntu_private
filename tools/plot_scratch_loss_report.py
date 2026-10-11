"""Create compact, readable report PNGs from actual scratch-search records.

Each batch/LR/evaluation-manifest/weight cohort gets separate figures. Input
may be the analyzer's aggregate.json or observations.json. Observation repeats
are averaged within generation seed, then generation seeds within training
seed; error bands describe training-seed SD, never sampler repetitions or CI.

Example:
  python tools/plot_scratch_loss_report.py \
    --aggregate docs/experiments/init_diffusion_scratch_loss_search_20261011/main_long_stage/aggregate.json \
    --cohort 8bcd9b085236 fd721a07c315 \
    --variants legacy,warm_winner,geometry_strong,geometry_strong_heading_l1 \
    --output docs/experiments/init_diffusion_scratch_loss_search_20261011/compact_plots

Only PNGs and provenance JSON/Markdown are written. No model, tensors, GPU,
training, evaluation, PDF, or external data are accessed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim_scratch_loss_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_VARIANTS = (
    "legacy", "warm_winner", "geometry_strong",
    "geometry_strong_heading_l1", "geo_mse32_col20", "balanced32_l1_quarter",
)
DEFAULT_LABELS = {
    "legacy": "Legacy", "warm_winner": "Old recipe",
    "geometry_strong": "Control", "geometry_strong_heading_l1": "Candidate",
    "geo_mse32_col20": "Collision 20", "balanced32_l1_quarter": "Size L1",
    "geometry_strong_width_p2": "Width p2",
    "geometry_strong_heading_p2": "Heading p2",
    "geometry_strong_heading_l1_width_p2": "Candidate + width p2",
    "geo_mse32_col20_heading_l1_shape2": "Geometry + size",
}
SCALES = {
    "nearest_dist_jsd": 1., "lat_dev_jsd": 1., "ang_dev_jsd": 10.,
    "length_jsd": 20., "width_jsd": 15., "speed_jsd": 20., "collision_rate": 5.,
}
VALUE_KEYS = (*SCALES, "score", "generated_onroad_fraction", "gt_onroad_fraction")
COHORT_KEYS = (
    "train_batch_size", "learning_rate", "evaluation_weights", "reference_mode",
    "num_samples", "eval_manifest_sha256", "initialization_scope",
)


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def stats(values):
    values = [value for value in values if finite(value) is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.}


def aggregate_observations(rows):
    """Respect the analyzer's hierarchy without counting generation repeats as N."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort_sha256"], row["variant"], row["budget"],
                row.get("recipe_sha256"))].append(row)
    result = []
    for _, observations in groups.items():
        cells = defaultdict(lambda: defaultdict(list))
        for row in observations:
            if row.get("seed") is None or row.get("eval_seed") is None:
                raise ValueError("Observation input needs actual training and generation seeds")
            cells[row["seed"]][row["eval_seed"]].append(row)
        seed_means = []
        for seed, generation_cells in cells.items():
            means = {"seed": seed}
            for key in VALUE_KEYS:
                generation_means = []
                for generation_rows in generation_cells.values():
                    numbers = [finite(item.get(key)) for item in generation_rows]
                    numbers = [value for value in numbers if value is not None]
                    if numbers:
                        generation_means.append(float(np.mean(numbers)))
                means[key] = float(np.mean(generation_means)) if generation_means else None
            seed_means.append(means)
        first = observations[0]
        item = {key: first.get(key) for key in (*COHORT_KEYS, "cohort_sha256",
                                                "variant", "recipe_sha256", "budget")}
        item.update(num_training_seeds=len(cells), num_observations=len(observations),
                    training_seed_means=seed_means)
        for key in VALUE_KEYS:
            item[key] = stats([value[key] for value in seed_means])
        result.append(item)
    return result


def read_rows(path):
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("aggregates", value.get("rows", value.get("observations")))
    if not isinstance(value, list) or not value:
        raise ValueError("Input must contain nonempty actual analyzer records")
    for row in value:
        if not isinstance(row, dict) or not row.get("cohort_sha256") or not row.get("variant"):
            raise ValueError("Each record needs cohort_sha256 and variant")
        budget = finite(row.get("budget"))
        if budget is None or budget <= 0 or int(budget) != budget:
            raise ValueError("Each record needs a positive actual optimizer-step budget")
    aggregated = [isinstance(row.get("score"), dict) for row in value]
    if any(aggregated) and not all(aggregated):
        raise ValueError("Do not mix aggregate and individual observation records")
    return value if all(aggregated) else aggregate_observations(value)


def summary(row, key):
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    mean, sd = finite(value.get("mean")), finite(value.get("sd"))
    if mean is None or mean < 0 or (sd is not None and sd < 0):
        return None
    return mean, sd if sd is not None else 0.


def check_cohorts(rows):
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row["cohort_sha256"]].append(row)
        score = summary(row, "score")
        parts = [summary(row, key) for key in SCALES]
        if score is None or any(part is None for part in parts):
            raise ValueError("Completed rows need score and all seven metric means")
        expected = sum(part[0] / scale for part, scale in zip(parts, SCALES.values())) / 7
        if not math.isclose(score[0], expected, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Input S does not match native-unit scales [1,1,10,20,15,20,5]")
    for cohort, items in cohorts.items():
        for key in COHORT_KEYS:
            values = {json.dumps(row.get(key), sort_keys=True) for row in items}
            if len(values) != 1:
                raise ValueError(f"Cohort {cohort[:12]} mixes {key}; regenerate analyzer groups")
        variants = defaultdict(set)
        points = set()
        for row in items:
            variants[row["variant"]].add(row.get("recipe_sha256"))
            identity = row["variant"], row["budget"]
            if identity in points:
                raise ValueError(f"Duplicate aggregate point in cohort {cohort[:12]}: {identity}")
            points.add(identity)
        if any(len(recipes) > 1 for recipes in variants.values()):
            raise ValueError("One variant name changes recipe within a cohort; label recipes separately")
    return cohorts


def cohort_title(row):
    lr = finite(row.get("learning_rate"))
    lr_text = f"{lr:g}" if lr is not None else "unknown"
    manifest = str(row.get("eval_manifest_sha256") or "unknown")[:12]
    return (f"batch {row.get('train_batch_size')} | LR {lr_text} | "
            f"{row.get('evaluation_weights')} | {row.get('reference_mode')}, n={row.get('num_samples')}\n"
            f"evaluation manifest {manifest}")


def panels(rows, *, variants, labels, colors, size, native_units):
    series = defaultdict(list)
    for row in rows:
        series[row["variant"]].append(row)
    multi_seed = any(int(row.get("num_training_seeds", 1)) > 1 for row in rows)
    specifications = (
        ("length_jsd", "Length JSD"), ("width_jsd", "Width JSD")
    ) if size else (
        ("score", "Fixed-scale S (lower is better)"),
        ("generated_onroad_fraction", "On-road vehicles (%)"),
        ("ang_dev_jsd", "Angle JSD"), ("speed_jsd", "Speed JSD"),
    )
    fig, axes = plt.subplots(2, 1 if size else 2, figsize=(10, 7))
    axes = np.asarray(axes).reshape(-1)
    for axis, (key, title) in zip(axes, specifications):
        factor = 100. if key == "generated_onroad_fraction" else (
            .01 if key.endswith("_jsd") and not native_units else 1.)
        if key.endswith("_jsd"):
            title += " x100 (native)" if native_units else " (nats; maximum ln 2)"
        for variant in variants:
            points = [(row, summary(row, key)) for row in series.get(variant, [])]
            points = sorted([(row, value) for row, value in points if value is not None],
                            key=lambda item: item[0]["budget"])
            if not points:
                continue
            x = np.asarray([row["budget"] for row, _ in points])
            y = np.asarray([value[0] for _, value in points]) * factor
            axis.plot(x, y, "o-", color=colors[variant], label=labels[variant],
                      linewidth=1.8, markersize=4)
            # Do not give single-seed points artificial uncertainty bands.
            for index, (row, value) in enumerate(points):
                if int(row.get("num_training_seeds", 1)) > 1:
                    axis.errorbar(x[index], y[index], yerr=value[1] * factor,
                                  color=colors[variant], capsize=3, linewidth=1.0)
            if all(int(row.get("num_training_seeds", 1)) > 1 for row, _ in points):
                sd = np.asarray([value[1] for _, value in points]) * factor
                axis.fill_between(x, np.maximum(0, y - sd), y + sd,
                                  color=colors[variant], alpha=.10)
        if key == "generated_onroad_fraction":
            actual_gt = [summary(row, "gt_onroad_fraction") for row in rows]
            actual_gt = [value[0] * 100 for value in actual_gt if value is not None]
            if actual_gt and np.ptp(actual_gt) < 1e-8:
                axis.axhline(actual_gt[0], color="black", linestyle="--", linewidth=1.,
                             label="Matched GT")
            axis.set_ylim(0, 102)
        else:
            axis.set_ylim(bottom=0)
        axis.set_title(title, fontsize=11, pad=8)
        axis.set_xlabel("Optimizer updates from random initialization", fontsize=11)
        axis.tick_params(labelsize=10)
        axis.grid(alpha=.25)
    handles, legends = [], []
    for axis in axes:
        actual_handles, actual_labels = axis.get_legend_handles_labels()
        for handle, label in zip(actual_handles, actual_labels):
            if label not in legends:
                handles.append(handle)
                legends.append(label)
    fig.suptitle(cohort_title(rows[0]), fontsize=12, y=.98)
    fig.legend(handles, legends, loc="lower center", bbox_to_anchor=(.5, .06),
               ncol=min(4, len(handles)), fontsize=11, frameon=False)
    uncertainty = ("Error bars/bands: training-seed SD, not confidence intervals."
                   if multi_seed else "One training seed; no training-seed SD bands.")
    fig.text(.5, .023, uncertainty, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .18, 1, .89), h_pad=1.5, w_pad=2.0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate", type=Path, required=True,
                        help="Actual analyzer aggregate.json or observations.json")
    parser.add_argument("--cohort", nargs="+", help="One or more unique cohort hash prefixes; default plots all separately")
    parser.add_argument("--variants", "--variants-csv", default=",".join(DEFAULT_VARIANTS),
                        help="Comma-separated actual variant names, in legend/color order")
    parser.add_argument("--labels-json", type=Path, help="JSON object mapping variant names to short readable labels")
    parser.add_argument("--jsd-units", choices=("native", "raw"), default="native",
                        help="native matches helper/Lightning x100; raw divides by100 to natural-log JSD (maximum ln2)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = read_rows(args.aggregate)
        cohorts = check_cohorts(rows)
        variants = list(dict.fromkeys(name.strip() for name in args.variants.split(",") if name.strip()))
        if not variants:
            raise ValueError("At least one actual variant is required")
        labels = {variant: DEFAULT_LABELS.get(variant, variant) for variant in variants}
        if args.labels_json:
            supplied = json.loads(args.labels_json.read_text())
            if not isinstance(supplied, dict) or any(not isinstance(value, str) or not value.strip()
                                                    for value in supplied.values()):
                raise ValueError("--labels-json must map full variant names to nonempty strings")
            labels.update({key: value for key, value in supplied.items() if key in labels})
        if len(set(labels.values())) != len(labels):
            raise ValueError("Short labels must be unique")
        if any(len(label) > 28 for label in labels.values()):
            raise ValueError("Legend labels must be at most 28 characters; supply --labels-json")
        selected = set(cohorts)
        if args.cohort:
            selected = set()
            for prefix in args.cohort:
                matches = [cohort for cohort in cohorts if cohort.startswith(prefix)]
                if len(matches) != 1:
                    raise ValueError(f"Cohort prefix {prefix!r} matches {len(matches)} cohorts; use a unique prefix")
                selected.add(matches[0])
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("tab10")
    colors = {variant: palette(index % 10) for index, variant in enumerate(variants)}
    generated, groups, absent = [], [], []
    with plt.rc_context({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 11}):
        for cohort in sorted(selected):
            group = [row for row in cohorts[cohort] if row["variant"] in variants]
            if not group:
                absent.append(cohort)
                continue
            for size in (False, True):
                figure = panels(group, variants=variants, labels=labels, colors=colors,
                                size=size, native_units=args.jsd_units == "native")
                path = output / f"{'size' if size else 'summary'}_{cohort[:12]}_{args.jsd_units}.png"
                figure.savefig(path, dpi=180)
                plt.close(figure)
                generated.append(path.name)
            groups.append({"cohort_sha256": cohort, **{key: group[0].get(key) for key in COHORT_KEYS},
                           "actual_variants": sorted({row["variant"] for row in group}),
                           "actual_budgets": sorted({row["budget"] for row in group}),
                           "num_aggregate_points": len(group),
                           "num_observations": sum(row.get("num_observations", 1) for row in group),
                           "training_seed_counts": sorted({row.get("num_training_seeds", 1) for row in group})})
    if not generated:
        parser.error("No selected actual rows; no plots generated")
    protocol = {"input": str(args.aggregate.resolve()),
                "input_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "jsd_units": args.jsd_units, "native_jsd_multiplier": 100,
                "jsd_log_base": "natural", "raw_jsd_theoretical_upper_bound": math.log(2),
                "score_native_scales": SCALES, "variant_labels": labels,
                "requested_variants": variants, "cohorts": groups,
                "empty_selected_cohorts": absent, "figures": generated,
                "figure_size_inches": [10, 7], "base_font_size_pt": 11,
                "uncertainty": "Training-seed sample SD; generation repetitions are averaged inside training seed; not CI",
                "training_seed_sd_ddof": 1, "single_training_seed_sd": 0,
                "scope": "Actual recorded points only; no simulated results, model, GPU, or PDF"}
    (output / "plot_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Compact scratch-loss report figures", "",
             "PNG only, based on actual analyzer records. Each cohort is plotted separately.", "",
             "| Short label | Full recipe variant |", "|---|---|"]
    lines.extend(f"| {label} | `{variant}` |" for variant, label in labels.items())
    lines += ["", f"JSD units: `{args.jsd_units}`. Current native angle/speed/size metrics are natural-log JSD x100; raw JSD has maximum ln2.",
              "Error bars/bands use independent training-seed SD, not confidence intervals.", ""]
    lines.extend(f"![{name}]({name})" for name in generated)
    (output / "figures.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"figures": generated, "cohorts": len(groups), "output": str(output)}))


if __name__ == "__main__":
    main()
