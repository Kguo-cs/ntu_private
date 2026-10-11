"""CPU-only figures from actual protocol-root frozen-map search observations.

Incomplete stages are recorded as incomplete/skipped. No missing observation is
imputed, no model is loaded, and generation repetitions never increase TRAIN N.
Refresh after later stages finish with the same command used for the first run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim_frozen_map_plot_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.text import Text
import numpy as np

from tools.analyze_frozen_map_search import (
    ATTR, EXCLUDED_PILOT_STAGES, GATES, METRICS, MULTIPLIERS,
    SCALES, arm_statistics, compare, discover, training_seed_means,
)
from tools.plot_component_loss_distributions import FEATURES, describe, validate_counts

ART = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
LABELS = dict(zip(METRICS, ("Nearest", "Lateral", "Angle", "Length", "Width", "Speed", "Collision")))
PALETTE = ("#333333", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
REPORT_WIDTH_PT = 511.0
REPORT_MAX_HEIGHT_PT = 490.0
FIGURE_WIDTH_IN = REPORT_WIDTH_PT / 72.0
MIN_SOURCE_FONT_PT = 9.0
MIN_REPORT_FONT_PT = 8.0
SAVE_DPI = 220
MAX_PHASE2_ROWS = 13
FIGURE_LAYOUTS = {}


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figure(fig, output, name):
    """Keep the PNG canvas at report width and audit visible text in points."""
    width, height = map(float, fig.get_size_inches())
    if width > FIGURE_WIDTH_IN + 1e-6:
        raise ValueError(f"{name}: native figure is wider than {REPORT_WIDTH_PT:g} pt")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    off_view_ticks = set()
    for axis in fig.axes:
        for labels, coordinate, bounds in ((axis.get_xticklabels(), 0, axis.get_xlim()),
                                            (axis.get_yticklabels(), 1, axis.get_ylim())):
            low, high = sorted(bounds)
            for label in labels:
                # Matplotlib retains locator labels just outside the displayed
                # limits, but does not draw them. They are not clipped PNG text.
                if not low - 1e-9 <= label.get_position()[coordinate] <= high + 1e-9:
                    off_view_ticks.add(id(label))
    texts = [artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text().strip()
             and id(artist) not in off_view_ticks]
    fonts = [float(artist.get_fontsize()) for artist in texts]
    minimum = min(fonts) if fonts else MIN_SOURCE_FONT_PT
    width_fit_scale = min(1.0, REPORT_WIDTH_PT / (width * 72.0))
    fit_scale = min(width_fit_scale, REPORT_MAX_HEIGHT_PT / (height * 72.0))
    if minimum < MIN_SOURCE_FONT_PT or minimum * fit_scale < MIN_REPORT_FONT_PT:
        raise ValueError(f"{name}: text would be smaller than the declared report font limit")
    canvas_width, canvas_height = fig.canvas.get_width_height()
    overflow = []
    for artist in texts:
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > canvas_width + 1 or bounds.y1 > canvas_height + 1:
            overflow.append(artist.get_text())
    if overflow:
        raise ValueError(f"{name}: text extends beyond the native canvas: {overflow[:3]}")
    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        # A tight bounding box can silently change physical width. Layout is
        # resolved before saving; the exported canvas retains native dimensions.
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches=None, facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        paths.append(str(path.resolve()))
    with (output / f"{name}.png").open("rb") as handle:
        handle.seek(16)
        png_width, png_height = struct.unpack(">II", handle.read(8))
    FIGURE_LAYOUTS[name] = {
        "native_width_in": width, "native_height_in": height,
        "native_width_pt": width * 72.0, "native_height_pt": height * 72.0,
        "PNG_width_px": png_width, "PNG_height_px": png_height, "PNG_DPI": SAVE_DPI,
        "visible_text_count": len(fonts), "source_font_sizes_pt": sorted(set(fonts)),
        "minimum_source_font_pt": minimum, "report_fit_width_pt": REPORT_WIDTH_PT,
        "report_fit_max_height_pt": REPORT_MAX_HEIGHT_PT,
        "width_fit_scale": width_fit_scale, "minimum_effective_font_at_report_width_pt": minimum * width_fit_scale,
        "report_box_fit_scale": fit_scale, "minimum_effective_font_at_report_box_pt": minimum * fit_scale,
        "minimum_display_width_for_8pt_font_pt": width * 72.0 * MIN_REPORT_FONT_PT / minimum,
        "minimum_display_height_for_8pt_font_pt": height * 72.0 * MIN_REPORT_FONT_PT / minimum,
        "height_limit_warning": "Any additional height fitting must retain at least the declared minimum display dimensions.",
        "nonempty_visible_text_outside_canvas": overflow, "files": paths,
    }
    plt.close(fig)
    return paths


def compact(name, width=25):
    return "\n".join(textwrap.wrap(name.replace("_", " "), width=width))


def compatible_controls(candidate, controls):
    """Match observed sample exposure and evaluation conditions, including GEN."""
    references = [row for row in controls
                  if row["cohort_sha256"] == candidate[0]["cohort_sha256"]
                  and row["seen_training_examples"] == candidate[0]["seen_training_examples"]
                  and row["evaluation_weights"] == candidate[0]["evaluation_weights"]]
    methods = {(row["budget"], row["train_batch_size"], row["arm_config_sha256"]) for row in references}
    if len(methods) > 1:
        raise ValueError("Ambiguous control methods at the same observed exposure/cohort")
    return references


def comparison_points(rows, controls):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort_sha256"], row["seen_training_examples"], row["evaluation_weights"], row["arm_id"])].append(row)
    points, skipped = [], []
    for _, cells in sorted(groups.items()):
        baseline = compatible_controls(cells, controls)
        if not baseline:
            skipped.append({"variant": cells[0]["variant"], "reason": "No same-cohort/exposure control"})
            continue
        comparison = compare(cells, baseline, external=any(row["run_path"] != cells[0]["run_path"] for row in baseline))
        if not comparison["generation_seed_pairing_complete"]:
            skipped.append({"variant": cells[0]["variant"], "reason": "TRAIN/GEN pairing incomplete"})
            continue
        ratios = [comparison["components"][key]["ratio_of_arm_means"] for key in METRICS]
        if any(value is None or not math.isfinite(value) for value in ratios):
            skipped.append({"variant": cells[0]["variant"], "reason": "Undefined metric ratio"})
            continue
        stats = arm_statistics(cells)
        points.append({"variant": cells[0]["variant"], "arm_id": cells[0]["arm_id"],
                       "cohort_sha256": cells[0]["cohort_sha256"],
                       "seen_training_examples": cells[0]["seen_training_examples"],
                       "budgets": sorted({row["budget"] for row in cells}),
                       "batch_sizes": sorted({row["train_batch_size"] for row in cells}),
                       "ratios": dict(zip(METRICS, ratios)), "comparison": comparison,
                       "statistics": stats, "sources": sorted({row["summary_source"] for row in cells})})
    return points, skipped


def heatmap(points, output, name, title, detail):
    values = np.asarray([[item["ratios"][key] for key in METRICS] for item in points])
    height = max(4.3, 2.15 + len(points) * .32)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, height))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2))
    row_labels = [compact(item["variant"], 18) + (" *" if not item["comparison"]["gates_pass"] else "")
                  for item in points]
    ax.set_xticks(range(len(METRICS)), ["Dist", "Lat", "Ang", "Len", "Wid", "Speed", "Col"], rotation=0)
    ax.set_yticks(range(len(points)), row_labels)
    for i in range(len(points)):
        for j in range(len(METRICS)):
            value = values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if value < .45 or value > 1.55 else "#111111")
    ax.tick_params(length=0, pad=5, labelsize=9.5)
    ax.set_xticks(np.arange(-.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(points), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(im, ax=ax, fraction=.038, pad=.03, extend="both")
    colorbar.set_label("Ratio (lower is better)", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, pad=12)
    footer = detail + "\nColor clipped to 0–2; numbers show actual ratios; * gate FAIL."
    fig.text(.5, .022, footer,
             ha="center", va="bottom", fontsize=9)
    footer_height_fraction = ((footer.count("\n") + 1) * 12 + 14) / (height * 72)
    fig.tight_layout(rect=(0, max(.18, footer_height_fraction), 1, .99))
    return save_figure(fig, output, name)


def phase1_plot(rows, artifacts, output, weights):
    protocol = read(artifacts / "search_protocol.json", {})
    expected = protocol.get("phase1", {}).get("arms", [])
    source = [row for row in rows if Path(row["run_path"]).name == "phase1"
              and row["budget"] == 512 and row["evaluation_weights"] == weights]
    control = [row for row in source if row["variant"] == "new_control"]
    points, skipped = comparison_points(source, control)
    order = {name: i for i, name in enumerate(expected)}
    points.sort(key=lambda item: (order.get(item["variant"], 999), item["variant"]))
    missing = [name for name in expected if name not in {item["variant"] for item in points}]
    info = {"status": "incomplete" if missing else "complete", "points": points,
            "missing_variants": missing, "skipped_observations": skipped, "files": []}
    if not points:
        info.update(status="skipped", reason="No paired phase1 EMA512 endpoints yet")
        return info
    cohort_ids = {item["cohort_sha256"] for item in points}
    if len(cohort_ids) != 1:
        raise ValueError("Phase1 heatmap cannot combine different evaluation cohorts")
    count = f"Observed {len(points)}/{len(expected)} arms" if expected else f"Observed {len(points)} arms"
    train_n = sorted({item["statistics"]["num_independent_training_seeds"] for item in points})
    detail = (f"{source[0]['num_samples']:,} scenes; B512; 512 updates; {weights.upper()}; {count}; TRAIN N={','.join(map(str, train_n))}.\n"
              "Gate: road ≥ control − 2 pp; vehicle/GT 0.9–1.1;\n"
              "length, width and speed ≤ 1.15× control.")
    info["files"] = heatmap(points, output, "phase1_relative_metrics", "Fixed map: initial loss sweep", detail)
    return info


def extended_loss_plots(rows, artifacts, output, weights, exposure):
    """Only the continued arms and recorded fresh interaction have late data."""
    protocol = read(artifacts / "search_protocol.json", {})
    continued = list(read(artifacts / "design/phase1_extend_resume.json", {}))
    joint_enabled = read(artifacts / "loss_interaction_protocol.json", {}).get("enabled", False)
    expected = continued + (["shape_p2_speed4"] if joint_enabled else [])
    source = [row for row in rows if Path(row["run_path"]).name in ("phase1_extend", "loss_interaction")
              and row["seen_training_examples"] == exposure and row["evaluation_weights"] == weights]
    control = [row for row in source if row["variant"] == "new_control"]
    points, skipped = comparison_points(source, control)
    points.sort(key=lambda item: (item["variant"] != "new_control", item["variant"] == "shape_p2_speed4", item["variant"]))
    observed = {item["variant"] for item in points}
    unobserved = [name for name in protocol.get("phase1", {}).get("arms", []) if name not in observed]
    missing = [name for name in expected if name not in observed]
    info = {"status": "incomplete" if missing else "complete", "equal_observed_sample_exposure": exposure,
            "points": points, "expected_continued_variants": continued, "fresh_interaction_enabled": joint_enabled,
            "missing_expected_variants": missing, "unobserved_768_competitors": unobserved,
            "missing_observation_policy": "Only four selected arms were continued; all other768-step competitors remain unobserved. No values are padded or inferred.",
            "skipped_observations": skipped, "files": []}
    if not points:
        info.update(status="skipped", reason="No paired extended/joint loss endpoints yet")
        return info
    if len({item["cohort_sha256"] for item in points}) != 1:
        raise ValueError("Extended loss comparison cannot combine evaluation cohorts")
    if any(item["batch_sizes"] != [512] or item["budgets"] != [768] for item in points):
        raise ValueError("Extended loss figure requires the declared B512/768 exposure")
    train_n = sorted({item["statistics"]["num_independent_training_seeds"] for item in points})
    detail = (f"{source[0]['num_samples']:,} scenes; {weights.upper()}; B512 × 768 = {exposure:,} TRAIN examples.\n"
              f"Observed {len(observed & set(continued))} continued arms + {int('shape_p2_speed4' in observed)} fresh joint; TRAIN N={','.join(map(str, train_n))}.\n"
              "Gate: road ≥ control − 2 pp; vehicle/GT 0.9–1.1; attributes ≤ 1.15×.\n"
              "Other 768-step competitors unobserved; no estimates.")
    info["files"] = heatmap(points, output, "phase1_extended_joint_relative_metrics",
                             "Fixed map: extended and joint losses", detail)
    # A changing control denominator is explicit: these are observed relative
    # scores, never extrapolated learning curves or predictions of final quality.
    progression = [row for row in rows if row["variant"] in observed
                   and Path(row["run_path"]).name in ("phase1", "phase1_extend", "loss_interaction")
                   and row["budget"] in (256, 512, 768) and row["evaluation_weights"] == weights
                   and row["cohort_sha256"] == points[0]["cohort_sha256"]]
    controls = [row for row in progression if row["variant"] == "new_control"]
    trend_points, trend_skipped = comparison_points(progression, controls)
    curves = defaultdict(list)
    for item in trend_points:
        curves[item["variant"]].append(item)
    for name, values in curves.items():
        if len(values) != len({item["seen_training_examples"] for item in values}):
            raise ValueError(f"Ambiguous loss progression arm at identical exposure: {name}")
    info["observed_progression"] = dict(curves)
    info["skipped_progression"] = trend_skipped
    for key, label in (("R_ratio_of_arm_means", "Seven-metric ratio R"),
                       ("R_attr_ratio_of_arm_means", "Attribute ratio R_attr")):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.7))
        for index, item in enumerate(points):
            values = sorted(curves.get(item["variant"], []), key=lambda cell: cell["seen_training_examples"])
            x = [cell["seen_training_examples"] / 1000 for cell in values]
            y = [cell["comparison"][key] for cell in values]
            ax.plot(x, y, "o-", label=compact(item["variant"], 21), color=PALETTE[index % len(PALETTE)], linewidth=1.8)
        ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
        ax.set_title("Observed loss arms: " + label, fontsize=12)
        ax.set_xlabel("Observed TRAIN examples (thousands)")
        ax.set_ylabel(label + " (lower is better)")
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(.5, .09), fontsize=9.5)
        fig.text(.5, .012, f"Matched control at each exposure; same {source[0]['num_samples']:,} scenes; {weights.upper()}; TRAIN N={','.join(map(str, train_n))}.\n"
                 "Only observed 256/512/768 endpoints; no extrapolation or missing-arm estimates.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .29, 1, .98))
        info["files"].extend(save_figure(fig, output, "phase1_loss_progression_" + ("R" if key == "R_ratio_of_arm_means" else "R_attr")))
    return info


def phase2_plot(rows, artifacts, output, weights, exposure):
    source = [row for row in rows if (Path(row["run_path"]).name.startswith("phase2")
              or Path(row["run_path"]).name.startswith("combo_"))
              and row["seen_training_examples"] == exposure and row["evaluation_weights"] == weights]
    controls = [row for row in rows if row["variant"] == "new_control"
                and Path(row["run_path"]).name in ("phase1", "phase1_extend")
                and row["seen_training_examples"] == exposure and row["evaluation_weights"] == weights]
    points, skipped = comparison_points(source, controls)
    expected = read(artifacts / "search_protocol.json", {}).get("phase2", {}).get("parameter_names", [])
    missing = [name for name in expected if name not in {item["variant"] for item in points}]
    info = {"status": "incomplete" if missing else "complete", "equal_observed_sample_exposure": exposure,
            "points": points, "missing_variants": missing, "skipped_observations": skipped, "files": []}
    if not points:
        info.update(status="skipped", reason="No paired phase2 endpoints at the declared equal exposure yet")
        return info
    cohorts = {item["cohort_sha256"] for item in points}
    if len(cohorts) != 1:
        raise ValueError("Phase2 comparison cannot combine different evaluation cohorts")
    points.sort(key=lambda item: (not item["comparison"]["gates_pass"], item["comparison"]["R_ratio_of_arm_means"], item["variant"]))
    batch_budgets = defaultdict(int)
    for item in points:
        batch_budgets[(item["batch_sizes"][0], item["budgets"][0])] += 1
    common_batch, common_budget = max(batch_budgets, key=batch_budgets.get)
    exceptions = sorted(method for method in batch_budgets if method != (common_batch, common_budget))
    exception_tags = {method: "†" + (str(index) if len(exceptions) > 1 else "")
                      for index, method in enumerate(exceptions, 1)}
    exception_detail = "; ".join(f"{exception_tags[method]} B{method[0]}/{method[1]} updates" for method in exceptions)
    info["batch_exceptions"] = [{"marker": exception_tags[method], "batch_size": method[0], "updates": method[1]}
                                for method in exceptions]
    keys = (("R_ratio_of_arm_means", "Seven-metric ratio R"),
            ("R_attr_ratio_of_arm_means", "Attribute ratio R_attr"))
    ranges = {}
    for key, _ in keys:
        values = [item["comparison"][key] for item in points]
        xmin, xmax = min(min(values), 1), max(max(values), 1)
        span = max(.12, xmax - xmin)
        ranges[key] = (max(0, xmin - span * .1), xmax + span * .7)
    chunks = [points[start:start + MAX_PHASE2_ROWS] for start in range(0, len(points), MAX_PHASE2_ROWS)]
    info["parts"] = []
    for part_index, part in enumerate(chunks, 1):
        # Bound physical height as well as width. All pages use the same x
        # ranges, preserving visual comparisons across the observed arms.
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, max(4.5, 1.8 + len(part) * .41)), sharey=True)
        y = np.arange(len(part))
        labels = [compact(item["variant"], 19) + (" *" if not item["comparison"]["gates_pass"] else "")
                  + (" " + exception_tags[(item["batch_sizes"][0], item["budgets"][0])]
                     if (item["batch_sizes"][0], item["budgets"][0]) in exception_tags else "")
                  for item in part]
        for ax, (key, label) in zip(axes, keys):
            values = [item["comparison"][key] for item in part]
            colors = ["#0072B2" if item["comparison"]["gates_pass"] else "#D55E00" for item in part]
            ax.scatter(values, y, c=colors, s=65, zorder=3)
            ax.axvline(1, color="#777777", linestyle="--", linewidth=1.2)
            ax.set_xlim(*ranges[key])
            for i, value in enumerate(values):
                ax.annotate(f"{value:.3f}", (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
            ax.set_xlabel(label + "\n(lower is better)", fontsize=10)
            ax.grid(axis="x", alpha=.2)
        axes[0].set_yticks(y, labels)
        axes[0].tick_params(axis="y", labelsize=9.5)
        axes[0].invert_yaxis()
        page_label = f" ({part_index}/{len(chunks)})" if len(chunks) > 1 else ""
        fig.suptitle("Fixed map: equal-exposure interventions" + page_label, fontsize=12)
        detail = f"{exposure:,} TRAIN examples; {weights.upper()}; default B{common_batch}/{common_budget} updates.\n"
        if exception_detail:
            detail += exception_detail + ".\n"
        detail += "Same cohort; GEN mean within TRAIN; * gate FAIL."
        fig.text(.5, .025, detail,
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .94))
        name = "phase2_equal_exposure" + (f"_part{part_index:02d}" if len(chunks) > 1 else "")
        files = save_figure(fig, output, name)
        info["files"].extend(files)
        info["parts"].append({"part": part_index, "variants": [item["variant"] for item in part],
                              "common_x_ranges": ranges, "files": files})
    return info


def long_curves(rows, output, weights, train_scenes):
    source = [row for row in rows if row["num_samples"] == 4096 and row["evaluation_weights"] == weights
              and (Path(row["run_path"]).name.startswith("long_") or Path(row["run_path"]).name.startswith("confirm_"))]
    info = {"status": "skipped", "reason": "No long/confirmation 4096-scene observations yet", "files": [], "cohorts": [],
            "skipped_incomplete_generation_cells": [], "generation_plan_unknown_cells": []}
    if not source:
        return info
    cohorts = defaultdict(list)
    for row in source:
        cohorts[row["cohort_sha256"]].append(row)
    info.pop("reason")
    info["status"] = "actual_observed_curves"
    generation_plans = {}
    for row in source:
        run = row["run_path"]
        if run not in generation_plans:
            protocol = read(Path(run) / "protocol.json", {})
            planned = [protocol.get("eval_seed"), protocol.get("extra_eval_seed")]
            generation_plans[run] = tuple(sorted({seed for seed in planned if type(seed) is int}))
    for cohort, cells in sorted(cohorts.items()):
        groups = defaultdict(list)
        for row in cells:
            groups[(row["arm_id"], row["seen_training_examples"])].append(row)
        curves = defaultdict(list)
        for (arm, exposure), values in sorted(groups.items()):
            methods = {(row["train_batch_size"], row["budget"]) for row in values}
            if len(methods) != 1:
                raise ValueError("One long-curve arm/exposure has conflicting batch/update methods")
            per_train = defaultdict(list)
            for row in values:
                per_train[row["seed"]].append(row)
            completed = []
            for train_seed, repetitions in sorted(per_train.items()):
                observed = sorted({row["eval_seed"] for row in repetitions})
                if len(repetitions) != len(observed):
                    raise ValueError("Conflicting duplicate long-curve TRAIN/GEN cell at identical exposure")
                plans = {generation_plans[row["run_path"]] for row in repetitions
                         if generation_plans[row["run_path"]]}
                if len(plans) > 1:
                    raise ValueError("Long-curve TRAIN cell has inconsistent planned GEN seeds")
                planned = list(next(iter(plans))) if plans else None
                cell = {"arm_id": arm, "training_seed": train_seed, "seen_training_examples": exposure,
                        "cohort_sha256": cohort, "observed_generation_seeds": observed,
                        "planned_generation_seeds": planned,
                        "sources": sorted({row["summary_source"] for row in repetitions})}
                if planned is not None and observed != planned:
                    info["skipped_incomplete_generation_cells"].append(cell)
                    continue
                if planned is None:
                    info["generation_plan_unknown_cells"].append(cell)
                completed.extend(repetitions)
            if not completed:
                continue
            stat = arm_statistics(completed)
            curves[arm].append({"x_observed_training_epochs": exposure / train_scenes,
                                "seen_training_examples": exposure, "statistics": stat,
                                "budgets": sorted({row["budget"] for row in values}),
                                "batch_sizes": sorted({row["train_batch_size"] for row in values}),
                                "sources": sorted({row["summary_source"] for row in completed}),
                                "generation_pairs_complete_when_plan_known": True})
        if not curves:
            continue
        ordered = sorted(curves, key=lambda arm: (curves[arm][0]["statistics"]["variant"] != "new_control", arm))
        files = []
        suffix = "" if len(cohorts) == 1 else "_" + cohort[:10]
        # One metric per report-width figure keeps axes and SD legible. An
        # eight-panel canvas would require unreadable reduction in a portrait PDF.
        for key in METRICS + ("training_seed_count",):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.35 if key in METRICS else 3.8))
            for index, arm in enumerate(ordered):
                points = sorted(curves[arm], key=lambda item: item["seen_training_examples"])
                color = PALETTE[index % len(PALETTE)]
                counts = sorted({point["statistics"]["num_independent_training_seeds"] for point in points})
                batches = sorted({batch for point in points for batch in point["batch_sizes"]})
                name = (compact(points[0]["statistics"]["variant"], 22)
                        + "\nB" + ",".join(map(str, batches)) + "; TRAIN N=" + ",".join(map(str, counts)))
                x = [item["x_observed_training_epochs"] for item in points]
                if key == "training_seed_count":
                    values = [item["statistics"]["num_independent_training_seeds"] for item in points]
                    ax.plot(x, values, "o-", color=color, linewidth=1.8, markersize=5, label=name)
                    continue
                means = np.asarray([item["statistics"]["statistics_native"][key]["mean"] / MULTIPLIERS[key] for item in points])
                deviations = np.asarray([item["statistics"]["statistics_native"][key]["sd"] / MULTIPLIERS[key] for item in points])
                n = np.asarray([item["statistics"]["num_independent_training_seeds"] for item in points])
                ax.plot(x, means, "o-", color=color, label=name, linewidth=1.8, markersize=5)
                # No SD band for a single TRAIN seed; do not invent uncertainty.
                ax.fill_between(x, means - deviations, means + deviations, where=n > 1, color=color, alpha=.14)
                if np.any(n > 1):
                    ax.errorbar(np.asarray(x)[n > 1], means[n > 1], yerr=deviations[n > 1], fmt="none", color=color, capsize=3, linewidth=1)
            if key == "training_seed_count":
                ax.set_title("Independent TRAIN seeds per observed checkpoint", fontsize=12)
                ax.set_ylabel("TRAIN N (GEN does not add N)")
                ax.set_yticks([1, 2, 3])
                ax.set_ylim(.6, 3.4)
            else:
                metric_title = "Collision rate (%)" if key == "collision_rate" else LABELS[key] + " JSD"
                ax.set_title(f"4,096 scenes; {weights.upper()}: {metric_title}", fontsize=12)
                ax.set_ylabel("Collision rate (%)" if key == "collision_rate" else "JSD (natural logarithm)")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ax.set_xlabel(f"TRAIN exposures / {train_scenes:,} (equivalent full epochs)")
            ax.grid(alpha=.2)
            handles, names = ax.get_legend_handles_labels()
            fig.legend(handles, names, loc="lower center", ncol=min(3, len(names)), frameon=False,
                       bbox_to_anchor=(.5, .09), fontsize=9.5, columnspacing=1.1, handlelength=1.6)
            seeds = sorted({seed for values in curves.values() for point in values
                            for seed in point["statistics"]["training_seed_ids"]})
            fig.text(.5, .012, "Observed TRAIN " + ",".join(map(str, seeds)) + "; GEN mean within TRAIN; sample SD across TRAIN.\n"
                     "Observed checkpoints; TRAIN membership may vary; N=1 has no SD band.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .29, 1, .98))
            files.extend(save_figure(fig, output, "long_confirmation_" + key + suffix))
        info["files"].extend(files)
        membership_changes = {arm: len({tuple(point["statistics"]["training_seed_ids"]) for point in points}) > 1
                              for arm, points in curves.items()}
        info["cohorts"].append({"cohort_sha256": cohort, "curves": dict(curves), "files": files,
                                "TRAIN_membership_changes_by_arm": membership_changes,
                                "uncertainty_scope": "Sample SD across independent TRAIN means after GEN averaging; never GEN-as-TRAIN N. A varying-N curve can change TRAIN membership across checkpoints."})
    if not info["cohorts"]:
        info.update(status="skipped", reason="No complete planned GEN cells in the observed long/confirmation records")
    return info


def full_histograms(rows, artifacts, output, weights, requested):
    source = [row for row in rows if row["num_samples"] == 50000 and row["evaluation_weights"] == weights
              and Path(row["run_path"]).name.startswith("full50000")]
    info = {"status": "skipped", "reason": "Two paired full50000 evaluator records are not available yet", "files": []}
    expected = requested or [arm["name"] for arm in read(artifacts / "frozen_confirmation_protocol.json", {}).get("arms", [])]
    if expected:
        source = [row for row in source if row["variant"] in expected]
    if len(source) < 2:
        info["actual_variants"] = sorted({row["variant"] for row in source})
        return info
    if len(source) != 2 or len({row["variant"] for row in source}) != 2:
        raise ValueError("Full50000 histograms require exactly two declared arm records; use --full-variants")
    source.sort(key=lambda row: (row["variant"] != "new_control", row["variant"]))
    if source[0]["cohort_sha256"] != source[1]["cohort_sha256"] or (source[0]["seed"], source[0]["eval_seed"]) != (source[1]["seed"], source[1]["eval_seed"]):
        raise ValueError("Full50000 arms have different evaluation cohort or TRAIN/GEN seed")
    loaded = []
    for row in source:
        histogram_path = Path(row["report_source"]).parent / "feature_histograms.json"
        if not histogram_path.is_file():
            info.update(reason=f"Missing saved histograms: {histogram_path}")
            return info
        loaded.append((row, histogram_path, read(histogram_path)))
    statistics, files = [], []
    for feature, spec in FEATURES.items():
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.5))
        edges, gt_counts, gt_probability = validate_counts(loaded[0][2][feature], "matched_gt", "Matched GT")
        ax.stairs(gt_probability, edges, label="Matched GT", color=PALETTE[0], linewidth=2)
        statistics.append(describe(feature, "Matched GT", loaded[0][1], edges, gt_counts, gt_probability))
        peaks = [gt_probability]
        for i, (row, path, histogram) in enumerate(loaded, 1):
            ref_edges, counts, _ = validate_counts(histogram[feature], "matched_gt", row["variant"] + " GT")
            if not np.array_equal(edges, ref_edges) or not np.array_equal(gt_counts, counts):
                raise ValueError("Matched GT histogram counts or edges differ across full50000 arms")
            gen_edges, counts, probability = validate_counts(histogram[feature], "generated", row["variant"])
            if not np.array_equal(edges, gen_edges):
                raise ValueError("Generated histogram edges differ across full50000 arms")
            ax.stairs(probability, edges, label=compact(row["variant"], 22), color=PALETTE[i], linewidth=1.8)
            peaks.append(probability)
            statistics.append(describe(feature, row["variant"], path, edges, counts, probability))
        limits = spec["zoom"] or (float(edges[0]), float(edges[-1]))
        visible = (edges[:-1] < limits[1]) & (edges[1:] > limits[0])
        ax.set_xlim(*limits)
        ax.set_ylim(0, 1.12 * max(float(value[visible].max()) for value in peaks))
        ax.set_title(f"50,000 scenes: {spec['label']} (bin {loaded[0][2][feature]['bin_width']:g} {spec['unit']})", fontsize=12)
        ax.set_xlabel(f"{spec['label']} ({spec['unit']})")
        ax.set_ylabel("Probability per official bin")
        ax.grid(axis="y", alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(.5, .065), fontsize=9.5)
        fig.text(.5, .012, "Saved official bins; bounds clip values. Size axes zoom without renormalization.\n"
                 "Metadata midpoint moments approximate clipped histograms, not raw values.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .24, 1, .98))
        files.extend(save_figure(fig, output, "full50000_" + feature + "_distribution"))
    first = source[0]
    if first["num_gt_vehicles"] != source[1]["num_gt_vehicles"] or first["num_gt_onroad_vehicles"] != source[1]["num_gt_onroad_vehicles"]:
        raise ValueError("Matched GT vehicle/on-road counts differ across full50000 arms")
    labels = ["Matched GT"] + [compact(row["variant"], 18) for row in source]
    coverage = [first["gt_onroad_fraction"]] + [row["generated_onroad_fraction"] for row in source]
    counts = [first["num_gt_vehicles"]] + [row["num_generated_vehicles"] for row in source]
    if any(value is None or not 0 <= value <= 1 for value in coverage):
        raise ValueError("Missing/invalid full50000 road coverage")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.1))
    ax.bar(range(3), np.asarray(coverage) * 100, color=PALETTE[:3], width=.58)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 121)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("50,000 scenes: vehicle / on-road coverage", fontsize=12)
    ax.set_ylabel("On-road vehicles (%)")
    ax.grid(axis="y", alpha=.2)
    for i, (value, count) in enumerate(zip(coverage, counts)):
        ax.text(i, value * 100 + 2, f"{value * 100:.1f}%\nN={int(count):,}", ha="center", va="bottom", fontsize=10)
    fig.text(.5, .02, "Same evaluation scenes and TRAIN/GEN seeds; counts show evaluated vehicles.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .98))
    files.extend(save_figure(fig, output, "full50000_road_coverage"))
    info.update(status="complete", statistics=statistics, source_rows=source,
                histogram_sources=[{"path": str(path.resolve()), "sha256": sha256(path)} for _, path, _ in loaded],
                files=files)
    info.pop("reason")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=[ROOT / "logs/component_loss_frozen_map"])
    parser.add_argument("--artifacts", type=Path, default=ART)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument("--train-scenes", type=int, default=493542)
    parser.add_argument("--phase2-exposure", type=int)
    parser.add_argument("--full-variants", help="Exactly two comma-separated declared full50000 arm names")
    args = parser.parse_args()
    if args.train_scenes < 1:
        parser.error("--train-scenes must be positive")
    output = args.output or args.artifacts / "figures"
    output.mkdir(parents=True, exist_ok=True)
    FIGURE_LAYOUTS.clear()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "svg.fonttype": "none"})
    initial_hashes = {str(path.resolve()): sha256(path) for run in args.runs for path in run.rglob("summary.json")
                      if (path.parent / "protocol.json").is_file()}
    rows, warnings, _ = discover(args.runs)
    declared = read(args.artifacts / "search_protocol.json", {})
    exposure = args.phase2_exposure or declared.get("phase2", {}).get("equal_exposure_to_phase1_extended", 393216)
    requested = args.full_variants.split(",") if args.full_variants else []
    if requested and (len(requested) != 2 or len(set(requested)) != 2):
        parser.error("--full-variants must contain exactly two distinct names")
    figures = {
        "phase1": phase1_plot(rows, args.artifacts, output, args.weights),
        "extended_joint_losses": extended_loss_plots(rows, args.artifacts, output, args.weights, exposure),
        "phase2": phase2_plot(rows, args.artifacts, output, args.weights, exposure),
        "long_confirmation": long_curves(rows, output, args.weights, args.train_scenes),
        "full50000": full_histograms(rows, args.artifacts, output, args.weights, requested),
    }
    sources = []
    for path in sorted({row["summary_source"] for row in rows}):
        current = sha256(path)
        stable = initial_hashes.get(path) == current
        sources.append({"path": path, "sha256_before_discovery": initial_hashes.get(path),
                        "sha256_after_render": current, "file_stable_during_render": stable})
        if not stable:
            warnings.append(f"Summary changed during render; figures reflect only the discovered rows: {path}")
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "tool": str(Path(__file__).resolve()), "tool_sha256": sha256(__file__),
                "actual_observation_count": len(rows), "source_roots": [str(path.resolve()) for path in args.runs],
                "summary_sources": sources,
                "excluded_pilot_stages": EXCLUDED_PILOT_STAGES, "weights": args.weights,
                "minimum_font_size": MIN_SOURCE_FONT_PT,
                "report_figure_layout": {"fit_width_pt": REPORT_WIDTH_PT, "fit_max_height_pt": REPORT_MAX_HEIGHT_PT,
                    "maximum_native_width_in": FIGURE_WIDTH_IN, "phase2_max_rows_per_figure": MAX_PHASE2_ROWS,
                    "required_source_font_pt": MIN_SOURCE_FONT_PT, "required_minimum_effective_report_font_pt": MIN_REPORT_FONT_PT,
                    "canvas_policy": "Fixed physical width; no tight-bbox rescaling. Phase2 splits at 13 arms with common x ranges. Each long-run metric/histogram is a separate figure. Fonts are checked after fitting both report width and height.",
                    "figures": FIGURE_LAYOUTS},
                "cpu_only_no_model_imports": True, "torch_imported": "torch" in sys.modules,
                "aggregation": "GEN mean within TRAIN; equally weighted independent TRAIN mean/sample SD; no GEN-as-TRAIN N.",
                "missing_observation_policy": "Skip missing/unpaired cells; no padding, imputation or extrapolation. See each stage's status and missing lists.",
                "warnings": warnings, "figures": figures}
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observations": len(rows), "stages": {key: value["status"] for key, value in figures.items()},
                      "files": [path for stage in figures.values() for path in stage["files"]]}, indent=2))


if __name__ == "__main__":
    main()
