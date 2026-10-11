"""Render the completed paired scratch heldout stage as a Chinese report section.

CPU-only: reads actual analyzer observations and the frozen paired decision;
never constructs models, trains, evaluates, or changes the selected recipe.
Both budgets must contain the same three training seeds and two generation
seeds for both finalists and weight modes. Optional PNGs use the existing
plot_scratch_loss_report.py, with Old recipe / New recipe labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
sys.path.insert(0, str(ROOT))

from tools.materialize_scratch_best_config import validate_decision
from tools.summarize_paired_scratch_finalists import (
    MULTIPLIERS, SCALES, VALUES, build_summary, make_cells, percent,
    read_protocol, select_rows, stats, training_seed_means,
)

DISPLAY = {
    "nearest_dist_jsd": "Nearest JSD", "lat_dev_jsd": "Lateral JSD",
    "ang_dev_jsd": "Angle JSD", "length_jsd": "Length JSD",
    "width_jsd": "Width JSD", "speed_jsd": "Speed JSD",
    "collision_rate": "Collision (%)", "score": "综合 S",
    "generated_onroad_fraction": "On-road (%)", "vehicle_count_ratio": "Vehicle / GT",
    "num_generated_vehicles": "生成车辆数", "num_gt_vehicles": "GT 车辆数",
}
REPORTED = (*SCALES, "score", "generated_onroad_fraction", "vehicle_count_ratio")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def divisor(key):
    return .01 if key == "generated_onroad_fraction" else MULTIPLIERS.get(key, 1)


def value_text(value, key, digits=None):
    digits = (2 if key == "generated_onroad_fraction" else 4 if key in ("score", "vehicle_count_ratio", "collision_rate") else 5) if digits is None else digits
    factor = divisor(key)
    return f"{value['mean'] / factor:.{digits}f} ± {value['sd'] / factor:.{digits}f}"


def delta_text(value, key):
    factor = divisor(key)
    digits = 2 if key == "generated_onroad_fraction" else 4 if key in ("score", "collision_rate", "vehicle_count_ratio") else 5
    return f"{value['mean'] / factor:+.{digits}f} ± {value['sd'] / factor:.{digits}f}"


def build_trends(rows, protocol, baseline, candidate):
    """Actual paired changes across exactly 512 optimizer updates, never forecasts."""
    variants = (baseline, candidate)
    result = {"budgets": [1536, 2048], "update_difference": 512,
              "training_seeds": protocol["training_seeds"],
              "generation_seeds": protocol["generation_seeds"],
              "scope": "Same training-seed checkpoints and generation seeds at both actual budgets",
              "weight_modes": {}, "warnings": []}
    for weights in ("ema", "online"):
        by_budget = {}
        for budget in (1536, 2048):
            cells, missing, cell_count, observation_count = make_cells(rows, budget, weights, variants, protocol, result["warnings"])
            if missing:
                raise ValueError(f"Incomplete actual {weights}{budget} bridge matrix: {missing}")
            by_budget[budget] = {"cells": cells, "training_seed_means": training_seed_means(cells, variants, protocol),
                                 "generation_cells": cell_count, "observation_rows": observation_count}
        arms = {}
        for variant in variants:
            entries = {}
            for key in VALUES:
                initial = by_budget[1536]["training_seed_means"][variant]
                final = by_budget[2048]["training_seed_means"][variant]
                deltas = [final[seed][key] - initial[seed][key] for seed in protocol["training_seeds"]]
                relative = [percent(final[seed][key], initial[seed][key]) for seed in protocol["training_seeds"]]
                direction = "higher" if key == "generated_onroad_fraction" else "lower"
                entries[key] = {"at1536_native": stats(initial[seed][key] for seed in protocol["training_seeds"]),
                                "at2048_native": stats(final[seed][key] for seed in protocol["training_seeds"]),
                                "delta_native": stats(deltas),
                                "slope_native_per_optimizer_update": stats(delta / 512 for delta in deltas),
                                "paired_percent_change": stats(relative),
                                "lower_count": sum(delta < 0 for delta in deltas),
                                "higher_count": sum(delta > 0 for delta in deltas),
                                "equal_count": sum(delta == 0 for delta in deltas),
                                "preferred_direction": direction,
                                "per_training_seed": [{"seed": seed, "at1536": initial[seed][key],
                                                       "at2048": final[seed][key],
                                                       "delta": final[seed][key] - initial[seed][key]}
                                                      for seed in protocol["training_seeds"]]}
            arms[variant] = entries
        result["weight_modes"][weights] = {"arms": arms, "actual_budgets": {
            str(budget): {"generation_cells": point["generation_cells"], "observation_rows": point["observation_rows"]}
            for budget, point in by_budget.items()}}
    return result


def per_seed_score(mode, baseline, candidate):
    lines = ["| TRAIN seed | Old recipe：S 均值 [GEN 最小, 最大] | New recipe：S 均值 [GEN 最小, 最大] | 配对 ΔS |",
             "|---:|---:|---:|---:|"]
    for item in mode["per_training_seed"]:
        base = item["baseline"]["score"]
        chosen = item["candidate"]["score"]
        ranges = item["generation_ranges"]["score"]
        lines.append(f"| {item['seed']} | {base:.4f} [{ranges['baseline']['minimum']:.4f}, {ranges['baseline']['maximum']:.4f}] | "
                     f"{chosen:.4f} [{ranges['candidate']['minimum']:.4f}, {ranges['candidate']['maximum']:.4f}] | {chosen-base:+.4f} |")
    return lines


def metric_table(mode, baseline, candidate, paired):
    lines = ["| 指标 | Old recipe：均值 ± TRAIN SD | New recipe：均值 ± TRAIN SD |" + (" New−Old：配对变化 | 改善 TRAIN seeds |" if paired else ""),
             "|---|---:|---:|" + ("---:|---:|" if paired else "")]
    for key in REPORTED:
        base = mode["arms"][baseline]["statistics_native"][key]
        chosen = mode["arms"][candidate]["statistics_native"][key]
        line = f"| {DISPLAY[key]} | {value_text(base, key)} | {value_text(chosen, key)} |"
        if paired:
            comparison = mode["paired_comparisons"][key]
            if key == "generated_onroad_fraction":
                change = comparison["absolute_delta_percentage_points"]
                text = f"{change['mean']:+.2f} ± {change['sd']:.2f} pp"
                count = comparison["candidate_higher_count"]
            else:
                change = comparison["paired_percent_change"]
                text = f"{change['mean']:+.2f} ± {change['sd']:.2f}%" if change else "控制分母为0，比例未定义"
                count = comparison["candidate_lower_count"]
            if key == "vehicle_count_ratio":
                # A smaller generated/GT ratio is not universally preferable.
                line += f" {text} | 下降 {count}/3，需接近1 |"
            else:
                line += f" {text} | {count}/3 |"
        lines.append(line)
    return lines


def render(summary, trends, figures):
    baseline, candidate = summary["baseline"], summary["candidate"]
    selected = summary["selected_variant"]
    primary = summary["modes"]["ema"]
    score_comparison = primary["paired_comparisons"]["score"]
    score_change = score_comparison["paired_percent_change"]
    selected_name = "Old recipe" if selected == baseline else "New recipe"
    lines = ["## 从头训练：batch512、三训练 seed 的 heldout4096 确认", "",
             f"冻结 EMA2048 终点选择 **{selected_name}（`{selected}`）**。"
             f"New recipe 相对 Old recipe 的配对 S 变化为 **{score_change['mean']:+.2f}% ± {score_change['sd']:.2f}%**，"
             f"S 在 {score_comparison['candidate_lower_count']}/3 个训练 seed 上下降。该结论来自实际从头训练和生成结果；未完成128 epochs，不能据此证明最终收敛最优。", "",
             f"Old recipe=`{baseline}`，New recipe=`{candidate}`。两者 map encoder 与 G1 都从随机参数训练，"
             "batch512、LR5e-4、默认 AdamW 和128真实epoch余弦调度、32-true、gradient clip0.5保持一致。"
             "2048 optimizer updates 对应实际1,048,524个训练样本、约2.124个full TRAIN epoch；这里使用G1 EMA和online共享map。", "",
             "训练 seeds 为817/1817/2817，生成 seeds 为2817/3817。先在每个训练 seed 内平均两次生成，再在3个独立训练seed之间等权平均；"
             "表中SD是训练seed样本SD（ddof=1），不是置信区间，6次生成不等于6个独立训练实验。", "",
             "下面六项JSD都是未缩放的自然对数JSD，理论上界ln2≈0.693147；helper中的nearest/lateral除10，其余JSD除100。"
             "collision保留百分比。S维持冻结的helper单位scale[1,1,10,20,15,20,5]，七项metric/scale后取均值，属于本次搜索的辅助排序分数。", "",
             "### 冻结 EMA2048 主结果及配对一致性", ""]
    lines += metric_table(primary, baseline, candidate, paired=True)
    lines += ["", "配对百分比先对每个训练seed的两次生成均值计算New/Old−1，再汇总3个seed的均值与样本SD；"
              "这不同于直接计算两臂总体均值的比例。On-road用百分点，V/GT应接近1；其下降不自动代表改善。", "",
              "Lateral/angle只统计on-road车辆，因此应连同on-road覆盖一起看，较低条件JSD不能补偿大量离路车辆。"
              "指标表完整保留尺寸、速度、碰撞和覆盖的取舍，没有将综合分数最低解释成所有指标最佳。", "",
              "### 各训练 seed 的实际生成波动", ""]
    lines += per_seed_score(primary, baseline, candidate)
    lines += ["", "方括号是同一checkpoint的两个生成seed范围，仅描述sampler波动；不同训练seed才是上表SD的样本单位。", "",
              "### 同一批训练 seed 的实际1536→2048变化", "",
              "两预算对每个配方均使用相同3个TRAIN seeds及相同2个GEN seeds。以下Δ=2048−1536，"
              "先在训练seed内配对，再汇总；它是512次真实optimizer更新的净变化。逐更新斜率Δ/512和逐seed原值保存在`heldout_stage_details.json`，"
              "不将短期斜率线性外推到128 epochs。", "",
              "| 指标 | Old recipe：实际Δ均值 ± TRAIN SD | New recipe：实际Δ均值 ± TRAIN SD | Old 改善seeds | New 改善seeds |",
              "|---|---:|---:|---:|---:|"]
    arms = trends["weight_modes"]["ema"]["arms"]
    for key in REPORTED:
        base, chosen = arms[baseline][key], arms[candidate][key]
        if key == "vehicle_count_ratio":
            before = after = "趋近1另看原值"
        else:
            count_key = "higher_count" if key == "generated_onroad_fraction" else "lower_count"
            before, after = f"{base[count_key]}/3", f"{chosen[count_key]}/3"
        lines.append(f"| {DISPLAY[key]} | {delta_text(base['delta_native'], key)} | {delta_text(chosen['delta_native'], key)} | {before} | {after} |")
    lines += ["", "JSD/碰撞/S的负Δ是改善，on-road的正Δ是覆盖增加；碰撞和on-road的Δ单位都是百分点。尺寸、速度和几何的短期趋势可能不同；"
              "相同seed的后期配对比混合不同训练seed的早晚总体均值更可核对。", "",
              "### Online2048：仅作诊断", "",
              "Online在相同训练、生成和场景矩阵上评价，只检查权重平均影响与排名稳定性。"
              "已冻结的部署终点为EMA2048，不因online表现切换权重范围、预算或选择规则。", ""]
    lines += metric_table(summary["modes"]["online"], baseline, candidate, paired=False)
    lines += ["", "### 完整门槛与复核记录", "",
              "部署选择按三训练seed的EMA2048均值，要求on-road不低于Old recipe−2个百分点，且生成/GT车辆比例在[0.9,1.1]，"
              "在合格的两个finalists中取最低S；该规则在heldout生成前冻结。", "",
              "| Recipe | 门槛合格 | Mean on-road (%) | 门槛最低on-road (%) | Mean V/GT | Mean S |",
              "|---|---|---:|---:|---:|---:|"]
    for variant, label in ((baseline, "Old recipe"), (candidate, "New recipe")):
        gate = summary["deployment_selection"]["eligibility"][variant]
        lines.append(f"| {label} | {gate['eligible']} | {100*gate['mean_onroad_fraction']:.2f} | "
                     f"{100*gate['minimum_onroad_fraction']:.2f} | {gate['mean_vehicle_count_ratio']:.4f} | {gate['mean_score']:.4f} |")
    lines += ["", "heldout场景未用于本次TRAIN或此前loss搜索；这里将其用于最终配方选择后，它就是选择验证集，不能再称为完全未参与选择的test。"
              "本节只报告实际4096场景确认，不填充未发生的全量评价或完整训练结果。", "",
              "[实际paired统计](paired_summary.json)、[逐训练seed表](per_training_seed.csv)、"
              "[实际阶段趋势及provenance](heldout_stage_details.json)保存原始单位、seed身份、checkpoint和数据清单hash。", ""]
    if figures:
        lines += ["### 同seed后期曲线", "", "图中误差线/带表示3个独立训练seed的样本SD，Old/New标签与上表一致。", ""]
        for figure, mode in figures:
            lines += [f"![实际1536与2048：{mode}，三训练seed，JSD单位为nats](compact_plots/{figure})", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-summary", type=Path, default=ARTIFACTS / "paired_finalists/paired_summary.json")
    parser.add_argument("--output", type=Path, help="Defaults to the paired summary directory")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    summary = read(args.paired_summary)
    validate_decision(summary)
    provenance = summary["provenance"]
    observations_path, protocol_path = Path(provenance["observations"]), Path(provenance["protocol"])
    if sha(observations_path) != provenance["observations_sha256"] or sha(protocol_path) != provenance["protocol_sha256"]:
        raise ValueError("The paired decision's actual observations/protocol were modified")
    observations = read(observations_path)
    protocol, gates = read_protocol(protocol_path)
    rebuilt = build_summary(observations, protocol, gates, summary["candidate"], summary["baseline"])
    if rebuilt["modes"] != summary["modes"] or rebuilt["selected_variant"] != summary["selected_variant"]:
        raise ValueError("Saved paired decision differs from recomputed actual observations")
    selected, exclusions, _ = select_rows(observations, (summary["baseline"], summary["candidate"]), protocol, 4096, 512, 5e-4)
    selected = [row for row in selected if row["budget"] in (1536, 2048)]
    trends = build_trends(selected, protocol, summary["baseline"], summary["candidate"])
    output = (args.output or args.paired_summary.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = {summary["baseline"]: "Old recipe", summary["candidate"]: "New recipe"}
    labels_path = output / "compact_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    plot_input = output / "matched_budget_observations.json"
    plot_input.write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
    figures = []
    if not args.skip_plots:
        subprocess.run([sys.executable, str(ROOT / "tools/plot_scratch_loss_report.py"),
                        "--aggregate", str(plot_input), "--variants", ",".join(labels),
                        "--labels-json", str(labels_path), "--jsd-units", "raw",
                        "--output", str(output / "compact_plots")], check=True, cwd=ROOT)
        plot_protocol = read(output / "compact_plots/plot_protocol.json")
        for group in plot_protocol["cohorts"]:
            figures.extend((figure, group["evaluation_weights"])
                           for figure in plot_protocol["figures"] if group["cohort_sha256"][:12] in figure)
    details = {"selected_variant": summary["selected_variant"], "primary_endpoint": "EMA2048",
               "baseline": summary["baseline"], "candidate": summary["candidate"],
               "actual_observation_rows": len(selected), "expected_unique_generation_cells": 48,
               "actual_unique_generation_cells": len({(row["variant"], row["budget"], row["seed"],
                                                        row["eval_seed"], row["evaluation_weights"])
                                                       for row in selected}),
               "exclusions": exclusions, "trends": trends,
               "primary_statistics": summary["modes"]["ema"], "online_diagnostics": summary["modes"]["online"],
               "provenance": {"paired_summary": str(args.paired_summary.resolve()), "paired_summary_sha256": sha(args.paired_summary),
                              **provenance, "section_script_sha256": sha(Path(__file__))},
               "units": {"raw_jsd": "natural-log nats, upper bound ln2", "collision": "percent",
                         "onroad_delta": "percentage points", "score_native_scales": SCALES},
               "uncertainty": "Sample SD across 3 independent TRAIN seeds, after averaging 2 GEN repeats; not CI",
               "scope": "Completed actual heldout4096 confirmation and matched late-budget changes; no full-training extrapolation"}
    (output / "heldout_stage_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "heldout_stage_zh.md").write_text(render(summary, trends, figures) + "\n")
    print(json.dumps({"section": str(output / "heldout_stage_zh.md"), "actual_observations": len(selected),
                      "selected_variant": summary["selected_variant"], "figures": len(figures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
