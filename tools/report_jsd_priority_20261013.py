"""Report six-JSD comparisons without silently reusing the old R7 ranking."""
from __future__ import annotations

import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import METRICS, read, save, sha, now

ART = ROOT / "output/reports/wandb_cc_20261013"
OLD = ROOT / "output/reports/init_diffusion_657_compact_20261013/live_report.json"


def scores(ratios):
    return {"R_jsd": statistics.fmean(ratios[k] for k in METRICS[:6]),
            "R_geometry": statistics.fmean(ratios[k] for k in METRICS[:3]),
            "R_attr": statistics.fmean(ratios[k] for k in METRICS[3:6]),
            "improved_jsd_components": sum(ratios[k] < 1 for k in METRICS[:6]),
            "worst_jsd_ratio": max(ratios[k] for k in METRICS[:6])}


def build():
    old = read(OLD)
    historical = []
    for row in old["comparisons"]:
        base = row.get("primary_fresh657", {})
        if not base.get("available", True) or "ratios" not in base:
            continue
        value = {k: row[k] for k in ("source", "variant", "tier", "budget", "num_samples", "TRAIN", "GEN")}
        value.update(scores(base["ratios"]), ratios=base["ratios"],
                     metrics=base["metrics"], control_metrics=base["control_metrics"],
                     previous_R7=base["R"], gates=base["gates"],
                     source_denominator_cells=base.get("denominator_cells", []),
                     posthoc_reranking=True)
        historical.append(value)
    remote = []
    for row in read(ART / "refresh_analysis.json")["comparisons"]:
        remote.append({**row, **scores(row["ratios"])})
    current_file = ART / "jsd_priority/comparisons.json"
    current = read(current_file)["comparisons"] if current_file.exists() else []
    current = [{**row, **scores(row["ratios"])} for row in current]
    obs_path = ART / "jsd_priority/observations.json"
    observations = read(obs_path) if obs_path.exists() else []
    for item in current:
        cells = []
        for row in observations:
            if row["variant"] != item["arm"] or row["budget"] != item["budget"]:
                continue
            control = next(c for c in observations if c["variant"] == "fresh657" and
                           (c["seed"],c["eval_seed"],c["budget"]) == (row["seed"],row["eval_seed"],row["budget"]))
            ratios = {k:row[k]/control[k] for k in METRICS}
            cells.append({"TRAIN":row["seed"],"GEN":row["eval_seed"],"ratios":ratios,**scores(ratios)})
        item["per_GEN_diagnostic"] = cells
        item["all_GEN_R_jsd_below1"] = bool(cells) and all(c["R_jsd"]<1 for c in cells)
    expanded_path = ART / "expanded/result.json"
    expanded = None
    if (ART / "expanded/verified_completion.json").exists():
        assert read(ART / "expanded/verified_completion.json")["result_sha256"] == sha(expanded_path)
        expanded = read(expanded_path)
        expanded.update(scores(expanded["ratios"]))
    report = {"generated_utc": now(), "primary": "R_jsd", "collision_in_ranking": False,
              "definition": "For each metric, equal GEN means within TRAIN, then equal TRAIN means; candidate mean / paired657 mean; average six ratios. Geometry first3; attributes last3. Lower is better; control=1.",
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in [OLD, ART / "refresh_analysis.json"]},
              "historical_posthoc": historical, "remote_observational": remote,
              "new_native_paired": current,
              "expanded_validation": expanded,
              "scope_warnings": [
                  "Do not compare R across differing training budgets or evaluation cohorts.",
                  "Historical reranking is posthoc; new second-TRAIN confirmation is registered separately.",
                  "Remote W&B same-epoch ratios lack remote data hashes and GEN pairing; not a paired replication.",
                  "128-scene remote evaluations are not ranked against256-scene runs.",
                  "Quick scratch training does not establish128-epoch convergence.",
                  "Report coverage and counts because lateral/angular metrics use onroad generated vehicles.",
                  "Collision reported as auxiliary; no collision threshold used to select primary winner.",
              ]}
    late_path = ART / "late_refresh/late_update_summary.json"
    if late_path.exists():
        report["late_remote_update"] = read(late_path)
        report["sources"][str(late_path.relative_to(ROOT))] = sha(late_path)
    final_remote = []
    for run_id in ("torc8p1w", "iege6y14"):
        path = ART / "final_refresh/histories" / f"{run_id}.json"
        if path.exists():
            snapshot = read(path)
            validations = [r for r in snapshot["rows"] if all(r.get(k) is not None for k in METRICS)]
            final_remote.append({"run_id":run_id,"name":snapshot["name"],
                "fetched_utc":snapshot["fetched_at_utc"],"validations":validations,
                "scope":"128-scene scratch K8" if run_id=="torc8p1w" else "256-scene continuation from cc epoch95; not scratch"})
            report["sources"][str(path.relative_to(ROOT))] = sha(path)
    report["final_remote_update"] = final_remote
    direct = []
    for budget in sorted({r["budget"] for r in observations if r["variant"]=="lane_only"}):
        lane = [r for r in observations if r["variant"]=="lane_only" and r["budget"]==budget]
        base = [r for r in observations if r["variant"]=="freshK8" and r["budget"]==budget]
        if lane and {(r["seed"],r["eval_seed"]) for r in lane} == {(r["seed"],r["eval_seed"]) for r in base}:
            ratios = {k:statistics.fmean(r[k] for r in lane)/statistics.fmean(r[k] for r in base) for k in METRICS}
            direct.append({"comparison":"lane_only / freshK8", "budget":budget,
                           "TRAIN":sorted({r["seed"] for r in lane}),"ratios":ratios,**scores(ratios)})
    report["centerline_ablation"] = direct
    save(ART / "jsd_priority_analysis.json", report)
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>JSD-first experiment report</title>",
             "<style>body{font:16px/1.55 system-ui,sans-serif;margin:36px auto;max-width:1450px;padding:0 24px;color:#172338}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:18px 0 30px;font-size:14px}td,th{border:1px solid #d8e0ec;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f9}small{color:#556477}img{max-width:100%}.note{padding:14px;background:#f0f6ff}</style>",
             "<h1>六项 JSD 优先：W&B、历史搜索与本轮复验</h1>",
             f"<small>更新 UTC {report['generated_utc']} · N20 · G1 8,083,978 参数 · 固定 AIRL map encoder</small>",
             "<p class='note'>主指标 R_jsd = 六项 JSD 相对同预算、同场景 657 对照的比值平均；越低越好，对照为 1。Geometry 为 nearest/lat/ang，Attr 为 length/width/speed。Collision 不参与排名。原报告 R7 保留用于解释排名变化，不混用名称。</p>"]

    def table(rows, label):
        parts.append("<h2>" + label + "</h2><table><tr><th>实验与范围</th><th>R_jsd</th><th>R_geometry</th><th>R_attr</th><th>改善项数</th>" +
                     "".join("<th>" + k.replace("_jsd", "") + " ratio</th>" for k in METRICS) + "</tr>")
        for title, row in rows:
            parts.append("<tr><td>" + html.escape(title) + "</td>" + "".join(f"<td>{row[k]:.4f}</td>" for k in ("R_jsd", "R_geometry", "R_attr")) +
                         f"<td>{row['improved_jsd_components']}/6</td>" +
                         "".join(f"<td>{row['ratios'][k]:.4f}</td>" for k in METRICS) + "</tr>")
        parts.append("</table>")

    table([(f"{r['arm']} · step {r['budget']} · TRAIN {r['TRAIN_seeds']}", r) for r in current], "本轮原生配对结果（只纳入完成且验证通过的阶段）")
    if direct:
        table([(f"{r['comparison']} · step{r['budget']} · TRAIN{r['TRAIN']}",r) for r in direct], "中心线输入单项消融（分母是K8，不是657）")
    if not current:
        parts.append("<p>训练仍在进行；尚无完成的候选/对照配对结果。</p>")
    else:
        table([(f"{r['arm']} · step{r['budget']} · TRAIN{c['TRAIN']} / GEN{c['GEN']}",c)
               for r in current for c in r["per_GEN_diagnostic"]], "生成种子敏感性（诊断，不平均这些R替代主评分）")
    if expanded:
        table([(f"{expanded['arm']} · step{expanded['budget']} · {expanded['scenes']} scenes",expanded)], "扩大场景集合复验")
        parts.append("<p>扩展集合保留搜索用4096场景，新增12288场景；它是扩展验证，并非完全独立的未见测试集。对照和候选均重新评价，使用同一个已保存权重与两个GEN。</p>")
    selected = [r for r in historical if r["budget"] in (2048,4096) and r["num_samples"] == 4096 and r["variant"] in ("lane_agentmax05", "a2a_k8", "sincos_context")]
    table([(f"{r['variant']} · step {r['budget']} · TRAIN {r['TRAIN']}", r) for r in sorted(selected, key=lambda r: (r["budget"], r["R_jsd"]))], "历史结果重新计分：相同阶段内比较")
    parts.append("<p>历史 lane_agentmax05 在 step4096 的 R_jsd≈0.6025，K8≈0.8257；原 R7 则分别≈0.8417、0.8162。排名翻转来自 collision 权重退出评分。Lane 组合仍有 width/speed 比 K8 差的情况，不能称为所有分项都最优。历史只含一个 TRAIN，需第二个 TRAIN 验证。</p>")
    longer = [r for r in historical if r["budget"] == 10240 and r["variant"] in ("a2a_k8", "sincos_context")]
    table([(f"{r['variant']} · {r['num_samples']} scenes · TRAIN {r['TRAIN']}", r) for r in longer], "较长预算校准：不把短训排序当作最终排序")
    parts.append("<p>K8 的六项 JSD 比值从历史 step4096 的约0.826，到 step10240 的4096场景约0.940；sincos context 也由约0.820变为约0.972。训练预算增加后早期优势减小。旧657的短训 lateral 误差本来很高，长期训练可明显追上，因此本轮中心线优势首先意味着学习更快，不能直接推断128 epoch同幅改善。</p>")
    table([(r["purpose"] + " · epochs " + str(r["epochs"]), r) for r in remote], "W&B 同 epoch 观察性比较（各自范围内解释）")
    parts.append("<p>cc N20/Map64 在最终 epoch 的 R_jsd≈0.9208，R_attr≈0.9655，angular 比 657 差约13.8%。碰撞从0.658%降到0.024%不能代表六项 JSD 同幅改善。N40 结果不满足本轮约束。历史 cc 的 map_neighbors 字段在合并后对应 map2agent_max_num_neighbors。</p>")
    if "late_remote_update" in report:
        late = report["late_remote_update"]
        parts.append("<h2>后续 W&B 刷新</h2><p>读取时间 UTC " + late["fetched_utc"] + "；通过当前配置 api.wandb.ai 只读读取 u3006612/bc，保留此前快照。</p><table><tr><th>run</th><th>最新训练epoch</th><th>实际最新完整评价epoch</th><th>评价场景 / steps</th></tr>")
        for r in late["runs"]:
            parts.append("<tr><td>" + html.escape(r["name"]) + "</td><td>" + str(r["last_training_epoch"]) + "</td><td>" + str(r["last_complete_validation_epoch"]) + "</td><td>" + str(r["val_scenes"]) + " / " + str(r["sampling_steps"]) + "</td></tr>")
        parts.append("</table><p>新K8与cc_drift在此次快照尚无完整JSD行，两者每8个epoch验证；cc_drift为从epoch95权重开始的finetune，不能作为从头训练的新结果参与本轮排名。</p>")
    if final_remote:
        parts.append("<h2>最终远端增量快照</h2><table><tr><th>run / 范围</th><th>epoch</th>"+"".join("<th>"+k+"</th>" for k in METRICS[:6])+"</tr>")
        for run in final_remote:
            for r in run["validations"]:
                parts.append("<tr><td>"+html.escape(run['name']+' / '+run['scope'])+"</td><td>"+str(r['epoch'])+"</td>"+"".join(f"<td>{r[k]:.5f}</td>" for k in METRICS[:6])+"</tr>")
        parts.append("</table><p>这些远端新行覆盖较早快照尚未完成评价的状态；不与不同场景数的657混算R，不将cc续训当作从头实验。</p>")
    parts.append("<h2>输入与损失的解释</h2><p>road_edges 返回道路边界节点；lanes_and_edges 在相同 GT 地图中额外返回车道中心线（类型0/1/2），没有使用 GT agent 属性。它不增加模型参数，但增加地图 query 数量和计算量，可能更直接支持 lateral/angular 分布。lane_agentmax05 同时改变了 collision reduction（pair_mean→agent_max）及权重（5→0.5），数值权重不能直接解释为总梯度缩小10倍，因此安排 lane-only 消融。</p>")
    parts.append("<p>已核查 AIRL 权重：没有 encoder.init_decoder / G1 权重，不能把657的优势解释为从该文件获得了预训练 diffusion。W&B 的657使用首批次归一化统计与 TRAIN seed2817；本轮使用固定TRAIN8192统计与 TRAIN4817/5817。原线上默认 matmul highest，本轮 medium；短训重复一个已打乱的文件顺序，完整训练每个 epoch 再打乱。这些差异要求分别使用匹配对照，不能把两套实验的绝对值直接拼接。</p>")
    image = ART / "historical_jsd_histograms.png"
    if image.exists():
        parts.append("<h2>历史 step4096 生成分布（同4096场景、两个GEN合并）</h2><img src='historical_jsd_histograms.png' alt='六项生成分布与真实分布'>")
        parts.append("<p>历史 lane 组合仍低估属性尾部：length≥6 m 的占比约2.46%（GT2.89%），width≥2.5 m 约2.74%（GT3.33%），speed≥15 m/s 约8.21%（GT9.30%）。0–1 m/s 比例约42.77%（GT41.84%），所以这个检查点的问题不是缺少低速车辆。以上为原生直方图阈值诊断，不能替代完整JSD。当前 collision 几何中 length/width 已 detach，没有直接收缩尺寸的梯度。</p>")
    parts.append("<h2>证据范围</h2><ul>" + "".join("<li>" + html.escape(w) + "</li>" for w in report["scope_warnings"]) + "</ul><p><a href='jsd_priority_analysis.json'>结构化结果与来源</a> · <a href='jsd_priority/protocol.json'>本轮实验注册</a> · <a href='refresh_analysis.html'>W&B 原始刷新分析</a></p></html>")
    (ART / "jsd_priority_analysis.html").write_text("\n".join(parts))
    print(json.dumps({"report": str(ART / "jsd_priority_analysis.html"), "new_comparisons": len(current)}))


if __name__ == "__main__":
    build()
