"""Render the JSD-first report from verified JSON; performs no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/tmp/sim_pdf_report_deps")
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from tools.build_scratch_loss_search_pdf import register_fonts, markup

ART = ROOT / "output/reports/wandb_cc_20261013"
WIDTH = A4[0] - 76
BLUE = colors.HexColor("#174d6b")
INK = colors.HexColor("#263342")
METRICS = ["nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd", "width_jsd", "speed_jsd"]
SHORT = ["Near", "Lat", "Ang", "Length", "Width", "Speed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    data = json.loads((ART / "jsd_priority_analysis.json").read_text())
    register_fonts()
    styles = {
        "body": ParagraphStyle("body", fontName="Latin", fontSize=9.4, leading=15, textColor=INK, spaceAfter=8, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName="Latin", fontSize=7.5, leading=11.5, textColor=INK, spaceAfter=5, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName="LatinBold", fontSize=21, leading=29, textColor=BLUE, spaceAfter=12, wordWrap="CJK"),
        "section": ParagraphStyle("section", fontName="LatinBold", fontSize=13, leading=20, textColor=BLUE, spaceBefore=11, spaceAfter=8, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName="Latin", fontSize=7.2, leading=10, textColor=INK, wordWrap="CJK"),
    }
    story = []
    def p(text, kind="body"):
        return Paragraph(markup(text), styles[kind])
    def add(text, kind="body"):
        story.append(p(text, kind))
    def table(headers, rows, widths):
        values = [[p(str(x), "cell") for x in row] for row in [headers] + rows]
        tab = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf2f7")),
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#ccd8e1")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.extend([tab, Spacer(1,9)])
    def num(x):
        return f"{x:.4f}"
    phase = "本轮完成结果" if args.final else "实验进行中 / DRAFT"
    add("InitDiffusion：JSD 优先探索", "title")
    add("2026-10-13 · " + phase + " · " + data["generated_utc"], "small")
    add("**目标：改善六项 JSD，collision rate 只作辅助观察。** 采样保持 N20，生成器 8,083,978 参数，使用 sim 环境和固定 AIRL map encoder。")
    add("评分口径", "section")
    add("每项指标先在同一个 TRAIN 内平均两个 GEN，再对 TRAIN 等权平均；候选均值除以同预算、同场景的657对照均值。R_jsd 为六个比值的平均，越低越好，control=1。R_geometry 为 Near/Lat/Ang；R_attr 为 Length/Width/Speed。旧 R7 含 collision，保留旧定义，不用于本轮排序。")
    add("本轮原生配对结果", "section")
    current = data["new_native_paired"]
    if current:
        table(["候选 / TRAIN / step", "R_jsd", "R_geom", "R_attr", "改善项", "检查"],
              [[f"{r['arm']} / {','.join(map(str,r['TRAIN_seeds']))} / {r['budget']}",num(r['R_jsd']),num(r['R_geometry']),num(r['R_attr']),f"{r['improved_jsd_components']}/6","PASS" if r['gates_pass'] else "见分项"] for r in current],
              [WIDTH*.37,WIDTH*.12,WIDTH*.12,WIDTH*.12,WIDTH*.12,WIDTH*.15])
        table(["候选 / step"] + SHORT,
              [[f"{r['arm']} / {r['budget']}"]+[num(r['ratios'][k]) for k in METRICS] for r in current],
              [WIDTH*.32]+[WIDTH*.68/6]*6)
        add("上表六项为相对对照比值，不是JSD绝对值。检查包括属性各项不退步超过15%、onroad比例不下降超过2个百分点、车辆数量接近GT；collision不参与检查。不同TRAIN/step行不直接按数值跨范围排名。", "small")
        qualified = [r for r in current if r['gates_pass'] and r['R_jsd'] < 1]
        if qualified:
            endpoint = min(qualified, key=lambda r:(-r['budget'],r['R_jsd']))
            add(f"较高预算复验：{endpoint['arm']}，TRAIN {endpoint['TRAIN_seeds']}，step {endpoint['budget']}。", "small")
            table(["原生JSD绝对值"] + SHORT,
                  [["配对657"]+[num(endpoint['control_metrics'][k]) for k in METRICS],
                   ["候选"]+[num(endpoint['metrics'][k]) for k in METRICS]],
                  [WIDTH*.25]+[WIDTH*.75/6]*6)
            values = [r['R_jsd'] for r in endpoint.get('per_GEN_diagnostic',[])]
            if values:
                add("两个GEN的R_jsd分别为"+" / ".join(num(v) for v in values)+"；原生GT直方图与对照完全相同。分项退步仍需保留，不能称为全部指标最优。", "small")
            add(f"辅助记录：collision rate候选{endpoint['metrics']['collision_rate']:.3f}%、配对657为{endpoint['control_metrics']['collision_rate']:.3f}%；它不参与R_jsd排序。", "small")
    else:
        add("候选训练尚未完成，当前不报告候选质量结论。已完成的原生对照与未完成阶段分别保存；不把中断输出纳入排名。")
    expanded = data.get('expanded_validation')
    if expanded:
        add(f"扩大到{expanded['scenes']}场景：{expanded['arm']}的R_jsd={expanded['R_jsd']:.4f}，R_attr={expanded['R_attr']:.4f}。保留原4096并新增12288场景，属于扩展验证，不是独立未见测试集。", "small")
        add("扩展验证六项比值（Near/Lat/Ang/Length/Width/Speed）："+" / ".join(num(expanded['ratios'][k]) for k in METRICS)+"；speed仍退步约9.5%。", "small")
    add("本轮最多训练约4.25个epoch；短训改善尚不证明128 epoch或50,000场景最终优势。", "small")

    story.append(PageBreak())
    add("已有结果：改用 JSD 后的判断", "title")
    add("历史搜索重新计分", "section")
    selected = [r for r in data["historical_posthoc"] if r["budget"]==4096 and r["num_samples"]==4096 and r["variant"] in ("lane_agentmax05","a2a_k8","sincos_context")]
    table(["历史候选 / TRAIN4817", "R_jsd", "R_attr", "旧R7", "Collision ratio"],
          [[r['variant'],num(r['R_jsd']),num(r['R_attr']),num(r['previous_R7']),num(r['ratios']['collision_rate'])] for r in sorted(selected,key=lambda r:r['R_jsd'])],
          [WIDTH*.36]+[WIDTH*.16]*4)
    add("lane_agentmax05 的 R_jsd≈0.6025，K8≈0.8257；但旧R7按碰撞一起计分时，K8略优。这是本轮转向中心线输入复验的原因。Lane候选上路车辆比例约92.6%，657约85.3%，低JSD没有伴随有效评价车辆比例下降。历史重排是事后分析，只用于提出假设。")
    add("W&B 与 cc 分支", "section")
    remote = [r for r in data["remote_observational"] if r['epochs']==[127] and r['candidate'] in ('4bo59qfn','c486qs4q')]
    table(["同epoch127观察", "R_jsd", "R_geom", "R_attr", "旧R7"],
          [["cc N20 Map64" if r['candidate']=='4bo59qfn' else "LR5e-4 / floor.01",num(r['R_jsd']),num(r['R_geometry']),num(r['R_attr']),num(r['observed_R7'])] for r in remote],
          [WIDTH*.36]+[WIDTH*.16]*4)
    add("cc N20的angular JSD比657高约13.8%；碰撞从0.658%降到0.024%，不能视为JSD同幅改善。cc的N40结果不满足本轮20步约束。新norm/time与depth5只验证128场景，旧657/cc为256场景，不跨这些范围排榜。远端数据哈希与GEN配对未验证，W&B表属于同epoch观察性比较。")
    for remote_run in data.get('final_remote_update',[]):
        if remote_run['validations']:
            r = remote_run['validations'][-1]
            add(f"最终增量快照：{remote_run['run_id']}，epoch {r['epoch']}，{remote_run['scope']}。六项绝对JSD（Near/Lat/Ang/Length/Width/Speed）："+" / ".join(num(r[k]) for k in METRICS)+"。不跨评价场景数或续训/从头范围排名。", "small")
    add("为什么短训优势没有稳定延续", "section")
    add("历史K8的R_jsd从step4096约0.826变为step10240、4096场景约0.940；sincos也从约0.820变为约0.972。早期对照尚未学好地图约束，一些改动主要加速学习，未必改善最终分布。线上还出现训练width/speed loss下降而生成JSD回升，说明不能只根据重建loss筛选。")

    story.append(PageBreak())
    add("分布诊断与下一步方向", "title")
    image = ART / "historical_jsd_histograms.png"
    story.append(Image(str(image), width=WIDTH, height=WIDTH*1280/2400))
    story.append(Spacer(1, 7))
    add("历史step4096、TRAIN4817、4096场景；两个GEN的直方图计数合并，仅作分布诊断。JSD排名仍使用每个GEN先单独计算指标再平均的口径。", "small")
    add("中心线输入改善几何约束", "section")
    add("road_edges只返回道路边界query；lanes_and_edges额外返回车道中心线0/1/2类型。两者来自相同GT地图，没有增加GT agent属性，也没有增加网络参数，但地图节点和计算量可能增加。lane-only实验保留原collision设置，以分离输入作用。")
    for r in data.get('centerline_ablation',[]):
        add(f"新单项消融：同TRAIN {r['TRAIN']}、step {r['budget']}，lane-only相对K8的R_jsd={r['R_jsd']:.4f}，R_attr={r['R_attr']:.4f}。六项比值："+" / ".join(num(r['ratios'][k]) for k in METRICS)+"。这里分母是K8，不是657。", "small")
    add("属性主要缺少尾部，不是缺少低速车辆", "section")
    table(["历史lane候选的分布区域", "生成比例", "GT比例"],
          [["length >= 6 m","2.46%","2.89%"],["width >= 2.5 m","2.74%","3.33%"],["speed >= 15 m/s","8.21%","9.30%"],["0 <= speed < 1 m/s","42.77%","41.84%"]],
          [WIDTH*.54,WIDTH*.23,WIDTH*.23])
    add("均值接近不能保证JSD低，峰值比例和尾部仍重要。当前collision几何已经detach length/width，没有直接收缩尺寸的梯度；共享网络仍可能产生间接影响。agent_max与pair_mean的缩放不同，权重0.5和5不能直接解释为惩罚强度差十倍。")

    story.append(PageBreak())
    add("复现范围与实验记录", "title")
    add("保持一致的条件", "section")
    table(["项目", "设置"], [
        ["训练数据", "training_map2_sd；新鲜G1/optimizer/EMA"],
        ["评价数据", "scenario_dreamer_val；固定4096清单；两个GEN"],
        ["模型与采样", "3层、hidden256、N20、8,083,978参数；collision guidance OFF"],
        ["输入条件", "固定ego位置/heading/shape/type；生成ego速度与其他agent类型"],
        ["地图", "AIRL预训练权重冻结；ego-local方形crop，半边长50 m"],
        ["优化器", "AdamW 5e-4；epoch cosine128；min ratio.05；clip.5"],
        ["统计与精度", "TRAIN8192固定normalizer；32-true；matmul medium"],
    ], [WIDTH*.27,WIDTH*.73])
    add("与线上完整训练的差异", "section")
    add("B512，训练集493,542场景。2048/4096 updates实际见到1,048,524/2,097,048场景，约2.12/4.25个epoch。评价使用GEN2817/3817和EMA。", "small")
    add("已检查AIRL checkpoint：没有InitDiffusion/G1权重，只有可复用的map等模块。线上657使用首批次normalizer、TRAIN2817、默认matmul highest；本轮使用固定TRAIN统计与TRAIN4817/5817。快速实验重复已打乱的顺序，完整训练每epoch重新打乱。不能跨这些设置拼接绝对指标或宣称等价最终性能。")
    add("完整训练配置", "section")
    add("`python src/run.py experiment=init_diffusion_657_n20_lane_agentmax05 optimizer=default`", "small")
    add("对照使用 `experiment=init_diffusion_657_n20_jsd_control`；单项消融使用 `experiment=init_diffusion_657_n20_lane_only`。每8epoch监控排序后的前4096场景，num_workers=8，保留定期checkpoint，不按collision选优。监控集合不同于快速搜索清单；完整eval评价50,000场景。", "small")
    add("输出R_jsd/R_attr使用已配置的657固定50k参考。4096场景验证的R仅作固定基准趋势，R_scope_matches_reference=0时不是本文配对分数；测试也需检查参考预算。本文搜索评分始终使用同预算、同场景的新对照。", "small")
    add("可审查的记录", "section")
    for label, path in [
        ("主报告与完整数值", "output/reports/wandb_cc_20261013/jsd_priority_analysis.json"),
        ("本轮注册与源码哈希", "output/reports/wandb_cc_20261013/jsd_priority/protocol.json"),
        ("本轮原生观测", "output/reports/wandb_cc_20261013/jsd_priority/observations.json"),
        ("W&B刷新与限制", "output/reports/wandb_cc_20261013/refresh_analysis.json"),
        ("直方图重算校验", "output/reports/wandb_cc_20261013/historical_histogram_diagnostics.json"),
    ]:
        add(label+"：`"+path+"`", "small")
    add("W&B工作区：[u3006612/bc](https://forge.coreweave.com/wandb/u3006612/bc/workspace)。每个完成阶段均检查原生完成标记、训练更新数、EMA更新数、固定map、源代码和数据清单哈希。36个历史直方图JSD重算与原生输出一致。", "small")
    add("结论边界", "section")
    add("按六项JSD选择候选，逐项披露退步；collision仅作附带观察。尚未完成的新阶段不会计分。最终推荐仍需更长预算、多TRAIN以及完整50,000场景确认，不把单个短训最小值当作已实现的完整训练提升。")

    out = ROOT / "output/pdf/init_diffusion_jsd_priority_20261013.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d5e0e7"))
        canvas.line(38,34,A4[0]-38,34)
        canvas.setFont("Latin",7)
        canvas.setFillColor(INK)
        canvas.drawString(38,22,"JSD-first / N20 / 2026-10-13" + (" / FINAL SNAPSHOT" if args.final else " / DRAFT"))
        canvas.drawRightString(A4[0]-38,22,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=34,bottomMargin=45,
                      title="InitDiffusion JSD-first experiments",author="Experiment analysis").build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({"pdf":str(out),"final_snapshot":args.final,"completed_comparisons":len(current)}))


if __name__ == "__main__":
    main()
