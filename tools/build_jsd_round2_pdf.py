"""Render verified round2 observations; no training or metric estimation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,"/tmp/sim_pdf_report_deps")
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from tools.build_scratch_loss_search_pdf import register_fonts,markup
ART=ROOT/"output/reports/init_diffusion_jsd_round2_20261013"
KEYS=['nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd']
SHORT=['Near','Lat','Ang','Length','Width','Speed']
W=A4[0]-76


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    data=json.loads((ART/'analysis.json').read_text());plan=json.loads((ART/'protocol.json').read_text())
    register_fonts();story=[]
    styles={k:ParagraphStyle(k,fontName='LatinBold' if k in ('title','section') else 'Latin',fontSize=size,
        leading=leading,textColor=colors.HexColor('#24364b'),spaceAfter=gap,wordWrap='CJK')
        for k,size,leading,gap in [('title',20,28,14),('section',13,19,9),('body',9.5,15,8),('small',7.5,11.5,5),('cell',7.2,10,0)]}
    def p(text,kind='body'):return Paragraph(markup(text),styles[kind])
    def add(text,kind='body'):story.append(p(text,kind))
    def table(headers,rows,widths):
        t=Table([[p(str(x),'cell') for x in row] for row in [headers]+rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf2f8')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#cad6e2')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([t,Spacer(1,9)])
    fmt=lambda x:f'{x:.4f}'
    add('InitDiffusion：JSD 第二轮探索','title')
    add('2026-10-13 · '+('完成快照' if args.final else '进行中 / DRAFT')+' · UTC '+data['updated_utc'],'small')
    add('采样N20，生成器8,083,978参数，固定AIRL map encoder。排名使用六项JSD，collision作辅助记录。新训练均从随机G1、全新optimizer与EMA开始。')
    add('实验设计与自适应调整','section')
    table(['新实验','改动与目的'],[
        ['lane-only / TRAIN5817 / 4096','K8 + 中心线；保留pair_mean collision、weight5。检查agent_max/.5组合是否造成属性退步。'],
        ['lane_attr_ot / TRAIN5817 / 4096','原lane_agentmax05 + sorted marginal MSE：speed .25、length .1、width .1；t>=.5，按GT类型分组，除以TRAIN std。'],
        ['适应性复验 / TRAIN4817 / 6826','从上述两项和原lane_agentmax05中，按TRAIN5817 step4096的配对R_jsd与门槛选一项；从头训练，在4096/6826评价。'],
    ],[W*.33,W*.67])
    if data.get('adaptive_amendment'):
        add('联合属性损失在完成4096评价后，speed改善但width明显恶化。较长预算任务停止，未完成输出排除；剩余时间改为speed-only（shape分布权重为0），再对照按GT类型分组与整体匹配，原15:00截止时间保持。','small')
    add('分数定义：每项先平均两个GEN，再除以同TRAIN/预算/场景的657均值。R_jsd平均六项比值，R_attr平均Length/Width/Speed；control=1，越低越好。不同预算和场景集合分别解释。属性单项退步超过15%、onroad下降超过2个百分点或车辆数偏离GT超过10%时不晋级。','small')
    rows=[r for r in data['comparisons'] if r['TRAIN_seeds']==[5817] and r['budget']==4096 and r['arm'] in ('lane_only','lane_attr_ot','lane_agentmax05','speed_only')]
    add('同TRAIN5817、step4096配对结果','section')
    if rows:
        table(['候选','R_jsd','R_attr']+SHORT,
            [[r['arm'],fmt(r['R_jsd']),fmt(r['R_attr'])]+[fmt(r['ratios'][k]) for k in KEYS] for r in rows],
            [W*.23]+[W*.77/8]*8)
    else:add('尚无完成的新TRAIN5817/4096评价，不能从训练loss推断生成质量。')
    for r in data['direct_ablations']:
        if r['budget']==4096:
            add(f"相对原lane_agentmax05：{r['arm']}，R_jsd={fmt(r['R_jsd'])}，R_attr={fmt(r['R_attr'])}；Speed比值={fmt(r['ratios']['speed_jsd'])}。这里分母是原lane方案。",'small')
    pooled=[r for r in data['comparisons'] if r['arm']=='speed_pooled' and r['budget']==2048]
    for r in pooled:
        add(f"整体speed匹配只完成2048预算：R_jsd={fmt(r['R_jsd'])}，R_attr={fmt(r['R_attr'])}，speed ratio={fmt(r['ratios']['speed_jsd'])}。与同2048阶段的分组匹配比较，不跨预算排名。",'small')
    add('分布损失属于训练启发式：匹配不同噪声时间下的x0预测边缘分布，不等价于直接优化最终采样分布的JSD，效果以原生生成评价为准。','small')

    story.append(PageBreak());add('较长预算、诊断与证据范围','title')
    long=[r for r in data['comparisons'] if r['TRAIN_seeds']==[4817] and r['budget']==6826]
    add('TRAIN4817、step6826','section')
    if long:
        for r in long:
            add(f"{r['arm']}：R_jsd={fmt(r['R_jsd'])}，R_attr={fmt(r['R_attr'])}；检查{'通过' if r['gates_pass'] else '未通过'}。")
            table(['指标']+SHORT,[['候选']+[fmt(r['metrics'][k]) for k in KEYS],['配对657']+[fmt(r['control_metrics'][k]) for k in KEYS],['比值']+[fmt(r['ratios'][k]) for k in KEYS]],[W*.22]+[W*.78/6]*6)
    else:add('较长预算任务未完成；根据联合损失的失败结果调整了实验方向，其部分训练输出未计分。')
    expanded=data.get('expanded')
    if expanded:
        add(f"扩展16384场景：{expanded['arm']}，R_jsd={fmt(expanded['R_jsd'])}，R_attr={fmt(expanded['R_attr'])}，Speed ratio={fmt(expanded['ratios']['speed_jsd'])}。保留搜索4096场景并新增12288，属于扩展验证。",'small')
    add('为什么speed需要检查整条分布','section')
    diagnosis=json.loads((ART/'speed_bin_diagnosis.json').read_text())
    add('上一轮TRAIN5817/4096、16384场景的speed退步主要来自1-2 m/s偏少、3-5 m/s偏多、13-15 m/s偏少。每个GEN的分箱JSD贡献均重算后求平均，合计与原生speed JSD一致。','small')
    table(['speed bin','GT %','657 %','lane %','额外JSD贡献'],
        [[f"{r['speed_bin_mps'][0]}-{r['speed_bin_mps'][1]}",f"{100*r['GT_probability']:.2f}",f"{100*r['control_probability']:.2f}",f"{100*r['lane_probability']:.2f}",f"{r['delta_jsd_contribution']:.5f}"] for r in diagnosis['positive_delta_bins'][:4]],
        [W*.24]+[W*.19]*4)
    probe=data.get('CPU_diagnostic')
    if probe and all('GT_near_zero_fraction' in r for r in probe['results']):
        add('CPU补充诊断：同256场景、GEN2817、已有step4096权重；只观察raw speed，不改变输出。','small')
        table(['方法','负speed %','物理speed<.01 %','speed<1 %'],
            [[r['arm'],f"{100*r['raw_negative_fraction']:.2f}",f"{100*r['physical_near_zero_fraction']:.2f}",f"{100*r['physical_below1_fraction']:.2f}"] for r in probe['results']],
            [W*.37,W*.21,W*.21,W*.21])
        add('GT speed<.01约17.48%，speed<1约41.63%。指标按1 m/s分箱，近零细节不会单独区分。负speed截断问题不能直接解释整个speed JSD，CPU诊断也不参与GPU排名。','small')
    add('验证与限制','section')
    add(f"独立数值校验：{data['numeric_QA']['checks']}项通过。检查原生直方图JSD、匹配初始权重、地图/归一化/数据哈希、实际曝光和EMA更新。完成状态：{data['status']['status']}。",'small')
    add('复用657对照的源码存在关闭的guidance/default和分数输出改动；解析配置和初始张量及GT直方图匹配，不声称源码完全一致。本轮完整完成的训练预算最高4096 updates（约4.25个epoch），仍不证明128epoch/50k最终改善。快速训练重复已打乱顺序，完整训练每epoch重排。','small')
    add('记录：`output/reports/init_diffusion_jsd_round2_20261013/analysis.json`；实验注册、原生输出和各阶段完成标记保存在相同目录及对应logs目录。','small')
    figure=ART/'attribute_histograms.png'
    if figure.exists():
        story.append(PageBreak());add('属性分布诊断','title')
        story.append(Image(str(figure),width=W,height=W*1383/2379));story.append(Spacer(1,10))
        add('同TRAIN5817/4096预算与4096场景；两个GEN计数合并，只用于画图。R和JSD仍使用每个GEN先计算再平均的口径。速度图放大1-8与8-25 m/s区间，评价包含完整范围和0-1 bin。','small')
        add('均值、标准差接近并不保证JSD低','section')
        add('联合属性损失的车辆width均值约2.144 m，原lane约2.149 m，GT约2.162 m；标准差约0.199/0.177/0.190 m。联合损失的标准差更接近GT，但2.1-2.2 m分箱占比降到约19%（GT22.4%），width JSD明显恶化。','small')
        add('采样分布与噪声时间下的预测分布','section')
        add('x0回归学习当前噪声状态下的条件预测；对它做边缘分布匹配并不等同于直接匹配最终ODE样本。多个属性共用denoiser，额外的速度或尺寸梯度也可能影响其他输出。本轮用单项对照检查实际影响，不把排序损失下降当作JSD改善。','small')
        add('目前应如何选配置','section')
        final_rows=[r for r in data['comparisons'] if r['budget']==4096 and r['TRAIN_seeds']==[5817]
                    and r['arm'] in ('lane_only','lane_agentmax05','speed_only') and r['gates_pass']]
        if final_rows:
            best=min(final_rows,key=lambda r:r['R_jsd'])
            add(f"同一4096范围内，{best['arm']}的R_jsd={fmt(best['R_jsd'])}最低。整体匹配只测试2048预算，不能替代4096结果。完整128epoch效果仍需独立确认。",'small')
    out=ROOT/'output/pdf/init_diffusion_jsd_round2_20261013.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Latin',7);canvas.drawString(38,23,'JSD round2 / N20 / '+('FINAL SNAPSHOT' if args.final else 'DRAFT'));canvas.drawRightString(A4[0]-38,23,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=34,bottomMargin=44,
        title='InitDiffusion JSD round2',author='Experiment analysis').build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'pdf':str(out),'final':args.final}))


if __name__=='__main__':main()
