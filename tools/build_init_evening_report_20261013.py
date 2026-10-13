"""Build a Chinese PDF report from actual completed experiment artifacts."""
from pathlib import Path
import json
import statistics
import sys
import math
from datetime import datetime
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/tmp/codex_report_pdf_support')
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.distance import jensenshannon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

BASE = ROOT / 'output/reports/init_diffusion_evening_20261013'
LONG = BASE / 'final_long/rebased/sampler_merge'
SAMPLER = BASE / 'n20_sampler'
OUT = ROOT / 'output/pdf/init_diffusion_evening_20261013.pdf'
JSD = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')
read = lambda p: json.loads(Path(p).read_text())


def pooled(rows):
    result = {}
    for i, key in enumerate(JSD):
        h = [read(Path(r['native_report']).parent/'feature_histograms.json')[key] for r in rows]
        generated = np.sum([v['generated'] for v in h], axis=0)
        real = np.sum([v['matched_gt'] for v in h], axis=0)
        result[key] = float(jensenshannon(generated, real)**2 * (10 if i < 2 else 100))
    return result


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body = ParagraphStyle('body', fontName='STSong-Light', fontSize=10, leading=15,
                          wordWrap='CJK', textColor=colors.HexColor('#183149'), spaceAfter=8)
    title = ParagraphStyle('title', parent=body, fontSize=23, leading=30, spaceAfter=14)
    head = ParagraphStyle('head', parent=body, fontSize=15, leading=21, spaceAfter=10)
    small = ParagraphStyle('small', parent=body, fontSize=8, leading=11)
    story = []
    def p(text, style=body): story.append(Paragraph(escape(str(text)).replace('\n', '<br/>'), style))
    def table(headers, rows, widths=None):
        content = [[Paragraph(escape(str(x)), small) for x in headers]] + [[Paragraph(escape(str(x)), small) for x in row] for row in rows]
        t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0fa')),
                               ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d5e1ea')),
                               ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
                               ('RIGHTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),5),
                               ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t, Spacer(1,12)])
    wb = read(BASE/'wandb/latest_analysis.json')
    records = {r['id']: r for r in wb['manifest']}
    current = records['zww9tc6r']
    state = read(LONG/'status.json')
    sampler_rows = read(SAMPLER/'observations.json') if (SAMPLER/'observations.json').exists() else []
    comparisons = read(SAMPLER/'comparisons.json') if (SAMPLER/'comparisons.json').exists() else []
    sampler_state = read(SAMPLER/'status.json') if (SAMPLER/'status.json').exists() else {}
    guidance_state={}
    for phase in ('guidance_full50000','guidance_final_only50000'):
        guidance_dir=BASE/phase
        guidance_state[phase]=read(guidance_dir/'status.json') if (guidance_dir/'status.json').exists() else {}
        if (guidance_dir/'observations.json').exists():
            all_rows={}
            for r in sampler_rows+read(guidance_dir/'observations.json'):
                all_rows[(r['arm'],r['GEN_seed'],r['option'],r['scenes'])]=r
            sampler_rows=list(all_rows.values())
        if (guidance_dir/'comparisons.json').exists():
            all_comparisons={(c['arm'],c['option'],c['scenes']):c for c in comparisons}
            for c in read(guidance_dir/'comparisons.json'):
                all_comparisons[(c['arm'],c['option'],c['scenes'])]=c
            comparisons=list(all_comparisons.values())
    ref = read(ROOT/'configs/metrics/iu3fa4gx_full50000.json')
    local = read(LONG/'observations.json')
    # Earlier completed short stages live in the preceding immutable phase.
    # Prefer the current phase for any duplicate actual TRAIN/GEN endpoint.
    merged = {}
    for r in read(BASE/'recovery/observations.json') + local:
        key=(r['variant'],r['seed'],r['budget'],r['eval_seed'],r['num_samples'],r['evaluation_only'])
        merged[key]=r
    local=list(merged.values())
    conclusions = dict(generated_local=datetime.now().astimezone().isoformat(), long_status=state,
                       sampler_status=sampler_state, guidance_status=guidance_state, sampler_completed_cells=len(sampler_rows),
                       full_training_recommendation='init_diffusion_657_n20_road_k8_full; unchanged loss, road_edges, N20, H256L3, K8, EMA;128epoch outcome not yet established.',
                       provenance='W&B observational curves and locally paired native tests are separate; numeric1 scope inference withdrawn.',
                       comparisons=comparisons)
    (BASE/'final_report.json').write_text(json.dumps(conclusions, ensure_ascii=False, indent=2)+'\n')
    p('InitDiffusion 晚间优化复核', title)
    p('2026-10-13 | speed、angle、length JSD 优先 | 20 次网络调用 | 约 808 万参数', head)
    p('推荐继续完整训练 init_diffusion_657_n20_road_k8_full。保留 road_edges、K8、TRAIN 归一化、逐层 time conditioning 和原有 component loss。当前推荐来自完整训练趋势与已完成对照，仍不能保证128 epoch最终超过657。')
    p(f"最新远程 K8：epoch {current['last_validation_epoch']}，speed={current['last_validation_metrics']['speed_jsd']:.5f}，angle={current['last_validation_metrics']['ang_dev_jsd']:.5f}，length={current['last_validation_metrics']['length_jsd']:.5f}。W&B读取时间 UTC {wb['inventory_fetched_at_utc']}。")
    p('collision guidance 应作为可选采样对照。它直接调整位置、降低重叠，不直接生成更合理的 speed/length/heading；后续去噪会受位置变化影响。其五圆几何对180度翻转基本不敏感，不能直接解决反向 heading。')
    final_guide=next((c for c in comparisons if c['option']=='guide_final_only' and c['scenes']==50000),None)
    if final_guide:
        anchors=[r for r in sampler_rows if r['arm']=='freshK8' and r['scenes']==50000 and r['option']=='off_uniform']
        guided=[r for r in sampler_rows if r['arm']=='freshK8' and r['scenes']==50000 and r['option']=='guide_final_only']
        p(f"建议提供最后一步引导作为可选评价：碰撞率从{statistics.fmean(r['metrics']['collision_rate'] for r in anchors):.5f}%降至{statistics.fmean(r['metrics']['collision_rate'] for r in guided):.5f}%，R_jsd={final_guide['R_jsd']:.5f}。每sample实测20次网络调用、1次引导。它保留speed/shape预测路径；GPU微小数值漂移和生成类别边界下，统计值不宣称bitwise一致。")
    full = [c for c in comparisons if c['scenes']==50000]
    if full:
        table(['完整50K配对','R_focus','R_jsd','S / A / L 比值','筛选约束'],
              [[c['arm']+' / '+c['option'],f"{c['R_focus']:.4f}",f"{c['R_jsd']:.4f}",
                ' / '.join(f"{c['ratios'][k]:.4f}" for k in FOCUS),c['gates_pass']] for c in full], [180,57,57,133,72])
    else:
        p(f"完整50K采样对照尚未完成。当前采样阶段：{sampler_state.get('status','未运行')}；已完成并核验 {len(sampler_rows)} 个实际TEST单元。此版是进度报告。")
    p('R定义：各指标分别除以参考值，再平均。R_jsd使用六项JSD；R_attr使用length、width、speed；R_focus使用speed、angle、length。数值小于1表示该口径改善。报告同时保留各分量，避免平均值掩盖angle退化。')
    p('完整训练默认每8 epoch验证全部50,000场景，num_workers=8。输出R/R_jsd/R_attr参考iu3fa4gx最终epoch127完整50K文件。短期17epoch模型对这一最终参考的比值只用于监控，不是同预算结论。')
    story.append(PageBreak())
    p('完整训练曲线：观察与限制', head)
    table(['run','最后验证epoch','speed','angle','length','collision %'],
          [[rid,int(records[rid]['last_validation_epoch']),*[f"{records[rid]['last_validation_metrics'][k]:.5f}" for k in FOCUS],f"{records[rid]['last_validation_metrics']['collision_rate']:.5f}"]
           for rid in ('iu3fa4gx','zww9tc6r','4bo59qfn','myqhg1rq','8bblh5do')], [90,73,83,83,83,87])
    p('cc_best_full_map64同时改变了地图邻居数、guidance和exact-x0；它的低碰撞率不能作为guidance单独收益的证据。lane_only与lane_agentmax05的长期angle/length未改善，撤回仅凭短期优胜将它们作为完整训练首选的建议。')
    p('范围更正：W&B会把1.0保存为1，无法据此区分Lightning整数1个batch与浮点1.0全部batch。此前据此断言K8仅128场景的判断已撤回；源full_base使用1.0。657最终参考有实际50K报告；其他远程run需要实际报告成员集合确认。lane两条曲线配置32 batches、每batch128，为4096场景上限。')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(10.1,3.2))
    palette = {'iu3fa4gx':'#333333','zww9tc6r':'#157f8b','myqhg1rq':'#dd7635','8bblh5do':'#a35c98'}
    for rid, color in palette.items():
        r=records[rid]; data=read(r['history_file'])['rows']; points={}
        for row in data:
            if row.get('epoch') is not None and all(isinstance(row.get(k),(float,int)) for k in FOCUS):points[int(row['epoch'])]=row
        for ax,key in zip(axes,FOCUS):
            ax.plot(sorted(points),[points[e][key] for e in sorted(points)],label=rid,color=color,linewidth=1.5)
            ax.set_title(key);ax.set_xlabel('epoch');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    chart=BASE/'wandb_focus_curves.png';fig.savefig(chart,dpi=190);plt.close(fig)
    story.append(Image(str(chart), width=499, height=159))
    p('曲线为观察性结果：不同TRAIN/GEN、可能不同验证范围和后端，不能直接当作配对因果实验。', small)
    story.append(PageBreak())
    p('本地较长训练：匹配预算对照', head)
    values=[]
    for arm in ('fresh657','freshK8','lane_only'):
        for budget in (4096,6826,8192,12288,16384):
            rr=[r for r in local if r['variant']==arm and r['seed']==4817 and r['budget']==budget and r['num_samples']==4096 and not r['evaluation_only']]
            if len(rr)!=2:continue
            values.append([arm,budget,*[f"{statistics.fmean(r[k] for r in rr):.5f}" for k in FOCUS],f"{statistics.fmean(r['lat_dev_jsd'] for r in rr):.5f}"])
    table(['模型','optimizer更新','speed','angle','length','lat'],values,[85,83,83,83,83,82])
    longer=read(LONG/'comparisons.json')['comparisons']
    table(['同预算vs657','更新数','R_focus','R_jsd','S / A / L 比值','约束'],
          [[c['arm'],c['budget'],f"{c['R_focus']:.4f}",f"{c['R_jsd']:.4f}",' / '.join(f"{c['ratios'][k]:.3f}" for k in FOCUS),c['gates_pass'] and c['R_focus']<1 and all(c['ratios'][k]<=1.10 for k in FOCUS)] for c in longer], [85,60,57,57,172,68])
    p('TRAIN4817、GEN2817/3817，随机固定4096场景，EMA、32-true、matmul=medium、冻结同一AIRL地图、20步，生成器8,083,978参数。每一阶段只续接自己的checkpoint，恢复optimizer/EMA/scheduler/RNG；验证过少量后端数值漂移，未宣称bitwise相同。')
    p('训练反复使用同一种子固定顺序，与完整训练每epoch shuffle不同；约17 epoch的结论不能保证128 epoch表现。lane后续长跑暂缓，保留已完成结果，把余量用于同权重采样和完整50K。发生原生崩溃的未保存更新不计入完成结果。')
    story.append(PageBreak())
    p('20步采样：guidance与时间网格', head)
    p('仅在TEST恢复同一EMA checkpoint后修改采样属性；每次sample实际监测恰好20次denoiser调用，不增加参数，不修改训练loss。关闭、弱guidance(scale=.25,t_max=.2,iters=5,margin=0)及cc强度(.5,.3,5,.1)使用相同场景和GEN。')
    p('非均匀时间网格来自e998分支：power或cosine。默认保留原torch.linspace(1,0,21)。terminal_time=.05控制最后一次模型查询在原训练t_eps，另测试raw网格以识别最后噪声时刻变化的混杂。')
    screen=[c for c in comparisons if c['scenes']==4096]
    table(['模型 / 采样','R_focus','speed','angle','length','lat','collision'],
          [[c['arm']+' / '+c['option'],f"{c['R_focus']:.3f}",*[f"{c['ratios'][k]:.3f}" for k in FOCUS],f"{c['ratios']['lat_dev_jsd']:.3f}",f"{c['ratios']['collision_rate']:.3f}"] for c in screen], [199,50,50,50,50,50,50])
    p('表内都是相对同模型off_uniform的比值。选择前要求三项重点均不恶化超过5%，near/width不超过15%，on-road下降不超过2百分点，车辆数变化不超过1%，且R_focus至少改善2%。collision不参与排序。每个方案须两GEN完整；full50K方案在查看full结果之前冻结。')
    p('额外完整50K guidance诊断：即使未达到2% JSD优胜，也测试弱、cc强度和final-only。可选引导的非劣约束为R_focus<=1.01、六项各<=1.05、on-road下降<=2百分点、collision至少减半。最后一步方案通过这一约束；其JSD变化很小，不声称JSD显著改善。')
    story.append(PageBreak())
    p('完整50K验证与pooling', head)
    for arm in ('fresh657','freshK8'):
        rr=[r for r in sampler_rows if r['arm']==arm and r['scenes']==50000]
        order={'off_uniform':0,'guide_light':1,'guide_cc':2,'guide_final_only':3}
        for option in sorted({r['option'] for r in rr},key=lambda name:order.get(name,99)):
            vv=[r for r in rr if r['option']==option]
            if len(vv)!=2:continue
            metric=pooled(vv)
            table([arm+' / '+option,'GEN2817','GEN3817','两GEN指标均值','合并直方图JSD'],
                  [[key,*[f"{next(r for r in vv if r['GEN_seed']==g)['metrics'][key]:.6f}" for g in (2817,3817)],f"{statistics.fmean(r['metrics'][key] for r in vv):.6f}",f"{metric[key]:.6f}"] for key in JSD], [167,83,83,83,83])
            ratios={k:statistics.fmean(r['metrics'][k] for r in vv)/ref['agent_metrics'][k] for k in JSD}
            p(f"对iu3最终127epoch参考的监控值（训练预算不等）：R_jsd={statistics.fmean(ratios.values()):.4f}，R_attr={statistics.fmean(ratios[k] for k in ('length_jsd','width_jsd','speed_jsd')):.4f}，R_focus={statistics.fmean(ratios[k] for k in FOCUS):.4f}。",small)
    if not any(r['scenes']==50000 for r in sampler_rows):p('本阶段完整50K尚未产生已核验结果；不会用4096指标替代50K。')
    p('合并直方图：先相加两GEN的生成/匹配GT bin counts，再按官方口径计算JSD（near/lat乘10，其余乘100）。它与两次JSD的均值不同；同一场景重复生成两次，也不是100,000个独立场景。报告保留两种数值，配对评分默认使用两GEN指标均值。')
    p('完整参考 iu3fa4gx：最终epoch127，EMA，N20，50,000场景。其值：near=.0126673，lat=.0037314，angle=.0372205，length=.2647131，width=.1016307，speed=.0566700，collision=.6581717%。该JSON从origin/cc_full原文件导入并核验，不由小范围结果推算。')
    p('当前17epoch模型在完整50K上的结果，用来确认场景覆盖及采样趋势；只有相同训练预算和条件的657对照可用于本地方法排名。最终128epoch训练仍须原生完整50K评价。')
    story.append(PageBreak())
    p('尺寸与速度分布：实际误差位置',head)
    if (BASE/'attribute_bins_full50000.png').exists():
        story.append(Image(str(BASE/'attribute_bins_full50000.png'),width=499,height=152))
    p('合并两GEN的完整50K直方图显示：K8在0-1m/s区间的质量为42.61%，GT为41.70%；最大的speed差异集中在15-16和20-21m/s区间的不足。这是低速区间质量，不等同于精确停车比例。')
    p('length在4.5-4.7m附近偏集中，较长车辆不足：5.7-5.8m区间GT约1.37%，K8生成约.72%。width在2.0-2.1m区间GT34.82%，生成37.36%。这些分布差异比总训练loss更直接描述当前JSD误差。')
    p('新增可选shape_x0_parameterization=time_residual：在线性或log表示中预测s0=s_t+t*r，再沿既有路径转回物理尺寸。仍计算原x0 component loss，没有增加层或参数；默认identity保持兼容，旧checkpoint按identity加载，改变输出坐标须从头训练。该方法的有效梯度随t变化，不声称优化轨迹等同。')
    followups=[BASE/'shape_matching_followup/comparisons.json',BASE/'shape_matching_followup/length_only/comparisons.json']
    available=[c for file in followups if file.exists() for c in read(file)['comparisons']]
    if available:
        table(['从头训练假设','更新数','R_focus vs K8','S / A / L','约束'],
              [[c['arm'],c['budget'],f"{c['R_focus']:.4f}",' / '.join(f"{c['ratios'][k]:.3f}" for k in FOCUS),c['eligible']] for c in available], [142,60,87,150,60])
    else:p('尺寸残差与matching4的从头短训正在运行；尚无完整评价结果时不填推测数值。')
    if available:
        p('两维size残差4096更新：三项重点改善，但width JSD恶化37%，未通过筛选。随后保留width的identity输出，只改length；其3413更新与既有3413 K8 control配对，不能直接与4096结果排名。matching4未执行，GPU检查遇到低显存ToDesk显示进程时中止，完成结果未丢弃；独立新阶段只忽略已核验的显示进程，未干扰任何外部进程。')
    p('每组使用TRAIN4817、两GEN、4096同场景、原loss/N20/冻结AIRL地图。短训结果只能提出候选，不能保证完整128epoch优胜。尺寸残差config单独保留为实验项，主完整训练推荐未替换。')
    story.append(PageBreak())
    p('融合内容与下一步', head)
    table(['e998分支内容','本目录处理'],[
        ['power/cosine采样网格','已融合；补充严格参数、单调性校验和terminal控制；默认均匀网格bitwise一致。'],
        ['R_jsd / R_attr与50K参考','已接入当前更完整的scope验证实现，保留最终iu3参考。'],
        ['collision guidance','本目录已有；保留固定ego、generated type、物理坐标修复，未覆盖为分支旧版。'],
        ['exact-x0 / EMA warm-start','已有可选项；本轮比较关闭，未更改训练初始化。'],
        ['5层denoiser','超过当前约810万参数上限，未采用。'],
        ['80epoch schedule','新增3层K8可选recipe；max_epochs和lr_total_steps均80，尚无验证结论。'],
    ],[160,339])
    probe=read(BASE/'matching_probe/results.json')
    a,b=probe['means']['heading4_speed4'],probe['means']['baseline']
    p(f"TRAIN-only Hungarian诊断（128场景、每batch4次同噪声）：heading/speed权重4使约{a['changed_fraction']*100:.1f}%配对变化，heading绝对路径缩短{(1-a['heading_abs_radian']/b['heading_abs_radian'])*100:.1f}%，speed归一化残差MSE下降{(1-a['speed_normalized_mse']/b['speed_normalized_mse'])*100:.1f}%，但position残差增加{(a['position_normalized_mse']/b['position_normalized_mse']-1)*100:.1f}%。这只是endpoint耦合代价，不是学习收益或JSD结果。")
    followup=BASE/'matching_followup/comparisons.json'
    if followup.exists():
        result=read(followup)['comparisons']
        table(['从头训练匹配','TRAIN','更新数','R_focus vs K8','S / A / L','约束'],
              [[c['arm'],c['seed'],c['budget'],f"{c['R_focus']:.4f}",' / '.join(f"{c['ratios'][k]:.3f}" for k in FOCUS),c['eligible']] for c in result], [105,52,52,85,145,60])
    p('验证：时间网格6 tests、collision坐标12 tests、heading路径22 tests、尺寸残差9 tests通过。独立复算42个实际TEST单元、1094checks PASS。一个旧heading测试误用了t^-3期望；按既有x0 MSE的t^-2修正测试期望，训练目标未改。')
    p('length_time_residual只修改长度输出，width保留既有x0坐标；新增单独配置init_diffusion_657_n20_road_k8_length_residual。没有增加参数。它与同时生成两维残差的time_residual是不同训练选项，checkpoint必须使用原选项恢复。')
    p('执行完整训练：python src/run.py experiment=init_diffusion_657_n20_road_k8_full。完整评价：设置INIT_DIFFUSION_ROAD_K8_CKPT后，使用init_diffusion_657_n20_road_k8_full_eval。可选80epoch：init_diffusion_657_n20_road_k8_full_e80。')
    p('最后一步guidance评价：同一checkpoint环境变量，experiment=init_diffusion_657_n20_road_k8_full_guidance_eval。s=.5、t_max=.05、iters=5、margin=.1m，仅预测车辆。最后一步固定均匀N20；改变时间网格时t_max可能触发不同次数，须单独核验。')
    p('审计文件：output/reports/init_diffusion_evening_20261013/final_long/rebased/sampler_merge/（source hashes、续训trace、原生报告）；n20_sampler/（冻结协议、逐单元checkpoint SHA、NFE审计、GEN、直方图、50K选择）；wandb/（只读原始快照）。',small)
    guide_status=' / '.join(v.get('status','未运行') for v in guidance_state.values())
    p(f"状态：训练 {state['status']}；采样 {sampler_state.get('status')}；完整guidance {guide_status}；PDF生成时间 {conclusions['generated_local']}。",small)
    def footer(canvas, doc):
        canvas.setFont('STSong-Light',8);canvas.setFillColor(colors.HexColor('#61758a'))
        canvas.drawString(48,25,'InitDiffusion / 2026-10-13 / 已核验实际结果')
        canvas.drawRightString(A4[0]-48,25,str(doc.page))
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=44,
                      title='InitDiffusion 晚间优化复核',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(OUT), flush=True)


if __name__=='__main__':main()
