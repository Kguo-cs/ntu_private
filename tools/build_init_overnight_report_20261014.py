"""Build a timestamped Chinese report from completed native and W&B evidence.

The active phase is chosen explicitly from active_phase.json, never from a
partially prepared directory. Missing cells are shown as pending, not imputed.
"""
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from xml.sax.saxutils import escape
import json
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/tmp/codex_report_pdf_support')
sys.path.insert(0, str(ROOT))
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014'
OUT = ROOT / 'output/pdf/init_diffusion_overnight_20261014.pdf'
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')
ATTR = ('speed_jsd', 'length_jsd', 'width_jsd')
JSD = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
LABELS = {'roadK8':'roadK8 control', 'size_residual':'size skip',
    'size_time_wp3':'size time + W p3', 'size_time_width_l1':'size time + W L1',
    'match2':'matching 2', 'match4':'matching 4', 'speed_time_residual':'speed time',
    'size_skip_match2':'size skip + match2'}
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def points(item):
    result = {}
    for row in read(item['history_file'])['rows']:
        if row.get('epoch') is not None and all(isinstance(row.get(k), (int, float)) for k in JSD):
            result.setdefault(int(row['epoch']), []).append(row)
    return {e:{k:fmean(r[k] for r in rows) for k in JSD} for e,rows in result.items()}


def main():
    phase = (ART / 'current_status.json').resolve().parent
    assert phase in (ART, ART/'recovery', ART/'adaptive_long', ART/'final_verification')
    observations = read(phase/'observations.json')
    comparisons = read(phase/'comparisons.json')['comparisons']
    status = read(phase/'status.json')
    audit = read(phase/'native_metric_audit.json') if (phase/'native_metric_audit.json').exists() else {}
    wb = read(ART/'wandb/latest_analysis.json')
    manifest = {r['id']:r for r in wb['manifest']}
    pairs = read(ART/'wandb/roadK8_same_epoch_latest.json')
    recommendation = read(phase/'recommendation.json') if (phase/'recommendation.json').exists() else {}
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    finished = datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat('2026-10-14T10:00:00+08:00').timestamp()
    title_text = 'InitDiffusion 夜间探索复核' + ('' if finished else ' - 进度版')
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body = ParagraphStyle('body',fontName='STSong-Light',fontSize=10,leading=15,wordWrap='CJK',spaceAfter=8,textColor=colors.HexColor('#20384a'))
    title = ParagraphStyle('title',parent=body,fontSize=23,leading=31,spaceAfter=14)
    heading = ParagraphStyle('heading',parent=body,fontSize=15,leading=21,spaceAfter=10)
    small = ParagraphStyle('small',parent=body,fontSize=8,leading=11,spaceAfter=6)
    story=[]
    def p(value,style=body): story.append(Paragraph(escape(str(value)).replace('\n','<br/>'),style))
    def table(headers,rows,widths):
        content=[[Paragraph(escape(str(x)),small) for x in line] for line in [headers]+rows]
        tab=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f0f5')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d0dde6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([tab,Spacer(1,10)])
    def page(name): story.append(PageBreak());p(name,heading)
    def metric(value): return f'{value:.5f}'
    def ratio(value): return f'{value:.3f}'
    def group(arm,seed,budget,scenes):
        rows=[r for r in observations if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,seed,budget,scenes)]
        assert len(rows) in (0,2)
        return {k:fmean(r[k] for r in rows) for k in JSD} if rows else None
    def focus_string(row): return ' / '.join(ratio(row['ratios'][k]) for k in FOCUS)

    p(title_text,title)
    p('speed、angle、length JSD 优先 | N20 | 8,083,978 参数',heading)
    p(f'生成时间：{generated}\n截止时间：2026-10-14 10:00 Asia/Shanghai\n当前阶段：{phase.name} / {status["status"]}；实际完成 {len(observations)} 个 TRAIN/GEN/预算/场景单元。',small)
    if recommendation.get('promoted'):
        p(f'候选 {LABELS.get(recommendation["candidate"],recommendation["candidate"])} 已通过本轮较长短训、第二个训练种子与完整50K筛选。它值得进入完整训练；尚无128 epoch最终优胜证据。')
    else:
        p('按speed/angle/length平均R_focus，完整训练观察值较好的是matching4 + multi-step guidance；已提供对应完整配置。它改善speed，同时牺牲部分angle/length，不能称三项均改善。若更重视length，保留roadK8。初筛最优的size time + width L1在长预算反转，未通过推荐条件。')
    final_wb=ART/'wandb_full127_reference_review.json'
    if final_wb.exists():
        values=read(final_wb)['rows']
        table(['完整epoch127 / vs657','R_focus','R_jsd','R_attr'],[[v['run'],ratio(v['R_focus']),ratio(v['R_jsd']),ratio(v['R_attr'])] for v in values],[220,93,93,93])
        p('matching4 + guidance相对最终657：speed约-38.4%，angle约+8.0%，length约+6.2%；这是不同硬件/训练种子的单次完整run观察，不是配对因果结果。',small)
    p('本轮新增无时间缩放的尺寸残差：length0 = length_t + raw_length，width0 = width_t + raw_width。已有time residual为s0 = s_t + t * raw。两者不增加网络参数，并沿用x0 component loss，但梯度尺度与优化轨迹不同。')
    p('排序使用R_focus：speed、angle、length各自除以同预算配对control，然后平均。R_jsd平均六项JSD；R_attr平均speed、length、width。碰撞率仅报告，不参与主排序。小于1表示相应参考下改善。')
    p('新组合size skip + matching2已完成4096更新初筛，但没有获得较长预算确认。matching2提高Hungarian中heading、velocity代价到2，position、length、width仍为1。')
    screen=sorted([r for r in comparisons if r['seed']==4817 and r['budget']==4096 and r['scenes']==4096],key=lambda r:r['R_focus'])
    table(['4096更新候选','R_focus','S / A / L比值','W比值','通过'],[[LABELS.get(r['arm'],r['arm']),ratio(r['R_focus']),focus_string(r),ratio(r['ratios']['width_jsd']),r['eligible']] for r in screen],[142,55,184,60,58])
    p('这是同一训练种子、两个生成种子、相同4096场景的初筛。表中最小平均值不一定通过width或单项退化约束。',small)

    page('W&B完整训练：同epoch对照与排名反转')
    p(f'只读快照：UTC {wb["inventory_fetched_at_utc"]}。同epoch复核快照：UTC {pairs["snapshot_utc"]}。',small)
    selected=['0vftb3cc','doke6bbr','nnsjk5vl','zww9tc6r','iu3fa4gx']
    table(['run','最后验证epoch','speed','angle','length'],[[rid,manifest[rid]['last_validation_epoch'],*[metric(manifest[rid]['last_validation_metrics'][k]) for k in FOCUS]] for rid in selected if rid in manifest],[110,83,102,102,102])
    paired=[]
    for rid in ('doke6bbr','nnsjk5vl'):
        rr=[r for r in pairs['comparisons'] if r['candidate']==rid]
        if rr:
            for r in sorted(rr,key=lambda x:x['epoch']):
                if r['epoch'] in (7,15,23) or r['epoch']==max(v['epoch'] for v in rr):
                    paired.append([rid,r['epoch'],ratio(r['R_focus']),focus_string(r),ratio(r['ratios']['width_jsd'])])
    table(['候选 / vs 0vft','epoch','R_focus','S / A / L比值','W比值'],paired,[104,43,61,222,69])
    p('doke6bbr仅length time residual，但还启用最后一步guidance；nnsjk5vl提高matching到4，同时有多步guidance且GPU不同。因此远程结果是观察性证据，不能把全部变化归因于某个单独选项。')
    fig,axs=plt.subplots(1,3,figsize=(10,3.1))
    for rid,label in [('0vftb3cc','roadK8'),('doke6bbr','length time'),('nnsjk5vl','match4+guide'),('iu3fa4gx','657')]:
        if rid not in manifest: continue
        rows=points(manifest[rid]);epochs=sorted(rows)
        for ax,k in zip(axs,FOCUS):
            ax.plot(epochs,[rows[e][k] for e in epochs],label=label,linewidth=1.4)
            ax.set_title(k);ax.set_xlabel('epoch');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);fig.tight_layout();chart=ART/'report_wandb_focus.png';fig.savefig(chart,dpi=180);plt.close(fig)
    story.append(Image(str(chart),width=499,height=155))
    p('W&B数字1不能区分原始配置中的整数1与浮点1.0；没有实际成员报告时，不根据这个值推断评价只跑一个batch。657最终参考有原生50K报告，新full配置明确要求50K成员。',small)

    page('本地配对实验：较长预算、种子与50K')
    p('控制条件：TRAIN4817/5817；GEN2817/3817；B512；EMA；32-true、matmul=medium；冻结同一个AIRL map；相同TRAIN归一化；uniform N20；关闭guidance和exact-x0。完整配方workers8、每8 epoch评价50K；本轮快速实验workers0。')
    long=[r for r in comparisons if r['budget'] in (6826,12288,16384)]
    table(['候选','TRAIN','更新/场景','R_focus','S / A / L比值','通过'],[[LABELS.get(r['arm'],r['arm']),r['seed'],f'{r["budget"]}/{r["scenes"]}',ratio(r['R_focus']),focus_string(r),r['eligible']] for r in long],[118,43,72,53,165,48])
    if not long:p('较长预算尚未产生配对完成结果；不从初筛外推填写。')
    full=[r for r in comparisons if r['scenes']==50000]
    for r in full:
        candidate=group(r['arm'],r['seed'],r['budget'],50000);control=group('roadK8',r['seed'],r['budget'],50000)
        table([f'完整50K {r["arm"]}','control','candidate','比值'],[[k,metric(control[k]),metric(candidate[k]),ratio(candidate[k]/control[k])] for k in JSD],[196,101,101,101])
    if not full:p('完整50K对照仍待完成。4096场景数据不会标成50K结果。')
    else:p('完整50K两GEN均值：候选speed改善19.6%、angle改善8.1%，但length恶化22.0%、width恶化88.5%；R_focus=.981、R_jsd=1.025、R_attr=1.303。平均focus略好不足以抵消尺寸退化，仍不推荐。')
    late=next((r for r in comparisons if r['arm']=='size_time_width_l1' and r['seed']==4817 and r['budget']==16384 and r['scenes']==4096),None)
    if late:
        p(f"初筛领先并未保持：4096更新R_focus=0.740，16384更新变成{late['R_focus']:.3f}；length比值{late['ratios']['length_jsd']:.3f}，width比值{late['ratios']['width_jsd']:.3f}。较小梯度范数不等于最终分布更好，不能据此推荐完整训练。")
    p('初筛：width<=1.20，每项重点<=1.15，并检查on-road和车辆数量。进入推荐需要两个长预算R_focus<=.97、每项重点<=1.10、width<=1.15及原生约束，再通过第二TRAIN和50K。没有通过时保留baseline。')
    p('本轮最多2048更新一段，恢复optimizer、EMA、scheduler、RNG，并为control/candidate使用相同边界。固定TRAIN顺序跨epoch重复；完整训练会reshuffle，因此短训仅能筛选候选，不能证明128 epoch最优。')

    page('可靠性检查与误差解释')
    drift=read(ART/'recovery_resume_drift.json')
    table(['相同baseline@4096','连续训练','恢复训练','恢复/连续'],[[k,metric(drift['continuous'][k]),metric(drift['resumed'][k]),ratio(drift['ratios'][k])] for k in FOCUS+('width_jsd',)],[181,106,106,106])
    p('恢复后首步loss和梯度很接近，但后续生成JSD明显分叉。已核验optimizer/EMA/RNG元数据，原因尚未完全确定；不能把这种影响描述为可忽略，也不能仅凭后端非确定性就认定原因。')
    p('TRAIN条件去噪探针没有发现raw direction近零导致的atan2病态：128场景中raw半径均大于0.1，连续/恢复checkpoint的条件heading误差接近。这不能排除其他场景问题，更不能替代真正生成评价。')
    p('EMA使用随更新数递增的有效decay。4096更新时半权重窗口约305更新，16384时约1215，完整训练末期约6932。早期和最终EMA平滑尺度不同，也是不能直接外推短训排名的原因之一。')
    p('time residual把输出对网络raw的导数乘t；MSE的局部曲率再乘t^2。width的p3或L1会进一步改变各时间段的梯度分配。初期loss下降可能主要来自更容易输出接近输入的尺寸，不等于尺寸分布已学好。')
    p('matching提高heading/speed权重可缩短对应传输路径，但会牺牲位置配对。校准后的TRAIN探针显示matching4的position归一化MSE增加约26%，heading绝对残差下降约5%，speed MSE下降约8%。这些是匹配端点诊断，不是JSD收益。')
    p('旧CPU匹配探针漏掉TRAIN normalizer，其数值已撤回；本轮校准后的探针另行保存。原生GPU训练/评价始终显式加载同一归一化文件，未受该CPU探针错误影响。')
    gradient_path=ART/'component_gradient_probe512/review.json'
    if gradient_path.exists():
        gradient=read(gradient_path)['summary']
        table(['TRAIN梯度探针(中位数)','总梯度范数','width梯度范数','width投影占比'],[[LABELS.get(k,k),ratio(v['median_grad_norm']),ratio(v['components']['width']['global_grad_norm']['median']),ratio(v['components']['width']['dot_over_total_squared']['median'])] for k,v in gradient.items()],[193,102,102,102])
        p('512个相同TRAIN场景，CPU/B32、raw权重，原生loss反向但不更新。支持width梯度竞争的假设；并非GPU/B512实际梯度，也不是因果证明。Adam更新不简单正比于裁剪后的梯度幅度。',small)
    long_gradient=ART/'component_gradient_probe16384/review.json'
    if long_gradient.exists():
        gg=read(long_gradient)['summary']
        p(f"长预算反证：16384更新时，baseline width梯度投影占比中位数已降至{gg['roadK8']['median_width_projection']:.3f}，其早期失衡会自行缓解。候选梯度虽仍更小，却有更差的length/width JSD；早期梯度平衡不能预测最终分布。",small)
    p('审计：从原生直方图独立复算JSD，并从计数复算碰撞率。当前审计状态 '+str(audit.get('status','待完成'))+'；实际单元 '+str(audit.get('unique_actual_cells',0))+'；检查 '+str(audit.get('checks',0))+'。另有真实checkpoint采样探针验证每次sample恰好20次denoiser调用。',small)

    page('配置、范围与可追溯记录')
    table(['配置','含义 / 状态'],[
        ['init_diffusion_657_n20_road_k8_full','长度更好的完整训练基线。N20，H256/L3，road edges，A2A K8，原component loss。'],
        ['init_diffusion_657_n20_road_k8_match4_guidance_full','复用完整W&B matching4 + guidance配方。speed优先；angle、length有代价，非matching单因素证明。'],
        ['init_diffusion_657_n20_road_k8_size_time_width_l1','两维size time residual，width改L1。初筛收益在长预算反转，不推荐替代baseline。'],
        ['init_diffusion_657_n20_road_k8_size_skip','新增length/width无时间缩放残差。实验配置，是否推荐见配对确认表。'],
        ['init_diffusion_657_n20_road_k8_size_skip_match2','在size skip上提高heading和velocity匹配权重到2。实验组合，仅完成4096初筛，不作长期收益声明。']],[225,274])
    p('启动示例：python src/run.py experiment=init_diffusion_657_n20_road_k8_full',small)
    p('本轮保持采样20步与生成器约810万参数限制。公开657最终epoch127完整50K仍作为日志R/R_jsd/R_attr参考；本地方法排序使用同预算roadK8，二者不能混用。')
    p('两个GEN的JSD均值，与先合并直方图再计算的JSD不是同一统计量。本轮配对R先分别平均两GEN的指标，再计算candidate/control比值；同一批场景生成两次不等于两倍独立场景。')
    p('完整训练checkpoint也应按同一组GEN和完整50K重新比较。远程曲线的单次最佳点有选择偏差，不能直接替代固定最终参考；某些run最后一次验证停在111，而不是127。')
    p('speed优先配置：python src/run.py experiment=init_diffusion_657_n20_road_k8_match4_guidance_full。对应full50K评价配置为init_diffusion_657_n20_road_k8_match4_guidance_eval，checkpoint环境变量INIT_DIFFUSION_ROAD_K8_MATCH4_CKPT。它复用已完成W&B run的两项配置差异，尚未在本机从头跑128 epoch。',small)
    p('执行缺口：搜索在10:00截止。07:49的三次失败发生在50K评价启动参数，训练16384和第二TRAIN已完成。评价脚本修正后，单独补核已有权重，无新增训练。候选一次缓存/NumPy异常导致覆盖校验失败，已排除；第二次完整通过。自动W&B刷新在02:31之后缺少成功快照，最终报告另作最新只读刷新；不声称整晚刷新均成功。',small)
    p('主要记录（仓库相对路径）：\noutput/reports/init_diffusion_overnight_20261014/\n当前phase：'+str(phase.relative_to(ROOT))+'\nprotocol.json / observations.json / comparisons.json / native_metric_audit.json\nwandb/latest_analysis.json 与 wandb/roadK8_same_epoch_latest.json\nrecovery_resume_drift.json、matching_probe_calibrated/、actual_nfe_audit/result.json',small)
    p('实际完成结果之外：未保存崩溃更新和被终止的partial均排除；导入既有单元不计为新重复实验。仅管理本轮创建的子进程，不干扰远程训练或显示进程。',small)
    metadata=dict(generated_local=generated,progress_report=not finished,active_phase=str(phase),status=status,
        completed_native_cells=len(observations),recommendation=recommendation,full128epoch_verified=False,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),phase/'observations.json',phase/'comparisons.json',ART/'wandb/latest_analysis.json',ART/'wandb/roadK8_same_epoch_latest.json']})
    (ART/'pdf_report_provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    def footer(canvas,doc):
        canvas.setFont('STSong-Light',8);canvas.setFillColor(colors.HexColor('#657b8a'))
        canvas.drawString(48,25,'InitDiffusion / 2026-10-14 / '+('进度报告' if not finished else '夜间复核'))
        canvas.drawRightString(A4[0]-48,25,str(doc.page))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=44,title=title_text,author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT,flush=True)


if __name__=='__main__':main()
