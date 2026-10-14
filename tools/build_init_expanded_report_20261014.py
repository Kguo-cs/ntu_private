"""Final report only: requires an explicit decision and a current metric audit."""
from datetime import datetime
from pathlib import Path
from statistics import fmean
from xml.sax.saxutils import escape
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tmp/codex_report_pdf_support')
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image

ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
OUT=ROOT/'output/pdf/init_diffusion_expanded_20261014.pdf'
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
JSD=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd')
LABELS={'roadK8':'roadK8 control','match4':'matching4','match_v4h2_s2':'V4/H2/L2/W2',
    'match4_h320d4':'H320 x4 + match4','match4_h256d5':'H256 x5 + match4',
    'match4_map64':'map64 + match4','match4_attrdist':'attribute loss + match4',
    'prior_speed':'TRAIN speed source','prior_shape_speed':'TRAIN speed + size source',
    'match4_speedhead_paired':'paired speed head + match4','match4_length_l1':'length L1 + match4',
    'match4_heading_p1':'heading power1 + match4','match4_shape_affine':'TRAIN affine size + match4',
    'match4_affine_heading_p1':'affine size + heading power1'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(decision_path=None, output_path=None):
    decision_path=Path(decision_path) if decision_path else ART/'final_decision.json'
    output_path=Path(output_path) if output_path else OUT
    decision=read(decision_path)
    rows=read(ART/'observations.json');comparisons=read(ART/'comparisons.json')['comparisons']
    audit=read(ART/'native_metric_audit.json');plan=read(ART/'protocol.json')
    assert audit['status']=='PASS' and audit['observations_sha256']==sha(ART/'observations.json')
    training_audit=read(ART/'training_schedule_audit.json')
    assert training_audit['status']=='PASS' and training_audit['observations_sha256']==sha(ART/'observations.json')
    assert decision['recommended_full_config'] and decision['scope'] and decision['summary']
    wb=read(ART/'wandb/latest_analysis.json');manifest={r['id']:r for r in wb['manifest']}
    target_file=ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json'
    target_spec=read(target_file);target_progress=read(ART/'all_seven_target_progress.json')
    assert target_progress['targets_sha256']==sha(target_file)
    assert target_progress['observations_sha256']==sha(ART/'observations.json')
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body=ParagraphStyle('body',fontName='STSong-Light',fontSize=10,leading=15,wordWrap='CJK',spaceAfter=8,textColor=colors.HexColor('#20384a'))
    title=ParagraphStyle('title',parent=body,fontSize=23,leading=30,spaceAfter=14)
    heading=ParagraphStyle('heading',parent=body,fontSize=15,leading=21,spaceAfter=10)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11,spaceAfter=6)
    story=[]
    def p(text,style=body):story.append(Paragraph(escape(str(text)).replace('\n','<br/>'),style))
    def page(text):story.append(PageBreak());p(text,heading)
    def table(headers,items,widths):
        cells=[[Paragraph(escape(str(v)),small) for v in line] for line in [headers]+items]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f0f5')),
            ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d0dde6')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,10)])
    def ratios(c):return ' / '.join(f'{c["ratios"][k]:.3f}' for k in FOCUS)
    def compare_table(items):
        table(['候选 / TRAIN / 更新数','R_focus','S / A / L 比值','W 比值','R_jsd'],[
            [LABELS.get(c['arm'],c['arm'])+' / '+str(c['seed'])+' / '+str(c['budget']),f'{c["R_focus"]:.3f}',
             ratios(c),f'{c["ratios"]["width_jsd"]:.3f}',f'{c["R_jsd"]:.3f}'] for c in items],
            [150,54,167,62,66])
    def image(path,width=499,height=170):
        if path.exists():story.extend([Image(str(path),width=width,height=height),Spacer(1,9)])

    p('InitDiffusion 扩大搜索报告',title)
    p('目标：完整50K七项胜过SD/VW | 最多24步 | 冻结AIRL地图',heading)
    p(f'截止：2026-10-14 22:30 Asia/Shanghai\n报告生成：{datetime.now().astimezone().isoformat(timespec="seconds")}\nW&B快照：{wb["inventory_fetched_at_utc"]}',small)
    p(decision['summary'])
    p('推荐完整训练配置：'+decision['recommended_full_config'])
    p('证据范围：'+decision['scope'])
    p(f'本地完成 {len(rows)} 个训练预算/训练种子/生成种子/场景数单元；独立审计 {audit["unique_actual_cells"]} 个单元、{audit["checks"]} 项检查。64步小测试仅验收原生流程，不参与候选排名。')
    p('成功标准：同一个checkpoint、同一套采样配置，在完整50K中逐项低于SD/VW两者更好的数值；两种预先指定GEN分别报告。任何一项未达标，都不能用其他项目的提升抵消。')
    p('R_focus平均speed、angle、length三项候选/control比值；R_jsd平均六项JSD比值；R_attr平均speed、length、width三项比值。这些平均值保留作趋势诊断，不再作为全指标达标证明。W&B完整训练与本地同预算短训分别呈现。')
    for text in decision.get('key_findings',[]):p(text)
    p('主要限制',heading)
    for text in decision.get('limitations',[]):p(text,small)

    page('固定目标与完整训练的实际差距')
    limits=target_spec['targets']
    target_labels={'ang_dev_jsd':'angle','length_jsd':'length','speed_jsd':'speed','width_jsd':'width',
        'lat_dev_jsd':'lateral deviation','nearest_dist_jsd':'nearest distance','collision_rate':'collision (%)'}
    reference=manifest.get('0vftb3cc',{}).get('last_validation_metrics') or {}
    entries=[]
    for key,limit in limits.items():
        baseline=min(target_spec['baselines'],key=lambda name:target_spec['baselines'][name]['user_reported_metrics'][key])
        value=reference.get(key)
        reduction=max(0.,1.-limit/value) if value else None
        entries.append([target_labels[key],f'< {limit:.6f}',baseline,
            f'{value:.6f}' if value is not None else '未取得',
            ('已低于门槛' if value<limit else f'需降低{reduction*100:.1f}%') if value is not None else '未知'])
    table(['指标','严格门槛','门槛来源','roadK8最终数值','差距'],entries,[102,86,100,100,111])
    p('SD与VW原始报告均为相同完整50000场景、相同GT特征计数、相同评价源码和reference map。数值门槛保留用户提供的终端精度，原始double精度JSON与SHA也已登记。',small)
    p('上表roadK8来自W&B最后验证记录。是否具备本地成员与源码哈希验证，单独记录；远端配置或相同epoch不能替代完整成员核对。',small)
    p('输入条件限制：当前InitDiffusion固定ego位置、朝向、尺寸和类型；SD/VW生成ego。因此达到数值门槛仍不能直接声称同等输入条件下全面胜出。',small)
    p('当前roadK8的SMART条件地图为ego-local半边长50m正方形，返回road-edge queries；SD/VW使用各自autoencoder的紧凑lane条件。相同的完整参考评价地图不意味着条件特征、裁剪和编码方式完全一致。',small)
    remote=[r for r in target_progress['remote_observations'] if r['run'] in ('iu3fa4gx','0vftb3cc','nnsjk5vl')]
    table(['完整训练参考','验证epoch','已过 / 7','尚未达标'],[
        [r['run'],r['epoch'],r['passed_metrics'],' / '.join(target_labels[k] for k in r['failed_metrics'])]
        for r in remote],[118,78,78,225])
    p('已有完整训练的共同短板为angle和length。提高speed但让这两项变差的方案，不能仅凭更低的平均R获得成功结论。')

    page('实验协议与新方法')
    p('数据：training_map2_sd；筛选使用固定4096个场景，确认使用完整50000场景。TRAIN种子4817/5817，GEN种子2817/3817；以实际完成单元为准，不补齐缺失结果。batch512，FP32，matmul medium，EMA，固定AIRL地图权重。')
    p('quick runner重复同一份有种子的训练顺序，完整recipe每个epoch重新打乱。不同网络不能共享所有初始权重；比较保持场景顺序、训练曝光量和评价种子匹配。同种子短跑也出现数值轨迹差异，因此不把单次早期名次当作最终排名。')
    if (ART/'training_recipe_crosscheck.json').exists():
        p('声明配置复核：本地与远端roadK8/affine均为batch512、LR0.0005、Trainer32-true、梯度裁剪0.5、累积1步。暂未发现这些设置不一致；远端实际matmul精度、运行时源码与张量哈希未独立核实。训练顺序、验证场景数和生成随机数仍存在协议差异，不能把短训与完整训练的排名反转归因于某一个因素。',small)
    arch=read(ART/'architecture_preflight.json')
    table(['结构检查','生成器参数','地图冻结'],[[r['arm'],f'{r["generator_params"]:,}',r['map_frozen']] for r in arch],[240,145,114])
    p('参数检查不代表每个结构都完成训练；实际完成候选见后续结果表。按训练集证据调整试验队列的记录保存在registrations/pre_attribute_prior及protocol.json。')
    p('新目标阶段新增两项单变量实验：matching4上仅把length的MSE改成L1；或仅把heading的时间幂从2改成1。前者检验尺寸回归形式对生成峰值的影响，后者提高高噪声状态的相对监督。两者均是假设，不预设优于原目标。',small)
    time_weights=ART/'heading_time_weight_diagnostic.json'
    if time_weights.exists():
        timing=read(time_weights)
        p('heading时间权重诊断：Uniform(0,1)，t_eps=.05，归一化平均权重为1。以下为权重积分占比，不是实际梯度份额。',small)
        table(['time power','t<0.1的权重占比','t>0.5的权重占比'],[
            [r['time_power'],f'{100*r["weight_integral_t_below_0p1"]:.1f}%',
             f'{100*r["weight_integral_t_above_0p5"]:.1f}%'] for r in timing['rows']],[115,192,192])
    p('新增TRAIN仿射尺寸输出：网络预测标准化的clean尺寸，经训练均值/标准差还原后使用原MSE目标。保留同一噪声、匹配、采样路径，不增加参数；14项尺寸测试通过，默认模型133个状态张量与RNG精确保持。该设计检验输出数值尺度，不能预设改善生成JSD。',small)
    p('新增一项交互对照：在matching4上组合TRAIN仿射尺寸输出和heading时间幂1，与两项各自单独启用及原matching4形成四组比较。假设是减轻提高角度监督后的尺寸退步；是否成立以配对生成结果为准，不将两项收益直接相加。',small)
    p('独立speed head的首轮初始化被排除出严格配对排名：外层Flow初始化令共享time MLP权重和RNG改变。原生数值保留，完成初始化隔离及10项测试后重新运行。默认共享head的权重与RNG保持相同。',small)
    p('恢复训练审计发现并修复一项实验执行错误：Trainer的reload_dataloaders_every_n_epoch误写为单数，导致首个恢复epoch的尾段被重复读取，后续epoch及学习率提前推进。旧roadK8的6144步结果及匹配4的未完成恢复阶段已排除；所有从头完成的2048步初筛保留，后续均从有效2048步checkpoint重跑。原2560状态哈希检查不足以证明后续数据轮次正确，不再作为有效恢复证据。',small)
    p('修复验收使用真实Lightning CPU小模型与AdamW：四个中途保存点恢复至第11步，逐项验证样本顺序、Python/NumPy/Torch随机数、模型与EMA、优化器及scheduler状态和连续训练完全相同。真实GPU阶段另审计每个epoch的实际场景数和每一步学习率。具体记录见cyclic_resume_regression.json和training_schedule_audit.json。',small)
    prior_notice=ART/'prior_resume_evidence_correction.json'
    if prior_notice.exists():
        correction=read(prior_notice)
        p(f'更正也涉及此前evening/overnight搜索的{correction["affected_stages"]}个阶段，其中{correction["affected_full50000_stages"]}个为旧权重的完整50K评价。那些数字仍是旧权重的原始测量，但训练不符合原定协议，相关短训排名和长期推断不再支持当前推荐。原始证据保留，逐阶段原因见prior_resume_evidence_correction.json。src/run.py完整训练不使用该helper，独立W&B结果不受此错误影响。',small)
    page('训练经验源：只改变噪声起点')
    p('TRAIN quantile source：只使用原normalizer的8192个训练场景、76027个agent，为speed/length/width保存2049个分位点。以已有高斯随机数的CDF查询训练分位表，不新增随机抽样、网络参数或监督目标。未按评价GT类型或GT属性选表；shape字段独立变换，没有保留原length/width联合相关。')
    image(ART/'attribute_prior/source_cdf_comparison.png',height=144)
    p('上图只检查t=1起点分布，不是生成场景的评价结果，也不用于判定模型胜出。表格随checkpoint保存；默认关闭时原生成器tensor hash保持完全相同。',small)

    page('短期筛选与较长预算验证')
    screen=sorted([c for c in comparisons if c['budget']==2048 and c['seed']==4817 and c['scenes']==4096],key=lambda c:max(c['ratios']['ang_dev_jsd'],c['ratios']['length_jsd']))
    p('2048更新初筛；参考同预算roadK8，两种GEN。',small);compare_table(screen)
    midpoint=sorted([c for c in comparisons if c['budget']==6144 and c['seed']==4817 and c['scenes']==4096],key=lambda c:max(c['ratios']['ang_dev_jsd'],c['ratios']['length_jsd']))
    if midpoint:p('6144更新的中期验证。',small);compare_table(midpoint)
    p('大网络学习速度可能不同，早期落后不足以证明长期无效。候选是否继续训练、是否采用探索性补位，以保存的selection文件为准。')

    page('新增方法自身的收益与speed误差来源')
    parent_path=ART/'parent_comparisons.json'
    if parent_path.exists():
        parents=read(parent_path)
        assert parents['source_observations_sha256']==sha(ART/'observations.json')
        own=[c for c in parents['comparisons'] if c['budget']==2048 and c['parent']=='match4']
        p('以下改为同预算matching4作参考，以免把共同使用matching4的收益归功于新增网络或噪声方案。',small)
        if own:compare_table(sorted(own,key=lambda c:c['R_focus']))
    interaction_arms=('match4','match4_shape_affine','match4_heading_p1','match4_affine_heading_p1')
    interaction_rows={arm:[r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,2048,4096)] for arm in interaction_arms}
    if all(len(values)==2 and {r['eval_seed'] for r in values}=={2817,3817} for values in interaction_rows.values()):
        page('组合实验与speed误差来源')
        p('组合与两个单项改动：同一TRAIN、相同2048步和4096场景，下表为两个GEN指标的算术均值。组合收益不能由单项收益相加推断。',small)
        table(['配置','angle JSD','length JSD','width JSD','speed JSD'],[
            [LABELS[arm],*[f'{fmean(r[k] for r in interaction_rows[arm]):.4f}' for k in ('ang_dev_jsd','length_jsd','width_jsd','speed_jsd')]]
            for arm in interaction_arms],[203,74,74,74,74])
    diagnostic_path=ART/'histogram_diagnostics.json'
    if diagnostic_path.exists():
        diagnostic=read(diagnostic_path)
        assert diagnostic['observations_sha256']==sha(ART/'observations.json')
        selected_hist=[r for r in diagnostic['results'] if r['budget']==2048 and r['arm'] in ('roadK8','match4','prior_speed','prior_shape_speed')]
        entries=[]
        for r in selected_hist:
            speed=r['features']['speed_jsd'];bands=speed['speed_bands']
            entries.append([LABELS.get(r['arm'],r['arm']),f'{speed["mean_jsd"]:.4f}',
                f'{100*bands[0]["generated_fraction"]:.2f}% / {100*bands[0]["gt_fraction"]:.2f}%',
                f'{100*bands[3]["fraction_of_jsd"]:.1f}%'])
        table(['2048更新模型','speed JSD','0-1m/s 生成 / GT','15-30m/s的JSD占比'],entries,[171,75,145,108])
        p('先分别计算每个GEN的分箱JSD贡献，再取平均；不是合并直方图后重新计算JSD。某速度区间占比正确仍可能在区间内部有误差。停车总量已接近GT时，单独改变初始停车分布不必然改善高速尾部。',small)
    p('训练属性源中约14.3%的speed恰为0；经验分位变换使对应标量出现原子质量，其他特征仍有随机性。它可能改变优化难度，但目前没有隔离实验能证明这是结果变差的原因。',small)

    marginal_path=ART/'affine_heading_marginal_errors.json'
    if marginal_path.exists():
        marginal=read(marginal_path)
        page('四组早期模型的分布误差')
        p('纵轴为生成概率减GT概率，每个bin按百分点计。两种GEN先分别归一化，曲线取均值；阴影只表示两次生成的范围，不是置信区间。显示常见值区间，原生JSD仍使用完整分箱。',small)
        image(ART/'affine_heading_marginal_errors.png',height=362)
        table(['配置','相对车道朝向在±5度内：GEN1 / GEN2'],[
            [LABELS[r['arm']],' / '.join(f'{100*x:.2f}%' for x in r['within_5_degrees_generated'])]
            for r in marginal['angle_peak_records']],[250,249])
        p('同一评价GT的对应比例为89.64%。组合改善角度中心峰，同时加重部分尺寸区间误差。这里的角度是生成位置处相对最近车道的偏差，与训练的clean heading重建loss含义不同。',small)

    if (ART/'train_eval_attribute_diagnostic.json').exists():
        page('TRAIN与完整50K数据分布差异')
        distribution=read(ART/'train_eval_attribute_diagnostic.json')
        image(ART/'train_eval_attribute_diagnostic.png',height=172)
        group=distribution['groups']['all_vehicle_targets']
        table(['车辆属性','TRAIN相对50K GT的JSD','模型目标门槛'],[
            [name,f'{group["features"][name]["train_vs_all_GT_jsd"]:.5f}',f'{limits[name+"_jsd"]:.5f}']
            for name in ('length','width','speed')],[122,197,180])
        p('这里使用8192个训练场景中的64535辆车，对照完整50K的GT车辆；没有生成场景，也没有拟合评价分布。TRAIN包含ego，和完整评价口径一致。场景构成与样本量仍不同。')
        p('三项TRAIN边缘分布与评价GT的差异均低于对应目标门槛；这支持优先检查模型生成误差，而不是把当前尺寸差距直接归因于严重的数据偏移。该结果不是不可突破的误差下限，也不能从模型JSD中相减。')
        p('另一个待验证假设：经验属性起点使用全部TRAIN agents，而当前ego尺寸固定、不需要生成。TRAIN车辆中约12.7%是ego，且尺寸集中在固定值；生成非ego时混入该部分可能改变起点混合比例。目前尚未隔离验证，不能据此解释已观察到的全部退步。',small)

    if (ART/'generated_size_source_comparison.png').exists():
        page('训练误差下降，但生成尺寸分布更集中')
        image(ART/'generated_size_source_comparison.png',height=240)
        diagnostics=read(ART/'histogram_diagnostics.json')
        size_rows=[r for r in diagnostics['results'] if r['budget']==2048 and r['seed']==4817
                   and r['scenes']==4096 and r['arm'] in ('match4','prior_speed','prior_shape_speed')]
        table(['模型','训练 L / W loss','生成 length / width JSD'],[
            [LABELS[r['arm']],
             ' / '.join(f'{r["training_last256"]["means"]["train/component/"+field+"_loss"]:.4f}' for field in ('length','width')),
             ' / '.join(f'{r["features"][field+"_jsd"]["mean_jsd"]:.4f}' for field in ('length','width'))]
            for r in size_rows if r.get('training_last256')],[178,146,175])
        p('训练loss取最后256次更新中对应component日志的平均值。初始分布变化会改变回归难度；更低的逐样本误差不保证生成边缘分布更准确。以上比较仍只证明这个早期训练预算下的现象。')
        p('宽度2.0-2.1m区间是明显偏差来源：经验speed+尺寸初始分布产生的车辆在该区间更集中。它不是主要由生成大量极小宽度车辆引起。分布损失实验的生成指标单独列出，不直接把低回归loss当作成功。')

    gradient_path=ART/'shape_affine/component_gradient_probe/summary.json'
    if gradient_path.exists():
        gradient=read(gradient_path)
        page('尺寸输出尺度与各项梯度的关系')
        p('额外只读检查：2048步的两个checkpoint，原始训练权重；128个相同TRAIN场景，每批32场景，CPU计算。匹配后的目标、噪声、时间哈希一致，没有参数更新。')
        p('当前TRAIN统计：length的均值约4.296、标准差约1.538；width的均值约1.958、标准差约0.494。仿射输出为 mean + std * raw，同一物理输出下传给raw的梯度多乘训练标准差std。它不改变x0目标，但会改变参数空间中的优化尺度和初始物理输出分布；目前未单独隔离这两种影响。')
        before=gradient['means']['match4']['components']
        after=gradient['means']['match4_shape_affine']['components']
        table(['损失项','原梯度范数','仿射梯度范数','原投影份额','仿射投影份额'],[
            [key,f'{before[key]["global_grad_norm"]:.3f}',f'{after[key]["global_grad_norm"]:.3f}',
             f'{100*before[key]["dot_over_total_squared"]:.1f}%',f'{100*after[key]["dot_over_total_squared"]:.1f}%']
            for key in ('position','heading','length','width','velocity','type','collision')],[103,99,99,99,99])
        p('这组样本上，width的梯度范数明显下降，length在总梯度方向上的相对投影增加，支持继续检查属性优化平衡。训练loss和生成分布不是同一个量，以上没有隔离因果，也不是实际GPU/B512训练全过程的梯度统计。')
        p('投影份额定义为 dot(g_component, g_total) / sum(g_total**2)，逐批计算后取均值。因梯度抵消，它可能为负或大于100%；不能解读为独立可加的性能贡献。',small)
        four_way_path=ART/'affine_heading_gradient_probe/summary.json'
        if four_way_path.exists():
            four_way=read(four_way_path)
            p('同一批TRAIN输入的四组补充诊断：下表是各损失项梯度范数。heading时间幂变化已包含在相应项权重中，不能当成相同loss的误差比较。',small)
            table(['配置','heading','length','width','velocity'],[
                [LABELS[arm],*[f'{four_way["means"][arm][field]["global_grad_norm"]:.3f}' for field in ('heading','length','width','velocity')]]
                for arm in interaction_arms],[203,74,74,74,74])
            p('组合确实改变了梯度平衡，但更大的length梯度没有带来更好的length JSD；最终选择仍依据原生生成评价，不能仅靠梯度大小推荐。',small)

    page('更长训练、第二训练种子与完整50K')
    extra_plan=ART/'long_confirmation/protocol.json'
    if extra_plan.exists():
        p('为减少短跑误判，预登记了额外等预算确认：现有训练和采样对照结束后，若剩余时间足够，延长roadK8及14336步选出的候选至18432-24576步，并再次完整评价50K。预算同时为两臂训练、双GEN评价和报告预留时间；是否实际执行以完成记录为准。',small)
    baseline_rows=[r for r in rows if r['variant']=='roadK8' and r['seed']==4817
                   and r['num_samples']==4096 and r['budget']>=6144]
    baseline_budgets=sorted({r['budget'] for r in baseline_rows})
    if len(baseline_budgets)>1:
        p('同一roadK8训练轨迹：固定TRAIN4817、4096场景，两个GEN的指标算术均值。',small)
        trajectory=[]
        for budget in baseline_budgets:
            group=[r for r in baseline_rows if r['budget']==budget]
            assert sorted(r['eval_seed'] for r in group)==[2817,3817]
            trajectory.append([budget,*[f'{fmean(r[k] for r in group):.5f}' for k in
                ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')]])
        table(['更新数','speed','angle','length','width'],trajectory,[99,100,100,100,100])
        p('这里展示增加训练预算的变化，不是候选相对control的收益，也不是合并直方图后重新计算的JSD。',small)
    longer=sorted([c for c in comparisons if c['budget']>6144 and c['scenes']==4096],key=lambda c:(c['budget'],c['R_focus']))
    if (ART/'match4_14336_budget_comparison.json').exists():
        p('matching4在14336更新的配对结果出现收益反转：相对roadK8，两个GEN平均speed改善17.0%，但angle、length、width分别恶化8.9%、18.0%、33.7%；后三项在两个GEN中均退步。其6144步的angle优势没有保持，length也比自身6144步更差。这里仍为4096场景、固定训练顺序的中期证据，不能视为128epoch结论。',small)
    coupling_path=ART/'matching_assignment_diagnostic/result.json'
    if coupling_path.exists():
        coupling=read(coupling_path)
        assert coupling['status']=='PASS' and coupling['source_sha256']==sha(ROOT/'tools/diagnose_init_matching_tradeoff_20261014.py')
        changes=coupling['match4_over_roadK8']
        p(f'匹配机制诊断：256个TRAIN场景、四组相同源噪声，matching4改变了{100*coupling["changed_assignment_fraction"]:.1f}%的非ego分配。归一化起终点平方距离比值：position={changes["position"]:.3f}、heading弦长={changes["heading_chord"]:.3f}、length={changes["length"]:.3f}、width={changes["width"]:.3f}、speed={changes["speed"]:.3f}。它以更长的位置/尺寸路径换取更短的朝向/速度路径；尺寸端变化较小，不能单独解释最终JSD差距。该诊断未调用网络预测，亦非生成指标。',small)
        if 'match2_over_roadK8' in coupling:
            changes2=coupling['match2_over_roadK8']
            p(f'权重2的对应诊断：分配变化{100*coupling["changed_assignment_fractions"]["match2"]:.1f}%，position路径平方距离比={changes2["position"]:.3f}、length={changes2["length"]:.3f}、width={changes2["width"]:.3f}、speed={changes2["speed"]:.3f}。它减轻了权重4的路径取舍，提供继续观察v2h2的机制依据，但不能据此预测最终生成质量。原权重4结果在扩展诊断中完全复现。',small)
    if longer:compare_table(longer)
    second=[c for c in comparisons if c['seed']==5817]
    if second:p('第二TRAIN种子；其预算须与对应control一致。',small);compare_table(second)
    full=[c for c in comparisons if c['scenes']==50000]
    if full:p('完整50K，两种GEN，N20原采样。',small);compare_table(full)
    else:p('没有通过验收的完整50K配对结果；不能据此声称完成大样本确认。')
    p('训练预算增加产生的新结果与同checkpoint扩大评价样本的结果属于不同证据。较短预算模型即使完整评价50K，也不等于完整128 epoch训练的性能。')
    variability_path=ART/'generation_variability_diagnostic.json'
    if variability_path.exists():
        variability=read(variability_path)
        matches=[r for r in variability['results'] if (r['variant'],r['TRAIN'],r['scenes'])==('roadK8',4817,4096)]
        if matches:
            latest=max(matches,key=lambda r:r['updates'])
            v=latest['features']['speed_jsd']
            p('评价波动检查：roadK8在'+str(latest['updates'])+'次更新、相同4096场景上，两个预先指定GEN的speed JSD分别为'+
              ' / '.join(f'{x:.5f}' for x in v['per_GEN'])+
              f'；生成速度直方图之间的总变差为{100*v["inter_GEN_total_variation"]:.2f}%。单种子小幅改善不足以支持晋级，仍使用配对GEN并扩大到50K。',small)
            p('另存独立多项分布抽样参照，仅用于解释有限样本波动量级；真实场景内agents相互依赖，该参照不是模型置信区间、显著性检验或误差下限。',small)
    full_targets=[r for r in target_progress['cells'] if r['scenes']==50000]
    if full_targets:
        p('各GEN独立检查全部七项门槛；下面不使用平均值替代。',small)
        table(['模型 / GEN','已过 / 7','最大目标比值','尚未达标'],[
            [LABELS.get(r['variant'],r['variant'])+' / '+str(r['GEN']),r['passed_metrics'],
             f'{r["worst_target_ratio"]:.3f}',' / '.join(target_labels[k] for k in r['failed_metrics']) or '全部低于']
            for r in full_targets],[160,55,75,209])

    latest_pair_path=ART/'paired_latest_attribute_diagnostic.json'
    if latest_pair_path.exists():
        latest_pair=read(latest_pair_path)
        if latest_pair['updates']>6144 or latest_pair['scenes']==50000:
            assert latest_pair['observations_sha256']==sha(ART/'observations.json')
            assert latest_pair['image_sha256']==sha(ART/'paired_latest_attribute_errors.png')
            page('较长预算的配对生成分布')
            p(f'对照臂{latest_pair["candidate"]}与roadK8，同为{latest_pair["updates"]}更新、{latest_pair["scenes"]}场景、TRAIN4817、N20与EMA。本图按指定的训练对照臂绘制，不据此另行挑选最终候选。',small)
            image(ART/'paired_latest_attribute_errors.png',height=285)
            entries=[]
            for arm,record in latest_pair['records'].items():
                for i,seed in enumerate(latest_pair['GEN']):
                    entries.append([LABELS.get(arm,arm)+' / '+str(seed),*[
                        f'{record["features"][feature]["per_GEN"][i]["jsd"]:.5f}'
                        for feature in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')]])
            table(['模型 / GEN','speed','angle','length','width'],entries,[203,74,74,74,74])
            p('曲线先按每个GEN归一化，再取均值；阴影仅为两次生成的范围，不是置信区间。表中JSD分别由完整分箱计算，不合并两种GEN的直方图。横轴只放大常见区间；未显示的尾部仍计入JSD。',small)

    midpoint_path=ART/'midpoint_attribute_diagnostic.json'
    if midpoint_path.exists():
        midpoint=read(midpoint_path)
        page('6144步确认：尺寸峰值与速度尾部')
        image(ART/'midpoint_attribute_errors.png',height=285)
        p('同一TRAIN、相同6144更新和4096场景，曲线为两个GEN的平均概率误差。灰色roadK8、蓝色matching4、绿色matching4加尺寸仿射。阴影表示两种GEN的范围，不是置信区间。',small)
        def find_band(arm,feature,bounds):
            return next(v for v in midpoint['records'][arm]['features'][feature]['bands'] if v['interval']==bounds)
        candidate='match4_shape_affine'
        parent_means={arm:{key:midpoint['records'][arm]['features'][key]['mean_jsd'] for key in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')} for arm in ('roadK8','match4',candidate)}
        table(['6144更新 / 同预算参考','speed比值','angle比值','length比值','width比值'],[
            [LABELS.get(parent,parent),*[f'{parent_means[candidate][key]/parent_means[parent][key]:.3f}' for key in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')]]
            for parent in ('roadK8','match4')],[203,74,74,74,74])
        p('拆分归因：相对单独matching4，尺寸仿射在6144步改善angle/length/width，但speed略差；相对roadK8的speed收益不能全部归因于新增尺寸参数化。2048步的speed优势没有完全保持，仍需更长训练确认。',small)
        length=find_band(candidate,'length_jsd',[5.7,5.9])
        width=find_band(candidate,'width_jsd',[2.,2.1])
        speed=find_band(candidate,'speed_jsd',[15.,30.])
        p(f'尺寸仿射版本在5.7-5.9m的生成占比{length["generated_mass_mean"]*100:.2f}%，GT为{length["GT_mass"]*100:.2f}%；这两个0.1m箱约占length JSD的{length["fraction_of_mean_jsd"]*100:.1f}%。这说明常见车长区间的分布形状仍有明显差距。')
        p(f'width的2.0-2.1m区间生成占比{width["generated_mass_mean"]*100:.2f}%，GT为{width["GT_mass"]*100:.2f}%。speed在15-30m/s区间贡献{speed["fraction_of_mean_jsd"]*100:.1f}%的JSD；停车区间并非主要误差来源。',small)
        table(['最后256更新','总梯度范数均值','发生clip比例'],[
            [LABELS.get(arm,arm),f'{midpoint["records"][arm]["training_last256"]["means"]["generator_grad_norm_before_clip"]:.3f}',
             f'{midpoint["records"][arm]["training_last256"]["fraction_clipped"]*100:.1f}%'] for arm in ('roadK8','match4','match4_shape_affine')],[245,130,124])
        p('尺寸仿射版本的梯度范数更小，但以上只是实际训练窗口的描述。不能将clip比例直接解释为Adam有效学习率，也不能据此认定JSD改善的因果机制。',small)

    page('20/24步及guidance分离实验')
    sampler=ART/'samplers/comparisons.json'
    if sampler.exists():
        data=read(sampler)
        table(['模型 / 采样','R_focus','S / A / L','R_jsd'],[
            [LABELS.get(r['arm'],r['arm'])+' / '+r['mode'],f'{r["score"]["R_focus"]:.3f}',
             ratios(r['score']),f'{r["score"]["R_jsd"]:.3f}'] for r in data],[191,58,181,69])
    else:p('没有完成采样对照；保持原20步，不能宣称24步改善。')
    p('采样对照使用同一checkpoint、同一场景、同一GEN。当前欧氏分量的采样向量场使用分母max(t,0.05)，均匀24步最后预测时间1/24小于0.05，因此最后更新会保留部分状态与x0预测之间的残差。terminal05保持最后预测时间为0.05，以区分更多积分步与末步处理差异。')
    terminal_path=ART/'sampler_terminal_diagnostic.json'
    if terminal_path.exists():
        terminal=read(terminal_path)
        table(['采样方案','模型调用数','最后预测时间','末步残差保留系数'],[
            [r['mode'],r['model_calls'],f'{r["last_model_time"]:.6f}',f'{r["last_step_residual_factor"]:.6f}'] for r in terminal['rows']],
            [179,85,110,125])
        p('1/6指最后一步状态相对最后x0预测的残差系数，不是保留1/6的原始高斯噪声。只有恒定x0预测器这一特例，整个轨迹对原始偏差的累计系数才是1/144；真实网络的x0随状态和时间变化。以上公式不构成N24指标改善的证据。',small)
        p('圆周x0 heading使用角度积分，分母为t，不受0.05这一欧氏下限限制。当前separate类型最终取最后一次clean logits的argmax；不能把它解释为直接对残留连续类型状态分类。',small)
    p('guidance调整采样时的几何；与matching训练权重分开验证。其碰撞改善不替代JSD改善，也不把位置推开造成的指标变化归因于模型学习。')
    expanded=ART/'samplers/full50000_comparison.json'
    if expanded.exists():
        value=read(expanded);p('完整50K采样确认：'+str(value['choices']))
        p(f'相应候选相对control：R_focus={value["score"]["R_focus"]:.4f}, R_jsd={value["score"]["R_jsd"]:.4f}, R_attr={value["score"]["R_attr"]:.4f}。')

    calibration_dir=ART/'map_architecture/size_calibration'
    if not (calibration_dir/'protocol.json').exists():calibration_dir=ART/'size_calibration'
    calibration_protocol=calibration_dir/'protocol.json'
    if calibration_protocol.exists():
        page('训练集尺寸校准：独立后处理对照')
        calibration_status=read(calibration_dir/'status.json')
        p('本实验修改实际生成车辆框的length和width，再交给原评价器；属于模型加后处理，不混入纯模型训练排名。当前状态：'+calibration_status['status'])
        p('拟合仅用原8192个TRAIN场景及同一checkpoint在这些场景上的生成结果，GEN6817、EMA、20步。分别筛选非ego车辆，固定257个log分位点建立单调映射；ego、非车辆和无效尺寸保持原值。不使用验证标签拟合、调参或逐场景修正。',small)
        p('先用双GEN在4096场景比较同一checkpoint的原始/校准输出。两种GEN的length与width均改善、已通过门槛保持、其余JSD不超过原值1.03倍时，才做完整50K确认。最终配置变化或checkpoint变化需重新用TRAIN拟合，不能复用旧映射。',small)
        p('验证：六项CPU单测通过；真实模型两场景证明原生encoder和正式rollout采样一致，已知尺寸映射只改变实际输出框，位置、heading、速度、类型和随机数状态完全不变。该验证不是生成指标改善的证据。',small)
        for filename,scope in [('quick_comparison.json','4096场景'),('full50000_comparison.json','完整50K')]:
            path=calibration_dir/filename
            if not path.exists():continue
            calibration_audit=read(calibration_dir/'native_metric_audit.json')
            assert calibration_audit['status'] in ('PASS','NATIVE_METRICS_PASS_OUTPUT_ISOLATION_REQUIRES_REVIEW'),calibration_audit['status']
            assert calibration_audit['comparison_sha256'][filename]==sha(path)
            record=read(path);p(scope+'：原始输出与TRAIN校准输出',heading)
            if calibration_audit['status']!='PASS':
                p('各项原生指标已由histogram/count独立重算通过，但部分非尺寸计数或histogram未与原始评价精确一致。输出隔离仍需复查；以下数值是有效实际测量，不能把非尺寸差异归因于尺寸映射。',small)
            entries=[]
            for calibrated in record['rows']:
                raw=next(r for r in record['raw_rows'] if r['eval_seed']==calibrated['eval_seed'])
                for label,row in [('raw',raw),('calibrated',calibrated)]:
                    entries.append([str(row['eval_seed'])+' / '+label,*[f'{row[k]:.5f}' for k in ('length_jsd','width_jsd','ang_dev_jsd','speed_jsd','collision_rate')]])
            table(['GEN / 输出','length','width','angle','speed','collision %'],entries,[139,72,72,72,72,72])
            if 'promote_full50K' in record:p('预先规则是否晋级50K：'+str(record['promote_full50K']),small)
        if not (calibration_dir/'quick_comparison.json').exists():
            p('尚无实际校准评价结果，不能声称该方法改善length或width。')
        p('TRAIN与VAL的属性分布可能存在差异；独立校准边缘尺寸分布也可能影响尺寸与场景的联合关系。碰撞率使用校准后的车辆框重新计算，不能沿用原始碰撞率。',small)
        if (calibration_dir/'full50000_shape_histograms.png').exists():
            page('尺寸分布的实际变化：固定GEN2817')
            plot_proof=read(calibration_dir/'full50000_shape_histograms.json')
            assert plot_proof['figure_sha256']==sha(calibration_dir/'full50000_shape_histograms.png')
            assert plot_proof['comparison_sha256']==sha(calibration_dir/'full50000_comparison.json')
            image(calibration_dir/'full50000_shape_histograms.png',height=180)
            p('图示同一6144 checkpoint的完整50K、GEN2817；曲线按完整原始histogram归一化，显示主要尺寸区间，图例JSD使用全部原评价bins。raw与calibrated是两次GPU重新生成的实际测量。',small)
            p('校准后length约0.1049、width约0.0879，第二GEN约0.1059/0.0820，也低于VW尺寸门槛。shape JSD对细bin内的峰值质量很敏感；原始框并非整体尺寸完全错误，峰值比例与尾部偏差仍可造成明显的缩放后JSD差距。',small)
            p('这支持进一步研究TRAIN统计驱动的尺寸表示或分布建模，不支持把后处理收益当作网络已学会正确条件分布。当前angle、speed、lat仍未达标；不同checkpoint必须重新在TRAIN拟合映射。',small)

    architecture=ART/'map_architecture'
    if (architecture/'source_application.json').exists():
        page('新增地图邻接与独立denoiser参数')
        application=read(architecture/'source_application.json')
        assert application['status']=='PASS'
        proof=read(architecture/'prepared_preflight.json')
        p('用户进一步允许调整map encoder的radius、neighbor、层数、输出mask，以及map/denoiser各自的hidden dim、频带数、heads和head dim。新实现解除地图输入固定128维的限制，并增加denoiser独立结构选项。')
        p('默认行为检查：旧权重严格加载、相同初始化张量、真实TRAIN2输出和RNG完全一致；40项相关测试通过。64维两层可训练地图与192维denoiser也完成真实前后向。结构检查证明选项可用，不能作为生成质量证据。',small)
        table(['CPU结构检查','生成器参数','地图输入维度'],[
            [r['name'],f'{r["generator_parameters"]:,}',r['map_feature_dim']] for r in proof['variants']], [263,130,106])
        p('原来的256维denoiser使用128个Fourier频带、8 heads、head dim32；地图编码器则使用128维、64频带、8 heads、head dim16。新字段分别控制这些结构，null保留原行为。',small)
        p('本轮原生GPU训练保持20步、657原目标、冻结AIRL地图和新control；地图radius40/neighbor64是联合邻接设置，频带32是单独denoiser设置。地图层数/维度发生结构不兼容时需走可训练地图分支，严格加载保留结构校验。',small)
        reproduction=architecture/'native_control_reproduction.json'
        if reproduction.exists():
            p('新旧control虽有相同初始张量、训练顺序和评价成员，2048步后仍出现指标差异：angle两种GEN相对旧值约为0.862/0.922。该现象不能单独归因于代码变更；本阶段始终用新control比较，避免跨轮波动被计作收益。CPU默认输出/RNG严格一致不代表训练后的GPU结果也严格一致。',small)
        if (architecture/'first_step_control_numeric_review.json').exists():
            p('第一批GPU记录的总loss同为40.335548，但heading加权分量已有约1.05e-4差值，生成器梯度范数相差0.0516/925.3。差异在首个前后向中就存在；这些日志尚不能区分具体数值来源，也不证明代码导致长期指标变化。',small)
        observations_path=architecture/'observations.json'
        if observations_path.exists() and read(observations_path):
            architecture_audit=read(architecture/'native_metric_audit.json')
            architecture_schedule=read(architecture/'training_schedule_audit.json')
            assert architecture_audit['status']=='PASS' and architecture_audit['observations_sha256']==sha(observations_path)
            assert architecture_schedule['status']=='PASS' and architecture_schedule['observations_sha256']==sha(observations_path)
            ac=read(architecture/'comparisons.json')['comparisons']
            p(f'当前独立审计{architecture_audit["unique_actual_cells"]}个新阶段单元、{architecture_audit["checks"]}项检查。',small)
            table(['新阶段 / 更新 / 场景','speed比值','angle比值','length比值','width比值'],[
                [r['arm']+' / '+str(r['budget'])+' / '+str(r['scenes']),*[f'{r["ratios"][k]:.3f}' for k in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')]]
                for r in ac],[203,74,74,74,74])
            full_rows=[r for r in read(observations_path) if r['num_samples']==50000]
            if full_rows:
                from tools import score_init_sd_vw_targets_20261014 as score_targets
                table(['完整50K / GEN','speed','angle','length','width','已过/7'],[
                    [r['variant']+' / '+str(r['eval_seed']),
                     *[f'{r[k]:.5f}' for k in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')],
                     score_targets.score(r)['passed_metrics']]
                    for r in full_rows],[183,67,67,67,67,48])
                p('这是6144更新checkpoint的完整评价样本，不是完整128epoch训练结果。逐种子的七项门槛检查单独保存；表中的平均比值不能替代单项达标。',small)
        if (architecture/'selection.json').exists():
            chosen=read(architecture/'selection.json')
            p('新阶段中期确认候选：'+chosen['candidate']+'；它是已完成初筛中的预登记规则选择，尚不等于值得完整训练的推荐。',small)
            if chosen.get('deferred_arms'):
                later_done=(architecture/'mask_followup/result.json').exists()
                missing=[a for a in chosen['deferred_arms'] if not (a=='lane_only_queries' and later_done)]
                p('为保留配对6144与完整50K预算，当时推迟了部分短训。'+
                  ('lanes_only后续已补跑，结果单独见后续mask章节。' if later_done else '')+
                  ('仍未执行GPU初筛：'+', '.join(missing)+'，仅完成CPU前后向检查。' if missing else ''),small)
        if (architecture/'budget_handoff.json').exists():
            p('预算调度调整仅暂停协调器，让正在运行的频带实验完整结束后再交接；原生训练子进程、模型、loss、种子、评价与候选排序均未改变。',small)
        context_path=architecture/'map_context_train_diagnostic.json'
        if context_path.exists():
            context=read(context_path)
            assert context['status']=='PASS'
            page('地图邻接改变了多少实际条件信息')
            p('下面使用16个固定TRAIN场景、CPU上的真实冻结AIRL encoder前向。地图为一层、ego-local正方形半边长50m；不使用评价标签，也不测生成质量。',small)
            table(['输出mask / radius / K','query节点','平均邻居','邻居平均距离m','特征cosine'],[
                [r['mask']+' / '+str(int(r['radius']))+' / '+str(r['neighbor_cap']),
                 r['query_nodes'],f'{r["mean_neighbors"]:.1f}',f'{r["mean_neighbor_distance"]:.1f}',
                 f'{r["mean_feature_cosine_vs_r20_k30"]:.3f}' if 'mean_feature_cosine_vs_r20_k30' in r else '同mask参考']
                for r in context['rows']],[192,62,76,87,82])
            p('road_edges的平均邻居由17.9增加到47.9，平均邻居距离由13.0m增加到23.1m；同一query的冻结特征平均cosine约0.698。增大图范围明显改变输入特征，不能当作等价的计算设置。',small)
            p('一种待验证解释是：新邻接与AIRL预训练时的上下文不同，增加远处节点也可能稀释局部道路信息。该诊断没有证明它导致任何JSD退步。输出mask同样改变下游可选query集合；地图attention的source节点在此比较中始终为1886。CPU近邻距离并列时的选择可能与GPU不同。',small)

    mask_followup=ART/'map_architecture/mask_followup'
    if (mask_followup/'queue_plan.json').exists():
        page('单独改变输出mask的后续初筛')
        mask_status=read(mask_followup/'status.json')
        p('固定候选lanes_only：其余结构、loss、优化器、TRAIN顺序和GEN沿用新结构阶段control。仅在优先队列结束、保守时长可在22:12前完成时启动；不影响原配对6144/full50K及尺寸诊断。为减少数十秒余量造成的整体弃跑，仅将空闲队列的截止延后两分钟，保留18分钟报告时间，未打断原生子任务。当前状态：'+mask_status['status'],small)
        if (mask_followup/'result.json').exists():
            result=read(mask_followup/'result.json')
            proof=read(mask_followup/'native_metric_audit.json')
            assert proof['status']=='PASS' and proof['observations_sha256']==sha(mask_followup/'observations.json')
            p('本轮新增两个实际评价单元，复用同源码阶段已有的2048 control两个单元；不把复用control重复计为新实验。两种GEN对照仍只是一种TRAIN种子的短期初筛。',small)
            compare_table([dict(arm='lane_only_queries',seed=4817,budget=2048,**result['score'])])
            table(['GEN','speed','angle','length','width'],[
                [r['eval_seed'],*[f'{r[k]:.5f}' for k in ('speed_jsd','ang_dev_jsd','length_jsd','width_jsd')]]
                for r in result['raw_rows']],[83,104,104,104,104])
            p('没有对应6144、完整50K或第二TRAIN确认，不能据此宣称优于现有完整训练结果。是否值得作为下一轮完整训练，另依据4096配对与第一页最终判断。',small)
        else:
            p('尚无已完成GPU结果；这里只记录排队或时间限制，CPU功能检查不替代效果验证。')
        midpoint=ART/'map_architecture/mask_midpoint'
        if (midpoint/'comparisons.json').exists():
            mc=read(midpoint/'comparisons.json')['comparisons']
            if mc:
                proof=read(midpoint/'native_metric_audit.json')
                assert proof['status']=='PASS' and proof['observations_sha256']==sha(midpoint/'observations.json')
                p('4096更新配对确认',heading)
                compare_table(mc)
                p('control使用此前真实6144训练过程中保存的step4096状态，候选仅恢复自己的2048 optimizer/EMA/RNG至4096。没有额外训练control，比较保持相同累计曝光和同一源码。此阶段仍为4096场景，未做完整50K或第二TRAIN确认。',small)
            else:
                p('4096更新确认尚无完整候选/control对，不能据单边结果计算收益。',small)
        if (ART/'map_architecture/lane_only_full_config_preflight.json').exists():
            p('新完整recipe已准备：init_diffusion_657_n20_lane_only_full；20步、G1约808万参数、冻结原AIRL map、batch512、128epochs、workers8、每8epoch验证完整50K。训练与评价Hydra组合及真实TRAIN2 CPU前后向通过；仅改变返回query的map类型，保留原loss和采样。是否推荐启动，以第一页最终决策为准。',small)

    repeatability=ART/'map_architecture/gpu_repeatability_result.json'
    if repeatability.exists():
        page('相同输入的GPU前后向重复检查')
        result=read(repeatability)
        p('所有原生队列结束后，在同一个当前模型、16个TRAIN场景、相同seed与FP32/matmul medium上连续做三次前后向。每次清空梯度并重置随机数；不做optimizer更新，不评价生成场景。',small)
        table(['重复','loss严格相同','grad严格相同','最大grad绝对差','不等grad元素'],[
            [r['repeat'],r['losses_exactly_equal'],r['gradients_exactly_equal'],
             f'{r["max_abs_gradient_difference"]:.7g}',r['unequal_gradient_elements']]
            for r in result['rows']],[66,103,103,127,100])
        p('此检查只描述当前代码在同进程GPU上的重复性，不能单独定位新旧source差异，也不能解释2048或6144后的指标差距。随机数状态hash和逐项gradient hash保存在原始JSON。',small)

    page('W&B完整训练与新run进展')
    selected=[manifest[rid] for rid in ('iu3fa4gx','0vftb3cc','nnsjk5vl','doke6bbr','ehnykqc4','efzhwmu3','wpya0vrc','cchgfuhr','kw1ayqd5','yh5n38xl','bq7z12zy') if rid in manifest]
    table(['Run / 状态','验证epoch','speed','angle','length'],[
        [r['id']+' / '+r['state'],r['last_validation_epoch'],*[f'{r["last_validation_metrics"][k]:.5f}' for k in FOCUS]]
        for r in selected if r['last_validation_metrics']], [174,65,87,87,86])
    p('finished仅表示该run已结束；停在epoch23的run不是完成128 epoch训练。表中按实际验证epoch比较，完整成员检查与训练时长分别确认。',small)
    for r in selected:
        if r['state']=='running' and not r['last_validation_metrics']:
            p('新运行 '+r['id']+' / '+r['name']+' 已出现，但尚无完整验证指标；不参与数值排名。',small)
    image(ART/'early_final_rank_review.png',height=191)
    p('相关性图来自最初读取的6个已完成run，仅为观察性回顾。硬件、源代码、seed等有混杂；第15epoch与最终排名相关性约.09，不构成通用预测定律。最新run表来自报告标注的W&B快照，未完成run不当作最终结果。',small)
    p('ehnykqc4同时改变邻居数、时间embedding、normalization、末步处理等，且未开启sd_require_full_set，不能据其单个run归因某项设置，或宣称已验证完整50K成员一致。',small)

    page('W&B新run的对应基线与完整训练复现')
    remote_parent_path=ART/'wandb/current_parent_comparisons.json'
    if remote_parent_path.exists():
        remote_parent=read(remote_parent_path)
        assert remote_parent['source_analysis_sha256']==sha(ART/'wandb/latest_analysis.json')
        relevant=[r for r in remote_parent['comparisons'] if r['parent'] in ('0vftb3cc','nnsjk5vl') and r['last_shared_epoch'] and r['candidate'] in ('efzhwmu3','wpya0vrc','cchgfuhr','kw1ayqd5','yh5n38xl','bq7z12zy')]
        parent_labels={'0vftb3cc':'roadK8','nnsjk5vl':'match4 + guidance'}
        table(['候选 / 共同epoch / 参考','R_focus','S / A / L 比值','W 比值'],[
            [r['candidate']+' / '+str(r['epoch'])+' / '+parent_labels[r['parent']],f'{r["R_focus"]:.3f}',ratios(r),f'{r["ratios"]["width_jsd"]:.3f}']
            for r in relevant],[179,60,190,70])
        p('两组参考均取各自与候选共同的epoch，避免将相对matching4的改善误认为相对roadK8也改善。相同epoch尚不足以证明场景成员、硬件、GEN种子与源代码都匹配；远端多个改动并存，不能单独归因采样步数或匹配权重。',small)
        if (ART/'wandb/1848_architecture_attribution.json').exists():
            p('新五层v2h2相对三层v2h2同时改变网络深度、20→24步和power1.2/末步0.05时间网格；相对三层24步matching4又同时改变深度与匹配权重。后续收益只能先归于整套配置。远端24步还启用碰撞guidance，与本地无guidance的24步对照不同，不能直接视为同一采样方案。',small)
        if (ART/'wandb/2134_run_review.json').exists():
            p('轨迹复核：五层v2h2在epoch15的angle相对roadK8为0.843，到epoch23为1.028，早期优势已反转；affine的length/width相对roadK8从epoch39的1.231/1.401扩大为epoch47的1.293/1.494。这些是各自共享epoch的观察，不把更低speed或早期angle名次当作完整训练收益。最新值以上表为准。',small)
    p('完整训练命令：python src/run.py experiment='+decision['recommended_full_config']+' optimizer=default')
    if decision.get('recommended_eval_config'):
        p('完整评价配置：'+decision['recommended_eval_config'])
    for command in decision.get('commands',[]):p(command,small)
    p('完整recipe：num_workers=8、check_val_every_n_epoch=8、完整50K、R参考iu3fa4gx_full50000。map初始化使用initial_map_checkpoint，完整模型从头训练时ckpt_path为null；不能把旧AIRL完整checkpoint当作本轮denoiser续训。',small)
    p('复核资料：output/reports/init_diffusion_expanded_20261014 下的protocol、observations、comparisons、native_metric_audit、samplers、wandb及final_decision。首批损坏诊断日志已保留并排除，采用重新从头训练的有效control；源代码阶段变化和默认兼容性证明单独记录。',small)
    page('复核运行链接')
    p('以下链接对应报告使用的远端训练记录；报告中的实际指标以标注时间的只读快照为准。',small)
    for r in selected:p('W&B '+r['id']+': https://forge.coreweave.com/wandb/u3006612/bc/runs/'+r['id'],small)
    def footer(canvas,doc):
        canvas.setFont('STSong-Light',8);canvas.setFillColor(colors.HexColor('#657786'))
        canvas.drawString(48,24,'InitDiffusion expanded search | 2026-10-14');canvas.drawRightString(A4[0]-48,24,str(doc.page))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(output_path),title='InitDiffusion expanded search - 2026-10-14',pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    sources=['observations.json','comparisons.json','native_metric_audit.json','protocol.json','wandb/latest_analysis.json']
    sources += [p for p in ('parent_comparisons.json','histogram_diagnostics.json','full_recipe_preflight.json','wandb/current_parent_comparisons.json','all_seven_target_progress.json','prior_resume_evidence_correction.json','cyclic_resume_issue.json','cyclic_resume_regression.json','cyclic_chained_resume_regression.json','training_schedule_audit.json','samplers/full50000_comparison.json','train_eval_attribute_diagnostic.json','heading_time_weight_diagnostic.json','generation_variability_diagnostic.json','shape_affine/component_gradient_probe/summary.json','shape_affine/component_gradient_probe/results.json','long_confirmation/protocol.json','long_confirmation/status.json','long_confirmation/selection.json','affine_heading_gradient_probe/results.json','affine_heading_gradient_probe/summary.json','affine_heading_result.json','affine_heading_marginal_errors.json') if (ART/p).exists()]
    sources += [p for p in ('size_calibration/protocol.json','size_calibration/status.json','size_calibration/selection.json','size_calibration/unit_tests.json','size_calibration/native_metric_audit.json','size_calibration/native_output_parity.json','size_calibration/quick_comparison.json','size_calibration/full50000_comparison.json') if (ART/p).exists()]
    sources += [p for p in ('sampler_terminal_diagnostic.json','shape_affine_budget_comparison.json','wandb/recommended_run_config_check.json','midpoint_attribute_diagnostic.json','midpoint_attribute_errors.png','current_recommendation.json','recommended_shape_affine_preflight.json','generated_size_source_comparison.png','affine_heading_marginal_errors.png','early_final_rank_review.png','map_architecture/map_context_train_diagnostic.json') if (ART/p).exists()]
    sources += [p for p in ('paired_latest_attribute_diagnostic.json','paired_latest_attribute_errors.png') if (ART/p).exists()]
    sources += [p for p in ('training_recipe_crosscheck.json',) if (ART/p).exists()]
    sources += [p for p in ('match4_14336_budget_comparison.json',) if (ART/p).exists()]
    sources += [p for p in ('matching_assignment_diagnostic/result.json',) if (ART/p).exists()]
    sources += [p for p in ('wandb/1846_run_triage.json','wandb/1848_architecture_attribution.json') if (ART/p).exists()]
    sources += [p for p in ('wandb/2134_run_review.json',) if (ART/p).exists()]
    sources += [p for p in ('map_architecture/queue_plan.json','map_architecture/protocol.json','map_architecture/prepared_preflight.json','map_architecture/prepared_unit_tests.json','map_architecture/source_application.json','map_architecture/native_control_reproduction.json','map_architecture/observations.json','map_architecture/comparisons.json','map_architecture/native_metric_audit.json','map_architecture/training_schedule_audit.json','map_architecture/effective_options.json','map_architecture/selection.json','map_architecture/budget_handoff.json','map_architecture/budget_handoff_status.json') if (ART/p).exists()]
    sources += [p for p in ('map_architecture/size_calibration/protocol.json','map_architecture/size_calibration/status.json','map_architecture/size_calibration/selection.json','map_architecture/size_calibration/native_metric_audit.json','map_architecture/size_calibration/quick_comparison.json','map_architecture/size_calibration/full50000_comparison.json') if (ART/p).exists()]
    sources += [p for p in ('map_architecture/size_calibration/quick_isolation_review.json','map_architecture/size_calibration/full_isolation_review.json','map_architecture/size_calibration/full50000_shape_histograms.json','map_architecture/size_calibration/full50000_shape_histograms.png','map_architecture/full50000_review.json') if (ART/p).exists()]
    sources += [p for p in ('map_architecture/first_step_control_numeric_review.json','map_architecture/all_seven_target_progress.json','map_architecture/gpu_repeatability_plan.json','map_architecture/gpu_repeatability_result.json','map_architecture/mask_followup/queue_plan.json','map_architecture/mask_followup/status.json','map_architecture/mask_followup/protocol.json','map_architecture/mask_followup/observations.json','map_architecture/mask_followup/result.json','map_architecture/mask_followup/native_metric_audit.json','map_architecture/mask_followup/training_schedule_audit.json') if (ART/p).exists()]
    sources += [p for p in ('map_architecture/lane_only_full_config_preflight.json','map_architecture/mask_midpoint/protocol.json','map_architecture/mask_midpoint/status.json','map_architecture/mask_midpoint/observations.json','map_architecture/mask_midpoint/comparisons.json','map_architecture/mask_midpoint/native_metric_audit.json','map_architecture/mask_midpoint/training_schedule_audit.json','map_architecture/mask_midpoint/control_fallback_plan.json','map_architecture/mask_midpoint/control_fallback_status.json') if (ART/p).exists()]
    manifest_path=ART/('report_manifest.json' if output_path==OUT else 'report_preview_manifest.json')
    manifest_path.write_text(json.dumps(dict(pdf=str(output_path),pdf_sha256=sha(output_path),decision_path=str(decision_path),decision_sha256=sha(decision_path),sources={p:sha(ART/p) for p in sources},target_spec_sha256=sha(target_file),report_builder_sha256=sha(__file__),visual_QA='pending'),indent=2)+'\n')
    print(output_path)


if __name__=='__main__':main()
