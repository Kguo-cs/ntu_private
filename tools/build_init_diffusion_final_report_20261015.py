"""Build a verified three-page PDF from genuine final128 paired50K evidence."""
from pathlib import Path
import json,hashlib,datetime,re,html,statistics
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'output/reports/init_diffusion_continuous_20261014'
OUT=ROOT/'output/pdf/init_diffusion_final128_full50k_report_20261015.pdf'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(B/'omega025_final128_confirmation_v1/status.json')
assert state['status']=='OMEGA025_FINAL128_CONFIRMATION_COMPLETE'
target=read(ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json')
rows=[v for v in read(B/'omega025_final128_confirmation_v1/observations.json') if v['scenes']==50000]
rows.sort(key=lambda v:v['seed']);assert [v['seed'] for v in rows]==[2817,3817]
scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
snap=B/'full_checkpoint_snapshots/omega025_final128_20261015';meta=read(snap/'provenance.json');assert sha(snap/'model.ckpt')==meta['sha256'];assert meta['epoch']==127 and meta['global_step']==meta['ema_updates']==123392
reports=[]
for v in rows:
 assert v['checkpoint_sha256']==meta['sha256'] and v['checkpoint_step']==123392
 assert v['effective_integrator']=='midpoint' and v['matmul_precision']=='medium'
 assert set(v['observed_denoiser_calls_per_sample_batch'])=={24}
 report_path=Path(v['output'])/'sd_agent_metrics.json';r=read(report_path)
 assert all(r[k]==target[k] for k in scope)
 assert r['decoder']['ema_num_updates']==123392
 assert r['agent_metrics']==v['metrics']
 assert all(v['metrics'][k]<value for k,value in target['targets'].items())
 reports.append(dict(seed=v['seed'],metrics=v['metrics'],report=str(report_path),sha256=sha(report_path),generated_vehicles=r['num_generated_vehicles'],onroad_fraction=r['feature_counts_generated'][1]/r['num_generated_vehicles']))
assert rows[0]['sampling_source_sha256_per_batch']!=rows[1]['sampling_source_sha256_per_batch']
assert rows[0]['sampler']==rows[1]['sampler']
worst={k:max(v['metrics'][k] for v in rows) for k in target['targets']}
summary=dict(status='SAME_FINAL128_CHECKPOINT_TWO_GEN_FULL50K_ALL7_VERIFIED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=meta,targets=target['targets'],reports=reports,worst_by_metric=worst,worst_improvement_percent={k:100*(1-worst[k]/value) for k,value in target['targets'].items()},source_scope_verified=list(scope),network_evaluations_per_batch=24,parameters=8083978,TRAIN_seed=4817,GEN=[2817,3817],full50000=True,conditioning_caveat='Repository benchmark only: GTego position/heading/shape/type fixed andSMART road-edge conditioning differ from SD/VW releasedgeneration conditions.',one_TRAIN_seed_only=True)
(B/'paired_final128_verified_report_20261015.json').write_text(json.dumps(summary,indent=2)+'\n')
# Combine unsampled W&B validation history with immutable native report archives.
curves=read(B/'omega_native_training_curves_20261015.json')['series']
for series,host,name in zip(curves,['host128','server9'],['init_diffusion_657_n24_road_k8_omega025_full','init_diffusion_657_n24_road_k8_omega10_full']):
 by_epoch={v['epoch']:v for v in series['rows']}
 for p in (B/'native_full50K_history'/host/name).glob('ema*.json'):
  r=read(p)['report'];step=r['decoder']['ema_num_updates'];by_epoch[step//964]=dict(epoch=step//964,**r['agent_metrics'])
 if host=='host128':
  r=read(snap/'native_report.json');by_epoch[128]=dict(epoch=128,**r['agent_metrics'])
 series['rows']=[by_epoch[k] for k in sorted(by_epoch)]
fig,axes=plt.subplots(1,3,figsize=(11.5,3.5))
for ax,k,label in zip(axes,['ang_dev_jsd','length_jsd','speed_jsd'],['Angle deviation','Length','Speed']):
 for s in curves:ax.plot([v['epoch'] for v in s['rows']],[v[k] for v in s['rows']],'-o',markersize=2.8,lw=1.5,color=s['color'],label=s['label'])
 ax.axhline(target['targets'][k],ls='--',lw=1,color='#667085',label='SD/VW target');ax.set_yscale('log');ax.set_xlabel('Completed epochs',fontsize=8);ax.set_title(label,fontsize=10);ax.tick_params(labelsize=7);ax.grid(alpha=.2)
axes[0].legend(fontsize=6);fig.tight_layout();plot=B/'omega_native_training_curves_final128_20261015.png';fig.savefig(plot,dpi=190);plt.close(fig)
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Mono','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
navy=HexColor('#16324F');teal=HexColor('#147D70');gray=HexColor('#596579');light=HexColor('#EEF3F7')
styles={
 'title':ParagraphStyle('title',fontName='DVB',fontSize=19,leading=27,textColor=navy,spaceAfter=8,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='DVB',fontSize=12,leading=19,textColor=navy,spaceBefore=10,spaceAfter=6,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='DV',fontSize=9,leading=15,textColor=navy,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='DV',fontSize=7.6,leading=12,textColor=gray,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=8,leading=12,textColor=navy,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='DVB',fontSize=8,leading=12,textColor=white,wordWrap='CJK'),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=7.2,leading=11,textColor=navy,backColor=light,borderPadding=6,spaceAfter=8,wordWrap='CJK')}
def P(text,kind='body'):
 escaped=html.escape(str(text));escaped=re.sub(r'([\u2e80-\uffff]+)',r'<font name="CJK">\1</font>',escaped)
 return Paragraph(escaped,styles[kind])
def table(data,widths):
 t=Table([[P(v,'head' if i==0 else 'cell') for v in row] for i,row in enumerate(data)],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,navy)]));return t
story=[]
story += [P('InitDiffusion 训练与独立评价复盘','title'),P('2026-10-15 | 同一最终权重 | 完整 50K | 两个固定生成种子','small'),P('结论：GEN 2817 与 3817 的七项指标均低于指定 SD/VW 本地基准。最接近约束的是角度：取两组较差值仍比目标低 %.2f%%。'%summary['worst_improvement_percent']['ang_dev_jsd'])]
data=[['指标（越低越好）','SD/VW 目标','GEN 2817','GEN 3817','较差值改善']]
for k in ['ang_dev_jsd','length_jsd','speed_jsd','width_jsd','lat_dev_jsd','nearest_dist_jsd','collision_rate']:
 data.append([k,f"{target['targets'][k]:.6f}",f"{rows[0]['metrics'][k]:.6f}",f"{rows[1]['metrics'][k]:.6f}",f"{summary['worst_improvement_percent'][k]:.1f}%"])
story += [table(data,[145,84,93,93,96]),Spacer(1,9),P('表中目标逐项取 SD 与 VectorWorld 的更优值。改善率使用两组结果中的较差值计算，没有将不同 checkpoint 或不同种子的最优项拼接。','small'),P('R 值与比较范围','h')]
story += [table([['参考 iu3fa4gx_full50000','GEN 2817','GEN 3817'],['R_jsd',f"{rows[0]['metrics']['R_jsd']:.6f}",f"{rows[1]['metrics']['R_jsd']:.6f}"],['R_attr',f"{rows[0]['metrics']['R_attr']:.6f}",f"{rows[1]['metrics']['R_attr']:.6f}"]],[245,133,133])]
story += [P('R_jsd 是相对该参考的六项 JSD 比值均值；R_attr 为 length、width、speed 的比值均值。这些相对分数与上表绝对 SD/VW 达标条件不同。','small'),P('条件限制：这是用户指定的仓库内 50K 比较。本方法固定 GT ego 的位置、朝向、尺寸与类型，并使用 SMART 道路边界条件；SD/VW 的公开生成条件不同，不能据此声称严格同条件复现论文。仅使用一个训练种子 4817，尚未完成独立训练种子的复现。','small')]
story += [PageBreak(),P('训练轨迹与搜索发现','title'),Image(str(plot),width=511,height=155),P('曲线为训练内原始完整 50K 验证；生成噪声随 epoch 改变。0.025 使用 RTX 5090，0.1 使用 A6000，跨硬件曲线不构成严格的单变量因果对照。','small'),P('短训提升没有保证最终提升','h')]
raw=read(B/'native_prefix8_pair_v1/observations.json');scaled=next(v for v in raw if v['arm']=='scaled025');control=next(v for v in raw if v['arm']=='raw025')
data=[['本机 8 epoch 候选','Angle / control','Length / control','Speed / control']]
ratios={k:scaled['metrics'][k]/control['metrics'][k] for k in target['targets']};data.append(['time-scaled floor .05',f"{ratios['ang_dev_jsd']:.3f}",f"{ratios['length_jsd']:.3f}",f"{ratios['speed_jsd']:.3f}"])
for label,path in [('time-scaled floor .2','native_floor02_prefix8_v1'),('circular arc matching','native_arc_prefix8_v1'),('heading weight .05','native_omega05_prefix8_v1')]:
 r=read(B/path/'decision.json')['ratios_to_raw8'];data.append([label,f"{r['ang_dev_jsd']:.3f}",f"{r['length_jsd']:.3f}",f"{r['speed_jsd']:.3f}"])
story += [table(data,[175,112,112,112]),P('比值小于 1 表示改善。各组保持原始初始化、实际样本顺序、128 epoch 学习率周期及训练预算；候选未全部通过预设的属性退化限制，因此没有由短训自动推广。完整 0.05 训练是此前基于长期 0.025/0.1 证据独立登记的探索。','small')]
old={r['seed']:r for r in read(B/'omega025_e80_confirmation_v1/observations.json') if r['scenes']==50000}
assert all(v['sampling_source_sha256_per_batch']==old[v['seed']]['sampling_source_sha256_per_batch'] for v in rows)
rat={k:statistics.mean(v['metrics'][k]/old[v['seed']]['metrics'][k] for v in rows) for k in ('ang_dev_jsd','length_jsd','speed_jsd','width_jsd')}
story += [P('完整训练确有角度与属性的取舍','h'),P('与同一训练任务第 80 epoch 相比，两组固定 GEN 的初始噪声逐批完全一致。最终 128 epoch 的角度平均改善 %.1f%%；length、speed、width 分别退化 %.1f%%、%.1f%%、%.1f%%，但仍低于指定基准。早期属性分布更好，不代表当时已满足角度约束。'%((1-rat['ang_dev_jsd'])*100,(rat['length_jsd']-1)*100,(rat['speed_jsd']-1)*100,(rat['width_jsd']-1)*100)),P('EMA 与梯度诊断','h'),P('EMA 实际使用自适应 decay：8 epoch 的参数平均年龄约 0.8 epoch，128 epoch 约 10.3 epoch。512 个训练场景的 CPU 梯度检查表明，碰撞项只在部分 batch 反向影响角速度，尚不足以支持直接删掉碰撞损失。这些诊断不是生成质量的因果证明。','small')]
story += [PageBreak(),P('复现入口与后续安排','title')]
cfg=yaml.safe_load((snap/'config.yaml').read_text());di=cfg['model']['model_config']['decoder']['init_diffusion']
recipe=[['项目','本次已验证配置'],['生成器','8,083,978 参数；hidden 256；3 层；8 heads；固定 AIRL map encoder'],['数据','training_map2_sd：493,542；scenario_dreamer_val：完整 50,000'],['训练','从头初始化 G1；128 epoch；batch 512；workers 8；每 8 epoch 完整验证'],['优化与 EMA','AdamW；lr 5e-4；cosine 128；最低比例 .05；EMA decay .9999（自适应）'],['角度与速度','共享输出行预测 angular velocity；heading MSE 权重 .025；标量 speed'],['采样','原生 midpoint，12 intervals = 24 denoiser evaluations；power 1.2；terminal .1'],['地图与邻居','road_edges；a2a K8；map2agent K30；二者半径 300 m'],['精度与匹配','medium matmul / 32-true；legacy 匹配；matching_weights=null（单位权重）']]
story += [table(recipe,[95,416]),Spacer(1,8),P('评价命令（权重已在本地保存）','h'),P('python src/run.py experiment=init_diffusion_657_n24_road_k8_omega025_final128_eval seed=2817','code'),P('python src/run.py experiment=init_diffusion_657_n24_road_k8_omega025_final128_eval seed=3817','code'),P('配置已完成真实最终权重 strict load，并核对全部 50K 文件及顺序。上述配置默认 workers 8；本报告实际复核使用带逐批噪声及 NFE 审计的原生 test 包装器、workers 0。实际执行配置及 verified.json 保存在各 GEN 目录。','small'),P('权重身份','h'),P('epoch 127 / global_step 123392 / EMA 123392','small'),P(meta['sha256'][:32]+'\n'+meta['sha256'][32:],'code'),P('后续探索','h'),P('host128 已从头启动 heading 权重 0.05 的完整任务；5 层 0.025/0.1 候选仍在训练，尚无最终质量结论。已保留第 104 epoch 权重用于后续早停比较；它没有被混入本次最终 128 epoch 结果。持续探索不会覆盖这份已验证权重。','small'),P('证据与定义','h'),P('报告目录：output/reports/init_diffusion_continuous_20261014/\npaired_final128_verified_report_20261015.json：原始报告路径、哈希、阈值和逐项复核。\nomega025_final128_confirmation_v1/：预登记、逐批噪声、两组原始完整评价。\n梯度与车道几何审查为只读诊断，没有改评价定义。JSD 沿用仓库 JS divergence 及其倍率；collision_rate 为百分比。','small')]
def footer(c,doc):
 c.setStrokeColor(light);c.line(42,31,553,31);c.setFillColor(gray);c.setFont('DV',7);c.drawString(42,20,'InitDiffusion | verified repository full50K benchmark | 2026-10-15');c.drawRightString(553,20,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=42,title='InitDiffusion final128 paired full50K report',author='sim experiment records').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(pdf=str(OUT),verified_summary=str(B/'paired_final128_verified_report_20261015.json'),both_GEN_all7=True,worst_improvement_percent=summary['worst_improvement_percent'])))
