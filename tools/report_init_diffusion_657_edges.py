"""Generate a live, verified-only report for the compact N20 graph search.

Read-only with respect to experiment outputs. Writes separate live_report JSON,
HTML and a Matplotlib SVG; CPU only. Never consumes partial-stage observations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,html,json,math,os,sys
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import METRICS,score
ART=ROOT/'output/reports/init_diffusion_657_edges_20261013'
RUNS=ROOT/'logs/init_diffusion_657_edges_20261013'
ANCHOR='norm_time_every_layer'
LABELS=('Near','Lat','Ang','Length','Width','Speed','Collision')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)
def maybe(path,default=None):
 return read(path) if Path(path).is_file() else default

def phase(name):
 return next((p for p in ('screen','medium','long','full') if name.startswith(p)),'other')
def pair_key(r):
 return (r['seed'],r['eval_seed'],r['seen_training_examples'],r['eval_manifest_sha256'])
def comparison(rows,controls):
 available={pair_key(r):r for r in controls}
 missing=[pair_key(r) for r in rows if pair_key(r) not in available]
 if missing:return {'available':False,'reason':'无相同TRAIN、GEN、训练曝光和场景清单的完整参考','missing_pairs':missing}
 try:
  result=score(rows,controls)
  seeds=sorted({r['seed'] for r in rows})
  per=result.get('per_TRAIN') or {str(seed):score([r for r in rows if r['seed']==seed],controls) for seed in seeds}
  per={key:{**value,'R_below_1':value['R']<1.,'verdict':'改善' if value['R']<1. else ('参考/相同' if math.isclose(value['R'],1.,rel_tol=1e-12,abs_tol=1e-12) else '未改善')} for key,value in per.items()}
  return {'available':True,**result,'per_TRAIN_results':per,
   'all_TRAIN_R_below_1':all(r['R_below_1'] for r in per.values()),
   'robust_improvement':result['all_TRAIN_gates_pass'] and all(r['R_below_1'] for r in per.values())}
 except (RuntimeError,ValueError,ZeroDivisionError) as e:return {'available':False,'reason':str(e)}

def load_historical_controls():
 rows=[];provenance=[]
 for name in ('train4817','train5817','screen4817','full50000_4817'):
  p=ROOT/f'logs/control657_paired_20261012/{name}/summary.json'
  if p.is_file():
   values=[r for r in read(p) if r['variant'].startswith('control657')];rows+=values
   provenance.append({'path':str(p),'sha256':sha(p),'observations':len(values)})
 for name in ('full_control5817','round2_full_control5817'):
  p=ROOT/f'logs/init_diffusion_657_iteration_20261012/{name}'
  marker=maybe(p/'verified_completion.json')
  if marker and marker['summary_sha256']==sha(p/'summary.json'):
   values=[r for r in read(p/'summary.json') if r['variant'].startswith('control657')];rows+=values
   provenance.append({'path':str(p/'summary.json'),'sha256':sha(p/'summary.json'),'observations':len(values)})
 return rows,provenance

def validate_stage(folder):
 marker=read(folder/'edge_verified_completion.json')
 if marker.get('status')!='VERIFIED_COMPLETE' or marker.get('returncode')!=0:raise ValueError('outer completion未成功')
 if marker.get('summary_sha256')!=sha(folder/'summary.json'):raise ValueError('summary SHA不匹配')
 if marker.get('contract_sha256')!=sha(folder/'edge_contract.json'):raise ValueError('contract SHA不匹配')
 rows=read(folder/'summary.json');contract=read(folder/'edge_contract.json');spec=contract['spec']
 expected=marker['metrics_reports']
 if len(rows)!=len(expected):raise ValueError('原生报告数量与summary不一致')
 report_hash={r['path']:r['sha256'] for r in expected}
 for row in rows:
  path=row['metrics_path']
  if path not in report_hash or sha(path)!=report_hash[path]:raise ValueError('原生metric报告SHA不匹配')
  native=read(path)
  if native['decoder']['sampling_steps']!=20:raise ValueError('实际采样步数不是20')
  if native['num_samples']!=spec['scenes'] or native['num_gt_samples']!=spec['scenes']:raise ValueError('场景数量不匹配')
  if row['evaluation_weights']!='ema' or row['seed']!=spec['seed']:raise ValueError('EMA/TRAIN不匹配')
  if native['runtime']['trainer_precision']!='32-true':raise ValueError('precision不匹配')
  if spec['scenes']==50000 and not native['require_full_set']:raise ValueError('未强制完整50k')
  for key,value in spec.get('expected_denoiser_options',{}).items():
   if native['decoder']['denoiser_options'].get(key)!=value:raise ValueError('实际denoiser选项不匹配:'+key)
  if any(not math.isfinite(float(row[k])) or float(row[k])<0 for k in METRICS):raise ValueError('metric非有限或负值')
  if any(not math.isclose(float(row[k]),float(native['agent_metrics'][k]),rel_tol=1e-12,abs_tol=1e-12) for k in METRICS):raise ValueError('summary指标与原生报告数值不一致')
 return rows,marker,spec

def stage_runtime(folder,rows):
 memory=[]
 p=folder/'console.log'
 if p.is_file():
  for line in p.read_text(errors='replace').splitlines():
   if line.startswith('RUNTIME_MEMORY '):
    try:memory.append(json.loads(line[len('RUNTIME_MEMORY '):]))
    except json.JSONDecodeError:pass
 def peak(key):
  values=[r[key] for r in memory if key in r]
  return max(values) if values else None
 train=max((r.get('train_seconds',0) for r in rows),default=0)
 steps=max((r['budget'] for r in rows),default=0)
 return {'train_seconds':train,'train_seconds_per_step':train/steps if train and steps else None,
  'eval_seconds':sum(r.get('eval_seconds',0) for r in rows),
  'cuda_peak_allocated_mib':peak('cuda_peak_allocated_mib'),
  'cuda_peak_reserved_mib':peak('cuda_peak_reserved_mib'),
  'peak_process_rss_mib':peak('peak_rss_kib')/1024 if peak('peak_rss_kib') is not None else None,
  'memory_scope':'训练回调采样记录中的进程累计CUDA峰值，原生checkpoint评价在fit结束后运行；不是精确每层内存测量',
  'memory_records':len(memory)}

def auxiliary_depth5():
 folder=ROOT/'logs/init_diffusion_657_iteration_20261012/round2_medium_norm_depth5'
 marker=maybe(folder/'verified_completion.json')
 if not marker:return None
 if marker.get('status')!='VERIFIED_COMPLETE' or marker['summary_sha256']!=sha(folder/'summary.json'):return None
 values=[r for r in read(folder/'summary.json') if r['budget']==4096]
 controls,_=load_historical_controls();s=comparison(values,controls)
 audit=maybe(ART/'cpu_edge_audit.json',{})
 return {'purpose':'此前5层网络的辅助背景，不参与本轮3层候选排名或统计池',
  'variant':'norm_depth5','budget':4096,'num_samples':4096,'TRAIN':[4817],'GEN':[2817,3817],
  'num_parameters':audit.get('parameters_5layer',{}).get('G1_trainable'),
  'parameter_count_source':'CPU实测相同配置架构；旧GPUsummary未记录此字段',
  'reference_3layer_parameters':audit.get('parameters_3layer',{}).get('G1_trainable'),
  'comparison':s,'stage':str(folder),'stage_summary_sha256':sha(folder/'summary.json')}

def build_data(art=ART,runs=RUNS):
 protocol=maybe(art/'protocol.json');plan=(protocol or {}).get('effective_plan',maybe(art/'plan.json',{}));controls,provenance=load_historical_controls()
 completed=[];stage_records=[];quarantined=[];partials=[];aliases=[];seen_reports=set()
 for folder in sorted(runs.glob('*')) if runs.exists() else []:
  if not folder.is_dir():continue
  if not (folder/'edge_verified_completion.json').is_file():
   partials.append({'stage':folder.name,'state': 'FAILED' if (folder/'edge_execution.json').is_file() else 'UNVERIFIED_OR_RUNNING',
    'execution':maybe(folder/'edge_execution.json'),'metrics_usable':False})
   continue
  try:rows,marker,spec=validate_stage(folder)
  except (ValueError,KeyError,OSError,json.JSONDecodeError) as e:
   quarantined.append({'stage':folder.name,'reason':str(e),'metrics_usable':False});continue
  unique=[r for r in rows if r['metrics_path'] not in seen_reports]
  if len(unique)!=len(rows):aliases.append({'stage':folder.name,'duplicate_metric_reports':len(rows)-len(unique),'independent_replicates':False})
  seen_reports.update(r['metrics_path'] for r in unique)
  for row in unique:completed.append({**row,'stage':folder.name,'phase':phase(folder.name)})
  stage_records.append({'stage':folder.name,'phase':phase(folder.name),'completion':marker,'spec':spec,
    'runtime':stage_runtime(folder,rows),'observations':len(unique)})
 # Fresh controls never extend or replace the historical denominator registry.
 repeat=[r for r in completed if r['variant']=='control657']
 anchor=[r for r in completed if r['variant']==ANCHOR]
 grouped=defaultdict(list)
 for row in completed:grouped[(row['phase'],row['variant'],row['budget'],row['num_samples'])].append(row)
 comparisons=[]
 for (p,name,budget,scenes),values in sorted(grouped.items(),key=lambda item:({'screen':0,'medium':1,'long':2,'full':3,'other':4}[item[0][0]],item[0][2],item[0][1])):
  # Distinct TRAIN/GEN scopes only. Identical scopes from separately verified
  # folders are provenance duplicates; do not turn them into extra seed samples.
  unique={}
  for row in values:
   key=pair_key(row)
   if key in unique:aliases.append({'stage':row['stage'],'duplicate_pair_scope':list(key),'independent_replicates':False})
   unique[key]=row
  values=list(unique.values());historical=comparison(values,controls)
  params=sorted({r['generator_trainable_parameters'] for r in values if 'generator_trainable_parameters' in r})
  total=sorted({r['generator_num_parameters'] for r in values if 'generator_num_parameters' in r})
  frozen=sorted({r['frozen_map_num_parameters'] for r in values if 'frozen_map_num_parameters' in r})
  comparisons.append({'phase':p,'variant':name,'budget':budget,'num_samples':scenes,
   'seen_training_examples':sorted({r['seen_training_examples'] for r in values}),
   'equivalent_epochs':statistics.fmean(r['seen_training_examples'] for r in values)/493542,
   'TRAIN_seeds':sorted({r['seed'] for r in values}),'GEN_seeds':sorted({r['eval_seed'] for r in values}),
   'paired_cells':len(values),'complete_stages':sorted({r['stage'] for r in values}),
   'historical657':historical,'primary_fresh657':comparison(values,[r for r in repeat if r['phase']==p]),
   'primary_denominator':'同phase新657；同TRAIN/GEN、曝光和场景清单，缺失时不借历史参考',
   'contemporaneous657':comparison(values,[r for r in repeat if r['phase']==p]),
   'native_mean_metrics':{k:statistics.fmean(statistics.fmean(r[k] for r in values if r['seed']==seed) for seed in sorted({r['seed'] for r in values})) for k in METRICS},
   'compact_anchor':comparison(values,anchor),
   'actual_generator_trainable_parameters':params,'actual_generator_total_parameters':total,
   'actual_frozen_map_parameters':frozen,'declared_parameters':plan.get('variant_metadata',{}).get(name,{}).get('num_parameters'),
   'native_observations':[{k:r[k] for k in ('stage','seed','eval_seed','budget','seen_training_examples','num_samples','metrics_path','checkpoint','train_seconds','eval_seconds')+METRICS} for r in values],
   'actual_sampling_steps':20,'parameter_count_source':'实际GPUrunner summary；空列表表示尚未记录，禁止用计划量替代实测'})
 status=maybe(art/'rootdriver_status.json') or maybe(art/'status.json') or {'status':'NOT_STARTED'}
 return {'generated_utc':datetime.now(timezone.utc).isoformat(),'status':status,'deadline':plan.get('deadline'),
   'protocol':protocol,'plan':plan,'constraints':maybe(art/'user_constraints.json'),
   'verified_stages':stage_records,'comparisons':comparisons,'historical_control_provenance':provenance,
   'failed_stages':maybe(art/'failed_stages.json',[]),'unverified_stages_not_evidence':partials,
   'quarantined_stages':quarantined,'aliases_not_independent_replicates':aliases,
   'decisions':{name:read(art/name) for name in ('screen.json','medium_selection.json','medium.json','long_selection.json','long.json','full.json','adaptive_registration.json') if (art/name).is_file()},
   'auxiliary_5layer':auxiliary_depth5(),
   'native128epoch_full_evaluation_verified':False,
   'scope_warning':'短期及约10.62epoch配对结果不能证明128epoch收敛；原始657的128epoch日志只有256场景监控。',
   'comparison_definition':'主R使用同phase新657的七项candidate/control原生指标比均值；历史657仅secondary。先GEN在每个TRAIN内均值，再TRAIN等权；R_attr为Length/Width/Speed比均值。',
   'gates_definition':'各Length/Width/Speed≤1.15；onroad下降≤.02；生成vehicle/GT∈[.9,1.1]；多TRAIN须各自和pooled都通过。'}

def plot_curves(data,art):
 # The only plotting dependency is imported inside this CPU-only function.
 os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
 os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/657_edges_mpl')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
 colors=plt.get_cmap('tab20');names=sorted({r['variant'] for r in data['comparisons']})
 palette={name:colors(i%20) for i,name in enumerate(names)}
 for row,(scenes,denominator,title) in zip(axes.flat,((1024,'primary_fresh657','Screen: N=1024 / fresh same-phase657'),(1024,'compact_anchor','Screen: N=1024 / compact anchor'),(4096,'primary_fresh657','Confirmation: N=4096 / fresh same-phase657'),(4096,'compact_anchor','Confirmation: N=4096 / compact anchor'))):
  row.axhline(1,color='black',ls='--',lw=1,alpha=.7)
  plotted=False
  for name in names:
   values=[r for r in data['comparisons'] if r['variant']==name and r['num_samples']==scenes and r[denominator].get('available')]
   values=sorted(values,key=lambda r:r['equivalent_epochs'])
   if values:
    scopes=sorted({tuple(r['TRAIN_seeds']) for r in values})
    for seeds in scopes:
     matched=[r for r in values if tuple(r['TRAIN_seeds'])==seeds]
     row.plot([r['equivalent_epochs'] for r in matched],[r[denominator]['R'] for r in matched],marker='o' if len(seeds)==1 else '^',ls='-' if len(seeds)==1 else ':',ms=4,lw=1.5,label=name,color=palette[name])
    plotted=True
  if not plotted:row.text(.5,.5,'No fully verified paired cells yet',ha='center',va='center',transform=row.transAxes,color='#64748b')
  row.set(title=title,xlabel='Equivalent TRAIN epochs (examples / 493542)',ylabel='R (lower is better)');row.grid(alpha=.2)
 fig.suptitle('Fixed N20; fresh G1; verified stages only',fontsize=15)
 legend={}
 for ax in axes.flat:
  handles,labels=ax.get_legend_handles_labels()
  for handle,label in zip(handles,labels):legend.setdefault(label,handle)
 if legend:fig.legend(legend.values(),legend.keys(),loc='lower center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9,frameon=False)
 fig.savefig(art/'live_learning_curves.svg',metadata={'Date':None},bbox_inches='tight');fig.savefig(art/'live_learning_curves.png',dpi=110,bbox_inches='tight');plt.close(fig)

def fmt(value,digits=4):return '—' if value is None else f'{value:.{digits}f}'
def td(value):return '<td>'+html.escape(str(value))+'</td>'
def gate(value):return ('PASS' if value.get('all_TRAIN_gates_pass') else 'FAIL') if value.get('available') else '待配对'
def render_html(data,art):
 comparisons=data['comparisons'];body=[];metricbody=[];runtimes=[];seedbody=[]
 for r in comparisons:
  s=r['primary_fresh657'];a=r['compact_anchor'];historic=r['historical657'];ratios=s.get('ratios',{})
  cells=[r['phase'],r['variant'],r['budget'],fmt(r['equivalent_epochs'],3),r['num_samples'],','.join(map(str,r['TRAIN_seeds'])),','.join(map(str,r['GEN_seeds'])),
   fmt(s.get('R')),fmt(s.get('R_attr')),fmt(a.get('R')),gate(a),fmt(historic.get('R')),gate(s),
   '; '.join(f"{seed}:{fmt(value['R'])} {value['verdict']}" for seed,value in s.get('per_TRAIN_results',{}).items()) or '待配对',
   ('是' if s.get('robust_improvement') else '否') if s.get('available') else '待配对']+[fmt(ratios.get(k)) for k in METRICS]
  body.append('<tr>'+''.join(map(td,cells))+'</tr>')
  m=r['native_mean_metrics'];am=a.get('ratios',{})
  metricbody.append('<tr>'+''.join(map(td,[r['phase'],r['variant'],r['budget'],r['num_samples']]+[fmt(m.get(k),6) for k in METRICS]+[fmt(am.get(k)) for k in METRICS]))+'</tr>')
  for reference,name in ((s,'同期657'),(a,'紧凑anchor')):
   for seed,value in reference.get('per_TRAIN_results',{}).items():
    seedbody.append('<tr>'+''.join(map(td,[r['phase'],r['variant'],r['budget'],r['num_samples'],name,seed,','.join(map(str,value['GEN_seeds'])),
     fmt(value['R']),fmt(value['R_attr']),'PASS' if value['gates_pass'] else 'FAIL',value['verdict']]+[fmt(value['ratios'].get(k)) for k in METRICS]))+'</tr>')
 for stage in data['verified_stages']:
  r=stage['runtime'];scope=next((x for x in comparisons if stage['stage'] in x['complete_stages']),{})
  counts=scope.get('actual_generator_trainable_parameters',[])
  runtimes.append('<tr>'+''.join(map(td,[stage['stage'],', '.join(f'{n:,}' for n in counts) or '实际待记录',20,
   fmt(r['train_seconds'],1),fmt(r['train_seconds_per_step'],3),fmt(r['eval_seconds'],1),fmt(r['cuda_peak_allocated_mib'],0),fmt(r['cuda_peak_reserved_mib'],0),fmt(r['peak_process_rss_mib'],0)]))+'</tr>')
 headers=['阶段','候选','steps','等价epoch','场景','TRAIN','GEN','主R/同期657↓','主R_attr↓','R/紧凑anchor↓','anchor门槛','历史657 R(secondary)↓','主门槛','各TRAIN主R/胜负','各TRAIN均改善且门槛通过']+[name+'比' for name in LABELS]
 metricheaders=['阶段','候选','steps','场景']+['原生'+name for name in LABELS]+['anchor '+name+'比' for name in LABELS]
 seedheaders=['阶段','候选','steps','场景','参考','TRAIN','GEN','该TRAIN R↓','该TRAIN R_attr↓','该TRAIN门槛','该TRAIN胜负']+[name+'比' for name in LABELS]
 runtimeheaders=['完整stage','实测G1可训参数','采样步数','训练秒','秒/step','评价秒合计','CUDA峰值 allocated MiB','CUDA峰值 reserved MiB','进程RSS峰值 MiB']
 def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+(''.join(rows) or '<tr><td colspan="'+str(len(headers))+'">等待完整核验的实验。运行中的结果不会提前计入。</td></tr>')+'</tbody></table></div>'
 status=data['status'];deadline=data['deadline'] or '2026-10-13 10:00 +08:00'
 aux=data.get('auxiliary_5layer');auxhtml=''
 if aux:
  s=aux['comparison'];auxhtml=f'<p>旧5层 norm_depth5：4096 steps、4096场景、TRAIN4817×GEN2817/3817，历史657参考R={fmt(s.get("R"))}，R_attr={fmt(s.get("R_attr"))}。网络参数12,562,954，相对本轮3层8,083,978增加55.4%。此项仅为辅助背景，<b>不参与本轮候选排名、配对统计池或曲线</b>。</p>'
 failures=data['failed_stages']+data['quarantined_stages']
 if data.get('constraints',{}).get('old_queue_failure'):failures.append({'prior_queue_failure':data['constraints']['old_queue_failure'],'current_observation':False})
 failhtml=''.join('<li><code>'+html.escape(json.dumps(r,ensure_ascii=False))+'</code></li>' for r in failures) or '<li>本轮暂无已记录失败。</li>'
 partialhtml=''.join('<li>'+html.escape(r['stage'])+'：'+html.escape(r['state'])+'，不计为评价证据。</li>' for r in data['unverified_stages_not_evidence']) or '<li>暂无运行中或未核验stage。</li>'
 scopeconditions='固定 AIRL map encoder；fresh G1；batch512；固定8192 TRAIN校准；EMA=.9999；32-true/matmul medium；N20、t_eps=.05；固定ego位置、朝向、尺寸、类型，生成ego速度及其他agent属性；道路边界map context、ego局部正方形半径50m。'
 htmltext=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>657 N20紧凑模型：实时优化报告</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;background:#f7fafc;color:#173047;margin:28px auto;max-width:1500px;padding:0 18px}}h1,h2{{line-height:1.3}}h1{{font-size:27px}}h2{{font-size:20px;margin-top:30px}}.box{{padding:16px;background:#edf4fb;border-left:5px solid #3075ae}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}th,td{{padding:8px;border:1px solid #d4e1ea;text-align:right;white-space:nowrap}}th{{background:#e4eef7}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}p{{max-width:1250px}}code{{font-size:12px;overflow-wrap:anywhere}}li{{margin:10px 0}}.note{{color:#5d7283}}img{{max-width:100%;background:white}}</style>
<h1>以657为baseline：固定N20的紧凑denoiser图结构优化</h1>
<div class="box">截止：{html.escape(deadline)}。状态：<b>{html.escape(str(status.get('status')))}</b>；当前stage：{html.escape(str(status.get('active_stage','—')))}。更新时间 UTC：{html.escape(data['generated_utc'])}。只统计退出成功、summary及原生metric报告SHA核验通过的完整stage。</div>
<p>{scopeconditions} 本轮优先3层256维；该参考架构有8,083,978个可训练参数。修改radius/K不会增加参数；实际参数和内存来自GPUrunner记录，计划数不替代实测值。</p>
<p><b>未验证128epoch最终完整评价。</b> 短期学习曲线用于筛选；最终约10.62epoch、两TRAIN×两GEN及50k确认仍不等于128epoch证明。原始657的128epoch日志仅256场景监控，不能混入4096/50k分母。快速实验重复seed固定的TRAIN顺序，完整训练每epoch重新shuffle也会改变轨迹。</p>
<h2>配对结果</h2>
<p>主R只使用同phase、同预算、TRAIN、GEN及场景清单的新657control；没有完整新参考时显示“—”，不借历史参考。control对自身的主R=1。历史657参考仅为secondary，不能把新control相对历史的短期波动解释为方法提升。R/紧凑anchor使用相同范围的3层归一化＋每层时间embedding。没有相同曝光的完整anchor时显示“—”，不会借用短预算或失败stage。R是七项指标比的均值；R_attr是Length/Width/Speed比均值，越小越好。先GEN在每个TRAIN内平均，再对TRAIN等权。门槛：各L/W/Speed≤1.15；onroad下降≤2个百分点；vehicle/GT∈[.9,1.1]；每个TRAIN及pooled均须通过。</p>
{table(headers,body)}
<h2>逐TRAIN配对结果</h2><p>每个TRAIN分别汇总其GEN，再独立比较。pooled R&lt;1可以与某个TRAIN R≥1同时发生；“各TRAIN均改善且门槛通过”才表示全部已完成种子均获益。表中的主门槛只约束属性、onroad和车辆数，不能替代R胜负判断。control对自身R=1属于参考，不标成改善。</p>
{table(seedheaders,seedbody)}
<h2>已核验学习曲线</h2><p>1024筛选和4096确认分开绘图，50k结果只进入其自身范围的表格。曲线只连接实际完成且TRAIN种子集合一致的配对预算：圆点实线为单TRAIN，三角点虚线为多TRAIN pooled；不同种子范围不连线。没有数据时显示占位文字。</p>
<img src="live_learning_curves.svg" alt="Verified N20 paired learning curves">
<h2>七项原生指标与anchor比值</h2><p>Near、Lat采用原生JSD×10；Ang、Length、Width、Speed采用JSD×100；Collision为生成vehicle中发生碰撞的百分比。不要将这些数值与未缩放JSD或不同样本范围直接混用。</p>
{table(metricheaders,metricbody)}
<h2>参数、时间和内存</h2><p>CUDA峰值来自训练回调的累计峰值记录，checkpoint评价在fit之后运行；这是进程训练范围的峰值，不能视为每层或每个step的精确内存。评价秒合计包含该stage所有完整checkpoint×GEN评价。</p>
{table(runtimeheaders,runtimes)}
<h2>辅助背景</h2>{auxhtml}
<p>32个TRAIN场景的CPU图结构审计已显式加载固定8192 TRAIN统计；先前32场景自动校准的v1仅归档追溯，不作生产源分布证据。详见<a href="report.html">独立图结构审计</a>及<a href="cpu_edge_audit.json">修正的CPU数据</a>。</p>
<h2>失败与未完成尝试</h2><ul>{failhtml}</ul><ul>{partialhtml}</ul>
<p class="note">失败、SIGSEGV、被中断或只完成部分预算的stage不进入评价表、曲线或选择证据。恢复重跑须fresh G1。完整实验记录、config与原生sd_agent_metrics.json保存在logs/init_diffusion_657_edges_20261013；<a href="live_report.json">机器可读完整报告</a>含原始每TRAIN/GEN的metric路径、checkpoint、各门槛细项及选择记录。</p>
</html>'''
 tmp=art/'live_report.html.tmp';tmp.write_text(htmltext);os.replace(tmp,art/'live_report.html')

def build(art=ART,runs=RUNS,plot=True):
 art.mkdir(parents=True,exist_ok=True);data=build_data(art,runs);write_json(art/'live_report.json',data)
 if plot:plot_curves(data,art)
 render_html(data,art)
 print(f"verified_stages={len(data['verified_stages'])}, compared_budgets={len(data['comparisons'])}, quarantined={len(data['quarantined_stages'])}, status={data['status'].get('status')}")
 return data

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-plot',action='store_true');parser.add_argument('--artifacts-dir',type=Path,default=ART);parser.add_argument('--runs-dir',type=Path,default=RUNS);args=parser.parse_args();build(args.artifacts_dir,args.runs_dir,not args.no_plot)
if __name__=='__main__':main()
