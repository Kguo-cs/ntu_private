"""Plot verified native full50K trajectories; no fitting or metric changes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
FEATURES=('ang_dev_jsd','length_jsd','speed_jsd','width_jsd','lat_dev_jsd','nearest_dist_jsd')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text());scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt');curves={}
 for label,host,experiment in [('heading w=.025','host128','init_diffusion_657_n24_road_k8_omega025_full'),('heading w=.05','host128','init_diffusion_657_n24_road_k8_omega05_full'),('heading w=.1','server9','init_diffusion_657_n24_road_k8_omega10_full')]:
  paths=list((BASE/'native_full50K_history'/host/experiment).glob('*.json'))
  if label=='heading w=.025':paths.append(BASE/'full_checkpoint_snapshots/omega025_e80_20261015/native_report.json')
  rows={}
  for p in paths:
   envelope=json.loads(p.read_text());r=envelope.get('report',envelope);assert all(r[k]==target[k] for k in scope);step=r['decoder']['ema_num_updates'];assert step%964==0;epoch=step//964;m={k:r['agent_metrics'][k] for k in FEATURES}
   if epoch in rows:assert rows[epoch]['metrics']==m
   rows[epoch]=dict(epoch=epoch,EMA=step,metrics=m,report=str(p),report_sha256=sha(p))
  curves[label]=[rows[k] for k in sorted(rows)]
 confirmations=[]
 for family in ('omega025_e96_confirmation_v1','omega025_e104_confirmation_v1','omega025_e104_holdout_confirmation_v1','omega025_final128_confirmation_v1'):
  for row in json.loads((BASE/family/'observations.json').read_text()):
   if row['scenes']!=50000:continue
   p=Path(row['output'])/'sd_agent_metrics.json';r=json.loads(p.read_text());assert all(r[k]==target[k] for k in scope)
   confirmations.append(dict(epoch=row['checkpoint_step']//964,GEN=row['seed'],metrics={k:r['agent_metrics'][k] for k in FEATURES},report=str(p),report_sha256=sha(p)))
 os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_mpl_continuous');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(12.4,7));colors={'heading w=.025':'#2369a0','heading w=.05':'#db8c20','heading w=.1':'#8053a3'}
 for ax,feature in zip(axes.ravel(),FEATURES):
  for label,rows in curves.items():
   points=[r for r in rows if r['epoch']>=32];ax.plot([r['epoch'] for r in points],[r['metrics'][feature] for r in points],marker='o',markersize=3,lw=1.5,label=label,color=colors[label])
  ax.scatter([r['epoch'] for r in confirmations],[r['metrics'][feature] for r in confirmations],s=22,facecolors='none',edgecolors=colors['heading w=.025'],label='.025 independent GEN',zorder=4)
  ax.axhline(target['targets'][feature],color='#ad4242',linestyle='--',lw=1,label='SD/VW target');ax.set_title(feature);ax.set_xlabel('Epoch');ax.set_ylabel('JSD (repository scale)');ax.grid(alpha=.2);ax.set_xlim(30,130);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Verified native full50K validation / late-training JSD trajectories')
 fig.text(.5,.018,'Solid lines: native validation; source draws and training hardware differ between runs. Open circles: all recorded GENs at96/104/128 (including the angle failure at epoch104, GEN5817).\nDescriptive comparisons; no confidence intervals or paired-training causal claims. The agreed repository conditioning differs from released SD/VW.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.065,1,.95))
 folder=BASE/('native_jsd_maturity_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir();plot=folder/'native_jsd_maturity.png';fig.savefig(plot,dpi=170);plt.close(fig)
 record=dict(status='VERIFIED_NATIVE_FULL50K_MATURITY_PLOT_COMPLETE',generated_utc=datetime.now(timezone.utc).isoformat(),curves=curves,confirmations=confirmations,targets={k:target['targets'][k] for k in FEATURES},plotted_min_epoch=32,source_sha256=sha(Path(__file__)),image_sha256=sha(plot),scope_verified_for_every_point=True,no_metric_or_model_changes=True,limitation='Native source draws/hardware are not proven paired. AllfourknownGENs at104andtwoat96/128 are shown; point ranges are not confidence intervals. Nativevalidation is additional separate evidence.')
 (folder/'data.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(image=str(plot),data=str(folder/'data.json'),points={k:len(v) for k,v in curves.items()})))
if __name__=='__main__':main()
