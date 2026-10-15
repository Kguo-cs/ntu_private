"""Locate full50K JSD errors across maturity, without fitting histograms."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,os
import numpy as np
from scipy.special import xlogy
from scipy.spatial.distance import jensenshannon

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
FEATURES=('ang_dev_jsd','length_jsd','width_jsd','speed_jsd')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
 out=BASE/'omega025_maturity_bin_audit_20261015';out.mkdir(exist_ok=False)
 curves={};details={};truth={};source={};coverage={}
 for epoch in (80,96,104,128):
  family='omega025_final128_confirmation_v1' if epoch==128 else f'omega025_e{epoch}_confirmation_v1'
  rows={r['seed']:r for r in read(BASE/family/'observations.json') if r['scenes']==50000}
  assert set(rows)=={2817,3817}
  details[str(epoch)]={};coverage[str(epoch)]=[]
  for seed,row in rows.items():
   if seed not in source:source[seed]=row['sampling_source_sha256_per_batch']
   assert row['sampling_source_sha256_per_batch']==source[seed]
   report=read(Path(row['output'])/'sd_agent_metrics.json')
   hp=Path(row['output'])/'feature_histograms.json';hist=read(hp)
   coverage[str(epoch)].append(dict(GEN=seed,onroad_fraction=report['feature_counts_generated'][1]/report['num_generated_vehicles'],GT_onroad_fraction=report['feature_counts_gt'][1]/report['num_gt_vehicles'],generated_vehicles=report['num_generated_vehicles'],GT_vehicles=report['num_gt_vehicles']))
   for feature in FEATURES:
    h=hist[feature];g=np.array(h['generated'],dtype=np.float64);q=np.array(h['matched_gt'],dtype=np.float64)
    edges=np.array(h['edges']);p=g/g.sum();q=q/q.sum();mix=(p+q)/2
    if feature in truth:assert np.array_equal(q,truth[feature])
    else:truth[feature]=q
    contribution=.5*(xlogy(p,np.divide(p,mix,out=np.ones_like(p),where=mix>0))+xlogy(q,np.divide(q,mix,out=np.ones_like(q),where=mix>0)))*h['multiplier']
    total=float(contribution.sum());assert math.isclose(total,report['agent_metrics'][feature],rel_tol=1e-9,abs_tol=1e-10)
    assert math.isclose(total,float(jensenshannon(g,h['matched_gt'])**2*h['multiplier']),rel_tol=1e-10,abs_tol=1e-12)
    curves[epoch,seed,feature]=dict(edges=edges,error=(p-q)*100,contributions=contribution,p=p,q=q)
    ranking=np.argsort(contribution)[-6:][::-1]
    details[str(epoch)].setdefault(feature,[]).append(dict(GEN=seed,JSD=total,histogram_sha256=sha(hp),total_variation=float(np.abs(p-q).sum()/2),
     first_bin_mass=dict(generated=float(p[0]),GT=float(q[0])),
     largest_contributors=[dict(lower=float(edges[i]),upper=float(edges[i+1]),generated_mass=float(p[i]),GT_mass=float(q[i]),JSD_contribution=float(contribution[i])) for i in ranking]))
 comparisons={}
 for feature in FEATURES:
  seed_rows=[]
  for seed in (2817,3817):
   old=curves[104,seed,feature];new=curves[128,seed,feature];d=new['contributions']-old['contributions'];top=np.argsort(d)[-6:][::-1]
   seed_rows.append(dict(GEN=seed,total_JSD_increase=float(d.sum()),largest_increases=[dict(lower=float(old['edges'][i]),upper=float(old['edges'][i+1]),JSD_increase=float(d[i]),error_pp_104=float(old['error'][i]),error_pp_128=float(new['error'][i])) for i in top]))
  comparisons[feature]=seed_rows
 # TwoGEN ranges describe sample sensitivity; they are not confidence intervals.
 os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_mpl_continuous')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(12,6.7))
 selected=('ang_dev_jsd','length_jsd','speed_jsd')
 bounds=((-30,30),(3.2,6.5),(0,30))
 for col,(feature,limits) in enumerate(zip(selected,bounds)):
  for epoch,color in zip((80,96,104,128),('#a3a8ae','#218a83','#2768b7','#c75c48')):
   vals=[curves[epoch,s,feature] for s in (2817,3817)];x=(vals[0]['edges'][:-1]+vals[0]['edges'][1:])/2;y=np.stack([v['error'] for v in vals])
   axes[0,col].plot(x,y.mean(0),color=color,lw=1.2,label=f'Epoch {epoch}')
   axes[0,col].fill_between(x,y.min(0),y.max(0),color=color,alpha=.1)
  vals=[curves[128,s,feature]['contributions']-curves[104,s,feature]['contributions'] for s in (2817,3817)]
  axes[1,col].bar(x,np.mean(vals,axis=0),width=np.diff(curves[104,2817,feature]['edges'])*.85,color='#8d5c56')
  for ax in axes[:,col]:ax.axhline(0,color='#777',lw=.7);ax.set_xlim(*limits);ax.grid(alpha=.16)
  axes[0,col].set_title(feature);axes[0,col].set_ylabel('Generated - GT (percentage points)')
  axes[1,col].set_ylabel('JSD contribution: epoch128 - 104')
  axes[1,col].set_xlabel('degrees' if feature=='ang_dev_jsd' else 'm/s' if feature=='speed_jsd' else 'm')
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Same TRAIN run and paired source draws / complete 50K / two GEN seeds')
 fig.text(.5,.01,'Shading: two-GEN range, not confidence intervals. JSD uses all bins; plots show selected ranges. No bin calibration or metric changes.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.035,1,.96));image=out/'maturity_errors.png';fig.savefig(image,dpi=170);plt.close(fig)
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PAIRED_FULL50K_MATURITY_BIN_AUDIT_COMPLETE',TRAIN_seed=4817,GEN=[2817,3817],source_noise_exact_across_epochs=True,all_metrics_recomputed_from_integer_histograms=True,details=details,epoch128_minus104=comparisons,coverage=coverage,image_sha256=sha(image),source_sha256=sha(Path(__file__)),limitations='Histogram decomposition localizes errors, not their trainingcause. No fitting/postprocessing, metricdefinition change or newacceptancegate.')
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],image=str(image),late_bins=comparisons)))

if __name__=='__main__':main()
