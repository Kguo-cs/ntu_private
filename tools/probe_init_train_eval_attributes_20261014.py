"""Read-only TRAIN marginal versus full50K GT diagnostic, never calibration."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_train_eval_attribute_plot')
from pathlib import Path
import json
import sqlite3
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_init_overnight_20261014 import divergence
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'


def main():
    train_path=ART/'attribute_prior/train_attributes.pt'
    payload=torch.load(train_path,map_location='cpu',weights_only=True)
    assert payload['fields']==['length','width','speed']
    train=payload['attributes'].numpy();kind=payload['type'].numpy();ego=payload['ego_mask'].numpy().astype(bool)
    assert np.isfinite(train).all() and (train[:,:2]>0).all() and (train[:,2]>=0).all()
    database=ROOT/'src/waymo_data/sd_real_metric_cache.sqlite'
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as con:
        meta={k:json.loads(v) for k,v in con.execute('SELECT key,value FROM metadata')}
    target=read(ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json')
    assert meta['completed'] and meta['num_scenes']==50000
    assert meta['eval_set_sha256']==target['eval_set_sha256']
    assert meta['source_manifest_sha256']==target['reference_source_manifest_sha256']
    assert meta['backend_signature']['source_sha256']==target['official_source_sha256']
    stats=meta['reference_statistics']
    assert stats['totals']==target['feature_counts_gt']
    groups={};plots={}
    for label,mask in [('all_vehicle_targets',kind==0),('movable_vehicle_targets',(kind==0)&~ego),('fixed_ego_vehicles',(kind==0)&ego)]:
        features={}
        for col,(name,lo,hi,step) in enumerate([('length',0.,25.,.1),('width',0.,5.,.1),('speed',0.,50.,1.)]):
            counts=np.histogram(np.clip(train[mask,col],lo,hi),bins=np.arange(lo,hi+step,step))[0]
            gt=stats['histograms'][col+3]
            result=divergence({'generated':counts,'matched_gt':gt},100)
            selected=np.argsort(result['contributions'])[::-1][:6]
            features[name]=dict(train_vs_all_GT_jsd=result['value'],train_counts=counts.tolist(),gt_counts=gt,
                bin_width=step,minimum=lo,maximum=hi,
                dominant_bins=[dict(lo=float(k*step),hi=float((k+1)*step),
                    train_mass=float(result['generated_mass'][k]),gt_mass=float(result['gt_mass'][k]),
                    jsd_contribution=float(result['contributions'][k])) for k in selected])
            if label=='all_vehicle_targets':plots[name]=(lo+(np.arange(len(counts))+.5)*step,result['generated_mass'],result['gt_mass'])
        groups[label]=dict(count=int(mask.sum()),features=features)
    result=dict(generated_utc=now(),probe_sha256=sha(__file__),train_source=str(train_path),
        train_source_sha256=sha(train_path),train_scenes=8192,train_agents=len(train),
        evaluation_cache=str(database),evaluation_metadata_sha256=sha_json(meta),
        evaluation_scenes=50000,evaluation_vehicles=stats['num_vehicles'],
        target_spec_sha256=sha(ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json'),
        metrics_scale=100.,groups=groups,
        train_vehicle_ego_fraction=float(ego[kind==0].mean()),
        current_source_includes_fixed_ego=True,
        interpretation='TRAIN calibration vehicle marginals compared with full50K evaluation GT, not generated-model performance. '
            'This diagnoses distribution mismatch using the same official bins; sample size and scene composition differ. '
            'Not an irreducible error floor, not additive to model JSD, and not a causal explanation. '
            'Movable/ego subgroups are each compared against ALL GT vehicles only to expose mixture effects; those subgroup JSDs are not like-for-like evaluations. '
            'No fitted mapping, threshold, source table, model or sampler is modified from these evaluation histograms.')
    save(ART/'train_eval_attribute_diagnostic.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    for ax,(name,limits) in zip(axes,[('length',(3.5,6.5)),('width',(1.5,2.8)),('speed',(0,30))]):
        x,p,q=plots[name];ax.plot(x,q,label='Full50K GT',color='#263844',lw=1.7)
        ax.plot(x,p,label='TRAIN calibration vehicles',color='#287db6',lw=1.5)
        ax.set_xlim(*limits);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        ax.set_xlabel(name.title()+(' (m/s)' if name=='speed' else ' (m)'))
        ax.set_ylabel('Probability per bin');ax.set_title('TRAIN vs GT JSD %.4f'%groups['all_vehicle_targets']['features'][name]['train_vs_all_GT_jsd'],fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=8.5)
    fig.text(.5,.015,'Read-only data diagnostic; no generated agents. Different scene cohorts; not a model score or error floor.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.89));fig.savefig(ART/'train_eval_attribute_diagnostic.png',dpi=180);plt.close(fig)
    print({k:v['train_vs_all_GT_jsd'] for k,v in groups['all_vehicle_targets']['features'].items()})


def sha_json(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


if __name__=='__main__':main()
