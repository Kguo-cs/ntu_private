"""Read-only periodic W&B snapshots until the user's local22:30 deadline."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import re

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import refresh_wandb_cc_readonly as api
from tools.run_init_diffusion_657_iteration import read,save,now,sha
ART=ROOT/'output/reports/init_diffusion_evening_20261013'
DEADLINE=datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
BASE='iu3fa4gx'

def model(c):
    import ast
    value=c.get('model',{}).get('model_config',{})
    return ast.literal_eval(value) if isinstance(value,str) else value

def analyze(folder):
    records={r['id']:r for r in read(folder/'inventory.json')['runs']}
    cache={};freshness={}
    # A later successful history supersedes an older one; never overwrite an
    # immutable downloaded snapshot or assume a failed fetch was current.
    for f in sorted((ART/'wandb').glob('*/histories/*.json')):
        p=read(f);rid=p['run_id'];old=freshness.get(rid,('',None))
        if p['fetched_at_utc']>old[0]:cache[rid]=p;freshness[rid]=(p['fetched_at_utc'],str(f))
    for f in (ROOT/'output/reports/wandb_cc_20261013/histories').glob('*.json'):
        p=read(f)
        if p['run_id'] not in cache:cache[p['run_id']]=p;freshness[p['run_id']]=(p['fetched_at_utc'],str(f))
    epochs={};manifest=[]
    for rid,p in cache.items():
        record=records.get(rid)
        if record is None:continue
        valid=[r for r in p['rows'] if r.get('epoch') is not None and all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) for k in METRICS)]
        groups={}
        for row in valid:groups.setdefault(int(row['epoch']),[]).append(row)
        epochs[rid]={e:{k:statistics.fmean(v[k] for v in rr) for k in METRICS} for e,rr in groups.items()}
        c=record['config'];mc=model(c);d=mc.get('decoder',{}).get('init_diffusion',{}) or {};trainer=c.get('trainer',{});data=c.get('data',{})
        item=dict(id=rid,name=record['name'],state=record['state'],training_epoch=record['summary'].get('epoch'),history_fetched_at_utc=freshness[rid][0],history_file=freshness[rid][1],history_sha256=sha(freshness[rid][1]),
            validation_epochs=sorted(groups),last_validation_epoch=max(groups) if groups else None,
            last_validation_metrics=epochs[rid][max(groups)] if groups else None,
            conditions=dict(seed=c.get('seed'),val_batch_size=data.get('val_batch_size'),limit_val_batches=trainer.get('limit_val_batches'),check_val_every_n_epoch=trainer.get('check_val_every_n_epoch'),
                sampling_steps=d.get('sampling_steps',d.get('ddpm_steps')),layers=d.get('denoiser_num_layers'),map_condition_types=d.get('map_condition_types','road_edges'),a2a_neighbors=d.get('a2a_max_num_neighbors',30),map_neighbors=d.get('map2agent_max_num_neighbors',d.get('map_neighbors',30)),
                precision=trainer.get('precision'),matmul=c.get('float32_matmul_precision'),ckpt_path=c.get('ckpt_path'),initial_map_checkpoint=mc.get('initial_map_checkpoint')))
        match=re.search(r'init_diffusion-(\d+)-',str(c.get('ckpt_path','')))
        item['continued_from_epoch']=int(match.group(1)) if match else None
        limit=trainer.get('limit_val_batches')
        # W&B normalizes numeric1.0 to1, including its downloadable config.yaml.
        # Lightning treats float1.0 as all batches and int1 as one batch. Never
        # infer a128/256-scene cohort from this lossy serialization alone.
        item['conditions']['validation_scope']=(
            {'status':'full_membership_enforced','scenes':mc.get('sd_expected_scenes',50000)}
            if mc.get('sd_require_full_set') else
            {'status':'ambiguous_numeric_one','scenes':None,'note':'Need actual sd_agent_metrics.json num_samples or original Hydra type; source full_base uses float1.0.'}
            if limit==1 else
            {'status':'configured_batch_cap','scenes':data.get('val_batch_size',0)*limit}
            if isinstance(limit,(int,float)) and limit>1 else
            {'status':'unverified','scenes':None})
        manifest.append(item)
    by={r['id']:r for r in manifest};comparisons=[]
    if BASE in epochs:
        for rid,ee in epochs.items():
            if rid==BASE:continue
            if by[rid]['continued_from_epoch'] is not None:continue
            for e in sorted(set(ee)&set(epochs[BASE])):
                a,b=ee[e],epochs[BASE][e];ratios={k:a[k]/b[k] for k in METRICS}
                ca,cb=by[rid]['conditions']['validation_scope'],by[BASE]['conditions']['validation_scope']
                same=ca['scenes'] is not None and ca['scenes']==cb['scenes']
                comparisons.append(dict(candidate=rid,baseline=BASE,epoch=e,ratios=ratios,observed_R_focus=statistics.fmean(ratios[k] for k in FOCUS),observed_R_jsd=statistics.fmean(ratios[k] for k in METRICS[:6]),same_configured_validation_size=same,
                    caveat='Single remote run; cohort hashes/GEN seeds unverified. Different validation sizes cannot rank. N40/depth5 outside current constraints.'))
    # The drift experiment starts from epoch95 weights. Its epoch31 is a
    # further32 epochs, not a fresh32-epoch model; compare it with its parent.
    continuation=[]
    if 'iege6y14' in epochs and '4bo59qfn' in epochs:
        offset=by['iege6y14']['continued_from_epoch']
        if offset is not None:
            for e,a in sorted(epochs['iege6y14'].items()):
                absolute=offset+1+e
                if absolute not in epochs['4bo59qfn']:continue
                b=epochs['4bo59qfn'][absolute];rr={k:a[k]/b[k] for k in METRICS}
                continuation.append(dict(candidate='iege6y14',parent='4bo59qfn',stage_epoch=e,parent_equivalent_epoch=absolute,ratios=rr,observed_R_focus=statistics.fmean(rr[k] for k in FOCUS),caveat='Weights-only continuation with changed LR schedule/optimizer and possible source changes; observational, not a paired LR ablation.'))
    analysis=dict(generated_utc=now(),inventory_fetched_at_utc=read(folder/'inventory.json')['fetched_at_utc'],read_only=True,manifest=manifest,comparisons=comparisons,continuation_comparisons=continuation,
        note='W&B numeric1 cannot distinguish int1 batch from float1.0 full validation. Earlier128/256-scene inference withdrawn. Actual report membership determines scope. Remote evidence guides ideas; local paired tests determine promotion.')
    save(folder/'analysis.json',analysis);save(ART/'wandb/latest_analysis.json',analysis)
    print('LATEST '+json.dumps([dict(id=r['id'],epoch=r['last_validation_epoch'],metrics={k:r['last_validation_metrics'][k] for k in FOCUS} if r['last_validation_metrics'] else None) for r in manifest if r['id'] in {'myqhg1rq','8bblh5do','zww9tc6r','4e89o3nz'}]),flush=True)
    return analysis

def snapshot():
    folder=ART/'wandb'/datetime.now().strftime('%H%M%S');folder.mkdir(parents=True,exist_ok=False)
    api.inventory(folder)
    inventory=read(folder/'inventory.json');records=inventory['runs']
    tracked={BASE,'4bo59qfn','iege6y14','arpfw0fr','az3mgqvh','myqhg1rq','8bblh5do','zww9tc6r','4e89o3nz','nhulkb5k'}
    tracked.update(r['id'] for r in records[:12] if 'init_diffusion' in r['name'] or r['name'].startswith('cc_'))
    previous=read(ART/'wandb/last_signatures.json') if (ART/'wandb/last_signatures.json').exists() else {}
    selected=[];signatures={}
    for r in records:
        if r['id'] not in tracked:continue
        sig=hashlib.sha256(json.dumps(r['summary'],sort_keys=True).encode()).hexdigest();signatures[r['id']]=sig
        if sig!=previous.get(r['id']):selected.append(r['id'])
    if selected:api.histories(folder,selected)
    failed={v['run_id'] for v in read(folder/'history_fetch_status.json')['errors']} if selected else set()
    for rid in failed:signatures.pop(rid,None)
    save(ART/'wandb/last_signatures.json',signatures)
    return analyze(folder)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--analyze',type=Path);args=parser.parse_args()
    if args.analyze:analyze(args.analyze)
    else:
        while time.time()<DEADLINE:
            try:
                result=snapshot();save(ART/'wandb/watcher_status.json',dict(status='SNAPSHOT_COMPLETE',updated_utc=now(),inventory_utc=result['inventory_fetched_at_utc'],pid=os.getpid()))
            except Exception as e:
                save(ART/'wandb/watcher_status.json',dict(status='READ_ONLY_REFRESH_FAILED',updated_utc=now(),error_type=type(e).__name__,pid=os.getpid()))
                print('REFRESH_FAILED '+type(e).__name__,flush=True)
            if not args.watch:break
            # Short sleeps maintain a responsive deadline/interrupt check.
            next_refresh=min(DEADLINE,time.time()+1800)
            while time.time()<next_refresh:time.sleep(min(10,next_refresh-time.time()))
