"""Registered recovery: completed4k prefix, same-arm cyclic resume, fresh trials.

Original failed6826 output is preserved and never counted as complete. Resumed
models keep the original fresh lineage, optimizer, EMA, scheduler and RNG.
"""
from __future__ import annotations
import argparse
import copy
import csv
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_evening_20261013 as core
from tools import run_init_diffusion_657_iteration as native
read,save,sha,now=core.read,core.save,core.sha,core.now
OLD_ART=core.ART;OLD_RUNS=core.RUNS
ART=OLD_ART/'recovery';RUNS=OLD_RUNS/'recovery'
core.ART,core.RUNS=ART,RUNS
old_sources=core.sources
core.sources=lambda:{**old_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
original_stage=core.run_stage
original_worker=core.worker
original_verify=core.verify
original_execute=core.execute
os.environ['PYTHONFAULTHANDLER']='1'

def verify(rows,spec):
    if not spec.get('resume'):return original_verify(rows,spec)
    import yaml
    cfg=yaml.safe_load((RUNS/spec['label']/spec['arm']/'config.yaml').read_text())
    d=cfg['model']['model_config']['decoder']['init_diffusion']
    assert d['component_loss']==core.recipe(spec['arm'])
    cp=spec['resume'][spec['arm']]
    saved=spec['resume_step']
    assert len(rows)==2
    for r in rows:
        assert r['budget']==spec['steps'] and r['seed']==spec['seed'] and r['variant']==spec['arm']
        assert r['optimizer_continued'] and r['resume_cyclic_epochs'] and r['resumed_from']==cp and r['resume_global_step']==saved
        assert r['seen_training_examples']==r['budget']//964*493542+r['budget']%964*512
        assert r['generator_num_parameters']==(8092299 if spec['arm']=='lane_speed_head32' else 8083978)
        assert r['map_frozen'] and r['evaluation_weights']=='ema'
        report=read(r['metrics_path']);actual=report['decoder']
        assert report['num_samples']==report['num_gt_samples']==4096
        assert report['map_source']=='reference' and report['reference_mode']=='matched'
        assert actual['sampling_steps']==20 and actual['ema_num_updates']==spec['steps'] and actual['use_ema']
        assert report['runtime']['trainer_precision']=='32-true' and report['runtime']['float32_matmul_precision']=='medium'
    train=read(RUNS/spec['label']/spec['arm']/'train_trace.json')
    assert len(train)==spec['steps']-saved and train[0]['step']==saved and train[-1]['step']==spec['steps']-1
    assert all(r['map_grad_norm_before_clip']==0 for r in train)
    assert not d.get('sampling_exact_x0') and not d.get('collision_guidance') and not d.get('warm_start_from_ema')

def worker(path):
    spec=read(path);core.setup()
    if not spec.get('resume'):return original_worker(path)
    rows=native.stage(spec['label'],[spec['arm']],spec['steps'],4096,seed=spec['seed'],checkpoints=(),gens=tuple(core.shared.GENS),resume=spec['resume'])
    verify(rows,spec);print('EVENING_NATIVE_SUCCESS',flush=True)

def run_stage(spec,plan):
    spec=copy.deepcopy(spec)
    if not spec.get('evaluation') and spec['steps']==6826:
        rows=read(ART/'observations.json')
        candidates=[r for r in rows if r['variant']==spec['arm'] and r['seed']==spec['seed'] and r['budget']==4096]
        if candidates:
            cp=candidates[0]['checkpoint'];assert all(r['checkpoint']==cp for r in candidates)
            spec['resume']={spec['arm']:cp};spec['resume_step']=4096;spec['checkpoints']=[]
    return original_stage(spec,plan)

def prepare():
    assert not (ART/'protocol.json').exists()
    old=read(OLD_ART/'protocol.json')
    save(ART/'variants.json',read(OLD_ART/'variants.json'));save(ART/'recipe.json',read(OLD_ART/'recipe.json'))
    rows=read(OLD_ART/'observations.json');save(ART/'observations.json',rows)
    plan=copy.deepcopy(old)
    plan.update(registered_utc=now(),source_hashes=core.sources(),
        original_registration_sha256=sha(OLD_ART/'protocol.json'),original_failure_sha256=sha(OLD_RUNS/'train4817_lane_only_6826/execution.json'),
        cyclic_resume_proof_sha256=sha(OLD_ART/'cyclic_resume_proof/verified_completion.json'),
        recovery_rule='Only same-arm completed4096 checkpoint continuation, with its optimizer/EMA/scheduler/RNG. Fresh candidates remain fresh. Tiny backend numerical divergence disclosed; no bitwise claim.',
        local_input_hashes={n:sha(ART/n) for n in ('variants.json','recipe.json')})
    plan['imported_sources'][str((OLD_ART/'cyclic_resume_proof/verified_completion.json').relative_to(ROOT))]=sha(OLD_ART/'cyclic_resume_proof/verified_completion.json')
    cp=OLD_RUNS/'train4817_lane_only_6826/lane_only/step_4096.ckpt'
    for p in [cp,cp.parent/'initialization.json',cp.parent/'csv/version_0/metrics.csv']:
        plan['imported_sources'][str(p.relative_to(ROOT))]=sha(p)
    save(ART/'protocol.json',plan);core.refresh(rows)
    return plan

def execute(plan):
    cp=str((OLD_RUNS/'train4817_lane_only_6826/lane_only/step_4096.ckpt').resolve())
    spec=dict(arm='lane_only',seed=4817,steps=4096,checkpoints=[],scenes=4096,evaluation={'lane_only':cp},label='recovered4096_train4817')
    rows=run_stage(spec,plan)
    if rows is None:return
    observations=read(ART/'observations.json')
    for r in rows:
        assert r['evaluation_only'] and r['budget']==4096
        r['training_origin']='Verified4096 optimizer updates in the original failed6826 process; prefix evaluated separately'
    observations+=rows;save(ART/'observations.json',observations);core.refresh(observations)
    original_execute(plan)

core.verify=verify;core.worker=worker;core.run_stage=run_stage
# Dispatch children through this wrapper without changing source hashing.
core.__file__=str(Path(__file__).resolve())
# old_sources consults core.__file__; keep the original file explicitly frozen.
orig_driver=ROOT/'tools/run_init_diffusion_evening_20261013.py'
base_sources=core.sources
core.sources=lambda:{**base_sources(),str(orig_driver.relative_to(ROOT)):sha(orig_driver)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',type=Path);args=p.parse_args()
    if args.worker:worker(args.worker)
    elif args.execute:execute(read(ART/'protocol.json'))
    else:plan=prepare();print(plan['registered_utc'],plan['deadline'])
