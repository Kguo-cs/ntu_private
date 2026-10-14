"""Same-checkpoint N20/N24 ablations after the owned training queue finishes."""
from __future__ import annotations
import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_expanded_20261014 as search
from tools import score_init_sd_vw_targets_20261014 as targets
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART/'samplers'
RUNS=search.RUNS/'samplers'
MODES={
    'n20':dict(sampling_steps=20,sampling_time_grid=None,collision_guidance=None),
    'n24_uniform':dict(sampling_steps=24,sampling_time_grid=None,collision_guidance=None),
    'n24_terminal05':dict(sampling_steps=24,sampling_time_grid={'terminal_time':.05},collision_guidance=None),
    'n20_guidance':dict(sampling_steps=20,sampling_time_grid=None,collision_guidance=dict(
        scale=.5,t_max=.3,iters=5,margin=.1,vehicles_only=True)),
}


def evaluate(arm,mode,scenes=4096):
    source=next(r for r in read(search.ART/'observations.json') if
        r['variant']==arm and r['seed']==4817 and r['budget']==14336 and r['num_samples']==4096)
    label=f'{arm}_{mode}_N{scenes}'
    folder=RUNS/label
    if (folder/'verified.json').exists():return read(folder/'summary.json')
    if search.remaining()<max(180,scenes/4096*110)+120:return None
    if search.engine.occupied():raise RuntimeError('GPU has another owner; no simultaneous evaluation')
    folder.mkdir(parents=True,exist_ok=False)
    sampler=ART/(label+'.sampler.json');save(sampler,MODES[mode])
    paths=ART/(label+'.checkpoint.json');save(paths,{arm:source['checkpoint']})
    before=sha(source['checkpoint'])
    command=[search.engine.shared.PYTHON,str(ROOT/'tools/runtime_checkpointed_sweep.py'),
        '--map-checkpoint',str(search.engine.shared.MAP),'--batch-size','512',
        '--scenes',str(scenes),'--eval-batch-size','128','--workers','0',
        '--defer-cyclic-gc','--young-gc-interval','0','--allow-multiple-epochs',
        '--seed','4817','--eval-seed','2817','--extra-eval-seed','3817',
        '--normalizer-json',str(search.engine.shared.NORMALIZER),'--evaluation-weights','ema',
        '--steps','14336','--checkpoint-steps','','--variants','',
        '--evaluation-checkpoints',str(paths),'--evaluation-sampler',str(sampler),
        '--output',str(folder),'--eval-manifest',str(search.ART/'full50000.json' if scenes==50000 else search.engine.shared.EVAL_MANIFEST)]
    if scenes==50000:command.append('--require-full-set')
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MPLCONFIGDIR':'/tmp/sim_expanded_sampler_mpl','PYTHONFAULTHANDLER':'1','SIM_SWEEP_DEFER_CYCLIC_GC':'1'}
    with (folder/'console.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(folder/'execution.json',dict(command=command,child_pid=child.pid,started_utc=now(),
            checkpoint_sha256=before,sampler_sha256=sha(sampler)))
        while child.poll() is None:
            save(ART/'status.json',dict(status='RUNNING',label=label,child_pid=child.pid,updated_utc=now()))
            if search.remaining()<=0:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                save(folder/'excluded.json',dict(reason='DEADLINE_PARTIAL',at_utc=now()));return None
            time.sleep(10)
    if child.returncode or 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():
        save(folder/'excluded.json',dict(reason='NATIVE_ERROR',code=child.returncode,at_utc=now()));return None
    assert sha(source['checkpoint'])==before
    rows=read(folder/'summary.json')
    assert len(rows)==2 and {r['eval_seed'] for r in rows}=={2817,3817}
    for r in rows:
        m=read(r['metrics_path'])
        assert r['budget']==14336 and r['num_samples']==scenes and r['evaluation_only']
        assert m['num_samples']==m['num_gt_samples']==scenes and m['reference_mode']=='matched'
        assert m['decoder']['sampling_steps']==MODES[mode]['sampling_steps']
        assert m['decoder']['ema_num_updates']==14336 and r['map_frozen']
        assert all(math.isfinite(r[k]) and r[k]==m['agent_metrics'][k] for k in search.engine.shared.METRICS)
    save(folder/'verified.json',dict(at_utc=now(),checkpoint_unchanged=True,
        sampler=MODES[mode],summary_sha256=sha(folder/'summary.json')))
    return rows


def execute():
    ART.mkdir(parents=True,exist_ok=True)
    save(ART/'plan.json',dict(registered_utc=now(),modes=MODES,source_sha256=sha(__file__),
        primary='Same-checkpoint all-seven SD/VW thresholds; angle and length bottlenecks',
        targets_sha256=sha(targets.TARGET_FILE),
        select='Preserve every per-GEN threshold already passed by N20. Require worst absolute target ratio improvement and angle/length ratios<=1.03, speed/width/lat/nearest<=1.05. Otherwise preserve N20.',
        interpretation='4096 scenes select inference settings; same50K evaluation gives a larger validation sample, not an untouched test set.'))
    while search.remaining()>180:
        status=read(search.ART/'status.json')
        if status['status']=='TRAINING_PLAN_COMPLETE':break
        save(ART/'status.json',dict(status='WAITING_FOR_OWN_TRAINING',updated_utc=now(),training_status=status['status']))
        time.sleep(30)
    else:return
    lock=(search.ART/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    winner=read(search.ART/'long_candidate.json')['candidate']
    records=[];choices={}
    original=read(search.ART/'observations.json')
    for arm in (search.BASE,winner):
        control=[r for r in original if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,14336,4096)]
        arm_scores=[]
        for mode in ('n24_uniform','n24_terminal05','n20_guidance'):
            values=evaluate(arm,mode)
            if values is None:continue
            score=search.paired_score(values,control)
            per_gen=[targets.score(r) for r in values]
            mean_target=targets.score({k:statistics.fmean(r[k] for r in values) for k in targets.TARGETS})
            item=dict(arm=arm,mode=mode,scenes=4096,score=score,rows=values,
                per_GEN_targets=per_gen,mean_target_progress=mean_target)
            records.append(item);arm_scores.append(item);save(ART/'comparisons.json',records)
        original_target=targets.score({k:statistics.fmean(r[k] for r in control) for k in targets.TARGETS})
        def retains_passed_thresholds(item):
            for row in item['rows']:
                before=next(r for r in control if r['eval_seed']==row['eval_seed'])
                if any(before[k]<limit and row[k]>=limit for k,limit in targets.TARGETS.items()):
                    return False
            return True
        eligible=[r for r in arm_scores if retains_passed_thresholds(r)
            and r['mean_target_progress']['worst_target_ratio']<original_target['worst_target_ratio']
            and all(r['score']['ratios'][k]<=1.03 for k in ('ang_dev_jsd','length_jsd'))
            and all(r['score']['ratios'][k]<=1.05 for k in ('speed_jsd','width_jsd','lat_dev_jsd','nearest_dist_jsd'))]
        choices[arm]=min(eligible,key=lambda r:(7-r['mean_target_progress']['passed_metrics'],
            r['mean_target_progress']['worst_target_ratio']))['mode'] if eligible else 'n20'
    save(ART/'selection.json',dict(at_utc=now(),choices=choices))
    full={}
    for arm,mode in choices.items():
        if mode=='n20':
            full[arm]=[r for r in original if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,14336,50000)]
        else:full[arm]=evaluate(arm,mode,50000)
    if all(full.values()):
        score=search.paired_score(full[winner],full[search.BASE])
        target_progress={arm:[dict(GEN=r['eval_seed'],scope=targets.report_scope(read(r['metrics_path'])),
            **targets.score(r)) for r in rows] for arm,rows in full.items()}
        save(ART/'full50000_comparison.json',dict(at_utc=now(),winner=winner,choices=choices,
             score=score,rows=full,full128epoch_verified=False,per_GEN_targets=target_progress,
             conditioning_caveat=read(targets.TARGET_FILE)['comparison_caveat']))
    save(ART/'status.json',dict(status='COMPLETE',updated_utc=now(),choices=choices,remaining_seconds=search.remaining()))


if __name__=='__main__':
    search.configure()
    execute()
