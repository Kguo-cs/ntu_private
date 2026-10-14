"""User-directed all-seven target phase, preserving matched training evidence."""
from __future__ import annotations
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import extend_init_expanded_speedhead_rng_20261014 as parity
from tools import score_init_sd_vw_targets_20261014 as targets
search=parity.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART
NATIVE_REFRESH=engine.refresh
SCREEN=parity.SCREEN+['match4_length_l1','match4_heading_p1']
RECOVERY=search.RUNS/'roadK8_T4817_6144_N4096/roadK8/step_2560.ckpt'


def sources():
    return {**parity.sources(),**{name:sha(ROOT/name) for name in (
        'tools/continue_init_allseven_20261014.py','tools/score_init_sd_vw_targets_20261014.py')}}


def refresh():
    result=NATIVE_REFRESH()
    targets.refresh()
    return result


def configure():
    parity.configure()
    search.SCREEN=SCREEN
    search.sources=engine.sources=sources
    engine.refresh=refresh


def register():
    import torch
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(ART/'protocol.json')
        assert old['phase']=='SPEED_HEAD_ADDITION',old['phase']
        changed={name for name,value in old['source_hashes'].items() if sha(ROOT/name)!=value}
        assert changed=={'src/smart/diffusion/scale_flow.py','tools/run_init_expanded_20261014.py'},changed
        archive=ART/'registrations/pre_all_seven_targets'
        archive.mkdir(exist_ok=False)
        for name in ('protocol.json','variants.json','observations.json','comparisons.json','status.json',
                     'midpoint_selection.json','speedhead_phase_pid.json','speedhead_rng_phase_pid.json','sampler_driver_pid.json'):
            if (ART/name).exists():shutil.copy2(ART/name,archive/name)
        proof=read(ART/'speedhead_initialization/after.json')
        assert proof['status']=='PASS' and all(proof[k] for k in ('default_tensor_values_unchanged',
             'default_rng_unchanged','separate_shared_tensors_equal','separate_rng_equal'))
        saved=torch.load(RECOVERY,map_location='cpu',weights_only=False)
        assert saved['global_step']==saved['scratch_sweep']['saved_global_step']==2560
        assert len(saved['optimizer_states'])==len(saved['lr_schedulers'])==1
        assert set(saved['scratch_sweep_rng'])=={'python','numpy','torch_cpu','torch_cuda'}
        assert saved['scratch_sweep_rng']['torch_cuda']
        prior=next(r for r in read(ART/'observations.json') if r['variant']==search.BASE and r['budget']==2048)
        for key in ('initial_generator_sha256','initial_map_sha256','train_order_sha256'):
            assert saved['scratch_sweep'][key]==prior[key]
        trace=read(RECOVERY.parent/'recovery_trace.json')
        assert [r['step'] for r in trace]==list(range(2048,2560))
        assert all(r['map_grad_norm_before_clip']==0 for r in trace)
        save(ART/'recovery_2560_proof.json',dict(at_utc=now(),checkpoint=str(RECOVERY),sha256=sha(RECOVERY),
            step=2560,previous_valid_step=2048,trace_sha256=sha(RECOVERY.parent/'recovery_trace.json'),
            complete_optimizer_scheduler_rng=True,default_source_compatibility=proof,
            cause='Previous native process exited -9; no Python traceback. Exact external kill cause unproven.',
            ignored_unsaved_steps='2560 onward from the interrupted attempt; replay begins at checkpoint2560.',
            repeated_epoch_policy='Native resume-cyclic-epochs; same seeded TRAIN order, preserved optimizer/EMA/RNG.'))
        variants=read(ART/'variants.json')
        variants['match4_speedhead_paired']=copy.deepcopy(variants['match4_speedhead'])
        variants['match4_length_l1']=copy.deepcopy(variants['match4'])
        variants['match4_length_l1']['length']={'kind':'l1'}
        variants['match4_heading_p1']=copy.deepcopy(variants['match4'])
        variants['match4_heading_p1']['heading']={'time_power':1}
        save(ART/'variants.json',variants)
        rows=read(ART/'observations.json');excluded=[r for r in rows if r['variant']=='match4_speedhead']
        save(ART/'speedhead_initialization/unpaired_native_results.json',dict(excluded_from_selection=True,
            reason='Full-Flow initialization changed shared SD time-MLP tensors/RNG. Numerical results preserved; rerun exact shared-init pair.',rows=excluded))
        save(ART/'observations.json',[r for r in rows if r['variant']!='match4_speedhead'])
        for folder in search.RUNS.glob('match4_speedhead_T*'):
            if (folder/'summary.json').exists():save(folder/'excluded.json',dict(at_utc=now(),
                reason='FULL_FLOW_SHARED_INITIALIZATION_MISMATCH',replacement='match4_speedhead_paired',valid_native_results=True))
        plan=copy.deepcopy(old)
        plan.update(phase='ALL_SEVEN_SD_VW_TARGETS',phase_registered_utc=now(),source_hashes=sources(),
            previous_protocol_sha256=sha(archive/'protocol.json'),screen=[search.BASE]+SCREEN,
            primary=read(targets.TARGET_FILE)['criterion'],
            selection='Preserve matching4 reference. Rank other same-budget paired trials by max(angle_ratio,length_ratio), then their mean, under speed/width/onroad/count gates. Full50K success requires EVERY threshold on BOTH prespecified GEN; no averaged-R success claim.',
            midpoint='matching4 plus top2 eligible non-matching4 candidates; do not mandate the 16M candidate that lost at screen. All get6144 updates versus equal-budget roadK8.',
            longer='matching4 plus best other midpoint candidate, each to14336 with roadK8. Select via paired angle/length bottleneck with speed/width gates; absolute seven-target progress always reported.',
            new_hypotheses={'match4_length_l1':'Change only length regression MSE to L1: test sensitivity of narrow generated-size peaks to regression estimator; may help or hurt, not known unbiased flow.',
                'match4_heading_p1':'Change only angle time_power2 to1: give noisy heading states more relative supervision; preserve angle MSE and objective. Test early direction recovery vs final denoising.',
                'match4_speedhead_paired':'Complete already prepared initialization-corrected small speed-head ablation; not a primary length hypothesis.'},
            interruption_recovery=str(ART/'recovery_2560_proof.json'),initialization_compatibility=proof,
            validation_caveat='50K is the same validation set used for model selection, not a held-out test. Short training cannot establish final128epoch performance.',
            conditioning_caveat=read(targets.TARGET_FILE)['comparison_caveat'],
            source_change_scope='Flow optional speed-head RNG fix, default tensors/RNG proven exact; supervisor verifier honors component overrides; new analysis/controller only. Evaluator/data/normalizer unchanged.')
        plan['local_input_hashes']['variants.json']=sha(ART/'variants.json')
        for p in (targets.TARGET_FILE,ART/'speedhead_initialization/after.json',ART/'recovery_2560_proof.json'):
            plan['inputs'][str(p.relative_to(ROOT))]=sha(p)
        save(ART/'protocol.json',plan)
        configure();engine.frozen();refresh()
        engine.status('REGISTERED_ALL_SEVEN_TARGETS',targets=targets.TARGETS)


def ranked(step):
    rows=[r for r in refresh() if r['budget']==step and r['seed']==4817 and r['scenes']==4096]
    guarded=[r for r in rows if r['ratios']['speed_jsd']<=1.10 and r['ratios']['width_jsd']<=1.15
             and r['gates']['onroad'] and r['gates']['vehicle_count']]
    def key(r):
        a,l=(r['ratios'][k] for k in ('ang_dev_jsd','length_jsd'))
        return (max(a,l),(a+l)/2,r['R_jsd'])
    return sorted(guarded or rows,key=key),not bool(guarded)


def execute():
    lock=(ART/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def halt(*unused):engine.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    # Finish interrupted control before new screens. No partial metrics are used.
    if engine.run(search.BASE,4817,6144,resume=str(RECOVERY),resume_step=2560,
                  attempt_label='_recovery2560') is None:return
    for arm in ('match4_length_l1','match4_heading_p1','match4_speedhead_paired'):
        if engine.run(arm,4817,2048) is None:return
    order,fallback=ranked(2048)
    selected=['match4']+[r['arm'] for r in order if r['arm']!='match4'][:2]
    save(ART/'midpoint_selection.json',dict(at_utc=now(),selected=selected,evidence=order,
        exploratory_fallback=fallback,criterion='Angle/length bottleneck; speed/width/native gates',remaining_seconds=search.remaining()))
    for arm in selected:
        if engine.run(arm,4817,6144,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
    order,fallback=ranked(6144)
    selected=['match4']+[r['arm'] for r in order if r['arm']!='match4'][:1]
    save(ART/'long_selection.json',dict(at_utc=now(),selected=selected,evidence=order,exploratory_fallback=fallback))
    for arm in [search.BASE]+selected:
        if engine.run(arm,4817,14336,resume=engine.checkpoint(arm,4817,6144),resume_step=6144) is None:return
    order,fallback=ranked(14336);winner=order[0]['arm']
    save(ART/'long_candidate.json',dict(at_utc=now(),candidate=winner,evidence=order,
        exploratory_fallback=fallback,full128epoch_verified=False,all_seven_target_progress=str(ART/'all_seven_target_progress.json')))
    # Prioritize the user's complete50K criterion over more short-seed screens.
    for arm in (search.BASE,winner):
        if engine.run(arm,4817,14336,scenes=50000,resume=engine.checkpoint(arm,4817,14336),evaluation=True) is None:return
    if search.remaining()>11000:
        for arm in (search.BASE,winner):
            if engine.run(arm,5817,4096) is None:return
        for arm in (search.BASE,winner):
            if engine.run(arm,4817,4096,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
    engine.status('TRAINING_PLAN_COMPLETE',winner=winner,remaining_seconds=search.remaining(),
        next='Same-checkpoint N24/guidance ablations; report all seven thresholds with ego-conditioning caveat.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','execute','refresh']);args=p.parse_args()
    if args.mode=='register':register()
    else:
        configure()
        if args.mode=='execute':execute()
        else:refresh()
