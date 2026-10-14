"""Paired fresh-G1 exploration; deadline 2026-10-14 22:30 Asia/Shanghai.

Larger models are allowed. Training comparisons use N20 without guidance;
N24 and guidance are evaluated separately on the same saved checkpoints.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_overnight_20261014 as engine

read, save, sha, now = engine.read, engine.save, engine.sha, engine.now
ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
RUNS = ROOT / 'logs/init_diffusion_expanded_20261014'
END = '2026-10-14T22:30:00+08:00'
DEADLINE = datetime.fromisoformat(END).timestamp()
P = engine.P
BASE = 'roadK8'
SCREEN = ['match4', 'match_v4h2_s2', 'match4_h320d4', 'match4_h256d5',
          'match4_map64', 'match4_attrdist', 'match4_speedhead']


def sources():
    files = set(engine.shared.SOURCE_FILES) | {
        'tools/run_init_overnight_20261014.py', 'tools/run_init_expanded_20261014.py',
        'tools/runtime_checkpointed_sweep.py', 'src/smart/diffusion/size_parameterization.py',
        'src/smart/diffusion/sampling_time_grid.py'}
    return {name: sha(ROOT / name) for name in sorted(files)}


def configure():
    engine.ART, engine.RUNS = ART, RUNS
    engine.END, engine.DEADLINE = END, DEADLINE
    engine.sources, engine.verify = sources, verify
    engine.evidence.score = paired_score


def paired_score(a, b):
    """Pair data/RNG budgets without requiring different networks' weights equal."""
    for row in a:
        control=next(x for x in b if (x['seed'],x['budget'],x['eval_seed'],x['eval_manifest_sha256'])==
                     (row['seed'],row['budget'],row['eval_seed'],row['eval_manifest_sha256']))
        assert row['initial_map_sha256']==control['initial_map_sha256']
        assert row['train_order_sha256']==control['train_order_sha256']
        ah=read(Path(row['metrics_path']).parent/'feature_histograms.json')
        bh=read(Path(control['metrics_path']).parent/'feature_histograms.json')
        assert all(ah[k]['matched_gt']==bh[k]['matched_gt'] for k in engine.shared.METRICS[:6])
    result=engine.evidence.native.score(a,b)
    for value in [result,*result['per_TRAIN'].values()]:
        value['R_jsd']=statistics.fmean(value['ratios'][k] for k in engine.shared.METRICS[:6])
        value['R_focus']=statistics.fmean(value['ratios'][k] for k in engine.evidence.FOCUS)
    return result


def prepare():
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Use a new phase rather than overwrite registered experiments')
    old = ROOT / 'output/reports/init_diffusion_overnight_20261014'
    base = copy.deepcopy(read(old / 'variants.json')[BASE])
    base['__config'].update({P+'map_condition_types': 'road_edges',
        P+'collision_reduction': 'pair_mean', P+'matching_weights': None,
        P+'distribution_loss': None, P+'speed_head_kind': 'shared',
        P+'speed_head_hidden_dim': None})
    variants = {BASE: base}
    for arm in SCREEN:
        variants[arm] = copy.deepcopy(base)
        variants[arm]['__config'][P+'matching_weights'] = dict(
            position=1., heading=4., length=1., width=1., velocity=4.)
    variants['match_v4h2_s2']['__config'][P+'matching_weights'].update(
        heading=2., length=2., width=2.)
    variants['match4_h320d4']['__config'].update({P+'denoiser_hidden_dim': 320,
                                                P+'denoiser_num_layers': 4})
    variants['match4_h256d5']['__config'][P+'denoiser_num_layers'] = 5
    variants['match4_map64']['__config'][P+'map2agent_max_num_neighbors'] = 64
    variants['match4_attrdist']['__config'][P+'distribution_loss'] = dict(
        speed=.1, length=.1, width=.05, time_min=.75, kind='l1', by_type=True)
    variants['match4_speedhead']['__config'].update({P+'speed_head_kind': 'separate',
                                                   P+'speed_head_hidden_dim': 256})
    save(ART / 'variants.json', variants)
    save(ART / 'recipe.json', read(old / 'recipe.json'))
    save(ART / 'full50000.json', read(old / 'full50000.json'))
    # Check the named weights are the known fixed-map checkpoint; never resume
    # the old generator or optimizer through ckpt_path.
    assert engine.shared.sha(engine.shared.MAP) == '610c1f94a0f573fd078e251af37c38c73255b969e52154e66546588996ef99fb'
    inputs = (engine.shared.MAP, engine.shared.NORMALIZER,
              engine.shared.TRAIN_MANIFEST, engine.shared.EVAL_MANIFEST)
    save(ART / 'protocol.json', dict(registered_utc=now(), deadline=END,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=sources(), inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        local_input_hashes={n: sha(ART/n) for n in ('variants.json','recipe.json','full50000.json')},
        screen=[BASE]+SCREEN, screen_steps=2048, midpoint_steps=6144, longer_steps=14336,
        TRAIN=[4817,5817], GEN=[2817,3817], screen_scenes=4096, final_scenes=50000,
        selection='R_focus mean paired speed/angle/length JSD ratios; also report six-JSD R_jsd and attribute R_attr. Collision never ranks candidates.',
        midpoint='Keep matching4 as a reference, best capacity candidate and best remaining candidate; width<=1.25 and focus each<=1.25 preferred, exploratory fallback disclosed.',
        longer='Top2 midpoint arms; compare equal-step roadK8. Rank by R_focus with width<=1.20 and each focus<=1.15 preferred. No early result proves final128epoch performance.',
        second_seed='Winner and roadK8 fresh TRAIN5817 to4096 if deadline allows; full50K prioritized over additional speculative trials.',
        inference='Same checkpoint, same scenes and GEN, compare N20/N24 and guidance separately. Explicit sampler override saved in evaluation config.',
        no_parameter_cap=True, training_sampler_steps=20, allowed_sampling_steps=[20,24],
        max_epochs_schedule=128, batch_size=512, map_frozen=True, pretrained_map=str(engine.shared.MAP),
        precision='32-true', float32_matmul_precision='medium', workers=0,
        pairing='Same TRAIN manifest/order and budget, same GEN. Different architectures cannot share all initial tensors.',
        training_order='Native quick runner repeats one seeded full order. Full run reshuffles, so short-run results remain proxies.',
        source_policy='Hash check before and after each stage; edits require a new registered phase.',
        process_policy='Serial GPU jobs; stop only owned child groups at deadline. Read-only W&B refresh every30min.'))
    save(ART / 'observations.json', [])
    engine.status('REGISTERED_EXPANDED_SEARCH')


def verify(rows, spec, folder):
    import yaml
    assert len(rows)==2 and {r['eval_seed'] for r in rows}=={2817,3817}
    cfg=yaml.safe_load((folder/spec['arm']/'config.yaml').read_text())
    mc=cfg['model']['model_config']; d=mc['decoder']['init_diffusion']
    variant=read(ART/'variants.json')[spec['arm']]
    assert cfg['ckpt_path'] is None and mc['initial_map_checkpoint']==str(engine.shared.MAP)
    expected_recipe = read(ART/'recipe.json')
    for component, override in variant.items():
        if not component.startswith('__'):
            expected_recipe['components'][component].update(override)
    assert d['component_loss']==expected_recipe
    for key,value in variant['__config'].items():
        if key.startswith(P): assert d.get(key[len(P):])==value, key
    for row in rows:
        assert (row['variant'],row['seed'],row['budget'],row['num_samples']) == (
            spec['arm'],spec['seed'],spec['steps'],spec['scenes'])
        assert row['map_frozen'] and row['evaluation_weights']=='ema'
        assert row['generator_trainable_parameters']==row['generator_num_parameters']>0
        report=read(row['metrics_path']); actual=report['decoder']
        assert report['num_samples']==report['num_gt_samples']==spec['scenes']
        assert report['reference_mode']=='matched' and report['map_source']=='reference'
        assert report['unverified_generation_timestep_count']==0
        assert actual['sampling_steps']==20 and actual['ema_num_updates']==spec['steps']
        assert report['runtime']['trainer_precision']=='32-true'
        assert report['runtime']['float32_matmul_precision']=='medium'
        assert all(math.isfinite(row[k]) and row[k]==report['agent_metrics'][k] for k in engine.shared.METRICS)
        if not spec.get('evaluation'):
            assert row['optimizer_continued']==bool(spec.get('resume'))
            assert row['resume_global_step']==spec.get('resume_step',0)
    if not spec.get('evaluation'):
        trace=read(folder/spec['arm']/'train_trace.json')
        assert [r['step'] for r in trace]==list(range(spec.get('resume_step',0),spec['steps']))
        assert all(r['map_grad_norm_before_clip']==0 for r in trace)


def remaining():
    return DEADLINE-time.time()


def candidates(step):
    rows=[r for r in engine.refresh() if r['budget']==step and r['seed']==4817 and r['scenes']==4096]
    guarded=[r for r in rows if r['ratios']['width_jsd']<=1.25 and
             all(r['ratios'][k]<=1.25 for k in engine.evidence.FOCUS) and
             r['gates']['onroad'] and r['gates']['vehicle_count']]
    return sorted(guarded or rows,key=lambda r:r['R_focus']), not bool(guarded)


def execute():
    lock=(ART/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def halt(*unused): engine.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    for arm in [BASE]+SCREEN:
        if engine.run(arm,4817,2048) is None:return
    ranked,fallback=candidates(2048)
    capacity=[r for r in ranked if r['arm'] in ('match4_h320d4','match4_h256d5')]
    if not capacity:
        capacity=sorted([r for r in engine.refresh() if r['budget']==2048 and
            r['arm'] in ('match4_h320d4','match4_h256d5')],key=lambda r:r['R_focus'])
    selected=list(dict.fromkeys(['match4',capacity[0]['arm'],ranked[0]['arm']]))
    if len(selected)<3:
        selected+=next(([r['arm']] for r in ranked if r['arm'] not in selected),[])
    save(ART/'midpoint_selection.json',dict(at_utc=now(),selected=selected,evidence=ranked,
                                         exploratory_fallback=fallback,remaining_seconds=remaining()))
    for arm in [BASE]+selected:
        if engine.run(arm,4817,6144,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
    ranked,fallback=candidates(6144); selected=[r['arm'] for r in ranked[:2]]
    save(ART/'long_selection.json',dict(at_utc=now(),selected=selected,evidence=ranked,
                                     exploratory_fallback=fallback,remaining_seconds=remaining()))
    for arm in [BASE]+selected:
        if engine.run(arm,4817,14336,resume=engine.checkpoint(arm,4817,6144),resume_step=6144) is None:return
    ranked,fallback=candidates(14336);winner=ranked[0]['arm']
    save(ART/'long_candidate.json',dict(at_utc=now(),candidate=winner,evidence=ranked,
                                     exploratory_fallback=fallback,full128epoch_verified=False))
    # Reserve at least 75min for full50K and same-checkpoint sampler comparisons.
    if remaining()>9000:
        for arm in (BASE,winner):
            if engine.run(arm,5817,4096) is None:return
        for arm in (BASE,winner):
            if engine.run(arm,4817,4096,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
    for arm in (BASE,winner):
        if engine.run(arm,4817,14336,scenes=50000,
                      resume=engine.checkpoint(arm,4817,14336),evaluation=True) is None:return
    engine.status('TRAINING_PLAN_COMPLETE',winner=winner,remaining_seconds=remaining(),
                  next='Evaluate N24/guidance separately; inspect latest W&B before recommendation.')


def recover():
    """Adopt a completed native job after a supervisor-only preflight correction.

    Source fencing makes the old driver stop after the current child finishes.
    No child is interrupted, and no incomplete metrics enter observations.
    """
    pid=read(ART/'owned_pids.json')['execute']
    while engine.shared.alive(pid):
        time.sleep(10)
    plan=read(ART/'protocol.json');current=sources()
    changed=[p for p in current if current[p]!=plan['source_hashes'][p]]
    assert changed==['tools/run_init_expanded_20261014.py'],changed
    save(ART/'supervisor_amendment.json',dict(at_utc=now(),changed=changed,
        previous_protocol_sha256=sha(ART/'protocol.json'),previous_sources=plan['source_hashes'],
        current_sources=current,reason='Architecture preflight found the legacy comparator requires identical generator hashes. Compare matched TRAIN/GEN/maps/GT for different architectures. No model, loss, data or native training source changed.'))
    plan['source_hashes']=current;save(ART/'protocol.json',plan)
    for folder in sorted(RUNS.glob('*')):
        if not (folder/'summary.json').exists():continue
        if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
        values=read(folder/'summary.json');record=read(folder/'execution.json')
        verify(values,record['spec'],folder)
        old=read(ART/'observations.json')
        if not any(r['variant']==values[0]['variant'] and r['budget']==values[0]['budget']
                   and r['seed']==values[0]['seed'] for r in old):
            engine.record_rows(values)
        save(folder/'verified_completion.json',dict(status='VERIFIED_NATIVE_COMPLETE_AFTER_SUPERVISOR_CORRECTION',
             updated_utc=now(),summary_sha256=sha(folder/'summary.json'),source_amendment=str(ART/'supervisor_amendment.json')))
    execute()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','execute','watch','refresh','recover'])
    args=parser.parse_args();configure()
    {'prepare':prepare,'execute':execute,'watch':engine.watch,'refresh':engine.refresh,'recover':recover}[args.mode]()
