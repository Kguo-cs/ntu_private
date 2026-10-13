"""Fresh, paired matching-weight followup after all frozen N20/50K tests.

At most two matching strengths; preserve the original component loss and map.
No full-training recommendation is based on this short coupling test alone.
"""
import argparse
import copy
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_long_audit_recovery_20261013 as runtime
from tools import run_init_diffusion_long_audit_20261013 as core
from tools import run_init_diffusion_evening_20261013 as evidence

read,save,sha,now=core.read,core.save,core.sha,core.now
BASE=ROOT/'output/reports/init_diffusion_evening_20261013'
ART=BASE/'matching_followup'
RUNS=ROOT/'logs/init_diffusion_evening_20261013/matching_followup'
original_sources=runtime.sources
original_verify=core.verify
P=core.shared.PREFIX


def sources():
    return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'src/smart/diffusion/sampling_time_grid.py':sha(ROOT/'src/smart/diffusion/sampling_time_grid.py')}


def prepare():
    if (ART/'protocol.json').exists():return
    base=read(BASE/'final_long/rebased/sampler_merge/variants.json')['freshK8']
    variants={}
    for strength in (4,2):
        arm=f'K8_match{strength}'
        variants[arm]=copy.deepcopy(base)
        variants[arm]['__config'][P+'matching_weights']=dict(position=1.,heading=float(strength),length=1.,width=1.,velocity=float(strength))
    save(ART/'variants.json',variants)
    save(ART/'recipe.json',read(BASE/'final_long/rebased/sampler_merge/recipe.json'))
    old=ROOT/'logs/init_diffusion_657_edges_20261013'
    controls=[]
    paths=[old/'medium_a2a_k8/summary.json',old/'long4817/summary.json',old/'long5817/summary.json']
    for path in paths:
        rows=read(path)
        proof=read(path.parent/'verified_completion.json')
        assert proof['status']=='VERIFIED_COMPLETE' and proof['summary_sha256']==sha(path)
        controls+=rows
    save(ART/'controls.json',controls)
    plan=dict(registered_utc=now(),deadline='2026-10-13T22:30:00+08:00',source_hashes=sources(),
              inputs={str(p.relative_to(ROOT)):sha(p) for p in (core.shared.MAP,core.shared.NORMALIZER,core.shared.TRAIN_MANIFEST,core.shared.EVAL_MANIFEST)},
              imported_sources={str(p.relative_to(ROOT)):sha(p) for p in paths +
                                [Path(r['metrics_path']) for r in controls] +
                                [Path(r['metrics_path']).parent/'feature_histograms.json' for r in controls]},
              local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','controls.json')},
              fresh_initial_G1_optimizer_EMA=True,training_component_loss_changed=False,source_distribution_changed=False,
              fit_condition='Wait for n20_sampler, full50K guidance and final-only guidance COMPLETE; no overlapping GPU jobs.',
              order='Fresh match4 TRAIN4817/4096; if focus fails, fresh match2 TRAIN4817/4096. Eligible option continues its own lineage to6826. Then fresh TRAIN5817/6826 if time permits.',
              selection='Matched same TRAIN/budget/GEN/manifest and initial shared hashes vs roadK8. R_focus<=.97, every focus<=1.10, native gates pass. No single-TRAIN short winner promotion.',
              precision='32-true',matmul='medium',sampling_steps=20,parameters=8083978,scenes=4096,
              warning='Coupling path cost improves heading/speed but worsens position; this test is a hypothesis, not a full128epoch claim.')
    save(ART/'protocol.json',plan);save(ART/'observations.json',[]);core.status('REGISTERED_WAITING')


def verify(rows,spec):
    if spec.get('resume'):
        original_verify(rows,spec)
        return
    evidence.ART,evidence.RUNS=ART,RUNS
    evidence.verify(rows,spec)
    # Legacy controls have the shared scalar head, so all generator tensors
    # are common; their original complete hash is also the shared hash.
    assert all(r['initial_shared_generator_sha256']==next(c.get('initial_shared_generator_sha256',c['initial_generator_sha256']) for c in read(ART/'controls.json')
                                                         if c['seed']==r['seed'] and c['budget']==spec['steps']) for r in rows)


def refresh(rows):
    control=read(ART/'controls.json');out=[]
    for arm in sorted({r['variant'] for r in rows}):
        for budget in sorted({r['budget'] for r in rows if r['variant']==arm}):
            for seed in sorted({r['seed'] for r in rows if r['variant']==arm and r['budget']==budget}):
                a=[r for r in rows if (r['variant'],r['budget'],r['seed'])==(arm,budget,seed)]
                b=[r for r in control if (r['budget'],r['seed'])==(budget,seed)]
                if len(a)!=2 or len(b)!=2:continue
                for r,c in zip(sorted(a,key=lambda r:r['eval_seed']),sorted(b,key=lambda r:r['eval_seed'])):
                    for key in ('eval_manifest_sha256','train_order_sha256','calibration_sha256','initial_map_sha256'):
                        assert r[key]==c[key],key
                    assert r['initial_shared_generator_sha256']==c.get('initial_shared_generator_sha256',c['initial_generator_sha256'])
                score=evidence.score(a,b)
                score['eligible']=score['gates_pass'] and score['R_focus']<=.97 and all(score['ratios'][k]<=1.10 for k in evidence.FOCUS)
                out.append(dict(arm=arm,budget=budget,seed=seed,reference='roadK8',**score))
    save(ART/'comparisons.json',dict(generated_utc=now(),comparisons=out))
    return out


def execute():
    def halt(*unused):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    while not core.STOP and time.time()<core.DEADLINE:
        status=read(BASE/'guidance_final_only50000/status.json')
        if status['status']=='COMPLETE':break
        if status['status'] not in ('WAITING_FOR_FULL_GUIDANCE_COMPLETE','RUNNING_NATIVE_TEST'):
            core.status('DEPENDENCY_NOT_COMPLETE',dependency=status);return
        time.sleep(10)
    if core.STOP:return
    plan=read(ART/'protocol.json');rows=read(ART/'observations.json')
    def run(arm,seed,target,resume=None,start=0):
        if core.STOP or core.DEADLINE-time.time()<(target-start)*.45+180:
            core.status('NO_TIME_FOR_STAGE',arm=arm,seed=seed,target=target);return False
        spec=dict(arm=arm,seed=seed,steps=target,scenes=4096,
                  manifest=str(core.shared.EVAL_MANIFEST),checkpoints=[],label=f'{arm}_T{seed}_{target}')
        if resume:spec.update(resume={arm:resume},resume_step=start)
        values=core.run_stage(spec,plan)
        if values is None:return False
        rows.extend(values);save(ART/'observations.json',rows);refresh(rows)
        return True
    selected=None
    for arm in ('K8_match4','K8_match2'):
        if not run(arm,4817,4096):return
        match=next(c for c in refresh(rows) if c['arm']==arm and c['budget']==4096)
        if match['eligible']:selected=arm;break
    if selected:
        cp=next(r['checkpoint'] for r in rows if r['variant']==selected and r['budget']==4096)
        if not run(selected,4817,6826,resume=cp,start=4096):return
        longer=next(c for c in refresh(rows) if c['arm']==selected and c['budget']==6826)
        if longer['eligible']:
            if not run(selected,5817,6826):return
    core.status('COMPLETE_MATCHING_HYPOTHESIS',selected=selected,observations=len(rows),full128epoch_verified=False)


runtime.ART,runtime.RUNS=ART,RUNS
core.ART,core.RUNS,core.sources,core.verify=ART,RUNS,sources,verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    execute() if args.execute else prepare()
