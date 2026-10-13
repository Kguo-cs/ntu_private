"""Fresh size-output and matching hypotheses after completed N20 guidance.

Same x0 component loss, same seed/map/source/parameter count. Neither short
trial is automatically promoted to a full-training recommendation.
"""
import argparse
import copy
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_matching_followup_20261013 as prior
core=prior.core
read,save,sha,now=prior.read,prior.save,prior.sha,prior.now
BASE=ROOT/'output/reports/init_diffusion_evening_20261013'
ART=BASE/'shape_matching_followup'
RUNS=ROOT/'logs/init_diffusion_evening_20261013/shape_matching_followup'
original_sources=prior.sources


def sources():
    return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'src/smart/diffusion/size_parameterization.py':sha(ROOT/'src/smart/diffusion/size_parameterization.py')}


def prepare():
    assert read(BASE/'guidance_final_only50000/status.json')['status']=='COMPLETE'
    assert not (ART/'protocol.json').exists()
    base=copy.deepcopy(read(BASE/'final_long/rebased/sampler_merge/variants.json')['freshK8'])
    shape=copy.deepcopy(base);shape['__config'][prior.P+'shape_x0_parameterization']='time_residual'
    match=copy.deepcopy(base);match['__config'][prior.P+'matching_weights']=dict(position=1.,heading=4.,length=1.,width=1.,velocity=4.)
    save(ART/'variants.json',dict(K8_size_residual=shape,K8_match4=match))
    save(ART/'recipe.json',read(BASE/'matching_followup/recipe.json'))
    save(ART/'controls.json',read(BASE/'matching_followup/controls.json'))
    old=read(BASE/'matching_followup/protocol.json')
    plan=copy.deepcopy(old)
    plan.update(registered_utc=now(),source_hashes=sources(),
                local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','controls.json')},
                initial_order=['K8_size_residual','K8_match4'],TRAIN=[4817],GEN=[2817,3817],
                source_change='Optional shape output s0=s_t+t*r in existing linear/log state. No layers, RNG draws, learned parameters or x0 loss changes. Default identity retained; legacy extra-state gains identity fallback.',
                unit_tests={'size_parameterization':8,'time_grid':6},
                deadlines='No stage unless remaining seconds>=updates*.40+180; enforce original22:30 cutoff. Actual completed K8 rate approximately.356 s/update.',
                selection='Paired vs same-budget roadK8. No short-only full-run promotion; if one trial eligible and time remains, continue its own lineage to6826 for a second-budget check.')
    save(ART/'protocol.json',plan);save(ART/'observations.json',[]);core.status('REGISTERED')


def execute():
    def halt(*unused):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    plan=read(ART/'protocol.json');rows=read(ART/'observations.json')
    def run(arm,target,resume=None,start=0):
        if core.STOP or core.DEADLINE-time.time()<(target-start)*.40+180:
            core.status('NO_TIME_FOR_STAGE',arm=arm,target=target);return False
        spec=dict(arm=arm,seed=4817,steps=target,scenes=4096,manifest=str(core.shared.EVAL_MANIFEST),
                  checkpoints=[],label=f'{arm}_T4817_{target}')
        if resume:spec.update(resume={arm:resume},resume_step=start)
        value=core.run_stage(spec,plan)
        if value is None:return False
        rows.extend(value);save(ART/'observations.json',rows);prior.refresh(rows)
        return True
    for arm in plan['initial_order']:
        if not run(arm,4096):return
    candidates=[c for c in prior.refresh(rows) if c['eligible'] and c['budget']==4096]
    if candidates:
        best=min(candidates,key=lambda c:c['R_focus'])['arm']
        cp=next(r['checkpoint'] for r in rows if r['variant']==best and r['budget']==4096)
        if not run(best,6826,resume=cp,start=4096):return
    core.status('COMPLETE_HYPOTHESES',observations=len(rows),full128epoch_verified=False)


prior.ART,prior.RUNS=ART,RUNS
prior.evidence.ART,prior.evidence.RUNS=ART,RUNS
prior.runtime.ART,prior.runtime.RUNS=ART,RUNS
core.ART,core.RUNS,core.sources,core.verify=ART,RUNS,sources,prior.verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    execute() if args.execute else prepare()
