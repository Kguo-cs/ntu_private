"""Full50K guidance diagnostics even when4096 focus gains fail promotion."""
import argparse
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import evaluate_init_n20_evening_20261013 as sampler
read,save,sha=sampler.read,sampler.save,sampler.sha
BASE=sampler.ART.parent
ART=BASE/'guidance_full50000'
RUNS=ROOT/'logs/init_diffusion_evening_20261013/guidance_full50000'
STOP=False


def prepare():
    if (ART/'protocol.json').exists():return
    save(ART/'protocol.json',dict(registered_local=__import__('datetime').datetime.now().astimezone().isoformat(),
                                 purpose='User asks whether collision_guidance should be considered. Test weak andcc strength onfull50K even if they do not beat focus by2% in screening.',
                                 options=['off_uniform','guide_light','guide_cc'],scenes=50000,TRAIN=[4817],GEN=[2817,3817],
                                 checkpoint_updates=16384,sampling_steps=20,NFE=20,parameters=8083978,
                                 fit_executed=False,primary='JSD preservation; collision auxiliary.',
                                 decision='Non-inferior optional guidance only if R_focus<=1.01, all six JSD ratios<=1.05, onroad decline<=.02, collision ratio<=.5. Not default for JSD unless focused score also improves.',
                                 sources={**sampler.source_hashes(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}))
    save(ART/'status.json',dict(status='WAITING_FOR_N20_COMPLETE'))


def execute():
    global STOP
    def halt(*unused):
        global STOP
        STOP=True;sampler.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    plan=read(ART/'protocol.json')
    while not STOP and time.time()<sampler.END:
        status=read(BASE/'n20_sampler/status.json')
        if status['status']=='COMPLETE':break
        if status['status'] not in ('WAITING_FOR_REGISTERED_TRAIN','RUNNING_NATIVE_TEST'):
            save(ART/'status.json',dict(status='DEPENDENCY_INCOMPLETE',dependency=status));return
        time.sleep(10)
    if STOP:return
    assert plan['sources']=={**sampler.source_hashes(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    oldrows=read(BASE/'n20_sampler/observations.json')
    rows=[]
    for r in oldrows:
        if r['arm']=='freshK8' and r['scenes']==50000 and r['option'] in plan['options']:
            proof=read(Path(r['folder'])/'verified_completion.json')
            assert proof['status']=='VERIFIED_COMPLETE'
            for name,value in proof['artifact_sha256'].items():assert sha(Path(r['folder'])/name)==value
            rows.append(r)
    assert len([r for r in rows if r['option']=='off_uniform'])==2
    save(ART/'imported_verified_results.json',dict(original_observations_sha256=sha(BASE/'n20_sampler/observations.json'),rows=rows))
    origin=read(BASE/'n20_sampler/frozen_endpoints.json')['freshK8']
    files=sorted(p.name for p in (ROOT/'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    assert len(files)==50000
    sampler.ART,sampler.RUNS=ART,RUNS
    for option in ('guide_light','guide_cc'):
        for gen in (2817,3817):
            if any(r['option']==option and r['GEN_seed']==gen for r in rows):continue
            value=sampler.cell('freshK8',origin,gen,option,files,'full50000')
            value['sources'].update({str(Path(__file__).relative_to(ROOT)):sha(__file__)})
            if STOP or not sampler.run_cell(value,rows):return
    comparisons=sampler.compare(rows,'freshK8',50000)
    for c in comparisons:
        c['optional_noninferior_guidance']=(c['option']!='off_uniform' and c['R_focus']<=1.01
                                            and all(c['ratios'][k]<=1.05 for k in sampler.JSD)
                                            and c['onroad_delta']>=-.02 and c['ratios']['collision_rate']<=.5)
    save(ART/'comparisons.json',comparisons)
    save(ART/'status.json',dict(status='COMPLETE',completed_cells=len(rows),actual_full50000=True,full128epoch_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    execute() if args.execute else prepare()
