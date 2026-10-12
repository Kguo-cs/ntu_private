"""Serial adaptive continuation of the registered 657 optimization experiment.

No GPU work overlaps the first-round driver. All arms start fresh, keep the
fixed map, and are evaluated with paired scene manifests and training exposure.
"""
import argparse
import copy
import os
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_init_diffusion_657_iteration import ART, RUNS, read, save, sha, score, stage, now


def alive(pid):
    try:
        os.kill(pid, 0)
        # Detached drivers may await reaping after finishing; zombies do not
        # own a GPU and must not hold the serial queue indefinitely.
        status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return status != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    plan = read(ART/'round2_plan.json')
    protocol = {
        'created_utc': now(), 'plan_sha256': sha(ART/'round2_plan.json'),
        'driver_sha256': sha(__file__), 'from_scratch': True,
        'screen': {'steps': [384,768], 'scenes':1024, 'TRAIN':4817, 'GEN':[2817],
                   'order':plan['order']},
        'medium': {'steps':[2048,4096], 'scenes':4096, 'TRAIN':4817, 'GEN':[2817,3817]},
        'selection': 'screen top3 by paired R subject to onroad >= control-.03; union of the best2 distinct deltas; medium gates, R<1 and late/early R<=1.15; best goes to10240steps, then secondTRAIN and50k only if both pass',
        'long': {'steps':[3413,6826,10240], 'scenes':4096, 'TRAIN':[4817,5817], 'GEN':[2817,3817]},
        'full': {'scenes':50000, 'TRAIN':[4817,5817], 'GEN':[2817,3817]},
        'wait_pid':args.wait_pid,
        'sources':{p:sha(ROOT/p) for p in ('src/smart/diffusion/denoiser.py','src/smart/diffusion/initial_diffusion.py','src/smart/diffusion/scale_flow.py','src/smart/diffusion/diffusion_utils.py','src/smart/model/smart.py','tools/overnight_init_diffusion_sweep.py','tools/scratch_component_loss_sweep.py')},
    }
    protocol_path = ART/'round2_protocol.json'
    if protocol_path.exists():
        previous = read(protocol_path)
        assert previous['driver_sha256'] == sha(__file__) and previous['plan_sha256'] == protocol['plan_sha256']
    else:
        save(protocol_path, protocol)
    if not args.execute:
        print('Registered round2 arms:',len(plan['order'])); return
    if args.wait_pid:
        save(ART/'round2_status.json', {'status':'WAITING_FOR_SERIAL_GPU','wait_pid':args.wait_pid,'pid':os.getpid()})
        print('WAIT',args.wait_pid,flush=True)
        while alive(args.wait_pid):time.sleep(5)
        prior = read(ART/'status.json')
        if prior['status'] not in ('COMPLETE','COMPLETE_NO_PROMOTION'):
            save(ART/'round2_status.json',{'status':'NEEDS_FIRST_ROUND_INSPECTION','prior':prior})
            raise RuntimeError('Inspect failed first round before continuing')
    assert not (ART/'round2_screen.json').exists(),'Inspect existing round2 results before retry'
    variants = read(ART/'variants.json');variants.update(plan['variants']);save(ART/'variants.json',variants)
    controls = []
    for label in ('train4817','train5817','screen4817','full50000_4817'):
        controls += read(ROOT/f'logs/control657_paired_20261012/{label}/summary.json')
    save(ART/'round2_status.json',{'status':'RUNNING','pid':os.getpid(),'stage':'screen'})
    screen=[]
    for name in plan['order']:
        screen += stage('round2_screen_'+name,[name],768,1024,checkpoints=(384,))
    repeat = [r for r in screen if r['variant']=='control657']
    rankings = []
    for name in plan['variants']:
        curves = [{'budget':b,**score([r for r in screen if r['variant']==name and r['budget']==b],controls)} for b in (384,768)]
        rankings.append({'variant':name,'curve':curves,'R':curves[-1]['R'],
            'contemporaneous_control_curve':[{'budget':b,**score([r for r in screen if r['variant']==name and r['budget']==b],repeat)} for b in (384,768)]})
    rankings.sort(key=lambda r:r['R'])
    save(ART/'round2_screen.json',rankings)
    selected = [r['variant'] for r in rankings if r['curve'][-1]['metrics']['generated_onroad_fraction']>=r['curve'][-1]['control_metrics']['generated_onroad_fraction']-.03][:3]
    if len(selected)>1:
        first = variants[selected[0]];second=variants[selected[1]];base=variants['control657']
        combo=copy.deepcopy(first)
        for key,value in second.items():
            if key=='__config':
                combo[key].update({k:v for k,v in value.items() if v!=base[key].get(k)})
            elif key!='__lr' and value!=base.get(key):
                combo.setdefault(key,{}).update(value)
        if combo!=first and combo!=second:
            variants['round2_combo']=combo;selected.append('round2_combo');save(ART/'variants.json',variants)
    save(ART/'round2_medium_selection.json',{'selected':selected,'evidence_sha256':sha(ART/'round2_screen.json'),'selected_utc':now()})
    if not selected:
        save(ART/'round2_status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'screen onroad gate'});return
    medium=[]
    for name in selected:
        medium += stage('round2_medium_'+name,[name],4096,4096,checkpoints=(2048,),gens=(2817,3817))
    results=[]
    for name in selected:
        curve=[{'budget':b,**score([r for r in medium if r['variant']==name and r['budget']==b],controls)} for b in (2048,4096)]
        results.append({'variant':name,'curve':curve,'R':curve[-1]['R'],
            'eligible':curve[-1]['gates_pass'] and curve[-1]['R']<1. and curve[-1]['R']<=curve[0]['R']*1.15})
    save(ART/'round2_medium.json',results)
    eligible=sorted([r for r in results if r['eligible']],key=lambda r:r['R'])
    if not eligible:
        save(ART/'round2_status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'medium gates/R/slope','finished_utc':now()});return
    winner=eligible[0]['variant'];long=[]
    save(ART/'round2_long_selection.json',{'winner':winner,'evidence_sha256':sha(ART/'round2_medium.json'),'selected_utc':now()})
    for seed in (4817,5817):
        long+=stage(f'round2_long{seed}',[winner],10240,4096,seed=seed,checkpoints=(3413,6826),gens=(2817,3817))
        last=score([r for r in long if r['budget']==10240],controls)
        if not last['all_TRAIN_gates_pass'] or last['R']>=1:
            save(ART/'round2_long.json',{'winner':winner,'curve':[{'budget':b,**score([r for r in long if r['budget']==b],controls)} for b in (3413,6826,10240)]})
            save(ART/'round2_status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'long confirmation fails','winner':winner,'finished_utc':now()});return
    save(ART/'round2_long.json',{'winner':winner,'curve':[{'budget':b,**score([r for r in long if r['budget']==b],controls)} for b in (3413,6826,10240)]})
    full=[]
    for seed in (4817,5817):
        checkpoint=str(RUNS/f'round2_long{seed}'/winner/'last.ckpt')
        full+=stage(f'round2_full{seed}',[],10240,50000,seed=seed,gens=(2817,3817),evaluation={winner:checkpoint})
        if seed==5817:
            c=str(ROOT/'logs/control657_paired_20261012/train5817/control657/last.ckpt')
            controls+=stage('full_control5817',[],10240,50000,seed=seed,gens=(2817,3817),evaluation={'control657':c})
    final=score(full,controls);save(ART/'round2_full.json',{'winner':winner,**final})
    save(ART/'round2_status.json',{'status':'COMPLETE','winner':winner,'beats657':final['all_TRAIN_gates_pass'] and final['R']<1,'finished_utc':now()})


if __name__=='__main__':main()
