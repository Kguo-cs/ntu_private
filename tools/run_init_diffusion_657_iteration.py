"""Adaptive, from-scratch optimization with exposure-matched 657 controls.

Each GPU stage uses existing SMART training/evaluation. Completed observations
only; budgets and selection rules are saved before execution. EMA, fixed map,
TRAIN calibration and scene manifests remain fixed. Single GPU, local logging.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / 'output/reports/init_diffusion_657_iteration_20261012'
RUNS = ROOT / 'logs/init_diffusion_657_iteration_20261012'
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
PREFIX = 'model.model_config.decoder.init_diffusion.'
METRICS = ('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)
def now(): return datetime.now(timezone.utc).isoformat()


def score(rows, controls):
    keys=lambda r:(r['seed'],r['eval_seed'],r['seen_training_examples'],r['eval_manifest_sha256'])
    base={keys(r):r for r in controls}
    for r in rows:
        if keys(r) not in base: raise RuntimeError('No matched657 control '+str(keys(r)))
        b=base[keys(r)]
        for k in ('batch_size','train_order_sha256','calibration_sha256','map_checkpoint_sha256','initial_map_sha256','evaluation_weights'):
            if r[k]!=b[k]:raise RuntimeError('Pair mismatch '+k)
    seeds=sorted({r['seed'] for r in rows})
    means=lambda values:{k:statistics.fmean(statistics.fmean(r[k] for r in values if r['seed']==s) for s in seeds) for k in METRICS+('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles')}
    a=means(rows);b=means([base[keys(r)] for r in rows]);ratios={k:a[k]/b[k] for k in METRICS}
    gates={k:ratios[k]<=1.15 for k in ('length_jsd','width_jsd','speed_jsd')}
    gates['onroad']=a['generated_onroad_fraction']>=b['generated_onroad_fraction']-.02
    gates['vehicle_count']=.9<=a['num_generated_vehicles']/a['num_gt_vehicles']<=1.1
    per={str(s):score([r for r in rows if r['seed']==s],controls) for s in seeds} if len(seeds)>1 else {}
    return {'R':statistics.fmean(ratios.values()),'R_attr':statistics.fmean(ratios[k] for k in ('length_jsd','width_jsd','speed_jsd')),
            'ratios':ratios,'metrics':a,'control_metrics':b,'gates':gates,'gates_pass':all(gates.values()),
            'all_TRAIN_gates_pass':all(gates.values()) and all(p['gates_pass'] for p in per.values()),
            'TRAIN_seeds':seeds,'GEN_seeds':sorted({r['eval_seed'] for r in rows}),'per_TRAIN':per}


def plans():
    base={'__config':{'model.model_config.optimizer':'default','model.model_config.lr_warmup_steps':0,'model.model_config.lr_min_ratio':.05},'__lr':.0005}
    changes={
        'control657':{},
        'width_p1':{'width':{'time_power':1}},
        'shape_l1':{'length':{'kind':'l1'},'width':{'kind':'l1'}},
        'speed_p2':{'velocity':{'time_power':2}},
        'train_features':{'__config':{PREFIX+'state_feature_normalization':'train'}},
        'lane_centers':{'__config':{PREFIX+'map_condition_types':'lanes_and_edges'}},
        'lr2e4':{'__lr':.0002},
        'depth5':{'__config':{PREFIX+'denoiser_num_layers':5}},
        'mlp_edges':{'__config':{PREFIX+'edge_embedding_type':'mlp'}},
    }
    variants={}
    for name,change in changes.items():
        v=copy.deepcopy(base);v['__config'].update(change.get('__config',{}));v.update({k:x for k,x in change.items() if k!='__config'});variants[name]=v
    return variants


def stage(label,variants,steps,scenes,seed=4817,checkpoints=(),gens=(2817,),evaluation=None,resume=None):
    folder=RUNS/label;done=folder/'verified_completion.json'
    if done.is_file():
        d=read(done)
        if d['summary_sha256']!=sha(folder/'summary.json'):raise RuntimeError('Summary changed')
        return read(folder/'summary.json')
    if (folder/'protocol.json').exists(): raise RuntimeError('Inspect incomplete stage before retry: '+label)
    subset={k:read(ART/'variants.json')[k] for k in variants}
    save(ART/f'{label}.variants.json',subset)
    cmd=[PYTHON,str(ROOT/'tools/runtime_cached_torch_loader_sweep.py'),'--map-checkpoint',str(ROOT/'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt'),
         '--batch-size','512','--scenes',str(scenes),'--eval-batch-size','128','--workers','0','--defer-cyclic-gc','--young-gc-interval','0',
         '--seed',str(seed),'--eval-seed',str(gens[0]),'--normalizer-json',str(ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'),
         '--evaluation-weights','ema','--steps',str(steps),'--checkpoint-steps',','.join(map(str,checkpoints)),
         '--recipe',str(ART/'recipe.json'),'--variant-file',str(ART/f'{label}.variants.json'),'--variants',','.join(variants),
         '--output',str(folder),'--allow-multiple-epochs']
    if len(gens)>1:cmd+=['--extra-eval-seed',str(gens[1])]
    if scenes<50000:cmd+=['--eval-manifest',str(ROOT/f'tmp/control657_paired_20261012/eval{scenes}_files.json')]
    else:cmd+=['--require-full-set']
    if evaluation:
        save(ART/f'{label}.checkpoints.json',evaluation);cmd+=['--evaluation-checkpoints',str(ART/f'{label}.checkpoints.json')]
    if resume:
        if evaluation or set(resume)!=set(variants):raise ValueError('Resume must name each trained arm and cannot evaluate only')
        save(ART/f'{label}.resume.json',resume);cmd+=['--resume-map',str(ART/f'{label}.resume.json'),'--resume-cyclic-epochs']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SIM_SWEEP_DEFER_CYCLIC_GC='1')
    folder.mkdir(parents=True,exist_ok=True)
    record={'stage':label,'started_utc':now(),'command':cmd}
    print('START '+label,flush=True)
    with (folder/'console.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(ART/'status.json',{'status':'RUNNING','driver_pid':os.getpid(),'child_pid':child.pid,'active_stage':label,'started_utc':record['started_utc']})
        record['returncode']=child.wait()
    rows=read(folder/'summary.json') if (folder/'summary.json').exists() else []
    expected={(v,b,g) for v in variants for b in list(checkpoints)+[steps] for g in gens}
    if evaluation: expected={(v,int(__import__('re').search(r'step_(\d+)',p).group(1)) if 'step_' in p else steps,g) for v,p in evaluation.items() for g in gens}
    actual={(r['variant'],r['budget'],r['eval_seed']) for r in rows}
    if record['returncode']!=0 or 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text() or actual!=expected or len(rows)!=len(expected):
        record.update(status='FAILED',actual=list(map(list,actual)),expected=list(map(list,expected)))
        save(folder/'execution.json',record);save(ART/'status.json',record);raise RuntimeError('Stage failed '+label)
    if any(r['num_samples']!=scenes or r['seed']!=seed or r['evaluation_weights']!='ema' for r in rows):raise RuntimeError('Observation scope mismatch')
    record.update(status='VERIFIED_COMPLETE',finished_utc=now(),summary_sha256=sha(folder/'summary.json'),observations=len(rows))
    save(done,record);print('DONE '+label,flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--screen-only',action='store_true');args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    variants=plans()
    if not (ART/'variants.json').exists():save(ART/'variants.json',variants)
    recipe=read(ROOT/'tmp/control657_paired_20261012/recipe.json');save(ART/'recipe.json',recipe)
    protocol={'control':'657f1bb8c017f39e06e4e5317f734d6799eb1040','created_utc':now(),
        'initialization':'fresh G1, fixed AIRL map; TRAIN-only calibration; no candidate warm start',
        'screen':{'steps':[384,768],'scenes':1024,'TRAIN':4817,'GEN':[2817],'arms':list(variants)},
        'medium':{'steps':[2048,4096],'scenes':4096,'TRAIN':4817,'GEN':[2817,3817],'top_candidates':3},
        'long':{'steps':[3413,6826,10240],'scenes':4096,'TRAIN':[4817,5817],'GEN':[2817,3817]},
        'full':{'scenes':50000,'TRAIN':[4817,5817],'GEN':[2817,3817]},
        'selection':'screen: top3 by final R with onroad decline<=.03, then add a union of two complementary single changes; medium: smallest endpoint R among arms with gates and improving/flat R slope, otherwise do not promote; long: both TRAIN must pass gates and pooled R<1; full: require both TRAIN gates and R<1',
        'warning':'Quick curves do not guarantee128epoch convergence; only actual observations count',
        'variants':variants,'recipe':recipe}
    if not (ART/'protocol.json').exists():save(ART/'protocol.json',protocol)
    if not args.execute:print(json.dumps(protocol,indent=2));return
    controls=read(ROOT/'output/reports/control657_paired_20261012/new_control_observations.json')+read(ROOT/'output/reports/control657_paired_20261012/auxiliary_observations.json')
    # Use raw runner summaries as pairing evidence: train_order and initialization are recorded there.
    controls=[]
    for label in ('train4817','train5817','screen4817','full50000_4817'):
        controls+=read(ROOT/f'logs/control657_paired_20261012/{label}/summary.json')
    screening=stage('screen',list(variants),768,1024,checkpoints=(384,))
    rankings=[]
    for name in variants:
        curve=[{'budget':b,**score([r for r in screening if r['variant']==name and r['budget']==b],controls)} for b in (384,768)]
        rankings.append({'variant':name,'curve':curve,'R':curve[-1]['R']})
    save(ART/'screen.json',sorted(rankings,key=lambda x:x['R']))
    if args.screen_only:save(ART/'status.json',{'status':'SCREEN_COMPLETE'});return
    candidates=sorted([r for r in rankings if r['variant']!='control657' and r['curve'][-1]['metrics']['generated_onroad_fraction']>=r['curve'][-1]['control_metrics']['generated_onroad_fraction']-.03],key=lambda x:x['R'])[:3]
    if not candidates:save(ART/'status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'screen onroad gate'});return
    picked=[r['variant'] for r in candidates]
    # A single predeclared adaptive combination tests complementary mechanisms.
    combined=copy.deepcopy(variants[picked[0]])
    for k,v in variants[picked[1]].items() if len(picked)>1 else []:
        if k=='__config':combined[k].update(v)
        elif k!='__lr':combined[k]=v
    if len(picked)>1 and combined!=variants[picked[0]]:
        variants['adaptive_combo']=combined;picked.append('adaptive_combo');save(ART/'variants.json',variants)
    save(ART/'medium_selection.json',{'selected':picked,'selected_utc':now(),'screen_evidence_sha256':sha(ART/'screen.json')})
    medium=stage('medium',picked,4096,4096,checkpoints=(2048,),gens=(2817,3817))
    results=[]
    for name in picked:
        curve=[{'budget':b,**score([r for r in medium if r['variant']==name and r['budget']==b],controls)} for b in (2048,4096)]
        results.append({'variant':name,'curve':curve,'R':curve[-1]['R'],'eligible':curve[-1]['gates_pass'] and curve[-1]['R']<=curve[0]['R']*1.05})
    save(ART/'medium.json',results)
    good=sorted([r for r in results if r['eligible'] and r['R']<1.05],key=lambda x:x['R'])
    if not good:save(ART/'status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'No medium candidate passes gates and slope'});return
    winner=good[0]['variant'];save(ART/'long_selection.json',{'winner':winner,'selected_utc':now(),'evidence_sha256':sha(ART/'medium.json')})
    long=[]
    for seed in (4817,5817):long+=stage(f'long{seed}',[winner],10240,4096,seed=seed,checkpoints=(3413,6826),gens=(2817,3817))
    curves=[{'budget':b,**score([r for r in long if r['budget']==b],controls)} for b in (3413,6826,10240)]
    save(ART/'long.json',{'winner':winner,'curve':curves})
    if not curves[-1]['all_TRAIN_gates_pass'] or curves[-1]['R']>=1:
        save(ART/'status.json',{'status':'COMPLETE_NO_PROMOTION','reason':'Long confirmation fails','winner':winner});return
    full=[]
    for seed in (4817,5817):
        checkpoint=str(RUNS/f'long{seed}'/winner/'last.ckpt')
        full+=stage(f'full{seed}',[],10240,50000,seed=seed,gens=(2817,3817),evaluation={winner:checkpoint})
        if seed==5817:
            c=str(ROOT/'logs/control657_paired_20261012/train5817/control657/last.ckpt')
            controls+=stage('full_control5817',[],10240,50000,seed=seed,gens=(2817,3817),evaluation={'control657':c})
    final=score(full,controls);save(ART/'full.json',{'winner':winner,**final})
    save(ART/'status.json',{'status':'COMPLETE','winner':winner,'beats657':final['all_TRAIN_gates_pass'] and final['R']<1,'finished_utc':now()})


if __name__=='__main__':main()
