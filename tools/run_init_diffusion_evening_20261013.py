"""Deadline-bounded, fresh training with paired speed/angle/length selection.

No sampling/evaluator changes. Frozen source, TRAIN calibration, N20 and fixed
AIRL map. Native completed observations only; legacy controls are hash recorded.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_jsd_round2_20261013 as core
from tools import run_init_diffusion_657_iteration as native
from tools import run_wandb_cc_scratch_20261013 as shared
read,save,sha,now=core.read,core.save,core.sha,core.now
ART=ROOT/'output/reports/init_diffusion_evening_20261013'
RUNS=ROOT/'logs/init_diffusion_evening_20261013'
DEADLINE='2026-10-13T22:30:00+08:00'
P=shared.PREFIX
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
METRICS=shared.METRICS
STOP=False

def remaining():return datetime.fromisoformat(DEADLINE).timestamp()-time.time()
def sources():
    return {**core.sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
def recipe(arm):
    value=read(ART/'recipe.json')
    for k,v in read(ART/'variants.json')[arm].items():
        if not k.startswith('__'):value['components'][k].update(v)
    return value
def status(s,**kw):save(ART/'status.json',dict(status=s,updated_utc=now(),driver_pid=os.getpid(),**kw))

def prepare():
    assert not (ART/'protocol.json').exists()
    old=ROOT/'output/reports/init_diffusion_jsd_round2_20261013/speed_isolation'
    vv=read(old/'variants.json')
    variants={k:copy.deepcopy(vv[k]) for k in ('fresh657','freshK8','lane_only','lane_agentmax05')}
    for name,delta in {
        'lane_speed_p0':{'velocity':{'time_power':0}},
        'lane_heading_p1':{'heading':{'time_power':1}},
        'lane_length_p0':{'length':{'time_power':0}},
        'lane_speed_head32':{'__config':{P+'speed_head_kind':'separate',P+'speed_head_hidden_dim':32}},
        'lane_map64':{'__config':{P+'map2agent_max_num_neighbors':64}},
    }.items():
        variants[name]=copy.deepcopy(variants['lane_only'])
        variants[name]['__config'].update(delta.get('__config',{}))
        variants[name].update({k:v for k,v in delta.items() if k!='__config'})
    # This union is registered now; activation depends on the completed singles.
    variants['lane_focus_combo']=copy.deepcopy(variants['lane_only'])
    variants['lane_focus_combo'].update(velocity={'time_power':0},heading={'time_power':1},length={'time_power':0})
    save(ART/'variants.json',variants);save(ART/'recipe.json',read(old/'recipe.json'))
    imports=[old/'observations.json']
    rows=read(imports[0]); rows=[r for r in rows if r['variant'] in ('fresh657','freshK8','lane_only','lane_agentmax05')]
    for seed in (4817,5817):
        path=ROOT/f'logs/control657_paired_20261012/train{seed}/summary.json';imports.append(path)
        for r in read(path):
            if r['budget'] in (4096,6826,10240):
                r=copy.deepcopy(r);r['variant']='fresh657';rows.append(r)
    unique={}
    for r in rows:
        key=(r['variant'],r['seed'],r['budget'],r['eval_seed'],r['num_samples'])
        # Prefer preceding fresh-source controls over older duplicates.
        if key not in unique:unique[key]=r
    rows=list(unique.values())
    plan=dict(registered_utc=now(),deadline=DEADLINE,primary='mean of matched speed/ang/length JSD ratios; smaller is better',
        secondary='six-JSD R_jsd and R_attr; collision auxiliary only',
        time_orientation='t=0 clean, t=1 noise; smaller inverse-time power redistributes weight toward noisy steps',
        initial_order=[dict(arm='lane_only',seed=4817,steps=6826,checkpoints=[4096])]+[
            dict(arm=a,seed=5817,steps=4096,checkpoints=[2048]) for a in
            ('lane_speed_p0','lane_heading_p1','lane_length_p0','lane_speed_head32','lane_map64')],
        combo_rule='Test registered three-power union only if at least two single powers improve their own target vs lane_only4096 and focus R<=1.03 vs lane_only; need time for confirmation.',
        selection='Among complete TRAIN5817/4096 arms, smallest focus ratio vs matched657 with all attributes<=1.15, onroad decline<=2pp, vehicle count within10%, near/width no more than15% worse than lane_only, lat absolute rise<=.02 or ratio<=1.15, each focus ratio<=1.10. Require >=3% focus improvement vs lane_only for promotion. Baseline may win.',
        confirmation=dict(steps=6826,seeds=[4817,5817],checkpoints=[4096],baseline='lane_only',from_scratch=True),
        expanded='Best eligible longer-budget model and lane_only/fresh657 at16384 scenes, two GEN; 4096 retained, not independent test. Full50000 optional if time allows.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=sources(),inputs={p:sha(ROOT/p) for p in [str(shared.MAP.relative_to(ROOT)),str(shared.NORMALIZER.relative_to(ROOT)),str(shared.TRAIN_MANIFEST.relative_to(ROOT)),str(shared.EVAL_MANIFEST.relative_to(ROOT))]},
        imported_sources={str(p.relative_to(ROOT)):sha(p) for p in imports},
        local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json')},
        sampling_steps=20,parameter_cap=8100000,shared_parameters=8083978,compact_head_parameters=8092299,
        GEN=[2817,3817],scenes=4096,batch_size=512,precision='32-true',matmul='medium',guidance='OFF',
        fresh_G1_optimizer_EMA=True,frozen_pretrained_map=True,quick_workers=0,
        full_profile='num_workers8; check_val_every_n_epoch8',
        no_final128epoch_claim=True,legacy_controls='Resolved configuration, TRAIN/exposure/GT/hash pairing required; differing inactive sources disclosed.',
        architecture_pairing='Common seeded G1 tensor hash excluding optional speed head; branch uses fork_rng to preserve later training noise.')
    save(ART/'protocol.json',plan);save(ART/'observations.json',rows);refresh(rows);return plan

def frozen(plan):
    assert sources()==plan['source_hashes'],'Registered source changed; stop and register a new phase'
    for table in ('inputs','imported_sources'):
        for p,d in plan[table].items():assert sha(ROOT/p)==d,p
    for p,d in plan['local_input_hashes'].items():assert sha(ART/p)==d,p

def setup():native.ART,native.RUNS=ART,RUNS
def verify(rows,spec):
    import yaml
    expected={(b,g) for b in spec['checkpoints']+[spec['steps']] for g in shared.GENS}
    assert {(r['budget'],r['eval_seed']) for r in rows}==expected and len(rows)==len(expected)
    arm=spec['arm']; folder=RUNS/spec['label']/arm
    cfg=yaml.safe_load((folder/'config.yaml').read_text());mc=cfg['model']['model_config'];d=mc['decoder']['init_diffusion']
    assert cfg['ckpt_path'] is None and mc['initial_map_checkpoint']==str(shared.MAP)
    assert d['component_loss']==recipe(arm)
    assert mc['optimizer']=='default' and mc['lr']==.0005 and mc['lr_min_ratio']==.05
    init=read(folder/'initialization.json')
    assert init['resumed_from'] is None and not init['optimizer_continued'] and init['actual_train_offset']==0
    assert read(folder/'train_summary.json')['steps']==spec['steps']
    with (folder/'csv/version_0/metrics.csv').open() as f:assert max(int(r['step']) for r in csv.DictReader(f))==spec['steps']-1
    for r in rows:
        assert r['variant']==arm and r['seed']==spec['seed'] and not r['evaluation_only']
        params=8092299 if arm=='lane_speed_head32' else 8083978
        assert r['generator_num_parameters']==r['generator_trainable_parameters']==params<=8100000
        assert r['seen_training_examples']==r['budget']//964*493542+r['budget']%964*512
        assert r['batch_size']==512 and r['evaluation_weights']=='ema' and r['map_frozen']
        report=read(r['metrics_path']); actual=report['decoder']
        assert report['num_samples']==report['num_gt_samples']==4096
        assert report['reference_mode']=='matched' and report['map_source']=='reference'
        assert report['unverified_generation_timestep_count']==0
        assert report['runtime']['trainer_precision']=='32-true' and report['runtime']['float32_matmul_precision']=='medium'
        assert actual['sampling_steps']==20 and actual['ema_num_updates']==r['budget'] and actual['use_ema']
        assert actual['velocity_representation']=='speed' and actual['type_process']=='separate'
        assert all(actual['fix_ego_'+k] for k in ('position','heading','shape','type')) and not actual['fix_ego_velocity']
        assert actual['init_map_half_extent_m']==50 and actual['init_map_crop']=='square'
        assert not d.get('sampling_exact_x0',False) and not d.get('warm_start_from_ema',False) and not d.get('collision_guidance')
        for k,v in read(ART/'variants.json')[arm]['__config'].items():
            if k.startswith(P):assert d[k[len(P):]]==v
        assert all(math.isfinite(r[k]) and r[k]==report['agent_metrics'][k] for k in METRICS)

def worker(path):
    spec=read(path);setup()
    rows=native.stage(spec['label'],[] if spec.get('evaluation') else [spec['arm']],spec['steps'],spec.get('scenes',4096),
        seed=spec['seed'],checkpoints=tuple(spec['checkpoints']),gens=tuple(shared.GENS),evaluation=spec.get('evaluation'))
    if not spec.get('evaluation'):verify(rows,spec)
    print('EVENING_NATIVE_SUCCESS',flush=True)

def score(a,b):
    for r in a:
        other=next(x for x in b if (x['seed'],x['budget'],x['eval_seed'],x['eval_manifest_sha256'])==(r['seed'],r['budget'],r['eval_seed'],r['eval_manifest_sha256']))
        assert r.get('initial_shared_generator_sha256',r['initial_generator_sha256'])==other.get('initial_shared_generator_sha256',other['initial_generator_sha256'])
        ah=read(Path(r['metrics_path']).parent/'feature_histograms.json');bh=read(Path(other['metrics_path']).parent/'feature_histograms.json')
        assert all(ah[k]['matched_gt']==bh[k]['matched_gt'] for k in METRICS[:6])
    result=native.score(a,b)
    for v in [result,*result['per_TRAIN'].values()]:
        v['R_jsd']=statistics.fmean(v['ratios'][k] for k in METRICS[:6])
        v['R_focus']=statistics.fmean(v['ratios'][k] for k in FOCUS)
    return result

def refresh(rows):
    results=[]
    for reference in ('fresh657','lane_only'):
        base=[r for r in rows if r['variant']==reference]
        for arm in sorted({r['variant'] for r in rows}-{reference,'fresh657'}):
            for budget in sorted({r['budget'] for r in rows if r['variant']==arm}):
                for seed in sorted({r['seed'] for r in rows if r['variant']==arm and r['budget']==budget}):
                    a=[r for r in rows if (r['variant'],r['budget'],r['seed'])==(arm,budget,seed)]
                    if len(a)!=2 or not all(any((c['seed'],c['budget'],c['eval_seed'])==(seed,budget,r['eval_seed']) for c in base) for r in a):continue
                    results.append(dict(arm=arm,budget=budget,reference=reference,**score(a,base)))
    save(ART/'comparisons.json',dict(generated_utc=now(),primary='R_focus',comparisons=results))

def run_stage(spec,plan):
    frozen(plan)
    occupied=shared.gpu_pids()
    if occupied:status('GPU_OCCUPIED',pids=occupied);return None
    spec=copy.deepcopy(spec)
    spec['label']=spec.get('label') or f"train{spec['seed']}_{spec['arm']}_{spec['steps']}"
    folder=RUNS/spec['label'];folder.mkdir(parents=True,exist_ok=False)
    path=ART/(spec['label']+'.spec.json');save(path,spec)
    started=time.monotonic()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/sim_evening_mpl')
    with (folder/'driver_console.log').open('w') as log:
        child=subprocess.Popen([shared.PYTHON,str(Path(__file__)), '--worker',str(path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            status('RUNNING',stage=spec,child_pid=child.pid,elapsed_seconds=time.monotonic()-started,remaining_seconds=remaining())
            if STOP or remaining()<=0:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                status('PARTIAL_EXCLUDED',stage=spec);return None
            time.sleep(10)
    if child.returncode or 'EVENING_NATIVE_SUCCESS' not in (folder/'driver_console.log').read_text():
        status('FAILED_STAGE',stage=spec,returncode=child.returncode);return None
    frozen(plan);marker=read(folder/'verified_completion.json')
    assert marker['status']=='VERIFIED_COMPLETE' and marker['summary_sha256']==sha(folder/'summary.json')
    values=read(folder/'summary.json')
    if not spec.get('evaluation'):verify(values,spec)
    save(folder/'evening_verified_completion.json',dict(status='VERIFIED_COMPLETE',finished_utc=now(),protocol_sha256=sha(ART/'protocol.json'),summary_sha256=sha(folder/'summary.json'),elapsed_seconds=time.monotonic()-started))
    return values

def execute(plan):
    global STOP
    lock=(ART/'driver.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def halt(n,f):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    rows=read(ART/'observations.json')
    def run(spec):
        if remaining()<spec['steps']*.47+180:status('NO_TIME_FOR_NEW_TRAIN',next=spec);return False
        values=run_stage(spec,plan)
        if values is None:return False
        # A longer fresh run re-evaluates the same prefix, replacing a shorter
        # duplicate instead of treating it as another independent replicate.
        keys={(r['variant'],r['seed'],r['budget'],r['eval_seed'],r['num_samples']) for r in values}
        rows[:]=[r for r in rows if (r['variant'],r['seed'],r['budget'],r['eval_seed'],r['num_samples']) not in keys]+values
        save(ART/'observations.json',rows);refresh(rows);return True
    for spec in plan['initial_order']:
        if not run(spec):return
    comparisons=read(ART/'comparisons.json')['comparisons']
    own={'lane_speed_p0':'speed_jsd','lane_heading_p1':'ang_dev_jsd','lane_length_p0':'length_jsd'}
    good=[c for c in comparisons if c['reference']=='lane_only' and c['budget']==4096 and c['TRAIN_seeds']==[5817]
          and c['arm'] in own and c['ratios'][own[c['arm']]]<1 and c['R_focus']<=1.03]
    save(ART/'combo_decision.json',dict(at_utc=now(),evidence=good,activated=len(good)>=2 and remaining()>3*6826*.47+2400))
    if len(good)>=2 and remaining()>3*6826*.47+2400:
        if not run(dict(arm='lane_focus_combo',seed=5817,steps=4096,checkpoints=[2048])):return
    # Verify the current best baseline at the longer exposure, even if no new
    # candidate improves it. This avoids promoting transient4k gains.
    if not run(dict(arm='lane_only',seed=5817,steps=6826,checkpoints=[4096])):return
    comparisons=read(ART/'comparisons.json')['comparisons']
    candidates=[]
    for c in comparisons:
        if c['reference']!='fresh657' or c['budget']!=4096 or c['TRAIN_seeds']!=[5817] or c['arm'] not in plan['initial_order_arms']+['lane_focus_combo']:continue
        direct=next(x for x in comparisons if x['arm']==c['arm'] and x['budget']==4096 and x['reference']=='lane_only' and x['TRAIN_seeds']==[5817])
        guard=all(direct['ratios'][k]<=1.15 for k in ('nearest_dist_jsd','width_jsd'))
        guard=guard and (direct['ratios']['lat_dev_jsd']<=1.15 or direct['metrics']['lat_dev_jsd']<=direct['control_metrics']['lat_dev_jsd']+.02)
        guard=guard and all(direct['ratios'][k]<=1.10 for k in FOCUS)
        if c['gates_pass'] and guard and direct['R_focus']<=.97:candidates.append((c,direct))
    winner=min(candidates,key=lambda v:v[0]['R_focus'])[0]['arm'] if candidates else 'lane_only'
    save(ART/'long_selection.json',dict(selected_utc=now(),winner=winner,candidates=candidates,comparisons_sha256=sha(ART/'comparisons.json'),scope='TRAIN5817/4096; actual longer confirmation follows'))
    if winner!='lane_only':
        for seed in (5817,4817):
            if not run(dict(arm=winner,seed=seed,steps=6826,checkpoints=[4096])):return
    # Need both TRAINs and improvement over the lane baseline for a new-method
    # recommendation. Otherwise the established baseline remains eligible.
    endpoint=[r for r in rows if r['budget']==6826]
    long=score([r for r in endpoint if r['variant']==winner],[r for r in endpoint if r['variant']=='fresh657'])
    direct=score([r for r in endpoint if r['variant']==winner],[r for r in endpoint if r['variant']=='lane_only'])
    promote=winner!='lane_only' and long['all_TRAIN_gates_pass'] and direct['R_focus']<1 and all(v['R_focus']<1 for v in direct['per_TRAIN'].values())
    selected=winner if promote else 'lane_only'
    save(ART/'long_result.json',dict(winner=winner,recommended=selected,promoted=promote,against657=long,against_lane_only=direct,full128epoch_verified=False))
    # Prefer cohort expansion to another tiny-budget search at the deadline.
    for scenes in (16384,50000):
        if remaining()<(scenes/4096)*6*35+180:break
        expanded=[]
        for seed in (4817,5817):
            names=list(dict.fromkeys([selected,'lane_only','fresh657']))
            checkpoints={name:next(r['checkpoint'] for r in endpoint if r['variant']==name and r['seed']==seed) for name in names}
            spec=dict(arm=selected,seed=seed,steps=6826,checkpoints=[],scenes=scenes,evaluation=checkpoints,label=f'expanded{scenes}_train{seed}')
            values=run_stage(spec,plan)
            if values is None:return
            expanded.extend(values)
        result=score([r for r in expanded if r['variant']==selected],[r for r in expanded if r['variant']=='fresh657'])
        direct=score([r for r in expanded if r['variant']==selected],[r for r in expanded if r['variant']=='lane_only'])
        save(ART/f'expanded{scenes}_observations.json',expanded)
        save(ART/f'expanded{scenes}_result.json',dict(arm=selected,budget=6826,scenes=scenes,against657=result,against_lane_only=direct))
    status('COMPLETE',observations=len(rows),remaining_seconds=remaining())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.execute:execute(read(ART/'protocol.json'))
    else:
        plan=prepare();plan['initial_order_arms']=[s['arm'] for s in plan['initial_order'] if s['arm']!='lane_only'];save(ART/'protocol.json',plan)
        print(plan['registered_utc'],plan['deadline'])
