"""New user-authorized map/denoiser phase, after the previous GPU queue."""
from pathlib import Path
import copy
import fcntl
import inspect
import math
import os
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_expanded_20261014 as previous
from tools import audit_init_expanded_schedule_20261014 as schedule
engine=previous.engine
read,save,sha,now=previous.read,previous.save,previous.sha,previous.now
PARENT=previous.ART
ART=PARENT/'map_architecture'
RUNS=previous.RUNS/'map_architecture'
P='model.model_config.decoder.init_diffusion.'
OUTER='model.model_config.decoder.'
ARMS=('map_r40_k64','denoise_f32','lane_only_queries','denoise_h4d32')

def register():
    proof=read(ART/'prepared_preflight.json')
    prepared=read(ART/'prepared_sources.json')
    assert proof['status']=='PASS' and proof['legacy_tensor_output_rng_and_checkpoint_parity']
    assert proof['prepared_sources']==prepared
    assert proof['script_sha256']==sha(ROOT/'tools/preflight_init_architecture_axes_20261014.py')
    base=read(PARENT/'variants.json')['roadK8']
    variants={'roadK8':base}
    updates={'map_r40_k64':{OUTER+'pl2pl_radius':40.,OUTER+'pt2pt_neighbor':64},
             'denoise_f32':{P+'denoiser_num_freq_bands':32},
             'lane_only_queries':{P+'map_condition_types':'lanes_only'},
             'denoise_h4d32':{P+'denoiser_num_heads':4,P+'denoiser_head_dim':32}}
    for arm,changes in updates.items():
        variants[arm]=copy.deepcopy(base);variants[arm]['__config'].update(changes)
    for name,value in [('variants.json',variants),('recipe.json',read(PARENT/'recipe.json')),
                       ('full50000.json',read(PARENT/'full50000.json'))]:
        path=ART/name
        if path.exists():assert read(path)==value
        else:save(path,value)
    plan=dict(registered_utc=now(),deadline=previous.END,source_sha256=sha(__file__),
        authorization='User explicitly allowed map radius/neighbors/layers/output mask and independent map/denoiser dimensions, frequencies and heads.',
        prerequisites='Original training coordinator AND original same-checkpoint sampler coordinator terminal, both completed normally. No production source edits before this.',
        screen={'TRAIN':4817,'GEN':[2817,3817],'updates':2048,'scenes':4096,'order':['roadK8',*ARMS]},
        midpoint={'updates':6144,'pair':'New-phase roadK8 and best completed guarded screen candidate. Restore each own optimizer/EMA/RNG.'},
        full='Evaluate both equal6144 checkpoints on complete50000 scenes with both GEN if entire pair plus report reserve fits.',
        ranking='Among completed candidates with speed/width<=1.15, onroad/count gates: min worst(angle_ratio,length_ratio), then R_focus, then R_jsd. Unguarded fallback is explicitly exploratory.',
        scope='N20/no guidance, unchanged657 objective, fixed AIRL map. Radius40/neighbors64 is a joint graph-setting recipe, not an isolated radius effect. All new models start from scratch.',
        capacity='Parameter cap relaxed. Default/legacy weights, real CPU outputs, RNG and old checkpoint loading proved unchanged; structural map training CPU path also checked.',
        reserve_seconds=1200,prepared_sources=prepared,preflight_sha256=sha(ART/'prepared_preflight.json'),
        original_protocol_sha256=sha(PARENT/'protocol.json'),
        superseded_queue='Idle longer confirmation is reallocated to architecture search. TRAIN-only calibration is deferred until this phase ends and fits the remaining budget.')
    path=ART/'queue_plan.json'
    if path.exists():assert {k:v for k,v in read(path).items() if k!='registered_utc'}=={k:v for k,v in plan.items() if k!='registered_utc'}
    else:save(path,plan)
    return plan

def sources():
    files=set(previous.sources())|set(read(ART/'prepared_sources.json'))|{
        'tools/run_init_architecture_axes_20261014.py','tools/runtime_init_architecture_axes_20261014.py',
        'tools/prepare_init_architecture_axes_20261014.py','tools/preflight_init_architecture_axes_20261014.py'}
    return {p:sha(ROOT/p) for p in sorted(files)}

def verify(rows,spec,folder):
    import yaml
    assert len(rows)==2 and {r['eval_seed'] for r in rows}=={2817,3817}
    cfg=yaml.safe_load((folder/spec['arm']/'config.yaml').read_text())
    variant=read(ART/'variants.json')[spec['arm']]
    for key,value in variant['__config'].items():
        actual=cfg
        for part in key.split('.'):actual=actual[part]
        assert actual==value,key
    assert cfg['ckpt_path'] is None
    assert cfg['model']['model_config']['decoder']['init_diffusion']['component_loss']==read(ART/'recipe.json')
    for row in rows:
        assert (row['variant'],row['seed'],row['budget'],row['num_samples'])==(spec['arm'],spec['seed'],spec['steps'],spec['scenes'])
        assert row['map_frozen'] and row['evaluation_weights']=='ema'
        assert row['generator_num_parameters']==row['generator_trainable_parameters']>0
        report=read(row['metrics_path'])
        assert report['num_samples']==report['num_gt_samples']==spec['scenes']
        assert report['map_source']=='reference' and report['reference_mode']=='matched'
        assert report['unverified_generation_timestep_count']==0
        assert report['decoder']['sampling_steps']==20 and report['decoder']['ema_num_updates']==spec['steps']
        assert report['runtime']['trainer_precision']=='32-true' and report['runtime']['float32_matmul_precision']=='medium'
        assert all(math.isfinite(row[k]) and row[k]==report['agent_metrics'][k] for k in engine.shared.METRICS)
    if not spec.get('evaluation'):
        audit=schedule.audit_row(rows[0])
        save(folder/'training_schedule_audit.json',dict(status='PASS',at_utc=now(),audit=audit))

def live_owned(path,needle):
    pid=read(path)['pid']
    if not engine.shared.alive(pid):return False
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert needle in cmd,(pid,cmd)
    return True

def execute():
    plan=register()
    while time.time()<previous.DEADLINE-1800:
        train=live_owned(PARENT/'seed_reference_phase_pid.json','tools/continue_init_seed_reference_20261014.py')
        sampler=live_owned(PARENT/'sampler_all_seven_pid.json','tools/eval_init_expanded_samplers_20261014.py')
        if not train and not sampler:break
        save(ART/'status.json',dict(status='WAITING_FOR_PRIOR_GPU_PHASES',at_utc=now(),training_live=train,sampler_live=sampler,pid=os.getpid()))
        time.sleep(30)
    else:
        save(ART/'status.json',dict(status='NO_TIME_AFTER_PRIOR_PHASES',at_utc=now()));return
    assert read(PARENT/'samplers/status.json')['status']=='COMPLETE'
    assert read(PARENT/'status.json')['status']=='TRAINING_PLAN_COMPLETE'
    with (PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not engine.occupied()
        assert sha(__file__)==plan['source_sha256']
        for name,hashes in plan['prepared_sources'].items():
            assert sha(ROOT/name)==hashes['original_sha256'],name
            staged=ART/'prepared_sources'/name
            assert sha(staged)==hashes['prepared_sha256']
            archived=ART/'applied_source_archive'/name
            archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,archived)
            shutil.copy2(staged,ROOT/name)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/sim_map_architecture_mpl'}
        with (ART/'applied_tests.log').open('w') as log:
            test_code="import sys,unittest;sys.path.insert(0,'tests');suite=unittest.defaultTestLoader.loadTestsFromNames(['test_init_diffusion_map_condition','test_init_diffusion_speed_heads','test_init_diffusion_sep_map']);result=unittest.TextTestRunner().run(suite);sys.exit(0 if result.wasSuccessful() else 1)"
            checks=subprocess.run([engine.shared.PYTHON,'-c',test_code],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if checks.returncode:
            for name in plan['prepared_sources']:shutil.copy2(ART/'applied_source_archive'/name,ROOT/name)
            save(ART/'status.json',dict(status='APPLIED_TEST_FAILED_RESTORED',at_utc=now()));return
        save(ART/'source_application.json',dict(status='PASS',at_utc=now(),prepared_preflight_sha256=plan['preflight_sha256'],applied_tests_sha256=sha(ART/'applied_tests.log')))
        engine.ART,engine.RUNS=ART,RUNS
        engine.DEADLINE,engine.END=previous.DEADLINE,previous.END
        engine.sources,engine.verify=sources,verify
        engine.evidence.score=previous.paired_score
        # Reuse the existing native coordinator verbatim except for its registered runner path.
        runner=inspect.getsource(engine.run)
        old="'tools/runtime_checkpointed_sweep.py'"
        assert runner.count(old)==1
        runner=runner.replace(old,"'tools/runtime_init_architecture_axes_20261014.py'")
        exec(compile(runner,str(ROOT/'tools/run_init_architecture_axes_20261014.py'),'exec'),engine.__dict__)
        input_paths=(engine.shared.MAP,engine.shared.NORMALIZER,engine.shared.EVAL_MANIFEST,engine.shared.TRAIN_MANIFEST)
        source_plan=dict(phase='MAP_DENOISER_ARCHITECTURE',source_hashes=sources(),
                         inputs={str(p):sha(p) for p in input_paths},local_input_hashes={p:sha(ART/p) for p in ('variants.json','recipe.json','full50000.json')},
                         queue_plan_sha256=sha(ART/'queue_plan.json'),parent_protocol_sha256=plan['original_protocol_sha256'])
        save(ART/'protocol.json',source_plan)
        save(ART/'observations.json',[])
        save(PARENT/'map_architecture_phase_transition.json',dict(at_utc=now(),prior_phase_closed=True,
            previous_protocol_sha256=plan['original_protocol_sha256'],new_protocol=str(ART/'protocol.json'),
            default_compatibility_proof=str(ART/'prepared_preflight.json')))
        def halt(*unused):engine.STOP=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        completed=[]
        for arm in ('roadK8',*ARMS):
            if time.time()+2048*.44+150+plan['reserve_seconds']>=previous.DEADLINE:break
            if engine.run(arm,4817,2048) is None:break
            completed.append(arm)
        scores=[x for x in engine.refresh() if x['budget']==2048 and x['scenes']==4096]
        guarded=[x for x in scores if x['ratios']['speed_jsd']<=1.15 and x['ratios']['width_jsd']<=1.15
                 and x['gates']['onroad'] and x['gates']['vehicle_count']]
        ranked=sorted(guarded or scores,key=lambda x:(max(x['ratios']['ang_dev_jsd'],x['ratios']['length_jsd']),x['R_focus'],x['R_jsd']))
        if not ranked:
            engine.status('SCREEN_ONLY_NO_CANDIDATE',completed=completed);return
        winner=ranked[0]['arm']
        save(ART/'selection.json',dict(at_utc=now(),candidate=winner,screen=ranked,exploratory_fallback=not bool(guarded)))
        # Use measured prior full50K runtime for both GEN, with startup and a20% margin.
        old_full=[x for x in read(PARENT/'observations.json') if x['num_samples']==50000]
        full_cost=max(450.,2.4*max(x['eval_seconds'] for x in old_full)+90) if old_full else 1050.
        needed=2*(6144-2048)*.44+300+2*full_cost+plan['reserve_seconds']
        if previous.DEADLINE-time.time()<needed:
            engine.status('SCREEN_COMPLETE_NO_TIME_FOR_PAIRED_MIDPOINT',candidate=winner,needed_seconds=needed);return
        for arm in ('roadK8',winner):
            if engine.run(arm,4817,6144,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
        for arm in ('roadK8',winner):
            if previous.DEADLINE-time.time()<full_cost+plan['reserve_seconds']:
                engine.status('MIDPOINT_COMPLETE_NO_TIME_FOR_FULL_PAIR',candidate=winner);return
            if engine.run(arm,4817,6144,scenes=50000,resume=engine.checkpoint(arm,4817,6144),evaluation=True) is None:return
        engine.status('ARCHITECTURE_PLAN_COMPLETE',candidate=winner,remaining_seconds=previous.DEADLINE-time.time(),
                      next='Independent metric audit, TRAIN-only calibration if time permits, full-config recommendation and final report.')

if __name__=='__main__':
    if '--register-only' in sys.argv:register()
    else:execute()
