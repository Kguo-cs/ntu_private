"""Finite final-depth5 study: native preflight, then two complete50K GENs."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.fetch_final_depth5_checkpoint_20261015 import fetch,DEST
from tools.run_full_sampler_eval_20261015 import occupied
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'deep_final128_midpoint_v1'
RUNS=ROOT/'logs/init_diffusion_continuous_20261014/deep_final128_midpoint_v1'
RUNNER=ROOT/'tools/eval_full_checkpoint_midpoint_20261015.py'

def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def source():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUNNER,Path(__file__),ROOT/'tools/fetch_final_depth5_checkpoint_20261015.py',*sorted((ROOT/'src/smart').rglob('*.py'))]}

def main():
    ART.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    assert not (ART/'protocol.json').exists()
    frozen=source()
    save(ART/'protocol.json',dict(registered_utc=now(),source_hashes=frozen,TRAIN=4817,GEN=[2817,3817],
        checkpoint='Actual completed bq7z12zy epoch127/step123392; snapshot only after native job exits.',
        stages=[dict(scenes=256,GEN=2817),dict(scenes=50000,GEN=2817),dict(scenes=50000,GEN=3817)],
        sampling=dict(integrator='midpoint',intervals=12,NFE=24,power=1.2,terminal_time=.1,final_network_time=.05),
        evaluation_precision='highest',no_training=True,no_subset_quality_selection=True,
        held_controller_pid=490695,held_controller_rule='Current child490698 continues; resume parent only after this finite evaluation succeeds.',
        limitation='Tests genuine final weights against the absolute same50K target gate. No training-causal claim from sampler or different hardware runs.'))
    while True:
        try:
            metadata=fetch()
        except subprocess.TimeoutExpired:
            save(ART/'status.json',dict(status='WAITING_FOR_SAME_REMOTE_CHECKPOINT_OBSERVATION',updated_utc=now()))
            time.sleep(30);continue
        except subprocess.CalledProcessError as exc:
            if exc.returncode!=255:raise
            save(ART/'status.json',dict(status='WAITING_FOR_SSH_RECONNECT',updated_utc=now()))
            time.sleep(30);continue
        if metadata['status']=='GENUINE_FINAL_CHECKPOINT_READY':break
        save(ART/'status.json',dict(status=metadata['status'],updated_utc=now(),pid=metadata.get('pid')));time.sleep(30)
    import yaml
    cfg=yaml.safe_load((DEST/'original_config.yaml').read_text())
    def relocate(x):
        if isinstance(x,str):return x.replace('/home/guoke/sim',str(ROOT))
        if isinstance(x,list):return [relocate(v) for v in x]
        if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
        return x
    cfg=relocate(cfg)
    cfg['paths']['root_dir']=str(ROOT/'src')
    cfg['paths']['cache_root']=str(ROOT/'src/waymo_data/full')
    for split,key in [('training_map2_sd','train_raw_dir'),('scenario_dreamer_val','val_raw_dir'),('scenario_dreamer_val','test_raw_dir')]:
        cfg['data'][key]=str(ROOT/'src/waymo_data/full'/split)
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
    cfg['paths']['output_dir']=str(ART/'portable_output')
    cfg['paths']['log_dir']=str(ART/'portable_output')
    resolved=OmegaConf.create(cfg)
    # Resolve reference/cache paths while root_dir still points to sim/src.
    # The unchanged evaluator subsequently rebinds its own runtime paths.
    OmegaConf.resolve(resolved)
    OmegaConf.save(resolved,DEST/'config.yaml',resolve=True)
    names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text())
    save(ART/'sampler.json',dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.1)))
    rows=[]
    for scenes,seed in [(256,2817),(50000,2817),(50000,3817)]:
        assert source()==frozen,'Source changed after registration'
        label=f'deep_final128_MP24_N{scenes}_GEN{seed}';folder=RUNS/label
        manifest=ART/(label+'.manifest.json');save(manifest,names if scenes==50000 else names[25000:25256])
        while occupied():
            save(ART/'status.json',dict(status='WAITING_FOR_CURRENT_LOCAL_GPU_STAGE',updated_utc=now(),label=label));time.sleep(15)
        command=[sys.executable,'-u',str(RUNNER),'--checkpoint',str(DEST/'model.ckpt'),
            '--source-config',str(DEST/'config.yaml'),'--sampler',str(ART/'sampler.json'),'--manifest',str(manifest),
            '--output',str(folder),'--seed',str(seed),'--integration','midpoint']
        if scenes==50000:command.append('--full-set')
        log=RUNS/(label+'.log')
        with log.open('x') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/sim_mpl_continuous',WANDB_MODE='disabled'))
            save(ART/'status.json',dict(status='RUNNING',label=label,pid=child.pid,log=str(log),updated_utc=now()))
            rc=child.wait()
        if rc or not (folder/'verified.json').exists():
            save(ART/'status.json',dict(status='FAILED_REQUIRES_REVIEW',label=label,returncode=rc,log=str(log),held_controller_pid=490695));return
        row=json.loads((folder/'verified.json').read_text());assert set(row['observed_denoiser_calls_per_sample_batch'])=={24}
        assert row['checkpoint_step']==123392
        if scenes==50000:
            target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
            report=json.loads((folder/'sd_agent_metrics.json').read_text())
            for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
                        'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):
                assert report[key]==target[key],key
            row['passed_targets']=[key for key,threshold in target['targets'].items() if row['metrics'][key]<threshold]
            row['all7_pass']=len(row['passed_targets'])==7
        rows.append(row);save(ART/'observations.json',rows)
    complete=[r for r in rows if r['scenes']==50000]
    assert len(complete)==2
    save(ART/'decision.json',dict(completed_utc=now(),both_GEN_full50k_all7_pass=all(r['all7_pass'] for r in complete),
        rows=[dict(seed=r['seed'],metrics=r['metrics'],passed_targets=r['passed_targets']) for r in complete],
        training_complete=True,independent_training_seed_unproven=True))
    p=Path('/proc/490695/cmdline')
    if p.exists() and 'run_shared_heading_velocity_isolated_v4_20261015.py' in p.read_bytes().decode():os.kill(490695,signal.SIGCONT)
    save(ART/'status.json',dict(status='FINAL128_MIDPOINT_FULL50K_STUDY_COMPLETE',completed_utc=now(),held_local_controller_resumed=True))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ART.exists():save(ART/'status.json',dict(status='FAILED_REQUIRES_REVIEW',error=type(exc).__name__+': '+str(exc),held_controller_pid=490695,updated_utc=now()))
        raise
