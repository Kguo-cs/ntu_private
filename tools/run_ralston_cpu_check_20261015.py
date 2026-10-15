"""CPU-only same-weight RK2/grid package screen against completed midpoint."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save
from tools.run_late_checkpoint_midpoint_20261015 import hashes,now,JSD

BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'ralston_cpu_v1'
RUNS=ROOT/'logs/init_diffusion_continuous_20261014/ralston_cpu_v1'
CONTROL=BASE/'ema_raw_cpu_v1'
SNAP=BASE/'full_checkpoint_snapshots/sampler_snapshots_20261015/roadK8'


def evaluate(scenes,seed):
    label=f'ralston24_N{scenes}_GEN{seed}'
    folder=RUNS/label
    assert not folder.exists()
    manifest=ART/(label+'.manifest.json')
    save(manifest,read(BASE/'full_sampler_50k_v2/full50000_files.json')[25000:25000+scenes])
    log=RUNS/(label+'.log')
    command=[sys.executable,'-u','tools/eval_full_checkpoint_ralston_20261015.py',
        '--checkpoint',str(SNAP/'model.ckpt'),'--source-config',str(CONTROL/'cpu_config.yaml'),
        '--sampler',str(ART/'ralston.json'),'--manifest',str(manifest),'--output',str(folder),
        '--seed',str(seed),'--integration','ralston']
    with log.open('x') as out:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
            start_new_session=True,env={**os.environ,'WANDB_MODE':'disabled','MPLCONFIGDIR':'/tmp/sim_mpl_continuous'})
        save(ART/'status.json',dict(status='RUNNING_CPU_ONLY',child_pid=child.pid,label=label,log=str(log)))
        rc=child.wait()
    if rc or not (folder/'verified.json').exists():
        save(ART/'status.json',dict(status='FAILED_REQUIRES_REVIEW',returncode=rc,label=label,log=str(log)))
        raise RuntimeError('Preserve failed trial; no automatic retry')
    row=read(folder/'verified.json')
    report=read(folder/'sd_agent_metrics.json')
    assert set(row['observed_denoiser_calls_per_sample_batch'])=={24}
    assert report['decoder']['sampling_integrator']=='experimental_ralston'
    assert report['decoder']['use_ema'] is True
    row.update(scenes=scenes,seed=seed,backend='CPU')
    return row


def compare(candidate):
    controls=read(CONTROL/'observations.json')
    control=next(r for r in controls if r['use_ema'] and r['scenes']==candidate['scenes'] and r['seed']==candidate['seed'])
    for key in ('checkpoint_sha256','source_config_sha256','manifest_sha256',
                'sampling_source_sha256_per_batch','smart_source_sha256','matmul_precision'):
        assert candidate[key]==control[key],key
    a,b=Path(candidate['output']),Path(control['output'])
    ah,bh=read(a/'feature_histograms.json'),read(b/'feature_histograms.json')
    assert all(ah[k]['matched_gt']==bh[k]['matched_gt'] for k in JSD)
    ar,br=read(a/'sd_agent_metrics.json'),read(b/'sd_agent_metrics.json')
    assert ar['feature_counts_gt']==br['feature_counts_gt']
    return dict(scenes=candidate['scenes'],GEN=candidate['seed'],candidate=candidate['metrics'],
        control=control['metrics'],ratios={k:candidate['metrics'][k]/control['metrics'][k] for k in JSD},
        generated_vehicle_count_ratio=ar['num_generated_vehicles']/br['num_generated_vehicles'],
        onroad_fraction_change=(ar['feature_counts_generated'][1]/ar['num_generated_vehicles']
            -br['feature_counts_generated'][1]/br['num_generated_vehicles']),
        weights_initial_noise_reference_histograms_equal=True)


def main():
    import hashlib,statistics
    ART.mkdir(parents=True,exist_ok=True)
    RUNS.mkdir(parents=True,exist_ok=True)
    assert not (ART/'protocol.json').exists()
    assert read(BASE/'ralston_sampling_proof_20261015.json')['status']=='PASS'
    assert read(CONTROL/'status.json')['status']=='EMA_RAW_CPU_PAIRED_SCREEN_COMPLETE'
    save(ART/'ralston.json',dict(sampling_steps=12,use_ema=True,
        sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.15)))
    frozen=hashes()
    for p in (Path(__file__),ROOT/'tools/rk2_sampling_20261015.py',ROOT/'tools/eval_full_checkpoint_ralston_20261015.py'):
        frozen[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(ART/'protocol.json',dict(registered_utc=now(),source_hashes=frozen,device='CPU',
        checkpoint=read(SNAP/'provenance.json'),GEN=[2817,3817],NFE=24,
        internal_stage_fraction=2/3,stage_weights=[.25,.75],terminal_time=.15,
        control='Same completedroadK8 EMA checkpoint,midpoint24,power1.2,terminal.1 on CPU.',
        selection='BothGEN angle<1,meanangle<=.95,meanspeed/length/width<=1.05,nearest/lateral<=1.1; '
                  'vehiclecount within.5percent,onroad fraction cannot drop>.3percentagepoints.',
        limitation='Integration and terminalgrid package, not a controlled estimate of either change alone. '
                   'CPU vsGPU RNG/numerics differ. RequireGPU/full50K before any recommendation.'))
    rows,pairs=[],[]
    for scenes,seeds in [(256,[2817]),(8192,[2817,3817])]:
        for seed in seeds:
            for name,digest in frozen.items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            row=evaluate(scenes,seed)
            rows.append(row)
            save(ART/'observations.json',rows)
            paired=compare(row)
            save(ART/(f'comparison_N{scenes}_GEN{seed}.json'),paired)
            if scenes==8192:pairs.append(paired)
    ratios={k:statistics.mean(p['ratios'][k] for p in pairs) for k in JSD}
    eligible=(all(p['ratios']['ang_dev_jsd']<1 and .995<=p['generated_vehicle_count_ratio']<=1.005
                  and p['onroad_fraction_change']>=-.003 for p in pairs)
        and ratios['ang_dev_jsd']<=.95
        and max(ratios[k] for k in ('speed_jsd','length_jsd','width_jsd'))<=1.05
        and max(ratios[k] for k in ('nearest_dist_jsd','lat_dev_jsd'))<=1.1)
    save(ART/'selection.json',dict(eligible_for_GPU_confirmation=eligible,ratios=ratios,paired=pairs,
        complete50K_target_success_unproven=True))
    save(ART/'status.json',dict(status='RALSTON_CPU_PAIRED_SCREEN_COMPLETE',completed_utc=now()))


if __name__=='__main__':main()
