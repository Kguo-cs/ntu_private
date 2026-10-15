"""CPU-only paired screen of EMA/raw from one genuine completed checkpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import gc
gc.disable()
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save
from tools.run_late_checkpoint_midpoint_20261015 import hashes,now,JSD

BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'ema_raw_cpu_v1'
RUNS=ROOT/'logs/init_diffusion_continuous_20261014/ema_raw_cpu_v1'
SNAP=BASE/'full_checkpoint_snapshots/sampler_snapshots_20261015/roadK8'


def evaluate(ema,scenes,seed):
    label=f'{"ema" if ema else "raw"}_N{scenes}_GEN{seed}'
    folder=RUNS/label
    assert not folder.exists()
    names=read(BASE/'full_sampler_50k_v2/full50000_files.json')[25000:25000+scenes]
    manifest=ART/(label+'.manifest.json')
    save(manifest,names)
    command=[sys.executable,'-u','tools/eval_full_checkpoint_midpoint_20261015.py',
             '--checkpoint',str(SNAP/'model.ckpt'),'--source-config',str(ART/'cpu_config.yaml'),
             '--sampler',str(ART/('ema.json' if ema else 'raw.json')),
             '--manifest',str(manifest),'--output',str(folder),'--seed',str(seed),'--integration','midpoint']
    log=RUNS/(label+'.log')
    start=time.monotonic()
    with log.open('x') as out:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
            start_new_session=True,env={**os.environ,'WANDB_MODE':'disabled','MPLCONFIGDIR':'/tmp/sim_mpl_continuous'})
        save(ART/'status.json',dict(status='RUNNING_CPU_ONLY',child_pid=child.pid,label=label,log=str(log)))
        rc=child.wait()
    if rc or not (folder/'verified.json').exists():
        save(ART/'status.json',dict(status='FAILED_REQUIRES_REVIEW',label=label,returncode=rc,log=str(log)))
        raise RuntimeError('Review CPU evaluation failure; no automatic retry')
    row=read(folder/'verified.json')
    report=read(folder/'sd_agent_metrics.json')
    assert report['decoder']['use_ema'] is ema
    assert set(row['observed_denoiser_calls_per_sample_batch'])=={24}
    row.update(use_ema=ema,scenes=scenes,seed=seed,seconds=time.monotonic()-start,backend='CPU')
    return row


def pair(a,b):
    assert not a['use_ema'] and b['use_ema']
    for key in ('checkpoint_sha256','source_config_sha256','manifest_sha256',
                'sampling_source_sha256_per_batch','smart_source_sha256','matmul_precision'):
        assert a[key]==b[key],key
    ah=read(Path(a['output'])/'feature_histograms.json')
    bh=read(Path(b['output'])/'feature_histograms.json')
    assert all(ah[k]['matched_gt']==bh[k]['matched_gt'] for k in JSD)
    return dict(GEN=a['seed'],scenes=a['scenes'],backend='CPU',raw=a['metrics'],ema=b['metrics'],
                raw_to_ema={k:(a['metrics'][k]/b['metrics'][k] if b['metrics'][k] else
                              1.0 if not a['metrics'][k] else None) for k in JSD},
                source_noise_and_reference_histograms_equal=True)


def main():
    ART.mkdir(parents=True,exist_ok=True)
    RUNS.mkdir(parents=True,exist_ok=True)
    assert not (ART/'protocol.json').exists()
    from omegaconf import OmegaConf,open_dict
    cfg=OmegaConf.load(SNAP/'config.yaml')
    assert not cfg.model.model_config.decoder.init_diffusion.get('warm_start_from_ema',False)
    with open_dict(cfg):
        cfg.trainer.accelerator='cpu'
        cfg.trainer.devices=1
        cfg.trainer.strategy='auto'
        cfg.trainer.precision='32-true'
        cfg.trainer.enable_progress_bar=False
    OmegaConf.save(cfg,ART/'cpu_config.yaml')
    for mode in ('ema','raw'):
        save(ART/(mode+'.json'),dict(sampling_steps=12,sampling_time_grid=dict(
             kind='power',power=1.2,terminal_time=.1),use_ema=mode=='ema'))
    frozen=hashes()
    save(ART/'protocol.json',dict(registered_utc=now(),checkpoint=read(SNAP/'provenance.json'),
        source_hashes=frozen,device='cpu',CUDA_VISIBLE_DEVICES='',TRAIN='completed checkpoint; no training',
        GEN=[2817,3817],NFE=24,preflight_scenes=256,screen_scenes=8192,
        decision='Check whether EMA helps angle/size/speed before adding generator parameters. '
                 'CPU-only, allowed concurrently with ownedGPU training.',
        limitation='CPU andGPU have different RNG streams/numerics. Compare pairedCPU arms only; '
                   'a winner requires same-cohortGPU and complete50K confirmation. No automatic promotion.'))
    rows=[]
    for scenes,seeds in [(256,[2817]),(8192,[2817,3817])]:
        for seed in seeds:
            assert hashes()==frozen
            ema=evaluate(True,scenes,seed)
            rows.append(ema)
            save(ART/'observations.json',rows)
            raw=evaluate(False,scenes,seed)
            rows.append(raw)
            save(ART/'observations.json',rows)
            comparison=pair(raw,ema)
            comparisons=read(ART/'comparisons.json') if (ART/'comparisons.json').exists() else []
            comparisons.append(comparison)
            save(ART/'comparisons.json',comparisons)
    assert hashes()==frozen
    save(ART/'status.json',dict(status='EMA_RAW_CPU_PAIRED_SCREEN_COMPLETE',completed_utc=now(),
         next='Review CPU-only ratios; requireGPU/full50K before changing weights policy.'))


if __name__=='__main__':
    main()
