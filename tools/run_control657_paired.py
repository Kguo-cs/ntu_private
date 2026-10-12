"""Run the 657 recipe with the saved overnight exposure/seed/evaluation protocol.

Uses existing model, data, optimizer and evaluator unchanged. Exact checkpoints
from two fresh TRAIN trajectories also supply the early screen/medium budgets.
These prefixes are not extra independent training seeds. No remote logging.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'tmp/control657_paired_20261012'
RUNS = ROOT / 'logs/control657_paired_20261012'
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
HISTORY = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011/analysis/actual_observations.json'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(t, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    historical = read(HISTORY)
    source = historical[0]['model_data_source_files']
    for name, digest in source.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Historical source changed: ' + name)
    map_path = ROOT / 'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt'
    normalizer = ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'
    assert sha(map_path) == historical[0]['map_checkpoint_sha256']
    assert sha(normalizer) == historical[0]['calibration_sha256']
    names = read(ROOT / 'docs/experiments/init_diffusion_frozen_map_search_20261011/confirmation4096_files.json')
    if isinstance(names, dict): names = names['files']
    for n in (1024, 4096): save(ART / f'eval{n}_files.json', names[:n])
    first = RUNS / 'train4817/control657'
    specs = [
        {'label': 'train4817', 'seed': 4817, 'budgets': [384,768,2048,3413,4096,6826,10240], 'scenes': 4096},
        {'label': 'train5817', 'seed': 5817, 'budgets': [3413,6826,10240], 'scenes': 4096},
        {'label': 'screen4817', 'seed': 4817, 'budgets': [384,768], 'scenes': 1024,
         'eval_map': {f'control657_step{s}': str(first / f'step_{s}.ckpt') for s in (384,768)}},
        {'label': 'full50000_4817', 'seed': 4817, 'budgets': [10240], 'scenes': 50000,
         'eval_map': {'control657_full': str(first / 'last.ckpt')}},
    ]
    for s in specs:
        ev = s.get('eval_map')
        cmd = [PYTHON, str(ROOT / 'tools/runtime_cached_torch_loader_sweep.py'), '--map-checkpoint', str(map_path),
               '--batch-size','512','--scenes',str(s['scenes']),'--eval-batch-size','128','--workers','0',
               '--defer-cyclic-gc','--young-gc-interval','0','--seed',str(s['seed']),'--eval-seed','2817',
               '--normalizer-json',str(normalizer),'--evaluation-weights','ema','--steps','10240',
               '--checkpoint-steps',','.join(map(str,s['budgets'][:-1])) if not ev else '',
               '--recipe',str(ART / 'recipe.json'),'--variant-file',str(ART / 'variants.json'),
               '--variants','' if ev else 'control657','--output',str(RUNS / s['label']),'--allow-multiple-epochs']
        if s['scenes'] < 50000: cmd += ['--eval-manifest',str(ART / f"eval{s['scenes']}_files.json")]
        else: cmd += ['--require-full-set']
        if s['scenes'] != 1024: cmd += ['--extra-eval-seed','3817']
        if ev:
            p = ART / f"{s['label']}.checkpoint_map.json"; save(p,ev)
            cmd += ['--evaluation-checkpoints',str(p)]
        s['command'] = cmd
    protocol = {'control_commit':'657f1bb8c017f39e06e4e5317f734d6799eb1040',
        'scope':'657 model/loss recipe with original frozen AIRL map and TRAIN calibration; not its native128 fit',
        'fresh_TRAIN_seeds':[4817,5817], 'GEN_seeds':[2817,3817], 'final_exposure_per_TRAIN':5242620,
        'final_equivalent_epochs':5242620/493542, 'reuse_prefix':'Exact early trajectory checkpoints; not independent TRAIN seeds',
        'precision':'32-true','float32_matmul_precision':'medium','source_files':source,
        'recipe_sha256':sha(ART/'recipe.json'),'variants_sha256':sha(ART/'variants.json'),
        'history_sha256':sha(HISTORY),'driver_sha256':sha(__file__),'specs':specs}
    save(ART / 'paired_protocol.json',protocol)
    if not args.execute:
        print(json.dumps(protocol,indent=2)); return
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SIM_SWEEP_DEFER_CYCLIC_GC='1')
    stages=[]
    for s in specs:
        folder=RUNS/s['label']; verified=folder/'verified_completion.json'
        if verified.is_file(): stages.append(read(verified)); continue
        if (folder/'protocol.json').exists(): raise RuntimeError('Inspect incomplete stage before retry: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True)
        record={'label':s['label'],'command':s['command'],'started_utc':datetime.now(timezone.utc).isoformat()}
        print('START '+s['label'],flush=True)
        with (folder/'console.log').open('w') as log:
            child=subprocess.Popen(s['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(ART/'driver_status.json',{'status':'RUNNING','pid':os.getpid(),'child_pid':child.pid,'active_stage':s['label'],'stages':stages})
            record['returncode']=child.wait()
        log=(folder/'console.log').read_text(errors='replace')
        rows=read(folder/'summary.json') if (folder/'summary.json').exists() else []
        expected={(b,g) for b in s['budgets'] for g in ([2817] if s['scenes']==1024 else [2817,3817])}
        good=record['returncode']==0 and 'RUNTIME_LOADER_SUCCESS' in log and len(rows)==len(expected) and {(r['budget'],r['eval_seed']) for r in rows}==expected
        good=good and all(r['num_samples']==s['scenes'] and r['seed']==s['seed'] and r['evaluation_weights']=='ema' for r in rows)
        record.update(status='VERIFIED_COMPLETE' if good else 'FAILED',finished_utc=datetime.now(timezone.utc).isoformat(),observation_count=len(rows),summary_sha256=sha(folder/'summary.json') if rows else None)
        save(folder/'execution.json',record);stages.append(record)
        if not good:
            save(ART/'driver_status.json',{'status':'FAILED','stages':stages}); raise RuntimeError('Incomplete or failed stage: '+s['label'])
        save(verified,record);print('VERIFIED_COMPLETE '+s['label'],flush=True)
    save(ART/'driver_status.json',{'status':'COMPLETE','finished_utc':datetime.now(timezone.utc).isoformat(),'stages':stages})
    print('ALL_PAIRED_CONTROL_STAGES_COMPLETE',flush=True)


if __name__=='__main__': main()
