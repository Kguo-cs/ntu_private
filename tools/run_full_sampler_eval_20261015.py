"""Serial genuine-checkpoint sampler evaluation after the owned head trial."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save

ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_sampler_eval_v3'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/full_sampler_eval_v3'
SNAPSHOTS = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_checkpoint_snapshots/sampler_snapshots_20261015'


def occupied():
    rows = csv.reader(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
         '--format=csv,noheader,nounits'], text=True).splitlines())
    for row in rows:
        if len(row) != 3:
            raise RuntimeError('Unrecognized GPU process record')
        if row[1].strip() == '/opt/todesk/bin/ToDesk_Session' and int(row[2].strip()) <= 1024:
            continue
        if alive(int(row[0].strip())):
            return True
    return False


def evaluate(model, mode, scenes, seed):
    label = f'{model}_{mode}_N{scenes}_GEN{seed}'
    folder = RUNS / label
    if folder.exists():
        raise RuntimeError('Evaluation directory exists; inspect before retry')
    full = read(ROOT / 'tmp/control657_paired_20261012/eval4096_files.json')
    names = full['files'] if isinstance(full, dict) else full
    manifest = ART / (label + '.manifest.json')
    save(manifest, names[:scenes])
    while occupied():
        time.sleep(10)
    snapshot_name = 'N24match4_final' if model == 'N24match4' else model
    checkpoint = SNAPSHOTS / snapshot_name / 'model.ckpt'
    command = ['/home/ke/miniconda3/envs/sim/bin/python', '-u',
               str(ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'),
               '--checkpoint', str(checkpoint), '--source-config', str(checkpoint.parent / 'config.yaml'),
               '--sampler', str(ART / (mode + '.json')), '--manifest', str(manifest),
               '--output', str(folder), '--seed', str(seed),
               '--integration', 'midpoint' if mode == 'midpoint24' else 'native']
    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / (label + '.log')
    with log.open('x') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, 'WANDB_MODE': 'disabled'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        result = child.wait()
    if result or not (folder / 'verified.json').exists():
        save(ART / 'status.json', dict(status='EVALUATION_FAILED_REQUIRES_REVIEW', label=label,
                                      returncode=result, log=str(log)))
        return None
    record = read(folder / 'verified.json')
    print('COMPLETE ' + label, flush=True)
    return {**record, 'model': model, 'mode': mode, 'scenes': scenes, 'seed': seed}


def main():
    pid = read(ART.parent / 'heads_driver_v2_pid.json')['pid']
    while alive(pid):
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_HEAD_TRIAL', pid=pid))
        time.sleep(10)
    observations = []
    for model in ('roadK8', 'N24match4'):
        for mode in ('native24', 'midpoint24'):
            record = evaluate(model, mode, 256, 2817)
            if record is None:
                return
            observations.append(record)
            save(ART / 'observations.json', observations)
    for model in ('roadK8', 'N24match4'):
        for mode in ('native20', 'native24', 'midpoint24'):
            for seed in (2817, 3817):
                record = evaluate(model, mode, 4096, seed)
                if record is None:
                    return
                observations.append(record)
                save(ART / 'observations.json', observations)
    save(ART / 'status.json', dict(status='SAMPLER_SCREEN_COMPLETE',
                                 next='Agent ranks paired4K results and runs complete50K for promising modes.'))


if __name__ == '__main__':
    main()
