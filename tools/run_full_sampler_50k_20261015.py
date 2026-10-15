"""Verify promising paired4K sampling changes on the complete official50K."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied, SNAPSHOTS

SCREEN = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_sampler_eval_v3'
ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_sampler_50k_v2'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/full_sampler_50k_v2'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ART.mkdir(parents=True, exist_ok=True)
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Full50K plan already registered; inspect instead of restarting')
    pid = read(SCREEN / 'driver_pid.json')['pid']
    while alive(pid):
        save(ART / 'status.json', dict(status='WAITING_FOR_PAIRED_SCREEN', parent_pid=pid))
        time.sleep(10)
    status = read(SCREEN / 'status.json')
    if status['status'] != 'SAMPLER_SCREEN_COMPLETE':
        save(ART / 'status.json', dict(status='SCREEN_NOT_SUCCESSFUL_REQUIRES_REVIEW', screen_status=status))
        return
    result = subprocess.run([sys.executable, str(ROOT / 'tools/analyze_full_sampler_eval_20261015.py'),
                             '--artifacts', str(SCREEN)], cwd=ROOT, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ART / 'screen_analysis.log').write_text(result.stdout)
    if result.returncode:
        save(ART / 'status.json', dict(status='PAIRING_AUDIT_FAILED_REQUIRES_REVIEW'))
        return
    scores = read(SCREEN / 'comparisons.json')['aggregate']
    selected = [r for r in scores if r['promising_for_full50000_check']
                or r['attribute_tradeoff_worth_full50000_check']]
    # Speed/angle/length gains take priority; retain the explicitly recorded
    # midpoint geometry tradeoff for full50K measurement, not as a claimed win.
    selected.sort(key=lambda r: (any(r['ratios'][k] > 1 for k in
                                    ('speed_jsd', 'ang_dev_jsd', 'length_jsd')), r['R_focus']))
    with (ROOT / 'src/waymo_data/waymo_eval_set.pkl').open('rb') as f:
        reference_names = pickle.load(f)['files']
    assert len(reference_names) == len(set(reference_names)) == 50000
    # The SMART cache preserves the full official stem (including lg_type and
    # map_id), with its official index prepended and a .pt extension.
    names = [f'{i:05d}_{Path(name).stem}.pt' for i, name in enumerate(reference_names)]
    raw_dir = ROOT / 'src/waymo_data/full/scenario_dreamer_val'
    missing = [name for name in names if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} exact official-index caches: {missing[:3]}')
    manifest = ART / 'full50000_files.json'
    save(manifest, names)
    runner_hash = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    plan = []
    seen = set()
    for score in selected:
        for seed in (2817, 3817):
            for mode in (score['mode'], score['control_mode']):
                key = (score['model'], mode, seed)
                if key not in seen:
                    seen.add(key)
                    plan.append(dict(model=key[0], mode=key[1], seed=key[2]))
    save(ART / 'protocol.json', dict(registered_utc=now(), selected_4K_scores=selected,
                                   plan=plan, scenes=50000, reference_mode='matched',
                                   require_full_set=True, runner_sha256=runner_hash,
                                   cache_filename_mapping='Official index prefix + unchanged official stem + .pt; all50000 exist.',
                                   limitation='Same final checkpoint and paired random sources. No training change. '
                                   'The screen is a compute-budget gate, not proof of full50K improvement.'))
    if not plan:
        save(ART / 'status.json', dict(status='NO_PROMISING_PAIRED_4K_CHANGE'))
        return
    observations = []
    RUNS.mkdir(parents=True, exist_ok=True)
    for cell in plan:
        label = f"{cell['model']}_{cell['mode']}_N50000_GEN{cell['seed']}"
        folder = RUNS / label
        if folder.exists():
            raise RuntimeError('Full50K output exists; inspect before retry')
        while occupied():
            save(ART / 'status.json', dict(status='WAITING_FOR_GPU', next_cell=cell))
            time.sleep(10)
        assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == runner_hash
        snapshot = 'N24match4_final' if cell['model'] == 'N24match4' else cell['model']
        checkpoint = SNAPSHOTS / snapshot / 'model.ckpt'
        command = [sys.executable, '-u', str(RUNNER), '--checkpoint', str(checkpoint),
                   '--source-config', str(checkpoint.parent / 'config.yaml'),
                   '--sampler', str(SCREEN / (cell['mode'] + '.json')),
                   '--manifest', str(manifest), '--output', str(folder),
                   '--seed', str(cell['seed']), '--full-set', '--integration',
                   'midpoint' if cell['mode'] == 'midpoint24' else 'native']
        log = RUNS / (label + '.log')
        with log.open('x') as output:
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, 'WANDB_MODE': 'disabled',
                                                                 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
            save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid,
                                          log=str(log), started_utc=now()))
            result = child.wait()
        if result or not (folder / 'verified.json').exists():
            save(ART / 'status.json', dict(status='FULL50K_FAILED_REQUIRES_REVIEW', label=label,
                                          returncode=result, log=str(log)))
            return
        record = read(folder / 'verified.json')
        observations.append({**record, **cell, 'scenes': 50000})
        save(ART / 'observations.json', observations)
        print('COMPLETE ' + label, flush=True)
    save(ART / 'status.json', dict(status='FULL50K_SAMPLER_PLAN_COMPLETE',
                                 completed_utc=now(), cells=len(observations),
                                 next='Review native full50K comparisons and all7 target thresholds.'))


if __name__ == '__main__':
    main()
