"""Finite paired terminal-angle study after the owned full50K sampler plan."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied, SNAPSHOTS

ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/terminal_heading_gain'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/terminal_heading_gain'
PARENT = ART.parent / 'full_sampler_50k_v2'
RUNNER = ROOT / 'tools/eval_terminal_heading_gain_20261015.py'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')


def now():
    return datetime.now(timezone.utc).isoformat()


def compare(rows):
    cells = {(r['gain'], r['seed']): r for r in rows if r['scenes'] == 8192}
    groups = []
    for gain in (1., 1.5, 2.5, 3.):
        paired = []
        for seed in (2817, 3817):
            a, b = cells.get((gain, seed)), cells.get((2., seed))
            if a is None or b is None:
                continue
            for key in ('checkpoint_sha256', 'manifest_sha256', 'source_config_sha256',
                        'sampling_source_sha256_per_batch', 'smart_source_sha256',
                        'runner_sha256', 'experimental_sampler_sha256'):
                assert a[key] == b[key], key
            ar = read(Path(a['output']) / 'sd_agent_metrics.json')
            br = read(Path(b['output']) / 'sd_agent_metrics.json')
            ratios = {k: a['metrics'][k] / b['metrics'][k] for k in METRICS}
            paired.append(dict(seed=seed, ratios=ratios,
                               vehicle_count_ratio=ar['num_generated_vehicles'] / br['num_generated_vehicles'],
                               onroad_fraction_change=ar['feature_counts_generated'][2] / ar['num_generated_vehicles']
                                   - br['feature_counts_generated'][2] / br['num_generated_vehicles']))
        if not paired:
            continue
        means = {k: statistics.mean(p['ratios'][k] for p in paired) for k in METRICS}
        eligible = (len(paired) == 2 and means['ang_dev_jsd'] < .97
                    and all(p['ratios']['ang_dev_jsd'] < .99 for p in paired)
                    and all(.995 <= p['vehicle_count_ratio'] <= 1.005
                            and abs(p['onroad_fraction_change']) <= .003 for p in paired)
                    and max(means[k] for k in METRICS if k != 'ang_dev_jsd') <= 1.05)
        groups.append(dict(gain=gain, ratios=means, paired=paired, eligible_for_full50000=eligible,
                           full50000_quality_unproven=True))
    save(ART / 'comparisons.json', dict(updated_utc=now(), comparisons=groups,
                                      limitation='Paired8K sampling study; select by angle and check unchanged feature distributions.'))
    return groups


def evaluate(gain, scenes, seed):
    label = f'gain{gain:g}_N{scenes}_GEN{seed}'
    folder = RUNS / label
    if folder.exists():
        raise RuntimeError('Terminal trial output exists; inspect before restarting')
    names = read(PARENT / 'full50000_files.json')
    selected = names if scenes == 50000 else names[25000:25000 + scenes]
    assert len(selected) == scenes
    manifest = ART / (label + '.manifest.json')
    save(manifest, selected)
    while occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_GPU', label=label))
        time.sleep(10)
    checkpoint = SNAPSHOTS / 'N24match4_final/model.ckpt'
    command = [sys.executable, '-u', str(RUNNER), '--checkpoint', str(checkpoint),
               '--source-config', str(checkpoint.parent / 'config.yaml'),
               '--sampler', str(ART / 'midpoint24.json'), '--manifest', str(manifest),
               '--output', str(folder), '--seed', str(seed), '--integration', 'midpoint',
               '--terminal-heading-gain', str(gain)]
    if scenes == 50000:
        command.append('--full-set')
    log = RUNS / (label + '.log')
    with log.open('x') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, 'WANDB_MODE': 'disabled',
                                                             'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        result = child.wait()
    if result or not (folder / 'verified.json').exists():
        save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', label=label,
                                      returncode=result, log=str(log)))
        return None
    record = {**read(folder / 'verified.json'), 'gain': gain, 'seed': seed, 'scenes': scenes}
    print('COMPLETE ' + label, flush=True)
    return record


def main():
    assert read(ART / 'proof.json')['status'] == 'PASS'
    pid = read(PARENT / 'driver_pid.json')['pid']
    while alive(pid):
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_FULL50K_PLAN', parent_pid=pid))
        time.sleep(10)
    status = read(PARENT / 'status.json')
    if status['status'] != 'FULL50K_SAMPLER_PLAN_COMPLETE':
        save(ART / 'status.json', dict(status='PARENT_FAILED_REQUIRES_REVIEW', parent_status=status))
        return
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Terminal phase already registered')
    RUNS.mkdir(parents=True, exist_ok=True)
    save(ART / 'midpoint24.json', read(ART.parent / 'full_sampler_eval_v3/midpoint24.json'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (RUNNER, ROOT / 'tools/terminal_heading_sampling_20261015.py')}
    save(ART / 'protocol.json', dict(registered_utc=now(), source_hashes=hashes,
                                    checkpoint='cchgfuhr epoch127 step123392', NFE=24,
                                    gains=[2., 1., 1.5, 2.5, 3.], GEN=[2817, 3817], scenes=8192,
                                    source_cohort='Official indexed caches25000:33192; same8192 for every gain.',
                                    objective_or_parameter_change=False,
                                    full_promotion='Best passing paired8K angle gain, then actual50K with both GENs.'))
    rows = []
    plan = [(2., 256, 2817), (1.5, 256, 2817)]
    plan += [(g, 8192, s) for g in (2., 1., 1.5, 2.5, 3.) for s in (2817, 3817)]
    for gain, scenes, seed in plan:
        for name, expected in hashes.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        row = evaluate(gain, scenes, seed)
        if row is None:
            return
        rows.append(row)
        save(ART / 'observations.json', rows)
        compare(rows)
    eligible = [r for r in compare(rows) if r['eligible_for_full50000']]
    if eligible:
        winner = min(eligible, key=lambda r: r['ratios']['ang_dev_jsd'])
        save(ART / 'selection.json', dict(selected=winner, full50000_quality_unproven=True))
        for seed in (2817, 3817):
            for gain in (winner['gain'], 2.):
                row = evaluate(gain, 50000, seed)
                if row is None:
                    return
                rows.append(row)
                save(ART / 'observations.json', rows)
    save(ART / 'status.json', dict(status='TERMINAL_GAIN_FINITE_PLAN_COMPLETE', completed_utc=now(),
                                 full50000_candidate_checked=bool(eligible),
                                 next='Review actual results; ongoing goal remains active.'))


if __name__ == '__main__':
    main()
