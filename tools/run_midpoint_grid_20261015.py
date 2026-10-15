"""Finite equal-NFE midpoint time-grid study on final trained checkpoints."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save
from tools.run_full_sampler_eval_20261015 import occupied, SNAPSHOTS

ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/midpoint_grid_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/midpoint_grid_v1'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
POWERS = (1.2, .8, 1., 1.6)


def now():
    return datetime.now(timezone.utc).isoformat()


def compare(rows):
    cells = {(r['model'], r['power'], r['seed']): r for r in rows if r['scenes'] == 8192}
    results = []
    for model in ('roadK8', 'N24match4'):
        for power in POWERS[1:]:
            paired = []
            for seed in (2817, 3817):
                a, b = cells.get((model, power, seed)), cells.get((model, 1.2, seed))
                if a is None or b is None:
                    continue
                for key in ('checkpoint_sha256', 'manifest_sha256', 'source_config_sha256',
                            'sampling_source_sha256_per_batch', 'smart_source_sha256', 'runner_sha256'):
                    assert a[key] == b[key], key
                ratios = {k: a['metrics'][k] / b['metrics'][k] for k in METRICS}
                ar, br = read(Path(a['output']) / 'sd_agent_metrics.json'), read(Path(b['output']) / 'sd_agent_metrics.json')
                paired.append(dict(seed=seed, ratios=ratios,
                                   vehicle_count_ratio=ar['num_generated_vehicles'] / br['num_generated_vehicles'],
                                   onroad_fraction_change=ar['feature_counts_generated'][2] / ar['num_generated_vehicles']
                                       - br['feature_counts_generated'][2] / br['num_generated_vehicles']))
            if not paired:
                continue
            ratios = {k: statistics.mean(p['ratios'][k] for p in paired) for k in METRICS}
            focus = statistics.mean(ratios[k] for k in ('ang_dev_jsd', 'speed_jsd', 'length_jsd'))
            eligible = (len(paired) == 2 and ratios['ang_dev_jsd'] < .97
                        and all(p['ratios']['ang_dev_jsd'] < .99 for p in paired)
                        and focus < .99 and ratios['length_jsd'] <= 1.03
                        and ratios['width_jsd'] <= 1.03 and ratios['speed_jsd'] <= 1.03
                        and all(.995 <= p['vehicle_count_ratio'] <= 1.005
                                and p['onroad_fraction_change'] >= -.003 for p in paired))
            results.append(dict(model=model, power=power, ratios=ratios, R_focus=focus,
                                paired=paired, eligible_for_full50000=eligible,
                                full50000_quality_unproven=True))
    save(ART / 'comparisons.json', dict(updated_utc=now(), comparisons=results,
                                      scope='Paired8K; identical12 intervals/24 actual model calls, terminal=.1. '
                                      'Only time-grid power changes.'))
    return results


def evaluate(model, power, scenes, seed):
    label = f'{model}_power{power:g}_N{scenes}_GEN{seed}'
    folder = RUNS / label
    if folder.exists():
        raise RuntimeError('Grid output exists; inspect before retry')
    names = read(ART.parent / 'full_sampler_50k_v2/full50000_files.json')
    names = names if scenes == 50000 else names[25000:25000 + scenes]
    assert len(names) == scenes
    manifest = ART / (label + '.manifest.json')
    save(manifest, names)
    while occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_GPU', label=label))
        time.sleep(10)
    snapshot = 'N24match4_final' if model == 'N24match4' else model
    checkpoint = SNAPSHOTS / snapshot / 'model.ckpt'
    command = [sys.executable, '-u', str(RUNNER), '--checkpoint', str(checkpoint),
               '--source-config', str(checkpoint.parent / 'config.yaml'),
               '--sampler', str(ART / f'power{power:g}.json'), '--manifest', str(manifest),
               '--output', str(folder), '--seed', str(seed), '--integration', 'midpoint']
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
    record = {**read(folder / 'verified.json'), 'model': model, 'power': power, 'seed': seed, 'scenes': scenes}
    assert set(record['observed_denoiser_calls_per_sample_batch']) == {24}
    print('COMPLETE ' + label, flush=True)
    return record


def main():
    ART.mkdir(parents=True, exist_ok=True)
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Grid plan already registered')
    RUNS.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [RUNNER, *sorted((ROOT / 'src/smart').rglob('*.py'))]}
    for power in POWERS:
        save(ART / f'power{power:g}.json', dict(sampling_steps=12,
                                             sampling_time_grid=dict(kind='power', power=power, terminal_time=.1)))
    save(ART / 'protocol.json', dict(registered_utc=now(), source_hashes=hashes, NFE=24, intervals=12,
                                    powers=POWERS, models=['roadK8', 'N24match4'], GEN=[2817, 3817],
                                    scenes=8192, cohort='Official indexed caches25000:33192',
                                    objective_architecture_weights_unchanged=True,
                                    full_promotion='Best angle gain passing both GEN and shape/speed gates, then paired actual50K.'))
    rows = []
    for model in ('roadK8', 'N24match4'):
        for power in POWERS:
            for seed in (2817, 3817):
                for name, expected in hashes.items():
                    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
                row = evaluate(model, power, 8192, seed)
                if row is None:
                    return
                rows.append(row)
                save(ART / 'observations.json', rows)
                compare(rows)
    for model in ('roadK8', 'N24match4'):
        eligible = [x for x in compare(rows) if x['model'] == model and x['eligible_for_full50000']]
        if not eligible:
            continue
        winner = min(eligible, key=lambda x: x['ratios']['ang_dev_jsd'])
        save(ART / (model + '.selection.json'), dict(selected=winner, full50000_quality_unproven=True))
        for seed in (2817, 3817):
            for power in (winner['power'], 1.2):
                row = evaluate(model, power, 50000, seed)
                if row is None:
                    return
                rows.append(row)
                save(ART / 'observations.json', rows)
    save(ART / 'status.json', dict(status='GRID_FINITE_PLAN_COMPLETE', completed_utc=now(),
                                 next='Review actual full50K gains and all7 thresholds; continuing goal stays active.'))


if __name__ == '__main__':
    main()
