"""Paired late-epoch EMA selection with fixed conditioning and 24 real calls."""
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

BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'late_checkpoint_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/late_checkpoint_v1'
LATE = BASE / 'full_checkpoint_snapshots/sampler_late_snapshots_20261015'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'
JSD = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
FOCUS = ('ang_dev_jsd', 'speed_jsd', 'length_jsd')
PAIRS = [('roadK8', 'roadK8_e111'), ('roadK8', 'roadK8_e119'), ('N24match4', 'N24match4_e119')]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [RUNNER, *sorted((ROOT / 'src/smart').rglob('*.py'))]}


def now():
    return datetime.now(timezone.utc).isoformat()


def compare(candidate, control):
    for k in ('source_config_sha256', 'manifest_sha256', 'smart_source_sha256', 'runner_sha256',
              'sampling_source_sha256_per_batch', 'matmul_precision', 'sampler'):
        assert candidate[k] == control[k], k
    assert set(candidate['observed_denoiser_calls_per_sample_batch']) == {24}
    ah = read(Path(candidate['output']) / 'feature_histograms.json')
    bh = read(Path(control['output']) / 'feature_histograms.json')
    for k in JSD:
        assert ah[k]['edges'] == bh[k]['edges'] and ah[k]['matched_gt'] == bh[k]['matched_gt']
    a = read(Path(candidate['output']) / 'sd_agent_metrics.json')
    b = read(Path(control['output']) / 'sd_agent_metrics.json')
    ratios = {k: candidate['metrics'][k] / control['metrics'][k] for k in JSD}
    if candidate['scenes'] == 50000:
        target = read(ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json')
        for report in (a, b):
            for k in ('num_samples', 'num_gt_samples', 'full_membership', 'eval_set_sha256',
                      'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt'):
                assert report[k] == target[k], k
    return dict(snapshot=candidate['snapshot'], seed=candidate['seed'], scenes=candidate['scenes'],
                ratios=ratios, R_focus=statistics.mean(ratios[k] for k in FOCUS),
                R_jsd=statistics.mean(ratios.values()),
                vehicle_count_ratio=a['num_generated_vehicles'] / b['num_generated_vehicles'],
                onroad_fraction_change=a['feature_counts_generated'][2] / a['num_generated_vehicles']
                    - b['feature_counts_generated'][2] / b['num_generated_vehicles'],
                candidate_metrics=candidate['metrics'], control_metrics=control['metrics'])


def evaluate(model, snapshot, scenes, seed):
    label = f'{snapshot}_N{scenes}_GEN{seed}'
    folder = RUNS / label
    assert not folder.exists(), 'Inspect existing output before retry'
    names = read(BASE / 'full_sampler_50k_v2/full50000_files.json')
    if scenes != 50000:
        names = names[25000:25000 + scenes]
    manifest = ART / (label + '.manifest.json')
    save(manifest, names)
    while occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_GPU', label=label))
        time.sleep(10)
    checkpoint = ((SNAPSHOTS / ('N24match4_final' if model == 'N24match4' else model))
                  if snapshot.endswith('_final') else LATE / snapshot) / 'model.ckpt'
    cmd = [sys.executable, '-u', str(RUNNER), '--checkpoint', str(checkpoint),
           '--source-config', str(checkpoint.parent / 'config.yaml'), '--sampler', str(ART / 'sampler.json'),
           '--manifest', str(manifest), '--output', str(folder), '--seed', str(seed), '--integration', 'midpoint']
    if scenes == 50000:
        cmd.append('--full-set')
    log = RUNS / (label + '.log')
    with log.open('x') as out:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'WANDB_MODE': 'disabled', 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        result = child.wait()
    if result or not (folder / 'verified.json').exists():
        save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', label=label, returncode=result, log=str(log)))
        raise RuntimeError('Evaluation failed; no automatic retry or promotion')
    print('COMPLETE ' + label, flush=True)
    return {**read(folder / 'verified.json'), 'model': model, 'snapshot': snapshot, 'scenes': scenes, 'seed': seed}


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists(), 'Plan already registered'
    proof = read(LATE / 'paired_conditions_verified.json')
    assert proof['status'] == 'PASS' and len(proof['rows']) == 3
    frozen = hashes()
    controls = [r for r in read(BASE / 'midpoint_grid_v1/observations.json') if r['power'] == 1.2]
    for r in controls:
        assert r['runner_sha256'] == frozen[str(RUNNER.relative_to(ROOT))]
        for k, v in r['smart_source_sha256'].items():
            assert frozen[k] == v
    save(ART / 'sampler.json', read(BASE / 'midpoint_grid_v1/power1.2.json'))
    save(ART / 'protocol.json', dict(registered_utc=now(), pairs=PAIRS, GEN=[2817, 3817],
         screen_scenes=8192, NFE=24, source_hashes=frozen, conditions_proof=proof,
         control_reuse='Completed exact paired8K final127 controls; source/config/noise/GT checks required.',
         full_gate='Both angle ratios<.99, mean angle<.97, focus<.99; speed/shape<=1.05; count/onroad guards.',
         limitation='Selects existing genuine checkpoints; does not prove a different training recipe.'))
    rows, comparisons, eligible = [], [], []
    for model, snapshot in PAIRS:
        paired = []
        for seed in (2817, 3817):
            assert hashes() == frozen
            row = evaluate(model, snapshot, 8192, seed)
            rows.append(row)
            control = next(r for r in controls if r['model'] == model and r['seed'] == seed)
            paired.append(compare(row, control))
            comparisons.append(paired[-1])
            save(ART / 'observations.json', rows)
            save(ART / 'comparisons.json', comparisons)
        ratios = {k: statistics.mean(r['ratios'][k] for r in paired) for k in JSD}
        passes = (ratios['ang_dev_jsd'] < .97 and all(r['ratios']['ang_dev_jsd'] < .99 for r in paired)
                  and statistics.mean(ratios[k] for k in FOCUS) < .99
                  and max(ratios[k] for k in ('length_jsd', 'width_jsd', 'speed_jsd')) <= 1.05
                  and all(.995 <= r['vehicle_count_ratio'] <= 1.005 and r['onroad_fraction_change'] >= -.003
                          for r in paired))
        if passes:
            eligible.append(dict(model=model, snapshot=snapshot, ratios=ratios))
        save(ART / 'eligible.json', eligible)
    for model in ('roadK8', 'N24match4'):
        candidates = [r for r in eligible if r['model'] == model]
        if not candidates:
            continue
        winner = min(candidates, key=lambda r: r['ratios']['ang_dev_jsd'])
        for seed in (2817, 3817):
            assert hashes() == frozen
            row = evaluate(model, winner['snapshot'], 50000, seed)
            control = evaluate(model, model + '_final', 50000, seed)
            rows.extend([row, control])
            comparisons.append(compare(row, control))
            save(ART / 'observations.json', rows)
            save(ART / 'comparisons.json', comparisons)
    save(ART / 'status.json', dict(status='LATE_CHECKPOINT_FINITE_PLAN_COMPLETE', completed_utc=now(),
         next='Review paired complete50K if any checkpoint passed; continuous optimization remains active.'))


if __name__ == '__main__':
    main()
