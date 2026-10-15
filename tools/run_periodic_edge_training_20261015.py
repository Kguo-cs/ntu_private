"""Fresh fixed-map periodic heading study; unchanged loss and24-call midpoint."""
from pathlib import Path
import copy
import hashlib
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save, MAP, NORMALIZER, TRAIN_MANIFEST
from tools.run_full_sampler_eval_20261015 import occupied
from tools.run_late_checkpoint_midpoint_20261015 import hashes, now, JSD

BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'periodic_edge_training_v2'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/periodic_edge_training_v2'
RUNNER = ROOT / 'tools/runtime_periodic_edge_sweep_20261015.py'
P = 'model.model_config.decoder.init_diffusion.'
ARMS = {'control': 0, 'harmonic4': 4, 'harmonic8': 8}


def source_hashes():
    result = hashes()
    for p in (RUNNER, Path(__file__), ROOT / 'tools/periodic_edge_heading_20261015.py',
              ROOT / 'tools/scratch_component_loss_sweep.py', ROOT / 'tools/overnight_init_diffusion_sweep.py',
              ROOT / 'tools/runtime_cached_torch_loader_sweep.py', ROOT / 'tools/torch_pickle_class_cache.py'):
        result[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def stage(arm, steps, scenes=8192, seed=4817, resume=None):
    label = f'{arm}_T{seed}_S{steps}_N{scenes}'
    folder = RUNS / label
    assert not folder.exists()
    variants = ART / (label + '.variant.json')
    save(variants, {arm: read(ART / 'variants.json')[arm]})
    manifest = ART / (label + '.manifest.json')
    save(manifest, read(ART / 'eval8192.json')[:scenes])
    while occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_GPU', label=label))
        time.sleep(10)
    command = [sys.executable, '-u', str(RUNNER), '--periodic-edge-harmonic', str(ARMS[arm]),
        '--map-checkpoint', str(MAP), '--batch-size', '512', '--scenes', str(scenes), '--eval-batch-size', '128',
        '--workers', '0', '--defer-cyclic-gc', '--young-gc-interval', '0', '--allow-multiple-epochs',
        '--seed', str(seed), '--eval-seed', '2817', '--extra-eval-seed', '3817',
        '--normalizer-json', str(NORMALIZER), '--evaluation-weights', 'ema', '--steps', str(steps),
        '--checkpoint-steps', '', '--recipe', str(ART / 'recipe.json'), '--variant-file', str(variants),
        '--variants', arm, '--output', str(folder), '--eval-manifest', str(manifest)]
    if resume:
        path = ART / (label + '.resume.json')
        save(path, {arm: str(resume)})
        command += ['--resume-map', str(path), '--resume-cyclic-epochs']
    log = RUNS / (label + '.log')
    with log.open('x') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'SIM_SWEEP_DEFER_CYCLIC_GC': '1', 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'WANDB_MODE': 'disabled'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        result = child.wait()
    if result or not (folder / 'summary.json').exists() or 'RUNTIME_LOADER_SUCCESS' not in log.read_text():
        save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', label=label, returncode=result, log=str(log)))
        raise RuntimeError('Training/evaluation failed; preserve output and inspect')
    rows = read(folder / 'summary.json')
    assert len(rows) == 2 and {r['eval_seed'] for r in rows} == {2817, 3817}
    for row in rows:
        assert row['budget'] == steps and row['seed'] == seed and row['num_samples'] == scenes
        assert row['map_frozen'] and row['evaluation_weights'] == 'ema'
        report = read(Path(row['metrics_path']))
        assert report['num_samples'] == report['num_gt_samples'] == scenes
        d = report['decoder']
        assert d['sampling_steps'] == 12 and d['sampling_integrator'] == 'midpoint'
        assert d['sampling_network_evaluations'] == 24 and d['ema_num_updates'] == steps
        if ARMS[arm]:
            assert d['denoiser_options']['edge_heading_encoding'] == 'integer_harmonics'
            assert d['denoiser_options']['edge_heading_max_harmonic'] == ARMS[arm]
        else:
            assert 'edge_heading_encoding' not in d['denoiser_options']
    trace = read(folder / arm / 'train_trace.json')
    assert all(r['map_grad_norm_before_clip'] == 0 for r in trace)
    save(folder / 'verified_training.json', dict(status='PASS', command=command, harmonic=ARMS[arm],
        steps=steps, full_training_complete=False, frozen_map_verified=True))
    print('PERIODIC_STAGE_COMPLETE ' + label, flush=True)
    checkpoint = Path(rows[0]['checkpoint'])
    assert checkpoint.is_file() and all(r['checkpoint'] == str(checkpoint) for r in rows)
    return rows, checkpoint


def compare(rows, arm, budget, seed=4817):
    controls = {r['eval_seed']: r for r in rows if r['variant'] == 'control' and r['budget'] == budget and r['seed'] == seed}
    candidates = [r for r in rows if r['variant'] == arm and r['budget'] == budget and r['seed'] == seed]
    paired = []
    assert len(controls) == len(candidates) == 2
    for r in candidates:
        c = controls[r['eval_seed']]
        for k in ('initial_map_sha256', 'initial_generator_sha256', 'train_order_sha256',
                  'eval_manifest_sha256', 'seen_training_examples'):
            assert r[k] == c[k], k
        assert r['generator_num_parameters'] == c['generator_num_parameters']
        ah = read(Path(r['metrics_path']).parent / 'feature_histograms.json')
        bh = read(Path(c['metrics_path']).parent / 'feature_histograms.json')
        assert all(ah[k]['matched_gt'] == bh[k]['matched_gt'] for k in JSD)
        ratios = {k: r[k] / c[k] for k in JSD}
        paired.append(dict(seed=r['eval_seed'], ratios=ratios))
    ratios = {k: statistics.mean(r['ratios'][k] for r in paired) for k in JSD}
    return dict(arm=arm, budget=budget, TRAIN=seed, ratios=ratios, paired=paired,
                R_focus=statistics.mean(ratios[k] for k in ('speed_jsd', 'ang_dev_jsd', 'length_jsd')),
                R_jsd=statistics.mean(ratios.values()), full128epoch_unproven=True)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists()
    assert read(BASE / 'periodic_edge_heading_proof.json')['status'] == 'PASS'
    assert read(BASE / 'periodic_checkpoint_guard_proof.json')['status'] == 'PASS'
    predecessor = read(BASE / 'deep_sampler_driver_pid.json')['pid']
    frozen = source_hashes()
    old = ROOT / 'output/reports/init_diffusion_expanded_20261014'
    base = copy.deepcopy(read(old / 'variants.json')['roadK8'])
    base['__config'].update({P + 'sampling_steps': 12, P + 'sampling_integrator': 'midpoint',
                            P + 'sampling_time_grid': dict(kind='power', power=1.2, terminal_time=.1)})
    save(ART / 'variants.json', {name: copy.deepcopy(base) for name in ARMS})
    save(ART / 'recipe.json', read(old / 'recipe.json'))
    save(ART / 'eval8192.json', read(BASE / 'full_sampler_50k_v2/full50000_files.json')[25000:33192])
    save(ART / 'protocol.json', dict(registered_utc=now(), source_hashes=frozen, predecessor_pid=predecessor,
        arms=ARMS, map_checkpoint=str(MAP), normalizer=str(NORMALIZER), TRAIN=[4817], GEN=[2817, 3817],
        preflight_steps=16, screen_steps=2048, continuation_steps=6144, NFE=24, architecture_parameters_unchanged=True,
        objective_and_sampling_unchanged=True, map_encoder_frozen=True,
        selection='Continue best angle ratio with mean focus<.99, angle<1.0, speed/length/width<=1.10; no promotion from short proxy.',
        limitation='From scratch on a repeated fixed full-data training order. Quick results do not prove final128epoch quality.'))
    while alive(predecessor) or occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_DEEP_SAMPLER_STUDY_AND_GPU'))
        time.sleep(10)
    all_rows, checkpoints = [], {}
    for arm in ('control', 'harmonic8'):
        assert source_hashes() == frozen
        stage(arm, 16, scenes=256)
    for arm in ARMS:
        assert source_hashes() == frozen
        rows, checkpoint = stage(arm, 2048)
        all_rows.extend(rows)
        checkpoints[arm] = checkpoint
        save(ART / 'observations.json', all_rows)
    comparisons = [compare(all_rows, arm, 2048) for arm in ARMS if arm != 'control']
    save(ART / 'comparisons.json', comparisons)
    eligible = [r for r in comparisons if r['R_focus'] < .99 and r['ratios']['ang_dev_jsd'] < 1
                and max(r['ratios'][k] for k in ('speed_jsd', 'length_jsd', 'width_jsd')) <= 1.10]
    if eligible:
        winner = min(eligible, key=lambda r: r['ratios']['ang_dev_jsd'])['arm']
        save(ART / 'selection.json', dict(winner=winner, evidence=eligible, full128epoch_unproven=True))
        for arm in ('control', winner):
            assert source_hashes() == frozen
            rows, checkpoint = stage(arm, 6144, resume=checkpoints[arm])
            all_rows.extend(rows)
            save(ART / 'observations.json', all_rows)
        comparisons.append(compare(all_rows, winner, 6144))
        save(ART / 'comparisons.json', comparisons)
    save(ART / 'status.json', dict(status='PERIODIC_TRAINING_FINITE_PLAN_COMPLETE', completed_utc=now(),
         next='Review actual training/eval curves; second TRAIN and full50K required before recommendation.'))


if __name__ == '__main__':
    main()
