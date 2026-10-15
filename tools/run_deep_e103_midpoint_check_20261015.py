"""Compare genuine epoch103 against existing epoch79 at the same24 model calls."""
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied
from tools.run_late_checkpoint_midpoint_20261015 import hashes, now, JSD

BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'deep_e103_midpoint_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/deep_e103_midpoint_v1'
SNAP = BASE / 'full_checkpoint_snapshots/deep_sampler_snapshots_20261015/deep_h2v2_e103'
CONTROL = ROOT / 'logs/init_diffusion_continuous_20261014/deep_sampler_v1'


def evaluate(scenes, seed):
    label = f'deep_e103_midpoint24_N{scenes}_GEN{seed}'
    folder = RUNS / label
    assert not folder.exists()
    names = read(BASE / 'full_sampler_50k_v2/full50000_files.json')
    if scenes != 50000:
        names = names[25000:25000 + scenes]
    manifest = ART / (label + '.manifest.json')
    save(manifest, names)
    assert not occupied(), 'Another GPU task appeared before launch'
    command = [sys.executable, '-u', 'tools/eval_full_checkpoint_midpoint_20261015.py',
               '--checkpoint', str(SNAP / 'model.ckpt'), '--source-config', str(SNAP / 'config.yaml'),
               '--sampler', str(ART / 'midpoint24.json'), '--manifest', str(manifest),
               '--output', str(folder), '--seed', str(seed), '--integration', 'midpoint']
    if scenes == 50000:
        command.append('--full-set')
    log = RUNS / (label + '.log')
    with log.open('x') as out:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'WANDB_MODE': 'disabled',
                                        'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        rc = child.wait()
    if rc or not (folder / 'verified.json').exists():
        save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', label=label, returncode=rc))
        raise RuntimeError('Preserve the failed evaluation and review before retrying')
    result = read(folder / 'verified.json')
    assert set(result['observed_denoiser_calls_per_sample_batch']) == {24}
    assert result['checkpoint_sha256'] == read(SNAP / 'provenance.json')['sha256']
    result.update(scenes=scenes, seed=seed, output=str(folder), training_complete=False)
    return result


def compare(candidate):
    scenes, seed = candidate['scenes'], candidate['seed']
    folder = CONTROL / f'deep_e79_midpoint24_N{scenes}_GEN{seed}'
    control = read(folder / 'verified.json')
    assert candidate['manifest_sha256'] == control['manifest_sha256']
    assert candidate['source_config_sha256'] == control['source_config_sha256']
    assert candidate['smart_source_sha256'] == control['smart_source_sha256']
    assert candidate['sampling_source_sha256_per_batch'] == control['sampling_source_sha256_per_batch']
    ah = read(Path(candidate['output']) / 'feature_histograms.json')
    bh = read(folder / 'feature_histograms.json')
    assert all(ah[k]['matched_gt'] == bh[k]['matched_gt'] for k in JSD)
    ar = read(Path(candidate['output']) / 'sd_agent_metrics.json')
    br = read(folder / 'sd_agent_metrics.json')
    assert ar['num_samples'] == br['num_samples'] == scenes
    assert ar['feature_counts_gt'] == br['feature_counts_gt']
    return dict(scenes=scenes, GEN=seed, candidate=candidate['metrics'], control=control['metrics'],
                ratios={k:candidate['metrics'][k] / control['metrics'][k] for k in JSD},
                vehicle_count_ratio=ar['num_generated_vehicles'] / br['num_generated_vehicles'],
                onroad_fraction_change=(ar['feature_counts_generated'][1] / ar['num_generated_vehicles']
                                         - br['feature_counts_generated'][1] / br['num_generated_vehicles']),
                same_initial_noise_and_reference_histograms=True)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists()
    assert read(SNAP / 'fixed_conditions_verified.json')['status'] == 'PASS'
    predecessor = read(BASE / 'periodic_edge_long_driver_pid.json')['pid']
    frozen = hashes()
    import hashlib
    frozen[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(ART / 'midpoint24.json', dict(sampling_steps=12,
         sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.1)))
    save(ART / 'protocol.json', dict(registered_utc=now(), predecessor_pid=predecessor,
        source_hashes=frozen, checkpoint=read(SNAP / 'provenance.json'), epochs=[79,103],
        GEN=[2817,3817], NFE=24, screen_scenes=8192, full_scenes=50000,
        selection='Both screen angle ratios<1, mean<.95; mean length<=1.08, speed<=1.8, width<=1.4. '
                  'Vehicle counts within.5%, onroad fraction cannot drop>.3 percentage points. '
                  'Attribute tradeoffs permitted for checkpoint screening only; full50K remains required.',
        limitation='Checkpoint-selection experiment on ongoing training, not a final128epoch result; '
                   'the screen gate cannot establish full50K baseline success.'))
    while alive(predecessor) or occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_STUDIES_AND_GPU'))
        time.sleep(10)
    def verify():
        for name, expected in frozen.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    rows, pairs = [], []
    for scenes, seeds in [(256,[2817]), (8192,[2817,3817])]:
        for seed in seeds:
            verify()
            rows.append(evaluate(scenes, seed))
            save(ART / 'observations.json', rows)
            if scenes == 8192:
                pairs.append(compare(rows[-1]))
    ratios = {k:statistics.mean(r['ratios'][k] for r in pairs) for k in JSD}
    eligible = (all(r['ratios']['ang_dev_jsd'] < 1 and .995 <= r['vehicle_count_ratio'] <= 1.005
                    and r['onroad_fraction_change'] >= -.003 for r in pairs)
                and ratios['ang_dev_jsd'] < .95 and ratios['length_jsd'] <= 1.08
                and ratios['speed_jsd'] <= 1.8 and ratios['width_jsd'] <= 1.4)
    save(ART / 'selection.json', dict(eligible=eligible, ratios=ratios, pairs=pairs,
                                     full50000_success_unproven=True))
    if eligible:
        for seed in (2817,3817):
            verify()
            rows.append(evaluate(50000, seed))
            save(ART / 'observations.json', rows)
            pairs.append(compare(rows[-1]))
    verify()
    save(ART / 'comparisons.json', pairs)
    save(ART / 'status.json', dict(status='DEEP_E103_PAIRED_PLAN_COMPLETE', completed_utc=now(),
        next='Assess all7 full50K targets and attribute tradeoffs before promoting a checkpoint.'))


if __name__ == '__main__':
    main()
