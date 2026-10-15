"""Finite fresh-training comparison after the time-scaled GPU smoke succeeds."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'native_heading_comparison_v2'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/native_heading_comparison_v2'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    prototype = json.loads((BASE / 'native_worker_heading_preflight_v1/driver.json').read_text())['command']
    control = json.loads((BASE / 'native_worker_heading_preflight_v1/variants.json').read_text())['control']
    raw = json.loads((BASE / 'shared_heading_velocity_isolated_v4/omega025_T4817_S2048_N8192.variant.json').read_text())['omega025']
    variants = dict(control=control, raw025=raw, scaled025=raw)
    with (ROOT / 'src/waymo_data/full/training_map2_sd_files.pkl').open('rb') as handle:
        train_size = len(pickle.load(handle)['files'])
    batches_per_epoch = (train_size + 511) // 512
    complete_epochs, remaining_batches = divmod(2048, batches_per_epoch)
    expected_exposure = complete_epochs * train_size + remaining_batches * 512
    save(ART / 'variants.json', variants)
    names = json.loads((BASE / 'full_sampler_50k_v2/full50000_files.json').read_text())
    save(ART / 'manifest8192.json', names[25000:33192])
    tracked = [ROOT / 'tools/runtime_native_worker_heading_20261015.py',
        ROOT / 'tools/shared_heading_velocity_20261015.py', ROOT / 'tools/time_scaled_heading_full/model.py',
        ROOT / 'tools/scratch_component_loss_sweep.py', Path(__file__),
        *sorted((ROOT / 'src/smart').rglob('*.py')), *sorted((ROOT / 'configs').rglob('*.yaml'))]
    frozen = {str(p): sha(p) for p in tracked}
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=frozen, steps=2048, TRAIN=4817, GEN=[2817, 3817], scenes=8192,
        train_size=train_size, actual_expected_training_examples=expected_exposure,
        replaces='v1 cancelled before any GPU training: correct the final partial batch accounting',
        fresh_training=True, frozen_map='AIRL512 legacy encoder', workers=8,
        paired_controls='Fresh x0 and rawomega025 controls in the same runtime; no reuse of historical weights.',
        arms=['control', 'raw025', 'scaled025'], generator_parameters=8083978,
        evaluation='EMA; midpoint12 intervals,24 neural evaluations; no final-quality claim.',
        gates=dict(each_GEN_angle_ratio_at_most=1.02, mean_focus_ratio_below=.95,
            mean_attribute_ratio_at_most=1.05, each_GEN_geometry_ratio_at_most=1.2),
        gate_usage='Exploratory only: nominate a longer paired study, never automatically deploy full training.',
        premise='Bounded constant arc-velocity target. Time scaling changes gradient conditioning, not a necessary1/t singularity.',
        GPU_precondition='Native time-scaled2step GPU smoke passes; at least8.5GiB device memory free.',
        failure_rule='Stop on first failed child; preserve outputs; no automatic retry.'))
    while True:
        smoke = BASE / 'time_scaled_native_worker_gpu_smoke_v1'
        if (smoke / 'failure.json').exists():
            raise RuntimeError('Native GPU smoke failed; review before training')
        status = json.loads((smoke / 'status.json').read_text())
        if status['status'] == 'GPU_SMOKE_PASS':
            break
        save(ART / 'status.json', dict(status='WAITING_FOR_TIME_SCALED_NATIVE_GPU_SMOKE'))
        time.sleep(30)
    observations = []
    for arm in variants:
        assert all(sha(Path(p)) == h for p, h in frozen.items()), 'Registered sources changed'
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 8704:
                break
            save(ART / 'status.json', dict(status='WAITING_FOR_LOCAL_MEMORY', free_mib=free))
            time.sleep(30)
        output = RUNS / arm
        command = prototype[:]
        replacements = {'--heading-readout': 'time_scaled' if arm == 'scaled025' else 'raw',
            '--steps': '2048', '--scenes': '8192', '--variants': arm,
            '--variant-file': str(ART / 'variants.json'), '--output': str(output),
            '--eval-manifest': str(ART / 'manifest8192.json')}
        for flag, value in replacements.items():
            command[command.index(flag) + 1] = value
        command += ['--extra-eval-seed', '3817']
        with (RUNS / (arm + '.log')).open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
            save(ART / 'status.json', dict(status='TRAIN_AND_EVAL_RUNNING', arm=arm,
                pid=child.pid, command=command, output=str(output)))
            rc = child.wait()
        assert rc == 0, f'{arm}: exit {rc}; inspect preserved logs'
        rows = json.loads((output / 'summary.json').read_text())
        assert {r['eval_seed'] for r in rows} == {2817, 3817}
        assert all(r['budget'] == 2048 and r['seen_training_examples'] == expected_exposure
                   and r['map_frozen'] and r['generator_num_parameters'] == 8083978 for r in rows)
        for r in rows:
            r['arm'] = arm
        if observations:
            first = observations[0]
            for r in rows:
                for key in ('seed', 'batch_size', 'train_order_sha256', 'calibration_sha256',
                            'initial_shared_generator_sha256', 'initial_map_sha256', 'eval_manifest_sha256'):
                    assert r[key] == first[key], key
        observations.extend(rows)
        save(ART / 'observations.json', observations)
    decisions = []
    for arm in ('raw025', 'scaled025'):
        pairs = []
        for seed in (2817, 3817):
            old = next(r for r in observations if r['arm'] == 'control' and r['eval_seed'] == seed)
            new = next(r for r in observations if r['arm'] == arm and r['eval_seed'] == seed)
            keys = ('ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd', 'lat_dev_jsd', 'nearest_dist_jsd')
            ratios = {k: new[k] / old[k] for k in keys}
            pairs.append(dict(seed=seed, ratios=ratios))
        focus = sum(p['ratios'][k] for p in pairs for k in ('ang_dev_jsd', 'length_jsd', 'speed_jsd')) / 6
        attrs = sum(p['ratios'][k] for p in pairs for k in ('length_jsd', 'width_jsd', 'speed_jsd')) / 6
        eligible = all(p['ratios']['ang_dev_jsd'] <= 1.02
            and max(p['ratios'][k] for k in ('lat_dev_jsd', 'nearest_dist_jsd')) <= 1.2 for p in pairs)
        eligible &= focus < .95 and attrs <= 1.05
        decisions.append(dict(arm=arm, pairs=pairs, mean_focus_ratio=focus,
            mean_attribute_ratio=attrs, nominate_longer_paired_study=eligible))
    save(ART / 'decision.json', dict(decisions=decisions, full_training_quality_unproven=True))
    save(ART / 'status.json', dict(status='FINITE_FRESH_PAIRED_STUDY_COMPLETE'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
