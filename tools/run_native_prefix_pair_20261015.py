"""Finite raw/scaled native8epoch pair in an immutable source snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'native_prefix8_pair_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/native_prefix8_pair_v1'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    proof = json.loads((BASE / 'native_prefix8_gpu_pair_verified.json').read_text())
    assert proof['status'] == 'PAIRED_NATIVE_TWO_STEP_SMOKES_PASS'
    smoke = json.loads((BASE / 'native_prefix8_gpu_smoke_v1/training_audit.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot = ROOT / '.codex_runs' / ('native_prefix8_pair_' + stamp)
    snapshot.mkdir(exist_ok=False)
    for directory in ('src/smart', 'src/utils', 'src/my_data_process', 'configs', 'tools'):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source in (ROOT / 'src').glob('*.py'):
        shutil.copy2(source, snapshot / 'src' / source.name)
    normalizer = Path('docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json')
    (snapshot / normalizer).parent.mkdir(parents=True)
    shutil.copy2(ROOT / normalizer, snapshot / normalizer)
    for directory in ('logs', 'output', 'videos', 'src/logs', 'src/waymo_data'):
        target = ROOT / directory
        if target.exists():
            (snapshot / directory).symlink_to(target, target_is_directory=True)
    manifest = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
                if p.is_file() and not p.is_symlink()}
    save(snapshot / 'SOURCE_MANIFEST.json', manifest)
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_snapshot=str(snapshot), source_manifest_sha256=sha(snapshot / 'SOURCE_MANIFEST.json'),
        native_entrypoint='src/run0.py', TRAIN=4817, epochs=8, optimizer_schedule_epochs=128,
        expected_steps=7712, expected_examples_per_epoch=493542, full50K_validation_every_epochs=8,
        per_epoch_shuffle=True, generator_from_scratch=True, frozen_map='AIRL512',
        G1_parameters=8083978, NFE=24, precision='medium/32-true',
        arms=['raw025', 'scaled025'], data_order_and_initial_parameters_must_match=True,
        prior_screen_failed=True,
        new_study_reason='2K time-scaled screen improves angle19% but misses aggregate gate. Rawomega2K angle is5.9x worse than x0 yet full native48epoch angle improves to.0378 with competitive attributes; do not infer final learning from2K alone. Test the native8epoch prefix with original epoch shuffle and128epoch scheduler.',
        no_automatic_full128_deployment=True,
        interpretation='Learning-curve evidence at8 epochs, not completed128epoch performance.'))
    observations = []
    for arm, experiment in [('raw025', 'init_diffusion_omega025_native8'),
                            ('scaled025', 'init_diffusion_omega025_timescaled_native8')]:
        assert all(sha(snapshot / p) == h for p, h in manifest.items()), 'Snapshot modified'
        while True:
            apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if len(apps) <= 1 and free >= 13312:
                break
            save(ART / 'status.json', dict(status='WAITING_FOR_LOCAL_GPU_HEADROOM',
                active_cuda_pids=apps, free_mib=free, arm=arm))
            time.sleep(30)
        output = RUNS / arm
        output.mkdir()
        audit = ART / arm
        audit.mkdir()
        command = ['taskset', '-c', '16-27', sys.executable, '-u',
            str(snapshot / 'tools/run_native_prefix_training_20261015.py'),
            '--gpu-memory-fraction', '.5', '--audit-output', str(audit),
            '--expected-initial-g1-sha256', smoke['initial_G1_sha256'],
            '--expected-initial-map-sha256', smoke['initial_map_sha256'],
            'experiment=' + experiment, 'optimizer=default', f'hydra.run.dir={output}',
            f'logger.wandb.name=init_diffusion_native8_{arm}_20261015',
            'logger.wandb.group=continuous_native8_20261015',
            'logger.wandb.tags=[native8,full50k,from_scratch,lr128]']
        with (output / 'console.log').open('x') as handle:
            child = subprocess.Popen(command, cwd=snapshot, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='online', MPLCONFIGDIR='/tmp/sim_mpl_continuous', HYDRA_FULL_ERROR='1'))
            save(ART / 'status.json', dict(status='NATIVE8_TRAINING', arm=arm, pid=child.pid,
                source_snapshot=str(snapshot), command=command, output=str(output)))
            rc = child.wait()
        assert rc == 0, f'{arm}: exit{rc}; preserve and review, no automatic native-fault retry'
        record = json.loads((audit / 'training_audit.json').read_text())
        assert record['status'] == 'NATIVE_PREFIX_COMPLETE' and record['actual_step'] == record['EMA_updates'] == 7712
        assert record['initial_EMA_updates'] == 0 and record['frozen_map_unchanged']
        assert len(record['epochs']) == 8 and all(r['actual_examples'] == 493542 for r in record['epochs'])
        assert len({r['order_sha256'] for r in record['epochs']}) == 8
        report = json.loads((output / 'sd_agent_metrics.json').read_text())
        target = json.loads((snapshot / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
        scope = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source', 'eval_set_sha256',
                 'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt')
        assert all(report[k] == target[k] for k in scope)
        assert report['decoder']['ema_num_updates'] == 7712 and report['decoder']['sampling_network_evaluations'] == 24
        checkpoint_paths = list(output.glob('bc/*/checkpoints/last.ckpt'))
        assert len(checkpoint_paths) == 1
        gc.disable()
        import torch
        torch.set_num_threads(1)
        state = torch.load(checkpoint_paths[0], map_location='cpu', weights_only=False)
        assert state['global_step'] == 7712
        assert state['state_dict']['encoder.init_decoder._extra_state']['ema']['num_updates'] == 7712
        del state
        row = dict(arm=arm, training_audit=record, metrics=report['agent_metrics'],
            checkpoint=str(checkpoint_paths[0]), report=str(output / 'sd_agent_metrics.json'),
            training_budget_epochs=8, full128epoch_quality_unproven=True)
        if observations:
            old = observations[0]['training_audit']
            for key in ('initial_G1_sha256', 'initial_map_sha256', 'epochs',
                        'final_CPU_RNG_sha256', 'final_CUDA_RNG_sha256'):
                assert record[key] == old[key], f'Paired native-prefix audit mismatch:{key}'
        observations.append(row)
        save(ART / 'observations.json', observations)
    ratios = {k: observations[1]['metrics'][k] / observations[0]['metrics'][k]
              for k in target['targets']}
    save(ART / 'decision.json', dict(paired_native8_ratios_scaled_to_raw=ratios,
        both_full50K_verified=True, actual_epoch_order_and_RNG_verified_equal=True,
        full128epoch_quality_unproven=True, next='Review curves and existing full runs before any128epoch deployment.'))
    save(ART / 'status.json', dict(status='NATIVE8_PAIRED_STUDY_COMPLETE'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
