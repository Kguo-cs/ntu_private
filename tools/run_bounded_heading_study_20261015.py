"""Finite paired16K study on one genuine intermediate rawomega checkpoint."""
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
ART = BASE / 'bounded_heading_study_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/bounded_heading_study_v1'
RUNNER = ROOT / 'tools/run_heading_support_probe_20261015.py'
SEEDS = (6817, 7817)


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    checkpoint = BASE / 'omega025_mid_checkpoint_20261015/model.ckpt'
    origin = json.loads((checkpoint.parent / 'provenance.json').read_text())
    assert sha(checkpoint) == origin['sha256'] and origin['global_step'] == 51092
    probe = json.loads((BASE / 'omega025_heading_support_probe_v3/support.json').read_text())
    assert probe['status'] == 'NATIVE_GPU_SUPPORT_PROBE_COMPLETE' and probe['mode'] == 'audit'
    names = json.loads((BASE / 'full_sampler_50k_v2/full50000_files.json').read_text())
    selected = names[16000:32384]
    assert len(selected) == len(set(selected)) == 16384
    manifest = ART / 'manifest16384.pkl'
    manifest.write_bytes(pickle.dumps({'files': selected}, protocol=4))
    files = [Path(__file__), RUNNER, ROOT / 'tools/bounded_heading_velocity_20261015.py',
             ROOT / 'tools/run_audited_capped_init_eval_20261015.py',
             *sorted((ROOT / 'src/smart').rglob('*.py')), *sorted((ROOT / 'configs').rglob('*.yaml'))]
    frozen = {str(p): sha(p) for p in files}
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=frozen, checkpoint=origin, checkpoint_training_complete=False,
        GEN=SEEDS, scenes=16384, manifest_sha256=sha(manifest), batch_size=64,
        precision='highest', NFE=24, no_training=True,
        treatments=['audit unchanged rawomega', 'clip free-headingomega to[-pi,pi]'],
        rationale='Every supervised shortest-arc angular-velocity target is in[-pi,pi]; projection is MSE nonexpansive on bridge inputs, not a distribution-quality theorem.',
        gates=dict(each_GEN_angle_ratio_below=1., mean_angle_ratio_at_most=.98,
            each_GEN_each_attribute_ratio_at_most=1.05, each_GEN_geometry_ratio_at_most=1.10,
            vehicle_count_ratio_range=[.995, 1.005]),
        resource_rule='At most two other CUDA applications and at least5.5GiB free; this evaluator capped at18%.',
        promotion='Subset results nominate a full50K paired confirmation; intermediate weights never represent completed128epoch training.',
        next_holdout_GEN_if_gate_passes=8817))
    observed = {}
    observations = []
    for mode in ('audit', 'clip'):
        for seed in SEEDS:
            assert all(sha(Path(p)) == h for p, h in frozen.items()), 'Source changed after registration'
            while True:
                apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], text=True).splitlines()
                free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if len(apps) <= 2 and free >= 5632:
                    break
                save(ART / 'status.json', dict(status='WAITING_FOR_HEADROOM', active_pids=apps, free_mib=free))
                time.sleep(30)
            label = f'{mode}_GEN{seed}_N16384'
            output = RUNS / label
            output.mkdir()
            folder = ART / label
            folder.mkdir()
            command = ['taskset', '-c', '20-31', sys.executable, '-u', str(RUNNER),
                '--heading-bound-mode', mode, '--heading-bound-report', str(folder / 'support.json'),
                '--gpu-memory-fraction', '.18', '--resource-report', str(folder / 'resources.json'),
                '--audit-report', str(folder / 'audit.json'), '--histogram-report', str(folder / 'histograms.json'),
                'experiment=init_diffusion_omega025_support_probe', f'seed={seed}',
                f'data.scenario_dreamer_eval_set={manifest}', 'trainer.limit_test_batches=1.0',
                f'hydra.run.dir={output}']
            with (output / 'console.log').open('x') as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous', HYDRA_FULL_ERROR='1'))
                save(ART / 'status.json', dict(status='RUNNING', mode=mode, seed=seed, pid=child.pid,
                    command=command, log=str(output / 'console.log')))
                rc = child.wait()
            assert rc == 0, f'{label} exited{rc}; preserve failure, no automatic retry'
            report = json.loads((output / 'sd_agent_metrics.json').read_text())
            audit = json.loads((folder / 'audit.json').read_text())
            hist = json.loads((folder / 'histograms.json').read_text())
            assert report['num_samples'] == report['num_gt_samples'] == 16384
            assert audit['decoder_ema_updates'] == 51092 and len(audit['rows']) == 256
            assert {r['NFE'] for r in audit['rows']} == {24}
            assert all(abs(report['agent_metrics'][k] - v) < 1e-12
                       for k, v in audit['recomputed_metrics'].items())
            row = dict(mode=mode, seed=seed, scenes=16384, metrics=report['agent_metrics'],
                vehicles=report['num_generated_vehicles'], report=str(output / 'sd_agent_metrics.json'),
                checkpoint_training_complete=False)
            if mode == 'audit':
                observed[seed] = row, audit, hist
            else:
                control, old_audit, old_hist = observed[seed]
                assert [r['source_sha256'] for r in audit['rows']] == [r['source_sha256'] for r in old_audit['rows']]
                assert all(hist[k]['matched_gt'] == old_hist[k]['matched_gt'] for k in hist)
                row['ratios'] = {k: row['metrics'][k] / control['metrics'][k]
                                 for k in audit['recomputed_metrics']}
                row['vehicle_ratio'] = row['vehicles'] / control['vehicles']
                row['source_noise_and_GT_histograms_matched'] = True
            observations.append(row)
            save(ART / 'observations.json', observations)
    candidates = [r for r in observations if r['mode'] == 'clip']
    mean_angle = sum(r['ratios']['ang_dev_jsd'] for r in candidates) / 2
    eligible = mean_angle <= .98 and all(r['ratios']['ang_dev_jsd'] < 1
        and max(r['ratios'][k] for k in ('length_jsd', 'width_jsd', 'speed_jsd')) <= 1.05
        and max(r['ratios'][k] for k in ('lat_dev_jsd', 'nearest_dist_jsd')) <= 1.10
        and .995 <= r['vehicle_ratio'] <= 1.005 for r in candidates)
    save(ART / 'decision.json', dict(nominate_full50K_paired_holdout=eligible,
        mean_angle_ratio=mean_angle, checkpoint_training_complete=False,
        full50K_quality_unproven=True, candidates=candidates))
    save(ART / 'status.json', dict(status='FINITE_16K_PAIRED_STUDY_COMPLETE'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
