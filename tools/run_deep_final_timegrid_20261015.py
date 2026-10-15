"""Finite matched 24-NFE time-grid study, with preregistered promotion gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'deep_final_timegrid_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/deep_final_timegrid_v1'
RUNNER = ROOT / 'tools/run_audited_capped_init_eval_20261015.py'
SEEDS = (2817, 3817)
POWERS = (1.2, 1.1, 1.3)
SCOPE_KEYS = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source',
              'eval_set_sha256', 'official_source_sha256',
              'reference_source_manifest_sha256', 'feature_counts_gt')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    source_files = [Path(__file__), RUNNER, *sorted((ROOT / 'src/smart').rglob('*.py')),
                    *sorted((ROOT / 'configs').rglob('*.yaml'))]
    frozen = {str(p): digest(p) for p in source_files}
    checkpoint = BASE / 'full_checkpoint_snapshots/deep_final128_20261015/model.ckpt'
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == 'f5e5eebd3250478761950cdc7086207e5b0e0781ebfe52c434cc4f36e0ab251d'
    names = json.loads((BASE / 'full_sampler_50k_v2/full50000_files.json').read_text())
    assert len(names) == 50000 and len(set(names)) == 50000
    manifests = {}
    for count, selected in ((8192, names[25000:33192]), (50000, names)):
        path = ART / f'manifest{count}.pkl'
        path.write_bytes(pickle.dumps({'files': selected}, protocol=4))
        manifests[count] = path
    targets = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    save(ART / 'protocol.json', dict(registered_utc=now(), source_hashes=frozen,
        checkpoint_sha256=checkpoint_hash, checkpoint_step=123392, seeds=SEEDS,
        powers=POWERS, terminal_time=.1, integration='midpoint', intervals=12,
        network_calls=24, batch_size=64, precision='highest', no_training=True,
        manifest_hashes={str(n): digest(p) for n, p in manifests.items()},
        gates=dict(each_seed_angle_ratio_below=1., mean_angle_ratio_at_most=.96,
            each_seed_each_attribute_ratio_at_most=1.10, mean_attribute_ratio_at_most=1.05,
            generated_vehicle_ratio_range=[.995, 1.005], onroad_fraction_change_at_least=-.005,
            geometry='candidate subset metrics below absolute SD/VW limits'),
        full_confirmation='Only a passing candidate runs two complete50K GENs; no pooled seed claim.',
        caveats=['All candidates use identical source-noise hashes per batch and matched GT histograms.',
                 'Changing grid timing can change the number of collision-guidance applications.',
                 'This is a sampler comparison on one completed training seed, not independent retraining.']))
    observations = []

    def run(power, seed, count):
        assert all(digest(Path(p)) == sha for p, sha in frozen.items()), 'Registered source modified'
        label = f'p{power:.1f}_GEN{seed}_N{count}'
        folder = RUNS / label
        folder.mkdir()
        audit_dir = ART / label
        audit_dir.mkdir()
        command = ['taskset', '-c', '20-31', sys.executable, '-u', str(RUNNER),
            '--gpu-memory-fraction', '.25', '--resource-report', str(audit_dir / 'resources.json'),
            '--audit-report', str(audit_dir / 'audit.json'),
            '--histogram-report', str(audit_dir / 'histograms.json'),
            'experiment=init_diffusion_657_n24_road_k8_match2_d5_final_midpoint_eval',
            f'seed={seed}', 'data.test_batch_size=64', 'data.num_workers=0',
            f'data.scenario_dreamer_eval_set={manifests[count]}', 'trainer.limit_test_batches=1.0',
            f'model.model_config.sd_require_full_set={str(count == 50000).lower()}',
            f'model.model_config.decoder.init_diffusion.sampling_time_grid.power={power}',
            f'hydra.run.dir={folder}']
        log = folder / 'console.log'
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous', HYDRA_FULL_ERROR='1'))
            save(ART / 'status.json', dict(status='RUNNING', label=label, pid=child.pid,
                command=command, log=str(log), updated_utc=now()))
            rc = child.wait()
        assert rc == 0, f'{label}: process exit {rc}; inspect {log}'
        audit = json.loads((audit_dir / 'audit.json').read_text())
        report = json.loads((folder / 'sd_agent_metrics.json').read_text())
        hist = json.loads((audit_dir / 'histograms.json').read_text())
        assert audit['status'] == 'PASS' and audit['decoder_ema_updates'] == 123392
        assert len(audit['rows']) == (count + 63) // 64
        assert {r['NFE'] for r in audit['rows']} == {24}
        assert report['num_samples'] == report['num_gt_samples'] == count
        assert report['decoder']['sampling_time_grid']['power'] == power
        assert report['runtime']['float32_matmul_precision'] == 'highest'
        assert all(abs(report['agent_metrics'][k] - v) < 1e-12
                   for k, v in audit['recomputed_metrics'].items())
        if count == 50000:
            assert all(report[k] == targets[k] for k in SCOPE_KEYS)
        row = dict(power=power, seed=seed, scenes=count, report=str(folder / 'sd_agent_metrics.json'),
            audit=str(audit_dir / 'audit.json'), histogram=str(audit_dir / 'histograms.json'),
            metrics=report['agent_metrics'], vehicles=report['num_generated_vehicles'],
            onroad_fraction=report['feature_counts_generated'][1] / report['num_generated_vehicles'],
            guidance_counts=sorted(set(r['guidance_calls'] for r in audit['rows'])))
        observations.append(row)
        save(ART / 'observations.json', observations)
        return row, audit, hist

    controls = {seed: run(1.2, seed, 8192) for seed in SEEDS}
    candidates = []
    for power in POWERS[1:]:
        rows = []
        for seed in SEEDS:
            row, audit, hist = run(power, seed, 8192)
            control, old_audit, old_hist = controls[seed]
            assert [r['source_sha256'] for r in audit['rows']] == [r['source_sha256'] for r in old_audit['rows']]
            assert all(hist[k]['matched_gt'] == old_hist[k]['matched_gt'] for k in hist)
            ratios = {k: row['metrics'][k] / control['metrics'][k]
                      for k in targets['targets'] if k.endswith('_jsd')}
            attrs = [ratios[k] for k in ('length_jsd', 'width_jsd', 'speed_jsd')]
            vehicle_ratio = row['vehicles'] / control['vehicles']
            onroad_change = row['onroad_fraction'] - control['onroad_fraction']
            passed = ratios['ang_dev_jsd'] < 1 and max(attrs) <= 1.10
            passed &= .995 <= vehicle_ratio <= 1.005 and onroad_change >= -.005
            passed &= all(row['metrics'][k] < targets['targets'][k]
                          for k in ('nearest_dist_jsd', 'lat_dev_jsd'))
            rows.append(dict(seed=seed, ratios=ratios, vehicle_ratio=vehicle_ratio,
                             onroad_change=onroad_change, seed_gate=passed))
        mean_angle = sum(r['ratios']['ang_dev_jsd'] for r in rows) / 2
        mean_attr = sum(r['ratios'][k] for r in rows
                        for k in ('length_jsd', 'width_jsd', 'speed_jsd')) / 6
        eligible = all(r['seed_gate'] for r in rows) and mean_angle <= .96 and mean_attr <= 1.05
        candidates.append(dict(power=power, mean_angle_ratio=mean_angle,
                               mean_attribute_ratio=mean_attr, eligible=eligible, rows=rows))
        save(ART / 'subset_decisions.json', candidates)
    eligible = [r for r in candidates if r['eligible']]
    if eligible:
        selected = min(eligible, key=lambda r: r['mean_angle_ratio'])['power']
        confirmations = [run(selected, seed, 50000)[0] for seed in SEEDS]
        save(ART / 'full_decision.json', dict(power=selected, confirmations=confirmations,
            both_GEN_same_checkpoint_absolute_metric_gate_pass=all(
                all(r['metrics'][k] < v for k, v in targets['targets'].items()) for r in confirmations),
            independent_training_seed_unproven=True))
    save(ART / 'status.json', dict(status='FINITE_STUDY_COMPLETE', completed_utc=now(),
        eligible_powers=[r['power'] for r in eligible], full_confirmation_run=bool(eligible)))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW',
                error=f'{type(exc).__name__}: {exc}', updated_utc=now()))
        raise
