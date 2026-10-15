"""Add same-batch full controls before inspecting candidate50K outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'deep_final_timegrid_full_controls_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/deep_final_timegrid_full_controls_v1'
RUNNER = ROOT / 'tools/run_audited_capped_init_eval_20261015.py'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    predecessor = BASE / 'deep_final_timegrid_v1'
    assert not (predecessor / 'full_decision.json').exists(), 'Register controls before seeing full outcomes'
    assert not (predecessor / 'p1.3_GEN2817_N50000/audit.json').exists()
    tracked = [RUNNER, Path(__file__), *sorted((ROOT / 'src/smart').rglob('*.py')),
               *sorted((ROOT / 'configs').rglob('*.yaml'))]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        before_candidate_full_outcomes=True, source_hashes=frozen, power=1.2,
        GEN=[2817, 3817], scenes=50000, batch_size=64, NFE=24,
        reason='Existing same-checkpoint full controls used batch128, which changes source noise streams.',
        purpose='Full paired attribution and absolute target confirmation; never compare batch64 treatment causally against batch128 controls.',
        resource_rule='Wait for predecessor terminal success, at most one other live CUDA application, and at least6.5GiB free.'))
    while True:
        if (predecessor / 'failure.json').exists():
            raise RuntimeError('Predecessor failed; do not allocate another evaluator')
        status = json.loads((predecessor / 'status.json').read_text())
        if status['status'] == 'FINITE_STUDY_COMPLETE':
            break
        save(ART / 'status.json', dict(status='WAITING_FOR_PREDECESSOR_FULL_EVALUATION'))
        time.sleep(30)
    observations = []
    target = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    for seed in (2817, 3817):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in frozen.items())
        while True:
            apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if len(apps) <= 1 and free >= 6656:
                break
            save(ART / 'status.json', dict(status='WAITING_FOR_LOCAL_GPU_HEADROOM',
                                          active_cuda_pids=apps, free_mib=free))
            time.sleep(30)
        label = f'p1.2_GEN{seed}_N50000'
        output = RUNS / label
        output.mkdir()
        folder = ART / label
        folder.mkdir()
        command = ['taskset', '-c', '20-31', sys.executable, '-u', str(RUNNER),
            '--gpu-memory-fraction', '.25', '--resource-report', str(folder / 'resources.json'),
            '--audit-report', str(folder / 'audit.json'), '--histogram-report', str(folder / 'histograms.json'),
            'experiment=init_diffusion_657_n24_road_k8_match2_d5_final_midpoint_eval',
            f'seed={seed}', 'data.test_batch_size=64', 'data.num_workers=0',
            f'data.scenario_dreamer_eval_set={predecessor / "manifest50000.pkl"}',
            'trainer.limit_test_batches=1.0', 'model.model_config.sd_require_full_set=true',
            'model.model_config.decoder.init_diffusion.sampling_time_grid.power=1.2',
            f'hydra.run.dir={output}']
        with (output / 'console.log').open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
            save(ART / 'status.json', dict(status='FULL_CONTROL_RUNNING', seed=seed, pid=child.pid))
            rc = child.wait()
        assert rc == 0, f'Full control exit{rc}; preserve and inspect outputs'
        report = json.loads((output / 'sd_agent_metrics.json').read_text())
        audit = json.loads((folder / 'audit.json').read_text())
        candidate_folder = predecessor / f'p1.3_GEN{seed}_N50000'
        candidate_audit = json.loads((candidate_folder / 'audit.json').read_text())
        hist = json.loads((folder / 'histograms.json').read_text())
        candidate_hist = json.loads((candidate_folder / 'histograms.json').read_text())
        assert audit['decoder_ema_updates'] == candidate_audit['decoder_ema_updates'] == 123392
        assert [r['source_sha256'] for r in audit['rows']] == [r['source_sha256'] for r in candidate_audit['rows']]
        assert all(hist[k]['matched_gt'] == candidate_hist[k]['matched_gt'] for k in hist)
        assert all(r['NFE'] == 24 for r in audit['rows'])
        scope = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source', 'eval_set_sha256',
                 'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt')
        assert all(report[k] == target[k] for k in scope)
        candidate = next(r for r in json.loads((predecessor / 'observations.json').read_text())
                         if r['scenes'] == 50000 and r['seed'] == seed)
        row = dict(seed=seed, scenes=50000, power=1.2, report=str(output / 'sd_agent_metrics.json'),
            metrics=report['agent_metrics'], source_noise_and_GT_histograms_match_candidate=True,
            candidate_to_control_ratios={k: candidate['metrics'][k] / report['agent_metrics'][k]
                                        for k in target['targets']})
        observations.append(row)
        save(ART / 'observations.json', observations)
    save(ART / 'status.json', dict(status='PAIRED_FULL50K_CONTROLS_COMPLETE', observations=observations))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
