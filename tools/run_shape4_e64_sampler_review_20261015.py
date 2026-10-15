"""Finite same-checkpoint native/MP24 review after the ongoing native8 pair."""
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
ART = BASE / 'shape4_e64_sampler_review_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/shape4_e64_sampler_review_v1'
SNAPSHOT = BASE / 'full_checkpoint_snapshots/shape4_e64_20261015'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'
WAIT_FOR_PID = 509864
REQUIRED_STUDY = None


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), RUNNER, ROOT / 'tools/runtime_cached_torch_loader_sweep.py',
                ROOT / 'tools/torch_pickle_class_cache.py', SNAPSHOT / 'config.yaml',
                SNAPSHOT / 'provenance.json', *sorted((ROOT / 'src/smart').rglob('*.py'))]}


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    proof = json.loads((BASE / 'shape4_e64_cpu_preflight_20261015.json').read_text())
    assert proof['status'] == 'SHAPE4_REAL64_NATIVE_CPU_STRICT_LOAD_PASS'
    assert proof['step'] == proof['EMA'] == 61696
    frozen = hashes()
    protocol = dict(registered_utc=now(), checkpoint_epoch=63, checkpoint_step=61696,
        checkpoint_sha256=json.loads((SNAPSHOT / 'provenance.json').read_text())['sha256'],
        interpretation='Actual intermediate64epoch weights; no final128epoch quality claim.',
        source_hashes=frozen, batch_size=64, precision='highest/32-true',
        stages=['MP256_functional_GEN8817', 'native_full50K_GEN2817',
                'MP24_full50K_GEN2817', 'MP24_full50K_GEN3817'],
        same_GEN_native_MP_source_and_GT_histograms_must_match=True,
        sampling_MP=dict(intervals=12, NFE=24, power=1.2, terminal_time=.1),
        resource_review_gate=dict(both_MP_at_least_six_of_seven_targets=True,
            both_MP_angle_target_ratio_at_most=1.5,
            both_MP_mean_angle_length_speed_target_ratio_at_most=1.0),
        resource_review_gate_is_not_final_success=True,
        no_automatic_training_stop_or_deployment=True,
        reason='Shape4 uses size matching weights4, not size loss weights. Its native Euler24 results cannot alone establish inferiority to native MP24 angular-velocity runs.',
        wait_for_pid=WAIT_FOR_PID, required_prior_study=None if REQUIRED_STUDY is None else str(REQUIRED_STUDY),
        no_automatic_failure_retries=True)
    save(ART / 'protocol.json', protocol)
    while alive(WAIT_FOR_PID):
        save(ART / 'status.json', dict(status='WAITING_FOR_REGISTERED_LIVE_PROCESS', pid=WAIT_FOR_PID, updated_utc=now()))
        time.sleep(30)
    if REQUIRED_STUDY is not None:
        prior = json.loads((REQUIRED_STUDY / 'status.json').read_text())
        assert prior['status'] == 'OMEGA025_E80_CONFIRMATION_COMPLETE', 'Review prior confirmation failure before using GPU'
    pair = json.loads((BASE / 'native_prefix8_pair_v1/status.json').read_text())
    assert pair['status'] == 'NATIVE8_PAIRED_STUDY_COMPLETE', 'Review failed native8 pair before using GPU'
    names = json.loads((BASE / 'full_sampler_50k_v2/full50000_files.json').read_text())
    assert len(names) == 50000 and len(set(names)) == 50000
    target = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    samplers = dict(native=dict(sampling_steps=24,
        sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.05)),
        midpoint=dict(sampling_steps=12,
        sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.1)))
    rows = []
    stages = [(256, 'midpoint', 8817), (50000, 'native', 2817),
              (50000, 'midpoint', 2817), (50000, 'midpoint', 3817)]
    scope = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source', 'eval_set_sha256',
             'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt')
    for count, integrator, seed in stages:
        assert hashes() == frozen, 'Evaluation source changed; review without retry'
        while subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip():
            save(ART / 'status.json', dict(status='WAITING_FOR_FREE_LOCAL_GPU', updated_utc=now()))
            time.sleep(30)
        label = f'shape4_e64_{integrator}_N{count}_GEN{seed}'
        manifest = ART / (label + '.manifest.json')
        sampler = ART / (label + '.sampler.json')
        save(manifest, names if count == 50000 else names[33000:33256])
        save(sampler, samplers[integrator])
        output = RUNS / label
        command = ['taskset', '-c', '16-31', sys.executable, '-u', str(RUNNER),
            '--checkpoint', str(SNAPSHOT / 'model.ckpt'), '--source-config', str(SNAPSHOT / 'config.yaml'),
            '--sampler', str(sampler), '--manifest', str(manifest), '--output', str(output),
            '--seed', str(seed), '--batch-size', '64', '--integration', integrator]
        if count == 50000:
            command += ['--full-set', '--r-reference', str(ROOT / 'configs/metrics/iu3fa4gx_full50000.json')]
        log = RUNS / (label + '.log')
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
            save(ART / 'status.json', dict(status='RUNNING', pid=child.pid, label=label,
                log=str(log), updated_utc=now()))
            result = child.wait()
        assert result == 0, f'{label}: exit{result}; preserve failure, no automatic retry'
        assert hashes() == frozen
        row = json.loads((output / 'verified.json').read_text())
        assert row['checkpoint_step'] == 61696 and set(row['observed_denoiser_calls_per_sample_batch']) == {24}
        report = json.loads((output / 'sd_agent_metrics.json').read_text())
        if count == 50000:
            assert all(key in report['agent_metrics'] for key in ['R', 'R_jsd', 'R_attr'])
            assert all(report[key] == target[key] for key in scope)
            row['target_ratios'] = {k: report['agent_metrics'][k] / value for k, value in target['targets'].items()}
            row['passed_targets'] = [k for k, value in row['target_ratios'].items() if value < 1]
        rows.append(row)
        save(ART / 'observations.json', rows)
    native, first, second = rows[1:]
    assert native['sampling_source_sha256_per_batch'] == first['sampling_source_sha256_per_batch']
    native_hist = json.loads((Path(native['output']) / 'feature_histograms.json').read_text())
    middle_hist = json.loads((Path(first['output']) / 'feature_histograms.json').read_text())
    assert all(native_hist[k]['matched_gt'] == middle_hist[k]['matched_gt'] for k in native_hist)
    viable = all(len(r['passed_targets']) >= 6 and r['target_ratios']['ang_dev_jsd'] <= 1.5
        and sum(r['target_ratios'][k] for k in ['ang_dev_jsd', 'length_jsd', 'speed_jsd']) / 3 <= 1
        for r in [first, second])
    save(ART / 'decision.json', dict(completed_utc=now(), both_MP_resource_gate_pass=viable,
        native_MP_GEN2817_noise_and_GT_exactly_matched=True,
        recommended_action='Retain training and review at96epochs' if viable else 'Review replacing shape4 with prepared five-layer omega0.1; preserve latest checkpoint before any stop',
        checkpoint_is_intermediate=True, final_goal_achieved=False,
        paired_MP_to_native_ratios={k: first['metrics'][k] / native['metrics'][k] for k in target['targets']},
        full_MP_results=[dict(seed=r['seed'],metrics=r['metrics'],passed_targets=r['passed_targets']) for r in [first, second]]))
    save(ART / 'status.json', dict(status='SHAPE4_E64_SAMPLER_REVIEW_COMPLETE', completed_utc=now()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-name', default='shape4_e64_sampler_review_v1')
    parser.add_argument('--wait-for-pid', type=int, default=509864)
    parser.add_argument('--required-study', type=Path)
    options = parser.parse_args()
    assert options.study_name and Path(options.study_name).name == options.study_name
    ART = BASE / options.study_name
    RUNS = ROOT / 'logs/init_diffusion_continuous_20261014' / options.study_name
    WAIT_FOR_PID = options.wait_for_pid
    REQUIRED_STUDY = options.required_study
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc), updated_utc=now()))
        raise
