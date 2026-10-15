"""Record native-worker preflight and the independent seed0 full50K result."""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
LOGS = ROOT / 'logs/init_diffusion_continuous_20261014'


def save(name, value):
    (BASE / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    new_dir = LOGS / 'native_worker_heading_preflight_v1/control16'
    old_dir = LOGS / 'periodic_edge_training_v2/control_T4817_S16_N256'
    new = torch.load(new_dir / 'control/last.ckpt', map_location='cpu', weights_only=False)
    old = torch.load(old_dir / 'control/step_16.ckpt', map_location='cpu', weights_only=False)
    assert new['global_step'] == old['global_step'] == 16
    assert new['scratch_sweep'] == old['scratch_sweep']
    assert new['state_dict']['encoder.init_decoder._extra_state']['ema']['num_updates'] == 16
    a, b = new['scratch_sweep_rng'], old['scratch_sweep_rng']
    rng = dict(python=a['python'] == b['python'],
        numpy=a['numpy'][0] == b['numpy'][0] and np.array_equal(a['numpy'][1], b['numpy'][1])
              and a['numpy'][2:] == b['numpy'][2:],
        torch_cpu=torch.equal(a['torch_cpu'], b['torch_cpu']),
        torch_cuda=all(torch.equal(v, w) for v, w in zip(a['torch_cuda'], b['torch_cuda'])))
    assert all(rng.values())
    tensors = [(k, v, old['state_dict'][k]) for k, v in new['state_dict'].items()
               if isinstance(v, torch.Tensor)]
    assert all(torch.isfinite(v).all() for _, v, _ in tensors)
    changed = [(k, float((v - w).abs().max())) for k, v, w in tensors if not torch.equal(v, w)]
    new_summary = json.loads((new_dir / 'summary.json').read_text())[0]
    old_summary = json.loads((old_dir / 'summary.json').read_text())[0]
    save('native_worker_heading_preflight_review_20261015.json', dict(
        status='FINITE_TRAIN_AND_RNG_CHECK_PASS_NOT_BITWISE_WEIGHT_EQUIVALENCE',
        reviewed_utc=datetime.now(timezone.utc).isoformat(), actual_steps=16, ema_updates=16,
        same_initial_weights_map_manifest_calibration=True, end_training_rng_equal=rng,
        finite_checkpoint_tensors=True, unchanged_tensor_count=len(tensors)-len(changed),
        changed_tensor_count=len(changed), maximum_tensor_difference=max(v for _, v in changed),
        largest_differences=sorted(changed, key=lambda r: r[1], reverse=True)[:8],
        train_seconds_native8=new_summary['train_seconds'],
        train_seconds_previous_native0=old_summary['train_seconds'],
        resource_report=str(new_dir / 'native_worker_resource_report.json'),
        cpu_input_parity_report=str(BASE / 'worker_loading_parity_v1/comparison.json'),
        caveats=['Training weights are not bitwise identical; no worker-only accuracy claim.',
                 'CPU input parity and matching end RNG are verified independently.',
                 'Native8 fresh studies need fresh paired controls; cyclic resume is still forbidden.',
                 '16 steps prove functionality, not long-run stability or final quality.']))
    path = LOGS / 'deep_final128_native_seed0_capped/full50000/sd_agent_metrics.json'
    report = json.loads(path.read_text())
    targets = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    keys = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source',
            'eval_set_sha256', 'official_source_sha256', 'reference_source_manifest_sha256',
            'feature_counts_gt')
    assert all(report[k] == targets[k] for k in keys)
    assert report['decoder']['ema_num_updates'] == 123392
    assert report['decoder']['sampling_network_evaluations'] == 24
    assert report['relative_score']['scope_matches_reference']
    assert all(math.isfinite(report['agent_metrics'][k]) for k in targets['targets'])
    passed = [k for k, v in targets['targets'].items() if report['agent_metrics'][k] < v]
    save('deep_final128_native_seed0_review_20261015.json', dict(status='VERIFIED_FULL50K',
        report=str(path), seed=0, batch_size=64, checkpoint_step=123392, NFE=24,
        metrics=report['agent_metrics'], passed_targets=passed, all7_pass=len(passed)==7,
        target_ratios={k: report['agent_metrics'][k]/v for k, v in targets['targets'].items()},
        caveat='Third generation seed, batch64; earlier seeds2817/3817 used batch128. One earlier seed still fails angle. Independent training seed remains unproved.'))
    print(json.dumps(dict(worker_preflight='PASS_WITH_WEIGHT_DIFFERENCE_CAVEAT',
        seed0_complete50K_passes=len(passed), metrics=report['agent_metrics'])))


if __name__ == '__main__':
    main()
