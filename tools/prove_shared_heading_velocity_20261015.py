"""Exercise real Flow modules, gradients, EMA copies and checkpoint guards."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import copy
    import datetime
    import hashlib
    import importlib.util
    import json
    import torch
    import yaml
    from unittest.mock import patch
    from tools.shared_heading_velocity_20261015 import install_training_policy
    from src.smart.diffusion.initial_diffusion import InitDiffusion

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('ego_fixture', root / 'tests/test_init_diffusion_ego_partial.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    test = fixture.InitDiffusionEgoPartialTest()
    config_path = root / 'output/reports/init_diffusion_continuous_20261014/full_checkpoint_snapshots/sampler_snapshots_20261015/roadK8/config.yaml'
    common = yaml.safe_load(config_path.read_text())['model']['model_config']['decoder']['init_diffusion']
    common.pop('normalizer_path')  # Both constructors use the same initial normalization buffers.
    def build(options):
        with patch.object(InitDiffusion, '_make_args', return_value=test.args()):
            return InitDiffusion(128, 8, 64, test.processor(), False, **options)
    torch.manual_seed(7301)
    baseline = build(common)
    baseline_rng = torch.get_rng_state().clone()
    install_training_policy()
    torch.manual_seed(7301)
    candidate = build(dict(common, heading_objective='angular_velocity'))
    assert torch.equal(baseline_rng, torch.get_rng_state())
    a = {k: v for k, v in baseline.state_dict().items() if torch.is_tensor(v)}
    b = {k: v for k, v in candidate.state_dict().items() if torch.is_tensor(v)}
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    model = candidate.G1.model
    features = torch.randn(5, model.hidden_dim, requires_grad=True)
    omega = model.to_out_heading_velocity(features)
    torch.testing.assert_close(omega, model.to_out_m_delta(features)[:, 3:4], rtol=0, atol=0)
    omega.square().mean().backward()
    final = model.to_out_m_delta.mlp[-1]
    assert final.weight.grad[3].abs().sum() > 0
    assert final.weight.grad[:3].count_nonzero() == 0
    assert final.weight.grad[4:].count_nonzero() == 0
    clone = copy.deepcopy(model)
    assert clone.to_out_heading_velocity.__self__ is clone
    with torch.no_grad():
        clone.to_out_m_delta.mlp[-1].bias[3].add_(1)
    torch.testing.assert_close(clone.to_out_heading_velocity(features), omega + 1, rtol=1e-6, atol=1e-6)
    # A large omega must not wrap back to a different velocity during sampling.
    state = torch.tensor([[0., 0., 1., 0., 4., 2., 0.]])
    time = torch.tensor([[.8]])
    prediction = torch.cat((state, torch.tensor([[5.]])), -1)
    velocity, clean = candidate.G1._prediction_velocity(state, time, prediction)
    torch.testing.assert_close(velocity[:, 2:4], torch.tensor([[0., 5.]]))
    torch.testing.assert_close(clean[:, 2:4], torch.stack((torch.cos(torch.tensor([-4.])), torch.sin(torch.tensor([-4.]))), -1))
    extra = candidate.get_extra_state()
    assert extra['denoiser_options']['heading_velocity_readout'] == 'shared_state_row3'
    incompatible = copy.deepcopy(extra)
    del incompatible['denoiser_options']['heading_velocity_readout']
    try:
        candidate.set_extra_state(incompatible)
    except ValueError:
        pass
    else:
        raise AssertionError('Native separate-head checkpoint was not rejected')
    result = dict(status='PASS', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identical_initial_state_tensors=len(a), identical_torch_rng=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        added_parameters=0, shared_row_gradient=True, deepcopy_ema_readout_bound_to_copy=True,
        unwrapped_velocity_sampling=True, incompatible_checkpoint_rejected=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='CPU correctness proof only. No training stability or JSD gain claimed.')
    out = root / 'output/reports/init_diffusion_continuous_20261014/shared_heading_velocity_proof_20261015.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
