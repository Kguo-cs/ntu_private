"""CPU forward/backward through the native mixed-objective Flow path."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import datetime
    import importlib.util
    import json
    import torch
    import yaml
    from unittest.mock import patch
    from src.smart.diffusion.initial_diffusion import InitDiffusion
    from tools.shared_heading_velocity_20261015 import install_training_policy

    torch.set_num_threads(1)
    torch.manual_seed(7302)
    root = Path(__file__).resolve().parents[1]
    base = root / 'output/reports/init_diffusion_continuous_20261014'
    spec = importlib.util.spec_from_file_location('ego_fixture', root / 'tests/test_init_diffusion_ego_partial.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    test = fixture.InitDiffusionEgoPartialTest()
    options = yaml.safe_load((base / 'full_checkpoint_snapshots/sampler_snapshots_20261015/roadK8/config.yaml').read_text())['model']['model_config']['decoder']['init_diffusion']
    options.pop('normalizer_path')
    options.update(heading_objective='angular_velocity')
    options['component_loss']['components']['heading'].update(time_power=0, weight=.025)
    install_training_policy()
    with patch.object(InitDiffusion, '_make_args', return_value=test.args()):
        wrapper = InitDiffusion(128, 8, 64, test.processor(), False, **options)
    x, agent, feature = test.inputs()
    x = torch.cat((x[:, :6], x[:, 6:8].norm(dim=-1, keepdim=True)), -1)
    agent['expert_input'] = x.clone()
    feature['pt_token'] = wrapper.G1.model.lane_embed(torch.randn(3, 128))
    wrapper.train()
    loss = wrapper.G1._supervised_loss(x, agent, feature)
    total = loss[0].mean()
    assert torch.isfinite(total)
    total.backward()
    head = wrapper.G1.model.to_out_m_delta.mlp[-1]
    gradient_by_row = head.weight.grad.abs().sum(-1)
    assert gradient_by_row[2] == 0
    assert all(gradient_by_row[i] > 0 for i in (0, 1, 3, 4, 5, 6))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.G1.parameters())
    metrics = agent['_init_diffusion_component_metrics']
    assert metrics['heading_weighted_loss'] >= 0
    result = dict(status='PASS', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        generator_parameters=sum(p.numel() for p in wrapper.G1.parameters()),
        native_component_forward_backward=True, finite_gradients=True,
        heading_loss=float(metrics['heading_loss']),
        weighted_heading_loss=float(metrics['heading_weighted_loss']),
        state_head_gradient_by_row=gradient_by_row.tolist(),
        unused_cosine_row_zero_gradient=True,
        limitation='Three-agent CPU correctness fixture. GPU preflight and paired training remain required.')
    (base / 'shared_heading_training_path_proof_20261015.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
