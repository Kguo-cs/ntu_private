"""Check the new five-layer objective candidate against the native x0 recipe."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=float, choices=[.025, .1], default=.025)
    options = parser.parse_args()
    suffix = '025' if options.weight == .025 else '10'
    candidate = 'init_diffusion_657_n24_road_k8_match2_d5_omega' + suffix + '_full'
    gc.disable()
    import jinja2, torch, hydra
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf, open_dict
    from tools.run_native_prefix_training_20261015 import digest_parameters
    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    models = []
    states = []
    configs = []
    for experiment in ('init_diffusion_657_n24_road_k8_match2_d5_midpoint_full',
                       candidate):
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
            cfg = compose(config_name='run', overrides=['experiment=' + experiment, 'optimizer=default'])
        with open_dict(cfg):
            cfg.paths.output_dir = '/tmp/sim_deep_shared_omega_proof'
        torch.manual_seed(4817)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        models.append(model)
        states.append(torch.get_rng_state().clone())
        configs.append(cfg)
    first, second = models
    assert torch.equal(*states)
    old, new = first.encoder.init_decoder.G1, second.encoder.init_decoder.G1
    assert digest_parameters(old) == digest_parameters(new)
    assert sum(p.numel() for p in new.parameters()) == 12562954
    old_tensors = {k: v for k, v in old.state_dict().items() if torch.is_tensor(v)}
    new_tensors = {k: v for k, v in new.state_dict().items() if torch.is_tensor(v)}
    assert old_tensors.keys() == new_tensors.keys()
    assert all(torch.equal(v, new_tensors[k]) for k, v in old_tensors.items())
    for cfg in configs:
        assert cfg.float32_matmul_precision == 'highest'
        matching = cfg.model.model_config.decoder.init_diffusion.matching_weights
        assert matching.heading == matching.velocity == 2
        assert cfg.model.model_config.lr_total_steps == 128 and cfg.data.num_workers == 8
    before = first.encoder.init_decoder.G1.component_loss.config['components']
    after = second.encoder.init_decoder.G1.component_loss.config['components']
    assert all(before[k] == after[k] for k in before if k != 'heading')
    assert after['heading'] == dict(kind='mse', time_power=0, weight=options.weight, scale='none')
    spec = importlib.util.spec_from_file_location('heading_fixture', ROOT / 'tests/test_init_diffusion_ego_partial.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    state, agent, feature = fixture.InitDiffusionEgoPartialTest().inputs()
    state = torch.cat((state[:, :6], state[:, 6:8].norm(dim=-1, keepdim=True)), -1)
    agent['expert_input'] = state.clone()
    feature['pt_token'] = new.model.lane_embed(torch.randn(3, 128)).detach()
    new.train()
    losses = new._supervised_loss(state, agent, feature)
    loss = losses[0].mean() + losses[1]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in new.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    row_grad = new.model.to_out_m_delta.mlp[-1].weight.grad.abs().sum(-1)
    assert row_grad[3] > 0 and row_grad[2] == 0
    report = dict(status='NATIVE_DEEP_SHARED_OMEGA_CPU_PROOF_PASS',
        initial_G1_sha256=digest_parameters(new), parameters=12562954,
        native_initial_tensors_and_CPU_RNG_equal=True, initial_tensor_count=len(new_tensors),
        matching_h2_v2_unchanged=True, precision_highest_unchanged=True,
        non_heading_loss_components_unchanged=True, native_full_backward_finite=True,
        shared_row3_gradient_active=True, unused_row2_gradient_zero=True,
        heading_objective_intentionally_changed=True, full_quality_unproven=True,
        heading_weight=options.weight,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / ('configs/experiment/' + candidate + '.yaml')]})
    name = 'deep_shared_omega' if suffix == '025' else 'deep_shared_omega10'
    path = ROOT / ('output/reports/init_diffusion_continuous_20261014/' + name + '_cpu_proof_20261015.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
