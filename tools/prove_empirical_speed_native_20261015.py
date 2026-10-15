"""Prove the TRAIN speed source integrates with the actual native recipe."""
from pathlib import Path
import copy
import gc
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
gc.disable()


def main():
    import jinja2
    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from src.smart.diffusion.attribute_source import AttributeQuantileSource
    from src.smart.diffusion.scale_flow import _noise_endpoint
    from tools.prove_native_circular_matching_20261015 import digest

    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    configs = []
    for experiment in ('init_diffusion_omega025_native8', 'init_diffusion_omega025_empirical_speed_native8'):
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
            cfg = compose(config_name='run', overrides=['experiment='+experiment, 'optimizer=default'])
        with open_dict(cfg):
            cfg.paths.output_dir = '/tmp/sim_empirical_speed_native_cpu'
        configs.append(cfg)
    models = []
    states = []
    for cfg in configs:
        torch.manual_seed(4817)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        models.append(model)
        states.append(torch.get_rng_state().clone())
    baseline, candidate = models
    flow = candidate.encoder.init_decoder.G1
    assert sum(p.numel() for p in flow.parameters()) == 8083978
    for name, value in baseline.named_parameters():
        assert torch.equal(value, dict(candidate.named_parameters())[name]), name
    assert torch.equal(states[0], states[1])
    legacy_buffers = dict(baseline.named_buffers())
    new_buffers = dict(candidate.named_buffers())
    assert all(torch.equal(v, new_buffers[k]) for k,v in legacy_buffers.items())
    source = flow.model.attribute_source
    assert source.speed and not source.shape and source.knots == 149659
    gen = torch.Generator().manual_seed(241015)
    z = torch.randn(100000, 7, generator=gen)
    before = torch.get_rng_state().clone()
    raw = _noise_endpoint(baseline.encoder.init_decoder.G1.model, z, 1.)
    endpoint = _noise_endpoint(flow.model, z, 1.)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(raw[:, :6], endpoint[:, :6])
    assert torch.isfinite(endpoint).all() and (endpoint[:,6]>=0).all()
    source_state = copy.deepcopy(source.state_dict())
    portable_options = OmegaConf.to_container(configs[1].model.model_config.decoder.init_diffusion.attribute_source)
    portable_options['path'] = None
    portable = AttributeQuantileSource(portable_options)
    portable.load_state_dict(source_state, strict=True)
    assert torch.equal(portable(z, raw), endpoint)
    # Native metadata must stop old Gaussian weights being resumed with a new source.
    try:
        candidate.encoder.init_decoder.set_extra_state(baseline.encoder.init_decoder.get_extra_state())
    except ValueError:
        incompatible_checkpoint_rejected = True
    else:
        raise AssertionError('Gaussian checkpoint silently accepted empirical speed source')
    spec = importlib.util.spec_from_file_location('fixture', ROOT/'tests/test_init_diffusion_ego_partial.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    state, agent, features = fixture.InitDiffusionEgoPartialTest().inputs()
    state = torch.cat((state[:,:6], state[:,6:8].norm(dim=-1, keepdim=True)), -1)
    agent['expert_input'] = state.clone()
    torch.manual_seed(142015)
    features['pt_token'] = flow.model.lane_embed(torch.randn(3,128)).detach()
    torch.manual_seed(142017)
    result = flow._supervised_loss(state, agent, features)
    loss = result[0].mean() + result[1]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)
    files = [Path(__file__), ROOT/'src/smart/diffusion/attribute_source.py',
        ROOT/'src/smart/diffusion/scale_flow.py', ROOT/'src/smart/diffusion/initial_diffusion.py',
        ROOT/'configs/experiment/init_diffusion_657_n24_road_k8_omega025_empirical_speed_full.yaml',
        ROOT/'configs/experiment/init_diffusion_omega025_empirical_speed_native8.yaml',
        ROOT/'docs/experiments/init_diffusion_train_attribute_source_full_20261015/quantiles.pt']
    summary = dict(status='EMPIRICAL_SPEED_NATIVE_FULL_CONFIG_CPU_PASS_NOT_TRAINED',
        parameters=8083978, source_knots=source.knots, source_sha256=source.source_sha256,
        all_initial_parameters_and_legacy_buffers_exact=True, initial_CPU_RNG_exact=True,
        endpoint_changes_only_speed_no_extra_RNG=True, full_native_backward_finite=True,
        portable_source_checkpoint_exact=True, incompatible_checkpoint_rejected=incompatible_checkpoint_rejected,
        initial_G1_sha256=digest(flow.named_parameters()),
        source_speed_zero_fraction=float((endpoint[:,6]==0).float().mean()),
        source_speed_under005_fraction=float((endpoint[:,6]<.05).float().mean()),
        source_speed_under05_fraction=float((endpoint[:,6]<.5).float().mean()),
        TRAIN_only=True, generation_quality_unproven=True,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output=ROOT/'output/reports/init_diffusion_continuous_20261014/empirical_speed_native_cpu_proof_20261015.json'
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    sys.stdout.flush()
    __import__('os')._exit(0)


if __name__ == '__main__':
    main()
