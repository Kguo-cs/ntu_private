"""Compare native legacy behavior to the frozen old implementation, then arc."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'native_circular_matching_cpu_v1'


def digest(tensors):
    result = hashlib.sha256()
    for name, tensor in tensors:
        result.update(name.encode())
        if tensor is None:
            result.update(b'None')
            continue
        value = tensor.detach().cpu().contiguous()
        result.update(str((str(value.dtype), tuple(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def capture(root, output, geometry):
    sys.path.insert(0, str(root))
    import gc
    gc.disable()
    import importlib.util
    import jinja2, torch, hydra
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf, open_dict
    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='run', overrides=['experiment=init_diffusion_omega025_native8', 'optimizer=default'])
    with open_dict(cfg):
        cfg.paths.output_dir = '/tmp/native_circular_matching_cpu_proof'
        if geometry != 'legacy':
            cfg.model.model_config.decoder.init_diffusion.heading_matching_geometry = geometry
    torch.manual_seed(4817)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    flow = model.encoder.init_decoder.G1
    record = dict(parameters=sum(p.numel() for p in flow.parameters()),
        all_initial_tensors=digest((k, v) for k, v in model.state_dict().items() if torch.is_tensor(v)),
        initial_CPU_RNG=digest([('rng', torch.get_rng_state())]),
        denoiser_options=model.encoder.init_decoder.denoiser_options,
        initial_parameters=digest(flow.named_parameters()))
    spec = importlib.util.spec_from_file_location('fixture', PROJECT / 'tests/test_init_diffusion_ego_partial.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    state, agent, features = fixture.InitDiffusionEgoPartialTest().inputs()
    state = torch.cat((state[:, :6], state[:, 6:8].norm(dim=-1, keepdim=True)), -1)
    agent['expert_input'] = state.clone()
    torch.manual_seed(142015)
    features['pt_token'] = flow.model.lane_embed(torch.randn(3, 128)).detach()
    torch.manual_seed(142016)
    noise = flow._sample_noise(state, agent)
    record['source_noise'] = digest([('noise', noise)])
    torch.manual_seed(142017)
    result = flow._supervised_loss(state, agent, features)
    loss = result[0].mean() + result[1]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [(name, p.grad) for name, p in flow.named_parameters()]
    assert all(torch.isfinite(g).all() for _, g in gradients if g is not None)
    record.update(losses=digest((str(i), t) for i, t in enumerate(result) if torch.is_tensor(t)),
        gradients=digest(gradients), final_CPU_RNG=digest([('rng', torch.get_rng_state())]),
        full_backward_finite=True, cuda_initialized=torch.cuda.is_initialized())
    decoder = model.encoder.init_decoder
    if geometry == 'legacy' and root == PROJECT:
        checkpoint = torch.load(BASE / 'full_checkpoint_snapshots/omega025_e80_20261015/model.ckpt',
                                map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert decoder.ema.num_updates == 77120
        record['real_legacy80_checkpoint_strict_load'] = True
    if geometry != 'legacy':
        import copy
        own = decoder.get_extra_state()
        clone = copy.deepcopy(decoder)
        clone.load_state_dict(decoder.state_dict(), strict=True)
        missing = copy.deepcopy(own)
        missing['denoiser_options'].pop('heading_matching_geometry')
        try:
            clone.set_extra_state(missing)
        except ValueError:
            record['wrong_coupling_checkpoint_rejected'] = True
        else:
            raise AssertionError('Old source coupling must not silently resume as arc')
        record['new_coupling_checkpoint_strict_load'] = True
    output.write_text(json.dumps(record, indent=2) + '\n')
    print('CAPTURED', geometry, str(root), flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--implementation-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--geometry', default='legacy', choices=['legacy', 'arc'])
    parser.add_argument('--artifact-name', default='native_circular_matching_cpu_v1')
    args = parser.parse_args()
    if args.capture:
        capture(args.implementation_root, args.output, args.geometry)
        return
    global ART
    ART = BASE / args.artifact_name
    ART.mkdir(exist_ok=False)
    original = Path(json.loads((BASE / 'native_prefix8_pair_v1/protocol.json').read_text())['source_snapshot'])
    cases = [('reference', original, 'legacy'), ('legacy', PROJECT, 'legacy'), ('arc', PROJECT, 'arc')]
    for label, root, geometry in cases:
        command = [sys.executable, str(Path(__file__)), '--capture', '--implementation-root', str(root),
                   '--output', str(ART / (label + '.json')), '--geometry', geometry]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        (ART / (label + '.stdout')).write_text(result.stdout)
        (ART / (label + '.stderr')).write_text(result.stderr)
        assert result.returncode == 0, result.stderr[-1800:]
        print(result.stdout, end='', flush=True)
    records = {name: json.loads((ART / (name + '.json')).read_text()) for name, _, _ in cases}
    keys = ('parameters', 'all_initial_tensors', 'initial_CPU_RNG', 'denoiser_options',
            'initial_parameters', 'source_noise', 'losses', 'gradients', 'final_CPU_RNG')
    assert all(records['reference'][k] == records['legacy'][k] for k in keys), 'Legacy numerical behavior changed'
    assert all(records['legacy'][k] == records['arc'][k] for k in keys[:3] + ('initial_parameters',))
    assert records['arc']['wrong_coupling_checkpoint_rejected']
    files = [Path(__file__), PROJECT / 'configs/model/smart.yaml',
             PROJECT / 'configs/experiment/init_diffusion_omega025_arc_native8.yaml',
             PROJECT / 'src/smart/diffusion/initial_diffusion.py', PROJECT / 'src/smart/diffusion/scale_flow.py',
             PROJECT / 'src/smart/diffusion/diffusion_utils.py', PROJECT / 'src/smart/model/smart.py']
    summary = dict(status='NATIVE_CIRCULAR_MATCHING_CPU_PROOF_PASS', parameters=records['arc']['parameters'],
        legacy_frozen_initial_tensors_RNG_noise_losses_gradients_exact=True,
        legacy80_real_checkpoint_strict_load=True, arc_initial_tensors_RNG_unchanged=True,
        arc_full_backward_finite=True, wrong_coupling_checkpoint_rejected=True,
        per_agent_bridge_targets_deliberately_changed=True, quality_unproven=True,
        source_hashes={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (BASE / 'native_circular_matching_cpu_proof_20261015.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
