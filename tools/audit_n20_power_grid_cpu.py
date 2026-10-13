"""Two-scene native CPU audit; never trains or starts a GPU evaluation queue."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gc
gc.disable()
gc.collect = lambda *args, **kwargs: 0
import copy
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.n20_power_grid_sampler import sample_with_power_grid, source_sha256, REVIEWED_FLOW_SHA256


def main():
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    import hydra
    from omegaconf import OmegaConf, open_dict
    from tools.scratch_component_loss_sweep import calibrate
    started = time.monotonic()
    torch.set_num_threads(1)
    require = lambda condition, message: None if condition else (_ for _ in ()).throw(AssertionError(message))
    artifact = ROOT / 'output/reports/init_diffusion_657_edges_20261013/N20_power_grid_cpu_audit.json'
    folder = ROOT / 'logs/control657_paired_20261012/train4817/control657'
    checkpoint_path, config_path = folder / 'last.ckpt', folder / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    cohort_path = ROOT / 'tmp/control657_paired_20261012/eval1024_files.json'
    cohort = json.loads(cohort_path.read_text())
    cohort = cohort.get('files', cohort) if isinstance(cohort, dict) else cohort
    selected = cohort[:2]
    manifest = Path('/tmp/N20_power_grid_cpu_validation2.pkl')
    manifest.write_bytes(pickle.dumps({'files': selected}))
    with open_dict(cfg):
        cfg.data.scenario_dreamer_eval_set = str(manifest)
        cfg.data.test_batch_size = 2
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        cfg.paths.output_dir = '/tmp/N20_power_grid_cpu'
        cfg.paths.log_dir = cfg.paths.output_dir
        cfg.model.model_config.sd_r_reference = None
        cfg.model.model_config.decoder.init_diffusion.sampling_steps = 20
    model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    calibration_path = ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'
    calibrate(model, json.loads(calibration_path.read_text()), verify=True)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup('test')
    batch = next(iter(dm.test_dataloader()))
    with torch.no_grad():
        tokenized_map, agent = model.token_processor(batch)
        map_feature = model.encoder._get_map_feature(tokenized_map, agent)
        model.encoder._prepare_initial_map_feature(tokenized_map, agent, map_feature)
    decoder, results, cases = model.encoder.init_decoder, [], []
    flow = decoder.G1
    before = {key: value.detach().clone() for key, value in model.state_dict().items() if torch.is_tensor(value)}
    loss_before, updates_before = copy.deepcopy(decoder.component_loss), decoder.ema.num_updates
    hooks_before = dict(flow.model._forward_pre_hooks)
    decoder._move_ema()
    with torch.no_grad(), decoder.ema.average_parameters(flow.parameters()):
        for kind, power, policy in (('legacy', 1., 'legacy_floor'), ('prototype', 1., 'legacy_floor'), ('prototype', .5, 'legacy_floor'), ('prototype', 2., 'legacy_floor'), ('prototype', 2., 'actual_time')):
            data = copy.deepcopy(agent)
            scene_pos, scene_heading, batch_index, graphs = decoder._prepare_ego_context(data)
            feature = decoder._initial_map_feature(data, scene_pos, scene_heading, graphs)
            torch.manual_seed(2817)
            audit = {'kind': kind, 'denominator_policy': policy}
            if kind == 'legacy':
                times = []
                def count(module, args):
                    times.append({'minimum': float(args[1].min()), 'maximum': float(args[1].max()),
                                  'input_rows': len(args[0])})
                handle = flow.model.register_forward_pre_hook(count)
                try:
                    latent = flow.sample(data, feature, steps=20)
                finally:
                    handle.remove()
                audit.update(power=power, actual_denoiser_forward_calls=len(times), model_call_times=times,
                             all_output_values_finite=bool(torch.isfinite(latent).all()))
            else:
                latent = sample_with_power_grid(flow, data, feature, power=power, denominator_policy=policy, audit=audit)
            outputs = flow.model.get_output(latent, data)
            audit.update(all_five_decoded_outputs_finite=all(bool(torch.isfinite(value).all()) for value in outputs),
                         heading_unit_norm_max_error=float((latent[:, 2:4].square().sum(-1) - 1).abs().max()),
                         fixed_ego_fields_exact=bool(torch.equal(
                             latent[flow._conditioned_state_mask(data, latent)],
                             flow._joint_clean(data['expert_input'], data)[flow._conditioned_state_mask(data, latent)])))
            require(audit['actual_denoiser_forward_calls'] == 20, 'NFE changed')
            require(audit['all_output_values_finite'] and audit['all_five_decoded_outputs_finite'], 'Nonfinite outputs')
            require(audit['fixed_ego_fields_exact'], 'Fixed ego conditions changed')
            results.append({'latent': latent.clone(), 'outputs': tuple(value.clone() for value in outputs),
                            'type': data['type'].clone(), 'gen_noise': data['gen_noise'].clone(),
                            'type_state': data['_init_diffusion_type_state'].clone(),
                            'type_source': data['_init_diffusion_type_source'].clone()})
            cases.append(audit)
    baseline, control = results[0], results[1]
    exact = {'latent': torch.equal(baseline['latent'], control['latent']),
             'decoded_five_outputs': [torch.equal(a, b) for a, b in zip(baseline['outputs'], control['outputs'])]}
    for field in ('type', 'gen_noise', 'type_state', 'type_source'):
        exact[field] = torch.equal(baseline[field], control[field])
    require(exact['latent'] and all(exact['decoded_five_outputs'])
            and all(exact[field] for field in ('type', 'gen_noise', 'type_state', 'type_source')),
            'Uniform prototype did not preserve legacy outputs exactly')
    unchanged = all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    require(unchanged and flow.t_eps == .05, 'Original checkpoint model changed')
    require(decoder.component_loss == loss_before and decoder.ema.num_updates == updates_before, 'Training objective/EMA changed')
    require(dict(flow.model._forward_pre_hooks) == hooks_before, 'Temporary native hooks leaked')
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    report = {'status': 'PASS', 'scope': 'CPU native SMART actual two VAL scenes; implementation audit only; no training, GPU queue, JSD or improvement evidence',
              'TRAIN_seed': 4817, 'GEN_seed': 2817, 'checkpoint': str(checkpoint_path),
              'checkpoint_sha256': source_sha256(checkpoint_path), 'source_config': str(config_path),
              'source_config_sha256': source_sha256(config_path), 'calibration_sha256': source_sha256(calibration_path),
              'selected_eval_files': selected, 'selected_eval_files_sha256': __import__('hashlib').sha256(('\n'.join(selected) + '\n').encode()).hexdigest(),
              'evaluation_batch_size': 2, 'precision': '32-true', 'sampling_steps': 20,
              'actual_float32_matmul_precision': torch.get_float32_matmul_precision(),
              'torch_version': str(torch.__version__),
              'generator_parameter_count': sum(p.numel() for p in flow.parameters()),
              'extra_trainable_parameters': 0, 'cuda_initialized': torch.cuda.is_initialized(),
              'EMA_updates_unchanged': updates_before, 'original_model_tensors_unchanged': unchanged,
              'objective_config_unchanged': decoder.component_loss == loss_before,
              'original_flow_t_eps_unchanged': flow.t_eps == .05,
              'original_forward_pre_hooks_unchanged': dict(flow.model._forward_pre_hooks) == hooks_before,
              'uniform_vs_legacy_exact_equality': exact, 'cases': cases,
              'nonuniform_vs_uniform_latent_max_abs_difference': {
                  label: float((results[i]['latent'] - control['latent']).abs().max())
                  for i, label in ((2, 'p05_legacy_floor'), (3, 'p2_legacy_floor'), (4, 'p2_actual_time'))},
              'p2_actual_time_vs_p2_legacy_floor_latent_max_abs_difference': float((results[4]['latent'] - results[3]['latent']).abs().max()),
              'source_sha256': {str(path.relative_to(ROOT)): source_sha256(path) for path in (
                  ROOT / 'tools/n20_power_grid_sampler.py', ROOT / 'tools/audit_n20_power_grid_cpu.py',
                  ROOT / 'tools/test_n20_power_grid_sampler.py', ROOT / 'src/smart/diffusion/scale_flow.py',
                  ROOT / 'src/smart/diffusion/initial_diffusion.py',
                  ROOT / 'src/smart/diffusion/denoiser.py',
                  ROOT / 'src/smart/tokens/token_processor.py',
                  ROOT / 'src/smart/modules/smart_decoder.py',
                  ROOT / 'src/smart/model/smart_gail.py',
                  ROOT / 'src/smart/datasets/scalable_dataset.py',
                  ROOT / 'src/smart/datamodules/scalable_datamodule.py',
                  ROOT / 'tools/scratch_component_loss_sweep.py',
                  ROOT / 'tools/runtime_cached_torch_loader_sweep.py',
                  ROOT / 'tools/torch_pickle_class_cache.py')},
              'production_flow_reviewed_sha256': REVIEWED_FLOW_SHA256,
              'evidence_limits': 'Two CPU scenes prove parity, actual NFE and finite/fixed-field behavior only. Power .5/2 quality needs separately registered paired native evaluation with the same checkpoint, TRAIN/GEN seeds, EMA, precision, ordered cohort and batch split. p2 legacy_floor is spacing-only and preserves damping; p2 actual_time is a separately labeled joint integration change removing physical/type floor damping below .05. Neither is t_eps-only testing.',
              'elapsed_seconds': time.monotonic() - started}
    artifact.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cuda_initialized', 'uniform_vs_legacy_exact_equality',
                                                'nonuniform_vs_uniform_latent_max_abs_difference', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
