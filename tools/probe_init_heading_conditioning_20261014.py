"""CPU TRAIN-only audit of the raw direction head's angular conditioning.

No training, CUDA, or generation-metric claims. The production loss is called
unchanged; its inputs and the pre-atan2 output are observed through local hooks.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gc
gc.disable()
gc.collect = lambda *args, **kwargs: 0
import hashlib
from pathlib import Path
import pickle
import random
import sys
import types
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now, NORMALIZER
from src.smart.diffusion.component_loss import heading_angle_residual

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014/heading_conditioning_probe'
USE_EMA = '--ema' in sys.argv
if USE_EMA:
    ART = ART / 'ema'
RUNS = ROOT / 'logs/init_diffusion_overnight_20261014'
CHECKPOINTS = {
    'shared2048': RUNS / 'roadK8_T4817_4096_N4096/roadK8/step_2048.ckpt',
    'continuous4096': RUNS / 'roadK8_T4817_4096_N4096/roadK8/step_4096.ckpt',
    'resumed4096': RUNS / 'recovery/roadK8_T4817_4096_N4096_attempt1/roadK8/step_4096.ckpt',
}


def quantiles(values):
    values = torch.cat(values).double()
    assert len(values) and torch.isfinite(values).all()
    return dict(count=len(values), mean=float(values.mean()), **{
        name: float(torch.quantile(values, q)) for name, q in
        [('min', 0.), ('p01', .01), ('p10', .1), ('p50', .5),
         ('p90', .9), ('p99', .99), ('max', 1.)]})


def main():
    torch.set_num_threads(1)
    ART.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'logs/component_loss_scratch/calibration/calibration_files.pkl'
    with source.open('rb') as handle:
        files = pickle.load(handle)
    files = (files['files'] if isinstance(files, dict) else files)[:128]
    assert len(files) == 128
    manifest = ART / 'train128.pkl'
    with manifest.open('wb') as handle:
        pickle.dump({'files': files}, handle)
    results = {}
    input_hashes = []
    for name, path in CHECKPOINTS.items():
        cfg = OmegaConf.load(path.parent / 'config.yaml')
        with open_dict(cfg):
            cfg.data.scenario_dreamer_train_sample_list = str(manifest)
            cfg.data.train_batch_size = 32
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.data.persistent_workers = False
            cfg.data.shuffle = False
        random.seed(241014); np.random.seed(241014); torch.manual_seed(241014)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu()
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        flow = model.encoder.init_decoder.G1
        if USE_EMA:
            decoder = model.encoder.init_decoder
            assert decoder.ema.num_updates == ckpt['global_step']
            decoder.ema.copy_to(flow.parameters())
        assert flow.heading_objective == 'x0'
        assert flow.component_loss.config['heading_space'] == 'angle'
        assert flow.component_loss.config['components']['heading']['kind'] == 'mse'
        stats = read(NORMALIZER)
        for field in ('normal_mean', 'normal_scale'):
            torch.testing.assert_close(getattr(flow.model, field),
                torch.tensor(stats[field], dtype=torch.float32), rtol=0, atol=0)
        model.train()
        assert not model.encoder.map_encoder.training
        dm = hydra.utils.instantiate(cfg.data); dm.setup('fit')
        original_output, original_loss = flow.model.output_transform, flow._component_supervised_loss
        observed = {key: [] for key in ('raw_radius', 'angle_jacobian_norm',
            'heading_error_rad', 'weighted_angle_loss_output_grad_norm')}
        holder = {}; digest = hashlib.sha256(); total_losses = []

        def output_hook(instance, raw, pos, theta):
            holder['radius'] = raw[:, 2:4].detach().norm(dim=-1)
            return original_output(raw, pos, theta)

        def loss_hook(instance, x, x0, noise, time, fixed, agent, prediction=None):
            active = (~fixed[:, 2:4]).all(-1) & (time[:, 0] > 0) & (time[:, 0] < 1)
            radius = holder.pop('radius')[active]
            assert (radius > 0).all()
            error = heading_angle_residual(x0[:, 2:4], x[:, 2:4])[active].abs()
            options = flow.component_loss.config['components']['heading']
            assert options['scale'] == 'none'
            weight = flow.component_loss.time_weight('heading', time[:, 0], flow.t_eps)[active]
            observed['raw_radius'].append(radius)
            observed['angle_jacobian_norm'].append(1 / radius)
            observed['heading_error_rad'].append(error)
            # Per-agent loss gradient before mean-over-agents and network chain.
            observed['weighted_angle_loss_output_grad_norm'].append(
                2 * error * weight * options['weight'] / radius)
            for value in (x, noise, time, fixed):
                digest.update(value.detach().contiguous().numpy().tobytes())
            return original_loss(x, x0, noise, time, fixed, agent, prediction)

        flow.model.output_transform = types.MethodType(output_hook, flow.model)
        flow._component_supervised_loss = types.MethodType(loss_hook, flow)
        with torch.no_grad():
            for batch_i, batch in enumerate(dm.train_dataloader()):
                seed = 241014 + batch_i
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                map_tokens, agent = model.token_processor(batch)
                result = model.encoder(map_tokens, agent)
                total_losses.append(float(result['initial_logit'][0]))
        radii = torch.cat(observed['raw_radius'])
        results[name] = dict(checkpoint=str(path), checkpoint_sha256=sha(path),
            step=ckpt['global_step'], raw_weights=not USE_EMA, EMA=USE_EMA, TRAIN_scenes=len(files),
            paired_input_sha256=digest.hexdigest(), mean_total_loss=float(np.mean(total_losses)),
            measurements={k: quantiles(v) for k, v in observed.items()},
            radius_fraction_below={str(v): float((radii < v).float().mean()) for v in (.01, .1, 1.)})
        input_hashes.append(digest.hexdigest())
        assert len(set(input_hashes)) == 1, 'Source/target/time must match across checkpoints'
        save(ART / 'results.json', dict(at_utc=now(), results=results,
            script_sha256=sha(__file__), manifest_sha256=sha(manifest),
            normalizer_sha256=sha(NORMALIZER), optimizer_updates=0, precision='CPU float32',
            scope='TRAIN conditional denoising; per-agent heading-output gradient excludes batch mean and other losses. Training-mode probe with the recorded raw/EMA weights, not generation. Does not establish the cause of trajectory drift.'))
        print('HEADING_CONDITIONING ' + name, flush=True)
        del model, flow, dm, ckpt
    assert not torch.cuda.is_initialized()
    os._exit(0)


if __name__ == '__main__':
    main()
