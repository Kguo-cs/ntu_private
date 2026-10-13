"""TRAIN-only paired endpoint coupling diagnostic; no generator forward/fit."""
import os
import gc
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
gc.disable()
gc.collect = lambda *unused, **kwargs: 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hydra
from omegaconf import OmegaConf, open_dict
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from tools.run_init_diffusion_657_iteration import read, save, now, sha
from src.smart.diffusion.component_loss import heading_angle_residual

ART = ROOT / 'output/reports/init_diffusion_evening_20261013/matching_probe'
OPTIONS = {'baseline': None, 'heading4_speed4': {'position': 1., 'heading': 4., 'length': 1., 'width': 1., 'velocity': 4.},
           'heading4_speed4_length2': {'position': 1., 'heading': 4., 'length': 2., 'width': 1., 'velocity': 4.}}


def main():
    torch.set_num_threads(1)
    rows = read(ROOT / 'output/reports/init_diffusion_evening_20261013/final_long/rebased/observations.json')
    row = next(r for r in rows if r['variant'] == 'freshK8' and r['budget'] == 8192)
    config = Path(row['checkpoint']).parent / 'config.yaml'
    cfg = OmegaConf.load(config)
    with open_dict(cfg):
        cfg.model.model_config.initial_map_checkpoint = None
        cfg.data.train_batch_size = 32
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
    torch.manual_seed(121013)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.eval()
    decoder = model.encoder.init_decoder
    flow = decoder.G1
    dm = hydra.utils.instantiate(cfg.data, _recursive_=False)
    dm.setup('fit')
    loader = iter(dm.train_dataloader())
    records = {name: [] for name in OPTIONS}
    count = 0
    with torch.no_grad():
        for batch_index in range(4):
            _, agent = model.token_processor(next(loader))
            decoder._prepare_ego_context(agent)
            clean, _ = flow.model.get_input(agent)
            movable = ~agent['ego_mask'].bool()
            for noise_index in range(4):
                anchor = None
                for name, weights in OPTIONS.items():
                    flow.matching_weights = weights
                    torch.manual_seed(101300 + batch_index * 4 + noise_index)
                    noise = flow._sample_noise(clean, agent)
                    if anchor is None:
                        anchor = noise.clone()
                    delta = (noise-clean) / flow.model.normal_scale
                    angle = heading_angle_residual(noise[:, 2:4], clean[:, 2:4]).abs()
                    item = dict(agents=int(movable.sum()), changed_fraction=float((noise[movable] != anchor[movable]).any(-1).float().mean()),
                                heading_abs_radian=float(angle[movable].mean()),
                                heading_over90_fraction=float((angle[movable] > torch.pi/2).float().mean()))
                    for field, columns in [('position', slice(0, 2)), ('length', slice(4, 5)), ('width', slice(5, 6)), ('speed', slice(6, 7))]:
                        item[field+'_normalized_mse'] = float(delta[movable, columns].square().mean())
                    records[name].append(item)
            count += agent['num_graphs']
    import statistics
    means = {name: {key: statistics.fmean(r[key] for r in rr) for key in rr[0]} for name, rr in records.items()}
    ratios = {name: {key: value/means['baseline'][key] if means['baseline'][key] else None for key, value in mean.items()}
              for name, mean in means.items()}
    save(ART / 'results.json', dict(created_utc=now(), config_sha256=sha(config), options=OPTIONS,
                                  TRAIN_scenes=count, noise_draws_per_batch=4, means=means, ratios=ratios, records=records,
                                  interpretation='Endpoint path lengths only, not learned JSD or a validation result. Same source draws, within-scene/type matching; ego excluded.'))
    print(means, flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
