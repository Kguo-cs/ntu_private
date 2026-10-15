"""TRAIN-only comparison of heading costs; no model fit or generated-JSD claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gc
gc.disable()
gc.collect = lambda *args, **kwargs: 0
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import hydra
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now, NORMALIZER
from src.smart.diffusion import scale_flow
from src.smart.diffusion.component_loss import heading_angle_residual

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014/circular_matching_probe'
CONFIG = ROOT / 'logs/init_diffusion_overnight_20261014/roadK8_T4817_4096_N4096/roadK8/config.yaml'
CHECKPOINT=None
BATCHES=4


def heading_cost(a, b, kind, strength):
    delta = (a[:, None] - b[None, :] + np.pi) % (2*np.pi) - np.pi
    return strength * (delta**2 if kind == 'arc' else 2*(1-np.cos(delta)))


def main():
    torch.set_num_threads(1)
    cfg = OmegaConf.load(CONFIG)
    with open_dict(cfg):
        cfg.model.model_config.initial_map_checkpoint = None
        # Saved scratch configs use null here: the training harness explicitly
        # calibrates after construction. This standalone probe must do so too.
        cfg.model.model_config.decoder.init_diffusion.normalizer_path = str(NORMALIZER)
        cfg.data.train_batch_size = 32
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
    torch.manual_seed(142015)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if CHECKPOINT is not None:
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval(); decoder = model.encoder.init_decoder; flow = decoder.G1
    calibration = read(NORMALIZER)
    for name in ('normal_mean', 'normal_scale'):
        expected=(torch.tensor(calibration[name],dtype=torch.float32) if CHECKPOINT is None else
                  checkpoint['state_dict']['encoder.init_decoder.G1.model.'+name])
        torch.testing.assert_close(getattr(flow.model,name),expected,rtol=0,atol=0)
    dm = hydra.utils.instantiate(cfg.data, _recursive_=False); dm.setup('fit')
    loader = iter(dm.train_dataloader())
    std = flow.model.normal_scale[0].detach().numpy()
    mean = flow.model.normal_mean[0].detach().numpy()
    # Match mean heading cost of independent circular source / TRAIN headings.
    # The chosen transport subsequently creates dependence; this is a fixed
    # coefficient calibration, not an equality of optimal assignments.
    heading_weight=1. if flow.matching_weights is None else float(flow.matching_weights['heading'])
    expected_old = heading_weight*float(((.5 + std[2:4]**2 + mean[2:4]**2) / std[2:4]**2).sum())
    arc_strength = expected_old / (np.pi**2/3)
    chord_strength = expected_old / 2
    options = [('legacy', None), ('isotropic_chord', 'chord'), ('arc', 'arc')]
    records = {name: [] for name, _ in options}
    source_permutation_checks = 0
    rotation_error = 0.
    scene_count = 0
    for batch_i in range(BATCHES):
        _, agent = model.token_processor(next(loader)); decoder._prepare_ego_context(agent)
        scene_count += int(agent['num_graphs'])
        clean, _ = flow.model.get_input(agent)
        movable = ~agent['ego_mask'].bool()
        for draw in range(4):
            anchor = None
            for name, kind in options:
                def matcher(fake_state, real_state, tokenized_agent, all_state=False, use_all_type=False):
                    assert all_state and not use_all_type
                    fake = fake_state.detach().float().cpu().numpy()
                    real = real_state.detach().float().cpu().numpy()
                    batch = tokenized_agent['batch'].numpy(); labels = tokenized_agent['type'].numpy()
                    out = np.arange(len(fake))
                    fields = [0, 1, 4, 5, 6]
                    for group in sorted(set(zip(batch.tolist(), labels.tolist()))):
                        ids = np.flatnonzero((batch == group[0]) & (labels == group[1]))
                        aa, bb = real[ids], fake[ids]
                        aa_theta = np.arctan2(aa[:, 3]*std[3], aa[:, 2]*std[2])
                        bb_theta = np.arctan2(bb[:, 3]*std[3], bb[:, 2]*std[2])
                        c = ((aa[:, None, fields] - bb[None, :, fields])**2).sum(-1)
                        strength = arc_strength if kind == 'arc' else chord_strength
                        c += heading_cost(aa_theta, bb_theta, kind, strength)
                        row, col = linear_sum_assignment(c)
                        out[ids[row]] = ids[col]
                    return torch.from_numpy(out).to(fake_state.device)
                torch.manual_seed(141000 + batch_i*4 + draw)
                if kind is None:
                    noise = flow._sample_noise(clean, agent)
                    anchor = noise.clone()
                else:
                    with patch.object(scale_flow, 'get_closest_sum_idx_fast', matcher):
                        noise = flow._sample_noise(clean, agent)
                    # Whole source row multisets are unchanged in every scene/type.
                    for batch_id in agent['batch'].unique():
                        for label in agent['type'].unique():
                            select = movable & (agent['batch'] == batch_id) & (agent['type'] == label)
                            first = sorted(map(tuple, noise[select].tolist()))
                            second = sorted(map(tuple, anchor[select].tolist()))
                            assert first == second
                            source_permutation_checks += 1
                delta = (noise-clean) / flow.model.normal_scale
                angular = heading_angle_residual(noise[:, 2:4], clean[:, 2:4]).abs()
                item = dict(agents=int(movable.sum()), changed_fraction=float((noise[movable] != anchor[movable]).any(-1).float().mean()),
                            heading_abs=float(angular[movable].mean()), heading_mse=float(angular[movable].square().mean()),
                            heading_over90=float((angular[movable] > torch.pi/2).float().mean()))
                for key, columns in [('position', slice(0, 2)), ('length', slice(4, 5)), ('width', slice(5, 6)), ('speed', slice(6, 7))]:
                    item[key + '_normalized_mse'] = float(delta[movable, columns].square().mean())
                records[name].append(item)
    means = {name: {k: sum(r[k]*r['agents'] for r in rows)/sum(r['agents'] for r in rows)
                   for k in rows[0] if k != 'agents'} for name, rows in records.items()}
    ratios = {name: {k: v/means['legacy'][k] if means['legacy'][k] else None for k,v in values.items()} for name,values in means.items()}
    for kind, strength in [('arc', arc_strength), ('chord', chord_strength)]:
        angles = np.array([-.99*np.pi, -.5*np.pi, .1, .8*np.pi], dtype=np.float64)
        x, y = heading_cost(angles, angles[::-1], kind, strength), heading_cost(angles+.7, angles[::-1]+.7, kind, strength)
        rotation_error = max(rotation_error, float(np.abs(x-y).max()))
    assert rotation_error < 1e-12
    save(ART / 'results.json', dict(created_utc=now(),TRAIN_scenes=scene_count,draws_per_scene=4,
        config_sha256=sha(CONFIG),probe_sha256=sha(__file__),expected_legacy_heading_cost=expected_old,
        checkpoint=None if CHECKPOINT is None else str(CHECKPOINT),
        checkpoint_sha256=None if CHECKPOINT is None else sha(CHECKPOINT),
        matching_heading_weight=heading_weight,
        normalizer_sha256=sha(NORMALIZER), normal_mean=mean.tolist(), normal_scale=std.tolist(),
        arc_strength=arc_strength,chord_strength=chord_strength,source_permutation_checks=source_permutation_checks,
        rotational_invariance_max_error=rotation_error,means=means,ratios=ratios,records=records,
        caveat='Endpoint coupling diagnostic only. No learned generator/JSD result. Ego excluded and type groups preserved.'))
    print(means,flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=ART)
    parser.add_argument('--batches',type=int,default=4)
    options=parser.parse_args();CONFIG=options.config;CHECKPOINT=options.checkpoint;ART=options.output;BATCHES=options.batches
    assert BATCHES>0
    main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
