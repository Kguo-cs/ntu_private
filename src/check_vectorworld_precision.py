"""Compare VectorWorld sampling precision with identical counts, weights and noise.

This is a small numerical audit, not a reproduction of full-set paper metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.smart.vectorworld.checkpoints import load_checkpoint, read_generator
from src.smart.vectorworld.decoder import _build_model
from src.smart.vectorworld.core import AutoEncoder, LDM, FlowLDM, MeanFlowLDM
from src.smart.vectorworld.core.utils.data_helpers import unnormalize_latents
from src.smart.vectorworld.data import build_generation_graph
from src.smart.scenario_dreamer.core.data_helpers import unnormalize_scene
from src.smart.scenario_dreamer.generation import SceneCountPrior, DEFAULT_COUNT_PRIOR
from src.smart.metrics.generated_map import split_generated_maps
from src.smart.metrics.metric_core import DistributionAccumulator, finalize_metrics
from src.smart.metrics.official_backend import load_official_backend

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'src/waymo_data/vectorworld/checkpoints/flow.ckpt')
    parser.add_argument('--scenes', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--precisions', nargs='+', choices=('medium', 'high', 'highest'),
                        default=['medium', 'highest'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/vectorworld_precision_audit.json')
    parser.add_argument('--minimum-free-gib', type=float, default=4.5)
    args = parser.parse_args()
    if args.scenes < 1 or args.batch_size < 1 or len(set(args.precisions)) < 2:
        parser.error('Use positive scenes/batch-size and at least two different precisions.')
    if args.minimum_free_gib < 0:
        parser.error('minimum-free-gib must be nonnegative.')
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if free / 2**30 < args.minimum_free_gib:
            raise SystemExit(f'Only {free / 2**30:.2f} GiB GPU memory free; '
                             f'need at least {args.minimum_free_gib:.2f}. Run when the active job finishes.')
    torch.set_num_threads(2)
    checkpoint = load_checkpoint(args.checkpoint)
    cfg, ae_cfg, weights, ae_weights, ema = read_generator(checkpoint)
    kind = str(cfg.model.ldm_type).lower()
    classes = {'flow': FlowLDM, 'diffusion': LDM, 'meanflow': MeanFlowLDM, 'mf': MeanFlowLDM}
    generator = _build_model(classes[kind], cfg, weights).eval()
    if ema is None:
        raise ValueError('Checkpoint has no EMA; audit expects the published EMA evaluation path.')
    params = list(generator.parameters())
    if len(params) != len(ema['shadow_params']):
        raise ValueError('EMA parameter count differs from generator.')
    # Use EMA directly, with no second GPU copy for context-manager restoration.
    for parameter, shadow in zip(params, ema['shadow_params']):
        if parameter.shape != shadow.shape or parameter.dtype != shadow.dtype:
            raise ValueError('EMA parameter shape/dtype differs from generator.')
        parameter.data = shadow
    generator.to(device)
    autoencoder = _build_model(AutoEncoder, ae_cfg.model, ae_weights).eval().to(device)
    updates = int(ema['num_updates'])
    del checkpoint, weights, ae_weights, ema, params
    prior = SceneCountPrior(DEFAULT_COUNT_PRIOR, max_num_agents=int(cfg.dataset.max_num_agents),
                            max_num_lanes=int(cfg.dataset.max_num_lanes), seed=args.seed)
    counts = prior.sample(args.scenes)
    official = load_official_backend()
    accumulators = {precision: DistributionAccumulator() for precision in args.precisions}
    differences = {precision: {'position_m': [], 'heading_rad': [], 'speed_mps': [],
                               'length_m': [], 'width_m': [], 'lane_point_m': [],
                               'agent_type_changes': 0, 'lane_edge_changes': 0}
                   for precision in args.precisions[1:]}
    backend_settings = {}
    s = cfg.dataset
    keys = ('fov', 'min_speed', 'max_speed', 'min_length', 'max_length', 'min_width',
            'max_width', 'min_lane_x', 'max_lane_x', 'min_lane_y', 'max_lane_y')

    @torch.inference_mode()
    def sample(data, precision, noise_seed):
        torch.set_float32_matmul_precision(precision)
        backend_settings[precision] = {'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                                       'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
                                       'autocast_enabled': torch.is_autocast_enabled()}
        # Reset for every precision: the sampler receives identical initial noise.
        torch.manual_seed(noise_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(noise_seed)
        a, l = generator(data, mode='initial_scene')
        a, l = unnormalize_latents(a, l, s.agent_latents_mean, s.agent_latents_std,
                                  s.lane_latents_mean, s.lane_latents_std)
        states, lanes, types, _, edges = autoencoder.forward_decoder_with_motion(a, l, data)
        states, lanes = unnormalize_scene(states, lanes, **{key: s[key] for key in keys})
        maps = split_generated_maps({'coordinate_frame': 'sd_local', 'road_points': lanes,
                'road_connection_types': edges, 'edge_index_lane_to_lane': data['lane', 'to', 'lane'].edge_index,
                'batch': data['lane'].batch, 'lg_type': data.lg_type}, data.batch_size)
        for index, road in enumerate(maps):
            mask = data['agent'].batch == index
            payload = dict(road, agent_states=states[mask, :7].cpu().numpy(),
                           agent_types=np.eye(3)[types[mask].cpu().numpy()])
            accumulators[precision].update(official.convert_data_to_unified_format(payload, 'waymo'),
                                          official, collision=True)
        return states[:, :7].cpu(), lanes.cpu(), types.cpu(), edges.argmax(-1).cpu()

    for batch_index, start in enumerate(range(0, args.scenes, args.batch_size)):
        data, *_ = build_generation_graph(counts[start:start + args.batch_size],
            agent_latent_dim=int(cfg.model.agent_latent_dim), lane_latent_dim=int(cfg.model.lane_latent_dim),
            device=device, dtype=torch.float32)
        baseline = sample(data, args.precisions[0], args.seed + batch_index)
        for precision in args.precisions[1:]:
            states, lanes, types, edges = sample(data, precision, args.seed + batch_index)
            before, roads, before_types, before_edges = baseline
            diff = differences[precision]
            for label, values in [('position_m', (states[:, :2] - before[:, :2]).norm(dim=-1)),
                                  ('lane_point_m', (lanes - roads).norm(dim=-1).flatten()),
                                  ('speed_mps', (states[:, 2] - before[:, 2]).abs()),
                                  ('length_m', (states[:, 5] - before[:, 5]).abs()),
                                  ('width_m', (states[:, 6] - before[:, 6]).abs())]:
                diff[label].extend(values.tolist())
            angle = torch.atan2(states[:, 4], states[:, 3]) - torch.atan2(before[:, 4], before[:, 3])
            diff['heading_rad'].extend(torch.atan2(angle.sin(), angle.cos()).abs().tolist())
            diff['agent_type_changes'] += int((types != before_types).sum())
            diff['lane_edge_changes'] += int((edges != before_edges).sum())
        print(f'Compared {min(start + args.batch_size, args.scenes)}/{args.scenes} scenes', flush=True)
    summary = {}
    for precision, diff in differences.items():
        summary[precision] = {key: {'mean': float(np.mean(value)), 'max': float(np.max(value)),
                                    'p95': float(np.percentile(value, 95))}
                              if isinstance(value, list) else value for key, value in diff.items()}
    distribution_shift = {}
    for precision in args.precisions[1:]:
        current, reference = accumulators[precision], accumulators[args.precisions[0]]
        if (current.totals == 0).any() or (reference.totals == 0).any():
            distribution_shift[precision] = {
                'undefined': 'Sample has empty feature distributions; increase --scenes.',
                'current_feature_counts': current.totals.tolist(),
                'reference_feature_counts': reference.totals.tolist()}
        else:
            metrics = finalize_metrics(current, reference)
            metrics.pop('collision_rate')  # Distribution shift is not a GT performance metric.
            distribution_shift[precision] = metrics
    report = {'scope': 'paired numerical audit; distribution JSD compares precisions, not generated vs GT',
        'checkpoint': str(args.checkpoint.resolve()), 'ldm_type': kind, 'use_ema': True,
        'ema_updates': updates, 'torch': torch.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'scenes': args.scenes, 'batch_size': args.batch_size, 'seed': args.seed,
        'counts': np.asarray(counts).tolist(), 'sampling_steps': int(getattr(generator, 'n_steps',
            getattr(generator, 'num_steps_eval', cfg.model.n_diffusion_timesteps))),
        'guidance_scale': float(cfg.train.guidance_scale),
        'baseline_precision': args.precisions[0], 'backend_settings': backend_settings,
        'physical_output_difference': summary, 'distribution_shift_between_precisions': distribution_shift,
        'sample_statistics': {precision: {'vehicles': a.num_vehicles,
            'onroad_vehicle_pct': a.totals[1] / a.num_vehicles * 100 if a.num_vehicles else None,
            'collision_pct': a.num_colliding / a.num_vehicles * 100 if a.num_vehicles else None}
            for precision, a in accumulators.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
