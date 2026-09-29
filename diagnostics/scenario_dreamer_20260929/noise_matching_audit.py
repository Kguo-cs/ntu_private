#!/usr/bin/env python3
"""Read-only CPU audit of conditioned-ego Hungarian matching.

Run with the project's Python environment:
  /home/ke/miniconda3/envs/sim/bin/python diagnostics/scenario_dreamer_20260929/noise_matching_audit.py

The default reproduces the original audit's os.scandir sampling order. The JSON
records every sampled filename; pass --sample-list <previous JSON> to reproduce
the same sample even if filesystem enumeration order changes.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dataset', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--trials-per-scene', type=int, default=8)
    parser.add_argument('--seed', type=int, default=817)
    parser.add_argument('--sample-list', type=Path, default=None)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('noise_matching_stats.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from src.smart.diffusion.diffusion_utils import get_closest_sum_idx_fast, get_diff_loss

    dataset = args.dataset or args.repo / 'src/waymo_data/full/training_map2_sd'
    checkpoint = args.checkpoint or args.repo / 'src/waymo_data/last.ckpt'
    if args.sample_list:
        previous = json.loads(args.sample_list.read_text())
        paths = [dataset / name for name in previous['sample_files']]
    else:
        paths = random.Random(args.seed).sample(
            [Path(e.path) for e in os.scandir(dataset) if e.name.endswith('.pt')],
            args.samples,
        )

    # These are trusted local project artifacts. No model training or inference
    # is run, and no cache/checkpoint is mutated.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved['state_dict']
    mean = state['encoder.init_decoder.G1.model.normal_mean'].clone()
    scale = state['encoder.init_decoder.G1.model.normal_scale'].clone()
    checkpoint_step = saved.get('global_step')
    del saved, state
    torch.manual_seed(args.seed)

    remap = trials = ego_only = non_ego_total = 0
    examples = []
    for path in paths:
        agent = torch.load(path, map_location='cpu', weights_only=False)['tokenized_agent']
        n = len(agent['initial_pos'])
        ego_only += int(n == 1)
        non_ego_total += n - 1
        # TokenProcessor's cached-data convention: the last agent is ego.
        pos = agent['initial_pos'] - agent['initial_pos'][-1]
        heading = agent['initial_heading'][-1]
        co, si = heading.cos(), heading.sin()
        pos = torch.stack([co * pos[:, 0] + si * pos[:, 1],
                           -si * pos[:, 0] + co * pos[:, 1]], -1)
        relative_heading = agent['initial_heading'] - heading
        clean = torch.cat([pos, relative_heading.cos()[:, None],
                           relative_heading.sin()[:, None], agent['shape'][:, :2],
                           agent['local_vel']], -1)
        meta = {'batch': torch.zeros(n, dtype=torch.long), 'type': agent['type']}
        for _ in range(args.trials_per_scene):
            # Exact source sampling/matching sequence in Flow._sample_noise.
            noise = torch.randn_like(clean) * scale + mean
            noise[-1] = clean[-1]
            index = get_closest_sum_idx_fast(noise, clean, meta, all_state=True)
            trials += 1
            if index[-1] != n - 1:
                remap += 1
                receiving_agent = int(torch.where(index == n - 1)[0].item())
                matched_noise = noise[index].clone()
                # Flow._fix_conditioned_agents then restores the ego.
                matched_noise[-1] = clean[-1]
                assert torch.equal(matched_noise[receiving_agent], matched_noise[-1])
                if len(examples) < 3:
                    examples.append({'file': path.name, 'n': n,
                                     'ego_got_noise_from': int(index[-1]),
                                     'non_ego_given_ego': receiving_agent})

    # The current supervised loss makes an ego-only batch a zero-gradient batch.
    clean = torch.zeros(1, 8)
    pred = torch.ones_like(clean, requires_grad=True)
    conditioned = torch.where(torch.tensor([[True]]), clean.detach(), pred)
    loss = get_diff_loss({'batch': torch.zeros(1, dtype=torch.long)}, conditioned,
                         clean, torch.zeros(1, 1), .05, x_pred=True)[0].mean()
    loss.backward()
    stats = {
        'seed': args.seed,
        'samples': len(paths),
        'trials_per_scene': args.trials_per_scene,
        'trials': trials,
        'ego_remap_trials': remap,
        'ego_remap_fraction': remap / trials,
        'ego_only_samples': ego_only,
        'ego_only_fraction': ego_only / len(paths),
        'non_ego_total': non_ego_total,
        'ego_only_loss': float(loss),
        'ego_only_max_gradient': float(pred.grad.abs().max()),
        'dataset': str(dataset),
        'checkpoint': str(checkpoint),
        'checkpoint_global_step': checkpoint_step,
        'normal_mean': mean.tolist(),
        'normal_scale': scale.tolist(),
        'examples': examples,
        'interpretation': 'An ego remap duplicates the conditioned ego source after '
                          'the post-matching ego overwrite. Inference does not '
                          'perform this matching. This audit proves a source '
                          'distribution mismatch, not its downstream metric effect.',
        'sample_files': [p.name for p in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'sample_files'}, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
