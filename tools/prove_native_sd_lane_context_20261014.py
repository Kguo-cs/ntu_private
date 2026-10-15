"""CPU proof of pretrained lane parity, context independence and transforms."""
from pathlib import Path
import sys
import json
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.smart.diffusion.native_lane_context import FrozenSDLaneContext
from src.smart.scenario_dreamer.core.autoencoder import ScenarioDreamerEncoder
from src.smart.scenario_dreamer.decoder import _checkpoint, _autoencoder_weights
from src.smart.scenario_dreamer.data import rotate


def main():
    torch.set_num_threads(1)
    checkpoint = ROOT / 'src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_waymo/last.ckpt'
    torch.manual_seed(4817)
    state = torch.get_rng_state().clone()
    context = FrozenSDLaneContext(checkpoint)
    rng_equal = torch.equal(state, torch.get_rng_state())
    cfg, weights = _autoencoder_weights(_checkpoint(checkpoint))
    reference = ScenarioDreamerEncoder(cfg).eval()
    reference.load_state_dict({k.removeprefix('encoder.'): v for k, v in weights.items()
                               if k.startswith('encoder.')}, strict=True)
    x = torch.linspace(-3., 3., 20)
    lanes = torch.stack([torch.stack((x, x * .0 + y), -1) for y in [-2., 2., -3., 3.]])
    batch = torch.tensor([0, 0, 1, 1])
    edges = torch.tensor([[i for i in range(4) for j in range(4) if batch[i] == batch[j]],
                          [j for i in range(4) for j in range(4) if batch[i] == batch[j]]])
    labels = torch.where(edges[0] == edges[1], 5, 1)
    native = dict(lanes=lanes, batch=batch, edges=edges, types=labels,
                  lg_type=torch.tensor([0, 1]),
                  center=torch.tensor([[10., -5.], [1000000., -2000000.]], dtype=torch.float64),
                  angle=torch.tensor([torch.pi / 2, -.7], dtype=torch.float64))
    before = (native['lg_type'][batch] == 1) & (lanes[:, 9, 1] <= 0)
    keep = before[edges[0]] == before[edges[1]]
    lane_input = (lanes / 32).flatten(1)
    relations = F.one_hot(labels[keep], 6).float()
    a2a = torch.arange(2).repeat(2, 1)
    bipartite = torch.stack((torch.arange(4), 4 + batch))
    agent_batch = torch.arange(2)
    with torch.no_grad():
        first = reference(torch.randn(2, 10), lane_input, relations, a2a,
                          edges[:, keep], bipartite, bipartite, agent_batch)[1]
        second = reference(torch.randn(2, 10) * 100, lane_input, relations, a2a,
                           edges[:, keep], bipartite, bipartite, agent_batch)[1]
    own = context.encode(native)
    agent_independent = torch.equal(first, second)
    original_branch_equal = torch.equal(first, own)
    shifted = dict(native, lanes=lanes.clone())
    shifted['lanes'][3, :, 0] += 10
    partition_independent = torch.equal(context.encode(shifted)[2], own[2])
    context.train()
    frozen_eval = not context.training and not any(p.requires_grad for p in context.parameters())
    raw = context(native)
    ego_pos = native['center'].clone()
    ego_heading = torch.tensor([-.5, .4], dtype=torch.float64)
    local = context(native, scene_pos=ego_pos, scene_heading=ego_heading)
    expected = rotate(lanes.double(), -native['angle'][batch, None] - ego_heading[batch, None])
    transform_error = float((local['position'] - expected.reshape(-1, 2).float()).abs().max())
    output_shape = raw['pt_token'].shape == (80, 128) and local['batch'].shape == (80,)
    latent_preserved = torch.equal(local['pt_token'].reshape(4, 20, 128)[:, 0, :24], own)
    analytic_world_point = torch.equal(raw['position'][0], torch.tensor([8., -2.]))
    analytic_world_heading = bool(torch.allclose(raw['orientation'][0], torch.tensor(-torch.pi / 2)))
    results = dict(rng_preserved=rng_equal, full_reference_lane_branch_equal=original_branch_equal,
                   independent_of_agent_state=agent_independent,
                   independent_of_other_partition=partition_independent,
                   frozen_eval_preserved=frozen_eval, transform_max_error=transform_error,
                   output_shape_valid=output_shape, latent_preserved=latent_preserved,
                   analytic_world_point=analytic_world_point,
                   analytic_world_heading=analytic_world_heading,
                   frozen_parameters=sum(p.numel() for p in context.parameters()),
                   status='PASS' if all([rng_equal, original_branch_equal, agent_independent,
                                        partition_independent, frozen_eval, transform_error == 0.,
                                        output_shape, latent_preserved, analytic_world_point,
                                        analytic_world_heading]) else 'FAIL',
                   scope='Synthetic geometry, real released weights, CPU only; no generation-quality claim and no InitDiffusion integration yet.')
    path = ROOT / 'output/reports/init_diffusion_continuous_20261014/native_lane_prototype/proof.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))
    assert results['status'] == 'PASS', results


if __name__ == '__main__':
    main()
