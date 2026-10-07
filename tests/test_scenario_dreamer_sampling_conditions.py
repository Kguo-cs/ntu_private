"""Fixed conditions reach both Scenario Dreamer CFG branches unchanged."""
from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import HeteroData
from omegaconf import OmegaConf

from src.smart.scenario_dreamer.core.constants import BEFORE_PARTITION
from src.smart.scenario_dreamer.core.ldm import LDM


def _config():
    return OmegaConf.create(dict(
        model=dict(hidden_dim=32, agent_hidden_dim=32, num_heads=2,
                   agent_num_heads=2, num_factorized_dit_blocks=1,
                   num_l2l_blocks=1, lane_latent_dim=4, agent_latent_dim=4,
                   dropout=0., label_dropout=.1, n_diffusion_timesteps=4,
                   lane_sampling_temperature=.75, diffusion_clip=.5),
        dataset=dict(num_map_ids=2, max_num_agents=4, max_num_lanes=4),
        train=dict(loss_type="l2", lane_weight=10., guidance_scale=4.),
    ))


def _graph():
    data = HeteroData()
    for kind in ("agent", "lane"):
        data[kind].num_nodes = 3
        data[kind].batch = torch.tensor([0, 0, 1])
        # Latents deliberately exceed the clip bound in every row.
        data[kind].latents = torch.arange(1., 13.).reshape(3, 4)
        data[kind].x = data[kind].latents.clone()
    data['agent'].partition_mask = torch.tensor([BEFORE_PARTITION, 0, 0])
    data['lane'].partition_mask = torch.tensor([0, BEFORE_PARTITION, 0])
    data['agent'].mask = torch.tensor([False, True, False])
    data['lane'].mask = torch.tensor([True, False, True])
    data.batch_size = 2
    return data


class _NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, agents, lanes, data, agent_timestep, lane_timestep,
                unconditional=False):
        self.calls.append((agents.clone(), lanes.clone(), unconditional,
                           agent_timestep.clone(), lane_timestep.clone()))
        # Nonzero, context-dependent predictions exercise the actual DDPM math.
        branch = .1 if unconditional else .2
        return (.03 * agents + .01 * lanes.mean() + branch,
                .05 * lanes + branch)


def _sample(model, data, mode, *, legacy=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        method = _legacy_sampling_loop if legacy else LDM.p_sample_loop
        return method(
            model, data['agent'].x[:, None, :].shape,
            data['lane'].x[:, None, :].shape, data,
            device="cpu", mode=mode, return_diffusion_chain=True,
        )


@torch.no_grad()
def _legacy_sampling_loop(self, agent_shape, lane_shape, data, *,
                               device, mode, return_diffusion_chain):
    """Keep the original draws, clipping and condition restoration order.

    This bounded test reference does not use the new fixed-context helper.
    """
    def restore_partial(agents, lanes):
        if mode in ("train", "inpainting"):
            for kind, value in (("agent", agents), ("lane", lanes)):
                mask = (data[kind].partition_mask == BEFORE_PARTITION
                        if mode == "train" else data[kind].mask)
                value[mask] = data[kind].latents[mask].unsqueeze(1)
        return agents, lanes
    agents = torch.randn(agent_shape, device=device)
    lanes = (data['lane'].latents[:, None, :].to(device) if mode == "lane_conditioned"
             else torch.randn(lane_shape, device=device) * self.lane_sampling_temperature)
    agents, lanes = restore_partial(agents, lanes)
    chain = [(agents, lanes)]
    for step in reversed(range(self.n_timesteps)):
        timestep = torch.full((data.batch_size,), step, device=device, dtype=torch.long)
        agents, lanes = self.p_sample(
            agents, lanes, data, timestep[data['agent'].batch], timestep[data['lane'].batch]
        )
        agents = torch.clip(agents, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
        if mode == "lane_conditioned":
            lanes = data['lane'].latents[:, None, :].to(device)
        else:
            lanes = torch.clip(lanes, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
        agents, lanes = restore_partial(agents, lanes)
        chain.append((agents, lanes))
    return agents[:, 0], lanes[:, 0], chain


class ScenarioDreamerSamplingConditionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self):
        model = LDM(_config()).eval()
        model.model = _NoisePredictor()
        return model

    def assert_cfg_calls(self, model):
        self.assertEqual(len(model.model.calls), 2 * model.n_timesteps)
        self.assertEqual([entry[2] for entry in model.model.calls], [False, True] * model.n_timesteps)
        for conditional, unconditional in zip(model.model.calls[::2], model.model.calls[1::2]):
            for index in (0, 1, 3, 4):
                torch.testing.assert_close(conditional[index], unconditional[index], rtol=0, atol=0)

    def test_all_lanes_are_fixed_in_every_cfg_call_and_chain_state(self):
        model, data = self.model(), _graph()
        agents, lanes, chain = _sample(model, data, "lane_conditioned")
        self.assert_cfg_calls(model)
        self.assertEqual(len(chain), model.n_timesteps + 1)
        for _, lane_input, *_ in model.model.calls:
            torch.testing.assert_close(lane_input[:, 0], data['lane'].latents, rtol=0, atol=0)
        for _, lane_state in chain:
            torch.testing.assert_close(lane_state[:, 0], data['lane'].latents, rtol=0, atol=0)
        torch.testing.assert_close(lanes, data['lane'].latents, rtol=0, atol=0)
        self.assertLessEqual(float(agents.abs().max()), model.cfg_model.diffusion_clip)

    def test_partial_conditions_survive_every_cfg_call_and_clipping(self):
        for mode in ("train", "inpainting"):
            with self.subTest(mode=mode):
                model, data = self.model(), _graph()
                agents, lanes, chain = _sample(model, data, mode)
                self.assert_cfg_calls(model)
                self.assertEqual(len(chain), model.n_timesteps + 1)
                for kind, index, generated in (("agent", 0, agents), ("lane", 1, lanes)):
                    mask = (data[kind].partition_mask == BEFORE_PARTITION
                            if mode == "train" else data[kind].mask)
                    for call in model.model.calls:
                        torch.testing.assert_close(call[index][mask, 0], data[kind].latents[mask], rtol=0, atol=0)
                    for state in chain:
                        torch.testing.assert_close(state[index][mask, 0], data[kind].latents[mask], rtol=0, atol=0)
                    torch.testing.assert_close(generated[mask], data[kind].latents[mask], rtol=0, atol=0)
                    self.assertLessEqual(float(generated[~mask].abs().max()), model.cfg_model.diffusion_clip)

    def test_all_modes_remain_bitwise_equal_to_original_loop(self):
        for mode in ("lane_conditioned", "initial_scene", "train", "inpainting"):
            with self.subTest(mode=mode):
                model, data = self.model(), _graph()
                previous = _sample(model, data, mode, legacy=True)
                model.model.calls.clear()
                current = _sample(model, data, mode)
                self.assert_cfg_calls(model)
                for expected, actual in zip(previous[:2], current[:2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(previous[2]), len(current[2]))
                for expected_state, actual_state in zip(previous[2], current[2]):
                    for expected, actual in zip(expected_state, actual_state):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
