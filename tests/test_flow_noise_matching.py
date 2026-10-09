import unittest
from itertools import product
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from src.smart.diffusion.scale_flow import Flow
from src.smart.diffusion.denoiser import InitDenoiser


def states(x_coordinates):
    result = torch.zeros(len(x_coordinates), 8)
    result[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
    return result


class FlowNoiseMatchingTest(unittest.TestCase):
    def flow(self, *, fix_ego=True, use_ego_embedding=False, generate_type=False):
        model = SimpleNamespace(denormalize=lambda x: x, normal_scale=torch.ones(8))
        model._ego_role_mask = MethodType(InitDenoiser._ego_role_mask, model)
        flow = SimpleNamespace(
            model=model, fix_ego=fix_ego,
            use_ego_embedding=use_ego_embedding, generate_type=generate_type,
        )
        flow._conditioned_agent_mask = MethodType(Flow._conditioned_agent_mask, flow)
        return flow

    def sample(self, clean, noise, batch, agent_type, ego_mask, **options):
        flow = self.flow(**options)
        agent = {
            "batch": torch.tensor(batch, dtype=torch.long),
            "type": torch.tensor(agent_type, dtype=torch.long),
            "ego_mask": torch.tensor(ego_mask, dtype=torch.bool),
            "num_graphs": max(batch, default=-1) + 1,
        }
        original_clean = clean.clone()
        with patch(
            "src.smart.diffusion.scale_flow.torch.randn_like",
            return_value=noise.clone(),
        ):
            result = Flow._sample_noise(flow, clean, agent)
        torch.testing.assert_close(clean, original_clean)
        return result, agent

    def test_fixed_ego_cannot_be_assigned_to_another_vehicle(self):
        # Unconstrained assignment would send the zero-valued ego source to
        # the vehicle at x=20, then restoring ego would duplicate that source.
        clean = states([10, 20, 0])
        result, agent = self.sample(
            clean, states([-10, -20, 999]),
            batch=[0, 0, 0], agent_type=[0, 0, 0],
            ego_mask=[False, False, True],
        )
        torch.testing.assert_close(result, states([-20, -10, 0]))

        # The later conditioning step must preserve the non-ego source set.
        time = torch.full((3, 1), 0.5)
        Flow._fix_conditioned_agents(self.flow(), clean, result, time, agent)
        torch.testing.assert_close(result, states([-20, -10, 0]))
        torch.testing.assert_close(time, torch.tensor([[0.5], [0.5], [0.0]]))

    def test_ego_keeps_its_own_source_across_scenes_and_training_options(self):
        clean = states([10, 0, 20, 100, 110, 120])
        endpoint = states([20, 10, 0, 110, 120, 100])
        # Mark every full source row, so the test catches partial-field swaps.
        endpoint[:, 2:] = torch.arange(6)[:, None] + torch.arange(6)[None, :] / 10
        original_endpoint = endpoint.clone()
        ego_mask = torch.tensor([False, True, False, True, False, False])

        # If ego participates, the optimal assignment takes another row for
        # ego and gives ego's source to a non-ego target in both scenes.
        for embedding, fixed, generation in product((False, True), repeat=3):
            with self.subTest(use_ego_embedding=embedding, fix_ego=fixed,
                              generate_type=generation):
                result, agent = self.sample(
                    clean, endpoint, batch=[0, 0, 0, 1, 1, 1],
                    agent_type=[0, 0, 1, 1, 0, 1], ego_mask=ego_mask.tolist(),
                    use_ego_embedding=embedding, fix_ego=fixed,
                    generate_type=generation,
                )
                expected = endpoint.clone()
                if generation:
                    # Type generation matches all non-ego types within each
                    # scene; known types keep each singleton type group.
                    expected[[0, 2, 4, 5]] = endpoint[[2, 0, 5, 4]]
                if fixed:
                    expected[ego_mask] = clean[ego_mask]
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                torch.testing.assert_close(agent["ego_mask"], ego_mask)
                torch.testing.assert_close(endpoint, original_endpoint, atol=0, rtol=0)

                # Reserving ego's source must not condition its training time
                # unless fix_ego is also enabled.
                time = torch.full((len(clean), 1), 0.5)
                flow = self.flow(fix_ego=fixed, use_ego_embedding=embedding,
                                 generate_type=generation)
                Flow._fix_conditioned_agents(flow, clean, result, time, agent)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                expected_time = torch.full_like(time, 0.5)
                if fixed:
                    expected_time[ego_mask] = 0
                torch.testing.assert_close(time, expected_time, atol=0, rtol=0)

    def test_matching_preserves_scene_and_type_groups(self):
        result, _ = self.sample(
            states([10, 30, 20, 0, 10, 10, 30, 20, 0]),
            states([-10, 50, -20, 999, 50, -10, 20, -20, 999]),
            batch=[0, 0, 0, 0, 2, 2, 2, 2, 2],
            agent_type=[0, 1, 0, 0, 2, 0, 2, 0, 0],
            ego_mask=[False, False, False, True, False, False, False, False, True],
        )
        torch.testing.assert_close(
            result, states([-20, 50, -10, 0, 20, -20, 50, -10, 0])
        )

    def test_ego_need_not_be_last_and_single_non_ego_keeps_its_source(self):
        result, _ = self.sample(
            states([0, 10]), states([999, -20]),
            batch=[0, 0], agent_type=[0, 0], ego_mask=[True, False],
        )
        torch.testing.assert_close(result, states([0, -20]))

    def test_empty_and_ego_only_batches(self):
        for n in (0, 1, 2):
            with self.subTest(num_agents=n):
                clean = states(list(range(n)))
                result, _ = self.sample(
                    clean, states([999] * n), batch=list(range(n)),
                    agent_type=[0] * n, ego_mask=[True] * n,
                )
                torch.testing.assert_close(result, clean)


if __name__ == "__main__":
    unittest.main()
