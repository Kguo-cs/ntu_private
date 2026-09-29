import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.smart.diffusion.scale_flow import Flow


def states(x_coordinates):
    result = torch.zeros(len(x_coordinates), 8)
    result[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
    return result


class FlowNoiseMatchingTest(unittest.TestCase):
    def sample(self, clean, noise, batch, agent_type, ego_mask):
        flow = SimpleNamespace(model=SimpleNamespace(denormalize=lambda x: x))
        agent = {
            "batch": torch.tensor(batch, dtype=torch.long),
            "type": torch.tensor(agent_type, dtype=torch.long),
            "ego_mask": torch.tensor(ego_mask, dtype=torch.bool),
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
        Flow._fix_conditioned_agents(clean, result, time, agent)
        torch.testing.assert_close(result, states([-20, -10, 0]))
        torch.testing.assert_close(time, torch.tensor([[0.5], [0.5], [0.0]]))

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
