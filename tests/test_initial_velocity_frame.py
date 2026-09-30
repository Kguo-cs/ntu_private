import math
import unittest

import torch
from torch import nn

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import cal_polygon_contour, rotate_to_global


def make_processor():
    processor = TokenProcessor.__new__(TokenProcessor)
    nn.Module.__init__(processor)
    processor.shift = 5
    processor.scenario_dreamer_init = False
    return processor


def make_agent():
    centers = torch.zeros(2, 5, 2)
    centers[1, :, 0] = torch.arange(1, 6.)
    angles = torch.zeros(2, 5)
    angles[1] = torch.arange(1, 6.) * (math.pi / 10)
    contours = cal_polygon_contour(centers, angles, torch.tensor([2., 4.]))
    return {
        'ego_mask': torch.tensor([False, False, True]),
        'sampled_pos': torch.tensor([
            [[5., 0.], [5., 0.]], [[0., 0.], [5., 0.]], [[20., 0.], [20., 0.]],
        ]),
        'sampled_heading': torch.tensor([
            [math.pi / 2, math.pi / 2], [.7, .7 + math.pi / 2], [0., 0.],
        ]),
        'shape': torch.tensor([[4., 2.]]).expand(3, -1).clone(),
        'sampled_idx': torch.tensor([[1, 0], [0, 1], [0, 0]]),
        'token_mask': torch.tensor([[True, True], [False, True], [True, True]]),
        'token_traj_all': contours[None].expand(3, -1, -1, -1, -1).clone(),
    }


class InitialVelocityFrameTest(unittest.TestCase):
    def setUp(self):
        self.processor = make_processor()

    def test_incoming_and_fallback_turning_tokens_use_different_frames(self):
        agent = make_agent()
        original_idx = agent['sampled_idx'].clone()
        self.processor.get_init(agent, None)
        # Incoming token is represented at its endpoint; outgoing fallback is
        # represented at its start. Both end contours move 5m and turn 90 deg.
        expected = torch.tensor([[0., -10.], [10., 0.], [0., 0.]])
        torch.testing.assert_close(agent['local_vel'], expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(agent['sampled_idx'], original_idx)
        world = rotate_to_global(agent['local_vel'], agent['initial_heading'])
        expected_world = torch.tensor([
            [10., 0.], [10 * math.cos(.7), 10 * math.sin(.7)], [0., 0.],
        ])
        torch.testing.assert_close(world, expected_world, atol=1e-5, rtol=0)

    def test_single_token_input_does_not_use_outgoing_frame(self):
        agent = make_agent()
        for key in ('sampled_idx', 'sampled_pos', 'sampled_heading', 'token_mask'):
            agent[key] = agent[key][:, :1]
        # An invalid lone token cannot serve as its own outgoing fallback.
        agent['sampled_idx'][1, 0] = 1
        self.processor.get_init(agent, None)
        torch.testing.assert_close(
            agent['local_vel'][:2], torch.tensor([[0., -10.], [0., -10.]]),
            atol=1e-5, rtol=0,
        )

    def test_invalid_next_token_is_not_used_as_fallback(self):
        agent = make_agent()
        agent['token_mask'][1, 1] = False
        self.processor.get_init(agent, None)
        # No reliable outgoing segment exists: this change does not introduce
        # a new label policy, and must not borrow the invalid turning token.
        torch.testing.assert_close(agent['local_vel'][1], torch.zeros(2))

    def test_missing_mask_preserves_incoming_token_velocity(self):
        agent = make_agent()
        del agent['token_mask']
        self.processor.get_init(agent, None)
        torch.testing.assert_close(
            agent['local_vel'], torch.tensor([[0., -10.], [0., 0.], [0., 0.]]),
            atol=1e-5, rtol=0,
        )

    def test_straight_token_has_same_velocity_in_either_frame(self):
        agent = make_agent()
        centers = torch.zeros(5, 2)
        centers[:, 0] = torch.arange(1, 6.)
        contour = cal_polygon_contour(centers, torch.zeros(5), torch.tensor([2., 4.]))
        agent['token_traj_all'][:, 1] = contour
        self.processor.get_init(agent, None)
        torch.testing.assert_close(agent['local_vel'][:2], torch.tensor([[10., 0.], [10., 0.]]))


if __name__ == '__main__':
    unittest.main()
