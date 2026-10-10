"""Raw type features share the physical state's deterministic objective."""

import unittest
from unittest.mock import patch

import torch

from src.smart.diffusion.diffusion_utils import (
    _parse_prediction,
    get_diff_loss,
    matching_loss,
)


class InitDiffusionFeatureLossTest(unittest.TestCase):
    @staticmethod
    def states(physical_dim, count=3):
        target = torch.arange(count*(physical_dim+3), dtype=torch.float64).reshape(count, -1)/19.
        target[:, 2:4] = torch.tensor([1., 0.], dtype=target.dtype)
        target[:, 4:6] = torch.tensor([4.5, 2.], dtype=target.dtype)
        target[:, physical_dim:] = torch.eye(3, dtype=target.dtype)[:count]
        prediction = target + torch.linspace(-.7, .9, physical_dim+3, dtype=target.dtype)
        agent = {'batch': torch.zeros(count, dtype=torch.long), 'type': torch.arange(count)%3}
        return prediction, target, agent

    def test_common_mse_uses_actual_feature_width_for_speed_and_vector(self):
        for physical_dim in (7, 8):
            with self.subTest(physical_dim=physical_dim):
                prediction, target, _ = self.states(physical_dim)
                total, position, heading, shape, velocity = matching_loss(
                    target, prediction, physical_state_dim=physical_dim,
                    use_l1=False, w_pos=.02, w_heading=12., w_shape=31., w_vel=9.,
                )
                error = (prediction-target).square()
                torch.testing.assert_close(total, error.mean(-1)*.02, atol=0, rtol=0)
                torch.testing.assert_close(position, error[:, :2].mean(-1))
                torch.testing.assert_close(heading, error[:, 2:4].mean(-1))
                torch.testing.assert_close(shape, error[:, 4:6].mean(-1))
                torch.testing.assert_close(velocity, error[:, 6:physical_dim].mean(-1))
                # A speed state has 10 real coordinates; there is no dummy vy.
                self.assertEqual(prediction.shape[-1], physical_dim+3)

    def test_type_prediction_is_raw_mse_without_softmax_or_ce(self):
        for physical_dim in (7, 8):
            prediction, target, _ = self.states(physical_dim)
            prediction[:, :physical_dim] = target[:, :physical_dim]
            prediction[:, physical_dim:] = torch.tensor([
                [-2., .5, 4.], [1.5, -3., .2], [7., -1., -4.],
            ], dtype=prediction.dtype)
            expected = (prediction[:, physical_dim:]-target[:, physical_dim:]).square().sum(-1)
            expected *= .02/(physical_dim+3)
            with (patch('torch.nn.functional.cross_entropy', side_effect=AssertionError('No type CE')),
                  patch('torch.nn.functional.softmax', side_effect=AssertionError('No type softmax'))):
                losses = matching_loss(target, prediction, physical_state_dim=physical_dim, w_pos=.02)
            torch.testing.assert_close(losses[0], expected)
            for physical_loss in losses[1:]:
                torch.testing.assert_close(physical_loss, torch.zeros_like(physical_loss), atol=0, rtol=0)

    def test_diff_loss_retains_common_time_weight_and_state_coefficient(self):
        for physical_dim in (7, 8):
            prediction, target, agent = self.states(physical_dim)
            time = torch.tensor([[.2], [.5], [.8]], dtype=prediction.dtype)
            loss = get_diff_loss(agent, prediction, target, time, .05,
                                 x_pred=True, physical_state_dim=physical_dim)
            expected = (prediction-target).square().mean(-1)*.02*time[:, 0].pow(-3)
            torch.testing.assert_close(loss[0], expected)
            self.assertEqual(loss[1].item(), 0.)
            # Representation scale has never normalized the raw deterministic
            # objective, and extra coordinates follow the same rule.
            scaled = get_diff_loss(agent, prediction, target, time, .05,
                                  x_pred=True, scale=torch.full_like(target, 13.),
                                  physical_state_dim=physical_dim)
            for actual, reference in zip(scaled, loss):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_time_zero_endpoints_and_weight_cap_apply_to_type_identically(self):
        prediction, target, agent = self.states(8)
        time = torch.tensor([[0.], [.01], [1.]], dtype=prediction.dtype)
        loss = get_diff_loss(agent, prediction, target, time, .05, x_pred=True,
                             max_loss_weight=4., physical_state_dim=8)[0]
        expected = (prediction-target).square().mean(-1)*.02*torch.tensor([0., 64., 0.])
        torch.testing.assert_close(loss, expected)

    def test_angular_reconstruction_selection_keeps_type_and_full_denominator(self):
        for physical_dim in (7, 8):
            prediction, target, _ = self.states(physical_dim)
            prediction.requires_grad_()
            dims = [0, 1, 4, 5]+list(range(6, physical_dim+3))
            loss = matching_loss(target, prediction, physical_state_dim=physical_dim,
                                 use_l1=False, w_pos=.02, reconstruction_dims=dims)
            expected = (prediction-target).square()[:, dims].sum(-1)*.02/(physical_dim+3)
            torch.testing.assert_close(loss[0], expected)
            loss[0].sum().backward()
            self.assertEqual(prediction.grad[:, 2:4].abs().sum().item(), 0.)
            self.assertGreater(prediction.grad[:, physical_dim:].abs().sum().item(), 0.)

    def test_full_feature_mask_ignores_nonfinite_values_and_blocks_gradients(self):
        for physical_dim in (7, 8):
            prediction, target, _ = self.states(physical_dim)
            mask = torch.ones_like(target, dtype=torch.bool)
            mask[0, physical_dim:] = False
            mask[1, [0, 3, 5, physical_dim+1]] = False
            prediction[~mask] = float('inf')
            target[~mask] = float('nan')
            prediction.requires_grad_()
            loss = matching_loss(target, prediction, physical_state_dim=physical_dim,
                                 reconstruction_mask=mask, use_l1=False, w_pos=.02)
            expected_error = torch.where(mask, prediction.detach(), 0.)-torch.where(mask, target, 0.)
            torch.testing.assert_close(loss[0], expected_error.square().mean(-1)*.02)
            self.assertTrue(all(torch.isfinite(component).all() for component in loss))
            loss[0].sum().backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertEqual(prediction.grad[~mask].abs().sum().item(), 0.)
            self.assertGreater(prediction.grad[mask].abs().sum().item(), 0.)

    def test_collision_decodes_only_physical_state_and_ignores_type_coordinates(self):
        for physical_dim in (7, 8):
            prediction, target, agent = self.states(physical_dim)
            target[:, :2] = torch.tensor([[0., 0.], [12., 0.], [25., 0.]])
            prediction[:, :2] = torch.tensor([[0., 0.], [.2, 0.], [25., 0.]])
            prediction[:, 4:6] = target[:, 4:6] = torch.tensor([4.5, 2.]).log()
            seen = []
            def decode(state):
                seen.append(state.shape[-1])
                self.assertEqual(state.shape[-1], physical_dim)
                state = state.clone()
                state[:, 4:6] = state[:, 4:6].exp()
                return state
            options = dict(x_pred=True, use_col=True, physical_state_dim=physical_dim,
                           state_to_physical=decode,
                           collision_valid_mask=torch.tensor([True, True, False]))
            time = torch.full((3, 1), .5, dtype=target.dtype)
            first = get_diff_loss(agent, prediction, target, time, .05, **options)
            changed = prediction.clone()
            changed[:, physical_dim:] = -900.
            second = get_diff_loss(agent, changed, target, time, .05, **options)
            self.assertEqual(seen, [physical_dim]*4)
            self.assertGreater(first[1].item(), 0.)
            torch.testing.assert_close(first[1], second[1], atol=0, rtol=0)
            self.assertGreater(second[0].sum().item(), first[0].sum().item())

    def test_explicit_physical_layout_is_validated(self):
        prediction, target, _ = self.states(7)
        for physical_dim in (6, 9):
            with self.assertRaisesRegex(ValueError, 'physical_state_dim'):
                matching_loss(target, prediction, physical_state_dim=physical_dim)
        for wrong_prediction in (prediction[:, :7], prediction[:, None, :], prediction[:, :6]):
            with self.assertRaisesRegex(ValueError, 'matching'):
                matching_loss(target, wrong_prediction, physical_state_dim=7)

    def test_legacy_deterministic_results_remain_identical(self):
        prediction, target, agent = self.states(8)
        prediction, target = prediction[:, :8], target[:, :8]
        time = torch.tensor([[.2], [.5], [.8]], dtype=target.dtype)
        legacy = get_diff_loss(agent, prediction, target, time, .05, x_pred=True, use_col=True)
        explicit = get_diff_loss(agent, prediction, target, time, .05, x_pred=True, use_col=True,
                                 physical_state_dim=8)
        for actual, reference in zip(explicit, legacy):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'Unsupported prediction dimension'):
            matching_loss(torch.cat((target, torch.zeros(3, 3)), -1),
                          torch.cat((prediction, torch.zeros(3, 3)), -1))

    def test_legacy_gaussian_and_mixture_parser_layouts_remain_available(self):
        gaussian = torch.arange(48.).reshape(3, 16)
        mode, mean, logits, std = _parse_prediction(gaussian)
        self.assertEqual(mode, 'gaussian')
        self.assertIsNone(logits)
        torch.testing.assert_close(mean, gaussian[:, :8], atol=0, rtol=0)
        torch.testing.assert_close(std, gaussian[:, 8:], atol=0, rtol=0)
        mixture = torch.arange(78.).reshape(3, 26)
        mode, mean, logits, std = _parse_prediction(mixture)
        self.assertEqual(mode, 'mixture')
        self.assertEqual(tuple(mean.shape), (3, 2, 8))
        torch.testing.assert_close(mean.flatten(1), mixture[:, :16], atol=0, rtol=0)
        torch.testing.assert_close(logits, mixture[:, 16:18], atol=0, rtol=0)
        torch.testing.assert_close(std, mixture[:, 18:], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
