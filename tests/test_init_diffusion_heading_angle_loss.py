"""Wrapped x0 heading loss keeps the existing state and sampling contracts."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.diffusion_utils import _heading_angle_mse, get_diff_loss, matching_loss
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_ego_partial as fixture_module


def pair(theta):
    return torch.stack((theta.cos(), theta.sin()), -1)


class InitDiffusionHeadingAngleLossTest(unittest.TestCase):
    fixtures = fixture_module.InitDiffusionEgoPartialTest

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def flow(self, **options):
        return Flow(self.fixtures.args(heading_x0_loss='angle_mse', **options), self.fixtures.processor(), False)

    def test_shortest_error_is_rotation_and_positive_scale_invariant(self):
        truth = torch.deg2rad(torch.tensor([179., -179., 45.], dtype=torch.float64))
        predicted = torch.deg2rad(torch.tensor([-179., 179., 75.], dtype=torch.float64))
        expected = torch.deg2rad(torch.tensor([2., -2., 30.], dtype=torch.float64)).square()
        torch.testing.assert_close(_heading_angle_mse(pair(predicted), pair(truth)), expected)
        torch.testing.assert_close(_heading_angle_mse(3*pair(predicted+1.7), .2*pair(truth+1.7)), expected)

    def test_antipodal_gradient_through_unit_projection_is_nonzero(self):
        theta = torch.deg2rad(torch.tensor([179.999, -179.999, 180., -180., 0.],
                                          dtype=torch.float64)).requires_grad_()
        loss = _heading_angle_mse(pair(theta), pair(torch.zeros_like(theta)))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        torch.testing.assert_close(theta.grad.abs(), 2*theta.detach().abs())
        self.assertGreater(theta.grad[0].abs().item(), 6.)
        self.assertEqual(theta.grad[-1].item(), 0.)

    def test_low_precision_and_zero_pairs_have_finite_backward(self):
        for dtype in (torch.float16, torch.bfloat16):
            prediction = torch.tensor([[-1., .01], [0., 0.]], dtype=dtype, requires_grad=True)
            target = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype)
            loss = _heading_angle_mse(prediction, target)
            self.assertEqual(loss.dtype, torch.float32)
            loss.sum().backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(prediction.grad[0].abs().sum().item(), 0.)

    def test_only_heading_changes_and_existing_time_weight_is_retained(self):
        clean, agent, _ = self.fixtures.inputs()
        predicted = clean + torch.tensor([.2, -.3, 0., 0., .1, -.1, .5, -.2])
        angles = torch.atan2(clean[:, 3], clean[:, 2]) + torch.tensor([3., -.8, .3])
        predicted[:, 2:4] = pair(angles)
        times = torch.tensor([[.1], [.5], [.9]])
        options = dict(scale=torch.full((1, 8), 3.), use_col=True, x_pred=True)
        default = get_diff_loss(agent, predicted, clean, times, .05, **options)
        legacy = get_diff_loss(agent, predicted, clean, times, .05,
                              heading_x0_loss='vector_mse', **options)
        angular = get_diff_loss(agent, predicted, clean, times, .05,
                               heading_x0_loss='angle_mse', **options)
        for actual, expected in zip(default, legacy):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for i in (1, 2, 4, 5):
            torch.testing.assert_close(angular[i], legacy[i], atol=0, rtol=0)
        torch.testing.assert_close(angular[0]-legacy[0], .1*times[:, 0].pow(-3)*(angular[3]-legacy[3]))

    def test_flow_masks_fixed_heading_and_keeps_speed_and_log_size_training(self):
        for velocity in ('vector', 'speed'):
            for size in ('linear', 'log'):
                for fixed_heading in (True, False):
                    with self.subTest(velocity=velocity, size=size, fixed_heading=fixed_heading):
                        flow = self.flow(input_dim=7 if velocity == 'speed' else 8,
                                         velocity_representation=velocity, size_representation=size,
                                         fix_ego=False, fix_ego_position=True, fix_ego_shape=True,
                                         fix_ego_heading=fixed_heading, fix_ego_velocity=False).train()
                        _, agent, feature = self.fixtures.inputs()
                        if size == 'log':
                            agent['expert_input'][0, 4] = float('nan')
                        clean, _ = flow.model.get_input(agent)
                        theta = (torch.atan2(clean[:, 3], clean[:, 2])
                                 + torch.tensor([3., .8, -.3])).requires_grad_()
                        motion = (clean[:, 6:] + .5).requires_grad_()
                        prediction = torch.cat((clean[:, :2], pair(theta), clean[:, 4:6], motion), -1)
                        time = torch.full((3, 1), .4)
                        with patch.object(flow, '_prepare_supervised_batch', return_value=(clean, time, clean)), \
                                patch.object(flow.model, 'forward', return_value=prediction):
                            losses = flow._supervised_loss(clean, agent, feature)
                        losses[0].mean().backward()
                        self.assertTrue(torch.isfinite(theta.grad).all())
                        self.assertTrue(torch.isfinite(motion.grad).all())
                        self.assertAlmostEqual(losses[3][1].item(), 0. if fixed_heading else .8**2, places=6)
                        if fixed_heading:
                            self.assertEqual(theta.grad[1].item(), 0.)
                        else:
                            self.assertGreater(theta.grad[1].abs().item(), 0.)
                        self.assertGreater(motion.grad[1].abs().sum().item(), 0.)

    def test_real_denoiser_backpropagates_angle_loss(self):
        flow = self.flow(fix_ego=False, use_ego_embedding=True).train()
        clean, agent, feature = self.fixtures.inputs()
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .7)):
            losses = flow._supervised_loss(clean, agent, feature)
        (losses[0].mean()+losses[1]).backward()
        gradients = [p.grad for p in flow.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(flow.model.to_out_m_delta.mlp[-1].weight.grad[2:4].abs().sum().item(), 0.)

    def test_legacy_parameters_and_seeded_sampling_are_identical(self):
        old = Flow(self.fixtures.args(fix_ego=False), self.fixtures.processor(), False).eval()
        new = self.flow(fix_ego=False).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        _, agent, feature = self.fixtures.inputs()
        torch.manual_seed(817)
        expected = old.sample(copy.deepcopy(agent), feature, steps=4)
        torch.manual_seed(817)
        actual = new.sample(copy.deepcopy(agent), feature, steps=4)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(list(dict(old.named_parameters())), list(dict(new.named_parameters())))
        with patch.object(InitDiffusion, '_make_args', side_effect=lambda: self.fixtures.args()):
            original = InitDiffusion(32, 2, 4, self.fixtures.processor(), False, use_ema=True)
            restored = InitDiffusion(32, 2, 4, self.fixtures.processor(), False, use_ema=True,
                                     heading_x0_loss='angle_mse')
        original.update_ema()
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(restored.heading_x0_loss, 'angle_mse')
        self.assertEqual(restored.G1.heading_x0_loss, 'angle_mse')
        self.assertEqual(restored.ema.num_updates, original.ema.num_updates)

    def test_configuration_and_incompatible_objective(self):
        with self.assertRaisesRegex(ValueError, 'heading_x0_loss must be'):
            Flow(self.fixtures.args(heading_x0_loss='invalid'), self.fixtures.processor(), False)
        with self.assertRaisesRegex(ValueError, 'requires heading_objective=x0'):
            self.flow(heading_objective='angular_velocity')
        clean, _, _ = self.fixtures.inputs()
        with self.assertRaisesRegex(ValueError, 'deterministic x0'):
            matching_loss(clean, torch.cat((clean, torch.zeros_like(clean)), -1),
                          heading_x0_loss='angle_mse')
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                self.assertEqual(config.model.model_config.decoder.init_diffusion.heading_x0_loss, 'vector_mse')
                config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                    'model.model_config.decoder.init_diffusion.heading_x0_loss=angle_mse'])
                options = config.model.model_config.decoder.init_diffusion
                self.assertEqual(options.heading_x0_loss, 'angle_mse')
                self.assertEqual(options.heading_objective, 'x0')


if __name__ == '__main__':
    unittest.main()
