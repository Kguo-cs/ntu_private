"""Scalar-speed initialization learns magnitude and reconstructs physical velocity."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate, _noise_endpoint
from src.smart.diffusion.diffusion_utils import get_diff_loss
from src.smart.utils import wrap_angle


class InitDiffusionSpeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(817)

    @staticmethod
    def args(**options):
        values = dict(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                      num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                      sampling_steps=20, use_rl=False, heading_noise='circular',
                      heading_objective='angular_velocity', velocity_representation='speed')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2) / dt)
        values.update(options)
        return SimpleNamespace(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, mode='speed', **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False,
                                 heading_noise='circular', heading_objective='angular_velocity',
                                 velocity_representation=mode, **options)

    @staticmethod
    def inputs():
        vector = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                               [0., 0., 1., 0., 4.5, 2., 0., 0.],
                               [-20., 30., 0., 1., 4., 1.8, -2., 0.]])
        speed = torch.cat((vector[:, :6], vector[:, 6:8].norm(dim=-1, keepdim=True)), -1)
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 1, 2]),
                     num_graphs=1, ego_mask=torch.tensor([False, True, False]),
                     expert_input=speed.clone(), local_vel=vector[:, 6:8].clone(),
                     batch_ego_pos=torch.zeros(3, 2), batch_ego_heading=torch.zeros(3),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return vector, speed, agent, feature

    def test_get_input_norms_body_velocity_and_keeps_heading_in_ego_frame(self):
        flow = self.flow()
        vector, expected, agent, _ = self.inputs()
        del agent['expert_input']
        ego_heading = torch.full((3,), .2)
        theta = torch.tensor([.6, -.5, 2.])
        pos = torch.tensor([[4., 5.], [-1., 2.], [3., -4.]])
        agent.update(initial_pos=pos, initial_heading=theta, shape=vector[:, 4:6],
                     batch_ego_pos=torch.zeros(3, 2), batch_ego_heading=ego_heading)
        first, target = flow.model.get_input(agent)
        self.assertEqual(first.shape, (3, 7))
        torch.testing.assert_close(first, target)
        torch.testing.assert_close(first[:, 6], torch.tensor([5., 0., 2.]))
        local_theta = theta - ego_heading
        torch.testing.assert_close(first[:, 2:4], torch.stack((local_theta.cos(), local_theta.sin()), -1))
        torch.testing.assert_close(agent['local_vel'], vector[:, 6:8])
        self.assertEqual(flow.model.normal_mean.shape, (1, 7))
        self.assertEqual(flow.model.normal_scale.shape, (1, 7))

    def test_cached_vector_input_converts_once_without_reinitializing_normalizer(self):
        flow = self.flow()
        vector, expected, agent, _ = self.inputs()
        agent['expert_input'] = vector.clone()
        with torch.no_grad():
            flow.model.normal_mean.fill_(9.)
            flow.model.normal_scale.fill_(2.)
        actual, target = flow.model.get_input(agent)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(target, expected)
        agent['expert_input'] = actual
        repeated, _ = flow.model.get_input(agent)
        torch.testing.assert_close(repeated, expected)
        torch.testing.assert_close(flow.model.normal_mean, torch.full((1, 7), 9.))
        torch.testing.assert_close(flow.model.normal_scale, torch.full((1, 7), 2.))
        torch.testing.assert_close(agent['local_vel'], vector[:, 6:8])

    def test_real_forward_and_training_update_scalar_speed_and_angular_heads(self):
        flow = self.flow().train()
        _, clean, agent, feature = self.inputs()
        latent = clean.clone().requires_grad_()
        time = torch.tensor([[.4], [0.], [.4]])
        prediction = flow.model(latent, time, agent, feature)
        self.assertEqual(flow.model.m_delta_dim, 7)
        self.assertEqual(flow.model.output_dim, 7)
        self.assertEqual(prediction.shape, (3, 8))
        prediction[:, [0, 1, 4, 5, 6, 7]].square().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad[:, 6].abs().sum().item(), 0.)
        flow.zero_grad()
        before_speed = flow.model.to_out_m_delta.mlp[-1].weight[6].detach().clone()
        before_heading = [p.detach().clone() for p in flow.model.to_out_heading_velocity.parameters()]
        optimizer = torch.optim.Adam(flow.parameters(), lr=1.e-3)
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = flow._supervised_loss(clean, agent, feature)
        (losses[0].mean() + losses[1]).backward()
        gradients = [p.grad for p in flow.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(flow.model.to_out_m_delta.mlp[-1].weight.grad[6].abs().sum().item(), 0.)
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in flow.model.to_out_heading_velocity.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before_speed, flow.model.to_out_m_delta.mlp[-1].weight[6]))
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before_heading, flow.model.to_out_heading_velocity.parameters())))

    def test_scalar_loss_matches_magnitude_target_and_corrects_negative_prediction(self):
        flow = self.flow()
        _, clean, agent, feature = self.inputs()
        time = torch.tensor([[.5], [0.], [.5]])
        prediction = torch.cat((clean, torch.zeros(3, 1)), -1)
        prediction[:, 6] = torch.tensor([-1., 500., -3.])
        prediction.requires_grad_()
        with patch.object(flow, '_prepare_supervised_batch', return_value=(clean, time, clean)), \
                patch.object(flow.model, 'forward', return_value=prediction), \
                patch('src.smart.diffusion.scale_flow.get_diff_loss', wraps=get_diff_loss) as loss_spy:
            losses = flow._supervised_loss(clean, agent, feature)
        expected = torch.tensor([36., 0., 25.])
        torch.testing.assert_close(losses[5], expected)
        fake, real = loss_spy.call_args.args[1:3]
        self.assertEqual(fake.shape, (3, 8))
        self.assertEqual(real.shape, (3, 8))
        torch.testing.assert_close(fake[:, 7], torch.zeros(3))
        torch.testing.assert_close(real[:, 7], torch.zeros(3))
        # The speed target is a magnitude, including GT reverse/lateral motion.
        torch.testing.assert_close(real[:, 6], torch.tensor([5., 0., 2.]))
        self.assertEqual(loss_spy.call_args.kwargs['reconstruction_dims'], (0, 1, 4, 5, 6))
        losses[0].sum().backward()
        self.assertLess(prediction.grad[0, 6].item(), 0.)
        self.assertLess(prediction.grad[2, 6].item(), 0.)
        self.assertEqual(prediction.grad[1, 6].item(), 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_signed_gaussian_speed_source_is_preserved_during_interpolation(self):
        flow = self.flow()
        _, clean, agent, _ = self.inputs()
        eps = torch.zeros(3, 7)
        eps[:, 2] = 1.
        eps[:, 6] = torch.tensor([-3., 4., -2.])
        endpoint = _noise_endpoint(flow.model, eps, None, 'circular')
        torch.testing.assert_close(endpoint[:, 6], eps[:, 6])
        with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=eps), \
                patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .75)):
            noise, time, latent = flow._prepare_supervised_batch(clean, agent)
        non_ego = ~agent['ego_mask']
        torch.testing.assert_close(noise[non_ego, 6], eps[non_ego, 6])
        torch.testing.assert_close(latent[non_ego, 6],
                                   (.25 * clean[:, 6] + .75 * eps[:, 6])[non_ego])
        self.assertTrue((latent[non_ego, 6] < 0).any())
        torch.testing.assert_close(latent[agent['ego_mask']], clean[agent['ego_mask']])

    def test_angular_velocity_uses_index_seven_and_oracle_recovers_seven_dimensional_state(self):
        flow = self.flow().eval()
        _, clean, agent, feature = self.inputs()
        theta0 = torch.deg2rad(torch.tensor([170., 0., -170.]))
        theta1 = torch.deg2rad(torch.tensor([-170., 0., 170.]))
        clean[:, 2:4] = torch.stack((theta0.cos(), theta0.sin()), -1)
        noise = clean.clone()
        noise[:, 2:4] = torch.stack((theta1.cos(), theta1.sin()), -1)
        delta = wrap_angle(theta1 - theta0)
        time = torch.tensor([[.4], [0.], [.025]])
        latent = _circular_interpolate(clean, noise, time)
        prediction = torch.cat((clean, delta[:, None]), -1)
        velocity, reconstructed = flow._prediction_velocity(latent, time, prediction)
        tangent = torch.stack((-latent[:, 3], latent[:, 2]), -1)
        torch.testing.assert_close((velocity[:, 2:4] * tangent).sum(-1), delta, atol=2e-6, rtol=0)
        torch.testing.assert_close(reconstructed, clean, atol=2e-6, rtol=0)
        agent['expert_input'] = clean.clone()
        for steps in (1, 20):
            with self.subTest(steps=steps), \
                    patch('src.smart.diffusion.scale_flow.torch.randn', return_value=noise.clone()), \
                    patch.object(flow.model, 'forward', return_value=prediction):
                generated = flow.sample(agent, feature, steps=steps)
            self.assertEqual(generated.shape, (3, 7))
            torch.testing.assert_close(generated, clean, atol=2e-5, rtol=0)

    def test_heading_and_speed_reconstruct_velocity_once_and_preserve_original_ego_vector(self):
        flow = self.flow()
        _, state, agent, _ = self.inputs()
        angle = torch.tensor([.6, -.3, 2.])
        state[:, 2:4] = torch.stack((angle.cos(), angle.sin()), -1)
        state[:, 6] = torch.tensor([5., 99., -7.])
        agent['batch_ego_heading'] = torch.full((3,), .4)
        agent['local_vel'] = torch.tensor([[8., 9.], [3., 4.], [-2., 5.]])
        token_vel = torch.tensor([[0., 0.], [5., 0.], [3., 4.]])
        agent['token_traj'] = (token_vel[None, :, None, :] * .5).expand(3, 3, 4, 2).clone()
        pos, heading, shape, velocity, token = flow.model.get_output(state, agent)
        self.assertEqual(velocity.shape, (3, 2))
        self.assertEqual(pos.shape, (3, 1, 2))
        global_heading = angle + .4
        torch.testing.assert_close(heading[:, 0], global_heading)
        expected = torch.zeros(3, 2)
        expected[0] = torch.tensor([5. * math.cos(1.), 5. * math.sin(1.)])
        theta = global_heading[1]
        expected[1] = torch.stack((3 * theta.cos() - 4 * theta.sin(),
                                   3 * theta.sin() + 4 * theta.cos()))
        torch.testing.assert_close(velocity, expected, atol=2e-6, rtol=0)
        torch.testing.assert_close(token[:, 0], torch.tensor([1, 2, 0]))
        torch.testing.assert_close(shape, state[:, 4:6])
        # Final reconstruction does not mutate the signed internal flow state.
        self.assertEqual(state[2, 6].item(), -7.)
        torch.testing.assert_close(velocity[0].norm(), state[0, 6])

    def test_x0_heading_mode_works_with_speed_state(self):
        flow = self.flow(heading_objective='x0').eval()
        _, clean, agent, feature = self.inputs()
        output = flow.model(clean, torch.full((3, 1), .4), agent, feature)
        self.assertEqual(output.shape, (3, 7))
        with patch.object(flow.model, 'forward', return_value=clean.clone()):
            generated = flow.sample(agent, feature, steps=20)
        torch.testing.assert_close(generated, clean, atol=2e-5, rtol=0)

    def test_vector_mode_retains_eight_dimensional_target_and_original_velocity(self):
        flow = self.flow(velocity_representation='vector')
        vector, _, agent, feature = self.inputs()
        agent['expert_input'] = vector.clone()
        source, target = flow.model.get_input(agent)
        torch.testing.assert_close(source, vector, atol=0, rtol=0)
        torch.testing.assert_close(target, vector, atol=0, rtol=0)
        self.assertEqual(flow.model.m_delta_dim, 8)
        output = flow.model(vector, torch.full((3, 1), .4), agent, feature)
        self.assertEqual(output.shape, (3, 9))
        agent['token_traj'] = torch.zeros(3, 2, 4, 2)
        actual = flow.model.get_output(vector, agent)[3]
        theta = torch.atan2(vector[:, 3], vector[:, 2])
        expected = torch.stack((vector[:, 6] * theta.cos() - vector[:, 7] * theta.sin(),
                                vector[:, 6] * theta.sin() + vector[:, 7] * theta.cos()), -1)
        torch.testing.assert_close(actual, expected)

    def test_validation_rejects_invalid_mode_and_unsupported_policy_paths(self):
        with self.assertRaisesRegex(ValueError, 'velocity_representation'):
            self.flow(velocity_representation='unknown')
        # Gaussian heading isolates the speed-mode restriction from circular SDE.
        with self.assertRaisesRegex(ValueError, '(?i)speed'):
            Flow(self.args(heading_noise='gaussian', heading_objective='x0'), self.processor(), True)
        with self.assertRaisesRegex(ValueError, '(?i)speed'):
            Flow(self.args(), self.processor(use_refiner=True), False)

    def test_public_wrapper_trains_and_evaluates_real_speed_flow_with_ema(self):
        for objective in ('x0', 'angular_velocity'):
            with self.subTest(heading_objective=objective):
                with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
                    model = InitDiffusion(32, 2, 4, self.processor(), False,
                                          velocity_representation='speed',
                                          heading_noise='circular', heading_objective=objective,
                                          use_ema=True, ema_decay=.9)
                vector, _, agent, feature = self.inputs()
                vector[1, 6:8] = torch.tensor([3., 4.])
                initial_heading = torch.atan2(vector[:, 3], vector[:, 2]) + .35
                agent.update(expert_input=vector.clone(), initial_pos=vector[:, :2].clone(),
                             initial_heading=initial_heading.clone(),
                             local_vel=vector[:, 6:8].clone(),
                             map_feature=dict(feature, pt_token=torch.randn(3, 128)))
                token_vel = torch.tensor([[0., 0.], [5., 0.], [3., 4.]])
                agent['token_traj'] = (token_vel[None, :, None, :] * .5).expand(3, 3, 4, 2).clone()
                original = {key: agent[key].clone() for key in ('initial_pos', 'initial_heading', 'local_vel')}
                raw_map_tokens = agent['map_feature']['pt_token'].clone()
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)
                with patch.object(model.G1, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = model(agent)
                self.assertEqual(len(losses), 6)
                self.assertTrue(all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses))
                self.assertEqual(agent['expert_input'].shape, (3, 7))
                torch.testing.assert_close(agent['expert_input'][:, 6], vector[:, 6:8].norm(dim=-1))
                sum(losses).backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(model.G1.model.to_out_m_delta.mlp[-1].weight.grad[6].abs().sum().item(), 0.)
                optimizer.step()
                model.update_ema()
                self.assertEqual(model.ema.num_updates, 1)
                online_parameters = [parameter.detach().clone() for parameter in model.G1.parameters()]
                model.eval()
                position, heading, token_index, shape, velocity = model(agent)
                self.assertEqual(position.shape, (3, 1, 2))
                self.assertEqual(heading.shape, (3, 1))
                self.assertEqual(token_index.shape, (3, 1))
                self.assertEqual(shape.shape, (3, 2))
                self.assertEqual(velocity.shape, (3, 2))
                self.assertTrue(all(torch.isfinite(output).all() for output in
                                    (position, heading, token_index, shape, velocity)))
                non_ego = ~agent['ego_mask']
                speed = agent['gen_z'][non_ego, 6].clamp_min(0.)
                direction = torch.stack((heading[non_ego, 0].cos(), heading[non_ego, 0].sin()), -1)
                torch.testing.assert_close(velocity[non_ego], speed[:, None] * direction, atol=2e-6, rtol=0)
                ego_theta = original['initial_heading'][1]
                ego_velocity = torch.stack((3 * ego_theta.cos() - 4 * ego_theta.sin(),
                                             3 * ego_theta.sin() + 4 * ego_theta.cos()))
                torch.testing.assert_close(velocity[1], ego_velocity, atol=2e-6, rtol=0)
                torch.testing.assert_close(velocity[1].norm(), torch.tensor(5.))
                for key, value in original.items():
                    torch.testing.assert_close(agent[key], value, atol=0, rtol=0)
                torch.testing.assert_close(agent['map_feature']['pt_token'], raw_map_tokens, atol=0, rtol=0)
                for before, after in zip(online_parameters, model.G1.parameters()):
                    torch.testing.assert_close(after, before, atol=0, rtol=0)

    def test_speed_parameters_and_ema_survive_strict_checkpoint_roundtrip(self):
        model = self.wrapper(use_ema=True, ema_decay=.9)
        self.assertEqual(model.G1.model.m_delta_dim, 7)
        with torch.no_grad():
            model.G1.model.to_out_m_delta.mlp[-1].weight[6].add_(1.)
            for parameter in model.G1.model.to_out_heading_velocity.parameters():
                parameter.add_(1.)
        model.update_ema()
        saved = copy.deepcopy(model.state_dict())
        restored = self.wrapper(use_ema=True, ema_decay=.9)
        restored.load_state_dict(saved, strict=True)
        self.assertEqual(restored.ema.num_updates, 1)
        self.assertEqual(len(restored.ema.shadow_params), len(list(restored.G1.parameters())))
        for expected, actual in zip(model.ema.shadow_params, restored.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for expected, actual in zip(model.G1.parameters(), restored.G1.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        vector = self.wrapper(mode='vector', use_ema=True)
        with self.assertRaises((RuntimeError, ValueError)):
            vector.load_state_dict(saved, strict=True)

    def test_train_eval_default_to_speed_and_allow_legacy_vector_override(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                for mode in ('speed', 'vector'):
                    overrides = [f'experiment={experiment}']
                    if mode == 'vector':
                        overrides.append('model.model_config.decoder.init_diffusion.velocity_representation=vector')
                    config = compose(config_name='run.yaml', overrides=overrides)
                    self.assertEqual(config.model.model_config.decoder.init_diffusion.velocity_representation, mode)


if __name__ == '__main__':
    unittest.main()
