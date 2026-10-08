"""Optional speed supervision leaves vector flow and checkpoint structure intact."""

import copy
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate
from src.smart.model.smart_gail import SMART_GAIL


class InitDiffusionSpeedMagnitudeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(863)

    @staticmethod
    def args(**options):
        values = dict(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                      num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                      sampling_steps=20, use_rl=False, heading_noise='circular',
                      heading_objective='x0', velocity_representation='vector')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2)/dt)
        values.update(options)
        return SimpleNamespace(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        values = dict(heading_noise='circular', heading_objective='x0',
                      velocity_representation='vector')
        values.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **values)

    @staticmethod
    def inputs():
        physical = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                                 [0., 0., 1., 0., 4.8, 2.1, 0., 0.],
                                 [-20., 30., 0., 1., .9, .6, -2., 0.]])
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 0, 2]),
                     num_graphs=1, ego_mask=torch.tensor([False, True, False]),
                     expert_input=physical.clone(), shape=physical[:, 4:6].clone(),
                     local_vel=physical[:, 6:8].clone(), batch_ego_pos=torch.zeros(3, 2),
                     batch_ego_heading=torch.zeros(3), initial_pos=physical[:, :2].clone(),
                     initial_heading=torch.atan2(physical[:, 3], physical[:, 2]),
                     token_traj=torch.zeros(3, 2, 4, 2),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return physical, agent, feature

    def test_known_velocity_norms_match_normalized_scalar_mse(self):
        flow = self.flow(speed_loss_weight=.7, speed_loss_scale=5.)
        prediction = torch.zeros(2, 8)
        target = prediction.clone()
        prediction[:, 6:8] = torch.tensor([[3., 4.], [-6., 8.]])
        target[:, 6:8] = torch.tensor([[0., 10.], [0., 0.]])
        loss, scale = flow._speed_magnitude_loss(prediction, target, torch.full((2, 1), .4),
                                                 torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(scale, torch.tensor(5.))
        torch.testing.assert_close(loss, torch.tensor((1. + 4.) / 2))
        self.assertEqual(loss.ndim, 0)

    def test_velocity_direction_changes_do_not_create_magnitude_loss(self):
        flow = self.flow(speed_loss_scale=2.)
        target = torch.zeros(3, 8)
        target[:, 6:8] = torch.tensor([[3., 4.], [0., -2.], [0., 0.]])
        prediction = target.clone()
        prediction[:, 6:8] = torch.tensor([[-4., 3.], [2., 0.], [0., 0.]])
        loss, _ = flow._speed_magnitude_loss(prediction, target, torch.full((3, 1), .2),
                                             torch.zeros(3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        self.assertGreater((prediction[:, 6:8] - target[:, 6:8]).square().sum().item(), 0.)

    def test_moving_and_stationary_targets_have_finite_expected_gradients(self):
        flow = self.flow(speed_loss_scale=5.)
        prediction = torch.zeros(3, 8)
        prediction[:, 6:8] = torch.tensor([[3., 4.], [3., 4.], [0., 0.]])
        prediction.requires_grad_()
        target = torch.zeros(3, 8, requires_grad=True)
        with torch.no_grad():
            target[0, 6:8] = torch.tensor([6., 8.])
        loss, _ = flow._speed_magnitude_loss(prediction, target, torch.full((3, 1), .4),
                                             torch.zeros(3, dtype=torch.bool))
        loss.backward()
        expected = torch.tensor([[-.08, -8./75.], [.08, 8./75.], [0., 0.]])
        torch.testing.assert_close(prediction.grad[:, 6:8], expected)
        torch.testing.assert_close(prediction.grad[:, :6], torch.zeros(3, 6), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertIsNone(target.grad)

    def test_zero_prediction_has_finite_zero_norm_gradient_and_vector_loss_supplies_direction(self):
        flow = self.flow(speed_loss_scale=1.)
        prediction = torch.zeros(1, 8, requires_grad=True)
        target = torch.zeros(1, 8)
        target[:, 6:8] = torch.tensor([[3., 4.]])
        loss, _ = flow._speed_magnitude_loss(prediction, target, torch.tensor([[.5]]),
                                             torch.tensor([False]))
        loss.backward(retain_graph=True)
        self.assertEqual(loss.item(), 25.)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)
        prediction.grad.zero_()
        (loss + (prediction[:, 6:8] - target[:, 6:8]).square().mean()).backward()
        torch.testing.assert_close(prediction.grad[:, 6:8], torch.tensor([[-3., -4.]]))

    def test_only_non_ego_interior_times_enter_active_agent_mean(self):
        flow = self.flow(speed_loss_scale=2.)
        prediction = torch.zeros(7, 8, requires_grad=True)
        target = torch.zeros_like(prediction)
        with torch.no_grad():
            prediction[:, 6] = torch.tensor([2., 4., 900., 900., 900., 900., 900.])
        time = torch.tensor([[.2], [.8], [0.], [1.], [-.1], [1.1], [.5]])
        ego = torch.tensor([False, False, False, False, False, False, True])
        loss, _ = flow._speed_magnitude_loss(prediction, target, time, ego)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad[2:], torch.zeros(5, 8), atol=0, rtol=0)
        self.assertGreater(prediction.grad[:2, 6].sum().item(), 0.)

    def test_all_ego_or_endpoint_agents_produce_differentiable_zero(self):
        flow = self.flow(speed_loss_scale=2.)
        for time, ego in ((torch.full((3, 1), .4), torch.ones(3, dtype=torch.bool)),
                          (torch.tensor([[0.], [1.], [-.2]]), torch.zeros(3, dtype=torch.bool))):
            prediction = torch.randn(3, 8, requires_grad=True)
            loss, scale = flow._speed_magnitude_loss(prediction, torch.zeros(3, 8), time, ego)
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(loss.requires_grad)
            self.assertEqual(scale.item(), 2.)
            loss.backward()
            torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)

    def test_auto_scale_uses_both_velocity_mean_and_population_variance(self):
        flow = self.flow()
        with torch.no_grad():
            flow.model.normal_mean[0, 6:8] = torch.tensor([3., 4.])
            flow.model.normal_scale[0, 6:8] = torch.tensor([12., 0.])
        before = {key: value.clone() for key, value in flow.model.state_dict().items()}
        target = torch.zeros(2, 8)
        prediction = target.clone()
        prediction[:, 6] = 13.
        loss, scale = flow._speed_magnitude_loss(prediction, target, torch.full((2, 1), .5),
                                                 torch.zeros(2, dtype=torch.bool))
        self.assertEqual(scale.item(), 13.)
        self.assertEqual(loss.item(), 1.)
        for key, value in before.items():
            torch.testing.assert_close(flow.model.state_dict()[key], value, atol=0, rtol=0)

    def test_auto_scale_floor_and_fixed_override_are_detached(self):
        for fixed, expected in ((None, 1.), (.25, .25)):
            flow = self.flow(speed_loss_scale=fixed)
            flow.model.normal_mean.zero_().requires_grad_()
            flow.model.normal_scale.zero_().requires_grad_()
            prediction = torch.ones(2, 8, requires_grad=True)
            loss, scale = flow._speed_magnitude_loss(prediction, torch.zeros(2, 8),
                                                     torch.full((2, 1), .5),
                                                     torch.zeros(2, dtype=torch.bool))
            self.assertEqual(scale.item(), expected)
            self.assertFalse(scale.requires_grad)
            loss.backward()
            self.assertIsNone(flow.model.normal_mean.grad)
            self.assertIsNone(flow.model.normal_scale.grad)

    def test_loss_and_empirical_rms_are_invariant_under_agent_frame_rotations(self):
        flow = self.flow()
        target = torch.zeros(4, 8, dtype=torch.float64)
        prediction = target.clone()
        target[:, 6:8] = torch.tensor([[3., 4.], [-2., 1.], [0., 0.], [5., -7.]])
        prediction[:, 6:8] = torch.tensor([[6., 8.], [1., 2.], [0., 3.], [-4., 2.]])
        time, ego = torch.full((4, 1), .5), torch.zeros(4, dtype=torch.bool)

        def fit_and_loss(p, t):
            with torch.no_grad():
                flow.model.normal_mean[0, 6:8] = t[:, 6:8].mean(0)
                flow.model.normal_scale[0, 6:8] = t[:, 6:8].std(0, unbiased=False)
            return flow._speed_magnitude_loss(p, t, time, ego)

        before = fit_and_loss(prediction, target)
        theta = torch.tensor([.2, -1.5, math.pi, 2.3], dtype=torch.float64)
        rotation = torch.stack((theta.cos(), -theta.sin(), theta.sin(), theta.cos()), -1).reshape(4, 2, 2)
        rotated_prediction, rotated_target = prediction.clone(), target.clone()
        for rotated, original in ((rotated_prediction, prediction), (rotated_target, target)):
            rotated[:, 6:8] = torch.einsum('nij,nj->ni', rotation, original[:, 6:8])
        after = fit_and_loss(rotated_prediction, rotated_target)
        for first, second in zip(before, after):
            torch.testing.assert_close(first, second, rtol=1.e-6, atol=1.e-7)

    def test_half_and_bfloat16_norms_use_float32_without_overflow(self):
        flow = self.flow(speed_loss_scale=100.)
        for dtype in (torch.float16, torch.bfloat16):
            prediction = torch.zeros(2, 8, dtype=dtype)
            prediction[:, 6:8] = torch.tensor([[300., 400.], [0., 0.]], dtype=dtype)
            prediction.requires_grad_()
            target = torch.zeros_like(prediction)
            target[:, 6:8] = torch.tensor([[0., 0.], [300., 400.]], dtype=dtype)
            loss, scale = flow._speed_magnitude_loss(prediction, target, torch.full((2, 1), .5),
                                                     torch.zeros(2, dtype=torch.bool))
            self.assertEqual(loss.dtype, torch.float32)
            self.assertEqual(scale.dtype, torch.float32)
            oracle = (prediction[:, 6:8].float().norm(dim=-1)
                      - target[:, 6:8].float().norm(dim=-1)).div(100.).square().mean()
            torch.testing.assert_close(loss, oracle)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_invalid_weight_scale_and_scalar_speed_duplicate_supervision_raise(self):
        for weight in (-.1, float('nan'), float('inf')):
            for constructor in (self.flow, self.wrapper):
                with self.subTest(weight=weight, constructor=constructor), self.assertRaises(ValueError):
                    constructor(speed_loss_weight=weight)
        for scale in (0., -.1, float('nan'), float('inf')):
            for constructor in (self.flow, self.wrapper):
                with self.subTest(scale=scale, constructor=constructor), self.assertRaises(ValueError):
                    constructor(speed_loss_scale=scale)
        for constructor in (self.flow, self.wrapper):
            with self.assertRaisesRegex(ValueError, '(?i)vector'):
                constructor(speed_loss_weight=.5, velocity_representation='speed')
            model = constructor(speed_loss_weight=0., velocity_representation='speed')
            self.assertEqual(model.G1.velocity_representation if isinstance(model, InitDiffusion)
                             else model.velocity_representation, 'speed')
        with self.assertRaisesRegex(ValueError, '(?i)refiner'):
            Flow(self.args(speed_loss_weight=.5), self.processor(use_refiner=True), False)

    def test_auxiliary_changes_only_total_and_keeps_collision_vector_and_heading_components(self):
        for objective in ('x0', 'angular_velocity'):
            with self.subTest(objective=objective):
                flow = self.flow(heading_objective=objective, speed_loss_weight=.7, speed_loss_scale=2.,
                                 size_representation='log')
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                noise = clean.clone()
                noise[:, 2:4] = torch.tensor([[.8, .6], [.8, .6], [-.6, .8]])
                time = torch.tensor([[.3], [0.], [.7]])
                latent = _circular_interpolate(clean, noise, time)
                predicted = clean.clone()
                predicted[:, :2] += .25
                predicted[:, 4:6] += .15
                predicted[:, 6:8] *= 1.5
                if objective == 'angular_velocity':
                    predicted = torch.cat((predicted, torch.full((3, 1), .2)), -1)
                with patch.object(flow, '_prepare_supervised_batch', return_value=(noise, time, latent)), \
                        patch.object(flow.model, 'forward', return_value=predicted):
                    enabled = flow._supervised_loss(clean, agent, feature)
                    speed_metrics = copy.deepcopy(agent['_init_diffusion_speed_metrics'])
                    flow.speed_loss_weight = 0.
                    disabled = flow._supervised_loss(clean, agent, feature)
                self.assertEqual(len(enabled), 6)
                for index in range(1, 6):
                    torch.testing.assert_close(enabled[index], disabled[index], atol=0, rtol=0)
                expected_prediction = predicted[:, :8].clone()
                expected_prediction[agent['ego_mask']] = clean[agent['ego_mask']]
                expected_raw, scale = flow._speed_magnitude_loss(expected_prediction, clean, time,
                                                                 agent['ego_mask'])
                torch.testing.assert_close(speed_metrics['loss'], expected_raw)
                torch.testing.assert_close(speed_metrics['scale'], scale)
                torch.testing.assert_close(enabled[0].mean() - disabled[0].mean(), .7*expected_raw)
                torch.testing.assert_close(speed_metrics['weighted_loss'], .7*expected_raw)
                self.assertNotIn('_init_diffusion_speed_metrics', agent)

    def test_noise_sampling_and_interpolation_are_identical_when_auxiliary_is_enabled(self):
        torch.manual_seed(863)
        baseline = self.flow()
        torch.manual_seed(863)
        enabled = self.flow(speed_loss_weight=2.)
        _, agent, _ = self.inputs()
        clean, _ = baseline.model.get_input(copy.deepcopy(agent))
        enabled.model.load_state_dict(baseline.model.state_dict(), strict=True)
        torch.manual_seed(174)
        expected = baseline._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_expected = torch.rand(3)
        torch.manual_seed(174)
        actual = enabled._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_actual = torch.rand(3)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(next_actual, next_expected, atol=0, rtol=0)

    def test_disabled_default_preserves_rng_parameters_buffers_and_strict_checkpoint_schema(self):
        torch.manual_seed(863)
        default = self.flow()
        after_default = torch.rand(5)
        torch.manual_seed(863)
        explicit = self.flow(speed_loss_weight=0., speed_loss_scale=None)
        after_explicit = torch.rand(5)
        torch.testing.assert_close(after_default, after_explicit, atol=0, rtol=0)
        torch.manual_seed(863)
        enabled = self.flow(speed_loss_weight=1., speed_loss_scale=3.)
        after_enabled = torch.rand(5)
        torch.testing.assert_close(after_default, after_enabled, atol=0, rtol=0)
        for other in (explicit, enabled):
            self.assertEqual(set(default.state_dict()), set(other.state_dict()))
            self.assertEqual(list(dict(default.named_parameters())), list(dict(other.named_parameters())))
            self.assertEqual(list(dict(default.named_buffers())), list(dict(other.named_buffers())))
            other.load_state_dict(default.state_dict(), strict=True)
            for key, value in default.state_dict().items():
                torch.testing.assert_close(other.state_dict()[key], value, atol=0, rtol=0)

    def test_enabled_ema_checkpoint_loads_strictly_into_disabled_runtime_option(self):
        source = self.wrapper(use_ema=True, ema_decay=.9, speed_loss_weight=.5, speed_loss_scale=3.)
        self.assertEqual(source.G1.speed_loss_weight, .5)
        self.assertEqual(source.G1.speed_loss_scale, 3.)
        with torch.no_grad():
            source.G1.model.to_out_m_delta.mlp[-1].weight[6:8].add_(.02)
        source.update_ema()
        saved_file = io.BytesIO()
        torch.save(source.state_dict(), saved_file)
        saved_file.seek(0)
        saved = torch.load(saved_file, weights_only=False)
        target = self.wrapper(use_ema=True, ema_decay=.7)
        self.assertEqual(set(target.state_dict()), set(saved))
        result = target.load_state_dict(saved, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(target.G1.speed_loss_weight, 0.)
        self.assertEqual(target.ema.num_updates, 1)
        self.assertEqual(target.ema.decay, .9)
        self.assertEqual(len(target.ema.shadow_params), len(source.ema.shadow_params))
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_metrics_are_detached_and_all_ego_enabled_batch_reports_zero(self):
        flow = self.flow(speed_loss_weight=.4, speed_loss_scale=2.)
        physical, agent, feature = self.inputs()
        agent['ego_mask'].fill_(True)
        losses = flow._supervised_loss(physical, agent, feature)
        metrics = agent['_init_diffusion_speed_metrics']
        self.assertEqual(set(metrics), {'loss', 'weighted_loss', 'scale'})
        self.assertEqual(metrics['loss'].item(), 0.)
        self.assertEqual(metrics['weighted_loss'].item(), 0.)
        self.assertEqual(metrics['scale'].item(), 2.)
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in metrics.values()))
        self.assertTrue(torch.isfinite(losses[0]).all())

    def test_real_training_updates_velocity_head_with_log_sizes_and_conditional_embeddings(self):
        for objective in ('x0', 'angular_velocity'):
            model = self.wrapper(speed_loss_weight=.7, speed_loss_scale=5., size_representation='log',
                                 heading_objective=objective, use_ema=True, ema_decay=.9,
                                 time_embedding_type='scenario_dreamer',
                                 count_embedding_type='scenario_dreamer',
                                 map_embedding_type='scenario_dreamer', map_label_dropout=0.)
            _, agent, feature = self.inputs()
            agent['initial_map_feature'] = feature
            optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
            head = model.G1.model.to_out_m_delta.mlp[-1].weight
            before = head[6:8].detach().clone()
            with patch.object(model.G1, '_sample_time', return_value=torch.full((3, 1), .4)):
                losses = model.train()(agent)
            self.assertEqual(len(losses), 6)
            logged = {}
            trainer = SimpleNamespace(encoder=SimpleNamespace(init_decoder=model),
                                      _log_train=lambda name, value: logged.update({name: value}))
            total = SMART_GAIL._initial_prediction_loss(trainer, {'initial_logit': losses}, agent,
                                                        losses[0])
            torch.testing.assert_close(total, losses[0] + losses[1])
            for name in ('loss', 'weighted_loss', 'scale'):
                torch.testing.assert_close(logged[f'train/speed_{name}'],
                                           agent['_init_diffusion_speed_metrics'][name])
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertGreater(head.grad[6:8].abs().sum().item(), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            self.assertFalse(torch.equal(before, head[6:8]))
            model.update_ema()
            output = model.eval()(agent)
            self.assertEqual(len(output), 5)
            self.assertTrue(torch.isfinite(output[4]).all())
            self.assertTrue((output[3] > 0).all())
            self.assertEqual(agent['_init_diffusion_size_representation'], 'log')

    def test_disabled_training_clears_stale_speed_metrics_and_omits_speed_log_keys(self):
        model = self.wrapper()
        _, agent, feature = self.inputs()
        agent.update(initial_map_feature=feature,
                     _init_diffusion_speed_metrics={'loss': torch.tensor(99.),
                                                    'weighted_loss': torch.tensor(99.),
                                                    'scale': torch.tensor(99.)})
        losses = model.train()(agent)
        self.assertNotIn('_init_diffusion_speed_metrics', agent)
        logged = {}
        trainer = SimpleNamespace(encoder=SimpleNamespace(init_decoder=model),
                                  _log_train=lambda name, value: logged.update({name: value}))
        SMART_GAIL._initial_prediction_loss(trainer, {'initial_logit': losses}, agent, losses[0])
        self.assertFalse(any(name.startswith('train/speed_') for name in logged))
        self.assertIn('train/vel_loss', logged)

    def test_train_eval_configs_default_disabled_and_accept_vector_speed_loss_override(self):
        root = Path(__file__).resolve().parents[1]
        generic = OmegaConf.load(root/'configs/model/smart.yaml').model_config.decoder.init_diffusion
        self.assertEqual(generic.speed_loss_weight, 0.)
        self.assertIsNone(generic.speed_loss_scale)
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                    'model.model_config.decoder.init_diffusion.velocity_representation=vector',
                    'model.model_config.decoder.init_diffusion.speed_loss_weight=0.7',
                    'model.model_config.decoder.init_diffusion.speed_loss_scale=5.0'])
                options = config.model.model_config.decoder.init_diffusion
                self.assertEqual(options.velocity_representation, 'vector')
                self.assertEqual(options.speed_loss_weight, .7)
                self.assertEqual(options.speed_loss_scale, 5.)


if __name__ == '__main__':
    unittest.main()
