"""Log sizes remain internal to the full flow; geometry and outputs use meters."""

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

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate, _noise_endpoint
from src.smart.diffusion.diffusion_utils import get_diff_loss


class InitDiffusionLogSizeTest(unittest.TestCase):
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
                      heading_objective='x0', velocity_representation='vector',
                      size_representation='log')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2) / dt)
        values.update(options)
        return SimpleNamespace(**values)

    def denoiser(self, **options):
        values = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                      num_layers=1, num_heads=2, dropout=0., size_representation='log')
        values.update(options)
        return InitDenoiser(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        values = dict(size_representation='log', heading_noise='circular',
                      heading_objective='x0', velocity_representation='vector')
        values.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **values)

    @staticmethod
    def inputs(representation='vector'):
        vector = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                               [0., 0., 1., 0., 4.8, 2.1, 0., 0.],
                               [-20., 30., 0., 1., .9, .6, -2., 0.]])
        physical = (torch.cat((vector[:, :6], vector[:, 6:8].norm(dim=-1, keepdim=True)), -1)
                    if representation == 'speed' else vector)
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 0, 2]),
                     num_graphs=1, ego_mask=torch.tensor([False, True, False]),
                     expert_input=physical.clone(), shape=vector[:, 4:6].clone(),
                     local_vel=vector[:, 6:8].clone(), batch_ego_pos=torch.zeros(3, 2),
                     batch_ego_heading=torch.zeros(3), initial_pos=vector[:, :2].clone(),
                     initial_heading=torch.atan2(vector[:, 3], vector[:, 2]),
                     token_traj=torch.zeros(3, 2, 4, 2),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return physical, agent, feature

    def test_roundtrip_changes_only_size_fields_and_preserves_input_tensors(self):
        for representation, width in (('vector', 8), ('speed', 7)):
            model = self.denoiser(input_dim=width, output_dim=width,
                                  velocity_representation=representation)
            physical, _, _ = self.inputs(representation)
            for dtype in (torch.float32, torch.float64):
                original = physical.to(dtype=dtype)
                before = original.clone()
                internal = model.state_to_model(original)
                torch.testing.assert_close(internal[:, 4:6], original[:, 4:6].log())
                torch.testing.assert_close(internal[:, :4], original[:, :4], atol=0, rtol=0)
                torch.testing.assert_close(internal[:, 6:], original[:, 6:], atol=0, rtol=0)
                self.assertEqual(internal.dtype, dtype)
                torch.testing.assert_close(model.state_to_physical(internal), original)
                torch.testing.assert_close(original, before, atol=0, rtol=0)

    def test_log_mode_rejects_invalid_clean_sizes_at_encoding_boundary(self):
        model = self.denoiser(invalid_size_policy="error")
        physical, agent, _ = self.inputs()
        for value in (0., -1., float('nan'), float('inf')):
            for field in (4, 5):
                invalid = physical.clone()
                invalid[0, field] = value
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    model.state_to_model(invalid)
                inputs = copy.deepcopy(agent)
                inputs['expert_input'] = invalid
                with self.subTest(cache_value=value, field=field), self.assertRaises(ValueError):
                    model.get_input(inputs)

    def test_exp_output_is_positive_continuous_and_not_clamped_to_physical_bounds(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        physical[:, 4:6] = torch.tensor([[.01, .02], [1000., 250.], [1001., 251.]])
        internal = model.state_to_model(physical)
        result = model.state_to_physical(internal)
        torch.testing.assert_close(result[:, 4:6], physical[:, 4:6])
        self.assertTrue((result[:, 4:6] > 0).all())
        self.assertGreater(result[2, 4].item(), result[1, 4].item())
        self.assertGreater(result[2, 5].item(), result[1, 5].item())

    def test_low_precision_physical_exp_uses_float32_and_double_stays_double(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        for dtype in (torch.float16, torch.bfloat16):
            internal = physical.to(dtype=dtype)
            internal[:, 4:6] = 12.
            result = model.state_to_physical(internal)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result[:, 4:6], torch.full((3, 2), 12.).exp(), atol=0, rtol=0)
            torch.testing.assert_close(result[:, :4], internal[:, :4].float(), atol=0, rtol=0)
            self.assertTrue(torch.isfinite(result).all())
        internal = physical.double()
        internal[:, 4:6] = 100.
        result = model.state_to_physical(internal)
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.isfinite(result).all())

    def test_nonfinite_underflow_and_overflow_do_not_silently_become_physical_boxes(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        internal = model.state_to_model(physical)
        for value in (float('nan'), float('inf'), -float('inf'), 1000., -1000.):
            invalid = internal.clone()
            invalid[0, 4] = value
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                model.state_to_physical(invalid)

    def test_constructed_input_logs_sizes_and_fits_internal_normalizer_without_changing_gt(self):
        for representation in ('vector', 'speed'):
            flow = self.flow(velocity_representation=representation)
            physical, agent, _ = self.inputs(representation)
            agent.pop('expert_input')
            shape_before = agent['shape'].clone()
            source, target = flow.model.get_input(agent)
            torch.testing.assert_close(source, target, atol=0, rtol=0)
            torch.testing.assert_close(source[:, 4:6], shape_before.log())
            torch.testing.assert_close(flow.model.normal_mean[:, 4:6], source[:, 4:6].mean(0, keepdim=True))
            expected_std = source[:, 4:6].std(0, unbiased=False, keepdim=True).clamp_min(1.e-6)
            torch.testing.assert_close(flow.model.normal_scale[:, 4:6], expected_std)
            torch.testing.assert_close(agent['shape'], shape_before, atol=0, rtol=0)
            torch.testing.assert_close(flow.model.state_to_physical(source)[:, 4:6], physical[:, 4:6])

    def test_cached_input_converts_once_for_vector_and_speed_and_records_representation(self):
        for representation in ('vector', 'speed'):
            flow = self.flow(velocity_representation=representation)
            vector, agent, _ = self.inputs('vector')
            original_cache = agent['expert_input']
            original_before = original_cache.clone()
            first, _ = flow.model.get_input(agent)
            self.assertEqual(first.shape[-1], 7 if representation == 'speed' else 8)
            self.assertEqual(agent['_init_diffusion_size_representation'], 'log')
            torch.testing.assert_close(first[:, 4:6], vector[:, 4:6].log())
            normal_mean = flow.model.normal_mean.clone()
            normal_scale = flow.model.normal_scale.clone()
            for _ in range(3):
                repeated, _ = flow.model.get_input(agent)
                torch.testing.assert_close(repeated, first, atol=0, rtol=0)
            torch.testing.assert_close(agent['expert_input'], first, atol=0, rtol=0)
            torch.testing.assert_close(original_cache, original_before, atol=0, rtol=0)
            torch.testing.assert_close(flow.model.normal_mean, normal_mean, atol=0, rtol=0)
            torch.testing.assert_close(flow.model.normal_scale, normal_scale, atol=0, rtol=0)
            if representation == 'speed':
                torch.testing.assert_close(first[:, 6], vector[:, 6:8].norm(dim=-1))

    def test_mismatched_cached_representation_is_rejected_in_both_directions(self):
        physical, agent, _ = self.inputs()
        log = self.denoiser()
        linear = self.denoiser(size_representation='linear')
        log.get_input(agent)
        with self.assertRaises(ValueError):
            linear.get_input(agent)
        agent['expert_input'] = physical.clone()
        agent['_init_diffusion_size_representation'] = 'linear'
        with self.assertRaises(ValueError):
            log.get_input(agent)

    def test_noise_endpoint_uses_gaussian_log_statistics_without_exponentiating(self):
        flow = self.flow(heading_noise='gaussian')
        _, agent, _ = self.inputs()
        flow.model.get_input(agent)
        eps = torch.linspace(-3., 2., 24).reshape(3, 8)
        actual = _noise_endpoint(flow.model, eps, None, 'gaussian')
        expected = flow.model.normal_mean + flow.model.normal_scale * eps
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 4:6], expected[:, 4:6], atol=0, rtol=0)
        self.assertFalse(torch.allclose(actual[:, 4:6], expected[:, 4:6].exp()))

    def test_size_interpolation_in_log_space_has_geometric_physical_midpoint(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        clean = model.state_to_model(physical)
        endpoint_physical = physical.clone()
        endpoint_physical[:, 4:6] = physical[:, 4:6] * 4.
        noise = model.state_to_model(endpoint_physical)
        time = torch.full((3, 1), .5)
        for latent in ((1-time)*clean + time*noise, _circular_interpolate(clean, noise, time)):
            result = model.state_to_physical(latent)
            torch.testing.assert_close(result[:, 4:6], physical[:, 4:6] * 2.)
            self.assertFalse(torch.allclose(result[:, 4:6], physical[:, 4:6] * 2.5))

    def test_sampling_converts_existing_cache_and_fits_log_prior_before_drawing_endpoint(self):
        flow = self.flow(heading_noise='gaussian').eval()
        physical, agent, feature = self.inputs()
        internal = flow.model.state_to_model(physical)
        eps = torch.linspace(-.4, .5, 24).reshape(3, 8)
        with patch('src.smart.diffusion.scale_flow.torch.randn', return_value=eps.clone()), \
                patch.object(flow.model, 'forward', return_value=internal.clone()):
            generated = flow.sample(agent, feature, steps=20)
        torch.testing.assert_close(agent['gen_noise'][:, 4:6],
                                   flow.model.denormalize(eps)[:, 4:6])
        torch.testing.assert_close(flow.model.normal_mean[:, 4:6], internal[:, 4:6].mean(0, keepdim=True))
        torch.testing.assert_close(generated, internal, atol=2.e-5, rtol=0)
        self.assertEqual(agent['_init_diffusion_size_representation'], 'log')
        torch.testing.assert_close(agent['expert_input'], internal, atol=0, rtol=0)

    def test_reverse_oracle_recovers_internal_log_target_and_physical_output_in_all_supported_paths(self):
        for representation in ('vector', 'speed'):
            for heading, objective in (('gaussian', 'x0'), ('circular', 'x0'),
                                       ('circular', 'angular_velocity')):
                with self.subTest(representation=representation, heading=heading, objective=objective):
                    flow = self.flow(velocity_representation=representation, heading_noise=heading,
                                     heading_objective=objective).eval()
                    physical, agent, feature = self.inputs(representation)
                    physical[:, 2:4] = torch.tensor([1., 0.])
                    agent['expert_input'] = physical.clone()
                    internal = flow.model.state_to_model(physical)
                    prediction = (torch.cat((internal, torch.zeros(3, 1)), -1)
                                  if objective == 'angular_velocity' else internal)
                    eps = torch.zeros_like(internal)
                    eps[:, 2] = 1.
                    for steps in (1, 20):
                        with patch('src.smart.diffusion.scale_flow.torch.randn', return_value=eps.clone()), \
                                patch.object(flow.model, 'forward', return_value=prediction.clone()):
                            generated = flow.sample(agent, feature, steps=steps)
                        torch.testing.assert_close(generated, internal, atol=2.e-5, rtol=0)
                        output = flow.model.get_output(generated, agent)
                        torch.testing.assert_close(output[2], physical[:, 4:6], atol=2.e-5, rtol=0)
                        self.assertTrue((output[2] > 0).all())

    def test_relative_log_size_error_has_equal_loss_for_different_physical_scales(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        physical = physical[:2].clone()
        physical[1, 4:6] = physical[0, 4:6] * 8.
        target = model.state_to_model(physical)
        prediction = target.clone()
        prediction[:, 4:6] += math.log(2.)
        agent = dict(batch=torch.tensor([0, 1]), type=torch.zeros(2, dtype=torch.long))
        losses = get_diff_loss(agent, prediction, target, torch.full((2, 1), .5), .05,
                               x_pred=True, use_col=True, w_pos=1.,
                               state_to_physical=model.state_to_physical)
        expected = torch.full((2,), math.log(2.) ** 2)
        torch.testing.assert_close(losses[4], expected)
        torch.testing.assert_close(losses[0], 2 * expected)
        self.assertEqual(losses[1].item(), 0.)

    def test_collision_uses_physical_boxes_and_preserves_gt_overlap_allowance(self):
        model = self.denoiser()
        physical, _, _ = self.inputs()
        physical[:, :2] = torch.tensor([[0., 0.], [3., 0.], [0., 30.]])
        physical[:, 2:4] = torch.tensor([1., 0.])
        physical[:, 4:6] = torch.tensor([4.5, 2.])
        prediction_physical = physical.clone()
        prediction_physical[0, 4:6] *= 2.
        prediction = model.state_to_model(prediction_physical)
        target = model.state_to_model(physical)
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.zeros(3, dtype=torch.long))
        time = torch.tensor([[.4], [0.], [.4]])
        actual = get_diff_loss(agent, prediction, target, time, .05, x_pred=True, use_col=True,
                              state_to_physical=model.state_to_physical)
        oracle = get_diff_loss(agent, prediction_physical, physical, time, .05,
                              x_pred=True, use_col=True)
        torch.testing.assert_close(actual[1], oracle[1])
        self.assertGreater(actual[1].item(), 0.)
        same = get_diff_loss(agent, target, target, time, .05, x_pred=True, use_col=True,
                            state_to_physical=model.state_to_physical)
        self.assertEqual(same[1].item(), 0.)

    def test_loss_transform_preserves_non_size_reconstruction_fields_and_weights(self):
        model = self.denoiser()
        physical, agent, _ = self.inputs()
        prediction_physical = physical.clone()
        prediction_physical[:, :2] += .4
        prediction_physical[:, 6:] += .3
        prediction_physical[:, 4:6] *= 1.5
        target = model.state_to_model(physical)
        prediction = model.state_to_model(prediction_physical)
        time = torch.tensor([[.2], [0.], [.7]])
        dims = (0, 1, 2, 3, 6, 7)
        options = dict(x_pred=True, use_col=True, reconstruction_dims=dims)
        actual = get_diff_loss(agent, prediction, target, time, .05,
                              state_to_physical=model.state_to_physical, **options)
        oracle = get_diff_loss(agent, prediction_physical, physical, time, .05, **options)
        for index in (0, 1, 2, 3, 5):
            torch.testing.assert_close(actual[index], oracle[index])

    def test_supervised_adapter_passes_log_states_and_physical_collision_transform(self):
        for representation in ('vector', 'speed'):
            for objective in ('x0', 'angular_velocity'):
                flow = self.flow(velocity_representation=representation, heading_objective=objective)
                _, agent, feature = self.inputs(representation)
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)), \
                        patch('src.smart.diffusion.scale_flow.get_diff_loss', wraps=get_diff_loss) as loss:
                    result = flow._supervised_loss(clean, agent, feature)
                torch.testing.assert_close(loss.call_args.args[2][:, 4:6], clean[:, 4:6])
                self.assertEqual(loss.call_args.args[1].shape, (3, 8))
                self.assertEqual(loss.call_args.kwargs['state_to_physical'], flow.model.state_to_physical)
                self.assertEqual(len(result), 6)

    def test_real_training_updates_log_size_head_and_generates_positive_physical_shapes(self):
        for representation in ('vector', 'speed'):
            for heading, objective in (('gaussian', 'x0'), ('circular', 'x0'),
                                       ('circular', 'angular_velocity')):
                with self.subTest(representation=representation, heading=heading, objective=objective):
                    flow = self.flow(velocity_representation=representation, heading_noise=heading,
                                     heading_objective=objective, time_embedding_type='scenario_dreamer',
                                     count_embedding_type='scenario_dreamer',
                                     map_embedding_type='scenario_dreamer', map_label_dropout=0.)
                    _, agent, feature = self.inputs(representation)
                    clean, _ = flow.model.get_input(agent)
                    before = flow.model.to_out_m_delta.mlp[-1].weight[4:6].detach().clone()
                    optimizer = torch.optim.Adam(flow.parameters(), lr=1.e-3)
                    with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                        losses = flow._supervised_loss(clean, agent, feature)
                    total = losses[0].mean() + losses[1]
                    self.assertTrue(torch.isfinite(total))
                    total.backward()
                    gradients = [p.grad for p in flow.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    size_gradient = flow.model.to_out_m_delta.mlp[-1].weight.grad[4:6]
                    self.assertGreater(size_gradient.abs().sum().item(), 0.)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, flow.model.to_out_m_delta.mlp[-1].weight[4:6]))
                    flow.eval()
                    generated = flow.sample(agent, feature, steps=20)
                    torch.testing.assert_close(generated[agent['ego_mask']], clean[agent['ego_mask']], atol=0, rtol=0)
                    output = flow.model.get_output(generated, agent)
                    self.assertTrue(torch.isfinite(output[2]).all())
                    self.assertTrue((output[2] > 0).all())
                    torch.testing.assert_close(output[2][agent['ego_mask']], agent['shape'][agent['ego_mask']])

    def test_linear_default_preserves_state_schema_rng_and_loss_call(self):
        options = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                       num_layers=1, num_heads=2, dropout=0.)
        torch.manual_seed(817)
        default = InitDenoiser(**options)
        next_default = torch.randn(8)
        torch.manual_seed(817)
        explicit = InitDenoiser(**options, size_representation='linear')
        next_explicit = torch.randn(8)
        torch.testing.assert_close(next_default, next_explicit, atol=0, rtol=0)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        explicit.load_state_dict(default.state_dict(), strict=True)
        physical, _, _ = self.inputs()
        torch.testing.assert_close(default.state_to_model(physical), physical, atol=0, rtol=0)
        torch.testing.assert_close(default.state_to_physical(physical), physical, atol=0, rtol=0)
        flow = self.flow(size_representation='linear')
        _, agent, feature = self.inputs()
        with patch('src.smart.diffusion.scale_flow.get_diff_loss', wraps=get_diff_loss) as loss:
            flow._supervised_loss(physical, agent, feature)
        self.assertNotIn('state_to_physical', loss.call_args.kwargs)

    def test_invalid_representation_and_unsupported_policy_paths_raise(self):
        for constructor in (lambda: self.denoiser(size_representation='unknown'),
                            lambda: self.flow(size_representation='unknown'),
                            lambda: self.wrapper(size_representation='unknown')):
            with self.assertRaises(ValueError):
                constructor()
        for args, processor, gail in ((self.args(heading_noise='gaussian'), self.processor(), True),
                                     (self.args(), self.processor(use_refiner=True), False),
                                     (self.args(use_rl=True), self.processor(), False)):
            with self.assertRaisesRegex(ValueError, '(?i)log'):
                Flow(args, processor, gail)

    def test_ema_roundtrip_and_checkpoint_representation_guard(self):
        for representation in ('vector', 'speed'):
            source = self.wrapper(use_ema=True, ema_decay=.9, velocity_representation=representation)
            self.assertEqual(source.get_extra_state()['size_representation'], 'log')
            with torch.no_grad():
                source.G1.model.to_out_m_delta.mlp[-1].weight[4:6].add_(.05)
            source.update_ema()
            buffer = io.BytesIO()
            torch.save(source.state_dict(), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=False)
            target = self.wrapper(use_ema=True, ema_decay=.7, velocity_representation=representation)
            loaded = target.load_state_dict(saved, strict=True)
            self.assertEqual(loaded.missing_keys, [])
            self.assertEqual(loaded.unexpected_keys, [])
            self.assertEqual(target.ema.num_updates, 1)
            self.assertEqual(target.ema.decay, .9)
            for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            linear = self.wrapper(size_representation='linear', velocity_representation=representation)
            with self.assertRaises(ValueError):
                linear.load_state_dict(saved, strict=True)
            old_saved = copy.deepcopy(linear.state_dict())
            old_saved['_extra_state'].pop('size_representation')
            linear.load_state_dict(old_saved, strict=True)
            with self.assertRaises(ValueError):
                target.load_state_dict(old_saved, strict=True)
            old_saved.pop('_extra_state')
            linear.load_state_dict(old_saved, strict=True)
            for strict in (True, False):
                with self.subTest(strict=strict), self.assertRaisesRegex(RuntimeError, '(?i)linear sizes'):
                    target.load_state_dict(old_saved, strict=strict)

    def test_wrapper_full_train_eval_uses_internal_log_state_and_physical_output_with_ema(self):
        for representation in ('vector', 'speed'):
            model = self.wrapper(use_ema=True, ema_decay=.9, velocity_representation=representation,
                                 heading_objective='angular_velocity')
            _, agent, feature = self.inputs(representation)
            agent['initial_map_feature'] = feature
            result = model.train()(agent)
            total = result[0] + result[1]
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertGreater(model.G1.model.to_out_m_delta.mlp[-1].weight.grad[4:6].abs().sum().item(), 0.)
            model.update_ema()
            output = model.eval()(agent)
            self.assertEqual(len(output), 5)
            self.assertTrue(torch.isfinite(output[3]).all())
            self.assertTrue((output[3] > 0).all())
            self.assertFalse(output[3].requires_grad)
            self.assertEqual(agent['_init_diffusion_size_representation'], 'log')

    def test_backbone_only_finetune_checkpoint_can_load_without_size_metadata(self):
        parent = torch.nn.Module()
        parent.backbone = torch.nn.Linear(2, 2)
        parent.init_decoder = self.wrapper(use_ema=True)
        before = copy.deepcopy(parent.init_decoder.G1.state_dict())
        checkpoint = {'backbone.weight': torch.ones(2, 2), 'backbone.bias': torch.ones(2)}
        parent.load_state_dict(checkpoint, strict=False)
        torch.testing.assert_close(parent.backbone.weight, torch.ones(2, 2), atol=0, rtol=0)
        self.assertEqual(parent.init_decoder.size_representation, 'log')
        for key, value in before.items():
            torch.testing.assert_close(parent.init_decoder.G1.state_dict()[key], value, atol=0, rtol=0)

    def test_training_and_evaluation_configs_keep_linear_default_and_allow_log_override(self):
        root = Path(__file__).resolve().parents[1]
        generic = OmegaConf.load(root/'configs/model/smart.yaml').model_config.decoder.init_diffusion
        self.assertEqual(generic.size_representation, 'linear')
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                for representation in ('linear', 'log'):
                    config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                        f'model.model_config.decoder.init_diffusion.size_representation={representation}'])
                    self.assertEqual(config.model.model_config.decoder.init_diffusion.size_representation,
                                     representation)


if __name__ == '__main__':
    unittest.main()
