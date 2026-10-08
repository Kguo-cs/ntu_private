"""Optional Euclidean uniform sources share the supervised train/sample path."""

import copy
import itertools
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
from src.smart.diffusion.scale_flow import Flow, _noise_endpoint


class InitDiffusionUniformSourceTest(unittest.TestCase):
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
                      sampling_steps=2, use_rl=False, heading_noise='gaussian',
                      velocity_representation='vector', size_representation='linear')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor():
        return SimpleNamespace(use_refiner=False, learn_init=True, init_map_range=50.)

    @staticmethod
    def normalizer(width=8, dtype=torch.float32):
        mean = torch.arange(width, dtype=dtype)[None] / 3 - 1
        scale = torch.linspace(.25, 2., width, dtype=dtype)[None]
        return SimpleNamespace(normal_mean=mean, normal_scale=scale,
                               denormalize=lambda x: mean + scale * x)

    @staticmethod
    def inputs():
        state = torch.tensor([[2., 3., 1., 0., 4.5, 2., 5., 0.],
                              [0., 0., 1., 0., 4.8, 2.1, 0., 0.],
                              [-4., 7., 0., 1., .9, .6, 3., 0.]])
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 1, 2]),
                     ego_mask=torch.tensor([False, True, False]), num_graphs=1,
                     expert_input=state.clone(), shape=state[:, 4:6].clone(),
                     local_vel=state[:, 6:8].clone(), batch_ego_pos=torch.zeros(3, 2),
                     batch_ego_heading=torch.zeros(3), initial_pos=state[:, :2].clone(),
                     initial_heading=torch.atan2(state[:, 3], state[:, 2]),
                     token_traj=torch.zeros(3, 2, 4, 2),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return state, agent, feature

    def test_default_gaussian_is_exact_and_consumes_no_additional_random_numbers(self):
        model = self.normalizer()
        standard = torch.randn(16, 8)
        rng = torch.get_rng_state().clone()
        endpoint = _noise_endpoint(model, standard, None)
        torch.testing.assert_close(endpoint, model.denormalize(standard), atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        explicit = _noise_endpoint(model, standard, None, pos_source='gaussian',
                                   shape_source='gaussian', velocity_source='gaussian')
        torch.testing.assert_close(explicit, endpoint, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_each_source_changes_only_its_group_and_preserves_input(self):
        model = self.normalizer()
        standard = torch.full((256, 8), 7.)
        before = standard.clone()
        expected = model.denormalize(standard)
        for option, selected in (('pos_source', [0, 1]), ('shape_source', [4, 5]),
                                  ('velocity_source', [6, 7])):
            with self.subTest(source=option):
                endpoint = _noise_endpoint(model, standard, None, **{option: 'uniform'})
                unselected = [index for index in range(8) if index not in selected]
                torch.testing.assert_close(endpoint[:, unselected], expected[:, unselected], atol=0, rtol=0)
                normalized = (endpoint - model.normal_mean) / model.normal_scale
                self.assertTrue((normalized[:, selected].abs() <= math.sqrt(3) + 1.e-6).all())
                self.assertFalse(torch.equal(endpoint[:, selected], expected[:, selected]))
                torch.testing.assert_close(standard, before, atol=0, rtol=0)

    def test_uniform_has_unit_variance_bounded_independent_coordinates(self):
        model = self.normalizer(dtype=torch.float64)
        standard = torch.randn(30000, 8, dtype=torch.float64)
        endpoint = _noise_endpoint(model, standard, None, pos_source='uniform',
                                   shape_source='uniform', velocity_source='uniform')
        selected = [0, 1, 4, 5, 6, 7]
        normalized = ((endpoint - model.normal_mean) / model.normal_scale)[:, selected]
        self.assertTrue((normalized.abs() <= math.sqrt(3) + 1.e-12).all())
        torch.testing.assert_close(normalized.mean(0), torch.zeros(6, dtype=torch.float64), atol=.025, rtol=0)
        torch.testing.assert_close(normalized.std(0, unbiased=False), torch.ones(6, dtype=torch.float64), atol=.02, rtol=0)
        torch.testing.assert_close(torch.corrcoef(normalized.T), torch.eye(6, dtype=torch.float64), atol=.025, rtol=0)
        torch.testing.assert_close(endpoint[:, 2:4], model.denormalize(standard)[:, 2:4], atol=0, rtol=0)

    def test_velocity_source_supports_vector_and_scalar_speed_dimensions(self):
        for width in (7, 8):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(width=width, dtype=dtype):
                    model = self.normalizer(width, dtype)
                    standard = torch.full((128, width), 9., dtype=dtype)
                    endpoint = _noise_endpoint(model, standard, 5., velocity_source='uniform')
                    self.assertEqual(endpoint.shape, standard.shape)
                    self.assertEqual(endpoint.dtype, dtype)
                    normalized = (endpoint[:, 6:] - model.normal_mean[:, 6:]) / model.normal_scale[:, 6:]
                    self.assertTrue((normalized.abs() <= math.sqrt(3) + 1.e-6).all())
                    torch.testing.assert_close(endpoint[:, :2], model.denormalize(standard)[:, :2], atol=0, rtol=0)
                    torch.testing.assert_close(endpoint[:, 4:6], model.denormalize(standard)[:, 4:6], atol=0, rtol=0)
                    torch.testing.assert_close(endpoint[:, 2:4], standard[:, 2:4] * 5., atol=0, rtol=0)

    def test_uniform_euclidean_sources_preserve_circular_heading_law(self):
        model = self.normalizer()
        standard = torch.randn(8192, 8)
        rng = torch.get_rng_state().clone()
        endpoint = _noise_endpoint(model, standard, 17., 'circular', pos_source='uniform',
                                   shape_source='uniform', velocity_source='uniform')
        torch.set_rng_state(rng)
        without_sigma = _noise_endpoint(model, standard, None, 'circular', pos_source='uniform',
                                        shape_source='uniform', velocity_source='uniform')
        torch.testing.assert_close(endpoint, without_sigma, atol=0, rtol=0)
        heading = endpoint[:, 2:4]
        torch.testing.assert_close(heading.norm(dim=-1), torch.ones(len(standard)), atol=2.e-7, rtol=0)
        bins = torch.floor((torch.atan2(heading[:, 1], heading[:, 0]) + torch.pi) /
                           (2*torch.pi) * 16).long().clamp_max(15)
        torch.testing.assert_close(torch.bincount(bins, minlength=16).float(),
                                   torch.full((16,), len(standard)/16), atol=90., rtol=0)

    def test_empty_endpoint_keeps_shape_dtype_and_device(self):
        for width in (7, 8):
            model = self.normalizer(width, torch.float64)
            standard = torch.empty(0, width, dtype=torch.float64)
            endpoint = _noise_endpoint(model, standard, None, 'circular', pos_source='uniform',
                                       shape_source='uniform', velocity_source='uniform')
            self.assertEqual(endpoint.shape, standard.shape)
            self.assertEqual(endpoint.dtype, standard.dtype)
            self.assertEqual(endpoint.device, standard.device)

    def test_uniform_shape_stays_in_log_units_and_decodes_to_positive_sizes(self):
        model = InitDenoiser(None, input_dim=8, hidden_dim=32, output_dim=8,
                             num_layers=1, num_heads=2, dropout=0., size_representation='log')
        physical, _, _ = self.inputs()
        internal = model.state_to_model(physical)
        with torch.no_grad():
            model.normal_mean.copy_(internal.mean(0, keepdim=True))
            model.normal_scale.copy_(internal.std(0, unbiased=False, keepdim=True).clamp_min(1.e-6))
        standard = torch.zeros(256, 8)
        endpoint = _noise_endpoint(model, standard, None, shape_source='uniform')
        lo = model.normal_mean[:, 4:6] - math.sqrt(3) * model.normal_scale[:, 4:6]
        hi = model.normal_mean[:, 4:6] + math.sqrt(3) * model.normal_scale[:, 4:6]
        self.assertTrue((endpoint[:, 4:6] >= lo - 1.e-6).all())
        self.assertTrue((endpoint[:, 4:6] <= hi + 1.e-6).all())
        decoded = model.state_to_physical(endpoint)
        torch.testing.assert_close(decoded[:, 4:6], endpoint[:, 4:6].exp(), atol=0, rtol=0)
        self.assertTrue((decoded[:, 4:6] > 0).all())
        self.assertTrue((decoded[:, 4:6] >= lo.exp() - 1.e-6).all())
        self.assertTrue((decoded[:, 4:6] <= hi.exp() + 1.e-6).all())

    def test_train_and_generation_replay_the_same_sources_and_keep_ego_fixed(self):
        # Unique non-ego types ensure matching cannot permute the replayed rows.
        for representation, size, heading, sources in itertools.product(
                ('vector', 'speed'), ('linear', 'log'), ('gaussian', 'circular'),
                itertools.product(('gaussian', 'uniform'), repeat=3)):
            options = dict(velocity_representation=representation, size_representation=size,
                           heading_noise=heading, **dict(zip(('pos_source', 'shape_source', 'velocity_source'), sources)))
            with self.subTest(**options):
                flow = Flow(self.args(**options), self.processor(), False)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                eps = torch.linspace(-2., 2., clean.numel()).reshape_as(clean)
                torch.manual_seed(619)
                with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=eps.clone()), \
                        patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    source, time, latent = flow._prepare_supervised_batch(clean, agent)
                sampled_agent = copy.deepcopy(agent)
                torch.manual_seed(619)
                with patch('src.smart.diffusion.scale_flow.torch.randn', return_value=eps.clone()), \
                        patch.object(flow.model, 'forward', return_value=clean.clone()):
                    generated = flow.eval().sample(sampled_agent, feature, steps=2)
                non_ego = ~agent['ego_mask']
                torch.testing.assert_close(sampled_agent['gen_noise'][non_ego], source[non_ego], atol=0, rtol=0)
                torch.testing.assert_close(source[~non_ego], clean[~non_ego], atol=0, rtol=0)
                torch.testing.assert_close(generated[~non_ego], clean[~non_ego], atol=0, rtol=0)
                torch.testing.assert_close(time[~non_ego], torch.zeros(1, 1), atol=0, rtol=0)
                self.assertTrue(torch.isfinite(generated).all())
                if heading == 'circular':
                    torch.testing.assert_close(latent[:, 2:4].norm(dim=-1), torch.ones(3), atol=2.e-7, rtol=0)

    def test_real_supervised_loss_backpropagates_with_uniform_log_sources(self):
        flow = Flow(self.args(pos_source='uniform', shape_source='uniform', velocity_source='uniform',
                              size_representation='log', heading_noise='circular', speed_loss_weight=.5),
                    self.processor(), False).train()
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = flow._supervised_loss(clean, agent, feature)
        self.assertEqual(len(losses), 6)
        objective = losses[0].mean() + losses[1]
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        gradients = [parameter.grad for parameter in flow.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_wrapper_passes_independent_sources_and_rejects_invalid_values(self):
        options = dict(pos_source='uniform', shape_source='gaussian', velocity_source='uniform')
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            model = InitDiffusion(32, 2, 4, self.processor(), False, **options)
        for option, expected in options.items():
            self.assertEqual(getattr(model, option), expected)
            self.assertEqual(getattr(model.G1, option), expected)
        for option in options:
            for value in ('normal', 'unform', '', None):
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    Flow(self.args(**{option: value}), self.processor(), False)
                with patch.object(InitDiffusion, '_make_args', return_value=self.args()), \
                        self.subTest(wrapper_option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    InitDiffusion(32, 2, 4, self.processor(), False, **{option: value})

    def test_uniform_sources_reject_gaussian_score_sde_path(self):
        for option in ('pos_source', 'shape_source', 'velocity_source'):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, '[Uu]niform|gaussian|Gaussian'):
                Flow(self.args(**{option: 'uniform'}), self.processor(), True)
        self.assertTrue(Flow(self.args(), self.processor(), True).use_sde)

    def test_source_option_does_not_change_parameter_or_buffer_shapes(self):
        gaussian = Flow(self.args(), self.processor(), False)
        uniform = Flow(self.args(pos_source='uniform', shape_source='uniform', velocity_source='uniform'),
                       self.processor(), False)
        gaussian_state, uniform_state = gaussian.state_dict(), uniform.state_dict()
        self.assertEqual(set(gaussian_state), set(uniform_state))
        for key in gaussian_state:
            self.assertEqual(gaussian_state[key].shape, uniform_state[key].shape)
        uniform.load_state_dict(gaussian_state, strict=True)

    def test_train_eval_config_supports_independent_uniform_overrides(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                defaults = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                for option in ('pos_source', 'shape_source', 'velocity_source'):
                    self.assertEqual(defaults.model.model_config.decoder.init_diffusion[option], 'gaussian')
                overrides = [f'experiment={experiment}',
                             'model.model_config.decoder.init_diffusion.pos_source=uniform',
                             'model.model_config.decoder.init_diffusion.shape_source=uniform',
                             'model.model_config.decoder.init_diffusion.velocity_source=gaussian']
                config = compose(config_name='run.yaml', overrides=overrides)
                actual = config.model.model_config.decoder.init_diffusion
                self.assertEqual(actual.pos_source, 'uniform')
                self.assertEqual(actual.shape_source, 'uniform')
                self.assertEqual(actual.velocity_source, 'gaussian')


if __name__ == '__main__':
    unittest.main()
