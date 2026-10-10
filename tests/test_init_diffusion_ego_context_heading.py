"""Continuous ego heading context preserves legacy models and checkpoint semantics."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_type_generation as fixture_module


class InitDiffusionEgoContextHeadingTest(unittest.TestCase):
    fixtures = fixture_module.InitDiffusionTypeGenerationTest

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(627)

    def flow(self, **options):
        return Flow(self.fixtures.args(**options), self.fixtures.processor(), False)

    def wrapper(self, **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.fixtures.args()):
            return InitDiffusion(32, 2, 4, self.fixtures.processor(), False, **options)

    def denoiser(self, **options):
        values = dict(token_processor=self.fixtures.processor(), input_dim=8,
                      hidden_dim=32, output_dim=8, num_layers=1, num_heads=2, dropout=0.)
        values.update(options)
        return InitDenoiser(**values)

    def test_option_validates_at_public_layers_and_reaches_refiner(self):
        for factory in (self.wrapper, self.flow, self.denoiser):
            for invalid in ('fourier', '', None, True):
                with self.subTest(factory=factory.__name__, value=invalid), \
                        self.assertRaisesRegex(ValueError, 'ego_context_heading_encoding'):
                    factory(ego_context_heading_encoding=invalid)
        model = self.wrapper(ego_context_heading_encoding='sincos')
        for layer in (model, model.G1, model.G1.model):
            self.assertEqual(layer.ego_context_heading_encoding, 'sincos')
        self.assertEqual(model.G1.model.ego_embed.mlp[0].in_features, 15)
        refiner = Flow(self.fixtures.args(ego_context_heading_encoding='sincos'),
                       self.fixtures.processor(use_refiner=True), False)
        self.assertEqual(refiner.refine_model.ego_context_heading_encoding, 'sincos')
        self.assertEqual(refiner.refine_model.ego_embed.mlp[0].in_features, 15)

    def test_reference_geometry_heading_order_and_counts(self):
        model = self.denoiser(ego_context_heading_encoding='sincos')
        model.ego_embed = nn.Identity()
        heads = torch.tensor([.2, -.8, 2.4])
        agent = dict(ego_feat=torch.tensor([[1., 3., .2, 4., -1., -.8,
                                           2., 5., 2.4, 2., 4., 6.]]))
        position = torch.tensor([[1., 2.], [-2., 3.]])
        heading = torch.tensor([0., math.pi/2])
        feature = model._ego_context_embedding(position, heading,
                                                torch.zeros(2, dtype=torch.long), agent)
        expected_position = torch.tensor([[0., 1., 3., -3., 1., 3.],
                                          [0., -3., -4., -6., 2., -4.]])
        relative = heads[None]-heading[:, None]
        expected = torch.cat((expected_position, relative.cos(), relative.sin(),
                              torch.tensor([[2., 4., 6.], [2., 4., 6.]])), -1)
        self.assertEqual(tuple(feature.shape), (2, 15))
        torch.testing.assert_close(feature, expected, atol=1.e-6, rtol=0)

    def test_seam_is_continuous_periodic_and_has_finite_heading_gradient(self):
        agent = dict(ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0.,
                                           0., 0., 0., 3., 0., 0.]]))
        position = torch.zeros(2, 2)
        batch = torch.zeros(2, dtype=torch.long)
        heading = torch.tensor([math.pi-1.e-4, -math.pi+1.e-4], requires_grad=True)
        model = self.denoiser(ego_context_heading_encoding='sincos')
        model.ego_embed = nn.Identity()
        feature = model._ego_context_embedding(position, heading, batch, agent)
        self.assertLess((feature[0]-feature[1]).abs().max().item(), 2.1e-4)
        periodic = model._ego_context_embedding(position, heading+2*math.pi, batch, agent)
        torch.testing.assert_close(periodic, feature, atol=1.e-6, rtol=0)
        feature[:, 9:12].sum().backward()
        self.assertTrue(torch.isfinite(heading.grad).all())
        torch.testing.assert_close(heading.grad, torch.full((2,), 3.), atol=1.e-5, rtol=0)
        legacy = self.denoiser(ego_context_heading_encoding='angle')
        legacy.ego_embed = nn.Identity()
        old_features = legacy._ego_context_embedding(position, heading.detach(), batch, agent)
        self.assertEqual(tuple(old_features.shape), (2, 12))
        self.assertGreater((old_features[0, 6:9]-old_features[1, 6:9]).abs().min().item(), 6.28)

    def test_default_angle_preserves_rng_parameters_and_forward(self):
        torch.manual_seed(181)
        default = self.flow(fix_ego=False)
        default_next = torch.rand(5)
        torch.manual_seed(181)
        explicit = self.flow(fix_ego=False, ego_context_heading_encoding='angle')
        explicit_next = torch.rand(5)
        self.assertEqual(default.ego_context_heading_encoding, 'angle')
        self.assertEqual(default.model.ego_embed.mlp[0].in_features, 12)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(default_next, explicit_next, atol=0, rtol=0)
        clean, agent, feature = self.fixtures.inputs()
        default.eval()
        explicit.eval()
        with torch.no_grad():
            first = default.model(clean, torch.full((3, 1), .4), copy.deepcopy(agent), feature)
            second = explicit.model(clean, torch.full((3, 1), .4), copy.deepcopy(agent), feature)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_actual_supervised_backward_and_sampling_speed_log_type_partial_ego(self):
        options = dict(ego_context_heading_encoding='sincos', use_ego_embedding=True,
                       fix_ego=False, fix_ego_position=True, fix_ego_heading=False,
                       fix_ego_shape=True, fix_ego_velocity=False, fix_ego_type=True,
                       generate_type=True, velocity_representation='speed',
                       size_representation='log', heading_x0_loss='angle_mse')
        flow = self.flow(**options).train()
        _, agent, feature = self.fixtures.inputs()
        clean, _ = flow.model.get_input(agent)
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = flow._supervised_loss(clean, agent, feature)
        total = losses[0].mean()+losses[1]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        context_gradient = flow.model.ego_embed.mlp[0].weight.grad
        self.assertIsNotNone(context_gradient)
        self.assertTrue(torch.isfinite(context_gradient).all())
        self.assertGreater(context_gradient.abs().sum().item(), 0.)
        self.assertGreater(context_gradient[:, 6:12].abs().sum().item(), 0.)
        with torch.no_grad():
            _, sample_agent, sample_feature = self.fixtures.inputs()
            truth, _ = flow.model.get_input(sample_agent)
            generated = flow.eval().sample(sample_agent, sample_feature, steps=2)
        self.assertEqual(tuple(generated.shape), (3, 7))
        self.assertTrue(torch.isfinite(generated).all())
        mask = sample_agent['ego_mask']
        torch.testing.assert_close(generated[mask, :2], truth[mask, :2], atol=0, rtol=0)
        torch.testing.assert_close(generated[mask, 4:6], truth[mask, 4:6], atol=0, rtol=0)

    def test_same_mode_strict_checkpoint_restores_context_weights_and_ema(self):
        options = dict(ego_context_heading_encoding='sincos', use_ema=True,
                       fix_ego=False, use_ego_embedding=True)
        source = self.wrapper(**options)
        target = self.wrapper(**options)
        self.assertEqual(source.get_extra_state()['ego_context_heading_encoding'], 'sincos')
        with torch.no_grad():
            source.G1.model.ego_embed.mlp[0].weight.add_(.2)
        source.update_ema()
        incompatible = target.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertEqual(source.ema.num_updates, target.ema.num_updates)
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name, expected in source.G1.state_dict().items():
            torch.testing.assert_close(target.G1.state_dict()[name], expected, atol=0, rtol=0)
        _, agent, feature = self.fixtures.inputs()
        with torch.no_grad():
            torch.manual_seed(409)
            first = source.eval()._infer(copy.deepcopy(agent), feature)
            torch.manual_seed(409)
            second = target.eval()._infer(copy.deepcopy(agent), feature)
        for actual, expected in zip(second, first):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_checkpoint_encoding_changes_are_rejected_even_non_strict(self):
        for original, requested in (('angle', 'sincos'), ('sincos', 'angle')):
            for strict in (True, False):
                with self.subTest(original=original, requested=requested, strict=strict):
                    source = self.wrapper(ego_context_heading_encoding=original, use_ema=True)
                    target = self.wrapper(ego_context_heading_encoding=requested, use_ema=True)
                    with self.assertRaisesRegex(ValueError, 'ego_context_heading_encoding'):
                        target.load_state_dict(source.state_dict(), strict=strict)
                    denoiser = self.denoiser(ego_context_heading_encoding=requested)
                    old_denoiser = self.denoiser(ego_context_heading_encoding=original)
                    with self.assertRaisesRegex(RuntimeError, 'ego_context_heading_encoding'):
                        denoiser.load_state_dict(old_denoiser.state_dict(), strict=strict)

    def test_legacy_metadata_defaults_angle_and_backbone_only_initializes_sincos(self):
        source = self.wrapper(use_ema=True)
        for omit_extra_state in (True, False):
            with self.subTest(omit_extra_state=omit_extra_state):
                state = copy.deepcopy(source.state_dict())
                if omit_extra_state:
                    state.pop('_extra_state')
                else:
                    state['_extra_state'].pop('ego_context_heading_encoding')
                restored = self.wrapper(use_ema=True)
                restored.load_state_dict(state, strict=True)
                new_model = self.wrapper(ego_context_heading_encoding='sincos', use_ema=True)
                with self.assertRaisesRegex((ValueError, RuntimeError), 'ego_context_heading_encoding'):
                    new_model.load_state_dict(state, strict=False)
        fresh = self.wrapper(ego_context_heading_encoding='sincos', use_ema=True)
        state_before = copy.deepcopy(fresh.G1.state_dict())
        incompatible = fresh.load_state_dict({}, strict=False)
        self.assertTrue(incompatible.missing_keys)
        for name, expected in state_before.items():
            torch.testing.assert_close(fresh.G1.state_dict()[name], expected, atol=0, rtol=0)
        for shadow, parameter in zip(fresh.ema.shadow_params, fresh.G1.parameters()):
            torch.testing.assert_close(shadow, parameter, atol=0, rtol=0)

    def test_train_and_eval_configs_enable_sincos_with_explicit_legacy_override(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        prefix = 'model.model_config.decoder.init_diffusion.ego_context_heading_encoding'
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                with self.subTest(experiment=experiment):
                    configured = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                    self.assertEqual(OmegaConf.select(configured, prefix), 'sincos')
                    legacy = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                                                                       f'{prefix}=angle'])
                    self.assertEqual(OmegaConf.select(legacy, prefix), 'angle')


if __name__ == '__main__':
    unittest.main()
