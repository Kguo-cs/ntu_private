"""Scenario Dreamer timestep features preserve InitDiffusion flow objectives."""

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
from torch import nn

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
from src.smart.scenario_dreamer.core.dit_layers import TimestepEmbedder


class InitDiffusionTimeEmbeddingTest(unittest.TestCase):
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
                      sampling_steps=4, use_rl=False, time_embedding_type='scenario_dreamer',
                      time_embedding_scale=99., heading_noise='circular',
                      heading_objective='x0', velocity_representation='vector')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50.)
        values.update(options)
        return SimpleNamespace(**values)

    def denoiser(self, **options):
        values = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                      num_layers=1, num_heads=2, dropout=0.,
                      time_embedding_type='scenario_dreamer', time_embedding_scale=99.)
        values.update(options)
        return InitDenoiser(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        values = dict(time_embedding_type='scenario_dreamer', time_embedding_scale=99.,
                      heading_noise='circular', heading_objective='angular_velocity',
                      velocity_representation='speed')
        values.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **values)

    @staticmethod
    def inputs(mode='vector'):
        vector = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                               [0., 0., 1., 0., 4.5, 2., 0., 0.],
                               [-20., 30., 0., 1., 4., 1.8, -2., 0.]])
        clean = (torch.cat((vector[:, :6], vector[:, 6:8].norm(dim=-1, keepdim=True)), -1)
                 if mode == 'speed' else vector)
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 1, 2]),
                     num_graphs=1, ego_mask=torch.tensor([False, True, False]),
                     expert_input=clean.clone(), local_vel=vector[:, 6:8].clone(),
                     batch_ego_pos=torch.zeros(3, 2), batch_ego_heading=torch.zeros(3),
                     initial_pos=vector[:, :2].clone(),
                     initial_heading=torch.atan2(vector[:, 3], vector[:, 2]),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return clean, agent, feature

    def test_reference_module_formula_endpoints_and_fractional_times_match(self):
        model = self.denoiser()
        self.assertIsInstance(model.noise_embedding, TimestepEmbedder)
        self.assertEqual(model.noise_embedding.frequency_embedding_size, 256)
        self.assertEqual([type(layer) for layer in model.noise_embedding.mlp],
                         [nn.Linear, nn.SiLU, nn.Linear])
        reference = TimestepEmbedder(32)
        reference.load_state_dict(model.noise_embedding.state_dict(), strict=True)
        beta = torch.tensor([0., .005, .25, .501, 1.])
        expected = reference(beta * 99.)
        torch.testing.assert_close(model._embed_time(beta, len(beta)), expected, atol=0, rtol=0)
        # Both convention and continuous fractional indices matter, not merely
        # the class name: a flipped time or integer rounding must fail this.
        self.assertFalse(torch.allclose(expected, reference((1-beta) * 99.)))
        self.assertFalse(torch.allclose(expected, reference((beta * 99.).round())))
        self.assertGreater(torch.pdist(expected).min().item(), 1.e-4)

    def test_nondefault_scale_is_used_without_changing_flow_time(self):
        model = self.denoiser(time_embedding_scale=37.)
        beta = torch.tensor([[0.], [.123], [1.]])
        expected = model.noise_embedding(beta[:, 0] * 37.)
        torch.testing.assert_close(model._embed_time(beta, 3), expected, atol=0, rtol=0)
        flow = self.flow(time_embedding_scale=37.)
        self.assertEqual(flow.model.time_embedding_scale, 37.)
        clean, agent, _ = self.inputs()
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            _, time, _ = flow._prepare_supervised_batch(clean, agent)
        torch.testing.assert_close(time, torch.tensor([[.4], [0.], [.4]]))

    def test_supported_scalar_time_layouts_agree_for_both_state_representations(self):
        time = torch.tensor([0., .125, .5, 1.])
        for state_dim, representation in ((8, 'vector'), (7, 'speed')):
            model = self.denoiser(input_dim=state_dim, output_dim=state_dim,
                                  velocity_representation=representation)
            expected = model._embed_time(time, 4)
            for layout in (time[:, None], time[:, None].expand(-1, state_dim),
                           time[:, None].repeat(1, state_dim), time[:, None, None],
                           time[:, None, None].expand(-1, 1, state_dim),
                           time[:, None, None].repeat(1, 1, state_dim)):
                with self.subTest(state_dim=state_dim, shape=tuple(layout.shape)):
                    torch.testing.assert_close(model._embed_time(layout, 4), expected,
                                               atol=0, rtol=0)
            model.eval()
            torch.testing.assert_close(model._embed_time(time, 4), expected, atol=0, rtol=0)

    def test_scenario_dreamer_rejects_nonuniform_group_times_and_invalid_layouts(self):
        model = self.denoiser()
        for invalid in (torch.tensor(.4), torch.ones(2, 1), torch.ones(3, 2),
                        torch.ones(3, 2, 8), torch.ones(3, 1, 1, 1)):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                model._embed_time(invalid, 3)
        grouped = torch.full((3, 8), .4)
        grouped[1, 5] = .5
        for invalid in (grouped, grouped[:, None]):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                model._embed_time(invalid, 3)

    def test_full_forward_preserves_time_shape_validation(self):
        model = self.denoiser()
        clean, agent, feature = self.inputs()
        for invalid in (torch.full((3, 2, 8), .4), torch.full((3, 1, 1, 1), .4)):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                model(clean, invalid, agent, feature)

    def test_expanded_scalar_time_does_not_require_consistency_synchronization(self):
        model = self.denoiser()
        time = torch.tensor([0., .4, 1.])
        expected = model._embed_time(time, 3)
        # torch.equal on CUDA would synchronize; standard scalar time uses an
        # expanded stride-zero view and needs no such per-batch check.
        for layout in (time, time[:, None], time[:, None].expand(-1, 8)):
            with self.subTest(shape=tuple(layout.shape)), \
                    patch('src.smart.diffusion.denoiser.torch.equal',
                          side_effect=AssertionError('unnecessary consistency check')):
                actual = model._embed_time(layout, 3)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    def test_invalid_modes_and_scales_are_rejected_at_public_entry_points(self):
        invalid_options = [{'time_embedding_type': 'unknown'}]
        invalid_options += [{'time_embedding_scale': value}
                            for value in (0., -1., float('nan'), float('inf'))]
        for options in invalid_options:
            for constructor in (lambda: self.denoiser(**options),
                                lambda: self.flow(**options),
                                lambda: self.wrapper(**options)):
                with self.subTest(options=options, constructor=constructor), self.assertRaises(ValueError):
                    constructor()

    def test_reference_initialization_survives_flow_outer_weight_initialization(self):
        flow = Flow(self.args(heading_noise='gaussian', time_embedding_scale=37.),
                    self.processor(use_refiner=True), False)
        for model in (self.denoiser(), flow.model, flow.refine_model):
            with self.subTest(model=type(model).__name__):
                self.assertIsInstance(model.noise_embedding, TimestepEmbedder)
                for layer in (model.noise_embedding.mlp[0], model.noise_embedding.mlp[2]):
                    self.assertAlmostEqual(layer.weight.std(unbiased=False).item(), .02, delta=.002)
                    self.assertLess(abs(layer.weight.mean().item()), .002)
                    torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), atol=0, rtol=0)
        self.assertEqual(flow.model.time_embedding_scale, 37.)
        self.assertEqual(flow.refine_model.time_embedding_scale, 37.)
        self.assertEqual(flow.refine_model.time_embedding_type, 'scenario_dreamer')

    def test_legacy_default_formula_parameter_layout_and_checkpoint_are_unchanged(self):
        options = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                       num_layers=1, num_heads=2, dropout=0.)
        default = InitDenoiser(**options)
        explicit = InitDenoiser(**options, time_embedding_type='legacy')
        self.assertEqual(default.time_embedding_type, 'legacy')
        state = default.state_dict()
        self.assertFalse(any(key.startswith('_time_') for key in state))
        self.assertEqual(tuple(state['noise_embedding.mlp.0.weight'].shape), (32, 8))
        self.assertEqual(tuple(state['noise_embedding.mlp.3.weight'].shape), (32, 32))
        self.assertIn('noise_embedding.mlp.1.weight', state)
        incompatible = explicit.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        beta = torch.tensor([0., .123, .5, 1.])
        dims = torch.arange(8)
        phase = (1-beta[:, None]) * (math.pi * 2. ** (dims // 2))
        features = torch.where(dims.remainder(2).bool(), phase.cos(), phase.sin())
        expected = default.noise_embedding(features)
        torch.testing.assert_close(default._embed_time(beta, 4), expected, atol=0, rtol=0)
        torch.testing.assert_close(explicit._embed_time(beta, 4), expected, atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            self.denoiser().load_state_dict(state, strict=True)

    def test_embedding_dtype_and_time_gradients_match_reference(self):
        for dtype in (torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.denoiser().to(dtype=dtype)
                time = torch.tensor([0., .25, .75, 1.], dtype=dtype, requires_grad=True)
                actual = model._embed_time(time, 4)
                features = TimestepEmbedder.timestep_embedding(time * 99., 256).to(dtype=dtype)
                expected = model.noise_embedding.mlp(features)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                actual.square().sum().backward()
                self.assertIsNotNone(time.grad)
                self.assertTrue(torch.isfinite(time.grad).all())
                self.assertGreater(time.grad.abs().max().item(), 0.)
                for parameter in model.noise_embedding.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_full_denoiser_time_gradients_and_train_sampling_for_existing_objectives(self):
        for representation in ('vector', 'speed'):
            for objective in ('x0', 'angular_velocity'):
                with self.subTest(representation=representation, objective=objective):
                    flow = self.flow(velocity_representation=representation,
                                     heading_objective=objective)
                    clean, agent, feature = self.inputs(representation)
                    time = torch.tensor([[.4], [0.], [.4]], requires_grad=True)
                    prediction = flow.model(clean, time, agent, feature)
                    self.assertEqual(prediction.shape,
                                     (3, clean.shape[1] + (objective == 'angular_velocity')))
                    prediction.square().mean().backward()
                    self.assertTrue(torch.isfinite(time.grad).all())
                    self.assertGreater(time.grad.abs().max().item(), 0.)
                    flow.zero_grad(set_to_none=True)
                    before = flow.model.noise_embedding.mlp[0].weight.detach().clone()
                    optimizer = torch.optim.Adam(flow.parameters(), lr=1.e-3)
                    with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                        losses = flow._supervised_loss(clean, agent, feature)
                    total = losses[0].mean() + losses[1]
                    self.assertTrue(torch.isfinite(total))
                    total.backward()
                    time_parameters = list(flow.model.noise_embedding.parameters())
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in time_parameters))
                    self.assertTrue(any(p.grad.abs().sum() > 0 for p in time_parameters))
                    optimizer.step()
                    self.assertFalse(torch.equal(before, flow.model.noise_embedding.mlp[0].weight))
                    flow.eval()
                    observed_time = []

                    def record_time(module, args):
                        observed_time.append(args[1].detach().clone())

                    hook = flow.model.register_forward_pre_hook(record_time)
                    try:
                        generated = flow.sample(agent, feature, steps=4)
                    finally:
                        hook.remove()
                    self.assertEqual(generated.shape, clean.shape)
                    self.assertTrue(torch.isfinite(generated).all())
                    torch.testing.assert_close(generated[agent['ego_mask']],
                                               clean[agent['ego_mask']], atol=0, rtol=0)
                    self.assertEqual(len(observed_time), 4)
                    self.assertEqual(observed_time[0][0].item(), 1.)
                    self.assertTrue(all(t[agent['ego_mask']].eq(0).all() for t in observed_time))

    def test_wrapper_ema_roundtrip_averages_new_time_parameters_and_restores_online_weights(self):
        source = self.wrapper(use_ema=True, ema_decay=.9)
        self.assertEqual(source.time_embedding_type, 'scenario_dreamer')
        self.assertEqual(source.G1.model.time_embedding_type, 'scenario_dreamer')
        self.assertEqual(source.G1.model.time_embedding_scale, 99.)
        names = list(dict(source.G1.named_parameters()))
        time_names = [name for name in names if name.startswith('model.noise_embedding.')]
        self.assertEqual(len(time_names), 4)
        self.assertTrue(all(name in source.get_extra_state()['ema_parameter_names'] for name in time_names))
        with torch.no_grad():
            for parameter in source.G1.model.noise_embedding.parameters():
                parameter.add_(.05)
        source.update_ema()
        saved = io.BytesIO()
        torch.save(source.state_dict(), saved)
        saved.seek(0)
        target = self.wrapper(use_ema=True, ema_decay=.7)
        incompatible = target.load_state_dict(torch.load(saved, weights_only=False), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(target.ema.decay, .9)
        self.assertEqual(target.ema.num_updates, 1)
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        online_before = copy.deepcopy(target.G1.state_dict())
        beta = torch.tensor([0., .4, 1.])
        with source.ema.average_parameters(source.G1.parameters()):
            expected = source.G1.model._embed_time(beta, 3).detach().clone()
        with target.ema.average_parameters(target.G1.parameters()):
            torch.testing.assert_close(target.G1.model._embed_time(beta, 3), expected, atol=0, rtol=0)
        for key, value in online_before.items():
            torch.testing.assert_close(target.G1.state_dict()[key], value, atol=0, rtol=0)

    def test_training_and_evaluation_configs_allow_same_optional_embedding(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                for mode in ('legacy', 'scenario_dreamer'):
                    overrides = [f'experiment={experiment}']
                    if mode != 'legacy':
                        overrides += ['model.model_config.decoder.init_diffusion.time_embedding_type=scenario_dreamer']
                    config = compose(config_name='run.yaml', overrides=overrides)
                    options = config.model.model_config.decoder.init_diffusion
                    self.assertEqual(options.time_embedding_type, mode)
                    self.assertEqual(options.time_embedding_scale, 99.)


if __name__ == '__main__':
    unittest.main()
