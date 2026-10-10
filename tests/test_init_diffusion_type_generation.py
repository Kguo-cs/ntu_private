"""Generated categorical agent state must not use GT labels as conditioning."""

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
import torch.nn.functional as F

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow


class InitDiffusionTypeGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(309)

    @staticmethod
    def args(**options):
        values = dict(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                      num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                      sampling_steps=2, use_rl=False, heading_noise='circular',
                      heading_objective='x0', velocity_representation='vector',
                      size_representation='linear')
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def library(types):
        """Two class-specific trajectories with distinct endpoint velocities."""
        types = types.long()
        shape = torch.stack((4.+types.float(), 1.+types.float()), -1)
        all_tokens = torch.zeros(len(types), 2, 2, 4, 2)
        all_tokens[:, 0, -1, :, 0] = (types.float()+1.)[:, None]
        all_tokens[:, 1, -1, :, 0] = (types.float()+4.)[:, None]
        return shape, all_tokens, all_tokens[:, :, -1].contiguous()

    @classmethod
    def processor(cls, **options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2)/dt,
                      _get_agent_tokens=cls.library)
        values.update(options)
        return SimpleNamespace(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    @classmethod
    def inputs(cls):
        state = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                              [0., 0., 1., 0., 4.8, 2.1, 1., 2.],
                              [-20., 30., 0., 1., .9, .6, -2., 0.]])
        types = torch.tensor([0, 1, 2])
        token_shapes, all_tokens, final_tokens = cls.library(types)
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=types,
                     ego_mask=torch.tensor([False, True, False]), num_graphs=1,
                     expert_input=state.clone(), shape=state[:, 4:6].clone(),
                     local_vel=state[:, 6:8].clone(), batch_ego_pos=torch.zeros(3, 2),
                     batch_ego_heading=torch.zeros(3), initial_pos=state[:, :2].clone(),
                     initial_heading=torch.atan2(state[:, 3], state[:, 2]),
                     token_agent_shape=token_shapes, token_traj=final_tokens,
                     token_traj_all=all_tokens,
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return state, agent, feature

    def wrapper(self, **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **options)

    def test_option_validation_and_unsupported_training_paths(self):
        for invalid in ('false', 0, 1, None):
            with self.subTest(option=invalid), self.assertRaisesRegex(ValueError, 'generate_type'):
                self.flow(generate_type=invalid)
            with self.subTest(wrapper=invalid), self.assertRaisesRegex(ValueError, 'generate_type'):
                self.wrapper(generate_type=invalid)
        for invalid in (-1., math.inf, math.nan):
            with self.subTest(weight=invalid), self.assertRaisesRegex(ValueError, 'type_loss_weight'):
                self.flow(generate_type=True, type_loss_weight=invalid)
        for options, processor, gail in ((dict(heading_noise='gaussian'), self.processor(), True),
                                         (dict(use_rl=True), self.processor(), False),
                                         ({}, self.processor(use_refiner=True), False)):
            with self.subTest(options=options, gail=gail), self.assertRaisesRegex(ValueError, 'generate_type'):
                Flow(self.args(generate_type=True, **options), processor, gail)

    def test_default_preserves_legacy_model_layout_and_random_sequence(self):
        torch.manual_seed(193)
        legacy = self.flow()
        torch.manual_seed(193)
        disabled = self.flow(generate_type=False)
        self.assertFalse(legacy.generate_type)
        self.assertFalse(hasattr(legacy.model, 'to_out_type'))
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        for name, parameter in legacy.state_dict().items():
            torch.testing.assert_close(parameter, disabled.state_dict()[name], atol=0, rtol=0)
        clean, agent, _ = self.inputs()
        torch.manual_seed(720)
        first = legacy._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_first = torch.rand(5)
        torch.manual_seed(720)
        second = disabled._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_second = torch.rand(5)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(next_first, next_second, atol=0, rtol=0)

    def test_type_interpolation_uses_same_time_and_keeps_continuous_dimensions(self):
        for representation, fixed in itertools.product(('vector', 'speed'), (True, False)):
            with self.subTest(velocity=representation, fix_ego=fixed):
                flow = self.flow(generate_type=True, fix_ego=fixed,
                                 velocity_representation=representation)
                _, agent, _ = self.inputs()
                clean, _ = flow.model.get_input(agent)
                times = torch.tensor([[0.], [.4], [1.]])
                with patch.object(flow, '_sample_time', return_value=times.clone()):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                self.assertEqual(latent.shape[-1], 7 if representation == 'speed' else 8)
                categorical = agent['_init_diffusion_type_state']
                self.assertEqual(categorical.shape, (3, 3))
                labels = F.one_hot(agent['type'], 3).to(categorical)
                torch.testing.assert_close(categorical[0], labels[0], atol=0, rtol=0)
                if fixed:
                    torch.testing.assert_close(categorical[1], labels[1], atol=0, rtol=0)
                    self.assertEqual(time[1].item(), 0.)
                else:
                    self.assertFalse(torch.equal(categorical[1], labels[1]))
                self.assertTrue(torch.isfinite(categorical).all())
                # At t=1 this is the unconstrained source, not a categorical label.
                self.assertFalse(torch.equal(categorical[2], labels[2]))
                self.assertFalse(torch.equal(categorical[2], categorical[2].softmax(-1)))

    def test_separate_type_generation_preserves_gt_type_grouped_matching(self):
        clean, agent, _ = self.inputs()
        for enabled in (True, False):
            flow = self.flow(generate_type=enabled, fix_ego=False)
            with patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                       return_value=torch.arange(2)) as match:
                flow._sample_noise(clean, agent)
            self.assertFalse(match.call_args.kwargs.get('use_all_type', False))
            self.assertEqual(match.call_args.args[0].shape[0], 2)
            torch.testing.assert_close(match.call_args.args[2]['batch'], agent['batch'][~agent['ego_mask']])
            torch.testing.assert_close(match.call_args.args[2]['type'], agent['type'][~agent['ego_mask']])
        fixed = self.flow(generate_type=True, fix_ego=True)
        with patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                   return_value=torch.arange(2)) as match:
            fixed._sample_noise(clean, agent)
        self.assertEqual(match.call_args.args[0].shape[0], 2)
        self.assertFalse(match.call_args.kwargs['use_all_type'])

    def test_type_ce_is_active_agent_mean_with_fixed_ego_and_time_boundary_masks(self):
        for fixed in (True, False):
            flow = self.flow(generate_type=True, fix_ego=fixed, type_loss_weight=.7)
            clean, agent, feature = self.inputs()
            logits = torch.tensor([[1., 0., -1.], [0., -1., 1.], [.2, -.1, .4]], requires_grad=True)
            times = torch.tensor([[.2], [.4], [1.]])
            def predict(latent, time, current, *args, **kwargs):
                current['_init_diffusion_type_logits'] = logits
                return clean.clone()
            zeros = (torch.zeros(3), torch.zeros(()), *(torch.zeros(3) for _ in range(4)))
            with patch.object(flow, '_sample_time', return_value=times.clone()), \
                    patch.object(flow.model, 'forward', side_effect=predict), \
                    patch('src.smart.diffusion.scale_flow.get_diff_loss', return_value=zeros):
                losses = flow._supervised_loss(clean, agent, feature)
            active = torch.tensor([True, not fixed, False])
            expected = F.cross_entropy(logits[active], agent['type'][active])
            metrics = agent['_init_diffusion_type_metrics']
            torch.testing.assert_close(metrics['loss'], expected.detach())
            torch.testing.assert_close(metrics['weighted_loss'], .7*expected.detach())
            torch.testing.assert_close(losses[0], (.7*expected).expand(3))
            self.assertEqual(len(losses), 6)
            losses[0].mean().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            torch.testing.assert_close(logits.grad[~active], torch.zeros_like(logits.grad[~active]), atol=0, rtol=0)
            self.assertTrue((logits.grad[active].abs().sum(-1) > 0).all())
            expected_accuracy = (logits[active].argmax(-1) == agent['type'][active]).float().mean()
            torch.testing.assert_close(metrics['accuracy'], expected_accuracy)

    def test_no_active_type_target_has_finite_differentiable_zero_loss(self):
        flow = self.flow(generate_type=True, fix_ego=False)
        clean, agent, feature = self.inputs()
        logits = torch.randn(3, 3, requires_grad=True)
        def predict(latent, time, current, *args, **kwargs):
            current['_init_diffusion_type_logits'] = logits
            return clean.clone()
        with patch.object(flow, '_sample_time', return_value=torch.tensor([[0.], [1.], [0.]])), \
                patch.object(flow.model, 'forward', side_effect=predict):
            losses = flow._supervised_loss(clean, agent, feature)
        self.assertEqual(agent['_init_diffusion_type_metrics']['loss'].item(), 0.)
        self.assertTrue(torch.isfinite(losses[0]).all())
        losses[0].mean().backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)

    def test_real_denoiser_ignores_gt_types_and_counts_when_type_state_is_fixed(self):
        for conditioned in (False, True):
            with self.subTest(sd_count_map=conditioned):
                options = dict(count_embedding_type='scenario_dreamer',
                               map_embedding_type='scenario_dreamer', map_id=1,
                               map_label_dropout=0.) if conditioned else {}
                flow = self.flow(generate_type=True, fix_ego=False, **options).eval()
                clean, agent, feature = self.inputs()
                state = torch.tensor([[-.2, .7, .1], [.3, -.4, .1], [.1, .2, .5]])
                first = copy.deepcopy(agent)
                second = copy.deepcopy(agent)
                first['_init_diffusion_type_state'] = state.clone()
                second['_init_diffusion_type_state'] = state.clone()
                second['type'] = torch.tensor([2, 0, 0])
                second['ego_feat'][:, -3:] = torch.tensor([[9., 1., 0.]])
                with torch.no_grad():
                    a = flow.model(clean, torch.full((3, 1), .4), first, feature)
                    b = flow.model(clean, torch.full((3, 1), .4), second, feature)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(first['_init_diffusion_type_logits'], second['_init_diffusion_type_logits'], atol=0, rtol=0)
                self.assertEqual(first['_init_diffusion_type_logits'].shape, (3, 3))

    def test_cached_wrapper_context_removes_per_type_gt_counts(self):
        wrapper = self.wrapper(generate_type=True, fix_ego=False)
        _, agent, _ = self.inputs()
        agent['ego_feat'][:, -3:] = torch.tensor([[17., 9., 12.]])
        expected_pose = agent['ego_feat'][:, :-3].clone()
        wrapper._prepare_ego_context(agent)
        torch.testing.assert_close(agent['ego_feat'][:, :-3], expected_pose, atol=0, rtol=0)
        torch.testing.assert_close(agent['ego_feat'][:, -3:], torch.tensor([[3., 0., 0.]]))

    def test_real_network_backpropagates_type_head_for_log_uniform_circular_states(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(generate_type=True, fix_ego=False, velocity_representation=representation,
                                 size_representation='log', heading_objective='angular_velocity',
                                 pos_source='uniform', shape_source='uniform', velocity_source='uniform').train()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                total = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                gradients = [p.grad for p in flow.model.to_out_type.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                embedding_gradient = flow.model.type_a_emb.weight.grad
                self.assertIsNotNone(embedding_gradient)
                self.assertTrue(torch.isfinite(embedding_gradient).all())
                self.assertGreater(embedding_gradient.abs().sum().item(), 0.)

    def test_sampler_evolves_noisy_type_state_and_generates_ego_when_unfixed(self):
        for representation, fixed in itertools.product(('vector', 'speed'), (True, False)):
            with self.subTest(velocity=representation, fix_ego=fixed):
                flow = self.flow(generate_type=True, fix_ego=fixed,
                                 velocity_representation=representation).eval()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                original_types = agent['type'].clone()
                oracle_logits = torch.tensor([[-3., 4., -2.], [3., -1., 0.], [4., -3., -2.]])
                recorded = []
                def predict(latent, time, current, *args, **kwargs):
                    recorded.append((time.clone(), current['_init_diffusion_type_state'].clone()))
                    current['_init_diffusion_type_logits'] = oracle_logits.clone()
                    return clean.clone()
                with patch.object(flow.model, 'forward', side_effect=predict):
                    generated = flow.sample(agent, feature, steps=2)
                self.assertEqual(generated.shape, clean.shape)
                source = recorded[0][1]
                probs = oracle_logits.softmax(-1)
                movable = ~agent['ego_mask'] if fixed else torch.ones(3, dtype=torch.bool)
                torch.testing.assert_close(recorded[1][1][movable], .5*source[movable]+.5*probs[movable], atol=2.e-7, rtol=0)
                expected = oracle_logits.argmax(-1)
                if fixed:
                    expected[agent['ego_mask']] = original_types[agent['ego_mask']]
                # Physical conversion must update types and token libraries before
                # assigning the nearest velocity token.
                output = flow.model.get_output(generated, agent)
                torch.testing.assert_close(agent['type'], expected, atol=0, rtol=0)
                shape, all_tokens, final_tokens = self.library(expected)
                torch.testing.assert_close(agent['token_agent_shape'], shape, atol=0, rtol=0)
                torch.testing.assert_close(agent['token_traj_all'], all_tokens, atol=0, rtol=0)
                torch.testing.assert_close(agent['token_traj'], final_tokens, atol=0, rtol=0)
                local_vel = clean[:, 6:8]
                if representation == 'speed':
                    local_vel = torch.cat((clean[:, 6:7].clamp_min(0), torch.zeros(3, 1)), -1)
                    if fixed:
                        local_vel[agent['ego_mask']] = agent['local_vel'][agent['ego_mask']]
                token_vel = final_tokens.mean(-2)/.5
                expected_idx = (token_vel-local_vel[:, None]).norm(dim=-1).argmin(-1)
                torch.testing.assert_close(output[-1][:, 0], expected_idx, atol=0, rtol=0)

    def test_generation_does_not_depend_on_gt_type_vector_when_ego_unfixed(self):
        flow = self.flow(generate_type=True, fix_ego=False).eval()
        _, agent, feature = self.inputs()
        first = copy.deepcopy(agent)
        second = copy.deepcopy(agent)
        second['type'] = torch.tensor([2, 2, 0])
        second['ego_feat'][:, -3:] = torch.tensor([[4., 9., 17.]])
        # Source normalizer is fixed before evaluation, as in a trained model.
        flow.model.normal_mean.zero_()
        flow.model.normal_scale.fill_(1.)
        with torch.no_grad():
            torch.manual_seed(81)
            a = flow.sample(first, feature, steps=2)
            torch.manual_seed(81)
            b = flow.sample(second, feature, steps=2)
            flow.model.get_output(a, first)
            flow.model.get_output(b, second)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(first['_init_diffusion_type_logits'], second['_init_diffusion_type_logits'], atol=0, rtol=0)
        torch.testing.assert_close(first['type'], second['type'], atol=0, rtol=0)

    def test_ema_tracks_new_type_head_and_enabled_checkpoint_round_trips(self):
        enabled = self.wrapper(generate_type=True, fix_ego=False, use_ema=True)
        self.assertTrue(enabled.generate_type)
        self.assertTrue(enabled.G1.generate_type)
        self.assertTrue(enabled.G1.model.generate_type)
        self.assertEqual(enabled.G1.model.m_delta_dim, 8)
        head_names = [name for name, _ in enabled.G1.named_parameters() if 'to_out_type' in name]
        self.assertTrue(head_names)
        self.assertEqual(len(enabled.ema.shadow_params), len(list(enabled.G1.parameters())))
        target = self.wrapper(generate_type=True, fix_ego=False, use_ema=True)
        target.load_state_dict(enabled.state_dict(), strict=True)
        target.update_ema()
        for key, value in enabled.G1.state_dict().items():
            torch.testing.assert_close(value, target.G1.state_dict()[key], atol=0, rtol=0)
        default = self.wrapper()
        default_copy = self.wrapper(generate_type=False)
        default_copy.load_state_dict(default.state_dict(), strict=True)
        self.assertFalse(any('to_out_type' in key for key in default.state_dict()))

    def test_legacy_checkpoint_requires_explicit_warm_start_for_new_type_head(self):
        legacy = self.wrapper(generate_type=False, use_ema=True)
        checkpoint = copy.deepcopy(legacy.state_dict())
        # A checkpoint predating type generation has no type metadata either.
        checkpoint['_extra_state'].pop('generate_type')
        enabled = self.wrapper(generate_type=True, fix_ego=False, use_ema=True)
        with self.assertRaisesRegex(RuntimeError, 'to_out_type'):
            enabled.load_state_dict(checkpoint, strict=True)
        enabled = self.wrapper(generate_type=True, fix_ego=False, use_ema=True)
        incompatible = enabled.load_state_dict(checkpoint, strict=False)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all('to_out_type' in key for key in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)
        for name, parameter in legacy.G1.state_dict().items():
            torch.testing.assert_close(parameter, enabled.G1.state_dict()[name], atol=0, rtol=0)
        # The old EMA layout must not be rebound to a larger parameter list.
        self.assertEqual(len(enabled.ema.shadow_params), len(list(enabled.G1.parameters())))
        for shadow, parameter in zip(enabled.ema.shadow_params, enabled.G1.parameters()):
            torch.testing.assert_close(shadow, parameter, atol=0, rtol=0)
        _, agent, feature = self.inputs()
        with patch.object(enabled.G1, 'sample') as sampler, \
                self.assertRaisesRegex(ValueError, 'type head'):
            enabled.eval()._infer(agent, feature)
        sampler.assert_not_called()
        # Full legacy inference remains available with generated types disabled.
        conditional = self.wrapper(generate_type=False, use_ema=True)
        conditional.load_state_dict(checkpoint, strict=True)
        output = conditional.eval()._infer(copy.deepcopy(agent), feature)
        self.assertTrue(all(torch.isfinite(item).all() for item in output))
        # A supervised optimizer update trains the new head and permits eval.
        enabled.train()
        losses = enabled._train(agent, feature, agent['batch'])
        (losses[0]+losses[1]).backward()
        head_parameter = next(enabled.G1.model.to_out_type.parameters())
        self.assertIsNotNone(head_parameter.grad)
        self.assertGreater(head_parameter.grad.abs().sum().item(), 0.)
        torch.optim.SGD(enabled.G1.parameters(), lr=.001).step()
        enabled.update_ema()
        output = enabled.eval()._infer(agent, feature)
        self.assertTrue(all(torch.isfinite(item).all() for item in output))

    def test_generated_types_reach_official_metric_vehicle_filter_and_export(self):
        import pickle
        import numpy as np
        import test_init_diffusion_metrics as metric_fixture
        from src.smart.model.smart import SMART

        fixture = metric_fixture.InitDiffusionMetricTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        generated = torch.tensor([1, 0, 2, 0, 1, 0])
        original = fixture.agent['type'].clone()
        prediction = fixture.init_decoder.return_value
        def generate(agent):
            agent['type'] = generated.clone()
            return prediction
        fixture.init_decoder.side_effect = generate
        output = fixture.rollout()
        torch.testing.assert_close(output['generated_type'][:, 0], generated, atol=0, rtol=0)
        # A cloned rollout can generate new labels without rewriting GT metadata.
        torch.testing.assert_close(fixture.agent['type'], original, atol=0, rtol=0)
        fixture.model._rollouts = lambda *args: output
        SMART._validate_closed_loop(fixture.model, fixture.data, {}, fixture.agent, 0)
        self.assertEqual(fixture.evaluator.report()['num_generated_vehicles'], 3)
        self.assertEqual(int((original == 0).sum()), 4)
        for scene in range(2):
            with (fixture.root/'export'/f'{scene:05d}.pkl').open('rb') as handle:
                exported = pickle.load(handle)
            expected_types = F.one_hot(generated[scene*3:scene*3+3], 3).numpy()
            np.testing.assert_array_equal(exported['agent_types'], expected_types)
            np.testing.assert_array_equal(exported['road_points'], fixture.raw[scene]['road_points'])

    def test_type_generation_requires_supervised_initial_scene_decoder(self):
        import inspect
        from src.smart.modules.smart_decoder import SMARTDecoder

        parameters = inspect.signature(SMARTDecoder.__init__).parameters
        arguments = {name: 1 for name, value in parameters.items()
                     if name != 'self' and value.default is inspect.Parameter.empty}
        arguments.update(dis_a2a_radius=0, token_processor=self.processor(pred_init=True),
                         init_decoder='flow', initial_scene_only=False,
                         init_diffusion={'generate_type': True})
        with self.assertRaisesRegex(ValueError, 'generate_type requires initial_scene_only'):
            SMARTDecoder(**arguments)

    def test_generated_types_require_evaluator_that_reads_generated_categories(self):
        from src.smart.model.smart import SMART

        for scenario_init, cached_evaluator in ((False, False), (True, False), (False, True)):
            with self.subTest(scenario_init=scenario_init, cached_evaluator=cached_evaluator):
                config = OmegaConf.create(dict(lr=1.e-4, lr_warmup_steps=10,
                    lr_total_steps=100, lr_min_ratio=.1, val_open_loop=False,
                    val_closed_loop=True, token_processor={}, decoder={'num_historical_steps': 11},
                    finetune=False, n_vis_batch=0, n_vis_scenario=0, n_vis_rollout=0,
                    n_batch_wosac_metric=0, sd_use_cached_evaluator=cached_evaluator))
                processor = SimpleNamespace(pred_init=True, n_token_agent=2,
                                            scenario_dreamer_init=scenario_init)
                encoder = SimpleNamespace(init_decoder_name='flow',
                                          init_decoder=SimpleNamespace(generate_type=True))
                with patch('src.smart.model.smart.TokenProcessor', return_value=processor), \
                        patch('src.smart.model.smart.SMARTDecoder', return_value=encoder), \
                        patch.object(SMART, '_configure_finetuning'), \
                        self.assertRaisesRegex(ValueError, 'scenario_dreamer_init=true and sd_use_cached_evaluator=true'):
                    SMART(config)

    def test_training_and_evaluation_config_allow_generated_or_gt_types(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        option = 'model.model_config.decoder.init_diffusion.generate_type'
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                with self.subTest(experiment=experiment):
                    default = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                    self.assertIsInstance(OmegaConf.select(default, option), bool)
                    for generated in (True, False):
                        selected = compose(config_name='run.yaml', overrides=[
                            f'experiment={experiment}', f'{option}={str(generated).lower()}'])
                        self.assertEqual(OmegaConf.select(selected, option), generated)
                    self.assertEqual(OmegaConf.select(default, 'model.model_config.decoder.init_diffusion.type_loss_weight'), 1.)


if __name__ == '__main__':
    unittest.main()
