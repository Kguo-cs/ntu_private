"""Explicit ego role conditions one designated slot without fixing its state."""

import copy
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.diffusion_utils import get_closest_sum_idx_fast
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_type_generation as type_fixtures
import test_init_diffusion_count_embedding as count_fixtures


class InitDiffusionEgoEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(391)

    args = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.args)
    processor = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.processor)
    inputs = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.inputs)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **options)

    def denoiser(self, **options):
        values = dict(token_processor=self.processor(), input_dim=8, hidden_dim=32,
                      output_dim=8, num_layers=1, num_heads=2, dropout=0.)
        values.update(options)
        return InitDenoiser(**values)

    def test_option_is_boolean_at_each_public_layer(self):
        for invalid in ('false', 0, 1, None):
            for factory in (self.flow, self.wrapper, self.denoiser):
                with self.subTest(value=invalid, factory=factory.__name__), \
                        self.assertRaisesRegex(ValueError, 'use_ego_embedding'):
                    factory(use_ego_embedding=invalid)
        wrapper = self.wrapper(use_ego_embedding=True)
        self.assertTrue(wrapper.use_ego_embedding)
        self.assertTrue(wrapper.G1.use_ego_embedding)
        self.assertTrue(wrapper.G1.model.use_ego_embedding)
        embedding = wrapper.G1.model.ego_a_emb
        self.assertIsInstance(embedding, nn.Embedding)
        self.assertEqual(tuple(embedding.weight.shape), (2, 32))

    def test_disabled_option_preserves_legacy_layout_rng_and_results(self):
        torch.manual_seed(927)
        legacy = self.flow(fix_ego=False)
        after_legacy = torch.rand(5)
        torch.manual_seed(927)
        disabled = self.flow(fix_ego=False, use_ego_embedding=False)
        after_disabled = torch.rand(5)
        self.assertFalse(legacy.use_ego_embedding)
        self.assertFalse(hasattr(legacy.model, 'ego_a_emb'))
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        for name, parameter in legacy.state_dict().items():
            torch.testing.assert_close(parameter, disabled.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(after_legacy, after_disabled, atol=0, rtol=0)
        clean, agent, feature = self.inputs()
        for seed in (189, 773):
            torch.manual_seed(seed)
            first = legacy._prepare_supervised_batch(clean, copy.deepcopy(agent))
            next_first = torch.rand(5)
            torch.manual_seed(seed)
            second = disabled._prepare_supervised_batch(clean, copy.deepcopy(agent))
            next_second = torch.rand(5)
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(next_first, next_second, atol=0, rtol=0)
        legacy.eval()
        disabled.eval()
        with torch.no_grad():
            a = legacy.model(clean, torch.full((3, 1), .4), copy.deepcopy(agent), feature)
            b = disabled.model(clean, torch.full((3, 1), .4), copy.deepcopy(agent), feature)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_role_features_are_additive_and_follow_mask_not_row_order(self):
        model = self.denoiser(use_ego_embedding=True).eval()
        clean, agent, feature = self.inputs()
        first = copy.deepcopy(agent)
        second = copy.deepcopy(agent)
        second['ego_mask'] = torch.tensor([True, False, False])
        with torch.no_grad():
            model.ego_a_emb.weight[0].fill_(-.25)
            model.ego_a_emb.weight[1].fill_(.75)
        captured = []
        def attention(**kwargs):
            captured.append(kwargs['feat_a'].detach().clone())
            return torch.zeros(len(kwargs['feat_a']), 8)
        with patch.object(model, '_apply_graph_attention', side_effect=attention):
            model(clean, torch.full((3, 1), .4), first, feature)
            model(clean, torch.full((3, 1), .4), second, feature)
        expected = model.ego_a_emb(second['ego_mask'].long()) - model.ego_a_emb(first['ego_mask'].long())
        torch.testing.assert_close(captured[1]-captured[0], expected, atol=2.e-7, rtol=0)
        torch.testing.assert_close(first['ego_mask'], agent['ego_mask'], atol=0, rtol=0)

    def test_missing_malformed_or_nonunique_full_scene_roles_fail_clearly(self):
        model = self.denoiser(use_ego_embedding=True).eval()
        clean, agent, feature = self.inputs()
        cases = ('missing', torch.tensor([0, 1, 0]), torch.tensor([[False], [True], [False]]),
                 torch.tensor([False, True]), torch.zeros(3, dtype=torch.bool),
                 torch.tensor([True, True, False]))
        for value in cases:
            current = copy.deepcopy(agent)
            if isinstance(value, str):
                current.pop('ego_mask')
            else:
                current['ego_mask'] = value
            with self.subTest(mask=value), self.assertRaisesRegex(ValueError, 'ego_mask|ego.*scene'):
                model(clean, torch.full((3, 1), .4), current, feature)
        # Correct aggregate count is insufficient: every individual scene needs one.
        clean, agent, feature = count_fixtures.InitDiffusionCountEmbeddingTest.inputs()
        agent['ego_mask'] = torch.tensor([True, True, False, True, False, False])
        with self.assertRaisesRegex(ValueError, 'ego.*scene'):
            model(clean, torch.full((6, 1), .4), agent, feature)

    def test_eval_mask_slices_roles_after_validating_complete_scene(self):
        model = self.denoiser(use_ego_embedding=True).eval()
        clean, agent, feature = count_fixtures.InitDiffusionCountEmbeddingTest.inputs()
        # Selected scenes have no ego; full input metadata has one per scene.
        selected = torch.tensor([True, True, False, False, True, False])
        invalid = copy.deepcopy(agent)
        invalid['ego_mask'][5] = False
        observed = []
        hook = model.ego_a_emb.register_forward_pre_hook(
            lambda _, values: observed.append(values[0].detach().clone()))
        self.addCleanup(hook.remove)
        with patch.object(model, '_apply_graph_attention',
                          side_effect=lambda **kw: torch.zeros(len(kw['feat_a']), 8)):
            output = model(clean, torch.full((6, 1), .4), agent, feature, eval_mask=selected)
        self.assertEqual(tuple(output.shape), (3, 8))
        self.assertEqual(len(observed), 1)
        torch.testing.assert_close(observed[0], agent['ego_mask'][selected].long(), atol=0, rtol=0)
        # Invalid unselected roles must still be caught before slicing.
        with self.assertRaisesRegex(ValueError, 'ego.*scene'):
            model(clean, torch.full((6, 1), .4), invalid, feature, eval_mask=selected)

    def test_ego_role_reserves_noise_source_without_clean_clamping_or_zero_time(self):
        for fixed in (True, False):
            with self.subTest(fix_ego=fixed):
                flow = self.flow(use_ego_embedding=True, fix_ego=fixed, heading_noise='gaussian')
                clean, agent, _ = self.inputs()
                endpoint = clean[torch.tensor([2, 0, 1])] + .25
                with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=endpoint.clone()), \
                        patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                              return_value=torch.tensor([1, 0])) as match:
                    matched = flow._sample_noise(clean, agent)
                self.assertEqual(match.call_args.args[0].shape[0], 2)
                torch.testing.assert_close(match.call_args.args[0], endpoint[[0, 2]]/flow.model.normal_scale)
                torch.testing.assert_close(match.call_args.args[1], clean[[0, 2]]/flow.model.normal_scale)
                torch.testing.assert_close(matched[[0, 2]], endpoint[[2, 0]], atol=0, rtol=0)
                torch.testing.assert_close(matched[1], clean[1] if fixed else endpoint[1], atol=0, rtol=0)
                with patch.object(flow, '_sample_noise', return_value=matched.clone()), \
                        patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                self.assertEqual(time[1].item(), 0. if fixed else torch.tensor(.4).item())
                torch.testing.assert_close(latent[1], clean[1] if fixed else .6*clean[1]+.4*endpoint[1])

    def test_non_ego_matching_keeps_existing_type_group_strategy(self):
        clean, agent, _ = self.inputs()
        for generation in (True, False):
            flow = self.flow(use_ego_embedding=True, fix_ego=False, generate_type=generation)
            with patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                       return_value=torch.arange(2)) as match:
                flow._sample_noise(clean, agent)
            self.assertEqual(match.call_args.kwargs.get('use_all_type', False), generation)
            torch.testing.assert_close(match.call_args.args[2]['type'], agent['type'][~agent['ego_mask']], atol=0, rtol=0)
            torch.testing.assert_close(match.call_args.args[2]['batch'], agent['batch'][~agent['ego_mask']], atol=0, rtol=0)

    def test_real_supervised_gradients_reach_both_roles_vector_speed_and_type_flow(self):
        for representation, generation in itertools.product(('vector', 'speed'), (True, False)):
            with self.subTest(velocity=representation, generate_type=generation):
                flow = self.flow(use_ego_embedding=True, fix_ego=False, generate_type=generation,
                                 velocity_representation=representation, size_representation='log',
                                 heading_objective='angular_velocity').train()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                total = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                grad = flow.model.ego_a_emb.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertTrue((grad.abs().sum(-1) > 0).all())
                if generation:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in flow.model.to_out_type.parameters()))

    def test_sampling_reuses_original_roles_without_fixing_unconditioned_ego(self):
        for fixed in (True, False):
            flow = self.flow(use_ego_embedding=True, fix_ego=fixed).eval()
            clean, agent, feature = self.inputs()
            original_role = agent['ego_mask'].clone()
            target = clean.clone()
            target[:, :2] += torch.tensor([8., -3.])
            target[:, 4:6] *= 1.2
            observed = []
            def predict(latent, time, current, *args, **kwargs):
                observed.append((current['ego_mask'].clone(), time.clone()))
                return target.clone()
            with patch.object(flow.model, 'forward', side_effect=predict):
                generated = flow.sample(agent, feature, steps=2)
            self.assertEqual(len(observed), 2)
            for role, time in observed:
                torch.testing.assert_close(role, original_role, atol=0, rtol=0)
                if fixed:
                    self.assertEqual(time[original_role].item(), 0.)
                else:
                    torch.testing.assert_close(time[original_role], time[[0]], atol=0, rtol=0)
                    self.assertGreater(time[original_role].item(), 0.)
            expected = target.clone()
            if fixed:
                expected[original_role] = clean[original_role]
            torch.testing.assert_close(generated, expected, atol=2.e-6, rtol=0)
            torch.testing.assert_close(agent['ego_mask'], original_role, atol=0, rtol=0)

    def test_ema_and_strict_checkpoint_round_trip_include_role_parameters(self):
        source = self.wrapper(use_ego_embedding=True, fix_ego=False, use_ema=True)
        target = self.wrapper(use_ego_embedding=True, fix_ego=False, use_ema=True)
        self.assertTrue(source.get_extra_state()['use_ego_embedding'])
        names = list(dict(source.G1.named_parameters()))
        index = names.index('model.ego_a_emb.weight')
        self.assertEqual(len(source.ema.shadow_params), len(names))
        with torch.no_grad():
            source.G1.model.ego_a_emb.weight.add_(.5)
        source.update_ema()
        self.assertGreater((source.ema.shadow_params[index]-target.ema.shadow_params[index]).abs().sum().item(), 0.)
        target.load_state_dict(source.state_dict(), strict=True)
        self.assertFalse(target._ego_embedding_missing_on_load)
        for name, value in source.G1.state_dict().items():
            torch.testing.assert_close(value, target.G1.state_dict()[name], atol=0, rtol=0)
        for a, b in zip(source.ema.shadow_params, target.ema.shadow_params):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        clean, agent, feature = self.inputs()
        source.eval()
        target.eval()
        with torch.no_grad():
            torch.manual_seed(882)
            a = source._infer(copy.deepcopy(agent), feature)
            torch.manual_seed(882)
            b = target._infer(copy.deepcopy(agent), feature)
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_legacy_ema_warm_start_resets_layout_and_requires_role_training_before_eval(self):
        legacy = self.wrapper(use_ego_embedding=False, use_ema=True)
        checkpoint = copy.deepcopy(legacy.state_dict())
        checkpoint['_extra_state'].pop('use_ego_embedding')
        enabled = self.wrapper(use_ego_embedding=True, fix_ego=False, use_ema=True)
        with self.assertRaisesRegex(RuntimeError, 'ego_a_emb'):
            enabled.load_state_dict(checkpoint, strict=True)
        enabled = self.wrapper(use_ego_embedding=True, fix_ego=False, use_ema=True)
        incompatible = enabled.load_state_dict(checkpoint, strict=False)
        self.assertEqual(incompatible.missing_keys, ['G1.model.ego_a_emb.weight'])
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(enabled._ego_embedding_missing_on_load)
        self.assertEqual(len(enabled.ema.shadow_params), len(list(enabled.G1.parameters())))
        for shadow, parameter in zip(enabled.ema.shadow_params, enabled.G1.parameters()):
            torch.testing.assert_close(shadow, parameter, atol=0, rtol=0)
        _, agent, feature = self.inputs()
        with patch.object(enabled.G1, 'sample') as sampler, \
                self.assertRaisesRegex(ValueError, 'ego.*embedding'):
            enabled.eval()._infer(copy.deepcopy(agent), feature)
        sampler.assert_not_called()
        enabled.train()
        losses = enabled._train(agent, feature, agent['batch'])
        (losses[0]+losses[1]).backward()
        gradient = enabled.G1.model.ego_a_emb.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.)
        torch.optim.SGD(enabled.G1.parameters(), lr=.001).step()
        enabled.update_ema()
        self.assertFalse(enabled._ego_embedding_missing_on_load)
        _, eval_agent, eval_feature = self.inputs()
        output = enabled.eval()._infer(eval_agent, eval_feature)
        self.assertTrue(all(torch.isfinite(item).all() for item in output))

    def test_role_mode_change_reverse_direction_starts_fresh_ema(self):
        enabled = self.wrapper(use_ego_embedding=True, use_ema=True)
        disabled = self.wrapper(use_ego_embedding=False, use_ema=True)
        incompatible = disabled.load_state_dict(enabled.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, ['G1.model.ego_a_emb.weight'])
        self.assertFalse(incompatible.missing_keys)
        self.assertEqual(len(disabled.ema.shadow_params), len(list(disabled.G1.parameters())))
        for shadow, parameter in zip(disabled.ema.shadow_params, disabled.G1.parameters()):
            torch.testing.assert_close(shadow, parameter, atol=0, rtol=0)

    def test_role_conditioned_denoiser_is_equivariant_to_within_scene_row_permutation(self):
        for generation in (True, False):
            with self.subTest(generate_type=generation):
                model = self.denoiser(use_ego_embedding=True, generate_type=generation).eval()
                clean, agent, feature = count_fixtures.InitDiffusionCountEmbeddingTest.inputs()
                if generation:
                    agent['_init_diffusion_type_state'] = torch.randn(6, 3)
                permutation = torch.tensor([1, 2, 0, 3, 5, 4])
                shuffled = copy.deepcopy(agent)
                for key, value in agent.items():
                    if torch.is_tensor(value) and value.ndim > 0 and len(value) == len(clean):
                        shuffled[key] = value[permutation].clone()
                with torch.no_grad():
                    first = model(clean, torch.full((6, 1), .4), agent, feature)
                    second = model(clean[permutation], torch.full((6, 1), .4), shuffled, feature)
                torch.testing.assert_close(second, first[permutation], atol=1.e-5, rtol=1.e-5)
                if generation:
                    torch.testing.assert_close(shuffled['_init_diffusion_type_logits'],
                        agent['_init_diffusion_type_logits'][permutation], atol=1.e-5, rtol=1.e-5)

    def test_single_agent_scene_keeps_ego_source_and_can_train_or_sample_it(self):
        for fixed in (True, False):
            with self.subTest(fix_ego=fixed):
                flow = self.flow(use_ego_embedding=True, fix_ego=fixed, heading_noise='gaussian')
                state, agent, feature = self.inputs()
                for key, value in tuple(agent.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and len(value) == 3:
                        agent[key] = value[1:2].clone()
                clean = state[1:2].clone()
                endpoint = clean + .25
                with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=endpoint.clone()), \
                        patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                              wraps=get_closest_sum_idx_fast) as match:
                    matched = flow._sample_noise(clean, agent)
                if match.called:
                    self.assertEqual(match.call_args.args[0].shape[0], 0)
                torch.testing.assert_close(matched, clean if fixed else endpoint, atol=0, rtol=0)
                with patch.object(flow, '_sample_noise', return_value=matched.clone()), \
                        patch.object(flow, '_sample_time', return_value=torch.full((1, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                self.assertTrue(all(torch.isfinite(loss).all() for loss in losses))
                if not fixed:
                    losses[0].mean().backward()
                    gradient = flow.model.ego_a_emb.weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertGreater(gradient[1].abs().sum().item(), 0.)
                target = clean.clone()
                target[:, :2] += 4.
                with patch.object(flow.model, 'forward', return_value=target.clone()):
                    generated = flow.eval().sample(agent, feature, steps=2)
                torch.testing.assert_close(generated, clean if fixed else target, atol=2.e-6, rtol=0)
                torch.testing.assert_close(agent['ego_mask'], torch.tensor([True]), atol=0, rtol=0)

    def test_training_eval_configs_enable_role_embedding_and_allow_legacy_override(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        prefix = 'model.model_config.decoder.init_diffusion'
        generic = OmegaConf.load(root/'configs/model/smart.yaml')
        self.assertFalse(OmegaConf.select(generic, 'model_config.decoder.init_diffusion.use_ego_embedding'))
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                with self.subTest(experiment=experiment):
                    default = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                    self.assertTrue(OmegaConf.select(default, f'{prefix}.use_ego_embedding'))
                    disabled = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                                            f'{prefix}.use_ego_embedding=false'])
                    self.assertFalse(OmegaConf.select(disabled, f'{prefix}.use_ego_embedding'))
                    self.assertEqual(OmegaConf.select(default, f'{prefix}.fix_ego'),
                                     OmegaConf.select(disabled, f'{prefix}.fix_ego'))
                    self.assertEqual(OmegaConf.select(default, f'{prefix}.generate_type'),
                                     OmegaConf.select(disabled, f'{prefix}.generate_type'))


if __name__ == '__main__':
    unittest.main()
