"""Training-only categorical source calibration, pairing, and checkpoint behavior."""

import copy
import io
import itertools
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_joint_type as joint_fixtures


class InitDiffusionTypeNoiseStatsTest(unittest.TestCase):
    args = staticmethod(joint_fixtures.InitDiffusionJointTypeTest.args)
    processor = staticmethod(joint_fixtures.InitDiffusionJointTypeTest.processor)
    inputs = staticmethod(joint_fixtures.InitDiffusionJointTypeTest.inputs)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(463)

    def flow(self, **options):
        defaults = dict(generate_type=True, type_process='joint',
                        type_noise_stats='data', fix_ego=False)
        defaults.update(options)
        return Flow(self.args(**defaults), self.processor(), False)

    def wrapper(self, **options):
        defaults = dict(generate_type=True, type_process='joint',
                        type_noise_stats='data', fix_ego=False)
        defaults.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **defaults)

    def calibrated(self, flow, types=None):
        _, agent, feature = self.inputs()
        if types is not None:
            agent['type'] = torch.tensor(types, dtype=torch.long)
        clean, _ = flow.model.get_input(agent)
        flow.train()._sample_noise(clean, copy.deepcopy(agent))
        return clean, agent, feature

    def assert_stats(self, flow, expected_mean, expected_scale):
        self.assertTrue(bool(flow.type_normal_initialized))
        torch.testing.assert_close(flow.type_normal_mean,
                                   torch.tensor([expected_mean], dtype=flow.type_normal_mean.dtype))
        torch.testing.assert_close(flow.type_normal_scale,
                                   torch.tensor([expected_scale], dtype=flow.type_normal_scale.dtype))

    def test_one_hot_population_statistics_include_fixed_ego_and_absent_class(self):
        for process in ('joint', 'separate'):
            with self.subTest(type_process=process):
                flow = self.flow(type_process=process, fix_ego_type=True,
                                 type_noise_std_floor=.03)
                self.calibrated(flow, [0, 1, 0])
                standard_deviation = math.sqrt(2./9.)
                # The only class-1 agent is fixed ego. It still contributes to
                # source data statistics; class 2 uses the explicit floor.
                self.assert_stats(flow, [2./3., 1./3., 0.],
                                  [standard_deviation, standard_deviation, .03])
                self.assertFalse(flow.type_normal_mean.requires_grad)
                self.assertFalse(flow.type_normal_scale.requires_grad)
                parameter_ids = {id(value) for value in flow.parameters()}
                self.assertNotIn(id(flow.type_normal_mean), parameter_ids)
                self.assertNotIn(id(flow.type_normal_scale), parameter_ids)

    def test_frozen_statistics_do_not_refit_on_later_training_labels(self):
        for process in ('joint', 'separate'):
            flow = self.flow(type_process=process)
            clean, agent, _ = self.calibrated(flow, [0, 0, 1])
            before = (flow.type_normal_mean.clone(), flow.type_normal_scale.clone())
            agent['type'] = torch.tensor([2, 2, 2])
            with patch('torch.distributed.is_available', return_value=True), \
                    patch('torch.distributed.is_initialized', return_value=True), \
                    patch('torch.distributed.all_reduce') as reduce:
                flow._sample_noise(clean, agent)
            self.assertEqual(reduce.call_count, 0)
            torch.testing.assert_close(flow.type_normal_mean, before[0], atol=0, rtol=0)
            torch.testing.assert_close(flow.type_normal_scale, before[1], atol=0, rtol=0)

    def test_draw_and_matching_use_affine_source_then_normalized_full_rows(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity_representation=representation):
                flow = self.flow(velocity_representation=representation,
                                 heading_noise='gaussian', type_match_weight=4.)
                clean, agent, _ = self.calibrated(flow, [0, 0, 1])
                dimension = clean.shape[-1]
                flow.model.normal_mean.fill_(.25)
                flow.model.normal_scale.copy_(torch.arange(1, dimension+1).reshape(1, -1))
                standard = torch.arange(3*(dimension+3)).reshape(3, dimension+3).float()/7.-1.
                expected_physical = standard[:, :dimension]*flow.model.normal_scale+flow.model.normal_mean
                expected_type = standard[:, dimension:]*flow.type_normal_scale+flow.type_normal_mean
                expected_source = torch.cat((expected_physical, expected_type), -1)
                clean_joint = torch.cat((clean, F.one_hot(agent['type'], 3).to(clean)), -1)
                movable = ~agent['ego_mask']
                permute = torch.tensor([1, 0])
                with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=standard.clone()) as draw, \
                        patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast', return_value=permute) as match:
                    sampled = flow._sample_noise(clean, agent)
                self.assertEqual(draw.call_count, 1)
                self.assertEqual(tuple(draw.call_args.args[0].shape), (3, dimension+3))
                self.assertTrue(match.call_args.kwargs['use_all_type'])
                source_cost, target_cost = match.call_args.args[:2]
                scale = torch.cat((flow.model.normal_scale, flow.type_normal_scale), -1)
                weight = torch.cat((torch.ones(dimension), torch.full((3,), 2.)))
                # Centering both representations is optional and cancels in
                # Hungarian pair distances, so verify all pair differences.
                expected_cost_delta = ((expected_source[movable, None]-clean_joint[None, movable])/scale)*weight
                torch.testing.assert_close(source_cost[:, None]-target_cost[None], expected_cost_delta)
                torch.testing.assert_close(sampled[movable], expected_source[movable][permute], atol=0, rtol=0)
                torch.testing.assert_close(sampled[~movable], expected_source[~movable], atol=0, rtol=0)
                torch.testing.assert_close(agent['type'], torch.tensor([0, 0, 1]), atol=0, rtol=0)

    def test_separate_source_uses_data_affine_transform_and_original_draw_order(self):
        flow = self.flow(type_process='separate', fix_ego_type=True)
        _, agent, _ = self.calibrated(flow, [0, 0, 1])
        clean, _ = flow.model.get_input(agent)
        expected_agent = copy.deepcopy(agent)
        torch.manual_seed(80)
        expected_physical = flow._sample_noise(clean, expected_agent)
        expected_time = flow._sample_time(clean, expected_agent)
        labels = F.one_hot(agent['type'], 3).to(clean)
        standard = torch.randn_like(labels)
        expected = standard*flow.type_normal_scale+flow.type_normal_mean
        expected[agent['ego_mask']] = labels[agent['ego_mask']]
        expected_next = torch.rand(5)
        torch.manual_seed(80)
        actual_agent = copy.deepcopy(agent)
        actual_physical, actual_time, _ = flow._prepare_supervised_batch(clean, actual_agent)
        actual_next = torch.rand(5)
        torch.testing.assert_close(actual_physical, expected_physical, atol=0, rtol=0)
        torch.testing.assert_close(actual_time, expected_time, atol=0, rtol=0)
        torch.testing.assert_close(actual_agent['_init_diffusion_type_source'], expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)

    def test_uninitialized_evaluation_fails_without_fitting_or_consuming_rng(self):
        for process, operation in itertools.product(('joint', 'separate'), ('sample', 'noise')):
            with self.subTest(type_process=process, operation=operation):
                flow = self.flow(type_process=process).eval()
                clean, agent, feature = self.inputs()
                before_mean = flow.type_normal_mean.clone()
                before_scale = flow.type_normal_scale.clone()
                before_rng = torch.get_rng_state().clone()
                with self.assertRaisesRegex(RuntimeError, 'stat|train|initializ'):
                    if operation == 'sample':
                        flow.sample(agent, feature, steps=2)
                    else:
                        flow._sample_noise(clean, agent)
                torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
                torch.testing.assert_close(flow.type_normal_mean, before_mean, atol=0, rtol=0)
                torch.testing.assert_close(flow.type_normal_scale, before_scale, atol=0, rtol=0)
                self.assertFalse(bool(flow.type_normal_initialized))

    def test_sampling_never_refits_even_in_training_mode(self):
        flow = self.flow().train()
        _, agent, feature = self.inputs()
        before_rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'stat|train|initializ'):
            flow.sample(agent, feature, steps=2)
        self.assertFalse(bool(flow.type_normal_initialized))
        torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)

    def test_saved_statistics_are_restored_and_generated_types_do_not_leak_gt_labels(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity_representation=representation):
                original = self.flow(velocity_representation=representation)
                _, agent, feature = self.calibrated(original, [0, 0, 1])
                target = self.flow(velocity_representation=representation).eval()
                # An actual serialization round trip includes all three buffers.
                stream = io.BytesIO()
                torch.save(original.state_dict(), stream)
                stream.seek(0)
                target.load_state_dict(torch.load(stream, weights_only=True), strict=True)
                self.assert_stats(target, [2./3., 1./3., 0.],
                                  [math.sqrt(2./9.), math.sqrt(2./9.), .01])
                first, second = copy.deepcopy(agent), copy.deepcopy(agent)
                second['type'] = torch.tensor([2, 2, 2])
                second['ego_feat'][:, -3:] = 17.
                before_stats = (target.type_normal_mean.clone(), target.type_normal_scale.clone())
                with torch.no_grad():
                    torch.manual_seed(684)
                    a = target.sample(first, feature, steps=2)
                    torch.manual_seed(684)
                    b = target.sample(second, feature, steps=2)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(first['_init_diffusion_type_logits'], second['_init_diffusion_type_logits'], atol=0, rtol=0)
                torch.testing.assert_close(first['type'], second['type'], atol=0, rtol=0)
                torch.testing.assert_close(target.type_normal_mean, before_stats[0], atol=0, rtol=0)
                torch.testing.assert_close(target.type_normal_scale, before_stats[1], atol=0, rtol=0)

    def test_legacy_checkpoint_can_finetune_but_incomplete_stats_fail_strict_loading(self):
        original = self.flow()
        self.calibrated(original, [0, 0, 1])
        state = original.state_dict()
        names = {'type_normal_mean', 'type_normal_scale', 'type_normal_initialized'}
        self.assertTrue(names.issubset(state))
        legacy = {key: value for key, value in state.items() if key not in names}
        target = self.flow()
        target.load_state_dict(legacy, strict=True)
        self.assertFalse(bool(target.type_normal_initialized))
        _, agent, feature = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'stat|train|initializ'):
            target.eval().sample(agent, feature, steps=2)
        self.calibrated(target, [2, 2, 2])
        self.assert_stats(target, [0., 0., 1.], [.01, .01, .01])
        for omitted in names:
            with self.subTest(missing=omitted), self.assertRaises(RuntimeError):
                self.flow().load_state_dict({key: value for key, value in state.items() if key != omitted}, strict=True)

    def test_distributed_first_batch_aggregates_counts_once(self):
        flow = self.flow(type_noise_std_floor=.02)
        _, agent, _ = self.inputs()
        agent['type'] = torch.tensor([0, 0, 1])
        clean, _ = flow.model.get_input(agent)
        def remote_counts(counts, *args, **kwargs):
            self.assertEqual(counts.numel(), 3)
            counts.add_(torch.tensor([0., 1., 2.], dtype=counts.dtype, device=counts.device))
        with patch('torch.distributed.is_available', return_value=True), \
                patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.all_reduce', side_effect=remote_counts) as reduce:
            flow._sample_noise(clean, copy.deepcopy(agent))
            flow._sample_noise(clean, copy.deepcopy(agent))
        self.assertEqual(reduce.call_count, 1)
        self.assert_stats(flow, [1./3., 1./3., 1./3.], [math.sqrt(2./9.)]*3)

    def test_training_backward_and_sampling_are_finite_with_missing_classes(self):
        for representation, process in itertools.product(('vector', 'speed'), ('joint', 'separate')):
            with self.subTest(velocity_representation=representation, type_process=process):
                flow = self.flow(velocity_representation=representation,
                                 type_process=process, type_noise_std_floor=.02,
                                 fix_ego_type=True, fix_ego_position=True,
                                 fix_ego_shape=True).train()
                _, agent, feature = self.inputs()
                agent['type'] = torch.tensor([0, 0, 0])
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                self.assert_stats(flow, [1., 0., 0.], [.02, .02, .02])
                total = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                gradients = [value.grad for value in flow.model.to_out_type.parameters()]
                self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
                self.assertTrue(any(value.abs().sum() > 0 for value in gradients))
                with torch.no_grad():
                    generated = flow.eval().sample(agent, feature, steps=2)
                self.assertEqual(generated.shape, clean.shape)
                self.assertTrue(torch.isfinite(generated).all())
                torch.testing.assert_close(agent['_init_diffusion_type_source'][agent['ego_mask']],
                                           torch.tensor([[1., 0., 0.]]), atol=0, rtol=0)

    def test_wrapper_ema_preserves_data_buffers_and_metadata(self):
        wrapper = self.wrapper(use_ema=True, type_noise_std_floor=.02)
        self.calibrated(wrapper.G1, [0, 0, 1])
        self.assertEqual(wrapper.type_noise_stats, 'data')
        self.assertEqual(wrapper.G1.type_noise_stats, 'data')
        self.assertEqual(wrapper.type_noise_std_floor, .02)
        self.assertEqual(len(wrapper.ema.shadow_params), len(list(wrapper.G1.parameters())))
        state = wrapper.state_dict()
        self.assertEqual(state['_extra_state']['type_noise_stats'], 'data')
        self.assertEqual(state['_extra_state']['type_noise_std_floor'], .02)
        target = self.wrapper(use_ema=True, type_noise_std_floor=.02)
        target.load_state_dict(state, strict=True)
        torch.testing.assert_close(target.G1.type_normal_mean, wrapper.G1.type_normal_mean, atol=0, rtol=0)
        torch.testing.assert_close(target.G1.type_normal_scale, wrapper.G1.type_normal_scale, atol=0, rtol=0)
        with target.ema.average_parameters(target.G1.parameters()):
            torch.testing.assert_close(target.G1.type_normal_mean, wrapper.G1.type_normal_mean, atol=0, rtol=0)
            torch.testing.assert_close(target.G1.type_normal_scale, wrapper.G1.type_normal_scale, atol=0, rtol=0)

    def test_default_standard_and_disabled_generation_keep_state_keys_and_rng(self):
        for generated, stats in ((True, 'standard'), (False, 'data')):
            with self.subTest(generate_type=generated, type_noise_stats=stats):
                torch.manual_seed(109)
                original = self.flow(generate_type=generated, type_noise_stats='standard')
                rng_original = torch.get_rng_state().clone()
                torch.manual_seed(109)
                candidate = self.flow(generate_type=generated, type_noise_stats=stats)
                rng_candidate = torch.get_rng_state().clone()
                self.assertEqual(set(original.state_dict()), set(candidate.state_dict()))
                self.assertNotIn('type_normal_mean', candidate.state_dict())
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, candidate.state_dict()[key], atol=0, rtol=0)
                torch.testing.assert_close(rng_original, rng_candidate, atol=0, rtol=0)
                clean, agent, _ = self.inputs()
                torch.manual_seed(283)
                a = original._prepare_supervised_batch(clean, copy.deepcopy(agent))
                after_a = torch.rand(5)
                torch.manual_seed(283)
                b = candidate._prepare_supervised_batch(clean, copy.deepcopy(agent))
                after_b = torch.rand(5)
                for first, second in zip(a, b):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
                torch.testing.assert_close(after_a, after_b, atol=0, rtol=0)

    def test_noise_statistics_options_validate(self):
        for factory in (self.flow, self.wrapper):
            for stats in (None, True, 'dataset', 'unknown'):
                with self.subTest(factory=factory.__name__, mode=stats), self.assertRaisesRegex(ValueError, 'type_noise_stats'):
                    factory(type_noise_stats=stats)
            for floor in (-1., 0., .51, math.inf, math.nan):
                with self.subTest(factory=factory.__name__, floor=floor), self.assertRaisesRegex(ValueError, 'type_noise_std_floor'):
                    factory(type_noise_std_floor=floor)


if __name__ == '__main__':
    unittest.main()
