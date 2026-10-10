"""Joint physical/categorical Flow sources share pairing and reverse updates."""

import copy
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from src.smart.diffusion.diffusion_utils import get_closest_sum_idx_fast
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_type_generation as type_fixtures


class InitDiffusionJointTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(581)

    args = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.args)
    processor = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.processor)
    inputs = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.inputs)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **options)

    def test_default_separate_preserves_parameters_results_and_rng(self):
        for generated in (False, True):
            with self.subTest(generate_type=generated):
                torch.manual_seed(712)
                default = self.flow(generate_type=generated, fix_ego=False)
                after_default = torch.rand(5)
                torch.manual_seed(712)
                separate = self.flow(generate_type=generated, fix_ego=False,
                                     type_process='separate', type_match_weight=1.)
                after_separate = torch.rand(5)
                self.assertEqual(set(default.state_dict()), set(separate.state_dict()))
                for name, value in default.state_dict().items():
                    torch.testing.assert_close(value, separate.state_dict()[name], atol=0, rtol=0)
                torch.testing.assert_close(after_default, after_separate, atol=0, rtol=0)
                clean, agent, _ = self.inputs()
                first_agent, second_agent = copy.deepcopy(agent), copy.deepcopy(agent)
                torch.manual_seed(198)
                first = default._prepare_supervised_batch(clean, first_agent)
                next_first = torch.rand(5)
                torch.manual_seed(198)
                second = separate._prepare_supervised_batch(clean, second_agent)
                next_second = torch.rand(5)
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(next_first, next_second, atol=0, rtol=0)
                if generated:
                    for key in ('_init_diffusion_type_source', '_init_diffusion_type_state'):
                        torch.testing.assert_close(first_agent[key], second_agent[key], atol=0, rtol=0)
                    # Reproduce the old separate draw order independently:
                    # physical endpoint, scene time, then categorical source.
                    torch.manual_seed(198)
                    expected_noise = separate._sample_noise(clean, copy.deepcopy(agent))
                    expected_time = separate._sample_time(clean, copy.deepcopy(agent))
                    labels = F.one_hot(agent['type'], 3).to(clean)
                    expected_type_source = torch.randn_like(labels)
                    expected_next = torch.rand(5)
                    torch.testing.assert_close(first[0], expected_noise, atol=0, rtol=0)
                    torch.testing.assert_close(first[1], expected_time, atol=0, rtol=0)
                    torch.testing.assert_close(first_agent['_init_diffusion_type_source'], expected_type_source, atol=0, rtol=0)
                    torch.testing.assert_close(first_agent['_init_diffusion_type_state'],
                                               (1.-expected_time)*labels+expected_time*expected_type_source, atol=0, rtol=0)
                    torch.testing.assert_close(next_first, expected_next, atol=0, rtol=0)

    def test_joint_option_is_inert_without_type_generation(self):
        torch.manual_seed(901)
        separate = self.flow(generate_type=False, fix_ego=False, type_process='separate')
        torch.manual_seed(901)
        joint = self.flow(generate_type=False, fix_ego=False, type_process='joint')
        clean, agent, _ = self.inputs()
        for model in (separate, joint):
            self.assertEqual(model.model.m_delta_dim, 8)
            self.assertFalse(hasattr(model.model, 'to_out_type'))
        torch.manual_seed(419)
        first = separate._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_first = torch.rand(5)
        torch.manual_seed(419)
        second = joint._prepare_supervised_batch(clean, copy.deepcopy(agent))
        next_second = torch.rand(5)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(next_first, next_second, atol=0, rtol=0)

    def test_joint_latent_has_type_tail_and_shared_time_for_both_physical_layouts(self):
        for representation, fixed in itertools.product(('vector', 'speed'), (False, True)):
            with self.subTest(velocity=representation, fixed=fixed):
                flow = self.flow(generate_type=True, type_process='joint', fix_ego=fixed,
                                 velocity_representation=representation)
                _, agent, _ = self.inputs()
                clean, _ = flow.model.get_input(agent)
                labels = F.one_hot(agent['type'], 3).to(clean)
                before_labels = agent['type'].clone()
                dimension = 7 if representation == 'speed' else 8
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                self.assertEqual(flow.model.m_delta_dim, dimension)
                self.assertEqual(flow.model.normal_scale.shape[-1], dimension)
                self.assertEqual(tuple(noise.shape), (3, dimension+3))
                self.assertEqual(tuple(latent.shape), (3, dimension+3))
                torch.testing.assert_close(noise[:, dimension:], agent['_init_diffusion_type_source'])
                torch.testing.assert_close(latent[:, dimension:], agent['_init_diffusion_type_state'])
                expected_type = (1.-time)*labels + time*noise[:, dimension:]
                torch.testing.assert_close(latent[:, dimension:], expected_type)
                torch.testing.assert_close(agent['type'], before_labels, atol=0, rtol=0)
                if fixed:
                    torch.testing.assert_close(latent[agent['ego_mask'], :dimension], clean[agent['ego_mask']], atol=0, rtol=0)
                    torch.testing.assert_close(latent[agent['ego_mask'], dimension:], labels[agent['ego_mask']], atol=0, rtol=0)
                    self.assertEqual(time[agent['ego_mask']].item(), 0.)
                self.assertTrue(torch.isfinite(latent).all())
                torch.testing.assert_close(latent[:, 2:4].norm(dim=-1), torch.ones(3), atol=1.e-6, rtol=0)

    def test_actual_hungarian_moves_whole_source_row_and_type_affects_cost(self):
        # Physical costs prefer identity. Type costs prefer swapping the two
        # non-ego rows, which have different GT categories.
        clean, agent, _ = self.inputs()
        clean = clean[0:1].expand(3, -1).clone()
        clean[:, :2] = torch.tensor([[0., 0.], [5., 5.], [1., 0.]])
        agent['expert_input'] = clean.clone()
        source_physical = clean.clone()
        source_physical[1] += .25
        source_type = torch.tensor([[0., 0., 1.], [8., -3., 4.], [1., 0., 0.]])
        standard_source = torch.cat((source_physical, source_type), -1)
        results = {}
        for weight in (0., 1000.):
            flow = self.flow(generate_type=True, type_process='joint', type_match_weight=weight,
                             fix_ego=False, fix_ego_type=True, heading_noise='gaussian')
            flow.model.normal_mean.zero_()
            flow.model.normal_scale.fill_(1.)
            current = copy.deepcopy(agent)
            with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=standard_source.clone()) as draw, \
                    patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast', wraps=get_closest_sum_idx_fast) as match:
                matched = flow._sample_noise(clean, current)
            self.assertEqual(tuple(draw.call_args.args[0].shape), (3, 11))
            self.assertEqual(draw.call_count, 1)
            self.assertTrue(match.call_args.kwargs['use_all_type'])
            self.assertTrue(match.call_args.kwargs['all_state'])
            self.assertEqual(match.call_args.args[0].shape, (2, 11))
            torch.testing.assert_close(current['type'], agent['type'], atol=0, rtol=0)
            results[weight] = matched
        torch.testing.assert_close(results[0.][[0, 2]], standard_source[[0, 2]], atol=0, rtol=0)
        torch.testing.assert_close(results[1000.][[0, 2]], standard_source[[2, 0]], atol=0, rtol=0)
        for matched in results.values():
            # Ego physical state remains generated, while ego type is fixed.
            torch.testing.assert_close(matched[1, :8], source_physical[1], atol=0, rtol=0)
            torch.testing.assert_close(matched[1, 8:], torch.tensor([0., 1., 0.]), atol=0, rtol=0)

    def test_matching_cannot_exchange_joint_sources_between_scenes(self):
        flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                         heading_noise='gaussian', type_match_weight=1000.)
        flow.model.normal_mean.zero_()
        flow.model.normal_scale.fill_(1.)
        clean, agent, _ = self.inputs()
        clean = clean[0:1].expand(6, -1).clone()
        agent = dict(batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                     type=torch.tensor([0, 1, 2, 0, 1, 2]),
                     ego_mask=torch.tensor([False, True, False, False, True, False]),
                     num_graphs=2)
        source = torch.cat((clean, F.one_hot(agent['type'], 3).float()), -1)
        source[:, 0] = torch.tensor([10., 11., 12., 20., 21., 22.])
        source[[0, 2, 3, 5], 8:] = source[[2, 0, 5, 3], 8:]
        with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=source.clone()):
            matched = flow._sample_noise(clean, agent)
        self.assertEqual(sorted(matched[:3, 0].tolist()), [10., 11., 12.])
        self.assertEqual(sorted(matched[3:, 0].tolist()), [20., 21., 22.])
        torch.testing.assert_close(matched[:, 8:], F.one_hot(agent['type'], 3).float(), atol=0, rtol=0)

    def test_type_source_is_standard_and_physical_normalization_stays_physical(self):
        for representation in ('vector', 'speed'):
            flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                             velocity_representation=representation, heading_noise='gaussian')
            _, agent, _ = self.inputs()
            clean, _ = flow.model.get_input(agent)
            dimension = clean.shape[-1]
            mean = torch.arange(dimension, dtype=clean.dtype).reshape(1, -1)
            scale = torch.arange(1, dimension+1, dtype=clean.dtype).reshape(1, -1)
            flow.model.normal_mean.copy_(mean)
            flow.model.normal_scale.copy_(scale)
            source = torch.arange(3*(dimension+3), dtype=clean.dtype).reshape(3, dimension+3)/7.-2.
            with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=source.clone()), \
                    patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast', return_value=torch.arange(2)):
                noise = flow._sample_noise(clean, agent)
            torch.testing.assert_close(noise[:, :dimension], source[:, :dimension]*scale+mean)
            torch.testing.assert_close(noise[:, dimension:], source[:, dimension:], atol=0, rtol=0)
            torch.testing.assert_close(flow.model.normal_mean, mean, atol=0, rtol=0)
            torch.testing.assert_close(flow.model.normal_scale, scale, atol=0, rtol=0)

    def test_real_joint_training_backpropagates_type_head_and_embedding(self):
        for representation, heading in itertools.product(('vector', 'speed'), ('x0', 'angular_velocity')):
            with self.subTest(velocity=representation, heading=heading):
                flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                                 velocity_representation=representation, size_representation='log',
                                 heading_objective=heading, pos_source='uniform', shape_source='uniform',
                                 velocity_source='uniform').train()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                total = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                gradients = [parameter.grad for parameter in flow.model.to_out_type.parameters()]
                self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in gradients))
                self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))
                self.assertGreater(flow.model.type_a_emb.weight.grad.abs().sum().item(), 0.)
                self.assertIn('_init_diffusion_type_metrics', agent)
                self.assertEqual(len(losses), 6)

    def test_joint_sampling_updates_type_and_returns_only_physical_state(self):
        for representation, fixed_type in itertools.product(('vector', 'speed'), (False, True)):
            with self.subTest(velocity=representation, fixed_type=fixed_type):
                flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                                 fix_ego_type=fixed_type, velocity_representation=representation).eval()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                original_types = agent['type'].clone()
                logits = torch.tensor([[-3., 4., -2.], [3., -1., 0.], [4., -3., -2.]])
                captured = []
                def predict(latent, time, current, *args, **kwargs):
                    captured.append(current['_init_diffusion_type_state'].clone())
                    current['_init_diffusion_type_logits'] = logits.clone()
                    return clean.clone()
                with patch.object(flow.model, 'forward', side_effect=predict):
                    generated = flow.sample(agent, feature, steps=2)
                self.assertEqual(generated.shape, clean.shape)
                active = ~agent['ego_mask'] if fixed_type else torch.ones(3, dtype=torch.bool)
                torch.testing.assert_close(captured[1][active], .5*captured[0][active]+.5*logits.softmax(-1)[active], atol=2.e-7, rtol=0)
                expected = logits.argmax(-1)
                if fixed_type:
                    expected[agent['ego_mask']] = original_types[agent['ego_mask']]
                    labels = F.one_hot(original_types[agent['ego_mask']], 3).float()
                    for state in captured:
                        torch.testing.assert_close(state[agent['ego_mask']], labels, atol=0, rtol=0)
                torch.testing.assert_close(agent['_init_diffusion_generated_type'], expected, atol=0, rtol=0)
                torch.testing.assert_close(agent['type'], expected, atol=0, rtol=0)
                self.assertEqual(agent['gen_z'].shape, clean.shape)
                flow.model.get_output(generated, agent)
                shape, all_tokens, final_tokens = type_fixtures.InitDiffusionTypeGenerationTest.library(expected)
                torch.testing.assert_close(agent['token_agent_shape'], shape, atol=0, rtol=0)
                torch.testing.assert_close(agent['token_traj_all'], all_tokens, atol=0, rtol=0)
                torch.testing.assert_close(agent['token_traj'], final_tokens, atol=0, rtol=0)

    def test_joint_generation_has_no_gt_type_or_count_leakage(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                                 velocity_representation=representation).eval()
                _, agent, feature = self.inputs()
                first, second = copy.deepcopy(agent), copy.deepcopy(agent)
                second['type'] = torch.tensor([2, 2, 0])
                second['ego_feat'][:, -3:] = torch.tensor([[4., 9., 17.]])
                # A zero mean is the current normalizer's uninitialized marker.
                # Use fixed nonzero statistics, as in a trained checkpoint, so
                # neither inference call calibrates against the target batch.
                flow.model.normal_mean.fill_(.01)
                flow.model.normal_scale.fill_(1.)
                with torch.no_grad():
                    torch.manual_seed(81)
                    a = flow.sample(first, feature, steps=2)
                    torch.manual_seed(81)
                    b = flow.sample(second, feature, steps=2)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(first['_init_diffusion_type_logits'], second['_init_diffusion_type_logits'], atol=0, rtol=0)
                torch.testing.assert_close(first['type'], second['type'], atol=0, rtol=0)

    def test_partial_ego_velocity_and_type_masks_are_independent(self):
        for velocity_fixed, type_fixed in ((True, False), (False, True)):
            with self.subTest(velocity_fixed=velocity_fixed, type_fixed=type_fixed):
                flow = self.flow(generate_type=True, type_process='joint', fix_ego=False,
                                 velocity_representation='speed',
                                 fix_ego_velocity=velocity_fixed, fix_ego_type=type_fixed)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                dimension = clean.shape[-1]
                labels = F.one_hot(agent['type'], 3).to(clean)
                source_type = torch.tensor([[-.5, .3, 1.2], [.3, -.7, .9], [1.5, -.8, .2]])
                endpoint = torch.cat((clean+.25, source_type), -1)
                times = torch.full((3, 1), .4)
                ego = agent['ego_mask']
                with patch.object(flow, '_draw_joint_source', side_effect=lambda _: endpoint.clone()), \
                        patch.object(flow, '_sample_time', return_value=times.clone()):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                mask = flow._conditioned_state_mask(agent, latent)
                self.assertEqual(tuple(mask.shape), tuple(latent.shape))
                self.assertEqual(mask[ego, 6].item(), velocity_fixed)
                self.assertEqual(mask[ego, dimension:].all().item(), type_fixed)
                # An ego with any free physical/category field keeps scene time.
                torch.testing.assert_close(time[ego], times[ego], atol=0, rtol=0)
                expected_speed_source = clean[ego, 6] if velocity_fixed else endpoint[ego, 6]
                expected_type_source = labels[ego] if type_fixed else source_type[ego]
                torch.testing.assert_close(noise[ego, 6], expected_speed_source, atol=0, rtol=0)
                torch.testing.assert_close(noise[ego, dimension:], expected_type_source, atol=0, rtol=0)
                torch.testing.assert_close(latent[ego, 6], .6*clean[ego, 6]+.4*expected_speed_source)
                torch.testing.assert_close(latent[ego, dimension:], .6*labels[ego]+.4*expected_type_source)

                # Isolate categorical CE to check that velocity conditioning
                # never suppresses the ego target when its type is generated.
                logits = torch.tensor([[1., 0., -1.], [.5, -.3, 1.], [.2, .7, -.1]], requires_grad=True)
                def predict(latent, time, current, *args, **kwargs):
                    current['_init_diffusion_type_logits'] = logits
                    return clean.clone()
                zeros = (torch.zeros(3), torch.zeros(()), *(torch.zeros(3) for _ in range(4)))
                with patch.object(flow, '_draw_joint_source', side_effect=lambda _: endpoint.clone()), \
                        patch.object(flow, '_sample_time', return_value=times.clone()), \
                        patch.object(flow.model, 'forward', side_effect=predict), \
                        patch('src.smart.diffusion.scale_flow.get_diff_loss', return_value=zeros):
                    loss = flow._supervised_loss(clean, agent, feature)
                active = ~ego if type_fixed else torch.ones(3, dtype=torch.bool)
                expected_ce = F.cross_entropy(logits[active], agent['type'][active])
                torch.testing.assert_close(loss[0].mean(), expected_ce)
                loss[0].mean().backward()
                if type_fixed:
                    torch.testing.assert_close(logits.grad[ego], torch.zeros_like(logits.grad[ego]), atol=0, rtol=0)
                else:
                    self.assertGreater(logits.grad[ego].abs().sum().item(), 0.)
                self.assertGreater(logits.grad[~ego].abs().sum().item(), 0.)

    def test_active_experiment_features_train_and_sample_joint_speed(self):
        flow = self.flow(generate_type=True, type_process='joint', velocity_representation='speed',
                         fix_ego=False, fix_ego_position=True, fix_ego_heading=False,
                         fix_ego_shape=True, fix_ego_velocity=False, fix_ego_type=True,
                         use_ego_embedding=True, ego_context_heading_encoding='sincos',
                         time_embedding_type='scenario_dreamer').train()
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        original_types = agent['type'].clone()
        physical_outputs, type_outputs = [], []
        def capture_physical(_, inputs, output):
            output.retain_grad()
            physical_outputs.append(output)
        def capture_type(_, inputs, output):
            output.retain_grad()
            type_outputs.append(output)
        hooks = [flow.model.to_out_m_delta.register_forward_hook(capture_physical),
                 flow.model.to_out_type.register_forward_hook(capture_type)]
        try:
            with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                loss = flow._supervised_loss(clean, agent, feature)
            total = loss[0].mean()+loss[1]
            self.assertTrue(torch.isfinite(total))
            total.backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(physical_outputs), 1)
        self.assertEqual(len(type_outputs), 1)
        gradient = physical_outputs[0].grad
        ego = agent['ego_mask']
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[ego, 2:4].abs().sum().item(), 0.)
        self.assertGreater(gradient[ego, 6].abs().sum().item(), 0.)
        torch.testing.assert_close(gradient[ego, :2], torch.zeros_like(gradient[ego, :2]), atol=0, rtol=0)
        torch.testing.assert_close(gradient[ego, 4:6], torch.zeros_like(gradient[ego, 4:6]), atol=0, rtol=0)
        torch.testing.assert_close(type_outputs[0].grad[ego], torch.zeros_like(type_outputs[0].grad[ego]), atol=0, rtol=0)
        self.assertGreater(type_outputs[0].grad[~ego].abs().sum().item(), 0.)
        self.assertGreater(flow.model.ego_a_emb.weight.grad.abs().sum().item(), 0.)
        time_gradients = [parameter.grad for parameter in flow.model.noise_embedding.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in time_gradients))
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in time_gradients))
        with torch.no_grad():
            generated = flow.eval().sample(agent, feature, steps=3)
        self.assertEqual(tuple(generated.shape), (3, 7))
        self.assertTrue(torch.isfinite(generated).all())
        torch.testing.assert_close(generated[ego, :2], clean[ego, :2], atol=0, rtol=0)
        torch.testing.assert_close(generated[ego, 4:6], clean[ego, 4:6], atol=0, rtol=0)
        torch.testing.assert_close(agent['type'][ego], original_types[ego], atol=0, rtol=0)
        torch.testing.assert_close(generated[:, 2:4].norm(dim=-1), torch.ones(3), atol=1.e-6, rtol=0)
        self.assertFalse(torch.equal(generated[ego, 2:4], clean[ego, 2:4]))
        self.assertFalse(torch.equal(generated[ego, 6], clean[ego, 6]))

    def test_wrapper_ema_and_joint_checkpoint_round_trip(self):
        options = dict(generate_type=True, type_process='joint', type_match_weight=.7,
                       fix_ego=False, use_ema=True)
        enabled = self.wrapper(**options)
        self.assertEqual(enabled.type_process, 'joint')
        self.assertEqual(enabled.G1.type_process, 'joint')
        self.assertEqual(enabled.G1.type_match_weight, .7)
        self.assertEqual(len(enabled.ema.shadow_params), len(list(enabled.G1.parameters())))
        target = self.wrapper(**options)
        target.load_state_dict(enabled.state_dict(), strict=True)
        for key, value in enabled.G1.state_dict().items():
            torch.testing.assert_close(value, target.G1.state_dict()[key], atol=0, rtol=0)
        # Representation changes pairing, not the physical or type head layout.
        separate = self.wrapper(generate_type=True, type_process='separate', fix_ego=False, use_ema=True)
        self.assertEqual(set(enabled.G1.state_dict()), set(separate.G1.state_dict()))
        separate.load_state_dict(enabled.state_dict(), strict=True)

    def test_options_validate_and_configs_allow_joint_selection(self):
        for process in ('unknown', True, None):
            for factory in (self.flow, self.wrapper):
                with self.subTest(process=process, factory=factory.__name__), self.assertRaisesRegex(ValueError, 'type_process'):
                    factory(generate_type=True, type_process=process)
        for weight in (-1., math.inf, math.nan):
            for factory in (self.flow, self.wrapper):
                with self.subTest(weight=weight, factory=factory.__name__), self.assertRaisesRegex(ValueError, 'type_match_weight'):
                    factory(generate_type=True, type_process='joint', type_match_weight=weight)
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        option = 'model.model_config.decoder.init_diffusion.type_process'
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                selected = compose(config_name='run.yaml', overrides=[f'experiment={experiment}', f'{option}=joint'])
                self.assertEqual(OmegaConf.select(selected, option), 'joint')
                self.assertEqual(OmegaConf.select(selected, 'model.model_config.decoder.init_diffusion.type_match_weight'), 1.)


if __name__ == '__main__':
    unittest.main()
