"""One-hot agent types follow the same Euclidean state and loss as features."""

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

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
import test_init_diffusion_type_generation as type_fixtures


class InitDiffusionFeatureTypeTest(unittest.TestCase):
    args = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.args)
    processor = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.processor)
    inputs = staticmethod(type_fixtures.InitDiffusionTypeGenerationTest.inputs)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(719)

    def flow(self, **options):
        defaults = dict(generate_type=True, type_process='feature', fix_ego=False)
        defaults.update(options)
        return Flow(self.args(**defaults), self.processor(), False)

    def wrapper(self, **options):
        defaults = dict(generate_type=True, type_process='feature', fix_ego=False)
        defaults.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **defaults)

    def test_feature_layout_has_one_shared_projection_and_head(self):
        for representation, heading in itertools.product(('vector', 'speed'), ('x0', 'angular_velocity')):
            with self.subTest(velocity=representation, heading=heading):
                flow = self.flow(velocity_representation=representation, heading_objective=heading)
                dimension = 7 if representation == 'speed' else 8
                self.assertTrue(flow.feature_type)
                self.assertTrue(flow.joint_type)
                self.assertTrue(flow.model.type_as_feature)
                self.assertEqual(flow.model.m_delta_dim, dimension)
                self.assertEqual(flow.model.feature_dim, dimension+3)
                self.assertEqual(flow.model.proj_in_m_delta.in_features, dimension-4+3)
                self.assertEqual(flow.model.to_out_m_delta.mlp[-1].out_features, dimension+3)
                self.assertFalse(hasattr(flow.model, 'type_a_emb'))
                self.assertFalse(hasattr(flow.model, 'to_out_type'))
                self.assertEqual(tuple(flow.model.normal_scale.shape), (1, dimension))
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                labels = F.one_hot(agent['type'], 3).to(clean)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                torch.testing.assert_close(latent[:, dimension:], .6*labels+.4*noise[:, dimension:])
                output = flow._denoise(latent, time, agent, feature)
                self.assertEqual(tuple(output.shape), (3, dimension+3+(heading == 'angular_velocity')))
                self.assertTrue(torch.isfinite(output).all())

    def test_feature_option_is_inert_without_generation_and_default_is_legacy(self):
        for process in ('separate', 'joint', 'feature'):
            torch.manual_seed(414)
            model = self.flow(generate_type=False, type_process=process)
            rng = torch.rand(4)
            if process == 'separate':
                reference = model
                reference_rng = rng
            else:
                self.assertFalse(model.feature_type)
                self.assertFalse(model.joint_type)
                self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
                torch.testing.assert_close(rng, reference_rng, atol=0, rtol=0)
        default = Flow(self.args(generate_type=True), self.processor(), False)
        self.assertEqual(default.type_process, 'separate')
        self.assertFalse(default.feature_type)
        self.assertTrue(hasattr(default.model, 'type_a_emb'))
        self.assertTrue(hasattr(default.model, 'to_out_type'))

    def test_type_uses_common_mse_denominator_time_weight_and_no_ce(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(velocity_representation=representation, type_loss_weight=99.)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                labels = F.one_hot(agent['type'], 3).to(clean)
                joint = torch.cat((clean, labels), -1)
                prediction = joint.clone()
                errors = torch.tensor([[2., -3., 4.], [-1., .5, 3.], [4., 2., -.5]])
                prediction[:, clean.shape[-1]:] += errors
                prediction.requires_grad_()
                with patch.object(flow, '_draw_joint_source', side_effect=lambda _: joint.clone()), \
                        patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)), \
                        patch.object(flow.model, 'forward', return_value=prediction), \
                        patch('src.smart.diffusion.scale_flow.F.cross_entropy', side_effect=AssertionError('feature mode must not use CE')):
                    losses = flow._supervised_loss(clean, agent, feature)
                expected = errors.square().sum(-1)/(clean.shape[-1]+3)*(.1/5.)*(1./.4)**3
                torch.testing.assert_close(losses[0], expected)
                self.assertEqual(len(losses), 6)
                self.assertEqual(losses[1].item(), 0.)
                for component in losses[2:]:
                    torch.testing.assert_close(component, torch.zeros_like(component), atol=0, rtol=0)
                losses[0].mean().backward()
                self.assertGreater(prediction.grad[:, clean.shape[-1]:].abs().sum().item(), 0.)
                torch.testing.assert_close(prediction.grad[:, :clean.shape[-1]], torch.zeros_like(clean), atol=0, rtol=0)
                self.assertIn('accuracy', agent['_init_diffusion_type_metrics'])
                torch.testing.assert_close(agent['_init_diffusion_type_metrics']['weighted_loss'], expected.mean())

    def test_type_loss_weight_does_not_change_feature_objective(self):
        _, agent, feature = self.inputs()
        results = []
        for weight in (.01, 100.):
            torch.manual_seed(882)
            flow = self.flow(type_loss_weight=weight)
            clean, _ = flow.model.get_input(copy.deepcopy(agent))
            torch.manual_seed(93)
            result = flow._supervised_loss(clean, copy.deepcopy(agent), feature)
            results.append(result)
        for first, second in zip(*results):
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_real_shared_projection_and_head_receive_type_gradients(self):
        for representation, heading in itertools.product(('vector', 'speed'), ('x0', 'angular_velocity')):
            with self.subTest(velocity=representation, heading=heading):
                flow = self.flow(velocity_representation=representation, heading_objective=heading,
                                 size_representation='log', type_noise_stats='data').train()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                dimension = clean.shape[-1]
                outputs = []
                def retain_output(module, inputs, output):
                    output.retain_grad()
                    outputs.append(output)
                hook = flow.model.to_out_m_delta.register_forward_hook(retain_output)
                try:
                    with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)), \
                            patch('src.smart.diffusion.scale_flow.F.cross_entropy', side_effect=AssertionError('unexpected CE')):
                        losses = flow._supervised_loss(clean, agent, feature)
                    total = losses[0].mean()+losses[1]
                    self.assertTrue(torch.isfinite(total))
                    total.backward()
                finally:
                    hook.remove()
                self.assertEqual(len(outputs), 1)
                self.assertEqual(outputs[0].shape[-1], dimension+3)
                gradient = outputs[0].grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient[:, dimension:].abs().sum().item(), 0.)
                input_gradient = flow.model.proj_in_m_delta.weight.grad
                self.assertTrue(torch.isfinite(input_gradient).all())
                self.assertGreater(input_gradient[:, -3:].abs().sum().item(), 0.)
                self.assertTrue(bool(flow.type_normal_initialized))
                with torch.no_grad():
                    generated = flow.eval().sample(agent, feature, steps=2)
                self.assertEqual(tuple(generated.shape), tuple(clean.shape))
                self.assertTrue(torch.isfinite(generated).all())
                self.assertEqual(tuple(agent['_init_diffusion_type_prediction'].shape), (3, 3))

    def test_raw_type_velocity_and_circular_heading_use_correct_angular_index(self):
        for representation, heading in itertools.product(('vector', 'speed'), ('x0', 'angular_velocity')):
            with self.subTest(velocity=representation, heading=heading):
                flow = self.flow(velocity_representation=representation, heading_objective=heading)
                _, agent, _ = self.inputs()
                clean, _ = flow.model.get_input(agent)
                dim = clean.shape[-1]
                latent = torch.cat((clean, torch.tensor([[2., -4., 7.], [3., 4., -2.], [-5., 2., 3.]])), -1)
                prediction = torch.cat((clean, torch.tensor([[-8., 5., 2.], [7., -1., 4.], [1., 9., -3.]])), -1)
                if heading == 'angular_velocity':
                    prediction = torch.cat((prediction, torch.full((3, 1), .7)), -1)
                time = torch.full((3, 1), .4)
                with patch('src.smart.diffusion.scale_flow.F.softmax', side_effect=AssertionError('feature must not project to simplex')):
                    velocity, x0 = flow._prediction_velocity(latent, time, prediction)
                torch.testing.assert_close(x0[:, dim:], prediction[:, dim:dim+3], atol=0, rtol=0)
                torch.testing.assert_close(velocity[:, dim:], (latent[:, dim:]-prediction[:, dim:dim+3])/.4)
                if heading == 'angular_velocity':
                    theta = torch.atan2(latent[:, 3], latent[:, 2])
                    torch.testing.assert_close(x0[:, 2:4], torch.stack(((theta-.28).cos(), (theta-.28).sin()), -1))
                    torch.testing.assert_close(velocity[:, 2:4], .7*torch.stack((-theta.sin(), theta.cos()), -1))

    def test_sampling_integrates_raw_type_once_without_simplex_projection(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(velocity_representation=representation).eval()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                dim = clean.shape[-1]
                target_type = torch.tensor([[-4., 7., 2.], [9., -3., 2.], [3., 8., -4.]])
                predicted = torch.cat((clean, target_type), -1)
                captured = []
                def predict(latent, time, current, *args, **kwargs):
                    captured.append(latent.clone())
                    return predicted.clone()
                with patch.object(flow.model, 'forward', side_effect=predict), \
                        patch('src.smart.diffusion.scale_flow.F.cross_entropy', side_effect=AssertionError('unexpected CE')):
                    generated = flow.sample(agent, feature, steps=2)
                self.assertEqual(len(captured), 2)
                self.assertEqual(captured[0].shape[-1], dim+3)
                torch.testing.assert_close(captured[1][:, dim:], .5*captured[0][:, dim:]+.5*target_type)
                torch.testing.assert_close(agent['_init_diffusion_type_state'], target_type, atol=1.e-6, rtol=0)
                torch.testing.assert_close(agent['type'], target_type.argmax(-1), atol=0, rtol=0)
                self.assertEqual(generated.shape[-1], dim)
                self.assertEqual(agent['gen_z'].shape[-1], dim)
                self.assertEqual(agent['gen_noise'].shape[-1], dim)

    def test_data_statistics_and_matching_treat_all_feature_channels_equally(self):
        for representation in ('vector', 'speed'):
            flow = self.flow(velocity_representation=representation, type_noise_stats='data',
                             type_match_weight=7., heading_noise='gaussian').train()
            _, agent, _ = self.inputs()
            agent['type'] = torch.tensor([0, 0, 1])
            clean, _ = flow.model.get_input(agent)
            flow._sample_noise(clean, copy.deepcopy(agent))
            dim = clean.shape[-1]
            standard = torch.arange(3*(dim+3)).float().reshape(3, dim+3)/9.-1.
            physical = standard[:, :dim]*flow.model.normal_scale+flow.model.normal_mean
            categories = standard[:, dim:]*flow.type_normal_scale+flow.type_normal_mean
            source = torch.cat((physical, categories), -1)
            target = torch.cat((clean, F.one_hot(agent['type'], 3).to(clean)), -1)
            movable = ~agent['ego_mask']
            scale = torch.cat((flow.model.normal_scale, flow.type_normal_scale), -1)
            with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=standard.clone()), \
                    patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast', return_value=torch.tensor([1, 0])) as match:
                matched = flow._sample_noise(clean, agent)
            self.assertTrue(match.call_args.kwargs['use_all_type'])
            source_cost, target_cost = match.call_args.args[:2]
            torch.testing.assert_close(source_cost[:, None]-target_cost[None],
                                       (source[movable, None]-target[None, movable])/scale)
            torch.testing.assert_close(matched[movable], source[movable][[1, 0]], atol=0, rtol=0)
            torch.testing.assert_close(matched[~movable], source[~movable], atol=0, rtol=0)
            torch.testing.assert_close(flow.type_normal_mean, torch.tensor([[2./3., 1./3., 0.]]))
            torch.testing.assert_close(flow.type_normal_scale, torch.tensor([[math.sqrt(2./9.), math.sqrt(2./9.), .01]]))

    def test_ego_type_and_velocity_masks_remain_independent_in_common_loss(self):
        for velocity_fixed, type_fixed in ((True, False), (False, True)):
            flow = self.flow(velocity_representation='speed', fix_ego_velocity=velocity_fixed,
                             fix_ego_type=type_fixed)
            _, agent, feature = self.inputs()
            clean, _ = flow.model.get_input(agent)
            dim = clean.shape[-1]
            labels = F.one_hot(agent['type'], 3).to(clean)
            source = torch.cat((clean+.25, torch.full((3, 3), -.4)), -1)
            predicted = torch.cat((clean+.3, labels+.7), -1).requires_grad_()
            with patch.object(flow, '_draw_joint_source', side_effect=lambda _: source.clone()), \
                    patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)), \
                    patch.object(flow.model, 'forward', return_value=predicted):
                losses = flow._supervised_loss(clean, agent, feature)
            losses[0].mean().backward()
            ego = agent['ego_mask']
            self.assertEqual(bool(predicted.grad[ego, 6].abs().sum()), not velocity_fixed)
            self.assertEqual(bool(predicted.grad[ego, dim:].abs().sum()), not type_fixed)
            self.assertGreater(predicted.grad[~ego, dim:].abs().sum().item(), 0.)
            with patch.object(flow, '_draw_joint_source', side_effect=lambda _: source.clone()), \
                    patch.object(flow.model, 'forward', return_value=predicted.detach()):
                generated = flow.eval().sample(agent, feature, steps=2)
            if velocity_fixed:
                torch.testing.assert_close(generated[ego, 6], clean[ego, 6], atol=0, rtol=0)
            if type_fixed:
                torch.testing.assert_close(agent['_init_diffusion_type_state'][ego], labels[ego], atol=0, rtol=0)
                torch.testing.assert_close(agent['type'][ego], labels[ego].argmax(-1), atol=0, rtol=0)

    def test_feature_denoising_has_no_gt_type_side_channel_and_supports_eval_mask(self):
        flow = self.flow(count_embedding_type='scenario_dreamer', map_embedding_type='scenario_dreamer',
                         map_id=1, map_label_dropout=0.).eval()
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        types = torch.tensor([[.2, -.8, 1.1], [.3, .9, -.6], [-.7, .4, .5]])
        latent = torch.cat((clean, types), -1)
        time = torch.full((3, 1), .4)
        first, second = copy.deepcopy(agent), copy.deepcopy(agent)
        first['_init_diffusion_type_state'] = torch.randn(3, 3)
        second['_init_diffusion_type_state'] = torch.randn(3, 3)*10.
        second['type'] = torch.tensor([2, 2, 0])
        second['ego_feat'][:, -3:] = torch.tensor([[99., 4., 12.]])
        with torch.no_grad():
            a = flow._denoise(latent, time, first, feature)
            b = flow._denoise(latent, time, second, feature)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        mask = torch.tensor([True, False, True])
        with torch.no_grad():
            masked = flow._denoise(latent, time, copy.deepcopy(agent), feature, eval_mask=mask)
        self.assertEqual(tuple(masked.shape), (2, flow.model.feature_dim))
        self.assertTrue(torch.isfinite(masked).all())

    def test_generated_types_do_not_depend_on_gt_labels_or_type_counts(self):
        flow = self.flow(type_noise_stats='data').train()
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        flow._sample_noise(clean, copy.deepcopy(agent))
        flow.model.normal_mean.fill_(.01)
        flow.model.normal_scale.fill_(1.)
        first, second = copy.deepcopy(agent), copy.deepcopy(agent)
        second['type'] = torch.tensor([2, 2, 0])
        second['ego_feat'][:, -3:] = torch.tensor([[22., 5., 13.]])
        with torch.no_grad():
            torch.manual_seed(341)
            a = flow.eval().sample(first, feature, steps=3)
            torch.manual_seed(341)
            b = flow.sample(second, feature, steps=3)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(first['_init_diffusion_type_prediction'], second['_init_diffusion_type_prediction'], atol=0, rtol=0)
        torch.testing.assert_close(first['type'], second['type'], atol=0, rtol=0)

    def test_empty_active_time_targets_have_finite_zero_common_loss_and_gradients(self):
        flow = self.flow(fix_ego=True)
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        full = torch.cat((clean, F.one_hot(agent['type'], 3).to(clean)), -1)
        prediction = (full+.7).requires_grad_()
        with patch.object(flow, '_sample_time', return_value=torch.tensor([[0.], [.4], [1.]])), \
                patch.object(flow.model, 'forward', return_value=prediction):
            losses = flow._supervised_loss(clean, agent, feature)
        torch.testing.assert_close(losses[0], torch.zeros(3), atol=0, rtol=0)
        losses[0].mean().backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(agent['_init_diffusion_type_metrics']['accuracy']))

    def test_active_experiment_features_backward_and_sample_with_shared_type_head(self):
        flow = self.flow(velocity_representation='speed', type_noise_stats='data',
                         fix_ego_position=True, fix_ego_heading=False, fix_ego_shape=True,
                         fix_ego_velocity=False, fix_ego_type=True, use_ego_embedding=True,
                         ego_context_heading_encoding='sincos', time_embedding_type='scenario_dreamer').train()
        _, agent, feature = self.inputs()
        clean, _ = flow.model.get_input(agent)
        original_types = agent['type'].clone()
        outputs = []
        def retain_output(module, inputs, output):
            output.retain_grad()
            outputs.append(output)
        hook = flow.model.to_out_m_delta.register_forward_hook(retain_output)
        try:
            with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                loss = flow._supervised_loss(clean, agent, feature)
            (loss[0].mean()+loss[1]).backward()
        finally:
            hook.remove()
        gradient = outputs[0].grad
        ego = agent['ego_mask']
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[ego, 2:4].abs().sum().item(), 0.)
        self.assertGreater(gradient[ego, 6].abs().sum().item(), 0.)
        for indices in (slice(0, 2), slice(4, 6), slice(7, 10)):
            torch.testing.assert_close(gradient[ego, indices], torch.zeros_like(gradient[ego, indices]), atol=0, rtol=0)
        self.assertGreater(gradient[~ego, 7:10].abs().sum().item(), 0.)
        self.assertGreater(flow.model.ego_a_emb.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            generated = flow.eval().sample(agent, feature, steps=3)
        self.assertEqual(tuple(generated.shape), (3, 7))
        torch.testing.assert_close(generated[ego, :2], clean[ego, :2], atol=0, rtol=0)
        torch.testing.assert_close(generated[ego, 4:6], clean[ego, 4:6], atol=0, rtol=0)
        torch.testing.assert_close(agent['type'][ego], original_types[ego], atol=0, rtol=0)
        torch.testing.assert_close(generated[:, 2:4].norm(dim=-1), torch.ones(3), atol=1.e-6, rtol=0)

    def test_feature_checkpoint_and_ema_roundtrip_reject_old_model_layout(self):
        enabled = self.wrapper(type_noise_stats='data', use_ema=True)
        _, agent, feature = self.inputs()
        clean, _ = enabled.G1.model.get_input(agent)
        with patch.object(enabled.G1, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = enabled.G1._supervised_loss(clean, agent, feature)
        (losses[0].mean()+losses[1]).backward()
        with torch.no_grad():
            for parameter in enabled.G1.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-.001)
        enabled.update_ema()
        target = self.wrapper(type_noise_stats='data', use_ema=True)
        target.load_state_dict(enabled.state_dict(), strict=True)
        self.assertEqual(target.type_process, 'feature')
        for key, value in enabled.G1.state_dict().items():
            torch.testing.assert_close(value, target.G1.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(len(target.ema.shadow_params), len(list(target.G1.parameters())))
        for first, second in zip(enabled.ema.shadow_params, target.ema.shadow_params):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        for process in ('separate', 'joint'):
            old = self.wrapper(type_process=process, type_noise_stats='data', use_ema=True)
            with self.subTest(process=process), self.assertRaisesRegex(RuntimeError, 'size mismatch|feature|type_as_feature|Missing key'):
                target.load_state_dict(old.state_dict(), strict=True)

    def test_legacy_warmstart_preserves_backbone_but_blocks_untrained_feature_evaluation(self):
        old = self.wrapper(type_process='joint', use_ema=True)
        target = self.wrapper(use_ema=True)
        incompatible = target.load_state_dict(old.state_dict(), strict=False)
        self.assertIn('G1.model.proj_in_m_delta.weight', incompatible.missing_keys)
        for name, value in old.G1.model.state_dict().items():
            if name.startswith(('a2a_attn_layers.', 'pt2a_attn_layers.', 'edge_encoder.', 'ego_embed.')):
                torch.testing.assert_close(value, target.G1.model.state_dict()[name], atol=0, rtol=0)
        for parameter, averaged in zip(target.G1.parameters(), target.ema.shadow_params):
            torch.testing.assert_close(parameter, averaged, atol=0, rtol=0)
        _, agent, feature = self.inputs()
        with self.assertRaisesRegex(ValueError, 'train|finetune|type-feature'):
            target.eval()._infer(agent, feature)
        with patch.object(target.G1, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = target.train()._train(agent, feature, agent['batch'])
        sum(losses).backward()
        self.assertFalse(target._type_head_missing_on_load)
        with torch.no_grad():
            inference = target.eval()._infer(agent, feature)
        self.assertEqual(len(inference), 5)
        self.assertTrue(all(torch.isfinite(value).all() for value in inference))

    def test_uninitialized_data_source_rejects_sampling_without_rng_or_stat_changes(self):
        for training in (True, False):
            flow = self.flow(type_noise_stats='data').train(training)
            _, agent, feature = self.inputs()
            before = torch.get_rng_state().clone()
            before_mean = flow.type_normal_mean.clone()
            before_scale = flow.type_normal_scale.clone()
            with self.assertRaisesRegex(RuntimeError, 'stat|train|initializ'):
                flow.sample(agent, feature, steps=2)
            torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
            torch.testing.assert_close(flow.type_normal_mean, before_mean, atol=0, rtol=0)
            torch.testing.assert_close(flow.type_normal_scale, before_scale, atol=0, rtol=0)
            self.assertFalse(bool(flow.type_normal_initialized))

    def test_current_train_and_eval_configs_select_feature_data_mode(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        prefix = 'model.model_config.decoder.init_diffusion'
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                selected = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                self.assertEqual(OmegaConf.select(selected, prefix+'.type_process'), 'feature')
                self.assertEqual(OmegaConf.select(selected, prefix+'.type_noise_stats'), 'data')


if __name__ == '__main__':
    unittest.main()
