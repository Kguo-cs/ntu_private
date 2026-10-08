"""Corrupt dimensions remain unsupervised while every agent stays in the flow."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.smart.diffusion.diffusion_utils import get_diff_loss
from test_init_diffusion_log_size import InitDiffusionLogSizeTest


class InitDiffusionInvalidSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(818)
        self.fixture = InitDiffusionLogSizeTest()

    def inputs(self, invalid=-1., field=4, representation='vector'):
        physical, agent, feature = self.fixture.inputs(representation)
        physical[0, field] = invalid
        agent['expert_input'] = physical.clone()
        agent['shape'] = physical[:, 4:6].clone()
        return physical, agent, feature

    def test_mask_policy_keeps_every_agent_and_raw_annotations(self):
        for representation in ('vector', 'speed'):
            for value in (-1., 0., float('nan'), float('inf')):
                for field in (4, 5):
                    with self.subTest(representation=representation, value=value, field=field):
                        physical, agent, _ = self.inputs(value, field, representation)
                        before = copy.deepcopy(agent)
                        flow = self.fixture.flow(velocity_representation=representation)
                        clean, target = flow.model.train().get_input(agent)
                        self.assertEqual(clean.shape, physical.shape)
                        self.assertTrue(torch.isfinite(clean).all())
                        torch.testing.assert_close(clean, target, atol=0, rtol=0)
                        valid = torch.isfinite(physical[:, 4:6]) & (physical[:, 4:6] > 0)
                        torch.testing.assert_close(agent['_init_diffusion_size_valid_mask'], valid)
                        torch.testing.assert_close(agent['shape'], before['shape'], equal_nan=True)
                        for name in ('batch', 'type', 'ego_mask', 'ego_feat'):
                            torch.testing.assert_close(agent[name], before[name], atol=0, rtol=0)
                        other = tuple(index for index in range(clean.shape[-1]) if index not in (4, 5))
                        torch.testing.assert_close(clean[:, other], physical[:, other], atol=0, rtol=0)
                        torch.testing.assert_close(clean[:, 4:6][valid], physical[:, 4:6][valid].log())

    def test_typed_geometric_fill_and_global_fallback_do_not_train_invalid_sizes(self):
        physical, agent, _ = self.fixture.inputs()
        physical = physical[[0, 1, 1, 2, 2]].clone()
        physical[:, 4:6] = torch.tensor([[-1., 2.], [4., 2.], [9., 8.], [0., 0.], [0., 0.]])
        agent['expert_input'] = physical
        agent['type'] = torch.tensor([0, 0, 0, 1, 2])
        agent['batch'] = torch.zeros(5, dtype=torch.long)
        agent['ego_mask'] = torch.zeros(5, dtype=torch.bool)
        agent['shape'] = physical[:, 4:6].clone()
        model = self.fixture.denoiser().train()
        clean, _ = model.get_input(agent)
        decoded = model.state_to_physical(clean)
        torch.testing.assert_close(decoded[0, 4], torch.tensor(6.))
        torch.testing.assert_close(decoded[3:, 4], torch.full((2,), 6.))
        torch.testing.assert_close(decoded[3:, 5], torch.full((2,), 32. ** (1./3.)))
        torch.testing.assert_close(model.normal_mean[0, 4], torch.tensor([4., 9.]).log().mean())
        torch.testing.assert_close(model.normal_scale[0, 4], torch.tensor([4., 9.]).log().std(unbiased=False))
        width_logs = torch.tensor([2., 2., 8.]).log()
        torch.testing.assert_close(model.normal_mean[0, 5], width_logs.mean())
        torch.testing.assert_close(model.normal_scale[0, 5], width_logs.std(unbiased=False))

    def test_all_missing_sizes_use_type_nominals_and_keep_unfitted_size_prior(self):
        _, agent, _ = self.fixture.inputs()
        agent['type'] = torch.tensor([0, 1, 2])
        agent['expert_input'][:, 4:6] = 0.
        agent['shape'][:] = 0.
        model = self.fixture.denoiser().train()
        clean, _ = model.get_input(agent)
        torch.testing.assert_close(model.state_to_physical(clean)[:, 4:6],
                                   torch.tensor([[4.8, 2.], [1., 1.], [2., 1.]]))
        torch.testing.assert_close(model.normal_mean[:, 4:6], torch.zeros(1, 2), atol=0, rtol=0)
        torch.testing.assert_close(model.normal_scale[:, 4:6], torch.ones(1, 2), atol=0, rtol=0)
        self.assertFalse(agent['_init_diffusion_size_valid_mask'].any())

    def test_normalizer_fits_each_dimension_using_only_its_own_valid_annotations(self):
        _, agent, _ = self.fixture.inputs()
        agent['expert_input'][:, 4] = torch.tensor([0., 4., 9.])
        agent['expert_input'][:, 5] = torch.tensor([2., float('nan'), 8.])
        model = self.fixture.denoiser().train()
        clean, _ = model.get_input(agent)
        for field, values in ((4, torch.tensor([4., 9.])), (5, torch.tensor([2., 8.]))):
            torch.testing.assert_close(model.normal_mean[0, field], values.log().mean())
            torch.testing.assert_close(model.normal_scale[0, field], values.log().std(unbiased=False))
        self.assertTrue(torch.isfinite(clean).all())

    def test_cached_masked_input_remains_idempotent_and_preserves_validity_metadata(self):
        _, agent, _ = self.inputs(float('nan'))
        model = self.fixture.denoiser().train()
        first, _ = model.get_input(agent)
        valid = agent['_init_diffusion_size_valid_mask'].clone()
        mean, scale = model.normal_mean.clone(), model.normal_scale.clone()
        for _ in range(3):
            repeated, _ = model.get_input(agent)
            torch.testing.assert_close(repeated, first, atol=0, rtol=0)
            torch.testing.assert_close(agent['_init_diffusion_size_valid_mask'], valid)
            torch.testing.assert_close(model.normal_mean, mean, atol=0, rtol=0)
            torch.testing.assert_close(model.normal_scale, scale, atol=0, rtol=0)
        self.assertEqual(agent['_init_diffusion_size_representation'], 'log')
        self.assertFalse(valid[0, 0])

    def test_encoding_boundary_stays_strict_even_when_training_mask_policy_is_enabled(self):
        for value in (0., -1., float('nan'), float('inf')):
            physical, _, _ = self.inputs(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.fixture.denoiser().train().state_to_model(physical)

    def test_eval_and_explicit_error_policy_report_the_invalid_agent(self):
        for policy, training in (('mask', False), ('error', True), ('error', False)):
            _, agent, _ = self.inputs(-.25)
            model = self.fixture.denoiser(invalid_size_policy=policy).train(training)
            with self.subTest(policy=policy, training=training), self.assertRaises(ValueError) as raised:
                model.get_input(agent)
            message = str(raised.exception)
            self.assertRegex(message, '(batch|scene)')
            self.assertIn('type', message)
            self.assertIn('-0.25', message)

    def test_linear_mode_is_unchanged_by_invalid_size_policy(self):
        for value in (-1., 0., float('nan'), float('inf')):
            physical, agent, _ = self.inputs(value)
            for policy in ('mask', 'error'):
                model = self.fixture.denoiser(size_representation='linear', invalid_size_policy=policy)
                actual, _ = model.get_input(copy.deepcopy(agent))
                torch.testing.assert_close(actual, physical, equal_nan=True)
                torch.testing.assert_close(model.state_to_model(physical), physical, equal_nan=True)
            self.assertNotIn('_init_diffusion_size_valid_mask', agent)

    def test_masked_reconstruction_retains_other_fields_and_valid_width_gradients(self):
        for selected in (None, (0, 1, 4, 5, 6, 7)):
            physical, agent, _ = self.inputs()
            model = self.fixture.denoiser().train()
            clean, _ = model.get_input(agent)
            prediction = (clean + .2).detach().requires_grad_(True)
            mask = torch.ones_like(clean, dtype=torch.bool)
            mask[:, 4:6] = agent['_init_diffusion_size_valid_mask']
            time = torch.full((3, 1), .5)
            losses = get_diff_loss(agent, prediction, clean, time, .05, w_pos=1.,
                                   x_pred=True, reconstruction_mask=mask,
                                   reconstruction_dims=selected)
            error = (prediction-clean).square() * mask
            oracle = (error.sum(-1) if selected is None else error[:, selected].sum(-1))
            # t=.5 has cubic inverse-time weight 8; retain the original /8 denominator.
            torch.testing.assert_close(losses[0], oracle)
            shape_oracle = error[:, 4:6].sum(-1) / 2
            torch.testing.assert_close(losses[4], shape_oracle)
            losses[0].sum().backward()
            self.assertEqual(prediction.grad[0, 4].item(), 0.)
            self.assertGreater(prediction.grad[0, 5].abs().item(), 0.)
            for field in (0, 1, 6, 7):
                self.assertGreater(prediction.grad[0, field].abs().item(), 0.)
            if selected is None:
                self.assertGreater(prediction.grad[0, 2:4].abs().sum().item(), 0.)

    def test_collision_mask_matches_only_valid_pairs_and_retains_their_denominator(self):
        physical, _, _ = self.fixture.inputs()
        physical = physical[[0, 0, 0, 0]].clone()
        physical[:, :2] = torch.tensor([[0., 0.], [0., 0.], [0., 0.], [12., 0.]])
        physical[:, 2:4] = torch.tensor([1., 0.])
        physical[:, 4:6] = torch.tensor([4.8, 2.])
        model = self.fixture.denoiser()
        clean = model.state_to_model(physical)
        prediction = clean.clone()
        prediction[3, :2] = torch.tensor([3., 0.])
        prediction.requires_grad_(True)
        agent = dict(batch=torch.zeros(4, dtype=torch.long), type=torch.zeros(4, dtype=torch.long))
        valid = torch.tensor([True, False, False, True])
        time = torch.full((4, 1), .4)
        actual = get_diff_loss(agent, prediction, clean, time, .05, x_pred=True, use_col=True,
                               state_to_physical=model.state_to_physical, collision_valid_mask=valid)
        subset = dict(batch=torch.zeros(2, dtype=torch.long), type=torch.zeros(2, dtype=torch.long))
        oracle = get_diff_loss(subset, prediction[valid], clean[valid], time[valid], .05,
                               x_pred=True, use_col=True, state_to_physical=model.state_to_physical)
        torch.testing.assert_close(actual[1], oracle[1])
        self.assertGreater(actual[1].item(), 0.)
        actual[1].backward()
        self.assertEqual(prediction.grad[~valid].abs().sum().item(), 0.)
        self.assertGreater(prediction.grad[valid, :2].abs().sum().item(), 0.)

    def test_all_missing_sizes_give_finite_loss_without_shape_or_collision_gradient(self):
        _, agent, _ = self.fixture.inputs()
        agent['expert_input'][:, 4:6] = 0.
        model = self.fixture.denoiser().train()
        clean, _ = model.get_input(agent)
        prediction = (clean + .1).detach().requires_grad_(True)
        mask = torch.ones_like(clean, dtype=torch.bool)
        mask[:, 4:6] = False
        losses = get_diff_loss(agent, prediction, clean, torch.full((3, 1), .5), .05,
                               x_pred=True, use_col=True, reconstruction_mask=mask,
                               collision_valid_mask=torch.zeros(3, dtype=torch.bool),
                               state_to_physical=model.state_to_physical)
        self.assertTrue(all(torch.isfinite(loss).all() for loss in losses))
        self.assertEqual(losses[1].item(), 0.)
        self.assertEqual(losses[4].sum().item(), 0.)
        (losses[0].sum() + losses[1]).backward()
        self.assertEqual(prediction.grad[:, 4:6].abs().sum().item(), 0.)
        self.assertGreater(prediction.grad[:, :2].abs().sum().item(), 0.)

    def test_supervised_adapter_passes_field_and_agent_masks_in_both_velocity_modes(self):
        for representation in ('vector', 'speed'):
            for objective in ('x0', 'angular_velocity'):
                _, agent, feature = self.inputs(representation=representation)
                flow = self.fixture.flow(velocity_representation=representation,
                                          heading_objective=objective)
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)), \
                        patch('src.smart.diffusion.scale_flow.get_diff_loss', wraps=get_diff_loss) as called:
                    losses = flow._supervised_loss(clean, agent, feature)
                field_mask = called.call_args.kwargs['reconstruction_mask']
                self.assertEqual(field_mask.shape, (3, 8))
                torch.testing.assert_close(field_mask[:, 4:6], agent['_init_diffusion_size_valid_mask'])
                self.assertTrue(field_mask[:, :4].all())
                self.assertTrue(field_mask[:, 6:].all())
                torch.testing.assert_close(called.call_args.kwargs['collision_valid_mask'],
                                           agent['_init_diffusion_size_valid_mask'].all(-1))
                self.assertTrue(all(torch.isfinite(loss).all() for loss in losses))
                metrics = agent['_init_diffusion_size_metrics']
                self.assertEqual(float(metrics['invalid_fields']), 1.)
                self.assertEqual(float(metrics['invalid_agents']), 1.)

    def test_policy_is_propagated_and_invalid_choices_are_rejected(self):
        for policy in ('mask', 'error'):
            wrapper = self.fixture.wrapper(invalid_size_policy=policy)
            self.assertEqual(wrapper.invalid_size_policy, policy)
            self.assertEqual(wrapper.G1.invalid_size_policy, policy)
            self.assertEqual(wrapper.G1.model.invalid_size_policy, policy)
        for constructor in (lambda: self.fixture.denoiser(invalid_size_policy='unknown'),
                            lambda: self.fixture.flow(invalid_size_policy='unknown'),
                            lambda: self.fixture.wrapper(invalid_size_policy='unknown')):
            with self.assertRaises(ValueError):
                constructor()

    def test_known_corrupt_real_scene_trains_and_updates_ema_without_removing_agents(self):
        path = Path(__file__).resolve().parents[1] / 'src/waymo_data/full/training_map2_sd/2f0cdd3fea29917c_0_26.pt'
        if not path.exists():
            self.skipTest('local failing cached scene is unavailable')
        source = torch.load(path, map_location='cpu', weights_only=False)['tokenized_agent']
        original_shape = source['shape'].clone()
        count = len(source['type'])
        physical = torch.cat((source['initial_pos'], source['initial_heading'].cos()[:, None],
                              source['initial_heading'].sin()[:, None], source['shape'][:, :2],
                              source['local_vel'][:, :2]), -1)
        agent = copy.deepcopy(source)
        agent.update(batch=torch.zeros(count, dtype=torch.long), num_graphs=1,
                     ego_mask=torch.arange(count) == 0, expert_input=physical,
                     batch_ego_pos=torch.zeros(count, 2), batch_ego_heading=torch.zeros(count),
                     token_traj=torch.zeros(count, 2, 4, 2),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., float(count), 0., 0.]]))
        _, _, feature = self.fixture.inputs()
        agent['initial_map_feature'] = feature
        wrapper = self.fixture.wrapper(use_ema=True, ema_decay=.9, count_embedding_type='scenario_dreamer')
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1.e-3)
        losses = wrapper.train()(agent)
        total = losses[0] + losses[1]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        size_head = wrapper.G1.model.to_out_m_delta.mlp[-1]
        self.assertGreater(size_head.weight.grad[4:6].abs().sum().item(), 0.)
        optimizer.step()
        wrapper.update_ema()
        self.assertTrue(all(torch.isfinite(p).all() for p in wrapper.ema.shadow_params))
        self.assertEqual(agent['expert_input'].shape[0], count)
        self.assertEqual(count, 8)
        self.assertFalse(agent['_init_diffusion_size_valid_mask'][3, 0])
        self.assertTrue(agent['_init_diffusion_size_valid_mask'][3, 1])
        self.assertEqual((~agent['_init_diffusion_size_valid_mask']).sum().item(), 1)
        torch.testing.assert_close(agent['shape'], original_shape)


if __name__ == '__main__':
    unittest.main()
