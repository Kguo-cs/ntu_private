"""Ego conditioning fixes selected state groups while training the remaining ones."""

import copy
import io
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate


GROUPS = ('position', 'heading', 'shape', 'velocity', 'type')
VELOCITY_ONLY = dict(fix_ego=False, fix_ego_position=True, fix_ego_heading=True,
                     fix_ego_shape=True, fix_ego_velocity=False, fix_ego_type=True)


class InitDiffusionEgoPartialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(930)

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
        shapes = torch.stack((4.+types.float(), 1.+types.float()), -1)
        tokens = torch.zeros(len(types), 2, 2, 4, 2)
        tokens[:, 0, -1, :, 0] = (types.float()+1.)[:, None]
        tokens[:, 1, -1, :, 0] = (types.float()+4.)[:, None]
        return shapes, tokens, tokens[:, :, -1].contiguous()

    @classmethod
    def processor(cls, **options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2)/dt,
                      _get_agent_tokens=cls.library)
        values.update(options)
        return SimpleNamespace(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **options)

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

    @staticmethod
    def expected_mask(clean, flags):
        result = torch.zeros_like(clean, dtype=torch.bool)
        for group, fields in (('position', slice(0, 2)), ('heading', slice(2, 4)),
                              ('shape', slice(4, 6)), ('velocity', slice(6, None))):
            result[1, fields] = flags[f'fix_ego_{group}']
        return result

    def test_unspecified_and_null_groups_inherit_the_legacy_switch(self):
        for master in (True, False):
            for nullable in (False, True):
                with self.subTest(fix_ego=master, explicit_null=nullable):
                    options = dict(fix_ego=master)
                    if nullable:
                        options.update({f'fix_ego_{group}': None for group in GROUPS})
                    wrapper = self.wrapper(**options)
                    for model in (wrapper, wrapper.G1, wrapper.G1.model):
                        self.assertEqual(model.fix_ego, master)
                        for group in GROUPS:
                            self.assertIs(getattr(model, f'fix_ego_{group}'), master)
        wrapper = self.wrapper(**VELOCITY_ONLY)
        for model in (wrapper, wrapper.G1, wrapper.G1.model):
            for group in GROUPS:
                self.assertIs(getattr(model, f'fix_ego_{group}'), group != 'velocity')

    def test_group_options_require_actual_booleans_or_null(self):
        for group, invalid in itertools.product(GROUPS, ('false', 0, 1, [], {})):
            option = f'fix_ego_{group}'
            for constructor in (self.flow, self.wrapper):
                with self.subTest(option=option, value=invalid, constructor=constructor):
                    with self.assertRaisesRegex(ValueError, option):
                        constructor(**{option: invalid})

    def test_field_masks_preserve_role_and_allow_a_nonzero_partial_ego_time(self):
        flow = self.flow(**VELOCITY_ONLY)
        clean, agent, _ = self.inputs()
        expected = self.expected_mask(clean, VELOCITY_ONLY)
        original = agent['ego_mask'].clone()
        torch.testing.assert_close(flow._conditioned_state_mask(agent, clean), expected)
        torch.testing.assert_close(flow._conditioned_agent_mask(agent), torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(flow._conditioned_type_mask(agent), original)
        latent = clean + .25
        time = torch.full((3, 1), .4)
        flow._fix_conditioned_agents(clean, latent, time, agent)
        torch.testing.assert_close(latent[expected], clean[expected], atol=0, rtol=0)
        torch.testing.assert_close(latent[~expected], (clean+.25)[~expected], atol=0, rtol=0)
        torch.testing.assert_close(time, torch.full((3, 1), .4), atol=0, rtol=0)
        torch.testing.assert_close(agent['ego_mask'], original)

    def test_noise_and_interpolation_fix_only_selected_coordinates_across_representations(self):
        for representation, size, heading, source in itertools.product(
                ('vector', 'speed'), ('linear', 'log'), ('circular', 'gaussian'),
                ('gaussian', 'uniform')):
            with self.subTest(velocity=representation, size=size, heading=heading, source=source):
                flow = self.flow(**VELOCITY_ONLY, velocity_representation=representation,
                                 size_representation=size, heading_noise=heading,
                                 pos_source=source, shape_source=source, velocity_source=source,
                                 generate_type=True)
                _, agent, _ = self.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                mask = self.expected_mask(clean, VELOCITY_ONLY)
                torch.testing.assert_close(noise[mask], clean[mask], atol=0, rtol=0)
                torch.testing.assert_close(latent[mask], clean[mask], atol=0, rtol=0)
                self.assertFalse(torch.equal(noise[1, 6:], clean[1, 6:]))
                torch.testing.assert_close(latent[1, 6:], .6*clean[1, 6:]+.4*noise[1, 6:])
                torch.testing.assert_close(time, torch.full((3, 1), .4), atol=0, rtol=0)
                one_hot = F.one_hot(agent['type'], 3).to(clean)
                torch.testing.assert_close(agent['_init_diffusion_type_source'][1], one_hot[1], atol=0, rtol=0)
                torch.testing.assert_close(agent['_init_diffusion_type_state'][1], one_hot[1], atol=0, rtol=0)

    def test_partial_noise_reserves_ego_source_before_hungarian_matching(self):
        flow = self.flow(**VELOCITY_ONLY, heading_noise='gaussian')
        clean, agent, _ = self.inputs()
        endpoint = clean[[2, 0, 1]] + .3
        with patch('src.smart.diffusion.scale_flow._noise_endpoint', return_value=endpoint.clone()), \
                patch('src.smart.diffusion.scale_flow.get_closest_sum_idx_fast',
                      return_value=torch.tensor([1, 0])) as matching:
            noise = flow._sample_noise(clean, agent)
        torch.testing.assert_close(noise[1, :6], clean[1, :6], atol=0, rtol=0)
        torch.testing.assert_close(noise[1, 6:], endpoint[1, 6:], atol=0, rtol=0)
        torch.testing.assert_close(noise[[0, 2]], endpoint[[2, 0]], atol=0, rtol=0)
        self.assertEqual(matching.call_args.args[0].shape[0], 2)

    def test_reconstruction_has_gradients_exactly_on_unfixed_ego_fields(self):
        patterns = [VELOCITY_ONLY]
        patterns += [dict(fix_ego=False, **{f'fix_ego_{group}': group == fixed for group in GROUPS})
                     for fixed in GROUPS[:4]]
        for representation, size, flags in itertools.product(('vector', 'speed'), ('linear', 'log'), patterns):
            with self.subTest(velocity=representation, size=size, flags=flags):
                flow = self.flow(**flags, velocity_representation=representation, size_representation=size)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                prediction = (clean + .2).detach().requires_grad_()
                with patch.object(flow, '_prepare_supervised_batch',
                                  return_value=(clean, torch.full((3, 1), .4), clean)), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    losses = flow._supervised_loss(clean, agent, feature)
                losses[0].sum().backward()
                mask = self.expected_mask(clean, flags)[1]
                torch.testing.assert_close(prediction.grad[1, mask], torch.zeros_like(prediction.grad[1, mask]), atol=0, rtol=0)
                self.assertTrue((prediction.grad[1, ~mask].abs() > 0).all())
                self.assertTrue(torch.isfinite(prediction.grad).all())
                for index, group in ((2, 'position'), (3, 'heading'), (4, 'shape'), (5, 'velocity')):
                    self.assertEqual(losses[index][1].item() == 0., flags[f'fix_ego_{group}'])

    def test_angular_heading_loss_masks_heading_independently_of_velocity(self):
        for representation, fixed_heading in itertools.product(('vector', 'speed'), (True, False)):
            flags = dict(fix_ego=False, fix_ego_position=True, fix_ego_shape=True,
                         fix_ego_heading=fixed_heading, fix_ego_velocity=not fixed_heading)
            flow = self.flow(**flags, velocity_representation=representation,
                             heading_objective='angular_velocity')
            _, agent, feature = self.inputs()
            clean, _ = flow.model.get_input(agent)
            theta = torch.atan2(clean[:, 3], clean[:, 2])
            delta = torch.tensor([.2, -.4, .6])
            noise = clean.clone()
            noise[:, 2:4] = torch.stack(((theta+delta).cos(), (theta+delta).sin()), -1)
            time = torch.full((3, 1), .4)
            omega = (delta+.5).requires_grad_()
            prediction = torch.cat((clean, omega[:, None]), -1)
            with patch.object(flow, '_prepare_supervised_batch',
                              return_value=(noise, time, _circular_interpolate(clean, noise, time))), \
                    patch.object(flow.model, 'forward', return_value=prediction):
                losses = flow._supervised_loss(clean, agent, feature)
            losses[3].sum().backward()
            self.assertAlmostEqual(losses[3][1].item(), 0. if fixed_heading else .25, places=6)
            self.assertAlmostEqual(omega.grad[1].item(), 0. if fixed_heading else 1., places=6)

    def test_magnitude_auxiliary_loss_uses_velocity_conditioning(self):
        for fixed_velocity in (True, False):
            flow = self.flow(fix_ego=False, fix_ego_position=True, fix_ego_heading=True,
                             fix_ego_shape=True, fix_ego_velocity=fixed_velocity,
                             speed_loss_weight=1., speed_loss_scale=1.)
            clean, agent, feature = self.inputs()
            prediction = clean.clone().requires_grad_()
            with torch.no_grad():
                prediction[1, 6:] *= 2
            with patch.object(flow, '_prepare_supervised_batch',
                              return_value=(clean, torch.full((3, 1), .4), clean)), \
                    patch.object(flow.model, 'forward', return_value=prediction):
                flow._supervised_loss(clean, agent, feature)
            self.assertAlmostEqual(agent['_init_diffusion_speed_metrics']['loss'].item(),
                                   0. if fixed_velocity else 5./3, places=6)

    def test_type_source_ce_and_gradients_follow_type_flag_with_partial_continuous_state(self):
        for fixed_type in (True, False):
            flags = dict(VELOCITY_ONLY, fix_ego_type=fixed_type)
            flow = self.flow(**flags, generate_type=True, type_loss_weight=.7)
            clean, agent, feature = self.inputs()
            logits = torch.tensor([[1., 0., -1.], [0., -1., 1.], [.2, -.1, .4]], requires_grad=True)

            def predict(latent, time, current, *args, **kwargs):
                current['_init_diffusion_type_logits'] = logits
                return clean.clone()

            zeros = (torch.zeros(3), torch.zeros(()), *(torch.zeros(3) for _ in range(4)))
            with patch.object(flow, '_sample_time', return_value=torch.tensor([[.2], [.4], [1.]])), \
                    patch.object(flow.model, 'forward', side_effect=predict), \
                    patch('src.smart.diffusion.scale_flow.get_diff_loss', return_value=zeros):
                losses = flow._supervised_loss(clean, agent, feature)
            active = torch.tensor([True, not fixed_type, False])
            expected = F.cross_entropy(logits[active], agent['type'][active])
            torch.testing.assert_close(agent['_init_diffusion_type_metrics']['loss'], expected.detach())
            torch.testing.assert_close(losses[0], (.7*expected).expand(3))
            losses[0].mean().backward()
            torch.testing.assert_close(logits.grad[~active], torch.zeros_like(logits.grad[~active]), atol=0, rtol=0)
            self.assertTrue((logits.grad[active].abs().sum(-1) > 0).all())
            one_hot = F.one_hot(agent['type'], 3).to(clean)
            source = agent['_init_diffusion_type_source']
            if fixed_type:
                torch.testing.assert_close(source[1], one_hot[1], atol=0, rtol=0)
            else:
                self.assertFalse(torch.equal(source[1], one_hot[1]))

    def test_free_type_with_all_continuous_fields_fixed_still_has_noise_time_ce_and_sampling(self):
        flow = self.flow(fix_ego=True, fix_ego_type=False, generate_type=True)
        clean, agent, feature = self.inputs()
        logits = torch.tensor([[1., 0., -1.], [3., -1., 0.], [.2, -.1, .4]], requires_grad=True)

        def predict(latent, time, current, *args, **kwargs):
            current['_init_diffusion_type_logits'] = logits
            return clean.clone()

        with patch.object(flow, '_sample_time', return_value=torch.tensor([[.2], [.4], [1.]])), \
                patch.object(flow.model, 'forward', side_effect=predict):
            losses = flow._supervised_loss(clean, agent, feature)
        torch.testing.assert_close(flow._conditioned_agent_mask(agent), torch.zeros(3, dtype=torch.bool))
        self.assertFalse(torch.equal(agent['_init_diffusion_type_source'][1], F.one_hot(agent['type'][1], 3).to(clean)))
        expected = F.cross_entropy(logits[:2], agent['type'][:2])
        torch.testing.assert_close(agent['_init_diffusion_type_metrics']['loss'], expected.detach())
        losses[0].mean().backward()
        self.assertGreater(logits.grad[1].abs().sum().item(), 0.)

        sampled_agent = copy.deepcopy(agent)
        recorded = []

        def sample_predict(latent, time, current, *args, **kwargs):
            recorded.append((latent.clone(), time.clone(), current['_init_diffusion_type_state'].clone()))
            current['_init_diffusion_type_logits'] = logits.detach().clone()
            return clean + .5

        with patch.object(flow.model, 'forward', side_effect=sample_predict):
            generated = flow.eval().sample(sampled_agent, feature, steps=2)
        for latent, time, _ in recorded:
            torch.testing.assert_close(latent[1], clean[1], atol=0, rtol=0)
            self.assertGreater(time[1].item(), 0.)
        torch.testing.assert_close(generated[1], clean[1], atol=0, rtol=0)
        torch.testing.assert_close(recorded[1][2][1],
                                   .5*recorded[0][2][1]+.5*logits.detach().softmax(-1)[1],
                                   atol=2.e-7, rtol=0)
        self.assertEqual(sampled_agent['type'][1].item(), 0)

    def test_fixed_velocity_preserves_world_velocity_when_heading_is_generated(self):
        heading_modes = (('circular', 'x0'), ('circular', 'angular_velocity'), ('gaussian', 'x0'))
        for representation, size, (noise_mode, objective) in itertools.product(
                ('vector', 'speed'), ('linear', 'log'), heading_modes):
            with self.subTest(velocity=representation, size=size, heading_noise=noise_mode, objective=objective):
                flow = self.flow(fix_ego=False, fix_ego_position=True, fix_ego_heading=False,
                                 fix_ego_shape=True, fix_ego_velocity=True, fix_ego_type=True,
                                 velocity_representation=representation, size_representation=size,
                                 heading_noise=noise_mode, heading_objective=objective).eval()
                _, agent, feature = self.inputs()
                del agent['expert_input']
                agent['batch_ego_pos'][:] = torch.tensor([100., 200.])
                agent['batch_ego_heading'].fill_(.4)
                agent['initial_pos'][1] = torch.tensor([101., 202.])
                agent['initial_heading'][1] = .7
                reference_heading = agent['initial_heading'][1].clone()
                reference_velocity = agent['local_vel'][1].clone()
                expected_world_velocity = torch.stack((
                    reference_velocity[0]*reference_heading.cos()-reference_velocity[1]*reference_heading.sin(),
                    reference_velocity[0]*reference_heading.sin()+reference_velocity[1]*reference_heading.cos()))
                clean, _ = flow.model.get_input(agent)
                target = clean.clone()
                theta = torch.tensor([-.2, -.6, .9])
                target[:, 2:4] = torch.stack((theta.cos(), theta.sin()), -1)
                target[:, 6:] += 9.
                noise = target.clone()
                theta1 = theta+.4
                noise[:, 2:4] = torch.stack((theta1.cos(), theta1.sin()), -1)
                prediction = (target if objective == 'x0' else
                              torch.cat((target, torch.full((3, 1), .4)), -1))
                with patch('src.smart.diffusion.scale_flow._noise_endpoint', return_value=noise.clone()), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    generated = flow.sample(agent, feature, steps=2)
                torch.testing.assert_close(generated[1, 6:], clean[1, 6:], atol=0, rtol=0)
                torch.testing.assert_close(generated[1, 2:4], target[1, 2:4], atol=2.e-6, rtol=0)
                output = flow.model.get_output(generated, agent)
                self.assertAlmostEqual(output[1][1, 0].item(), -.2, places=6)
                torch.testing.assert_close(output[3][1], expected_world_velocity, atol=2.e-6, rtol=0)

    def test_physical_output_restores_exact_fixed_pose_and_shape_after_log_roundtrip(self):
        for representation in ('vector', 'speed'):
            flow = self.flow(**VELOCITY_ONLY, velocity_representation=representation,
                             size_representation='log')
            _, agent, _ = self.inputs()
            del agent['expert_input']
            agent['batch_ego_pos'][:] = torch.tensor([100., 200.])
            agent['batch_ego_heading'].fill_(.23456)
            agent['initial_pos'][1] = torch.tensor([101.12345, 202.98765])
            agent['initial_heading'][1] = .71321
            clean, _ = flow.model.get_input(agent)
            prediction = clean.clone()
            prediction[1, :2] += 50.
            prediction[1, 2:4] = torch.tensor([0., 1.])
            prediction[1, 4:6] += 1.
            prediction[1, 6:] = 9.
            output = flow.model.get_output(prediction, agent)
            torch.testing.assert_close(output[0][1, 0], agent['initial_pos'][1], atol=0, rtol=0)
            torch.testing.assert_close(output[1][1, 0], agent['initial_heading'][1], atol=0, rtol=0)
            torch.testing.assert_close(output[2][1], agent['shape'][1], atol=0, rtol=0)
            local = torch.tensor([9., 0.]) if representation == 'speed' else torch.tensor([9., 9.])
            theta = agent['initial_heading'][1]
            expected_velocity = torch.stack((local[0]*theta.cos()-local[1]*theta.sin(),
                                             local[0]*theta.sin()+local[1]*theta.cos()))
            torch.testing.assert_close(output[3][1], expected_velocity, atol=2.e-6, rtol=0)

    def test_fixed_velocity_uses_cached_reference_heading_when_world_heading_is_absent(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(fix_ego=False, fix_ego_velocity=True, fix_ego_heading=False,
                                 velocity_representation=representation)
                _, agent, _ = self.inputs()
                del agent['initial_heading']
                agent['batch_ego_heading'].fill_(.4)
                prediction, _ = flow.model.get_input(agent)
                prediction = prediction.clone()
                prediction[1, 2:4] = torch.tensor([torch.cos(torch.tensor(.6)), torch.sin(torch.tensor(.6))])
                prediction[1, 6:] += 9.
                output = flow.model.get_output(prediction, agent)
                reference_velocity = agent['local_vel'][1]
                reference_heading = torch.tensor(.4)
                expected = torch.stack((
                    reference_velocity[0]*reference_heading.cos()-reference_velocity[1]*reference_heading.sin(),
                    reference_velocity[0]*reference_heading.sin()+reference_velocity[1]*reference_heading.cos()))
                self.assertAlmostEqual(output[1][1, 0].item(), 1., places=6)
                torch.testing.assert_close(output[3][1], expected, atol=2.e-6, rtol=0)

    def test_real_ego_only_network_trains_velocity_with_fixed_type_pose_and_shape(self):
        for representation, objective in itertools.product(('vector', 'speed'), ('x0', 'angular_velocity')):
            with self.subTest(velocity=representation, objective=objective):
                flow = self.flow(**VELOCITY_ONLY, generate_type=True,
                                 velocity_representation=representation, size_representation='log',
                                 heading_objective=objective, velocity_source='uniform').train()
                _, agent, feature = self.inputs()
                for key, value in list(agent.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 3:
                        agent[key] = value[1:2].clone()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((1, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                loss = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                head = flow.model.to_out_m_delta.mlp[-1]
                self.assertGreater(head.weight.grad[6:].abs().sum().item(), 0.)
                torch.testing.assert_close(head.weight.grad[:6], torch.zeros_like(head.weight.grad[:6]), atol=0, rtol=0)
                for parameter in flow.model.to_out_type.parameters():
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=0, rtol=0)
                self.assertEqual(agent['_init_diffusion_type_metrics']['loss'].item(), 0.)
                self.assertTrue(all(torch.isfinite(parameter.grad).all()
                                    for parameter in flow.parameters() if parameter.grad is not None))

    def test_oracle_sampler_clamps_fields_every_step_and_generates_velocity_and_other_types(self):
        for representation, size, heading in itertools.product(
                ('vector', 'speed'), ('linear', 'log'), ('circular', 'gaussian')):
            with self.subTest(velocity=representation, size=size, heading=heading):
                flow = self.flow(**VELOCITY_ONLY, generate_type=True,
                                 velocity_representation=representation, size_representation=size,
                                 heading_noise=heading).eval()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                target = clean.clone()
                target[:, :2] += torch.tensor([8., -3.])
                angle = torch.tensor([.2, -.6, 1.8])
                target[:, 2:4] = torch.stack((angle.cos(), angle.sin()), -1)
                target[:, 4:6] += .3
                target[:, 6:] += 4.
                noise = target.clone()
                noise[:, :2] += 3.
                noise[:, 6:] += 2.
                logits = torch.tensor([[-3., 4., -2.], [4., -1., 0.], [4., -3., -2.]])
                original_types = agent['type'].clone()
                recorded = []

                def predict(latent, time, current, *args, **kwargs):
                    recorded.append((latent.clone(), time.clone(), current['_init_diffusion_type_state'].clone()))
                    current['_init_diffusion_type_logits'] = logits.clone()
                    return target.clone()

                with patch('src.smart.diffusion.scale_flow._noise_endpoint', return_value=noise.clone()), \
                        patch.object(flow.model, 'forward', side_effect=predict):
                    generated = flow.sample(agent, feature, steps=2)
                mask = self.expected_mask(clean, VELOCITY_ONLY)
                torch.testing.assert_close(generated, torch.where(mask, clean, target), atol=2.e-6, rtol=0)
                torch.testing.assert_close(agent['gen_z'], generated, atol=0, rtol=0)
                for latent, time, categorical in recorded:
                    torch.testing.assert_close(latent[mask], clean[mask], atol=0, rtol=0)
                    self.assertGreater(time[1].item(), 0.)
                    torch.testing.assert_close(categorical[1], F.one_hot(original_types[1], 3).to(categorical), atol=0, rtol=0)
                torch.testing.assert_close(recorded[1][0][1, 6:], .5*noise[1, 6:]+.5*target[1, 6:])
                expected_types = logits.argmax(-1)
                expected_types[1] = original_types[1]
                torch.testing.assert_close(agent['type'], expected_types, atol=0, rtol=0)
                output = flow.model.get_output(generated, agent)
                torch.testing.assert_close(output[0][1, 0], agent['initial_pos'][1], atol=0, rtol=0)
                torch.testing.assert_close(output[1][1, 0], agent['initial_heading'][1], atol=0, rtol=0)
                torch.testing.assert_close(output[2][1], agent['shape'][1], atol=0, rtol=0)
                expected_velocity = (torch.tensor([target[1, 6], 0.]) if representation == 'speed'
                                     else target[1, 6:])
                torch.testing.assert_close(output[3][1], expected_velocity, atol=2.e-6, rtol=0)
                self.assertFalse(torch.equal(output[3][1], agent['local_vel'][1]))

    def test_partial_options_preserve_strict_checkpoints_and_resolved_runtime_ema_flags(self):
        source = self.wrapper(fix_ego=True, use_ema=True, ema_decay=.9, generate_type=True)
        target = self.wrapper(**VELOCITY_ONLY, use_ema=True, ema_decay=.7, generate_type=True)
        self.assertEqual(source.get_extra_state()['ego_conditioning'], {group: True for group in GROUPS})
        self.assertEqual(target.get_extra_state()['ego_conditioning'],
                         {group: group != 'velocity' for group in GROUPS})
        self.assertEqual(set(source.state_dict()), set(target.state_dict()))
        for key, value in source.G1.state_dict().items():
            self.assertEqual(value.shape, target.G1.state_dict()[key].shape)
        source.update_ema()
        saved = io.BytesIO()
        torch.save(source.state_dict(), saved)
        saved.seek(0)
        result = target.load_state_dict(torch.load(saved, weights_only=False), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for model in (target, target.G1, target.G1.model):
            for group in GROUPS:
                self.assertIs(getattr(model, f'fix_ego_{group}'), group != 'velocity')
        self.assertEqual(target.get_extra_state()['ego_conditioning'],
                         {group: group != 'velocity' for group in GROUPS})
        self.assertEqual(target.ema.num_updates, 1)
        self.assertEqual(target.ema.decay, .9)
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        target.update_ema()
        self.assertEqual(target.ema.num_updates, 2)

    def test_partial_conditioning_rejects_sde_rl_and_refiner(self):
        for options, processor, gail in ((dict(heading_noise='gaussian'), self.processor(), True),
                                         (dict(use_rl=True), self.processor(), False),
                                         ({}, self.processor(use_refiner=True), False)):
            with self.subTest(options=options, gail=gail), self.assertRaisesRegex(ValueError, '(?i)ego'):
                Flow(self.args(fix_ego=True, fix_ego_velocity=False, **options), processor, gail)

    def test_current_configs_fix_pose_shape_type_and_generate_velocity(self):
        root = Path(__file__).resolve().parents[1]
        generic = OmegaConf.load(root/'configs/model/smart.yaml').model_config.decoder.init_diffusion
        for group in GROUPS:
            self.assertIsNone(generic[f'fix_ego_{group}'])
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                with self.subTest(experiment=experiment):
                    config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                    decoder = config.model.model_config.decoder.init_diffusion
                    self.assertFalse(decoder.fix_ego)
                    for group in GROUPS:
                        self.assertIs(decoder[f'fix_ego_{group}'], group != 'velocity')
                    self.assertEqual(config.model.model_config.token_processor.init_map_half_extent, 32.)
                    self.assertEqual(config.model.model_config.token_processor.init_map_crop, 'circle')


if __name__ == '__main__':
    unittest.main()
