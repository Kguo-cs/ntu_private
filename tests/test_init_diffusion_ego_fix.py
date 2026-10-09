"""Optional all-agent initialization supervises and generates the original ego row."""

import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate
from src.smart.utils import wrap_angle


class InitDiffusionEgoFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(902)

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
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50., shift=5,
                      token_velocity_in_current_frame=lambda contour, dt: contour.mean(-2) / dt)
        values.update(options)
        return SimpleNamespace(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    @staticmethod
    def inputs():
        state = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                              [0., 0., 1., 0., 4.8, 2.1, 1., 2.],
                              [-20., 30., 0., 1., .9, .6, -2., 0.]])
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

    def test_conditioning_mask_defaults_to_legacy_ego_and_false_keeps_metadata(self):
        _, agent, _ = self.inputs()
        for option in (None, True, False):
            with self.subTest(fix_ego=option):
                flow = self.flow(**({} if option is None else dict(fix_ego=option)))
                original = agent['ego_mask'].clone()
                expected = original if option is not False else torch.zeros_like(original)
                torch.testing.assert_close(flow._conditioned_agent_mask(agent), expected)
                torch.testing.assert_close(agent['ego_mask'], original)
        # Legacy lightweight Flow fixtures did not carry the new option.
        del flow.fix_ego
        torch.testing.assert_close(flow._conditioned_agent_mask(agent), agent['ego_mask'])

    def test_no_fix_hungarian_matches_full_source_rows_including_ego(self):
        flow = self.flow(fix_ego=False, heading_noise='gaussian')
        clean, agent, _ = self.inputs()
        agent['type'].zero_()
        # Positional matching has an unambiguous optimum which moves source
        # rows across the original ego index. All other state fields move too.
        delta = torch.tensor([.1, -.2, .2, .1, .3, .4, .5, .6])
        endpoint = (clean + delta)[torch.tensor([2, 0, 1])]
        with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=endpoint.clone()):
            matched = flow._sample_noise(clean, agent)
        torch.testing.assert_close(matched, clean + delta)
        self.assertFalse(torch.equal(matched[1], clean[1]))
        torch.testing.assert_close(agent['ego_mask'], torch.tensor([False, True, False]))

    def test_fixed_ego_source_cannot_be_reassigned_to_non_ego_targets(self):
        flow = self.flow(fix_ego=True, heading_noise='gaussian')
        clean, agent, _ = self.inputs()
        agent['type'].zero_()
        endpoint = clean[torch.tensor([2, 0, 1])] + .25
        with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=endpoint.clone()):
            matched = flow._sample_noise(clean, agent)
        torch.testing.assert_close(matched[1], clean[1])
        # Original ego source row is excluded before matching, preserving the
        # multiset of source rows available to the two generated agents.
        actual_rows = sorted(tuple(row.tolist()) for row in matched[[0, 2]])
        expected_rows = sorted(tuple(row.tolist()) for row in endpoint[[0, 2]])
        self.assertEqual(actual_rows, expected_rows)

    def test_all_agent_training_replays_generation_source_and_nonzero_ego_time(self):
        for representation, size, uniform in itertools.product(
                ('vector', 'speed'), ('linear', 'log'), (False, True)):
            with self.subTest(velocity=representation, size=size, uniform=uniform):
                source = 'uniform' if uniform else 'gaussian'
                flow = self.flow(fix_ego=False, velocity_representation=representation,
                                 size_representation=size, pos_source=source,
                                 shape_source=source, velocity_source=source)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                eps = torch.linspace(-2., 2., clean.numel()).reshape_as(clean)
                torch.manual_seed(139)
                with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=eps.clone()), \
                        patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    noise, time, latent = flow._prepare_supervised_batch(clean, agent)
                sampled_agent = copy.deepcopy(agent)
                torch.manual_seed(139)
                with patch('src.smart.diffusion.scale_flow.torch.randn', return_value=eps.clone()), \
                        patch.object(flow.model, 'forward', return_value=clean.clone()):
                    flow.eval().sample(sampled_agent, feature, steps=2)
                torch.testing.assert_close(noise, sampled_agent['gen_noise'], atol=0, rtol=0)
                torch.testing.assert_close(time, torch.full((3, 1), .4), atol=0, rtol=0)
                self.assertFalse(torch.equal(noise[1], clean[1]))
                self.assertFalse(torch.equal(latent[1], clean[1]))
                torch.testing.assert_close(latent[:, 2:4].norm(dim=-1), torch.ones(3), atol=2.e-7, rtol=0)

    def test_x0_reconstruction_supervises_all_ego_coordinates_only_when_unfixed(self):
        for fixed, representation, size in itertools.product(
                (True, False), ('vector', 'speed'), ('linear', 'log')):
            with self.subTest(fix_ego=fixed, velocity=representation, size=size):
                flow = self.flow(fix_ego=fixed, velocity_representation=representation,
                                 size_representation=size)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                time = torch.full((3, 1), .4)
                prediction = (clean + .2).detach().requires_grad_()
                with patch.object(flow, '_prepare_supervised_batch', return_value=(clean, time, clean)), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    losses = flow._supervised_loss(clean, agent, feature)
                losses[0].sum().backward()
                ego_gradient = prediction.grad[1]
                self.assertTrue(torch.isfinite(ego_gradient).all())
                if fixed:
                    torch.testing.assert_close(ego_gradient, torch.zeros_like(ego_gradient), atol=0, rtol=0)
                    self.assertEqual(losses[3][1].item(), 0.)
                    self.assertEqual(losses[5][1].item(), 0.)
                else:
                    self.assertTrue((ego_gradient.abs() > 0).all())
                    self.assertGreater(losses[3][1].item(), 0.)
                    self.assertGreater(losses[5][1].item(), 0.)

    def test_angular_velocity_target_and_gradient_include_unfixed_ego(self):
        for fixed, representation in itertools.product((True, False), ('vector', 'speed')):
            with self.subTest(fix_ego=fixed, velocity=representation):
                flow = self.flow(fix_ego=fixed, velocity_representation=representation,
                                 heading_objective='angular_velocity', heading_flow_loss_weight=.7)
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                theta0 = torch.atan2(clean[:, 3], clean[:, 2])
                theta1 = theta0 + torch.tensor([.2, -.4, .6])
                noise = clean.clone()
                noise[:, 2:4] = torch.stack((theta1.cos(), theta1.sin()), dim=-1)
                time = torch.full((3, 1), .4)
                latent = _circular_interpolate(clean, noise, time)
                omega = (wrap_angle(theta1-theta0) + .5).detach().requires_grad_()
                prediction = torch.cat((clean, omega[:, None]), dim=-1)
                with patch.object(flow, '_prepare_supervised_batch', return_value=(noise, time, latent)), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    losses = flow._supervised_loss(clean, agent, feature)
                losses[3].sum().backward()
                self.assertAlmostEqual(losses[3][1].item(), 0. if fixed else .25, places=6)
                self.assertAlmostEqual(omega.grad[1].item(), 0. if fixed else 1., places=6)

    def test_magnitude_loss_uses_unfixed_ego_in_active_agent_mean(self):
        for fixed in (True, False):
            with self.subTest(fix_ego=fixed):
                flow = self.flow(fix_ego=fixed, speed_loss_weight=1., speed_loss_scale=1.)
                clean, agent, feature = self.inputs()
                prediction = clean.clone().requires_grad_()
                with torch.no_grad():
                    prediction[1, 6:8] *= 2
                time = torch.full((3, 1), .4)
                with patch.object(flow, '_prepare_supervised_batch', return_value=(clean, time, clean)), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    flow._supervised_loss(clean, agent, feature)
                expected = 0. if fixed else 5./3
                self.assertAlmostEqual(agent['_init_diffusion_speed_metrics']['loss'].item(), expected, places=6)

    def test_oracle_sampler_generates_ego_without_end_or_step_restoration(self):
        for fixed, representation, size, objective in itertools.product(
                (True, False), ('vector', 'speed'), ('linear', 'log'), ('x0', 'angular_velocity')):
            with self.subTest(fix_ego=fixed, velocity=representation, size=size, objective=objective):
                flow = self.flow(fix_ego=fixed, velocity_representation=representation,
                                 size_representation=size, heading_objective=objective).eval()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                physical = flow.model.state_to_physical(clean)
                physical[:, :2] += torch.tensor([8., -3.])
                theta = torch.tensor([.2, -.6, 1.8])
                physical[:, 2:4] = torch.stack((theta.cos(), theta.sin()), dim=-1)
                physical[:, 4:6] *= 1.2
                physical[:, 6:] += 2
                target = flow.model.state_to_model(physical)
                noise = target.clone()
                noise[:, :2] += 3
                theta1 = theta + .4
                noise[:, 2:4] = torch.stack((theta1.cos(), theta1.sin()), dim=-1)
                prediction = target if objective == 'x0' else torch.cat((target, torch.full((3, 1), .4)), -1)
                with patch('src.smart.diffusion.scale_flow._noise_endpoint', return_value=noise.clone()), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    generated = flow.sample(agent, feature, steps=2)
                expected = target.clone()
                if fixed:
                    expected[1] = clean[1]
                torch.testing.assert_close(generated, expected, atol=2.e-6, rtol=0)
                torch.testing.assert_close(agent['gen_z'], generated)
                if not fixed:
                    self.assertGreater(generated[1, :2].norm().item(), 1.)

    def test_speed_output_uses_generated_ego_heading_and_speed(self):
        for fixed in (True, False):
            flow = self.flow(fix_ego=fixed, velocity_representation='speed')
            _, agent, _ = self.inputs()
            state, _ = flow.model.get_input(agent)
            state = state.clone()
            state[1, :2] = torch.tensor([4., -2.])
            theta = .6
            state[1, 2:4] = torch.tensor([torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))])
            state[1, 6] = 9.
            agent['batch_ego_pos'][:] = torch.tensor([100., 200.])
            agent['batch_ego_heading'].fill_(.4)
            pos, heading, shape, velocity, _ = flow.model.get_output(state, agent)
            self.assertAlmostEqual(heading[1, 0].item(), 1., places=6)
            self.assertFalse(torch.equal(pos[1, 0], agent['batch_ego_pos'][1]))
            # Physical velocity is reconstructed in the generated heading frame.
            angle = torch.tensor(1.)
            local = agent['local_vel'][1] if fixed else torch.tensor([9., 0.])
            expected = torch.stack((local[0]*angle.cos()-local[1]*angle.sin(),
                                    local[0]*angle.sin()+local[1]*angle.cos()))
            torch.testing.assert_close(velocity[1], expected, atol=2.e-6, rtol=0)
            torch.testing.assert_close(shape, state[:, 4:6])

    def test_real_network_backpropagates_with_all_agent_log_uniform_flow(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                flow = self.flow(fix_ego=False, velocity_representation=representation,
                                 size_representation='log', heading_objective='angular_velocity',
                                 pos_source='uniform', shape_source='uniform', velocity_source='uniform').train()
                _, agent, feature = self.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
                    losses = flow._supervised_loss(clean, agent, feature)
                objective = losses[0].mean()+losses[1]
                self.assertTrue(torch.isfinite(objective))
                objective.backward()
                gradients = [parameter.grad for parameter in flow.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_option_preserves_checkpoint_parameter_shapes_and_ema(self):
        wrappers = []
        for fixed in (True, False):
            with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
                wrapper = InitDiffusion(32, 2, 4, self.processor(), False, fix_ego=fixed,
                                       heading_noise='circular', use_ema=True)
            self.assertEqual(wrapper.fix_ego, fixed)
            self.assertEqual(wrapper.G1.fix_ego, fixed)
            self.assertEqual(wrapper.G1.model.fix_ego, fixed)
            wrappers.append(wrapper)
        fixed, generated = wrappers
        self.assertEqual(set(fixed.G1.state_dict()), set(generated.G1.state_dict()))
        for key, value in fixed.G1.state_dict().items():
            self.assertEqual(value.shape, generated.G1.state_dict()[key].shape)
        generated.load_state_dict(fixed.state_dict(), strict=True)
        generated.update_ema()
        self.assertFalse(generated.fix_ego)
        self.assertFalse(generated.G1.model.fix_ego)
        self.assertEqual(len(generated.ema.shadow_params), len(list(generated.G1.parameters())))

    def test_reject_non_boolean_option_and_unsupported_training_paths(self):
        for invalid in ('false', 0, 1, None):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'fix_ego'):
                self.flow(fix_ego=invalid)
            with patch.object(InitDiffusion, '_make_args', return_value=self.args()), \
                    self.subTest(wrapper=invalid), self.assertRaisesRegex(ValueError, 'fix_ego'):
                InitDiffusion(32, 2, 4, self.processor(), False, fix_ego=invalid)
        for options, processor, gail in ((dict(heading_noise='gaussian'), self.processor(), True),
                                         (dict(use_rl=True), self.processor(), False),
                                         ({}, self.processor(use_refiner=True), False)):
            with self.subTest(options=options, gail=gail), self.assertRaisesRegex(ValueError, 'fix_ego=false'):
                Flow(self.args(fix_ego=False, **options), processor, gail)

    def test_training_and_evaluation_config_enable_all_agent_generation(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        option = 'model.model_config.decoder.init_diffusion.fix_ego'
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                with self.subTest(experiment=experiment):
                    default = compose(config_name='run.yaml', overrides=[f'experiment={experiment}'])
                    self.assertFalse(OmegaConf.select(default, option))
                    old = compose(config_name='run.yaml', overrides=[f'experiment={experiment}', f'{option}=true'])
                    self.assertTrue(OmegaConf.select(old, option))


if __name__ == '__main__':
    unittest.main()
