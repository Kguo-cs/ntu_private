"""Isotropic heading endpoints agree across supervision, sampling and loaded weights."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _noise_endpoint


class InitDiffusionHeadingNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def args(sigma_h=None):
        return SimpleNamespace(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                               num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                               sampling_steps=2, use_rl=False, sigma_h=sigma_h)

    @staticmethod
    def processor():
        return SimpleNamespace(use_refiner=False, learn_init=True, init_map_range=50.)

    def flow(self, sigma_h=5.):
        model = Flow(self.args(sigma_h), self.processor(), False)
        with torch.no_grad():
            model.model.normal_mean.copy_(torch.tensor([[3., -2., .8, -.3, 4., 2., 7., 1.]]))
            model.model.normal_scale.copy_(torch.tensor([[2., 3., .2, 9., 1., .5, 4., 2.]]))
        return model

    @staticmethod
    def inputs():
        state = torch.tensor([[2., 3., 1., 0., 4.5, 2., 5., 0.],
                              [0., 0., 1., 0., 4.5, 2., 5., 0.],
                              [-4., 7., 0., 1., 4., 1.8, 3., 0.]])
        agent = dict(batch=torch.zeros(3, dtype=torch.long), type=torch.tensor([0, 1, 2]),
                     ego_mask=torch.tensor([False, True, False]), num_graphs=1,
                     expert_input=state.clone(),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]))
        feature = dict(batch=torch.zeros(3, dtype=torch.long),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
                       orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, 32))
        return state, agent, feature

    def test_train_and_sampler_use_identical_endpoint_and_keep_ego_conditioned(self):
        flow = self.flow()
        state, agent, feature = self.inputs()
        eps = torch.linspace(-1.7, 2.1, 24).reshape(3, 8)
        expected = flow.model.denormalize(eps)
        expected[:, 2:4] = eps[:, 2:4] * 5.
        with patch('src.smart.diffusion.scale_flow.torch.randn_like', return_value=eps.clone()), \
                patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            noise, time, latent = flow._prepare_supervised_batch(state, agent)
        non_ego = ~agent['ego_mask']
        torch.testing.assert_close(noise[non_ego], expected[non_ego])
        torch.testing.assert_close(noise[~non_ego], state[~non_ego])
        self.assertEqual(time[~non_ego].item(), 0.)
        torch.testing.assert_close(latent, (1-time)*state + time*noise)
        flow.eval()
        with patch('src.smart.diffusion.scale_flow.torch.randn', return_value=eps.clone()), torch.no_grad():
            generated = flow.sample(agent, feature, steps=2)
        torch.testing.assert_close(agent['gen_noise'], expected)
        torch.testing.assert_close(agent['gen_noise'][non_ego], noise[non_ego])
        torch.testing.assert_close(generated[~non_ego], state[~non_ego])
        self.assertTrue(torch.isfinite(generated).all())

    def test_new_endpoint_has_zero_mean_equal_std_and_rotation_symmetry(self):
        flow = self.flow()
        torch.manual_seed(817)
        eps = torch.randn(20000, 8)
        noise = _noise_endpoint(flow.model, eps, flow.sigma_h)
        heading = noise[:, 2:4]
        torch.testing.assert_close(heading.mean(0), torch.zeros(2), atol=.12, rtol=0)
        torch.testing.assert_close(heading.std(0, unbiased=False), torch.full((2,), 5.), atol=.12, rtol=0)
        indices = [0, 1, 4, 5, 6, 7]
        torch.testing.assert_close(noise[:, indices], flow.model.denormalize(eps)[:, indices])
        rotated_eps = eps.clone()
        rotated_eps[:, 2], rotated_eps[:, 3] = -eps[:, 3], eps[:, 2]
        rotated_noise = _noise_endpoint(flow.model, rotated_eps, flow.sigma_h)
        torch.testing.assert_close(rotated_noise[:, 2], -noise[:, 3])
        torch.testing.assert_close(rotated_noise[:, 3], noise[:, 2])

    def test_loaded_normalizer_does_not_override_explicit_heading_sigma(self):
        old = self.flow(sigma_h=None)
        state = old.state_dict()
        new = self.flow(sigma_h=2.5)
        new.load_state_dict(state, strict=True)
        eps = torch.ones(3, 8, dtype=torch.float64)
        new.double()
        noise = _noise_endpoint(new.model, eps, new.sigma_h)
        self.assertEqual(noise.dtype, torch.float64)
        torch.testing.assert_close(noise[:, 2:4], torch.full((3, 2), 2.5, dtype=torch.float64))
        torch.testing.assert_close(new.model.normal_mean, state['model.normal_mean'].double())
        torch.testing.assert_close(new.model.normal_scale, state['model.normal_scale'].double())

    def test_null_retains_exact_empirical_endpoint(self):
        flow = self.flow(sigma_h=None)
        eps = torch.randn(10, 8)
        torch.testing.assert_close(_noise_endpoint(flow.model, eps, None),
                                   flow.model.denormalize(eps), atol=0, rtol=0)

    def test_real_supervised_objective_backpropagates_with_isotropic_heading_noise(self):
        flow = self.flow().train()
        state, agent, feature = self.inputs()
        with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .4)):
            losses = flow._supervised_loss(state, agent, feature)
        loss = losses[0].mean() + losses[1]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in flow.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_wrapper_passes_sigma_h_and_validates_invalid_values(self):
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            model = InitDiffusion(32, 2, 4, self.processor(), False, sigma_h=2.5)
        self.assertEqual(model.sigma_h, 2.5)
        self.assertEqual(model.G1.sigma_h, 2.5)
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(sigma_h=value), self.assertRaisesRegex(ValueError, 'sigma_h must be finite and positive'):
                Flow(self.args(value), self.processor(), False)

    def test_train_eval_config_inherits_default_and_override(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                for value in ('5.0', '2.5', 'null'):
                    overrides = [f'experiment={experiment}']
                    if value != '5.0':
                        overrides.append(f'model.model_config.decoder.init_diffusion.sigma_h={value}')
                    config = compose(config_name='run.yaml', overrides=overrides)
                    actual = config.model.model_config.decoder.init_diffusion.sigma_h
                    self.assertEqual(actual, None if value == 'null' else float(value))


if __name__ == '__main__':
    unittest.main()
