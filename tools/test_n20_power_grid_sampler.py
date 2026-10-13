"""CPU contracts for the independent N20 power-grid prototype."""
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.n20_power_grid_sampler import (
    power_time_grid, sample_with_power_grid, validate_reviewed_flow, REVIEWED_RECIPE,
)


class PowerGridTests(unittest.TestCase):
    def test_uniform_control_is_bit_identical_legacy_grid(self):
        actual = power_time_grid(1., device='cpu', dtype=torch.float32)
        self.assertTrue(torch.equal(actual, torch.linspace(1., 0., 21)))

    def test_all_three_grids_are_exactly_twenty_calls_and_end_at_zero(self):
        for p in (.5, 1., 2.):
            with self.subTest(power=p):
                grid = power_time_grid(p, device='cpu', dtype=torch.float32)
                self.assertEqual(len(grid[:-1]), 20)
                self.assertTrue(bool((grid[:-1] > 0).all()))
                self.assertTrue(bool((grid[:-1] > grid[1:]).all()))
                self.assertEqual(float(grid[0]), 1.)
                self.assertEqual(float(grid[-1]), 0.)
                self.assertAlmostEqual(float(grid[-2]), .05 ** p, places=6)

    def test_actual_dt_telescopes_to_minus_one(self):
        for p in (.5, 1., 2.):
            grid = power_time_grid(p, device='cpu', dtype=torch.float32)
            delta = grid[1:] - grid[:-1]
            self.assertTrue(bool((delta < 0).all()))
            self.assertAlmostEqual(float(delta.sum()), -1., places=6)

    def test_power_two_allocates_more_evaluations_at_small_times(self):
        grids = [power_time_grid(p, device='cpu', dtype=torch.float32) for p in (.5, 1., 2.)]
        low = [int((g[:-1] < .25).sum()) for g in grids]
        self.assertLess(low[0], low[1])
        self.assertLess(low[1], low[2])

    def test_x0_euler_exact_dt_over_t_reaches_clean_endpoint(self):
        clean = torch.tensor([[3., -2., 4., 1.8, .25]])
        source = torch.tensor([[20., -12., 8., -1., 10.]])
        for p in (.5, 1., 2.):
            grid = power_time_grid(p, device='cpu', dtype=torch.float32)
            state = source.clone()
            for t, next_t in zip(grid[:-1], grid[1:]):
                state = state + (next_t - t) * (state - clean) / t
            self.assertTrue(torch.allclose(state, clean, atol=2e-6, rtol=0))

    def test_legacy_floor_is_not_valid_exact_x0_transport_below_point_zero_five(self):
        grid = power_time_grid(2., device='cpu', dtype=torch.float32)
        state = torch.tensor([[9.]])
        clean = torch.tensor([[2.]])
        for t, next_t in zip(grid[:-1], grid[1:]):
            state = state + (next_t - t) * (state - clean) / t.clamp_min(.05)
        self.assertGreater(abs(float(state - clean)), .001)

    def test_circle_shortest_arc_dt_preserves_unit_norm_and_endpoint(self):
        for p in (.5, 1., 2.):
            grid = power_time_grid(p, device='cpu', dtype=torch.float32)
            theta = torch.tensor([math.pi - .1])
            clean = torch.tensor([-math.pi + .1])
            wrap = lambda value: (value + math.pi).remainder(2 * math.pi) - math.pi
            for t, next_t in zip(grid[:-1], grid[1:]):
                velocity = wrap(theta - clean) / t
                theta = wrap(theta + (next_t - t) * velocity)
                pair = torch.stack([theta.cos(), theta.sin()], -1)
                self.assertTrue(torch.allclose(pair.square().sum(-1), torch.ones(1), atol=1e-6))
            self.assertLess(float(wrap(theta - clean).abs().max()), 1e-6)

    def test_unreviewed_budget_dtype_and_power_rejected(self):
        for steps in (10, 40, 20., True):
            with self.assertRaises(ValueError):
                power_time_grid(1., device='cpu', dtype=torch.float32, steps=steps)
        for p in (0., -1., 3., True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                power_time_grid(p, device='cpu', dtype=torch.float32)
        for dtype in (torch.float64, torch.float16):
            with self.assertRaises(ValueError):
                power_time_grid(1., device='cpu', dtype=dtype)


class FailureCleanupTests(unittest.TestCase):
    """A tiny fake denoiser isolates cleanup contracts; native parity is separate."""

    @staticmethod
    def make_flow():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('normal_mean', torch.ones(1, 7))
                self.register_buffer('normal_scale', torch.ones(1, 7))
                self.m_delta_dim = 7
                self._normalizer_from_file = True
                self.eval()

            def get_input(self, data):
                return data['expert_input'], None

            def forward(self, latent, time, data):
                return latent

        class Flow(SimpleNamespace):
            def _require_type_normalizer(self):
                pass

            def _draw_type_source(self, standard):
                return standard

            def _conditioned_type_mask(self, data):
                return torch.zeros(len(data['batch']), dtype=torch.bool)

            def _sample_step(self, latent, time, next_time, data, feature, **kwargs):
                self.seen_guard.append(self.t_eps)
                self.model(latent, time.expand(len(latent), 1), data)
                raise RuntimeError('injected failure after denoiser forward')

        return Flow(**REVIEWED_RECIPE, model=Model(), t_eps=.05,
                    training=False, use_sde=False, use_refiner=False,
                    use_ego_embedding=False, sigma_h=None, seen_guard=[])

    @staticmethod
    def make_data():
        return {'batch': torch.tensor([0, 0]), 'type': torch.tensor([0, 1]),
                'expert_input': torch.ones(2, 7)}

    @staticmethod
    def fake_native_module():
        return SimpleNamespace(_noise_endpoint=lambda model, noise, *args, **kwargs: noise)

    def test_reviewed_recipe_only_accepts_native_audited_modes(self):
        validate_reviewed_flow(self.make_flow())
        rejected = {'heading_noise': 'gaussian', 'heading_objective': 'angular_velocity',
                    'velocity_representation': 'vector', 'size_representation': 'log',
                    'generate_type': False, 'type_process': 'joint', 'type_noise_stats': 'data',
                    'joint_type': True, 'feature_type': True, 'pos_source': 'uniform',
                    'shape_source': 'uniform', 'velocity_source': 'uniform', 't_eps': .01}
        for name, value in rejected.items():
            with self.subTest(field=name):
                flow = self.make_flow()
                setattr(flow, name, value)
                with self.assertRaises(ValueError):
                    validate_reviewed_flow(flow)
        flow = self.make_flow()
        flow.model.m_delta_dim = 8
        with self.assertRaises(ValueError):
            validate_reviewed_flow(flow)

    def test_exception_removes_only_temporary_hook_and_leaves_original_flow_guard(self):
        for policy, expected in (('legacy_floor', .05),
                                 ('actual_time', torch.finfo(torch.float32).eps)):
            with self.subTest(policy=policy):
                flow = self.make_flow()
                existing = flow.model.register_forward_pre_hook(lambda module, args: None)
                hooks_before = dict(flow.model._forward_pre_hooks)
                tensors_before = {key: value.clone() for key, value in flow.model.state_dict().items()}
                with mock.patch.dict(sys.modules, {'src.smart.diffusion.scale_flow': self.fake_native_module()}):
                    with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                        sample_with_power_grid(flow, self.make_data(), {}, power=2., denominator_policy=policy)
                self.assertEqual(flow.t_eps, .05)
                self.assertEqual(flow.seen_guard, [expected])
                self.assertEqual(dict(flow.model._forward_pre_hooks), hooks_before)
                self.assertTrue(all(torch.equal(value, flow.model.state_dict()[key])
                                    for key, value in tensors_before.items()))
                existing.remove()

    def test_invalid_policy_or_actual_time_on_other_powers_rejected_before_hook(self):
        for power, policy in ((1., 'actual_time'), (.5, 'actual_time'), (2., 'unknown')):
            with self.subTest(power=power, policy=policy):
                flow = self.make_flow()
                with mock.patch.dict(sys.modules, {'src.smart.diffusion.scale_flow': self.fake_native_module()}):
                    with self.assertRaises(ValueError):
                        sample_with_power_grid(flow, self.make_data(), {}, power=power,
                                               denominator_policy=policy)
                self.assertEqual(len(flow.model._forward_pre_hooks), 0)
                self.assertEqual(flow.t_eps, .05)

    def test_denominator_policy_must_be_explicit(self):
        flow = self.make_flow()
        with self.assertRaises(TypeError):
            sample_with_power_grid(flow, self.make_data(), {}, power=2.)
        self.assertEqual(len(flow.model._forward_pre_hooks), 0)
        self.assertEqual(flow.t_eps, .05)

    def test_bad_normalizer_rejected_before_model_hook(self):
        for mean, scale in ((float('nan'), 1.), (1., 0.), (1., float('inf'))):
            with self.subTest(mean=mean, scale=scale):
                flow = self.make_flow()
                flow.model.normal_mean.fill_(mean)
                flow.model.normal_scale.fill_(scale)
                with mock.patch.dict(sys.modules, {'src.smart.diffusion.scale_flow': self.fake_native_module()}):
                    with self.assertRaises(ValueError):
                        sample_with_power_grid(flow, self.make_data(), {}, denominator_policy='legacy_floor')
                self.assertEqual(len(flow.model._forward_pre_hooks), 0)
                self.assertEqual(flow.t_eps, .05)


if __name__ == '__main__':
    unittest.main()
