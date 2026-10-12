"""Independent component objectives retain masks and predictable time scaling."""

import copy
import itertools
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.smart.diffusion.component_loss import ComponentLoss, heading_angle_residual
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow, _circular_interpolate
import test_init_diffusion_type_generation as type_fixture


class InitDiffusionComponentLossTest(unittest.TestCase):
    @staticmethod
    def objective(**components):
        return ComponentLoss({
            'normalize_time_weights': False,
            'components': components,
        })

    @staticmethod
    def states():
        prediction = torch.tensor([[2., -4.], [3., 1.]], dtype=torch.float64)
        target = torch.zeros_like(prediction)
        time = torch.tensor([[.25], [.5]], dtype=torch.float64)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        return prediction, target, time, valid

    def test_each_component_selects_its_kind_power_and_coefficient(self):
        prediction, target, time, valid = self.states()
        objective = self.objective(
            position={'kind': 'l1', 'time_power': 1, 'weight': 2., 'scale': 'none'},
            heading={'kind': 'mse', 'time_power': 2, 'weight': .5, 'scale': 'none'},
            length={'kind': 'l1', 'time_power': 3, 'weight': 3., 'scale': 'none'},
            width={'kind': 'mse', 'time_power': 0, 'weight': 4., 'scale': 'none'},
        )
        for name, kind, power, coefficient in (
            ('position', 'l1', 1, 2.), ('heading', 'mse', 2, .5),
            ('length', 'l1', 3, 3.), ('width', 'mse', 0, 4.),
        ):
            with self.subTest(component=name):
                raw, weighted = objective.regression(name, prediction, target, time, valid)
                error = prediction.abs() if kind == 'l1' else prediction.square()
                row = error.mean(-1)
                torch.testing.assert_close(raw, row.mean())
                torch.testing.assert_close(weighted, coefficient*(row*time[:, 0].pow(-power)).mean())
        self.assertEqual(objective.config['components']['position']['kind'], 'l1')
        self.assertEqual(objective.config['components']['heading']['time_power'], 2)

    def test_uniform_time_normalization_has_analytic_means(self):
        times = torch.tensor([0., .01, .05, .2, .7, 1.], dtype=torch.float64)
        expectations = {0: 1., 1: 1.+math.log(20.), 2: 39., 3: 599.5}
        for power, normalizer in expectations.items():
            with self.subTest(power=power):
                objective = ComponentLoss({
                    'normalize_time_weights': True,
                    'components': {'position': {'time_power': power}},
                })
                expected = times.clamp_min(.05).pow(-power)/normalizer
                expected[[0, -1]] = 0.
                torch.testing.assert_close(objective.time_weight('position', times), expected)
                torch.testing.assert_close(objective.time_weight('position', times[:, None]), expected)

    def test_normalization_uses_supplied_time_epsilon(self):
        times = torch.tensor([.02, .25, .75], dtype=torch.float64)
        epsilon = .1
        for power in (1, 2, 3):
            objective = ComponentLoss({'normalize_time_weights': True,
                                       'components': {'velocity': {'time_power': power}}})
            normalizer = (1.+math.log(1./epsilon) if power == 1
                          else epsilon**(1-power)+(epsilon**(1-power)-1.)/(power-1))
            expected = times.clamp_min(epsilon).pow(-power)/normalizer
            torch.testing.assert_close(objective.time_weight('velocity', times, t_eps=epsilon), expected)

    def test_mask_reduction_means_valid_coordinates_then_active_rows(self):
        objective = self.objective(position={'kind': 'l1', 'time_power': 0, 'weight': 2.})
        prediction = torch.tensor([[2., 100.], [4., 8.], [100., 100.]], dtype=torch.float64)
        target = torch.zeros_like(prediction)
        valid = torch.tensor([[True, False], [True, True], [False, False]])
        raw, weighted = objective.regression('position', prediction, target,
                                             torch.full((3,), .5), valid)
        # Each active row has equal weight, despite unequal coordinate counts.
        torch.testing.assert_close(raw, torch.tensor(4., dtype=torch.float64))
        torch.testing.assert_close(weighted, torch.tensor(8., dtype=torch.float64))

    def test_inactive_nonfinite_values_and_endpoints_have_finite_zero_gradients(self):
        for kind in ('l1', 'mse'):
            with self.subTest(kind=kind):
                objective = self.objective(position={'kind': kind, 'time_power': 3, 'weight': .7})
                prediction = torch.tensor([[float('nan'), float('inf')],
                                           [2., float('inf')],
                                           [float('nan'), float('-inf')],
                                           [float('inf'), float('nan')]],
                                          dtype=torch.float64, requires_grad=True)
                target = torch.tensor([[float('inf'), float('nan')], [0., float('nan')],
                                       [float('nan'), float('nan')], [float('nan'), float('nan')]],
                                      dtype=torch.float64)
                times = torch.tensor([0., .5, .4, 1.], dtype=torch.float64)
                valid = torch.tensor([[True, True], [True, False], [False, False], [True, True]])
                raw, weighted = objective.regression('position', prediction, target, times, valid)
                expected = 2. if kind == 'l1' else 4.
                torch.testing.assert_close(raw, torch.tensor(expected, dtype=torch.float64))
                torch.testing.assert_close(weighted, torch.tensor(.7*8.*expected, dtype=torch.float64))
                weighted.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertGreater(prediction.grad[1, 0].abs().item(), 0.)
                expected_grad = torch.zeros_like(prediction)
                expected_grad[1, 0] = .7*8.*(1. if kind == 'l1' else 4.)
                torch.testing.assert_close(prediction.grad, expected_grad)

    def test_all_fixed_batch_has_finite_differentiable_zero(self):
        objective = self.objective(length={'kind': 'mse', 'time_power': 3})
        prediction = torch.full((2, 1), float('nan'), dtype=torch.float64, requires_grad=True)
        target = torch.full_like(prediction, float('inf'))
        raw, weighted = objective.regression('length', prediction, target,
                                             torch.tensor([.2, .7]), torch.zeros_like(prediction, dtype=torch.bool))
        self.assertEqual(raw.item(), 0.)
        self.assertEqual(weighted.item(), 0.)
        weighted.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)

    def test_data_and_numeric_scales_apply_before_l1_or_mse(self):
        prediction, target, time, valid = self.states()
        for kind in ('l1', 'mse'):
            with self.subTest(kind=kind):
                objective = self.objective(position={'kind': kind, 'time_power': 0, 'scale': 'data'})
                raw, weighted = objective.regression('position', prediction, target, time, valid,
                                                     scale=torch.tensor([[2., 4.]], dtype=torch.float64))
                scaled_error = prediction/torch.tensor([2., 4.], dtype=torch.float64)
                expected = (scaled_error.abs() if kind == 'l1' else scaled_error.square()).mean()
                torch.testing.assert_close(raw, expected)
                torch.testing.assert_close(weighted, expected)
                numeric = self.objective(position={'kind': kind, 'time_power': 0, 'scale': 2.})
                numeric_raw, _ = numeric.regression('position', prediction, target, time, valid)
                numeric_error = prediction/2.
                expected = (numeric_error.abs() if kind == 'l1' else numeric_error.square()).mean()
                torch.testing.assert_close(numeric_raw, expected)

    def test_standard_deviation_floor_prevents_division_by_zero(self):
        objective = ComponentLoss({'normalize_time_weights': False, 'std_floor': .2,
                                  'components': {'width': {'kind': 'l1', 'time_power': 0, 'scale': 'data'}}})
        prediction = torch.tensor([[1.]], dtype=torch.float64, requires_grad=True)
        raw, weighted = objective.regression('width', prediction, torch.zeros_like(prediction),
                                             torch.tensor([.5]), torch.ones_like(prediction, dtype=torch.bool),
                                             scale=torch.zeros((1, 1), dtype=torch.float64))
        torch.testing.assert_close(raw, torch.tensor(5., dtype=torch.float64))
        weighted.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_zero_coefficient_disables_nonfinite_component_and_gradient(self):
        prediction = torch.full((2, 2), float('inf'), dtype=torch.float64, requires_grad=True)
        target = torch.full_like(prediction, float('nan'))
        time = torch.tensor([.2, .7], dtype=torch.float64)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        objective = self.objective(velocity={'kind': 'l1', 'time_power': 3, 'weight': 0.})
        raw, weighted = objective.regression('velocity', prediction, target, time, valid)
        self.assertEqual(raw.item(), 0.)
        self.assertEqual(weighted.item(), 0.)
        weighted.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)

    def test_heading_angle_residual_wraps_shortest_arc_and_has_antipodal_gradient(self):
        truth = torch.deg2rad(torch.tensor([179., -179., 0., 0.], dtype=torch.float64))
        theta = torch.deg2rad(torch.tensor([-179., 179., 179.999, -180.],
                                          dtype=torch.float64)).requires_grad_()
        target = torch.stack((truth.cos(), truth.sin()), -1)
        prediction = torch.stack((theta.cos(), theta.sin()), -1)
        residual = heading_angle_residual(prediction, target)
        expected = torch.deg2rad(torch.tensor([2., -2., 179.999, -180.], dtype=torch.float64))
        torch.testing.assert_close(residual, expected)
        # Squared shortest-arc error retains a correcting signal near 180 degrees.
        residual.square().sum().backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        torch.testing.assert_close(theta.grad, 2.*expected)
        self.assertGreater(theta.grad[2].abs().item(), 6.)
        for scale in (.2, 3.):
            torch.testing.assert_close(heading_angle_residual(prediction.detach()*scale,
                                                               target/scale), expected)

    def test_heading_zero_pair_fallback_is_finite_and_differentiable(self):
        prediction = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        residual = heading_angle_residual(prediction, target)
        torch.testing.assert_close(residual, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)
        residual.square().sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0)

    def test_feature_type_regression_remains_raw_without_softmax(self):
        objective = self.objective(type={'kind': 'mse', 'time_power': 1, 'weight': .2})
        prediction = torch.tensor([[-2., .5, 4.], [1.5, -3., .2]],
                                  dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
        time = torch.tensor([.25, .5], dtype=torch.float64)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        raw, weighted = objective.regression('type', prediction, target, time, valid)
        row = (prediction-target).square().mean(-1)
        torch.testing.assert_close(raw, row.mean())
        torch.testing.assert_close(weighted, .2*(row/time).mean())
        weighted.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.)

    def test_collision_penalty_has_independent_kind_power_and_weight(self):
        objective = self.objective(collision={'kind': 'mse', 'time_power': 2, 'weight': .1})
        overlap = torch.tensor([[.2], [.4], [.8]], dtype=torch.float64, requires_grad=True)
        time = torch.tensor([.2, .5, .8], dtype=torch.float64)
        valid = torch.tensor([[True], [False], [True]])
        raw, weighted = objective.regression('collision', overlap, torch.zeros_like(overlap), time, valid)
        active = valid[:, 0]
        row = overlap[:, 0].square()
        torch.testing.assert_close(raw, row[active].mean())
        torch.testing.assert_close(weighted, .1*(row[active]/time[active].square()).mean())
        weighted.backward()
        self.assertEqual(overlap.grad[1].item(), 0.)
        self.assertGreater(overlap.grad[active].abs().sum().item(), 0.)

    def test_categorical_ce_and_probability_regression_keep_finite_gradients(self):
        labels = torch.tensor([0, 2, 1])
        times = torch.tensor([.2, .5, .8], dtype=torch.float64)
        valid = torch.tensor([True, False, True])
        for kind in ('ce', 'l1', 'mse'):
            with self.subTest(kind=kind):
                logits = torch.tensor([[2., -.2, .7], [-.4, 1., 3.], [1., 2., -.5]],
                                      dtype=torch.float64, requires_grad=True)
                objective = self.objective(type={'kind': kind, 'time_power': 2, 'weight': .3})
                raw, weighted = objective.categorical(logits, labels, times, valid)
                if kind == 'ce':
                    row = F.cross_entropy(logits, labels, reduction='none')
                else:
                    difference = logits.softmax(-1)-F.one_hot(labels, 3)
                    row = (difference.abs() if kind == 'l1' else difference.square()).mean(-1)
                torch.testing.assert_close(raw, row[valid].mean())
                torch.testing.assert_close(weighted, .3*(row[valid]*times[valid].pow(-2)).mean())
                weighted.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                torch.testing.assert_close(logits.grad[1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
                self.assertGreater(logits.grad[[0, 2]].abs().sum().item(), 0.)

    def test_categorical_masks_nonfinite_logits_and_invalid_labels_before_arithmetic(self):
        logits = torch.tensor([[float('nan')]*3, [1., -1., 0.],
                               [float('inf')]*3, [float('nan')]*3],
                              dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([-999, 0, -999, -999])
        times = torch.tensor([0., .5, .3, 1.], dtype=torch.float64)
        valid = torch.tensor([True, True, False, True])
        objective = self.objective(type={'kind': 'ce', 'time_power': 1, 'weight': .4})
        raw, weighted = objective.categorical(logits, labels, times, valid)
        reference = F.cross_entropy(logits[1:2], labels[1:2])
        torch.testing.assert_close(raw, reference)
        torch.testing.assert_close(weighted, .8*reference)
        weighted.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[[0, 2, 3]], torch.zeros((3, 3), dtype=torch.float64), atol=0, rtol=0)

    def test_empty_categorical_batch_is_differentiable_zero(self):
        objective = self.objective(type={'kind': 'ce'})
        logits = torch.empty((0, 3), dtype=torch.float64, requires_grad=True)
        raw, weighted = objective.categorical(logits, torch.empty(0, dtype=torch.long),
                                              torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.bool))
        self.assertEqual(raw.item(), 0.)
        self.assertEqual(weighted.item(), 0.)
        weighted.backward()
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_invalid_configuration_is_rejected_early(self):
        invalid_configs = [
            {'unexpected': True},
            {'normalize_time_weights': 'yes'},
            {'components': {'type': {'kind': 'ce', 'scale': 'data'}}},
            {'components': {'positon': {'kind': 'l1'}}},
            {'components': {'position': {'unexpected': 1.}}},
            {'components': {'position': {'kind': 'huber'}}},
            {'components': {'position': {'kind': 'ce'}}},
            {'components': {'position': {'time_power': -1}}},
            {'components': {'position': {'time_power': 4}}},
            {'components': {'position': {'time_power': 1.5}}},
            {'components': {'position': {'weight': -1.}}},
            {'components': {'position': {'weight': float('inf')}}},
            {'components': {'position': {'weight': float('nan')}}},
            {'components': {'position': {'scale': 'invalid'}}},
            {'components': {'position': {'scale': 0.}}},
            {'components': {'position': {'scale': -1.}}},
            {'components': {'position': {'scale': float('nan')}}},
            {'std_floor': 0.},
            {'std_floor': -1.},
            {'std_floor': float('inf')},
            {'heading_space': 'degrees'},
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                ComponentLoss(config)

    def test_default_type_kind_and_legacy_weights_can_be_resolved(self):
        objective = ComponentLoss({}, type_kind='mse', heading_space='angle',
                                  type_weight=.25, heading_weight=2.)
        self.assertEqual(objective.config['components']['type']['kind'], 'mse')
        self.assertEqual(objective.config['components']['type']['weight'], .25)
        self.assertEqual(objective.config['components']['heading']['weight'], 2.)
        self.assertEqual(objective.config['heading_space'], 'angle')


class InitDiffusionComponentFlowTest(unittest.TestCase):
    fixtures = type_fixture.InitDiffusionTypeGenerationTest
    names = ('position', 'heading', 'length', 'width', 'velocity', 'type', 'collision', 'speed_magnitude')

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(863)

    @classmethod
    def config(cls, heading_space='vector', **components):
        entries = {name: {'weight': 0.} for name in cls.names}
        for name, options in components.items():
            entries[name].update(options)
        return dict(normalize_time_weights=False, heading_space=heading_space, components=entries)

    @classmethod
    def flow(cls, config, **options):
        return Flow(cls.fixtures.args(component_loss=config, **options), cls.fixtures.processor(), False)

    @staticmethod
    def mocked_loss(flow, clean, agent, feature, prediction, time, logits=None, noise=None):
        def predict(*args, **kwargs):
            if logits is not None:
                agent['_init_diffusion_type_logits'] = logits
            return torch.zeros_like(prediction), prediction
        with patch.object(flow, '_prepare_supervised_batch',
                          return_value=(clean if noise is None else noise, time, clean)), \
                patch.object(flow, '_model_velocity', side_effect=predict):
            return flow._supervised_loss(clean, agent, feature)

    def test_mixed_component_objectives_control_total_and_metrics(self):
        config = self.config(
            position={'kind': 'l1', 'time_power': 1, 'weight': 2.},
            length={'kind': 'mse', 'time_power': 2, 'weight': .5},
            width={'kind': 'mse', 'time_power': 3, 'weight': .3},
            velocity={'kind': 'mse', 'time_power': 2, 'weight': .7},
        )
        flow = self.flow(config, fix_ego=False)
        clean, agent, feature = self.fixtures.inputs()
        delta = torch.tensor([2., -4., 0., 0., .5, -.25, 3., -4.])
        prediction = (clean+delta).detach().requires_grad_()
        time = torch.tensor([[.2], [.4], [.8]])
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, time)
        position = delta[:2].abs().mean()*2.*time[:, 0].pow(-1).mean()
        length = delta[4].square()*.5*time[:, 0].pow(-2).mean()
        width = delta[5].square()*.3*time[:, 0].pow(-3).mean()
        velocity = delta[6:].square().mean()*.7*time[:, 0].pow(-2).mean()
        torch.testing.assert_close(losses[0], position+length+width+velocity)
        metrics = agent['_init_diffusion_component_metrics']
        weighted_sum = sum(value for name, value in metrics.items() if name.endswith('_weighted_loss'))
        torch.testing.assert_close(losses[0]+losses[1], weighted_sum)
        torch.testing.assert_close(metrics['position_loss'], delta[:2].abs().mean())
        torch.testing.assert_close(metrics['length_loss'], delta[4].square())
        losses[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:, 2:4].abs().sum().item(), 0.)
        all_l1 = copy.deepcopy(config)
        for name in ('length', 'width', 'velocity'):
            all_l1['components'][name]['kind'] = 'l1'
        comparison = self.flow(all_l1, fix_ego=False)
        other = self.mocked_loss(comparison, clean, copy.deepcopy(agent), feature, prediction.detach(), time)
        self.assertNotAlmostEqual(losses[0].item(), other[0].item(), places=5)

    def test_collision_is_added_once_and_weighted_metrics_sum_matches_optimized_loss(self):
        flow = self.flow(self.config(position={'kind': 'l1', 'time_power': 0, 'weight': .2},
                                     collision={'kind': 'mse', 'time_power': 1, 'weight': .3}),
                         fix_ego=False)
        clean, agent, feature = self.fixtures.inputs()
        clean = clean.clone()
        clean[:, :2] = torch.tensor([[0., 0.], [12., 0.], [24., 0.]])
        clean[:, 4:6] = torch.tensor([4.8, 2.])
        prediction = clean.clone()
        prediction[1, :2] = torch.tensor([.3, .1])
        prediction.requires_grad_()
        time = torch.full((3, 1), .4)
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, time)
        metrics = agent['_init_diffusion_component_metrics']
        self.assertGreater(losses[1].item(), 0.)
        torch.testing.assert_close(losses[1], metrics['collision_weighted_loss'])
        weighted_sum = sum(value for name, value in metrics.items() if name.endswith('_weighted_loss'))
        torch.testing.assert_close(losses[0]+losses[1], weighted_sum)
        (losses[0]+losses[1]).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[:, :2].abs().sum().item(), 0.)

    def test_collision_already_blocks_size_gradient_but_keeps_repulsion(self):
        config = self.config(collision={'kind': 'l1', 'time_power': 1, 'weight': 20.})
        for representation in ('linear', 'log'):
            with self.subTest(representation=representation):
                clean, agent, feature = self.fixtures.inputs()
                clean = clean.clone()
                clean[:, :2] = torch.tensor([[0., 0.], [12., 0.], [24., 0.]])
                clean[:, 2:4] = torch.tensor([1., 0.])
                clean[:, 4:6] = torch.tensor([4.8, 2.])
                if representation == 'log':
                    clean[:, 4:6] = clean[:, 4:6].log()
                sample = clean.clone()
                sample[1, :2] = torch.tensor([.8, .7])
                angle = torch.tensor(.35)
                sample[1, 2:4] = torch.stack((angle.cos(), angle.sin()))
                flow = self.flow(config, fix_ego=False, size_representation=representation)
                prediction = sample.clone().requires_grad_()
                losses = self.mocked_loss(flow, clean, agent, feature, prediction, torch.full((3, 1), .4))
                losses[1].backward()
                self.assertGreater(losses[1].item(), 0.)
                self.assertTrue(torch.isfinite(prediction.grad).all())
                torch.testing.assert_close(prediction.grad[:, 4:6], torch.zeros((3, 2)), atol=0, rtol=0)
                self.assertGreater(prediction.grad[:, :2].abs().sum().item(), 0.)
                self.assertGreater(prediction.grad[:, 2:4].abs().sum().item(), 0.)

    def test_agent_max_counts_each_colliding_actor_instead_of_all_pairs(self):
        config = self.config(collision={'kind': 'l1', 'time_power': 0, 'weight': 1.})
        clean, agent, feature = self.fixtures.inputs()
        clean = clean.clone()
        clean[:, :2] = torch.tensor([[0., 0.], [12., 0.], [24., 0.]])
        clean[:, 2:4] = torch.tensor([1., 0.])
        clean[:, 4:6] = torch.tensor([4.8, 2.])
        sample = clean.clone()
        sample[1, :2] = torch.tensor([.8, .7])
        results = []
        for reduction in ('pair_mean', 'agent_max'):
            flow = self.flow(config, fix_ego=False, collision_reduction=reduction)
            prediction = sample.clone().requires_grad_()
            losses = self.mocked_loss(flow, clean, copy.deepcopy(agent), feature,
                                     prediction, torch.full((3, 1), .4))
            losses[1].backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            results.append((losses[1].detach(), prediction.grad))
        # Only one of three pairs collides, so 2/3 actors vs 1/3 pairs.
        torch.testing.assert_close(results[1][0], 2.*results[0][0])
        torch.testing.assert_close(results[1][1][:, :4], 2.*results[0][1][:, :4])
        torch.testing.assert_close(results[1][1][:, 4:6], torch.zeros((3, 2)), atol=0, rtol=0)

    def test_agent_max_retains_fixed_ego_zero_time_collision_condition(self):
        config = self.config(collision={'kind': 'l1', 'time_power': 1, 'weight': 1.})
        clean, agent, feature = self.fixtures.inputs()
        clean = clean.clone()
        clean[:, :2] = torch.tensor([[0., 0.], [12., 0.], [24., 0.]])
        clean[:, 2:4] = torch.tensor([1., 0.])
        clean[:, 4:6] = torch.tensor([4.8, 2.])
        sample = clean.clone()
        sample[0, :2] = torch.tensor([12.8, .7])
        observed = []
        for reduction in ('pair_mean', 'agent_max'):
            flow = self.flow(config, fix_ego=True, fix_ego_velocity=False,
                             collision_reduction=reduction)
            prediction = sample.clone().requires_grad_()
            losses = self.mocked_loss(flow, clean, copy.deepcopy(agent), feature,
                                     prediction, torch.tensor([[.4], [0.], [.4]]))
            losses[1].backward()
            self.assertGreater(losses[1].item(), 0.)
            torch.testing.assert_close(prediction.grad[1, :6], torch.zeros(6), atol=0, rtol=0)
            self.assertGreater(prediction.grad[0, :2].abs().sum().item(), 0.)
            observed.append(losses[1].detach())
        torch.testing.assert_close(observed[1], 2.*observed[0])

    def test_agent_max_single_actor_is_finite_zero(self):
        flow = self.flow(self.config(collision={'weight': 1.}), fix_ego=False,
                         collision_reduction='agent_max')
        clean, agent, feature = self.fixtures.inputs()
        agent['batch'] = torch.arange(3)
        prediction = clean.clone().requires_grad_()
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, torch.full((3, 1), .4))
        self.assertEqual(losses[1].item(), 0.)
        (losses[0]+losses[1]).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_disabled_collision_skips_geometry_and_log_size_exponentiation(self):
        flow = self.flow(self.config(length={'kind': 'mse', 'time_power': 0, 'weight': 1.}),
                         fix_ego=False, size_representation='log').train()
        _, agent, feature = self.fixtures.inputs()
        clean, _ = flow.model.get_input(agent)
        prediction = clean.clone()
        prediction[:, 4] = 1000.
        prediction.requires_grad_()
        with patch('src.smart.diffusion.scale_flow.compute_penetration',
                   side_effect=AssertionError('Disabled collision must not evaluate geometry')), \
                patch.object(flow.model, 'state_to_physical',
                             side_effect=AssertionError('Disabled collision must not exponentiate sizes')):
            losses = self.mocked_loss(flow, clean, agent, feature, prediction, torch.full((3, 1), .5))
        self.assertEqual(losses[1].item(), 0.)
        self.assertTrue(torch.isfinite(losses[0]))
        losses[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_separate_ce_type_is_counted_once_and_masks_fixed_ego(self):
        config = self.config(type={'kind': 'ce', 'time_power': 2, 'weight': .7})
        flow = self.flow(config, generate_type=True, type_process='separate',
                         type_noise_stats='standard', fix_ego=False, fix_ego_type=True)
        clean, agent, feature = self.fixtures.inputs()
        logits = torch.tensor([[1., 0., -1.], [.1, -.2, .5], [.2, -.1, .4]], requires_grad=True)
        time = torch.tensor([[.2], [.4], [1.]])
        losses = self.mocked_loss(flow, clean, agent, feature, clean, time, logits=logits)
        expected = .7*F.cross_entropy(logits[:1], agent['type'][:1])/.2**2
        torch.testing.assert_close(losses[0], expected)
        torch.testing.assert_close(agent['_init_diffusion_type_metrics']['weighted_loss'], expected)
        torch.testing.assert_close(agent['_init_diffusion_component_metrics']['type_weighted_loss'], expected)
        losses[0].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.)
        torch.testing.assert_close(logits.grad[1:], torch.zeros_like(logits.grad[1:]), atol=0, rtol=0)

    def test_feature_type_raw_l1_or_mse_is_counted_once(self):
        for kind in ('l1', 'mse'):
            with self.subTest(kind=kind):
                flow = self.flow(self.config(type={'kind': kind, 'time_power': 1, 'weight': .4}),
                                 generate_type=True, type_process='feature', type_noise_stats='standard',
                                 fix_ego=False, fix_ego_type=True)
                clean, agent, feature = self.fixtures.inputs()
                categorical = torch.tensor([[-2., .5, 4.], [1.5, -3., .2], [7., -1., -4.]])
                prediction = torch.cat((clean, categorical), -1).detach().requires_grad_()
                time = torch.tensor([[.2], [.4], [1.]])
                with patch('torch.nn.functional.cross_entropy', side_effect=AssertionError('Feature type is raw state')):
                    losses = self.mocked_loss(flow, clean, agent, feature, prediction, time)
                error = categorical[0]-F.one_hot(agent['type'][0], 3)
                raw = (error.abs() if kind == 'l1' else error.square()).mean()
                expected = .4*raw/.2
                torch.testing.assert_close(losses[0], expected)
                torch.testing.assert_close(agent['_init_diffusion_type_metrics']['weighted_loss'], expected)
                losses[0].backward()
                torch.testing.assert_close(prediction.grad[:, :8], torch.zeros_like(prediction.grad[:, :8]), atol=0, rtol=0)
                torch.testing.assert_close(prediction.grad[1:, 8:], torch.zeros_like(prediction.grad[1:, 8:]), atol=0, rtol=0)
                self.assertGreater(prediction.grad[0, 8:].abs().sum().item(), 0.)

    def test_fixed_ego_and_invalid_log_size_annotations_block_nonfinite_gradients(self):
        config = self.config(heading_space='angle',
                             position={'kind': 'l1', 'time_power': 1, 'weight': 1.},
                             heading={'kind': 'mse', 'time_power': 2, 'weight': 1.},
                             length={'kind': 'mse', 'time_power': 2, 'weight': 1.},
                             width={'kind': 'mse', 'time_power': 2, 'weight': 1.},
                             velocity={'kind': 'mse', 'time_power': 2, 'weight': 1.})
        flow = self.flow(config, fix_ego=False, fix_ego_position=True, fix_ego_heading=True,
                         fix_ego_shape=True, fix_ego_velocity=False, size_representation='log').train()
        _, agent, feature = self.fixtures.inputs()
        agent['expert_input'][0, 4] = float('nan')
        clean, _ = flow.model.get_input(agent)
        self.assertFalse(agent['_init_diffusion_size_valid_mask'][0, 0])
        prediction = clean.clone()+.2
        prediction[0, 4] = float('inf')
        prediction[1, :6] = float('nan')
        prediction.requires_grad_()
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, torch.full((3, 1), .4))
        self.assertTrue(all(torch.isfinite(value).all() for value in losses))
        losses[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[0, 4].item(), 0.)
        self.assertGreater(prediction.grad[0, 5].abs().item(), 0.)
        torch.testing.assert_close(prediction.grad[1, :6], torch.zeros_like(prediction.grad[1, :6]), atol=0, rtol=0)
        self.assertGreater(prediction.grad[1, 6:].abs().sum().item(), 0.)

    def test_angular_heading_masks_nonfinite_endpoint_predictions_before_geometry(self):
        flow = self.flow(self.config(heading_space='angle',
                                     heading={'kind': 'mse', 'time_power': 3, 'weight': 1.}), fix_ego=False)
        clean, agent, feature = self.fixtures.inputs()
        prediction = clean.clone()
        prediction[0, 2:4] = float('nan')
        prediction[1, 2:4] = torch.tensor([.5, .5])
        prediction[2, 2:4] = float('inf')
        prediction.requires_grad_()
        time = torch.tensor([[0.], [.4], [1.]])
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, time)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses))
        losses[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[[0, 2]], torch.zeros_like(prediction.grad[[0, 2]]), atol=0, rtol=0)
        self.assertGreater(prediction.grad[1, 2:4].abs().sum().item(), 0.)

    def test_real_denoiser_backpropagates_for_vector_speed_and_type_layouts(self):
        for representation, process in itertools.product(('vector', 'speed'), ('separate', 'feature')):
            with self.subTest(velocity=representation, process=process):
                config = self.config(heading_space='angle',
                                     position={'kind': 'l1', 'time_power': 1, 'weight': 1., 'scale': 'data'},
                                     heading={'kind': 'mse', 'time_power': 2, 'weight': .5},
                                     length={'kind': 'mse', 'time_power': 2, 'weight': 1., 'scale': 'data'},
                                     width={'kind': 'mse', 'time_power': 2, 'weight': 1., 'scale': 'data'},
                                     velocity={'kind': 'mse', 'time_power': 2, 'weight': 1., 'scale': 'data'},
                                     type={'kind': 'ce' if process == 'separate' else 'mse', 'time_power': 0, 'weight': 1.},
                                     collision={'kind': 'l1', 'time_power': 1, 'weight': .02})
                flow = self.flow(config, input_dim=7 if representation == 'speed' else 8,
                                 velocity_representation=representation, size_representation='log',
                                 generate_type=True, type_process=process, type_noise_stats='standard',
                                 fix_ego=False, use_ego_embedding=True).train()
                _, agent, feature = self.fixtures.inputs()
                clean, _ = flow.model.get_input(agent)
                with patch.object(flow, '_sample_time', return_value=torch.full((3, 1), .6)):
                    losses = flow._supervised_loss(clean, agent, feature)
                (losses[0]+losses[1]).backward()
                gradients = [parameter.grad for parameter in flow.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                head = flow.model.to_out_m_delta.mlp[-1].weight.grad
                self.assertGreater(head[:2].abs().sum().item(), 0.)
                if process == 'separate':
                    self.assertGreater(flow.model.to_out_type.mlp[-1].weight.grad.abs().sum().item(), 0.)
                else:
                    self.assertGreater(head[-3:].abs().sum().item(), 0.)

    def test_angular_velocity_uses_own_l1_power_and_masks_fixed_ego(self):
        for representation in ('vector', 'speed'):
            with self.subTest(velocity=representation):
                config = self.config(heading={'kind': 'l1', 'time_power': 3, 'weight': 2.})
                flow = self.flow(config, input_dim=7 if representation == 'speed' else 8,
                                 velocity_representation=representation, heading_objective='angular_velocity',
                                 fix_ego=False, fix_ego_heading=True)
                _, agent, feature = self.fixtures.inputs()
                clean, _ = flow.model.get_input(agent)
                theta = torch.atan2(clean[:, 3], clean[:, 2])
                delta = torch.tensor([.2, -.4, .6])
                noise = clean.clone()
                noise[:, 2:4] = torch.stack(((theta+delta).cos(), (theta+delta).sin()), -1)
                time = torch.full((3, 1), .4)
                omega = (delta+.5).detach().requires_grad_()
                prediction = torch.cat((clean, omega[:, None]), -1)
                with patch.object(flow, '_prepare_supervised_batch',
                                  return_value=(noise, time, _circular_interpolate(clean, noise, time))), \
                        patch.object(flow.model, 'forward', return_value=prediction):
                    losses = flow._supervised_loss(clean, agent, feature)
                torch.testing.assert_close(losses[0], torch.tensor(2.*.5/.4**3))
                torch.testing.assert_close(losses[3], torch.tensor(.5))
                losses[0].backward()
                self.assertTrue(torch.isfinite(omega.grad).all())
                self.assertEqual(omega.grad[1].item(), 0.)
                self.assertGreater(omega.grad[[0, 2]].abs().sum().item(), 0.)

    def test_null_component_config_preserves_layout_rng_and_seeded_supervised_loss(self):
        options = dict(generate_type=True, type_process='separate', type_noise_stats='standard', fix_ego=False)
        torch.manual_seed(371)
        omitted = Flow(self.fixtures.args(**options), self.fixtures.processor(), False).train()
        torch.manual_seed(371)
        explicit = self.flow(None, **options).train()
        self.assertIsNone(explicit.component_loss)
        self.assertEqual(list(dict(omitted.named_parameters())), list(dict(explicit.named_parameters())))
        for name, value in omitted.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
        _, agent, feature = self.fixtures.inputs()
        first_agent = copy.deepcopy(agent)
        second_agent = copy.deepcopy(agent)
        first_clean, _ = omitted.model.get_input(first_agent)
        second_clean, _ = explicit.model.get_input(second_agent)
        torch.manual_seed(790)
        expected = omitted._supervised_loss(first_clean, first_agent, feature)
        first_after = torch.rand(5)
        torch.manual_seed(790)
        actual = explicit._supervised_loss(second_clean, second_agent, feature)
        second_after = torch.rand(5)
        for one, two in zip(expected, actual):
            torch.testing.assert_close(one, two, atol=0, rtol=0)
        torch.testing.assert_close(first_after, second_after, atol=0, rtol=0)
        self.assertNotIn('_init_diffusion_component_metrics', second_agent)

    def test_speed_magnitude_has_own_kind_power_scale_and_endpoint_masks(self):
        config = self.config(speed_magnitude={'kind': 'l1', 'time_power': 1, 'weight': .3, 'scale': 2.})
        flow = self.flow(config, fix_ego=False, fix_ego_velocity=True)
        clean, agent, feature = self.fixtures.inputs()
        prediction = clean.clone()
        prediction[0, 6:8] = torch.tensor([6., 0.])
        prediction[1, 6:8] = float('inf')
        prediction[2, 6:8] = float('nan')
        prediction.requires_grad_()
        time = torch.tensor([[.2], [.4], [1.]])
        losses = self.mocked_loss(flow, clean, agent, feature, prediction, time)
        # Only row 0 is active: speed increases from 5 to 6, divided by scale 2.
        torch.testing.assert_close(losses[0], torch.tensor(.3*.5/.2))
        metrics = agent['_init_diffusion_component_metrics']
        torch.testing.assert_close(metrics['speed_magnitude_loss'], torch.tensor(.5))
        torch.testing.assert_close(metrics['speed_magnitude_weighted_loss'], losses[0])
        torch.testing.assert_close(agent['_init_diffusion_speed_metrics']['weighted_loss'], losses[0])
        losses[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[1:], torch.zeros_like(prediction.grad[1:]), atol=0, rtol=0)
        self.assertGreater(prediction.grad[0, 6].abs().item(), 0.)

    def test_speed_magnitude_rejects_duplicate_scalar_speed_objective(self):
        config = self.config(speed_magnitude={'kind': 'mse', 'time_power': 2, 'weight': .3})
        with self.assertRaisesRegex(ValueError, '(scalar speed|velocity_representation=vector)'):
            self.flow(config, input_dim=7, velocity_representation='speed')
        defaults = self.config()
        defaults['components'].pop('speed_magnitude')
        inherited = self.flow(defaults, speed_loss_weight=.4, speed_loss_scale=2.)
        options = inherited.component_loss.config['components']['speed_magnitude']
        self.assertEqual(options['weight'], .4)
        self.assertEqual(options['scale'], 2.)

    def test_wrapper_forwards_and_saves_resolved_component_recipe(self):
        config = self.config(position={'kind': 'l1', 'time_power': 1, 'weight': 2.})
        with patch.object(InitDiffusion, '_make_args', return_value=self.fixtures.args()):
            wrapper = InitDiffusion(32, 2, 4, self.fixtures.processor(), False, component_loss=config)
            restored = InitDiffusion(32, 2, 4, self.fixtures.processor(), False, component_loss=config)
        self.assertEqual(wrapper.component_loss, wrapper.G1.component_loss.config)
        self.assertEqual(wrapper.get_extra_state()['component_loss'], wrapper.component_loss)
        restored.load_state_dict(wrapper.state_dict(), strict=True)
        self.assertEqual(restored.component_loss, wrapper.component_loss)
        self.assertEqual(restored.G1.component_loss.config, wrapper.G1.component_loss.config)


if __name__ == '__main__':
    unittest.main()
