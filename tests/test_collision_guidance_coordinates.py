"""Collision guidance uses generated physical geometry and actual fixed context."""

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.smart.diffusion.scale_flow import Flow, _noise_endpoint
from src.smart.metrics.sd_metric_helpers import compute_vehicle_circles


class CollisionGuidanceCoordinatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    @staticmethod
    def flow(**options):
        args = dict(input_dim=7, hidden_dim=32, num_heads=2, dropout=0.,
                    num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                    sampling_steps=20, use_rl=False, heading_noise="circular",
                    heading_objective="x0", velocity_representation="speed",
                    size_representation="linear", fix_ego=False,
                    collision_guidance=dict(scale=.5, t_max=.3, iters=1,
                                            margin=0., vehicles_only=False))
        args.update(options)
        processor = SimpleNamespace(use_refiner=False, learn_init=True,
                                    init_map_range=50.)
        return Flow(SimpleNamespace(**args), processor, False).eval()

    @staticmethod
    def state():
        # Five-circle centerlines coincide in x, so lateral penetration has an
        # independent analytic answer. Nonzero positions expose unit confusion.
        return torch.tensor([[20., 10., 1., 0., 4., 2., 3.],
                             [20., 11., 1., 0., 4., 2., 5.]])

    @staticmethod
    def agent(state, **overrides):
        result = dict(batch=torch.zeros(len(state), dtype=torch.long),
                      type=torch.zeros(len(state), dtype=torch.long),
                      ego_mask=torch.zeros(len(state), dtype=torch.bool),
                      expert_input=state.clone(), num_graphs=1)
        result.update(overrides)
        return result

    def test_train_normalization_keeps_positions_in_meters_and_gaussian_endpoint_units(self):
        none = self.flow(state_feature_normalization="none")
        normalized = self.flow(state_feature_normalization="train")
        mean = torch.tensor([[8., -13., .2, -.3, 4.5, 1.9, 10.]])
        std = torch.tensor([[20., 7., .5, .4, 1.2, .7, 8.]])
        normalized.model.normal_mean.copy_(mean)
        normalized.model.normal_scale.copy_(std)
        state = self.state()
        actual = normalized._collision_guided_x0(state, self.agent(state))
        reference = none._collision_guided_x0(state, self.agent(state))
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        displacement = .5 * (4. / math.sqrt(3.8) - 1.)
        expected = state.clone()
        expected[0, 1] -= displacement
        expected[1, 1] += displacement
        torch.testing.assert_close(actual, expected, atol=2.e-6, rtol=0)
        torch.testing.assert_close(state, self.state(), atol=0, rtol=0)
        standard = torch.randn_like(state)
        endpoint = _noise_endpoint(normalized.model, standard, None, "gaussian")
        torch.testing.assert_close(endpoint, standard * std + mean, atol=0, rtol=0)
        torch.testing.assert_close(normalized.model.normalize(endpoint), standard,
                                   atol=3.e-7, rtol=0)

    def test_normalized_denoiser_forward_outputs_physical_position_not_normalized_position(self):
        flow = self.flow(state_feature_normalization="train")
        state = self.state()
        flow.model.normal_mean.copy_(torch.tensor([[8., -13., .2, -.3, 4.5, 1.9, 10.]]))
        flow.model.normal_scale.copy_(torch.tensor([[20., 7., .5, .4, 1.2, .7, 8.]]))
        agent = self.agent(state)
        agent["ego_feat"] = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.]])
        local_prediction = torch.tensor([[1., 2., 1., 0., 4., 2., 3.],
                                         [1., 2., 1., 0., 4., 2., 5.]])
        # Real forward embeds standardized attributes, but its geometric output
        # transform still adds local meter offsets to the meter-valued input.
        with patch.object(flow.model, "_apply_graph_attention", return_value=local_prediction):
            actual = flow.model(state, torch.full((2, 1), .2), agent, {})
        torch.testing.assert_close(actual[:, :2], state[:, :2] + local_prediction[:, :2],
                                   atol=0, rtol=0)

    def test_log_sizes_decode_for_geometry_and_leave_state_tail_unchanged(self):
        linear = self.flow()
        logged = self.flow(size_representation="log", state_feature_normalization="train")
        state = self.state()
        encoded = logged.model.state_to_model(state)
        torch.testing.assert_close(logged.model.state_to_physical(encoded), state,
                                   atol=0, rtol=0)
        actual = logged._collision_guided_x0(encoded, self.agent(encoded))
        expected = linear._collision_guided_x0(state, self.agent(state))
        torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 2:], encoded[:, 2:], atol=0, rtol=0)

    def test_displacement_matches_metric_five_circle_penetration_gradient(self):
        flow = self.flow()
        theta = torch.tensor([.2, -.8], dtype=torch.float64)
        state = torch.tensor([[-2., 4., 0., 0., 4.4, 1.8, 3.],
                              [0., 5.3, 0., 0., 3.9, 2.2, 5.]], dtype=torch.float64)
        state[:, 2:4] = torch.stack((theta.cos(), theta.sin()), -1)
        physical = state.numpy()

        def objective(position):
            circles = [compute_vehicle_circles(position[i], float(theta[i]),
                                               physical[i, 4], physical[i, 5])[0]
                       for i in range(2)]
            distances = np.linalg.norm(circles[0][:, None] - circles[1][None, :], axis=-1)
            depth = max(0., (physical[0, 5] + physical[1, 5]) / math.sqrt(3.8)
                        - float(distances.min()))
            return .5 * depth ** 2

        self.assertGreater(objective(physical[:, :2]), 0.)
        step = 1.e-5
        gradient = np.empty((2, 2))
        for row in range(2):
            for col in range(2):
                plus, minus = physical[:, :2].copy(), physical[:, :2].copy()
                plus[row, col] += step
                minus[row, col] -= step
                gradient[row, col] = (objective(plus) - objective(minus)) / (2 * step)
        expected = state[:, :2] - .5 * torch.from_numpy(gradient)
        actual = flow._collision_guided_x0(state, self.agent(state))
        torch.testing.assert_close(actual[:, :2], expected, atol=1.e-8, rtol=0)

    def test_scene_boundaries_and_nonoverlap_are_preserved(self):
        flow = self.flow()
        state = self.state()
        separate = self.agent(state, batch=torch.tensor([0, 1]), num_graphs=2)
        torch.testing.assert_close(flow._collision_guided_x0(state, separate), state,
                                   atol=0, rtol=0)
        far = state.clone()
        far[1, 1] += 30.
        torch.testing.assert_close(flow._collision_guided_x0(far, self.agent(far)), far,
                                   atol=0, rtol=0)

    def test_fixed_ego_geometry_uses_conditioned_position_heading_and_shape(self):
        flow = self.flow(fix_ego=True)
        state = self.state()
        agent = self.agent(state, ego_mask=torch.tensor([True, False]))
        reference = flow._collision_guided_x0(state, agent)
        for field, indices, values in (("position", slice(0, 2), [100., -100.]),
                                       ("heading", slice(2, 4), [0., 1.]),
                                       ("shape", slice(4, 6), [20., 8.])):
            with self.subTest(field=field):
                erroneous = state.clone()
                erroneous[0, indices] = torch.tensor(values)
                actual = flow._collision_guided_x0(erroneous, agent)
                torch.testing.assert_close(actual[1], reference[1], atol=0, rtol=0)
                # Guidance itself changes only movable positions. Sampling restores
                # fixed coordinates, and no shape/heading/speed prediction is altered.
                torch.testing.assert_close(actual[0], erroneous[0], atol=0, rtol=0)
                torch.testing.assert_close(actual[:, 2:], erroneous[:, 2:], atol=0, rtol=0)
                self.assertGreater(float(actual[1, 1] - erroneous[1, 1]), .5)

    def test_actual_sample_step_guides_against_fixed_ego_before_final_restoration(self):
        flow = self.flow(fix_ego=True)
        latent = self.state()
        prediction = latent.clone()
        prediction[0, :6] = torch.tensor([100., -100., 0., 1., 20., 8.])
        agent = self.agent(latent, ego_mask=torch.tensor([True, False]))
        time, next_time = torch.tensor(.2), torch.tensor(.15)

        def velocity(value, step, *_args, **_kwargs):
            return (value - prediction) / step.clamp_min(flow.t_eps), prediction

        with patch.object(flow, "_model_velocity", side_effect=velocity):
            result, x0, *_ = flow._sample_step(latent.clone(), time, next_time,
                                               copy.deepcopy(agent), {})
        torch.testing.assert_close(result[0], latent[0], atol=0, rtol=0)
        torch.testing.assert_close(x0[0], latent[0], atol=0, rtol=0)
        self.assertGreater(float(result[1, 1] - latent[1, 1]), .1)
        torch.testing.assert_close(result[:, 2:], latent[:, 2:], atol=0, rtol=0)

    def test_vehicles_only_uses_known_and_generated_types_in_all_processes(self):
        guidance = dict(scale=.5, t_max=.3, iters=1, margin=0., vehicles_only=True)
        state = self.state()
        for generated, process in ((False, "separate"), (True, "separate"),
                                   (True, "joint"), (True, "feature")):
            with self.subTest(generated=generated, process=process):
                flow = self.flow(generate_type=generated, type_process=process,
                                 collision_guidance=guidance)
                agent = self.agent(state, type=torch.tensor([0, 1]))
                # Generated states intentionally disagree with the GT types.
                if generated:
                    types = torch.tensor([[0., 0., 9.], [9., 0., 0.]])
                    agent["_init_diffusion_type_logits"] = types
                    joint = torch.cat((state, types), -1) if flow.joint_type else state
                else:
                    joint = state
                actual = flow._collision_guided_x0(joint, agent)
                torch.testing.assert_close(actual, joint, atol=0, rtol=0)
                if generated:
                    # Both generated cars must interact even though GT has a
                    # pedestrian: generated type, not leaked GT, defines active.
                    agent["_init_diffusion_type_logits"] = torch.tensor([[9., 0., 0.],
                                                                          [9., 0., 0.]])
                    if flow.feature_type:
                        joint = torch.cat((state, agent["_init_diffusion_type_logits"]), -1)
                    actual = flow._collision_guided_x0(joint, agent)
                    self.assertGreater(float(actual[1, 1] - joint[1, 1]), .5)

    def test_fixed_ego_type_overrides_class_prediction_in_both_directions(self):
        guidance = dict(scale=.5, t_max=.3, iters=1, margin=0., vehicles_only=True)
        flow = self.flow(generate_type=True, fix_ego=True, collision_guidance=guidance)
        state = self.state()
        for ego_type, expected_move in ((0, True), (1, False)):
            with self.subTest(ego_type=ego_type):
                logits = torch.tensor([[0., 9., 0.], [9., 0., 0.]])
                if ego_type == 1:
                    logits[0] = torch.tensor([9., 0., 0.])
                agent = self.agent(state, type=torch.tensor([ego_type, 0]),
                                   ego_mask=torch.tensor([True, False]),
                                   _init_diffusion_type_logits=logits)
                actual = flow._collision_guided_x0(state, agent)
                self.assertEqual(bool(actual[1, 1] > state[1, 1]), expected_move)

    def test_disabled_guidance_is_bit_identical_in_actual_sampling(self):
        latent = self.state()
        prediction = latent.clone()
        prediction[:, 1] += .1
        agent = self.agent(latent)
        flow = self.flow()

        def velocity(value, step, *_args, **_kwargs):
            return (value - prediction) / step.clamp_min(flow.t_eps), prediction

        with patch.object(flow, "_model_velocity", side_effect=velocity):
            flow.collision_guidance["scale"] = 0.
            disabled = flow._sample_step(latent.clone(), torch.tensor(.2), torch.tensor(.15),
                                         copy.deepcopy(agent), {})
            flow.collision_guidance = None
            absent = flow._sample_step(latent.clone(), torch.tensor(.2), torch.tensor(.15),
                                       copy.deepcopy(agent), {})
        for first, second in zip(disabled, absent):
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_training_never_applies_inference_guidance_and_inference_mode_is_supported(self):
        latent = self.state()
        agent = self.agent(latent)
        flow = self.flow().train()

        def velocity(value, step, *_args, **_kwargs):
            return (value - latent) / step.clamp_min(flow.t_eps), latent

        with patch.object(flow, "_model_velocity", side_effect=velocity), \
                patch.object(flow, "_collision_guided_x0", side_effect=AssertionError("training guidance")):
            flow._sample_step(latent.clone(), torch.tensor(.2), torch.tensor(.15), agent, {})
        flow.eval()
        with torch.inference_mode():
            actual = flow._collision_guided_x0(latent, agent)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual[1, 1] - latent[1, 1]), .5)

    def test_final_only_guidance_preserves_attribute_predictions_without_feedback(self):
        state = self.state()
        base = self.flow(fix_ego=True)
        guided = self.flow(fix_ego=True, collision_guidance=dict(
            scale=.5, t_max=.05, iters=5, margin=.1, vehicles_only=False))
        base.collision_guidance['scale'] = 0.
        agent = self.agent(state, ego_mask=torch.tensor([True, False]))

        def prediction(latent, *_args, **_kwargs):
            result = state.clone()
            # Attributes depend on the previous position: earlier guidance
            # would affect the later prediction, final-only guidance cannot.
            result[:, 4:] += latent[:, :1] * .001
            return result

        outputs = []
        for flow in (base, guided):
            torch.manual_seed(903)
            with torch.inference_mode(), \
                    patch.object(flow.model, 'forward', side_effect=prediction) as model, \
                    patch.object(flow, '_collision_guided_x0', wraps=flow._collision_guided_x0) as guide:
                outputs.append(flow.sample(copy.deepcopy(agent), {}, steps=20))
            self.assertEqual(model.call_count, 20)
            self.assertEqual(guide.call_count, 0 if flow is base else 1)
        torch.testing.assert_close(outputs[0][:, 2:], outputs[1][:, 2:], atol=0, rtol=0)
        torch.testing.assert_close(outputs[1][0], outputs[0][0], atol=0, rtol=0)
        self.assertGreater(float((outputs[1][1, :2] - outputs[0][1, :2]).norm()), .5)


if __name__ == "__main__":
    unittest.main()
