"""Opt-in speed parameterizations preserve the legacy scalar-x0 path."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.smart.diffusion.denoiser import InitDenoiser


class InitDiffusionSpeedHeadsTest(unittest.TestCase):
    @staticmethod
    def model(**options):
        args = dict(token_processor=SimpleNamespace(), input_dim=7,
                    hidden_dim=32, output_dim=7, num_layers=0,
                    num_heads=4, dropout=0., velocity_representation="speed")
        args.update(options)
        return InitDenoiser(**args)

    @staticmethod
    def inputs():
        state = torch.tensor([[1., 2., 1., 0., 4., 2., -3.],
                              [3., -2., 0., 1., 5., 2.2, 6.]])
        beta = torch.tensor([[.2], [.7]])
        agent = dict(batch=torch.zeros(2, dtype=torch.long),
                     type=torch.zeros(2, dtype=torch.long), num_graphs=1,
                     ego_mask=torch.tensor([True, False]),
                     ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 2., 0., 0.]]))
        return state, beta, agent

    def test_explicit_defaults_retain_all_seeded_parameters_and_return_identity(self):
        torch.manual_seed(817)
        legacy = self.model()
        torch.manual_seed(817)
        explicit = self.model(state_feature_normalization="none",
                              speed_x0_parameterization="identity",
                              speed_head_kind="shared", speed_head_detach_context=False)
        self.assertEqual(set(legacy.state_dict()), set(explicit.state_dict()))
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        state, beta, _ = self.inputs()
        self.assertIs(explicit._parameterize_speed_x0(state, state, beta), state)

    def test_separate_head_preserves_seeded_shared_parameters(self):
        torch.manual_seed(817)
        shared = self.model()
        torch.manual_seed(817)
        separate = self.model(speed_head_kind="separate")
        for key, value in shared.state_dict().items():
            self.assertTrue(torch.equal(value, separate.state_dict()[key]), key)
        extra = set(separate.state_dict()) - set(shared.state_dict())
        self.assertTrue(extra)
        self.assertTrue(all(key.startswith("to_out_speed.") for key in extra))

    def test_train_normalization_uses_saved_statistics_and_preserves_type_tail(self):
        model = self.model(state_feature_normalization="train", generate_type=True,
                           type_as_feature=True)
        state, beta, _ = self.inputs()
        state = torch.cat((state, torch.tensor([[1., 0., 0.], [0., 1., 0.]])), -1)
        original = state.clone()
        with torch.no_grad():
            model.normal_mean.copy_(torch.tensor([[0., 0., 0., 0., 4., 2., 3.]]))
            model.normal_scale.copy_(torch.tensor([[1., 1., 1., 1., 2., .5, 3.]]))
        captured = []
        handle = model.proj_in_m_delta.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        model._original_state_embedding(state, beta, None)
        handle.remove()
        expected = torch.cat(((state[:, 4:7] - model.normal_mean[:, 4:])
                              / model.normal_scale[:, 4:], state[:, 7:]), -1)
        self.assertTrue(torch.equal(captured[0], expected))
        self.assertTrue(torch.equal(state, original))

    def test_softplus_supervises_positive_speed_before_final_clamp(self):
        model = self.model(speed_x0_parameterization="softplus")
        state, beta, _ = self.inputs()
        prediction = state.clone().requires_grad_()
        output = model._parameterize_speed_x0(prediction, state, beta)
        self.assertTrue((output[:, 6] > 0).all())
        self.assertTrue(torch.equal(output[:, :6], prediction[:, :6]))
        output[:, 6].sum().backward()
        self.assertTrue((prediction.grad[:, 6] > 0).all())
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_time_residual_clean_identity_and_noise_endpoint(self):
        model = self.model(speed_x0_parameterization="time_residual")
        state, _, _ = self.inputs()
        raw = torch.ones_like(state, requires_grad=True)
        time = torch.tensor([[0.], [1.]])
        output = model._parameterize_speed_x0(raw, state, time)
        self.assertTrue(torch.equal(output[:, 6], state[:, 6] + time[:, 0]))
        output[:, 6].sum().backward()
        self.assertTrue(torch.equal(raw.grad[:, 6], time[:, 0]))
        self.assertTrue(torch.equal(output[:, :6], raw[:, :6]))

    def test_speed_head_detach_routes_speed_gradient_only_to_speed_head(self):
        model = self.model(speed_head_kind="separate", speed_head_detach_context=True)
        context = torch.randn(2, 32, requires_grad=True)
        with patch.object(model.edge_encoder, "build_interaction_edge",
                          return_value=(torch.empty((2, 0), dtype=torch.long), None)):
            output = model._apply_graph_attention(context, torch.zeros(2, 2), torch.zeros(2),
                                                  torch.zeros(2, dtype=torch.long), {}, {}, 1,
                                                  use_map_condition=False)
        output[:, 6].sum().backward()
        self.assertTrue(context.grad is None or torch.count_nonzero(context.grad) == 0)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.to_out_speed.parameters()))
        model.zero_grad(set_to_none=True)
        context = torch.randn(2, 32, requires_grad=True)
        with patch.object(model.edge_encoder, "build_interaction_edge",
                          return_value=(torch.empty((2, 0), dtype=torch.long), None)):
            output = model._apply_graph_attention(context, torch.zeros(2, 2), torch.zeros(2),
                                                  torch.zeros(2, dtype=torch.long), {}, {}, 1,
                                                  use_map_condition=False)
        output[:, :6].square().sum().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0.)

    def test_forward_applies_residual_to_selected_agent_latents(self):
        model = self.model(speed_x0_parameterization="time_residual")
        state, beta, agent = self.inputs()
        mask = torch.tensor([False, True])
        prediction = torch.tensor([[0., 0., 1., 0., 4., 2., 2.]])
        with patch.object(model, "_apply_graph_attention", return_value=prediction):
            output = model(state, beta, copy.deepcopy(agent), {}, eval_mask=mask,
                           use_map_condition=False)
        self.assertAlmostEqual(float(output[0, 6]), 6. + .7 * 2., places=5)

    def test_invalid_cross_representation_options_fail_early(self):
        cases = [dict(speed_x0_parameterization="clamp"),
                 dict(state_feature_normalization="batch"),
                 dict(speed_head_detach_context=True),
                 dict(speed_head_kind="separate", input_dim=8, output_dim=8,
                      velocity_representation="vector"),
                 dict(speed_x0_parameterization="softplus", input_dim=8, output_dim=8,
                      velocity_representation="vector")]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options)


if __name__ == "__main__":
    unittest.main()
