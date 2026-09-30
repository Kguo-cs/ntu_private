import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.smart.diffusion.denoiser import InitDenoiser


def make_model(x_pred=True):
    return InitDenoiser(
        token_processor=None, input_dim=8, hidden_dim=32,
        output_dim=8 if x_pred else 16, num_freq_bands=4,
        num_layers=1, num_heads=2, head_dim=8, dropout=0.0,
        x_pred=x_pred,
    )


class DenoiserHeadingMagnitudeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.model = make_model()

    @staticmethod
    def state_embedding(model, state):
        return model._original_state_embedding(
            state, torch.full((len(state), 1), 0.5),
            torch.zeros(len(state), model.hidden_dim),
        )

    def test_same_angle_different_magnitude_changes_embedding_and_prediction(self):
        state = torch.tensor([
            [2., 3., 0., .05, 4.5, 2., 5., 0.],
            [0., 0., 1., 0., 4.5, 2., 5., 0.],
        ])
        larger = state.clone()
        larger[0, 3] = 2.0
        small_embedding = self.state_embedding(self.model, state)
        large_embedding = self.state_embedding(self.model, larger)
        self.assertGreater((small_embedding[0] - large_embedding[0]).norm().item(), 0.01)

        agent = {
            'batch': torch.tensor([0, 0]), 'type': torch.tensor([0, 0]),
            'num_graphs': 1,
            'ego_feat': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.]]),
        }
        map_feature = {
            'batch': torch.tensor([0, 0, 0]),
            'position': torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
            'orientation': torch.zeros(3), 'pt_token': torch.randn(3, 32),
        }
        time = torch.tensor([[.5], [0.]])
        state.requires_grad_(True)
        predicted_small = self.model(state, time, agent, map_feature)
        predicted_large = self.model(larger, time, agent, map_feature)
        self.assertGreater(
            (predicted_small[0, :2] - predicted_large[0, :2]).norm().item(), 1e-5
        )
        gradient = torch.autograd.grad(predicted_small[0, :2].sum(), state)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(abs(gradient[0, 3].item()), 1e-5)

    def test_added_state_feature_is_rotation_invariant(self):
        state = torch.tensor([[2., 3., .3, .4, 4.5, 2., 5., 0.]])
        rotated = state.clone()
        rotated[:, 2] = -state[:, 3]
        rotated[:, 3] = state[:, 2]
        torch.testing.assert_close(
            self.state_embedding(self.model, state),
            self.state_embedding(self.model, rotated),
        )

    def test_zero_and_near_zero_magnitude_embedding_has_finite_gradients(self):
        # Scope is the new state projection, not atan2's existing singularity.
        state = torch.zeros(3, 8)
        state[:, 3] = torch.tensor([0., 1e-6, 2.])
        state.requires_grad_(True)
        embedded = self.state_embedding(self.model, state)
        self.assertTrue(torch.isfinite(embedded).all())
        embedded.square().mean().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(self.model.proj_in_m_delta.weight.grad).all())
        self.assertGreater(
            self.model.proj_in_m_delta.weight.grad[:, -1].abs().max().item(), 0.
        )

    def test_legacy_weights_load_recursively_and_new_column_can_learn(self):
        # Loading SMART invokes descendant _load_from_state_dict, rather than
        # calling each denoiser's public load_state_dict directly.
        parent = nn.ModuleDict({'generator': self.model})
        state_dict = parent.state_dict()
        key = 'generator.proj_in_m_delta.weight'
        old_weight = torch.randn(self.model.hidden_dim, 4)
        old_bias = torch.randn(self.model.hidden_dim)
        state_dict[key] = old_weight.clone()
        state_dict['generator.proj_in_m_delta.bias'] = old_bias.clone()
        parent.load_state_dict(state_dict, strict=True)
        self.assertEqual(tuple(state_dict[key].shape), (32, 4))
        torch.testing.assert_close(self.model.proj_in_m_delta.weight[:, :4], old_weight)
        torch.testing.assert_close(self.model.proj_in_m_delta.weight[:, -1], torch.zeros(32))

        state = torch.tensor([[2., 3., 0., 2., 4.5, 2., 5., 0.]])
        expected = F.linear(state[:, 4:], old_weight, old_bias)
        expected = expected + self.model._embed_time(torch.tensor([[.5]]), 1)
        actual = self.state_embedding(self.model, state)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.all(self.model.proj_in_m_delta.weight.grad[:, -1] != 0))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        optimizer.step()
        smaller = state.clone()
        smaller[:, 3] = 0.05
        self.assertGreater(
            (self.state_embedding(self.model, state) - self.state_embedding(self.model, smaller)).norm().item(),
            0.01,
        )

    def test_new_weights_round_trip_without_resetting_magnitude_column(self):
        state_dict = self.model.state_dict()
        self.assertGreater(state_dict['proj_in_m_delta.weight'][:, -1].norm().item(), 0.1)
        restored = make_model()
        restored.load_state_dict(state_dict, strict=True)
        state = torch.tensor([[2., 3., 0., 2., 4.5, 2., 5., 0.]])
        torch.testing.assert_close(
            self.state_embedding(self.model, state),
            self.state_embedding(restored, state),
        )

    def test_non_x0_model_keeps_full_state_projection(self):
        model = make_model(x_pred=False)
        self.assertEqual(model.proj_in_m_delta.in_features, 8)
        state = torch.randn(3, 8)
        expected = model.proj_in_m_delta(state) + model._embed_time(torch.full((3, 1), .5), 3)
        torch.testing.assert_close(self.state_embedding(model, state), expected)
        restored = make_model(x_pred=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(self.state_embedding(restored, state), expected)


if __name__ == '__main__':
    unittest.main()
