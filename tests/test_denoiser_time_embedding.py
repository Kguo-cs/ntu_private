import unittest

import torch

from src.smart.diffusion.denoiser import InitDenoiser


def make_model(x_pred=True):
    return InitDenoiser(
        token_processor=None,
        input_dim=8,
        hidden_dim=32,
        output_dim=8 if x_pred else 16,
        num_freq_bands=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
        x_pred=x_pred,
    )


class DenoiserTimeEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.model = make_model()

    @staticmethod
    def state_embedding(model, time):
        n = len(time)
        return model._original_state_embedding(
            torch.zeros(n, 8), time, torch.zeros(n, model.hidden_dim)
        )

    def test_fresh_model_distinguishes_noise_levels_and_endpoints(self):
        # Exercise the actual state-embedding path before any bias is learned.
        # Repeated scalar inputs followed by LayerNorm fail this regression.
        for x_pred in (True, False):
            with self.subTest(x_pred=x_pred):
                model = make_model(x_pred)
                time = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
                embedding = self.state_embedding(model, time)
                norms = embedding.norm(dim=-1)
                self.assertTrue(torch.all(norms > 0.1))
                distances = torch.pdist(embedding)
                self.assertGreater((distances.min() / norms.max()).item(), 0.05)

    def test_supported_time_layouts_and_train_eval_agree(self):
        time = torch.tensor([0.0, 0.1, 0.5, 1.0])
        expected = self.state_embedding(self.model, time)
        for layout in (
            time[:, None],
            time[:, None].expand(-1, 8),
            time[:, None, None],
            time[:, None, None].expand(-1, 1, 8),
        ):
            with self.subTest(shape=tuple(layout.shape)):
                torch.testing.assert_close(
                    self.state_embedding(self.model, layout), expected
                )
        self.model.eval()
        torch.testing.assert_close(self.state_embedding(self.model, time), expected)

    def test_full_forward_has_finite_time_and_parameter_gradients(self):
        state = torch.tensor([
            [2., 3., 1., 0., 4.5, 2., 5., 0.],
            [0., 0., 1., 0., 4.5, 2., 5., 0.],
        ])
        agent = {
            'batch': torch.tensor([0, 0]),
            'type': torch.tensor([0, 0]),
            'num_graphs': 1,
            'ego_feat': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.]]),
        }
        map_feature = {
            'batch': torch.tensor([0, 0, 0]),
            'position': torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
            'orientation': torch.zeros(3),
            'pt_token': torch.randn(3, self.model.hidden_dim),
        }
        for t in (0.0, 0.5, 1.0):
            with self.subTest(time=t):
                self.model.zero_grad(set_to_none=True)
                time = torch.full((2, 1), t, requires_grad=True)
                prediction = self.model(state, time, agent, map_feature)
                self.assertEqual(tuple(prediction.shape), (2, 8))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(time.grad).all())
                self.assertGreater(time.grad.abs().max().item(), 1e-6)
                for parameter in self.model.noise_embedding.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0)

    def test_time_features_add_no_checkpoint_parameters(self):
        state = self.model.state_dict()
        self.assertFalse(any(key.startswith('_time_') for key in state))
        self.assertEqual(tuple(state['noise_embedding.mlp.0.weight'].shape), (32, 8))
        restored = make_model()
        restored.load_state_dict(state, strict=True)
        time = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        torch.testing.assert_close(
            self.state_embedding(restored, time), self.state_embedding(self.model, time)
        )


if __name__ == '__main__':
    unittest.main()
