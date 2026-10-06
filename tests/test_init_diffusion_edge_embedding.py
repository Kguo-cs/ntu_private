"""Graph-relation embedding selection in the initial-state denoiser."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.edge_encoder import EdgeEncoder


class InitDiffusionEdgeEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(817)

    @staticmethod
    def processor(use_refiner=False):
        return SimpleNamespace(use_refiner=use_refiner, learn_init=True, init_map_range=50.)

    @staticmethod
    def args(**options):
        settings = dict(
            input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
            num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
            sampling_steps=2, use_rl=False,
        )
        settings.update(options)
        return SimpleNamespace(**settings)

    def denoiser(self, embedding=None, *, x_pred=True):
        options = dict(
            token_processor=None, input_dim=8, hidden_dim=32,
            output_dim=8 if x_pred else 16, num_layers=1, num_heads=2,
            dropout=0., x_pred=x_pred,
        )
        if embedding is not None:
            options["edge_embedding_type"] = embedding
        return InitDenoiser(**options)

    def wrapper(self, embedding=None, **options):
        if embedding is not None:
            options["edge_embedding_type"] = embedding
        with patch.object(InitDiffusion, "_make_args", return_value=self.args()):
            return InitDiffusion(128, 8, 64, self.processor(), False, **options)

    @staticmethod
    def inputs(hidden_dim=32):
        state = torch.tensor([
            [2., 3., 1., 0., 4.5, 2., 5., 0.],
            [0., 0., .98, .2, 4., 1.8, 3., .5],
        ])
        agent = dict(
            batch=torch.tensor([0, 0]), type=torch.tensor([0, 0]), num_graphs=1,
            ego_mask=torch.tensor([False, True]), initial_pos=state[:, :2].clone(),
            initial_heading=torch.tensor([0., .2]), expert_input=state.clone(),
            ego_feat=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.]]),
        )
        feature = dict(
            batch=torch.zeros(3, dtype=torch.long),
            position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.]]),
            orientation=torch.tensor([0., .3, -.4]), pt_token=torch.randn(3, hidden_dim),
        )
        return state, agent, feature

    def assert_embeddings(self, model, expected):
        self.assertIsInstance(model.edge_encoder.r_a2a_emb, expected)
        self.assertIsInstance(model.edge_encoder.r_pt2a_emb, expected)

    def test_actual_mlp_agent_and_map_relations_have_finite_input_and_parameter_gradients(self):
        encoder = EdgeEncoder(
            hidden_dim=32, num_freq_bands=4, use_a2a=True, use_pl2a=True,
            embedding_type="mlp",
        )
        self.assertIsInstance(encoder.r_a2a_emb, MLPEmbedding)
        self.assertIsInstance(encoder.r_pt2a_emb, MLPEmbedding)
        positions = torch.tensor([[0., 0.], [2., 3.]], requires_grad=True)
        headings = torch.tensor([.2, -.4], requires_grad=True)
        vectors = torch.stack([headings.cos(), headings.sin()], dim=-1)
        batch = torch.zeros(2, dtype=torch.long)
        edges = torch.tensor([[0, 1], [1, 0]])
        actual_edges, interaction, *_ = encoder.build_interaction_edge(
            positions, headings, vectors, batch, None, 4, 50., a2a_edge_index=edges,
        )
        torch.testing.assert_close(actual_edges, edges)
        map_positions = torch.tensor([[-1., 2.], [4., -3.]], requires_grad=True)
        map_headings = torch.tensor([.3, -.1], requires_grad=True)
        map_edges, map_relation = encoder.build_map2agent_edge(
            map_positions, map_headings, positions, headings, vectors, None,
            batch, batch, 50., 4, l2a_edge_index=edges,
        )
        torch.testing.assert_close(map_edges, edges)
        for relation in (interaction, map_relation):
            self.assertEqual(relation.shape, (2, 32))
            self.assertTrue(torch.isfinite(relation).all())
        (interaction.square().mean() + map_relation.square().mean()).backward()
        for value in (positions, headings, map_positions, map_headings):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.)
        for embedding in (encoder.r_a2a_emb, encoder.r_pt2a_emb):
            gradient = embedding.mlp[0].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_real_mlp_denoiser_forward_and_backward_work_in_training_and_evaluation(self):
        for x_pred in (True, False):
            with self.subTest(x_pred=x_pred):
                model = self.denoiser("mlp", x_pred=x_pred)
                self.assert_embeddings(model, MLPEmbedding)
                state, agent, feature = self.inputs()
                state.requires_grad_(True)
                time = torch.tensor([[.5], [.0]], requires_grad=True)
                prediction = model(state, time, agent, feature)
                self.assertEqual(prediction.shape, (2, 8 if x_pred else 16))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertTrue(torch.isfinite(time.grad).all())
                for embedding in (model.edge_encoder.r_a2a_emb, model.edge_encoder.r_pt2a_emb):
                    gradient = embedding.mlp[0].weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertGreater(gradient.abs().sum().item(), 0.)
                model.eval()
                with torch.no_grad():
                    evaluated = model(state.detach(), time.detach(), agent, feature)
                torch.testing.assert_close(evaluated, prediction.detach())

    def test_mlp_map_relations_preserve_categorical_tensor_values_and_gradients(self):
        encoder = EdgeEncoder(32, 4, use_pl2a=True, embedding_type="mlp")
        map_positions = torch.tensor([[-1., 2.], [4., -3.]])
        map_headings = torch.tensor([.3, -.1])
        positions = torch.tensor([[0., 0.], [2., 3.]])
        headings = torch.tensor([.2, -.4])
        vectors = torch.stack([headings.cos(), headings.sin()], dim=-1)
        batch = torch.zeros(2, dtype=torch.long)
        edges = torch.tensor([[0, 1], [1, 0]])
        categorical = torch.randn(2, 32, requires_grad=True)
        arguments = (map_positions, map_headings, positions, headings, vectors,
                     batch, batch, 50., 4)
        _, base = encoder.build_map2map_edge(*arguments, l2l_edge_index=edges)
        actual_edges, relation = encoder.build_map2map_edge(
            *arguments, l2l_edge_index=edges, l2l_feature=categorical,
        )
        torch.testing.assert_close(actual_edges, edges)
        torch.testing.assert_close(relation, base + categorical)
        self.assertTrue(torch.isfinite(relation).all())
        relation.square().mean().backward()
        for gradient in (categorical.grad, encoder.r_pt2a_emb.mlp[0].weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_mlp_relations_accept_empty_maps_and_empty_edges(self):
        model = self.denoiser("mlp")
        state, agent, feature = self.inputs()
        feature = {key: value[:0] for key, value in feature.items()}
        prediction = model(state, torch.full((2, 1), .5), agent, feature)
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.sum().backward()
        encoder = model.edge_encoder
        empty = torch.empty(2, 0, dtype=torch.long)
        _, relation, *_ = encoder.build_interaction_edge(
            state[:, :2], agent["initial_heading"],
            torch.stack([agent["initial_heading"].cos(), agent["initial_heading"].sin()], dim=-1),
            agent["batch"], None, 4, 50., a2a_edge_index=empty,
        )
        self.assertEqual(relation.shape, (0, 32))

    def test_flow_and_wrapper_pass_selection_to_both_base_and_refiner_denoisers(self):
        for selection, expected in (("fourier", FourierEmbedding), ("mlp", MLPEmbedding)):
            for use_refiner in (False, True):
                with self.subTest(selection=selection, refiner=use_refiner):
                    flow = Flow(self.args(edge_embedding_type=selection), self.processor(use_refiner), False)
                    self.assert_embeddings(flow.model, expected)
                    if use_refiner:
                        self.assert_embeddings(flow.refine_model, expected)
        wrapper = self.wrapper("mlp")
        self.assert_embeddings(wrapper.G1.model, MLPEmbedding)
        legacy_flow = Flow(self.args(), self.processor(), False)
        self.assert_embeddings(legacy_flow.model, FourierEmbedding)
        self.assert_embeddings(self.wrapper().G1.model, FourierEmbedding)

    def test_default_fourier_layout_and_strict_legacy_checkpoint_roundtrip_are_preserved(self):
        source = self.wrapper()
        target = self.wrapper("fourier")
        state = source.state_dict()
        self.assertIn("G1.model.edge_encoder.r_a2a_emb.freqs.weight", state)
        self.assertIn("G1.model.edge_encoder.r_pt2a_emb.mlps.0.0.weight", state)
        self.assertFalse(any("edge_encoder.r_a2a_emb.mlp." in key for key in state))
        state.pop("_extra_state")
        incompatible = target.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for key, value in source.G1.state_dict().items():
            torch.testing.assert_close(target.G1.state_dict()[key], value, rtol=0, atol=0)

    def test_mlp_wrapper_ema_checkpoint_restores_parameters_and_average_graph_inference(self):
        source = self.wrapper("mlp", use_ema=True, ema_decay=.9)
        with torch.no_grad():
            for parameter in source.G1.parameters():
                parameter.add_(.05)
        source.update_ema()
        source.eval()
        state, agent, feature = self.inputs(hidden_dim=128)
        agent["map_feature"] = feature

        def infer(wrapper, inputs, map_feature):
            return wrapper.G1.model(inputs["expert_input"], torch.full((2, 1), .5), inputs, map_feature)

        with patch.object(source, "_infer", side_effect=lambda inputs, feature: infer(source, inputs, feature)):
            expected = source(agent)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertFalse(expected.requires_grad)
        target = self.wrapper("mlp", use_ema=True, ema_decay=.7)
        incompatible = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(target.ema.decay, .9)
        self.assertEqual(target.ema.num_updates, 1)
        for actual, saved in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        target.eval()
        with patch.object(target, "_infer", side_effect=lambda inputs, feature: infer(target, inputs, feature)):
            actual = target(agent)
        torch.testing.assert_close(actual, expected)
        for key, value in source.G1.state_dict().items():
            torch.testing.assert_close(target.G1.state_dict()[key], value, rtol=0, atol=0)

    def test_embedding_architecture_changes_require_matching_checkpoint_weights(self):
        with self.assertRaises(RuntimeError):
            self.denoiser("mlp").load_state_dict(self.denoiser("fourier").state_dict(), strict=True)

    def test_invalid_embedding_types_are_rejected_at_each_public_entry_point(self):
        for constructor in (
            lambda: EdgeEncoder(32, 4, use_a2a=True, embedding_type="unknown"),
            lambda: self.denoiser("unknown"),
            lambda: Flow(self.args(edge_embedding_type="unknown"), self.processor(), False),
            lambda: self.wrapper("unknown"),
        ):
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()


if __name__ == "__main__":
    unittest.main()
