"""Official scene priors, independent graph generation, and SMART output contracts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
from src.smart.scenario_dreamer.generation import SceneCountPrior, build_generation_graph
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from test_scenario_dreamer_init_decoder import make_checkpoints, official_scene


class ScenarioDreamerPriorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.ae, cls.ldm = make_checkpoints(cls.root)
        cls.prior_path = cls.root / "prior.npz"
        cls.probabilities = np.zeros((2, 6, 7), dtype=np.float32)
        cls.probabilities[0, 2, 4], cls.probabilities[0, 3, 5] = .4, .6
        cls.probabilities[1, 4, 6], cls.probabilities[1, 5, 4] = .5, .5
        np.savez(cls.prior_path, probabilities=cls.probabilities)
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def prior(self, path=None):
        return SceneCountPrior(path or self.prior_path, max_num_agents=30, max_num_lanes=100, seed=17)

    def model(self, **options):
        args = dict(ae_checkpoint=self.ae, ldm_checkpoint=self.ldm, map_source="exact",
                    scene_count_source="official_prior", count_prior_path=self.prior_path, sampling_seed=17)
        args.update(options)
        return ScenarioDreamerInitDecoder(self.processor, **args)

    def inputs(self):
        batch = Batch.from_data_list([HeteroData(adapt_preprocessed_scene(official_scene(), f"{i}.pkl"))
                                     for i in range(2)])
        tokens, agent = self.processor(batch)
        agent["tokenized_map"] = tokens
        return agent

    def test_prior_preserves_float32_and_official_multinomial_order(self):
        prior = self.prior()
        torch.testing.assert_close(prior.probabilities, torch.from_numpy(self.probabilities.reshape(2, -1)),
                                   atol=0, rtol=0)
        global_state = torch.random.get_rng_state()
        sampled = prior.sample(30)
        torch.testing.assert_close(global_state, torch.random.get_rng_state(), atol=0, rtol=0)
        rng = torch.Generator().manual_seed(17)
        expected = []
        for _ in range(30):
            m = int(torch.multinomial(torch.tensor([.62, .38]), 1, generator=rng))
            count = int(torch.multinomial(torch.from_numpy(self.probabilities[m].reshape(-1)), 1, generator=rng))
            expected.append((m, count // 7, count % 7))
        np.testing.assert_array_equal(sampled, expected)
        prior.reset()
        np.testing.assert_array_equal(sampled, np.concatenate((prior.sample(9), prior.sample(21))))
        report = prior.report()
        self.assertEqual(report["num_scenes"], 30)
        self.assertEqual(report["num_agents"], int(sampled[:, 2].sum()))
        self.assertEqual(report["num_lanes"], int(sampled[:, 1].sum()))
        self.assertEqual(sum(report["map_id_counts"].values()), 30)
        self.assertEqual(len(report["count_prior_sha256"]), 64)

    def test_generated_graph_counts_edges_and_ego_order(self):
        graph, rows, centers, angles = build_generation_graph([(0, 2, 4), (1, 3, 5)],
            agent_latent_dim=8, lane_latent_dim=24, device="cpu", dtype=torch.float32)
        self.assertEqual(graph.num_agents.tolist(), [4, 5])
        self.assertEqual(graph.num_lanes.tolist(), [2, 3])
        self.assertEqual(graph.map_id.tolist(), [0, 1])
        self.assertEqual(rows.tolist(), [3, 0, 1, 2, 8, 4, 5, 6, 7])
        self.assertFalse(graph.lg_type.any())
        self.assertFalse(centers.any())
        self.assertFalse(angles.any())
        for source, target, total in (("agent", "agent", 41), ("lane", "lane", 13), ("lane", "agent", 23)):
            edge = graph[source, "to", target].edge_index
            self.assertEqual(edge.shape, (2, total))
            torch.testing.assert_close(graph[source].batch[edge[0]], graph[target].batch[edge[1]])

    def test_real_decoder_changes_counts_ignores_gt_and_resets(self):
        model = self.model().eval()
        agent = self.inputs()
        original = copy.deepcopy(agent)
        with patch.object(model, "_build_graph", side_effect=AssertionError("must not access GT graph")):
            torch.manual_seed(5)
            first = model(agent)
            self.assertGreater(len(first[0]), len(original["initial_pos"]))
            self.assertEqual(len(agent["batch"]), len(first[0]))
            self.assertEqual(agent["ego_mask"].sum().item(), 2)
            self.assertEqual(agent["shape"].shape, (len(first[0]), 2))
            self.assertEqual(len(agent["type"]), len(first[0]))
            self.assertNotIn("sd_states", agent)
            self.assertNotIn("sd_map", agent)
            first_map = agent["generated_map"]["road_points"].clone()
            report = model.sampling_report()
            self.assertEqual(report["num_agents"], len(first[0]))
            self.assertEqual(report["num_lanes"], len(first_map))
            self.assertEqual(report["num_scenes"], 2)
            model.reset_sampling()
            self.assertEqual(model.sampling_report()["num_scenes"], 0)
            original["sd_states"].fill_(float("nan"))
            original["sd_map"]["lanes"].fill_(float("nan"))
            original["initial_pos"].fill_(1234.)
            torch.manual_seed(5)
            second = model(original)
        for actual, expected in zip(second, first):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(original["generated_map"]["road_points"], first_map, atol=0, rtol=0)

    def test_training_keeps_existing_objective_and_guard_rejects_other_data(self):
        model = self.model().train()
        with patch.object(model, "_build_generation_graph", side_effect=AssertionError("inference only")):
            losses = model(self.inputs())
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(model.sampling_report()["num_scenes"], 0)
        agent = self.inputs()
        agent.pop("initial_scene_only")
        with self.assertRaisesRegex(ValueError, "direct preprocessed"):
            model.eval()(agent)
        with self.assertRaisesRegex(ValueError, "initial_scene"):
            self.model(generation_mode="lane_conditioned")

    def test_prior_rejects_invalid_probability_arrays(self):
        invalid = []
        zero_agent = self.probabilities.copy()
        zero_agent[0, 1, 0] = 1
        invalid.append(zero_agent)
        negative = self.probabilities.copy()
        negative[0, 2, 4] = -1
        invalid.append(negative)
        invalid.append(np.zeros_like(self.probabilities))
        invalid.append(np.ones((2, 102, 31), dtype=np.float32))
        invalid.append(np.full_like(self.probabilities, np.nan))
        for i, probabilities in enumerate(invalid):
            path = self.root / f"invalid_{i}.npz"
            np.savez(path, probabilities=probabilities)
            with self.subTest(index=i), self.assertRaises(ValueError):
                self.prior(path)


if __name__ == "__main__":
    unittest.main()
