"""Cache/online equivalence, source provenance, ordering and resumable generation."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
from src.smart.scenario_dreamer.latent_cache import cache_path, precache, read_manifest
from src.smart.tokens.token_processor import TokenProcessor
from test_scenario_dreamer_init_decoder import make_checkpoints, official_scene


class ScenarioDreamerLatentCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.cache = self.root / "raw", self.root / "cache"
        self.raw.mkdir()
        ae_path, ldm_path = make_checkpoints(self.root)
        self.ldm_config = torch.load(ldm_path, weights_only=False)["hyper_parameters"]["cfg"]
        self.ae = ScenarioDreamerInitDecoder(self.processor, ae_checkpoint=ae_path,
                                            training_stage="autoencoder", map_source="exact")
        self.ldm = ScenarioDreamerInitDecoder(self.processor, ae_checkpoint=ae_path, map_source="exact",
                                             ldm_config={"model": self.ldm_config.model})
        for index, kind in enumerate((0, 1, 1, 0)):
            scene = official_scene(kind)
            if index == 2:
                scene["agent_states"] = scene["agent_states"][[0, 2, 1]]
                scene["agent_types"] = scene["agent_types"][[0, 2, 1]]
                scene["road_points"] = scene["road_points"][[1, 0]]
            if index == 3:
                scene["agent_states"] = scene["agent_states"][:1]
                scene["agent_types"] = scene["agent_types"][:1]
                scene["num_agents"] = 1
            with (self.raw / f"{index}.pkl").open("wb") as handle:
                pickle.dump(scene, handle)

    def dataset(self, cached=False, recording=False):
        return MultiDataset(str(self.raw), WaymoTargetBuilderVal(), scenario_dreamer_preprocessed=True,
                            scenario_dreamer_latent_cache=str(self.cache) if cached else None,
                            record_latent_source=recording)

    def generate(self, **options):
        return precache(self.ae, self.dataset(recording=True), self.cache,
                        batch_size=2, device="cpu", **options)

    def agents(self, cached):
        batch = next(iter(DataLoader(self.dataset(cached=cached), batch_size=4)))
        tokenized_map, agent = self.processor(batch)
        agent["tokenized_map"] = tokenized_map
        return agent

    def test_cached_posterior_matches_encoder_across_orders_partitions_and_batch_sizes(self):
        summary = self.generate()
        self.assertEqual(summary["written"], 4)
        self.assertFalse(summary["partial"])
        self.assertGreater(summary["latent_statistics"]["agent_latents_std"], 0)
        raw, cached = self.agents(False), self.agents(True)
        raw_graph = self.ldm._build_graph(raw)[0]
        graph = self.ldm._build_graph(cached)[0]
        with torch.no_grad():
            am, lm, av, lv = self.ldm.autoencoder.forward_encoder(raw_graph, return_stats=True)
        for actual, expected in ((graph["agent"].posterior_mu, am), (graph["agent"].posterior_log_var, av),
                                 (graph["lane"].posterior_mu, lm), (graph["lane"].posterior_log_var, lv)):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        for kind in ("agent", "lane"):
            torch.testing.assert_close(graph[kind].partition_mask, raw_graph[kind].partition_mask)
        expected = self.ldm.eval()._encode(raw)[0]
        with patch.object(self.ldm.autoencoder, "forward_encoder", side_effect=AssertionError("AE must be skipped")):
            actual = self.ldm._encode(cached)[0]
        for kind in ("agent", "lane"):
            torch.testing.assert_close(actual[kind].latents, expected[kind].latents, atol=2e-5, rtol=2e-5)

    def test_cached_training_preserves_posterior_resampling_loss_and_gradients(self):
        self.generate()
        raw, cached = self.agents(False), self.agents(True)
        self.ldm.train()
        torch.manual_seed(29)
        expected = self.ldm(raw)
        torch.manual_seed(29)
        with patch.object(self.ldm.autoencoder, "forward_encoder", side_effect=AssertionError("AE must be skipped")):
            actual = self.ldm(cached)
            first = self.ldm._encode(cached)[0]["agent"].latents
            second = self.ldm._encode(cached)[0]["agent"].latents
        self.assertFalse(torch.equal(first, second))
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-5)
        actual["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.ldm.diff_model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.ldm.autoencoder.parameters()))
        torch.optim.SGD(self.ldm.diff_model.parameters(), lr=.01).step()
        self.ldm.update_ema()
        self.assertEqual(self.ldm.ema.num_updates, 1)

    def test_cache_resume_fills_missing_scenes_and_reuses_valid_records(self):
        first = self.generate(max_scenes=2)
        self.assertTrue(first["partial"])
        second = self.generate()
        self.assertEqual((second["written"], second["reused"]), (2, 2))
        before = cache_path(self.cache, "0.pkl").stat().st_mtime_ns
        with patch.object(self.ae.autoencoder, "forward_encoder", side_effect=AssertionError("Already cached")):
            third = self.generate()
        self.assertEqual((third["written"], third["reused"]), (0, 4))
        self.assertEqual(cache_path(self.cache, "0.pkl").stat().st_mtime_ns, before)
        self.assertEqual(second["latent_statistics"], third["latent_statistics"])

    def test_stale_input_missing_and_corrupt_records_fail_explicitly(self):
        self.generate()
        path = self.raw / "0.pkl"
        with path.open("rb") as handle:
            scene = pickle.load(handle)
        scene["agent_states"][1, 0] += 1
        with path.open("wb") as handle:
            pickle.dump(scene, handle)
        with self.assertRaisesRegex(ValueError, "source scene changed"):
            self.dataset(cached=True)[0]
        with self.assertRaisesRegex(ValueError, "source scene changed"):
            self.generate()
        self.generate(overwrite=True)
        record_path = cache_path(self.cache, "0.pkl")
        record = torch.load(record_path, weights_only=True)
        record["agent_mu"] = torch.zeros(1, 1)
        torch.save(record, record_path)
        with self.assertRaisesRegex(ValueError, "Invalid agent_mu"):
            self.dataset(cached=True)[0]
        record_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing latent cache"):
            self.dataset(cached=True)[0]

    def test_ae_reload_invalidates_fingerprint_and_rejects_old_cache(self):
        self.generate()
        cached = self.agents(True)
        self.ldm._encode(cached)
        previous = self.ldm._latent_cache_fingerprint
        state = {k: v.clone() for k, v in self.ldm.autoencoder.state_dict().items()}
        state[next(iter(state))].add_(.1)
        self.ldm.autoencoder.load_state_dict(state)
        self.assertIsNone(self.ldm._latent_cache_fingerprint)
        with self.assertRaisesRegex(ValueError, "does not match the current AE"):
            self.ldm._encode(cached)
        self.assertNotEqual(previous, self.ldm._latent_cache_fingerprint)
        manifest = read_manifest(self.cache)
        self.ae.autoencoder.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "different AE/config"):
            self.generate(overwrite=True)
        self.assertEqual(read_manifest(self.cache), manifest)

    def test_cache_keeps_raw_latents_for_new_normalization_and_rejects_ae_training(self):
        self.generate()
        cached = self.agents(True)
        self.ldm.cfg.dataset.agent_latents_mean = 3
        self.ldm.cfg.dataset.agent_latents_std = 2
        graph = self.ldm.eval()._encode(cached)[0]
        torch.testing.assert_close(graph["agent"].latents, (graph["agent"].posterior_mu - 3) / 2)
        with self.assertRaisesRegex(ValueError, "disable latent caches"):
            self.ae(cached)
        self.ldm.map_source = "tokens"
        with self.assertRaisesRegex(ValueError, "exact preprocessed map"):
            self.ldm._encode(cached)

    def test_datamodule_uses_cache_only_for_selected_split(self):
        from src.smart.datamodules.scalable_datamodule import MultiDataModule
        self.generate()
        module = MultiDataModule(
            train_batch_size=2, val_batch_size=2, test_batch_size=2,
            train_raw_dir=str(self.raw), val_raw_dir=str(self.raw), test_raw_dir=str(self.raw),
            val_tfrecords_splitted=None, shuffle=False, num_workers=0, pin_memory=False,
            persistent_workers=False, train_max_num=30, scenario_dreamer_preprocessed=True,
            scenario_dreamer_train_latent_cache=str(self.cache),
        )
        module.setup("fit")
        self.assertIn("posterior_mu", next(iter(module.train_dataloader()))["sd_agent"])
        self.assertNotIn("posterior_mu", next(iter(module.val_dataloader()))["sd_agent"])
        module.setup("test")
        self.assertNotIn("posterior_mu", next(iter(module.test_dataloader()))["sd_agent"])


if __name__ == "__main__":
    unittest.main()
