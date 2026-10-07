"""Split-aware labels reach denoisers; legacy cache behavior stays explicit."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
from src.smart.scenario_dreamer.map_categories import (
    scene_category_key, classify_category, load_category_keys, import_category_keys,
    SPLIT_POLICY, NATIVE_POLICY, SOURCE_RAW_SPLITS,
)
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from test_scenario_dreamer_init_decoder import make_checkpoints, official_scene


class ScenarioDreamerMapCategoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.ae, cls.ldm = make_checkpoints(cls.root)
        cls.index = cls.root / "categories.json"
        for name, keys in {
            "nocturne_train_filenames.pkl": ["tfrecord-00001-of-01000_8"],
            "nocturne_val_filenames.pkl": ["tfrecord-00001-of-00150_8"],
            "nocturne_test_filenames.pkl": ["tfrecord-00001-of-00150_10"],
        }.items():
            (cls.root / name).write_bytes(pickle.dumps(keys))
        import_category_keys(cls.root, cls.index)
        cls.prior = cls.root / "prior.npz"
        probabilities = np.zeros((2, 3, 4), dtype=np.float32)
        probabilities[:, 2, 3] = 1
        np.savez(cls.prior, probabilities=probabilities)
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def model(self, **options):
        args = dict(ae_checkpoint=self.ae, ldm_checkpoint=self.ldm,
                    map_source="exact", generation_mode="lane_conditioned",
                    map_category_index=self.index)
        args.update(options)
        return ScenarioDreamerInitDecoder(self.processor, **args)

    def inputs(self, kinds=(0, 0), labels=(None, None)):
        scenes = []
        for i, (kind, label) in enumerate(zip(kinds, labels)):
            scene = official_scene(kind)
            if label is not None:
                scene["nocturne_compatible"] = label
            filename = f"validation.tfrecord-00001-of-00150_{8+i}_{kind}_9.pkl"
            scenes.append(HeteroData(adapt_preprocessed_scene(scene, filename)))
        tokens, agent = self.processor(Batch.from_data_list(scenes))
        agent["tokenized_map"] = tokens
        return agent

    def test_lane_eval_config_uses_sim_data_weights_and_split_aware_category_index(self):
        from hydra import compose, initialize_config_dir
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver("sim_root", lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
            cfg = compose(config_name="run.yaml", overrides=[
                "experiment=scenario_dreamer_lane_conditioned", "paths.root_dir=/alternate/run/root"])
        options = cfg.model.model_config.decoder.scenario_dreamer
        self.assertEqual(Path(options.map_category_index), root /
                         "src/waymo_data/scenario_dreamer/metadata/nocturne_compatible_keys.json")
        if Path(options.map_category_index).exists():
            self.assertEqual(load_category_keys(options.map_category_index).policy, SPLIT_POLICY)
        self.assertEqual(Path(cfg.data.test_raw_dir), root /
                         "src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test")
        for field in ("ae_checkpoint", "ldm_checkpoint"):
            self.assertTrue(str(options[field]).startswith(str(root / "src/waymo_data/scenario_dreamer/checkpoints")))
        for field in ("sd_eval_set", "sd_real_cache"):
            self.assertTrue(str(cfg.model.model_config[field]).startswith(str(root / "src/waymo_data")))

    def test_filename_key_uses_record_and_ignores_split_partition_and_timestep(self):
        for split in ("training", "validation", "testing"):
            for kind in (0, 1):
                self.assertEqual(scene_category_key(
                    f"/cache/{split}.tfrecord-00001-of-00150_8_{kind}_37.pkl"),
                    "tfrecord-00001-of-00150_8")

    def test_lane_evaluation_passes_per_scene_labels_to_actual_dit(self):
        model, agent = self.model().eval(), self.inputs()
        ids, valid = agent["vectorworld_map_id"].clone(), agent["vectorworld_map_valid_mask"].clone()
        embedder = model.diff_model.model.scene_type_embedder
        with patch.object(embedder, "forward", wraps=embedder.forward) as embed:
            output = model(agent)
        self.assertTrue(embed.called)
        for call in embed.call_args_list:
            self.assertEqual(call.args[0].tolist(), [1, 0])
        self.assertEqual(len(output), 5)
        self.assertNotIn("generated_map", agent)
        torch.testing.assert_close(agent["vectorworld_map_id"], ids, atol=0, rtol=0)
        torch.testing.assert_close(agent["vectorworld_map_valid_mask"], valid, atol=0, rtol=0)
        report = model.sampling_report()
        self.assertEqual(report["map_condition_id_counts"], {"1": 1, "0": 1})
        self.assertEqual(report["map_condition_sources"], {"split_aware_nocturne_filename_index": 2})
        model.reset_sampling()
        self.assertEqual(model.sampling_report()["map_condition_id_counts"], {})
        self.assertEqual(model.sampling_report()["map_condition_sources"], {})

    def test_explicit_labels_override_index_and_do_not_require_valid_filenames(self):
        model = self.model().eval()
        agent = self.inputs(labels=(0, 1))
        agent["scenario_dreamer_cache_file"] = ["native_a.pkl", "native_b.pkl"]
        graph = model._build_graph(agent)[0]
        self.assertEqual(graph.map_id.tolist(), [0, 1])
        self.assertEqual(graph.map_condition_sources, ["metadata_nocturne_compatible"] * 2)

    def test_mixed_metadata_and_filename_labels_are_filled_per_scene(self):
        model = self.model().eval()
        agent = self.inputs(labels=(None, 1))
        graph = model._build_graph(agent)[0]
        self.assertEqual(graph.map_id.tolist(), [1, 1])
        self.assertEqual(graph.map_condition_sources,
                         ["split_aware_nocturne_filename_index", "metadata_nocturne_compatible"])
        self.assertEqual(agent["vectorworld_map_valid_mask"].tolist(), [False, True])

    def test_no_index_honors_explicit_labels_and_configured_fallback(self):
        model = self.model(map_category_index=None, map_id=1)
        graph = model._build_graph(self.inputs(labels=(0, None)))[0]
        self.assertEqual(graph.map_id.tolist(), [0, 1])
        self.assertEqual(graph.map_condition_sources,
                         ["metadata_nocturne_compatible", "fallback_config1"])

    def test_missing_or_malformed_filenames_fail_before_ae_encoding(self):
        for mutation, message in (
            (lambda a: a.pop("scenario_dreamer_cache_file"), "needs scenario_dreamer_cache_file"),
            (lambda a: a.update(scenario_dreamer_cache_file=["one.pkl"]), "align with map category"),
            (lambda a: a.update(scenario_dreamer_cache_file=["a.pkl", "b.pkl"]), "Cannot derive native Waymo"),
        ):
            with self.subTest(message=message):
                model, agent = self.model().eval(), self.inputs()
                mutation(agent)
                with patch.object(model.autoencoder, "forward_encoder") as encoder:
                    with self.assertRaisesRegex(ValueError, message):
                        model(agent)
                encoder.assert_not_called()

    def test_training_labels_reach_dit_for_joint_and_lane_objectives_all_graph_types(self):
        for mode in ("joint", "lane_conditioned"):
            with self.subTest(training_mode=mode):
                model = self.model(training_mode=mode).train()
                embedder = model.diff_model.model.scene_type_embedder
                with patch.object(embedder, "forward", wraps=embedder.forward) as embed:
                    losses = model(self.inputs(kinds=(0, 1)))
                self.assertEqual(embed.call_args.args[0].tolist(), [1, 2])
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                self.assertTrue(any(p.grad is not None for p in model.diff_model.parameters()))
                self.assertTrue(all(p.grad is None for p in model.autoencoder.parameters()))

    def test_official_prior_outputs_ignore_input_categories_and_filename_index(self):
        options = dict(generation_mode="initial_scene", scene_count_source="official_prior",
                       count_prior_path=self.prior, sampling_seed=17)
        indexed = self.model(**options).eval()
        previous = self.model(map_category_index=None, **options).eval()
        agent = self.inputs(labels=(1, 0))
        agent["scenario_dreamer_cache_file"] = ["a.pkl", "b.pkl"]
        torch.manual_seed(13)
        with patch.object(indexed, "_build_graph", side_effect=AssertionError("inference must use prior")):
            actual = indexed(agent)
        torch.manual_seed(13)
        expected = previous(self.inputs())
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        report = indexed.sampling_report()
        self.assertEqual(report["map_condition_sources"], {"official_count_prior": 2})
        self.assertEqual(report["map_condition_id_counts"], report["map_id_counts"])


class MapCategoryIndexPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_key = "tfrecord-00001-of-01000_8"
        self.val_key = "tfrecord-00001-of-00150_8"
        self.heldout_key = "tfrecord-00001-of-00150_10"
        self.sources = {
            "nocturne_train_filenames.pkl": [self.train_key],
            "nocturne_val_filenames.pkl": [self.val_key],
            "nocturne_test_filenames.pkl": [self.heldout_key],
        }
        for name, keys in self.sources.items():
            (self.root / name).write_bytes(pickle.dumps(keys))
        self.path = import_category_keys(self.root, self.root / "index.json")
        self.index = load_category_keys(self.path)

    def test_default_import_keeps_raw_split_provenance_and_all_source_hashes(self):
        data = json.loads(self.path.read_text())
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(self.index.policy, SPLIT_POLICY)
        self.assertEqual(data["source_raw_splits"], SOURCE_RAW_SPLITS)
        self.assertEqual(set(data["source_sha256"]), set(self.sources))
        self.assertEqual(self.index.training_keys, {self.train_key})
        self.assertEqual(self.index.validation_keys, {self.val_key, self.heldout_key})

    def test_collision_testing_stays_zero_and_heldout_validation_is_positive(self):
        for key in (self.val_key, self.heldout_key):
            self.assertEqual(classify_category(f"validation.{key}_0_9.pkl", self.index), 1)
            self.assertEqual(classify_category(f"testing.{key}_0_9.pkl", self.index), 0)
            self.assertEqual(classify_category(f"training.{key}_0_9.pkl", self.index), 0)
        self.assertEqual(classify_category(f"training.{self.train_key}_1_37.pkl", self.index), 1)
        self.assertEqual(classify_category(f"validation.{self.train_key}_1_37.pkl", self.index), 0)
        self.assertEqual(classify_category("validation.tfrecord-00001-of-00150_11_0_9.pkl", self.index), 0)

    def test_legacy_policy_requires_explicit_import_and_warns_on_load(self):
        path = import_category_keys(self.root, self.root / "legacy.json", policy=NATIVE_POLICY)
        with self.assertWarnsRegex(RuntimeWarning, "legacy splitless"):
            index = load_category_keys(path)
        self.assertEqual(index.policy, NATIVE_POLICY)
        # Reproduce the released bug only under this explicit legacy policy.
        self.assertEqual(classify_category(f"testing.{self.val_key}_0_9.pkl", index), 1)
        self.assertEqual(classify_category(f"validation.{self.heldout_key}_0_9.pkl", index), 0)

    def test_invalid_schema_provenance_and_key_types_fail(self):
        original = json.loads(self.path.read_text())
        mutations = [
            {"schema_version": True},
            {"schema_version": 1},
            {"policy": "unknown"},
            {"source_raw_splits": {"nocturne_test_filenames.pkl": "testing"}},
            {"source_sha256": {}},
            {"source_sha256": {name: 7 for name in self.sources}},
            {"compatible_keys_by_raw_split": {"training": [], "validation": [], "testing": [self.val_key]}},
            {"compatible_keys_by_raw_split": {"training": [False], "validation": [], "testing": []}},
            {"compatible_keys_by_raw_split": {"training": [self.train_key] * 2, "validation": [], "testing": []}},
            {"compatible_keys_by_raw_split": {"validation": []}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.path.write_text(json.dumps(dict(original, **mutation)))
                with self.assertRaises(ValueError):
                    load_category_keys(self.path)
        self.path.write_text('[]')
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_category_keys(self.path)
        with self.assertRaises(TypeError):
            classify_category(f"testing.{self.val_key}_0_9.pkl", frozenset([self.val_key]))

    def test_missing_test_whitelist_fails_instead_of_silently_using_native_policy(self):
        (self.root / "nocturne_test_filenames.pkl").unlink()
        with self.assertRaises(FileNotFoundError):
            import_category_keys(self.root, self.root / "missing.json")

    def test_import_rejects_duplicate_or_non_string_keys(self):
        for values in ([self.train_key] * 2, [2], [{"key": self.train_key}]):
            with self.subTest(values=values):
                (self.root / "nocturne_train_filenames.pkl").write_bytes(pickle.dumps(values))
                with self.assertRaises(ValueError):
                    import_category_keys(self.root, self.root / "bad.json")


if __name__ == "__main__":
    unittest.main()
