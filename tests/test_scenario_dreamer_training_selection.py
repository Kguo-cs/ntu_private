"""Full-lane training selects actual lg_type=0 scenes for raw and cached inputs."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.smart.datamodules.scalable_datamodule import MultiDataModule
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.scenario_dreamer.graph_type_index import DEFAULT_INDEX_NAME
from src.smart.scenario_dreamer.latent_cache import FORMAT_VERSION, atomic_save, cache_path


def scene(kind):
    return dict(
        lg_type=kind, scene_timestep=37, num_agents=2, num_lanes=1,
        agent_states=np.array([[0., 0., 3., 0., 1., 4., 2.], [1., 6., 8., 0., 1., 4.5, 2.]]),
        agent_types=np.eye(3)[[0, 0]],
        road_points=np.stack((np.zeros(20), np.linspace(-10., 20., 20)), -1)[None],
        edge_index_lane_to_lane=np.array([[0], [0]]), road_connection_types=np.eye(6)[[5]],
    )


class TrainingSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        # Deliberately contradict apparent filename categories.
        self.files = ["a_named_partitioned_1.pkl", "b_named_full_0.pkl", "c.pkl"]
        for name, kind in zip(self.files, (0, 1, np.int64(0))):
            self.write(name, scene(kind))

    def write(self, name, value):
        path = self.raw / name
        with path.open("wb") as handle:
            pickle.dump(value, handle)
        return path

    def dataset(self, **options):
        settings = dict(scenario_dreamer_preprocessed=True, scenario_dreamer_non_partitioned_only=True)
        settings.update(options)
        return MultiDataset(str(self.raw), WaymoTargetBuilderVal(), **settings)

    def test_mixed_raw_selection_reads_fields_and_reuses_index_without_reopening(self):
        dataset = self.dataset()
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[0], self.files[2]])
        self.assertEqual(dataset.non_partitioned_selection["total"], 3)
        self.assertEqual(dataset.non_partitioned_selection["inspected"], 3)
        self.assertEqual([dataset.get(i)["sd_lg_type"] for i in range(len(dataset))], [0, 0])
        index = self.raw / DEFAULT_INDEX_NAME
        modified = index.stat().st_mtime_ns
        with patch("src.smart.scenario_dreamer.graph_type_index._inspect", side_effect=AssertionError("must reuse")):
            again = self.dataset()
        self.assertEqual(again.non_partitioned_selection["reused"], 3)
        self.assertEqual(again.non_partitioned_selection["inspected"], 0)
        self.assertEqual(index.stat().st_mtime_ns, modified)
        self.assertEqual(again.raw_paths, dataset.raw_paths)

    def test_changed_and_new_sources_are_reinspected_and_deleted_sources_do_not_return(self):
        self.dataset()
        path = self.raw / self.files[1]
        old_time = path.stat().st_mtime_ns
        self.write(path.name, scene(0))
        os.utime(path, ns=(old_time + 1_000_000_000, old_time + 1_000_000_000))
        self.write("d.pkl", scene(torch.tensor(0)))
        (self.raw / self.files[0]).unlink()
        dataset = self.dataset()
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[1], self.files[2], "d.pkl"])
        self.assertEqual(dataset.non_partitioned_selection["inspected"], 2)
        self.assertEqual(dataset.non_partitioned_selection["reused"], 1)

    def test_explicit_sample_list_order_and_index_location_are_preserved(self):
        manifest = self.root / "selected.pkl"
        with manifest.open("wb") as handle:
            pickle.dump({"files": list(reversed(self.files))}, handle)
        index = self.root / "indices" / "types.npz"
        dataset = self.dataset(sample_list=str(manifest), scenario_dreamer_graph_type_index=str(index))
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[2], self.files[0]])
        self.assertTrue(index.is_file())
        self.assertFalse((self.raw / DEFAULT_INDEX_NAME).exists())

    def test_get_rechecks_actual_graph_type_after_selection(self):
        dataset = self.dataset()
        self.write(self.files[0], scene(1))
        with self.assertRaisesRegex(ValueError, "scene changed since selection"):
            dataset.get(0)
        bad = scene(0)
        del bad["lg_type"]
        self.write(self.files[0], bad)
        with self.assertRaisesRegex(ValueError, "Missing lg_type"):
            dataset.get(0)

    def test_missing_malformed_and_unreadable_sources_fail_without_publishing_partial_index(self):
        cases = [(None, "Missing lg_type"), (.5, "Expected lg_type"), (2, "Expected lg_type"),
                 (np.array([0, 1]), "must be scalar"), (float("nan"), "Expected lg_type")]
        for kind, message in cases:
            with self.subTest(kind=kind):
                malformed = scene(kind)
                if kind is None:
                    del malformed["lg_type"]
                self.write("bad.pkl", malformed)
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset()
                self.assertFalse((self.raw / DEFAULT_INDEX_NAME).exists())
        (self.raw / "bad.pkl").write_bytes(b"not a pickle")
        with self.assertRaisesRegex(ValueError, "Cannot inspect.*bad.pkl"):
            self.dataset()
        self.assertFalse((self.raw / DEFAULT_INDEX_NAME).exists())

    def test_corrupt_or_wrong_root_index_and_empty_selection_are_rejected(self):
        index = self.raw / DEFAULT_INDEX_NAME
        index.write_bytes(b"not a numpy archive")
        with self.assertRaisesRegex(ValueError, "Invalid graph-type index"):
            self.dataset()
        index.unlink()
        self.dataset()
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "source directory does not match"):
            MultiDataset(str(other), None, scenario_dreamer_preprocessed=True,
                         scenario_dreamer_non_partitioned_only=True,
                         scenario_dreamer_graph_type_index=str(index))
        for name in self.files:
            self.write(name, scene(1))
        with self.assertRaisesRegex(ValueError, "No non-partitioned lg_type=0"):
            self.dataset()

    def make_cached_records(self):
        cache = self.root / "cache"
        cache.mkdir()
        manifest = dict(version=FORMAT_VERSION, encoder_fingerprint="test-encoder",
                        agent_latent_dim=2, lane_latent_dim=3)
        (cache / "manifest.json").write_text(json.dumps(manifest))
        # No record for the partitioned input: filtering must happen before cache reads.
        for name in (self.files[0], self.files[2]):
            record = dict(version=FORMAT_VERSION, filename=name, encoder_fingerprint="test-encoder",
                          source_sha256=hashlib.sha256((self.raw / name).read_bytes()).hexdigest(),
                          agent_mu=torch.arange(4, dtype=torch.float32).reshape(2, 2),
                          agent_log_var=torch.zeros(2, 2), lane_mu=torch.zeros(1, 3),
                          lane_log_var=torch.zeros(1, 3))
            atomic_save(cache_path(cache, name), record)
        return cache

    def test_filtered_raw_and_cached_inputs_match_and_existing_cache_checks_remain(self):
        cache = self.make_cached_records()
        raw = self.dataset()
        cached = self.dataset(scenario_dreamer_latent_cache=str(cache))
        self.assertEqual(raw.raw_paths, cached.raw_paths)
        for i in range(len(raw)):
            expected, actual = raw.get(i), cached.get(i)
            self.assertEqual(actual["sd_lg_type"], 0)
            torch.testing.assert_close(actual["sd_agent"]["x"], expected["sd_agent"]["x"])
            torch.testing.assert_close(actual["sd_agent"]["posterior_mu"], torch.tensor([[2., 3.], [0., 1.]]))
            self.assertEqual(actual["sd_latent_cache_fingerprint"], "test-encoder")
        changed = scene(0)
        changed["agent_states"][0, 2] = 4.
        self.write(self.files[0], changed)
        with self.assertRaisesRegex(ValueError, "Stale latent cache"):
            cached.get(0)
        self.write(self.files[0], scene(1))
        with self.assertRaisesRegex(ValueError, "scene changed since selection"):
            cached.get(0)

    def test_datamodule_filters_only_training_and_supports_train_cache(self):
        cache = self.make_cached_records()
        module = MultiDataModule(
            train_batch_size=2, val_batch_size=2, test_batch_size=2,
            train_raw_dir=str(self.raw), val_raw_dir=str(self.raw), test_raw_dir=str(self.raw),
            val_tfrecords_splitted=None, shuffle=False, num_workers=0, pin_memory=False,
            persistent_workers=False, train_max_num=30, scenario_dreamer_preprocessed=True,
            scenario_dreamer_train_non_partitioned_only=True,
            scenario_dreamer_train_latent_cache=str(cache))
        module.setup("fit")
        self.assertEqual(len(module.train_dataset), 2)
        self.assertEqual(len(module.val_dataset), 3)
        batch = next(iter(module.train_dataloader()))
        self.assertTrue((batch["sd_lg_type"] == 0).all())
        self.assertIn("posterior_mu", batch["sd_agent"])
        self.assertNotIn("posterior_mu", next(iter(module.val_dataloader()))["sd_agent"])
        self.assertFalse(module.val_dataset.non_partitioned_only)
        module.setup("test")
        self.assertEqual(len(module.test_dataset), 3)
        self.assertFalse(module.test_dataset.non_partitioned_only)

    def test_filter_is_opt_in_and_requires_official_preprocessed_inputs(self):
        dataset = self.dataset(scenario_dreamer_non_partitioned_only=False)
        self.assertEqual(len(dataset), 3)
        self.assertFalse((self.raw / DEFAULT_INDEX_NAME).exists())
        with self.assertRaisesRegex(ValueError, "scenario_dreamer_preprocessed=true"):
            self.dataset(scenario_dreamer_preprocessed=False)
        with self.assertRaisesRegex(ValueError, "requires non_partitioned_only=true"):
            self.dataset(scenario_dreamer_non_partitioned_only=False,
                         scenario_dreamer_graph_type_index=str(self.root / "types.npz"))


if __name__ == "__main__":
    unittest.main()
