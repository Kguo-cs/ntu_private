"""Filename-only full-lane selection keeps raw/cache metadata checks at get time."""
import hashlib
import json
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
from src.smart.scenario_dreamer.graph_type_index import filename_graph_type, select_non_partitioned
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
        self.files = ["scene_0_0_37.pkl", "scene_1_1_37.pkl", "scene_2_0_37.pkl"]
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

    def test_initialization_selects_filenames_without_reading_scenes_or_writing_an_index(self):
        with patch("pickle.load", side_effect=AssertionError("no source reads")), \
                patch("torch.load", side_effect=AssertionError("no source reads")):
            dataset = self.dataset()
            again = self.dataset()
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[0], self.files[2]])
        self.assertEqual(dataset.non_partitioned_selection,
                         dict(total=3, selected=2, partitioned=1, selection_source="filename"))
        self.assertEqual(again.raw_paths, dataset.raw_paths)
        self.assertEqual([dataset.get(i)["sd_lg_type"] for i in range(len(dataset))], [0, 0])
        self.assertEqual(set(p.name for p in self.raw.iterdir()), set(self.files))

    def test_parser_uses_the_type_field_and_selection_needs_no_filesystem_access(self):
        paths = [Path("/not-a-directory/training_0_1_0.pkl"),
                 Path("/not-a-directory/training_1_0_1.pt"),
                 Path("/not-a-directory/training_9_0_99.pkl")]
        with patch.object(Path, "open", side_effect=AssertionError("no file reads")), \
                patch.object(Path, "stat", side_effect=AssertionError("no source stats")):
            selected, report = select_non_partitioned(paths, "/not-a-directory")
        self.assertEqual(selected, paths[1:])
        self.assertEqual(report["selection_source"], "filename")
        for path, expected in zip(paths, (1, 0, 0)):
            self.assertEqual(filename_graph_type(path), expected)
        for name in ("scene.pkl", "scene_0_2_37.pkl", "scene_0_0.5_37.pkl", "scene_0_0_bad.pkl"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Cannot determine lg_type"):
                filename_graph_type(name)

    def test_new_and_removed_files_are_selected_without_an_index(self):
        self.dataset()
        self.write("scene_3_0_37.pkl", scene(torch.tensor(0)))
        (self.raw / self.files[0]).unlink()
        dataset = self.dataset()
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[2], "scene_3_0_37.pkl"])
        self.assertEqual(dataset.non_partitioned_selection["total"], 3)

    def test_sample_list_order_and_legacy_index_paths_are_preserved_without_index_io(self):
        manifest = self.root / "selected.pkl"
        with manifest.open("wb") as handle:
            pickle.dump({"files": list(reversed(self.files))}, handle)
        index = self.root / "legacy-types.npz"
        index.write_bytes(b"legacy index is not read")
        dataset = self.dataset(sample_list=str(manifest), scenario_dreamer_graph_type_index=str(index))
        self.assertEqual([Path(p).name for p in dataset.raw_paths], [self.files[2], self.files[0]])
        self.assertEqual(index.read_bytes(), b"legacy index is not read")
        absent = self.root / "never-created" / "types.npz"
        self.dataset(scenario_dreamer_graph_type_index=str(absent))
        self.assertFalse(absent.parent.exists())

    def test_get_rejects_filename_metadata_mismatch_and_invalid_actual_fields(self):
        dataset = self.dataset()
        self.write(self.files[0], scene(1))
        with self.assertRaisesRegex(ValueError, "filename disagrees with scene metadata"):
            dataset.get(0)
        for kind, message in ((None, "Missing lg_type"), (.5, "Expected lg_type"),
                              (2, "Expected lg_type"), (np.array([0, 1]), "must be scalar")):
            with self.subTest(kind=kind):
                malformed = scene(kind)
                if kind is None:
                    del malformed["lg_type"]
                self.write(self.files[0], malformed)
                with self.assertRaisesRegex(ValueError, message):
                    dataset.get(0)

    def test_empty_selection_is_rejected_by_name_without_reading_partitioned_files(self):
        full_paths = [self.raw / name for name in (self.files[0], self.files[2])]
        for path in full_paths:
            path.unlink()
        (self.raw / self.files[1]).write_bytes(b"unused partitioned source")
        with patch("pickle.load", side_effect=AssertionError("no source reads")):
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
        with self.assertRaisesRegex(ValueError, "filename disagrees with scene metadata"):
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
        self.write("arbitrary_filename.pkl", scene(1))
        dataset = self.dataset(scenario_dreamer_non_partitioned_only=False)
        self.assertEqual(len(dataset), 4)
        with self.assertRaisesRegex(ValueError, "Cannot determine lg_type"):
            self.dataset()
        with self.assertRaisesRegex(ValueError, "scenario_dreamer_preprocessed=true"):
            self.dataset(scenario_dreamer_preprocessed=False)
        # Legacy index config remains loadable even when filtering is disabled.
        dataset = self.dataset(scenario_dreamer_non_partitioned_only=False,
                               scenario_dreamer_graph_type_index=str(self.root / "legacy.npz"))
        self.assertEqual(len(dataset), 4)
        self.assertFalse((self.root / "legacy.npz").exists())


if __name__ == "__main__":
    unittest.main()
