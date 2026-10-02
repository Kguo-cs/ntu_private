"""Pre-saved basenames avoid directory scans and eager scene stats at startup."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from src.smart.datamodules.scalable_datamodule import MultiDataModule
from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.scenario_dreamer.file_list import write_file_list
from test_scenario_dreamer_training_selection import scene


class ScenarioDreamerFileListTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "train"
        self.raw.mkdir()
        self.names = ["scene_0_0_37.pkl", "scene_1_1_37.pkl", "scene_2_0_37.pkl"]
        for name, kind in zip(self.names, (0, 1, 0)):
            with (self.raw / name).open("wb") as handle:
                pickle.dump(scene(kind), handle)
        self.manifest = self.root / "train_files.pkl"

    def dataset(self, **options):
        settings = dict(scenario_dreamer_preprocessed=True,
                        sample_list=str(self.manifest), sample_list_check_exists=False)
        settings.update(options)
        return MultiDataset(str(self.raw), None, **settings)

    def test_generation_saves_sorted_supported_basenames_without_reading_scenes(self):
        (self.raw / "notes.txt").write_text("ignored")
        (self.raw / "directory.pkl").mkdir()
        (self.raw / "scene_3_1_37.pt").write_bytes(b"scene content is not read")
        with patch("pickle.load", side_effect=AssertionError("no scene content reads")):
            report = write_file_list(self.raw)
        self.assertEqual(report["num_files"], 4)
        self.assertEqual(report["file_list"], str(self.manifest))
        with self.manifest.open("rb") as handle:
            payload = pickle.load(handle)
        self.assertEqual(payload["files"], self.names + ["scene_3_1_37.pt"])
        self.assertTrue(all(Path(name).name == name for name in payload["files"]))

    def test_loaded_list_avoids_directory_scan_and_per_scene_stats(self):
        write_file_list(self.raw)
        with patch.object(Path, "iterdir", side_effect=AssertionError("no directory scan")), \
                patch("os.scandir", side_effect=AssertionError("no directory scan")), \
                patch.object(Path, "is_file", side_effect=AssertionError("no eager scene stat")):
            dataset = self.dataset(scenario_dreamer_non_partitioned_only=True)
        self.assertEqual([Path(path).name for path in dataset.raw_paths], [self.names[0], self.names[2]])
        self.assertEqual(dataset.get(0)["sd_lg_type"], 0)
        self.assertEqual(dataset.non_partitioned_selection["total"], 3)

    def test_presaved_and_enumerated_lists_keep_identical_training_order(self):
        write_file_list(self.raw)
        enumerated = MultiDataset(str(self.raw), None, scenario_dreamer_preprocessed=True,
                                  scenario_dreamer_non_partitioned_only=True)
        self.assertEqual(self.dataset(scenario_dreamer_non_partitioned_only=True).raw_paths,
                         enumerated.raw_paths)
        self.assertEqual(len(self.dataset()), 3)

    def test_missing_scene_is_reported_on_access_and_eager_checks_remain_available(self):
        write_file_list(self.raw)
        (self.raw / self.names[0]).unlink()
        dataset = self.dataset()
        self.assertEqual(len(dataset), 3)
        with self.assertRaises(FileNotFoundError):
            dataset.get(0)
        with self.assertRaisesRegex(FileNotFoundError, "Missing 1 selected scenes"):
            self.dataset(sample_list_check_exists=True)

    def test_regeneration_updates_added_and_removed_files_and_excludes_its_own_output(self):
        inside = self.raw / "file_list.pkl"
        write_file_list(self.raw, inside)
        (self.raw / self.names[0]).unlink()
        new = "scene_4_0_37.pkl"
        (self.raw / new).write_bytes(b"new scene")
        write_file_list(self.raw, inside)
        with inside.open("rb") as handle:
            self.assertEqual(pickle.load(handle)["files"], self.names[1:] + [new])

    def test_invalid_basenames_and_duplicates_fail_even_without_eager_stats(self):
        for names in ([], ["../outside.pkl"], [".."], [self.names[0], self.names[0]], [42]):
            with self.subTest(names=names):
                with self.manifest.open("wb") as handle:
                    pickle.dump(dict(files=names), handle)
                with self.assertRaisesRegex(ValueError, "unique cache basenames"):
                    self.dataset()

    def test_datamodule_uses_train_list_without_affecting_eval_list_or_checks(self):
        write_file_list(self.raw)
        eval_list = self.root / "eval.pkl"
        with eval_list.open("wb") as handle:
            pickle.dump(dict(files=[self.names[2], self.names[0]]), handle)
        module = MultiDataModule(
            train_batch_size=2, val_batch_size=2, test_batch_size=2,
            train_raw_dir=str(self.raw), val_raw_dir=str(self.raw), test_raw_dir=str(self.raw),
            val_tfrecords_splitted=None, shuffle=False, num_workers=0, pin_memory=False,
            persistent_workers=False, train_max_num=30, scenario_dreamer_preprocessed=True,
            scenario_dreamer_train_sample_list=str(self.manifest),
            scenario_dreamer_eval_set=str(eval_list), scenario_dreamer_train_non_partitioned_only=True)
        with patch.object(Path, "iterdir", side_effect=AssertionError("no directory scan")):
            module.setup("fit")
        self.assertEqual(len(module.train_dataset), 2)
        self.assertEqual([Path(path).name for path in module.val_dataset.raw_paths],
                         [self.names[2], self.names[0]])
        (self.raw / self.names[2]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing 1 selected scenes"):
            module.setup("test")


if __name__ == "__main__":
    unittest.main()
