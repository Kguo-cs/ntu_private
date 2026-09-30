import copy
import importlib.util
import json
import pickle
from pathlib import Path

import tempfile
import unittest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/my_data_process/fix_initial_velocity_cache.py"
SPEC = importlib.util.spec_from_file_location("fix_initial_velocity_cache", MODULE_PATH)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def make_inputs():
    # A next token travels 5m in its start frame and finishes at 90 degrees.
    # Correct start-frame velocity is [10, 0], whereas the old endpoint-frame
    # velocity is [0, -10]. The second agent's token is not a fallback.
    contour = torch.tensor([[4., 2.], [6., 2.], [6., -2.], [4., -2.]])
    endpoints = contour[None, None].expand(3, 2, 4, 2).clone()
    pos = torch.tensor([[[1., 2.], [6., 2.]], [[8., 9.], [9., 9.]], [[20., 21.], [21., 21.]]])
    heading = torch.tensor([[0., torch.pi / 2], [0., 0.], [0., 0.]])
    shape = torch.tensor([[4., 2., 1.], [4., 2., 1.], [4., 2., 1.]])
    source = {"tokenized_agent": {
        "type": torch.tensor([0, 1, 2], dtype=torch.uint8), "num_nodes": 3,
        "sampled_pos": pos, "sampled_heading": heading, "shape": shape,
        "sampled_idx": torch.zeros((3, 2), dtype=torch.int16),
        "token_mask": torch.tensor([[False, True], [True, True], [False, False]]),
    }}
    cache = {"tokenized_agent": {
        "type": source["tokenized_agent"]["type"].clone(), "num_nodes": 3,
        "initial_pos": pos[:, 0].clone(), "initial_heading": heading[:, 0].clone(),
        "initial_shape": shape.clone(),
        "local_vel": torch.tensor([[0., -10.], [2., 3.], [4., 5.]]),
        "ego_pos2": torch.tensor([[[20., 20.], [20., 21.], [20., 22.]]]),
        "ego_heading2": torch.zeros((1, 3)),
    }, "tokenized_map": {"position": torch.tensor([[4., 5.]]), "num_nodes": 1},
        "extra_metadata": ["preserve", 3]}
    return cache, source, endpoints


def make_directories(tmp_path):
    cache, source, endpoints = make_inputs()
    cache_dir, source_dir, output_dir = [tmp_path/name for name in ("cache", "source", "corrected")]
    cache_dir.mkdir()
    source_dir.mkdir()
    torch.save(cache, cache_dir/"scene.pt")
    torch.save(source, source_dir/"scene.pt")
    token_file = tmp_path/"tokens.pkl"
    library = endpoints[0, :, None].expand(2, 6, 4, 2).clone()
    with token_file.open("wb") as file:
        pickle.dump({"token_all": {name: library.numpy() for name in migration.AGENT_NAMES}}, file)
    return cache_dir, source_dir, output_dir, token_file



class InitialVelocityCacheMigrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)


    def test_only_eligible_local_velocity_changes_and_repair_is_idempotent(self):
        cache, source, endpoints = make_inputs()
        original = copy.deepcopy(cache)
        corrected, stats = migration.repair_cache_data(cache, source, endpoints)
        assert corrected["tokenized_agent"]["local_vel"].tolist() == [[10., 0.], [2., 3.], [4., 5.]]
        assert stats["eligible_fallback_agents"] == 1
        assert stats["changed_agents"] == 1
        assert stats["unavailable_next_token_agents"] == 1
        assert migration._same_data(cache, original)
        expected = copy.deepcopy(original)
        expected["tokenized_agent"]["local_vel"][0] = torch.tensor([10., 0.])
        assert migration._same_data(corrected, expected)
        twice, second_stats = migration.repair_cache_data(corrected, source, endpoints)
        assert migration._same_data(corrected, twice)
        assert second_stats["changed_agents"] == 0


    def test_rejects_misaligned_cache_source(self):
        for field in ("type", "initial_pos", "initial_heading", "initial_shape"):
            with self.subTest(field=field):
                cache, source, endpoints = make_inputs()
                cache["tokenized_agent"][field].flatten()[0] += 1
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    migration.repair_cache_data(cache, source, endpoints)


    def test_dry_run_write_and_repeat_preserve_originals(self):
        tmp_path = self.tmp_path
        cache_dir, source_dir, output_dir, token_file = make_directories(tmp_path)
        original_cache = (cache_dir/"scene.pt").read_bytes()
        original_source = (source_dir/"scene.pt").read_bytes()
        dry = migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=output_dir)
        assert dry["mode"] == "dry-run" and dry["files_written"] == 0
        assert not output_dir.exists()
        first = migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=output_dir, write=True)
        assert first["files_written"] == 1
        assert sorted(p.name for p in output_dir.iterdir()) == ["scene.pt"]
        assert (tmp_path/"corrected.velocity-fix-report.json").is_file()
        corrected = torch.load(output_dir/"scene.pt", weights_only=False)
        assert corrected["tokenized_agent"]["local_vel"][0].tolist() == [10., 0.]
        written_bytes = (output_dir/"scene.pt").read_bytes()
        second = migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=output_dir, write=True)
        assert second["files_written"] == 0 and second["existing_outputs_verified"] == 1
        assert (output_dir/"scene.pt").read_bytes() == written_bytes
        assert (cache_dir/"scene.pt").read_bytes() == original_cache
        assert (source_dir/"scene.pt").read_bytes() == original_source
        assert not list(tmp_path.glob(".velocity-cache-*"))


    def test_existing_different_output_is_not_overwritten(self):
        tmp_path = self.tmp_path
        cache_dir, source_dir, output_dir, token_file = make_directories(tmp_path)
        output_dir.mkdir()
        torch.save({"unrelated": 42}, output_dir/"scene.pt")
        before = (output_dir/"scene.pt").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=output_dir, write=True)
        assert (output_dir/"scene.pt").read_bytes() == before


    def test_output_and_report_cannot_pollute_input_or_output_datasets(self):
        tmp_path = self.tmp_path
        cache_dir, source_dir, output_dir, token_file = make_directories(tmp_path)
        for invalid_output in (cache_dir, source_dir, cache_dir/"nested", source_dir/"nested"):
            with self.assertRaisesRegex(ValueError, "separate"):
                migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=invalid_output, write=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            migration.migrate_directory(cache_dir, source_dir, token_file, write=True)
        for invalid_report in (output_dir/"report.json", cache_dir/"report.json", source_dir/"report.json"):
            with self.assertRaisesRegex(ValueError, "Report must be outside"):
                migration.migrate_directory(cache_dir, source_dir, token_file, output_dir=output_dir,
                                            report=invalid_report, write=True)
        assert not output_dir.exists()


    def test_limit_marks_output_incomplete_and_missing_source_stops(self):
        tmp_path = self.tmp_path
        cache_dir, source_dir, output_dir, token_file = make_directories(tmp_path)
        (cache_dir/"second.pt").write_bytes((cache_dir/"scene.pt").read_bytes())
        limited = migration.migrate_directory(cache_dir, source_dir, token_file, limit=1)
        assert limited["selected_files"] == 1 and not limited["complete_dataset"]
        with self.assertRaisesRegex(ValueError, "Missing paired source"):
            migration.migrate_directory(cache_dir, source_dir, token_file)



    def test_partial_write_can_resume_and_report_stays_incomplete_on_error(self):
        cache_dir, source_dir, output_dir, token_file = make_directories(self.tmp_path)
        (cache_dir/"second.pt").write_bytes((cache_dir/"scene.pt").read_bytes())
        report = self.tmp_path/"corrected.velocity-fix-report.json"
        with self.assertRaisesRegex(ValueError, "Missing paired source"):
            migration.migrate_directory(cache_dir, source_dir, token_file,
                                        output_dir=output_dir, write=True)
        assert not json.loads(report.read_text())["completed"]
        assert sorted(path.name for path in output_dir.iterdir()) == ["scene.pt"]
        (source_dir/"second.pt").write_bytes((source_dir/"scene.pt").read_bytes())
        resumed = migration.migrate_directory(cache_dir, source_dir, token_file,
                                               output_dir=output_dir, write=True)
        assert resumed["existing_outputs_verified"] == 1
        assert resumed["files_written"] == 1 and resumed["completed"]
        assert resumed["complete_dataset"]

    def test_output_rejects_unexpected_entries(self):
        cache_dir, source_dir, output_dir, token_file = make_directories(self.tmp_path)
        output_dir.mkdir()
        (output_dir/"report.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unexpected entries"):
            migration.migrate_directory(cache_dir, source_dir, token_file,
                                        output_dir=output_dir, write=True)


if __name__ == "__main__":
    unittest.main()
