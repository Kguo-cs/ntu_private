"""Exact motion labels, source lookup and ordered output for local preprocessing."""
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.prepare_vectorworld_data import (
    WaymoSourceReader, convert_full_scene, parse_source_name, prepare, select_inputs, strict_state_order,
)


def full_scene():
    n, steps = 3, 10
    # Native order is ego, non-ego A, non-ego B; SMART is B, ego, A.
    positions = torch.zeros(n, steps, 2, dtype=torch.float64)
    positions[0, :, 0] = torch.arange(steps)
    positions[1, :, 0] = 10 + torch.arange(steps) * .5
    positions[2, :, 1] = 20
    native_states = np.array([[9, 0, 2, 1, 0, 4, 2],
                              [14.5, 0, 1, 1, 0, 4, 2],
                              [0, 20, 0, 1, 0, 4, 2]], dtype=np.float64)
    road = np.stack([np.column_stack((np.zeros(20), np.linspace(-5, 5, 20))),
                     np.column_stack((np.ones(20), np.linspace(-5, 5, 20)))])
    graph = dict(road_points=road, num_lanes=2, pre_adj=np.array([[0, 1], [0, 0]]),
                 suc_adj=np.zeros((2, 2)), left_adj=np.zeros((2, 2)), right_adj=np.zeros((2, 2)))
    info = dict(valid_scene=True, lg_type=0, scene_timestep=9, num_agents=n, num_lanes=2,
                agent_states=native_states, agent_types=np.eye(3)[[0, 1, 0]],
                source_index=torch.tensor([10, 20, 30]), output_source_index=torch.tensor([30, 10, 20]),
                graphs={"regular": graph, "partitioned": graph})
    velocities = torch.zeros(n, steps, 2, dtype=torch.float64)
    velocities[0, :, 0], velocities[1, :, 0] = 2, 1
    order = torch.tensor([2, 0, 1])
    return dict(scenario_dreamer=info, scenario_dreamer_cache_file="testing.tfrecord-00000-of-00150_2_0_9.pkl",
                agent=dict(position=positions[order], velocity=velocities[order],
                           heading=torch.zeros(n, steps), valid_mask=torch.ones(n, steps, dtype=torch.bool)))


class PrepareVectorWorldDataTest(unittest.TestCase):
    def test_source_basename_recovers_split_record_partition_and_time(self):
        actual = parse_source_name("training.tfrecord-00551-of-01000_335_1_73.pkl")
        self.assertEqual(actual, dict(tfrecord="training.tfrecord-00551-of-01000", split="training",
                                      record_index=335, lg_type=1, scene_timestep=73))
        with self.assertRaisesRegex(ValueError, "Cannot locate raw Waymo source"):
            parse_source_name("scenario_id_0_11.pkl")

    def test_full_tracks_motion_reorders_by_source_ids_and_preserves_geometry(self):
        data = full_scene()
        output = convert_full_scene(data, "full.pt")
        torch.testing.assert_close(torch.from_numpy(output["agent_motion_raw"][:, -2:]), torch.zeros(3, 2))
        self.assertEqual(output["agent_motion_is_static"].tolist(), [False, False, True])
        self.assertTrue(output["agent_motion_valid_mask"].all())
        np.testing.assert_array_equal(output["agent_states"], data["scenario_dreamer"]["agent_states"])
        np.testing.assert_array_equal(output["road_points"], data["scenario_dreamer"]["graphs"]["regular"]["road_points"])
        self.assertEqual(output["road_connection_types"].argmax(-1).tolist(), [5, 0, 1, 5])
        self.assertNotIn("agent_motion_raw", data)

    def test_native_target_partition_reordering_keeps_motion_aligned(self):
        data = full_scene()
        first = convert_full_scene(data, "full.pt")
        permutation = np.array([0, 2, 1])
        target = dict(first, agent_states=first["agent_states"][permutation], agent_types=first["agent_types"][permutation])
        actual = convert_full_scene(data, "full.pt", target=target)
        np.testing.assert_array_equal(actual["agent_motion_raw"], first["agent_motion_raw"][permutation])
        self.assertEqual(actual["agent_motion_is_static"].tolist(), [False, True, False])

    def test_state_matching_rejects_missing_or_ambiguous_sources(self):
        scene = convert_full_scene(full_scene(), "full.pt")
        changed = dict(scene, agent_states=scene["agent_states"].copy())
        changed["agent_states"][1, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "unique exact source matches"):
            strict_state_order(changed, scene)
        duplicates = dict(scene, agent_states=np.repeat(scene["agent_states"][:1], 3, axis=0),
                          agent_types=np.repeat(scene["agent_types"][:1], 3, axis=0))
        with self.assertRaisesRegex(ValueError, "unique exact source matches"):
            strict_state_order(duplicates, duplicates)

    def test_current_invalid_motion_and_tokenized_only_fallback_fail(self):
        data = full_scene()
        data["agent"]["valid_mask"][0, 9] = False
        with self.assertRaisesRegex(ValueError, "valid real source observation"):
            convert_full_scene(data, "full.pt")
        with self.assertRaisesRegex(ValueError, "saved scenario_dreamer metadata"):
            convert_full_scene({"tokenized_agent": {}, "tokenized_map": {}}, "tokenized.pt")

    def test_missing_source_has_actionable_error_without_importing_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            request = parse_source_name("training.tfrecord-00000-of-01000_0_0_9.pkl")
            with self.assertRaisesRegex(FileNotFoundError, "Initial/tokenized-only caches cannot supply"):
                WaymoSourceReader(directory).get(request)

    def test_prepare_retains_manifest_order_basenames_and_original_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source/test", root / "output/vae/test"
            source.mkdir(parents=True)
            names = ["b.pt", "a.pt"]
            for index, name in enumerate(names):
                data = full_scene()
                data["scenario_dreamer_cache_file"] = f"testing.tfrecord-00000-of-00150_{index + 1}_0_9.pkl"
                torch.save(data, source / name)
            original = {name: (source / name).read_bytes() for name in names}
            sample_list = root / "selected.pkl"
            with sample_list.open("wb") as handle:
                pickle.dump({"files": names}, handle)
            report = prepare(input_dir=source, output_dir=output, split="test", sample_list=sample_list)
            with Path(report["sample_list"]).open("rb") as handle:
                selected = pickle.load(handle)["files"]
            self.assertEqual(selected, ["testing.tfrecord-00000-of-00150_1_0_9.pkl",
                                        "testing.tfrecord-00000-of-00150_2_0_9.pkl"])
            for name in names:
                self.assertEqual((source / name).read_bytes(), original[name])
            for name in selected:
                with (output / name).open("rb") as handle:
                    result = pickle.load(handle)
                self.assertEqual(result["agent_motion_raw"].shape, (3, 12))
                self.assertTrue(result["agent_motion_valid_mask"].all())
            self.assertEqual(select_inputs(source, sample_list, 1), [source / "b.pt"])
            previous = Path(report["manifest"]).read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(input_dir=source, output_dir=output, split="test", sample_list=sample_list)
            self.assertEqual(Path(report["manifest"]).read_bytes(), previous)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_raw_scene_source_rows_follow_unique_native_state_permutation(self):
        from src.prepare_vectorworld_data import convert_raw_scene
        from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
        data = full_scene()
        reference = convert_full_scene(data, "full.pt")
        order = np.array([0, 2, 1])
        target = dict(reference, agent_states=reference["agent_states"][order], agent_types=reference["agent_types"][order])
        info = dict(data["scenario_dreamer"], source_index=np.array([0, 1, 2]))
        positions = data["agent"]["position"][[1, 2, 0]].numpy()
        vel = data["agent"]["velocity"][[1, 2, 0]].numpy()
        states = np.zeros((3, 10, 9))
        states[..., :2], states[..., 7:9] = positions, vel
        tracks = dict(states=states, valid=np.ones((3, 10), dtype=bool))
        request = parse_source_name("testing.tfrecord-00000-of-00150_2_0_9.pkl")
        with patch("scenario_dreamer_filter.decode_tracks_from_proto", return_value=tracks), \
                patch("scenario_dreamer_filter.get_agent_features", return_value=({}, info)):
            actual = convert_raw_scene(target, "scene.pkl", SimpleNamespace(current_time_index=10, scenario_id="source"), request)
        np.testing.assert_array_equal(actual["agent_motion_raw"], reference["agent_motion_raw"][order])
        self.assertEqual(actual["vectorworld_motion_metadata"]["source_agent_indices"], order.tolist())
        adapt_preprocessed_scene(actual, "scene.pkl")


if __name__ == "__main__":
    unittest.main()
