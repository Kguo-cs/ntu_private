"""Agent metrics must use generated lane geometry and predicted topology."""
import copy
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from src.smart.metrics.generated_map import split_generated_maps
from src.smart.metrics.gen_metrics import ScenarioDreamerEvaluator, make_generated_scene
from src.smart.metrics.metric_core import DistributionAccumulator
from src.smart.metrics.official_backend import load_official_backend
from src.smart.metrics.real_cache import prepare_real_cache


def raw_scene(x=0.):
    points = np.stack((np.full(20, x), np.linspace(-10., 20., 20)), -1)[None]
    return dict(lg_type=0, scene_timestep=37, num_lanes=1, num_agents=2,
                road_points=points, edge_index_lane_to_lane=np.array([[0], [0]]),
                road_connection_types=np.eye(6)[[5]],
                agent_states=np.array([[.25, 0., 3., 0., 1., 4., 2.], [.5, 6., 8., 0., 1., 4.5, 2.]]),
                agent_types=np.eye(3)[[0, 0]])


def outputs():
    states = np.concatenate([raw_scene()["agent_states"]] * 2)
    out = dict(traj=states[:, None, None, :2], head=np.full((4, 1, 1), np.pi / 2),
               size=states[:, None, None, 5:7], vel=np.stack((np.zeros(4), states[:, 2]), -1)[:, None])
    out["generated_map"] = dict(
        coordinate_frame="sd_local", road_points=np.concatenate([raw_scene(1.)["road_points"], raw_scene(.75)["road_points"]]),
        road_connection_types=np.eye(6)[[5, 5]], edge_index_lane_to_lane=np.array([[0, 1], [0, 1]]),
        batch=np.array([0, 1]), lg_type=np.array([0, 0]))
    return out


class GeneratedMapTest(unittest.TestCase):
    def setUp(self):
        self.backend = load_official_backend()

    def test_smart_rollout_and_evaluator_use_generated_agent_counts(self):
        from types import SimpleNamespace
        import torch
        from src.smart.model.smart import SMART

        expected = outputs()
        expected = {k: ({mk: torch.as_tensor(mv) if not isinstance(mv, str) else mv
                         for mk, mv in value.items()} if isinstance(value, dict)
                        else torch.as_tensor(value)) for k, value in expected.items()}
        original_agent = {"batch": torch.tensor([0, 1]), "type": torch.zeros(2, dtype=torch.long)}
        generated_batch = torch.tensor([0, 0, 1, 1])

        def inference(agent):
            agent["batch"] = generated_batch.clone()
            agent["type"] = torch.zeros(4, dtype=torch.long)
            agent["generated_map"] = expected["generated_map"]
            return {"pred_traj_10hz": expected["traj"][:, 0],
                    "pred_head_10hz": expected["head"][:, 0],
                    "pred_z_10hz": torch.zeros(4, 1),
                    "shape": expected["size"][:, 0, 0],
                    "initial_local_vel": expected["vel"][:, 0]}

        model = SimpleNamespace(encoder=SimpleNamespace(init_decoder_name="scenario_dreamer", inference=inference),
                                scenario_dreamer_init=True, n_rollout_closed_val=1,
                                n_vis_batch=0, challenge_type=None, use_sd_evaluator=True)
        out = SMART._rollouts(model, {}, original_agent, None)
        torch.testing.assert_close(out["generated_batch"], generated_batch)
        torch.testing.assert_close(original_agent["batch"], torch.tensor([0, 1]))
        self.assertEqual(out["traj"].shape[0], 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["a.pkl", "b.pkl"]
            for name in files:
                with (root / name).open("wb") as handle:
                    pickle.dump(raw_scene(), handle)
            manifest = root / "eval.pkl"
            with manifest.open("wb") as handle:
                pickle.dump({"files": files}, handle)
            database = root / "real.sqlite"
            prepare_real_cache(root, manifest, database, expected_scenes=2, progress=False)
            model.sd_evaluator = ScenarioDreamerEvaluator(
                real_cache=database, eval_set=manifest, expected_scenes=2, gen_timestep=0,
                prediction_frame="sd_local", require_full_set=True,
                reference_mode="full", map_source="generated")
            model._rollouts = lambda *args: out
            records = [dict(scenario_dreamer_cache_file=name, scene_timestep=37,
                            generation_scene_timestep=37) for name in files]
            SMART._validate_closed_loop(model, records, {}, original_agent, 0)
            self.assertEqual(model.sd_evaluator.report()["num_generated_vehicles"], 4)
            self.assertEqual(len(model.sd_evaluator.compute()), 7)

    def test_geometry_replaces_gt_onroad_and_deviation_reference(self):
        out = outputs()
        batch, types = np.array([0, 0, 1, 1]), np.zeros(4, dtype=np.int64)
        gt = self.backend.convert_data_to_unified_format(raw_scene(100.), "waymo_gt")
        gt["metric_lanes"] = np.full((1, 100, 2), 100.)
        maps = split_generated_maps(out["generated_map"], 2)
        generated, _, _ = make_generated_scene(out, batch, types, 0, 0, {}, gt,
                                               prediction_frame="sd_local", generated_map=maps[0])
        self.assertNotIn("metric_lanes", generated)
        np.testing.assert_array_equal(generated["lanes"], maps[0]["road_points"])
        generated_acc, reference_acc = DistributionAccumulator(), DistributionAccumulator()
        generated_acc.update(generated, self.backend, collision=True)
        reference_acc.update(dict(gt, vehicles=generated["vehicles"]), self.backend)
        self.assertEqual(generated_acc.totals[1:3].tolist(), [2, 2])
        self.assertEqual(reference_acc.totals[1:3].tolist(), [0, 0])

    def test_world_agent_predictions_align_with_sd_local_generated_lanes(self):
        local = outputs()
        world = copy.deepcopy(local)
        center = np.array([123., -456.])
        # Decoder maps SD -> world with rotation -pi/2; metrics must invert it.
        xy = local["traj"].copy()
        world["traj"][..., 0] = xy[..., 1] + center[0]
        world["traj"][..., 1] = -xy[..., 0] + center[1]
        world["head"] -= np.pi / 2
        world["vel"] = local["vel"][..., ::-1].copy()
        batch, types = np.array([0, 0, 1, 1]), np.zeros(4, dtype=np.int64)
        lane_map = split_generated_maps(local["generated_map"], 2)[0]
        expected, _, _ = make_generated_scene(local, batch, types, 0, 0, {}, {},
                                              prediction_frame="sd_local", generated_map=lane_map)
        actual, _, _ = make_generated_scene(world, batch, types, 0, 0,
                                            dict(sd_center_world=center, sd_rotation_angle=np.pi / 2), {},
                                            prediction_frame="world", generated_map=lane_map)
        np.testing.assert_allclose(actual["vehicles"], expected["vehicles"], atol=1e-12)
        np.testing.assert_array_equal(actual["lanes"], expected["lanes"])

    def test_predicted_predecessor_edges_control_lane_compaction(self):
        source = raw_scene()
        lower, upper = source["road_points"][0].copy(), source["road_points"][0].copy()
        lower[:, 1], upper[:, 1] = np.linspace(-10, 0, 20), np.linspace(0, 10, 20)
        source.update(num_lanes=2, road_points=np.stack((lower, upper)),
                      edge_index_lane_to_lane=np.array([[1, 0, 1, 0], [1, 1, 0, 0]]),
                      road_connection_types=np.eye(6)[[5, 1, 2, 5]])
        connected = self.backend.convert_data_to_unified_format(source, "waymo")
        self.assertEqual(len(connected["lanes"]), 1)
        source["road_connection_types"] = np.eye(6)[[5, 0, 0, 5]]
        disconnected = self.backend.convert_data_to_unified_format(source, "waymo")
        self.assertEqual(len(disconnected["lanes"]), 2)

    def test_graph_split_handles_interleaved_nodes_and_rejects_cross_scene_edges(self):
        payload = outputs()["generated_map"]
        payload["road_points"] = np.concatenate([payload["road_points"], payload["road_points"][:1]])
        payload["batch"] = np.array([0, 1, 0])
        payload["edge_index_lane_to_lane"] = np.array([[0, 2, 0, 2, 1], [0, 0, 2, 2, 1]])
        payload["road_connection_types"] = np.eye(6)[[5, 2, 1, 5, 5]]
        records = split_generated_maps(payload, 2)
        np.testing.assert_array_equal(records[0]["edge_index_lane_to_lane"], [[0, 1, 0, 1], [0, 0, 1, 1]])
        np.testing.assert_array_equal(records[1]["edge_index_lane_to_lane"], [[0], [0]])
        payload["edge_index_lane_to_lane"][1, 0] = 1
        with self.assertRaisesRegex(ValueError, "cross scene"):
            split_generated_maps(payload, 2)

    def test_generated_map_contract_rejects_wrong_frame_or_connections(self):
        payload = outputs()["generated_map"]
        payload["coordinate_frame"] = "world"
        with self.assertRaisesRegex(ValueError, "SD-local"):
            split_generated_maps(payload, 2)
        payload["coordinate_frame"] = "sd_local"
        payload["road_connection_types"][0] = 0
        with self.assertRaisesRegex(ValueError, "one-hot"):
            split_generated_maps(payload, 2)

    def test_evaluator_matches_official_metrics_and_exports_generated_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["a.pkl", "b.pkl"]
            for name in files:
                with (root / name).open("wb") as handle:
                    pickle.dump(raw_scene(), handle)
            manifest = root / "eval.pickle"
            with manifest.open("wb") as handle:
                pickle.dump({"files": files}, handle)
            database = root / "real.sqlite"
            prepare_real_cache(root, manifest, database, expected_scenes=2, progress=False)
            kwargs = dict(real_cache=database, eval_set=manifest, expected_scenes=2, gen_timestep=0,
                          prediction_frame="sd_local", require_full_set=True, reference_mode="full")
            evaluator = ScenarioDreamerEvaluator(**kwargs, map_source="generated", export_dir=root / "export")
            records = [dict(scenario_dreamer_cache_file=name, scene_timestep=37, generation_scene_timestep=37) for name in files]
            agent = dict(batch=np.array([0, 0, 1, 1]), type=np.zeros(4, dtype=np.int64))
            out = outputs()
            missing = copy.deepcopy(out); del missing["generated_map"]
            with self.assertRaisesRegex(ValueError, "refusing to evaluate against GT"):
                evaluator.update(records, agent, missing)
            reference_evaluator = ScenarioDreamerEvaluator(**kwargs)
            with self.assertRaisesRegex(ValueError, "reference-map evaluator"):
                reference_evaluator.update(records, agent, out)
            evaluator.update(records, agent, out)
            result = evaluator.compute()
            generated = []
            for index, lane_map in enumerate(split_generated_maps(out["generated_map"], 2)):
                with (root / "export" / f"{index:05d}.pkl").open("rb") as handle:
                    exported = pickle.load(handle)
                np.testing.assert_array_equal(exported["road_points"], lane_map["road_points"])
                np.testing.assert_array_equal(exported["road_connection_types"], lane_map["road_connection_types"])
                generated.append(self.backend.convert_data_to_unified_format(exported, "waymo"))
            real = [self.backend.convert_data_to_unified_format(raw_scene(), "waymo_gt")] * 2
            expected = self.backend.compute_agent_metrics(generated, real)
            for key in expected:
                self.assertAlmostEqual(result[key], expected[key], places=10)
            report = evaluator.report()
            self.assertEqual(report["map_source"], "generated")
            self.assertTrue(report["full_membership"])


if __name__ == "__main__":
    unittest.main()
