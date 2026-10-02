"""Full-lane conditioning evaluates agents on their identified reference maps."""
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.smart.metrics.gen_metrics import ScenarioDreamerEvaluator
from src.smart.metrics.official_backend import load_official_backend
from src.smart.metrics.real_cache import prepare_real_cache
from src.smart.model.smart import SMART


def full_scene(x):
    return dict(
        lg_type=0, scene_timestep=37, num_lanes=1, num_agents=2,
        road_points=np.stack((np.full(20, x), np.linspace(-10., 20., 20)), -1)[None],
        edge_index_lane_to_lane=np.array([[0], [0]]),
        road_connection_types=np.eye(6)[[5]],
        agent_states=np.array([[x + .25, 0., 3., 0., 1., 4., 2.],
                               [x + .5, 6., 8., 0., 1., 4.5, 2.]]),
        agent_types=np.eye(3)[[0, 0]],
    )


class LaneConditionedMetricTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.files = ["a.pkl", "b.pkl"]
        self.raw = [full_scene(0.), full_scene(10.)]
        for name, scene in zip(self.files, self.raw):
            with (self.root / name).open("wb") as handle:
                pickle.dump(scene, handle)
        self.manifest = self.root / "eval.pkl"
        with self.manifest.open("wb") as handle:
            pickle.dump({"files": self.files}, handle)
        self.database = self.root / "real.sqlite"
        prepare_real_cache(self.root, self.manifest, self.database,
                           expected_scenes=2, progress=False)
        self.records = [dict(scenario_dreamer_cache_file=name, scene_timestep=37,
                             generation_scene_timestep=37) for name in self.files]
        self.states = np.concatenate([scene["agent_states"] for scene in self.raw])
        self.types = torch.tensor([0, 1, 0, 0])
        self.agent = dict(batch=torch.tensor([0, 0, 1, 1]), type=torch.ones(4, dtype=torch.long))
        self.out = dict(
            traj=self.states[:, None, None, :2], head=np.full((4, 1, 1), np.pi / 2),
            size=self.states[:, None, None, 5:7],
            vel=np.stack((np.zeros(4), self.states[:, 2]), -1)[:, None],
        )

    def evaluator(self):
        evaluator = ScenarioDreamerEvaluator(
            real_cache=self.database, eval_set=self.manifest, expected_scenes=2,
            gen_timestep=0, prediction_frame="sd_local", require_full_set=True,
            reference_mode="full", map_source="reference", export_dir=self.root / "export")
        self.addCleanup(evaluator.store.close)
        return evaluator

    def test_smart_reference_map_route_uses_generated_types_and_exports_input_lanes(self):
        expected = {key: torch.as_tensor(value) for key, value in self.out.items()}

        def inference(agent):
            agent["type"] = self.types.clone()
            return dict(pred_traj_10hz=expected["traj"][:, 0],
                        pred_head_10hz=expected["head"][:, 0],
                        pred_z_10hz=torch.zeros(4, 1), shape=expected["size"][:, 0, 0],
                        initial_local_vel=expected["vel"][:, 0])

        model = SimpleNamespace(
            encoder=SimpleNamespace(init_decoder_name="scenario_dreamer", inference=inference),
            scenario_dreamer_init=True, n_rollout_closed_val=1, n_vis_batch=0,
            challenge_type=None, use_sd_evaluator=True, sd_evaluator=self.evaluator())
        out = SMART._rollouts(model, {}, self.agent, self.records)
        self.assertNotIn("generated_map", out)
        torch.testing.assert_close(self.agent["type"], torch.ones(4, dtype=torch.long))
        model._rollouts = lambda *args: out
        SMART._validate_closed_loop(model, self.records, {}, self.agent, 0)

        backend = load_official_backend()
        generated = []
        for index, raw in enumerate(self.raw):
            with (self.root / "export" / f"{index:05d}.pkl").open("rb") as handle:
                exported = pickle.load(handle)
            self.assertEqual(exported["lg_type"], 0)
            for field in ("road_points", "road_connection_types", "edge_index_lane_to_lane"):
                np.testing.assert_array_equal(exported[field], raw[field])
            expected_types = self.types[index * 2:index * 2 + 2].numpy()
            np.testing.assert_array_equal(exported["agent_types"].argmax(-1), expected_types)
            generated.append(backend.convert_data_to_unified_format(exported, "waymo"))
        real = [backend.convert_data_to_unified_format(raw, "waymo_gt") for raw in self.raw]
        expected_metrics = backend.compute_agent_metrics(generated, real)
        for name, value in model.sd_evaluator.compute().items():
            self.assertAlmostEqual(value, expected_metrics[name], places=10)
        report = model.sd_evaluator.report()
        self.assertEqual(report["map_source"], "reference")
        self.assertEqual(report["num_generated_vehicles"], 3)
        self.assertTrue(report["full_membership"])

    def test_full_set_and_frame_checks_still_apply_to_reference_maps(self):
        evaluator = self.evaluator()
        agent = dict(batch=np.zeros(2, dtype=np.int64), type=np.zeros(2, dtype=np.int64))
        out = {key: value[:2] for key, value in self.out.items()}
        evaluator.update(self.records[:1], agent, out)
        with self.assertRaisesRegex(ValueError, "Incomplete official evaluation: 1/2"):
            evaluator.compute()
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation sample"):
            evaluator.update(self.records[:1], agent, out)
        evaluator.reset()
        wrong_frame = [dict(self.records[0], generation_scene_timestep=38)]
        with self.assertRaisesRegex(ValueError, "Model/reference frame mismatch"):
            evaluator.update(wrong_frame, agent, out)
        self.assertEqual(evaluator.report()["num_samples"], 0)

    def test_reference_mode_rejects_generated_map_payload(self):
        evaluator = self.evaluator()
        with self.assertRaisesRegex(ValueError, "reference-map evaluator"):
            evaluator.update(self.records, self.agent, dict(self.out, generated_map={}))
        self.assertEqual(evaluator.report()["num_samples"], 0)


if __name__ == "__main__":
    unittest.main()
