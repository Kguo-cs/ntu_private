"""CPU regression checks for main-search versus optional artifact inventory.

All fixtures live in temporary directories. No real search state, checkpoint,
training protocol, GPU, or final report is changed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_frozen_map_final_artifacts as audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class OptionalInventoryRegression(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="frozen_map_inventory_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.patch = patch.object(audit, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.make_main()

    def make_main(self):
        stage = self.runs / "phase1"
        arm = stage / "new_control"
        write_json(stage / "protocol.json", {"initialization_scope": "frozen_pretrained_map_fresh_G1", "steps": 2})
        values = {key: .1 for key in audit.METRICS}
        row = dict(values, variant="new_control", budget=2, batch_size=8, seed=817, eval_seed=2817,
                   evaluation_weights="ema", evaluation_only=False, map_frozen=True,
                   initialization_scope="frozen_pretrained_map_fresh_G1", seen_training_examples=16,
                   num_samples=4, num_generated_vehicles=100, num_gt_vehicles=100,
                   num_generated_onroad_vehicles=80, num_gt_onroad_vehicles=80,
                   generated_onroad_fraction=.8, gt_onroad_fraction=.8,
                   train_seconds=1., eval_seconds=.25, eval_manifest_sha256="fixture_manifest",
                   checkpoint=str(arm / "last.ckpt"), metrics_path=str(arm / "metrics.json"))
        write_json(stage / "summary.json", [row])
        write_json(arm / "metrics.json", {"num_samples": 4, "num_gt_samples": 4,
                                         "reference_mode": "matched", "agent_metrics": values})
        write_json(arm / "initialization.json", {"seed": 817, "resume_global_step": 0,
            "batch_size": 8, "optimizer_continued": False, "calibration_sha256": "fixture_calibration"})
        config = {"model": {"model_config": {"decoder": {}, "optimizer": "default", "lr": .0005,
                   "lr_warmup_steps": 0, "lr_total_steps": 128, "lr_min_ratio": .05}},
                  "trainer": {"gradient_clip_val": .5}}
        (arm / "config.yaml").write_text(yaml.safe_dump(config))
        write_json(arm / "train_trace.json", [{"step": step, "map_grad_norm_before_clip": 0.} for step in range(2)])
        log = self.root / "logs/component_loss_frozen_map_phase1.log"
        log.parent.mkdir()
        log.write_text("DONE " + str(stage) + "\n")

    def inventory(self):
        return audit.collect_inventory(self.runs, 100)

    def test_optional_same_scope_never_changes_main_counts_or_cost(self):
        baseline = self.inventory()[0]
        optional = self.runs / "map_policy_comparison_current"
        write_json(optional / "protocol.json", {"initialization_scope": "frozen_pretrained_map_fresh_G1"})
        # Deliberately no main-stage log/config/trace/report: optional diagnostics
        # must not be subjected to the main search's completion requirements.
        write_json(optional / "summary.json", [{"draft_diagnostic": True}])
        counts, stages, auxiliary, rows, _ = self.inventory()
        for key, value in baseline.items():
            if not key.startswith("excluded_optional_"):
                self.assertEqual(counts[key], value, key)
        self.assertEqual(len(stages), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts["excluded_optional_protocol_roots"], 1)
        self.assertEqual(counts["excluded_optional_reported_metric_observations"], 1)
        self.assertEqual(auxiliary[0]["exclusion_kind"], "optional_map_policy_diagnostic")
        self.assertTrue(auxiliary[0]["excluded_from_main_metrics_training_recipes_and_cost"])

    def test_nested_optional_protocol_uses_ancestor_exclusion(self):
        optional = self.runs / "map_policy_comparison_current" / "nested_evaluator"
        write_json(optional / "protocol.json", {"initialization_scope": "frozen_pretrained_map_fresh_G1"})
        write_json(optional / "summary.json", [{"draft": 1}, {"draft": 2}])
        counts, _, auxiliary, _, _ = self.inventory()
        self.assertEqual(counts["main_metric_observations"], 1)
        self.assertEqual(counts["excluded_optional_reported_metric_observations"], 2)
        self.assertEqual(auxiliary[0]["optional_root"], "map_policy_comparison_current")

    def test_unfinished_optional_protocol_does_not_block_final_audit(self):
        optional = self.runs / "map_policy_comparison_pending"
        optional.mkdir()
        (optional / "protocol.json").write_text('{"unfinished":')
        counts, _, auxiliary, _, _ = self.inventory()
        self.assertEqual(counts["main_metric_observations"], 1)
        self.assertEqual(auxiliary[0]["status"], "no_summary")

    def test_unreadable_optional_summary_is_explicitly_excluded(self):
        optional = self.runs / "map_policy_comparison_pending"
        write_json(optional / "protocol.json", {})
        (optional / "summary.json").write_text('{"unfinished":')
        counts, _, auxiliary, _, hashes = self.inventory()
        self.assertEqual(counts["excluded_optional_reported_metric_observations"], 0)
        self.assertEqual(auxiliary[0]["status"], "unreadable_summary_not_audited_as_search")
        self.assertIn(str((optional / "summary.json").resolve()), hashes)

    def test_incomplete_main_stage_still_blocks_final_audit(self):
        write_json(self.runs / "phase2" / "protocol.json", {"initialization_scope": "frozen_pretrained_map_fresh_G1"})
        with self.assertRaisesRegex(ValueError, "main stage is incomplete"):
            self.inventory()


if __name__ == "__main__":
    unittest.main(verbosity=2)
