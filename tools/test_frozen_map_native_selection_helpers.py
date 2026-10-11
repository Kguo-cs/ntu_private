"""CPU-only regression checks for native final-selection audit helpers.

Fixtures live only in memory: no final_selection.json, checkpoint or best
experiment config is created, and no training or GPU API is invoked.
"""
from __future__ import annotations

import copy
import unittest

from tools import audit_frozen_map_final_artifacts as audit
from tools import materialize_frozen_map_best as materialize


def fixture(*, candidate_ratio=.9, candidate_batch=256, changes=None):
    candidate = "synthetic_candidate"
    frozen = {
        "arms": [{"name": "new_control", "batch": 512},
                 {"name": candidate, "batch": candidate_batch}],
        "train_seeds": list(audit.TRAIN_SEEDS),
        "generation_seeds": list(audit.GEN_SEEDS),
        "full50000_no_reselection": True,
    }
    rows = []
    for name, batch in (("new_control", 512), (candidate, candidate_batch)):
        for seed_i, seed in enumerate(audit.TRAIN_SEEDS):
            for gen_i, gen in enumerate(audit.GEN_SEEDS):
                scale = (1. + .1 * seed_i) * (1. + .02 * gen_i)
                row = {
                    "variant": name, "seed": seed, "eval_seed": gen,
                    "batch_size": batch, "budget": 8192 if batch == 256 else 4096,
                    "checkpoint": f"/tmp/IN_MEMORY_ONLY/{name}/{seed}.ckpt",
                    "eval_manifest_sha256": "synthetic_helper_fixture_only",
                    "evaluation_weights": "ema", "num_samples": 4096,
                    "generated_onroad_fraction": .6,
                    "num_generated_vehicles": 100, "num_gt_vehicles": 100,
                }
                row.update({key: (i + 1) * scale * (candidate_ratio if name == candidate else 1.)
                            for i, key in enumerate(audit.METRICS)})
                if name == candidate:
                    row.update(changes or {})
                rows.append(row)
    means = {name: materialize.aggregate([row for row in rows if row["variant"] == name])
             for name in ("new_control", candidate)}
    rankings = [{"variant": name, "R": materialize.compare(values, means["new_control"])[0],
                 "eligible": materialize.compare(values, means["new_control"])[1]}
                for name, values in means.items()]
    improving = sorted((row for row in rankings if row["eligible"] and row["R"] < 1.),
                       key=lambda row: (row["R"], row["variant"]))
    selected = improving[0]["variant"] if improving else "new_control"
    decision = {"selected": selected, "selected_arm": next(arm for arm in frozen["arms"]
                                                            if arm["name"] == selected),
                "rows": rows, "all": rankings}
    return decision, frozen


class NativeSelectionHelpers(unittest.TestCase):
    def check_both(self, decision, frozen):
        selected, _, _, scores = materialize.validate_decision(decision, frozen)
        independent = audit.audit_final_matrix(decision, frozen, decision["rows"])
        self.assertEqual(selected, independent["selected"])
        for name, (score, eligible) in scores.items():
            self.assertAlmostEqual(score, independent["relative_R_and_gates"][name]["R"])
            self.assertEqual(eligible, independent["relative_R_and_gates"][name]["eligible"])
        return independent

    def test_equal_exposure_256_8192_and_512_4096(self):
        self.assertEqual(audit.exposure(8192, 256, 493542), 2097048)
        self.assertEqual(audit.exposure(4096, 512, 493542), 2097048)

    def test_cross_batch_matrix_and_seed_roles(self):
        decision, frozen = fixture()
        result = self.check_both(decision, frozen)
        self.assertEqual(result["selected"], "synthetic_candidate")
        self.assertEqual(result["actual_cells"], 12)
        self.assertEqual(result["actual_independent_training_initializations"], 3)
        self.assertEqual(result["fully_post_freeze_confirmation_training_N"], 2)
        self.assertEqual(result["post_freeze_training_seeds"], [1817, 2817])
        self.assertTrue(result["post_freeze_only_is_diagnostic_no_reselection"])
        self.assertAlmostEqual(result["paired_R_sample_stats_all3"]["mean"], .9)

    def test_same_batch_matrix_supported(self):
        decision, frozen = fixture(candidate_batch=512)
        self.check_both(decision, frozen)

    def test_no_improvement_retains_control(self):
        decision, frozen = fixture(candidate_ratio=1.01)
        self.assertEqual(self.check_both(decision, frozen)["selected"], "new_control")

    def test_onroad_gate_rejects_smaller_R(self):
        decision, frozen = fixture(changes={"generated_onroad_fraction": .579})
        self.assertEqual(self.check_both(decision, frozen)["selected"], "new_control")

    def test_vehicle_count_gate_rejects_smaller_R(self):
        decision, frozen = fixture(changes={"num_generated_vehicles": 111})
        self.assertEqual(self.check_both(decision, frozen)["selected"], "new_control")

    def test_shape_gate_rejects_smaller_R(self):
        decision, frozen = fixture(changes={"length_jsd": 10.})
        self.assertEqual(self.check_both(decision, frozen)["selected"], "new_control")

    def test_incomplete_gen_pair_rejected(self):
        decision, frozen = fixture()
        decision["rows"].pop()
        for check in (lambda: materialize.validate_decision(decision, frozen),
                      lambda: audit.audit_final_matrix(decision, frozen, decision["rows"])):
            with self.assertRaises(ValueError):
                check()

    def test_wrong_B256_endpoint_rejected(self):
        decision, frozen = fixture()
        for row in decision["rows"]:
            if row["batch_size"] == 256:
                row["budget"] = 4096
        for check in (lambda: materialize.validate_decision(decision, frozen),
                      lambda: audit.audit_final_matrix(decision, frozen, decision["rows"])):
            with self.assertRaises(ValueError):
                check()

    def test_row_order_does_not_change_equal_train_aggregation(self):
        decision, frozen = fixture()
        expected = self.check_both(decision, frozen)
        reordered = copy.deepcopy(decision)
        reordered["rows"].reverse()
        result = self.check_both(reordered, frozen)
        self.assertEqual(result["arm_means_native"], expected["arm_means_native"])


if __name__ == "__main__":
    unittest.main()
