"""CPU checks for opt-in SMART lane-centerline query conditioning."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.smart.modules.map_decoder import SMARTMapDecoder, initial_map_output_mask


def make_model(tp, layers=1):
    return SMARTMapDecoder(32, 40., 4, layers, 4, 8, 0., 8, tp).eval()


def fixture():
    tp = SimpleNamespace(
        init_map_crop="square", init_map_half_extent=50., init_map_range=100.,
        map_token_traj_src=torch.arange(4 * 22).reshape(4, 22).float() / 20.,
    )
    tokens = {
        "type": torch.arange(10), "batch": torch.zeros(10, dtype=torch.long),
        "position": torch.stack((torch.arange(10).float(), torch.ones(10)), -1),
        "orientation": torch.zeros(10), "token_idx": torch.arange(10) % 4,
        "light_type": torch.zeros(10, dtype=torch.long),
    }
    agent = {
        "initial_pos": torch.zeros(1, 2), "initial_heading": torch.zeros(1),
        "ego_mask": torch.ones(1, dtype=torch.bool),
    }
    return tp, tokens, agent


class MapConditionTests(unittest.TestCase):
    def test_only_driving_centerlines_and_road_edges_are_returned(self):
        types = torch.arange(10)
        self.assertEqual(torch.where(initial_map_output_mask(types, "road_edges"))[0].tolist(), [4, 5])
        self.assertEqual(torch.where(initial_map_output_mask(types, "lanes_and_edges"))[0].tolist(), [0, 1, 2, 4, 5])

    def test_invalid_output_policy_fails_early(self):
        with self.assertRaisesRegex(ValueError, "init_map_output_types"):
            initial_map_output_mask(torch.arange(10), "all")

    def test_default_with_or_without_flag_has_exact_seeded_state_and_output(self):
        tp, tokens, agent = fixture()
        torch.manual_seed(917)
        old_default = make_model(tp)
        explicit_tp = copy.copy(tp)
        explicit_tp.init_map_output_types = "road_edges"
        torch.manual_seed(917)
        explicit_default = make_model(explicit_tp)
        for key, tensor in old_default.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit_default.state_dict()[key]), key)
        with torch.no_grad():
            a, b = old_default(tokens, agent), explicit_default(tokens, agent)
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_query_policy_keeps_source_geometry_and_weights(self):
        tp, tokens, agent = fixture()
        torch.manual_seed(917)
        baseline = make_model(tp)
        opt_tp = copy.copy(tp)
        opt_tp.init_map_output_types = "lanes_and_edges"
        torch.manual_seed(917)
        option = make_model(opt_tp)
        self.assertEqual(set(baseline.state_dict()), set(option.state_dict()))
        for key, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, option.state_dict()[key]), key)
        original = {k: v.clone() for k, v in tokens.items()}
        with patch.object(option.edge_encoder, "build_map2map_edge", wraps=option.edge_encoder.build_map2map_edge) as edge:
            with torch.no_grad():
                result = option(tokens, agent)
        arguments = edge.call_args.args
        self.assertEqual(len(arguments[0]), 10)  # every cropped source remains
        self.assertEqual(len(arguments[2]), 5)  # only query nodes change
        self.assertEqual(arguments[-2:], (40., 8))
        selected = [0, 1, 2, 4, 5]
        self.assertTrue(torch.equal(result["position"], tokens["position"][selected]))
        for key in original:
            self.assertTrue(torch.equal(original[key], tokens[key]), key)

    def test_new_policy_works_without_attention_and_empty_lanes(self):
        tp, tokens, agent = fixture()
        tp.init_map_output_types = "lanes_and_edges"
        output = make_model(tp, layers=0)(tokens, agent)
        self.assertEqual(len(output["position"]), 5)
        road_only = {key: value[3:6] for key, value in tokens.items()}
        output = make_model(tp, layers=0)(road_only, agent)
        self.assertTrue(torch.equal(output["position"], tokens["position"][[4, 5]]))


if __name__ == "__main__":
    unittest.main()
