"""Initial SMART map squares crop attention sources in the ego coordinate frame."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.model.smart import SMART
from src.smart.modules.map_decoder import SMARTMapDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_global
from src.smart.utils.map_crop import square_map_mask, validate_init_map_crop
import test_init_diffusion_sep_map as fixtures


class InitDiffusionMapCropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(391)

    @staticmethod
    def processor(*, crop="square", half_extent=32.):
        return SimpleNamespace(
            pred_init=True, learn_init=True, init_map_range=32.,
            init_map_crop=crop, init_map_half_extent=half_extent,
            map_token_traj_src=torch.linspace(-1., 1., 66).reshape(3, 22),
        )

    def map_encoder(self, *, layers=1, crop="square", half_extent=32.):
        return SMARTMapDecoder(
            hidden_dim=8, pl2pl_radius=15., num_freq_bands=4,
            num_layers=layers, num_heads=2, head_dim=4, dropout=0.,
            pt2pt_neighbor=8,
            token_processor=self.processor(crop=crop, half_extent=half_extent),
        ).eval()

    def decoder(self, *, sep_map=False, layers=1, crop="square"):
        return fixtures.InitDiffusionSeparateMapTest().decoder(
            sep_map=sep_map, num_map_layers=layers,
            token_processor=self.processor(crop=crop),
        ).eval()

    @staticmethod
    def origin_agent():
        return dict(
            num_graphs=1, batch=torch.tensor([0]), ego_mask=torch.tensor([True]),
            initial_pos=torch.zeros(1, 2), initial_heading=torch.zeros(1),
        )

    @staticmethod
    def tokens(position, *, types=None, batch=None, orientation=None):
        position = torch.as_tensor(position, dtype=torch.float32).reshape(-1, 2)
        count = len(position)
        return dict(
            position=position,
            type=torch.full((count,), 4, dtype=torch.long) if types is None else torch.as_tensor(types),
            batch=torch.zeros(count, dtype=torch.long) if batch is None else torch.as_tensor(batch),
            orientation=torch.linspace(-.3, .4, count) if orientation is None else torch.as_tensor(orientation),
            token_idx=torch.arange(count).remainder(3),
            light_type=torch.zeros(count, dtype=torch.long),
        )

    @staticmethod
    def select(tokens, keep):
        return {name: value[keep] for name, value in tokens.items()}

    @staticmethod
    def world_tokens():
        agent = fixtures.InitDiffusionSeparateMapTest.agents()
        agent["initial_heading"][0] = math.pi / 4.
        batch = torch.tensor([0, 0, 0, 1])
        local_pos = torch.tensor([[31., 31.], [40., 0.], [1., 2.], [4., -3.]])
        local_heading = torch.tensor([.2, -.4, .6, -.8])
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        position, orientation = transform_to_global(
            local_pos, local_heading, scene_pos[batch], scene_heading[batch],
        )
        tokens = InitDiffusionMapCropTest.tokens(
            position, types=[4, 5, 0, 5], batch=batch,
            orientation=orientation,
        )
        return tokens, agent, local_pos, local_heading

    @staticmethod
    def wrapper(processor):
        with patch("src.smart.diffusion.initial_diffusion.Flow", fixtures.ProjectionFlow):
            return InitDiffusion(8, 2, 4, processor, False, use_ema=False)

    def assert_feature_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0,
                                       msg=lambda message, name=name: name + ": " + message)

    def test_square_mask_is_strict_on_each_side_and_keeps_corners(self):
        position = torch.tensor([
            [31.999, 0.], [-31.999, 0.], [0., 31.999], [0., -31.999],
            [32., 0.], [-32., 0.], [0., 32.], [0., -32.],
            [31., 31.], [-31., -31.], [32.001, 0.], [0., -32.001],
        ])
        actual = square_map_mask(position, torch.zeros(len(position), dtype=torch.long),
                                 torch.zeros(1, 2), torch.zeros(1), 32.)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(actual.tolist(), [True] * 4 + [False] * 4 + [True, True, False, False])
        # Half extent, rather than the old radius, controls both local axes.
        smaller = square_map_mask(position, torch.zeros(len(position), dtype=torch.long),
                                  torch.zeros(1, 2), torch.zeros(1), 16.)
        self.assertFalse(smaller.any())

    def test_square_mask_uses_scene_ego_rotation_and_does_not_require_final_map_scene(self):
        tokens, agent, _, _ = self.world_tokens()
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        mask = square_map_mask(tokens["position"], tokens["batch"],
                               scene_pos, scene_heading, 32.)
        self.assertEqual(mask.tolist(), [True, False, True, True])
        # The first corner is outside a world-aligned square; the second point
        # is inside that world square but outside the rotated ego square.
        world_axis_inside = ((tokens["position"] - scene_pos[tokens["batch"]]).abs() < 32.).all(-1)
        self.assertEqual(world_axis_inside[:2].tolist(), [False, True])
        final_scene_empty = tokens["batch"] == 0
        cropped = self.select(tokens, final_scene_empty)
        self.assertEqual(square_map_mask(cropped["position"], cropped["batch"],
                                         scene_pos, scene_heading, 32.).tolist(), [True, False, True])
        empty = square_map_mask(torch.empty(0, 2), torch.empty(0, dtype=torch.long),
                                scene_pos, scene_heading, 32.)
        self.assertEqual(tuple(empty.shape), (0,))

    def test_square_encoder_filters_output_types_and_boundaries_at_every_layer_count(self):
        tokens = self.tokens(
            [[31., 31.], [32., 0.], [0., -32.], [1., 2.], [5., -6.]],
            types=[4, 4, 5, 0, 5],
        )
        for layers in (0, 1, 2):
            with self.subTest(layers=layers):
                feature = self.map_encoder(layers=layers)(tokens, tokenized_agent=self.origin_agent())
                torch.testing.assert_close(feature["position"], tokens["position"][[0, 4]])
                torch.testing.assert_close(feature["orientation"], tokens["orientation"][[0, 4]])
                self.assertEqual(feature["pt_token"].shape, (2, 8))
                self.assertEqual(feature["batch"].tolist(), [0, 0])

    def test_square_crops_all_attention_sources_without_an_outer_halo(self):
        tokens = self.tokens(
            [[30., 0.], [31., 1.], [33., 0.], [34., 1.], [100., 0.]],
            types=[4, 0, 0, 5, 4],
        )
        inside = self.select(tokens, torch.tensor([True, True, False, False, False]))
        mutated = deepcopy(tokens)
        mutated["token_idx"][2:] = (mutated["token_idx"][2:] + 1).remainder(3)
        mutated["light_type"][2:] = 4
        for layers in (1, 2):
            with self.subTest(layers=layers), torch.no_grad():
                encoder = self.map_encoder(layers=layers)
                with patch.object(encoder.edge_encoder, "build_map2map_edge",
                                  wraps=encoder.edge_encoder.build_map2map_edge) as edges:
                    actual = encoder(tokens, tokenized_agent=self.origin_agent())
                self.assertEqual(edges.call_count, 1)
                # The real attention graph includes the retained lane-type
                # source as well as road edges, and no outside source.
                torch.testing.assert_close(edges.call_args.args[0], inside["position"])
                expected = encoder(inside, tokenized_agent=self.origin_agent())
                changed = encoder(mutated, tokenized_agent=self.origin_agent())
                self.assert_feature_equal(actual, expected)
                self.assert_feature_equal(changed, expected)
                changed_inside = deepcopy(inside)
                changed_inside["light_type"][1] = 4
                affected = encoder(changed_inside, tokenized_agent=self.origin_agent())
                self.assertGreater((affected["pt_token"] - expected["pt_token"]).abs().max().item(), 1e-7)

    def test_square_empty_outside_and_missing_final_scene_are_valid_for_all_layer_counts(self):
        tokens, agent, local_pos, local_heading = self.world_tokens()
        cases = (
            (self.select(tokens, tokens["batch"] == 0), local_pos[[0]], local_heading[[0]], [0]),
            (self.select(tokens, torch.zeros(4, dtype=torch.bool)), torch.empty(0, 2), torch.empty(0), []),
            (self.select(tokens, torch.tensor([False, True, False, False])), torch.empty(0, 2), torch.empty(0), []),
        )
        for layers in (0, 1, 2):
            for selected, expected_pos, expected_heading, expected_batch in cases:
                with self.subTest(layers=layers, nodes=len(selected["batch"])):
                    feature = self.map_encoder(layers=layers)(selected, tokenized_agent=agent)
                    torch.testing.assert_close(feature["position"], expected_pos, atol=1e-5, rtol=0)
                    torch.testing.assert_close(feature["orientation"], expected_heading, atol=1e-6, rtol=0)
                    self.assertEqual(feature["batch"].tolist(), expected_batch)
                    self.assertEqual(feature["pt_token"].shape, (len(expected_batch), 8))
                    self.assertTrue(torch.isfinite(feature["pt_token"]).all())

    def test_shared_map_encoding_returns_world_coordinates_then_wrapper_transforms_once(self):
        tokens, agent, local_pos, local_heading = self.world_tokens()
        decoder = self.decoder(sep_map=False)
        with patch.object(decoder.map_encoder, "forward", wraps=decoder.map_encoder.forward) as encode:
            feature = decoder._get_map_feature(tokens, agent)
        self.assertEqual(encode.call_count, 1)
        self.assertIs(encode.call_args.kwargs["tokenized_agent"], agent)
        self.assertFalse(encode.call_args.kwargs["return_local"])
        torch.testing.assert_close(feature["position"], tokens["position"][[0, 3]])
        torch.testing.assert_close(feature["orientation"], tokens["orientation"][[0, 3]])
        initial = self.wrapper(decoder.token_processor)
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        actual = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
        torch.testing.assert_close(actual["position"], local_pos[[0, 3]], atol=1e-5, rtol=0)
        torch.testing.assert_close(actual["orientation"], local_heading[[0, 3]], atol=1e-6, rtol=0)
        torch.testing.assert_close(actual["pt_token"], initial.G1.model.lane_embed(feature["pt_token"]))
        self.assertIs(decoder._get_map_feature(tokens, agent), feature)

    def test_shared_square_dispatch_requires_initial_flow_prediction_and_works_when_frozen(self):
        tokens, _, _, _ = self.world_tokens()
        for name, pred_init, learn_init, square in (
            ("flow", True, True, True), ("flow", True, False, True),
            ("flow", False, False, False), ("scenario_dreamer", True, True, False),
            ("vectorworld", True, True, False),
        ):
            with self.subTest(name=name, pred_init=pred_init, learn_init=learn_init):
                decoder = self.decoder(sep_map=False)
                decoder.init_decoder_name = name
                decoder.token_processor.pred_init = pred_init
                decoder.token_processor.learn_init = learn_init
                _, agent, _, _ = self.world_tokens()
                with patch.object(decoder.map_encoder, "forward", wraps=decoder.map_encoder.forward) as encode:
                    feature = decoder._get_map_feature(tokens, agent)
                if square:
                    self.assertIs(encode.call_args.kwargs["tokenized_agent"], agent)
                    self.assertFalse(encode.call_args.kwargs["return_local"])
                    self.assertEqual(len(feature["position"]), 2)
                else:
                    encode.assert_called_once_with(tokens)
                    self.assertEqual(len(feature["position"]), 3)

    def test_wrapper_crops_world_square_independently_of_legacy_circle_radius(self):
        tokens, agent, local_pos, local_heading = self.world_tokens()
        initial = self.wrapper(self.processor())
        # A prepared world feature can have come from an existing cache.
        road = (tokens["type"] == 4) | (tokens["type"] == 5)
        agent["map_feature"] = dict(
            position=tokens["position"][road], orientation=tokens["orientation"][road],
            batch=tokens["batch"][road], pt_token=torch.randn(3, 8),
        )
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        actual = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
        torch.testing.assert_close(actual["position"], local_pos[[0, 3]], atol=1e-5, rtol=0)
        torch.testing.assert_close(actual["orientation"], local_heading[[0, 3]], atol=1e-6, rtol=0)
        self.assertEqual(actual["batch"].tolist(), [0, 1])

    def test_separate_map_stays_local_and_is_never_transformed_twice(self):
        tokens, agent, local_pos, local_heading = self.world_tokens()
        decoder = self.decoder(sep_map=True)
        feature = decoder._prepare_initial_map_feature(tokens, agent, None)
        torch.testing.assert_close(feature["position"], local_pos[[0, 3]], atol=1e-5, rtol=0)
        initial = self.wrapper(decoder.token_processor)
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        with patch("src.smart.diffusion.initial_diffusion.transform_to_local",
                   side_effect=AssertionError("map already uses ego coordinates")):
            actual = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
        torch.testing.assert_close(actual["position"], local_pos[[0, 3]], atol=1e-5, rtol=0)
        torch.testing.assert_close(actual["orientation"], local_heading[[0, 3]], atol=1e-6, rtol=0)
        torch.testing.assert_close(actual["pt_token"], initial.G1.model.lane_embed(feature["pt_token"]))

    def test_cached_local_features_cannot_bypass_strict_square_crop_or_double_project(self):
        for cache in ("raw", "projected", "ema_raw"):
            with self.subTest(cache=cache):
                agent = fixtures.InitDiffusionSeparateMapTest.agents()
                feature = dict(
                    position=torch.tensor([[31., 31.], [32., 0.], [0., -32.], [3., 4.]]),
                    orientation=torch.tensor([.1, .2, .3, .4]), batch=torch.tensor([0, 0, 1, 1]),
                    pt_token=torch.randn(4, 4 if cache == "projected" else 8),
                )
                if cache == "ema_raw":
                    agent["_initial_map_raw_feature"] = feature
                else:
                    agent["initial_map_feature"] = feature
                    agent["_initial_map_feature_is_raw"] = cache == "raw"
                initial = self.wrapper(self.processor())
                expected_token = (feature["pt_token"][[0, 3]] if cache == "projected" else
                                  initial.G1.model.lane_embed(feature["pt_token"][[0, 3]]))
                with patch("src.smart.diffusion.initial_diffusion.transform_to_local",
                           side_effect=AssertionError("cached map is already local")), \
                        patch.object(initial.G1.model.lane_embed, "forward",
                                     wraps=initial.G1.model.lane_embed.forward) as project:
                    actual = initial._initial_map_feature(
                        agent, agent["initial_pos"][agent["ego_mask"]],
                        agent["initial_heading"][agent["ego_mask"]], 2,
                    )
                self.assertEqual(project.call_count, 0 if cache == "projected" else 1)
                torch.testing.assert_close(actual["position"], feature["position"][[0, 3]])
                torch.testing.assert_close(actual["pt_token"], expected_token)
                self.assertEqual(actual["batch"].tolist(), [0, 1])
                self.assertIs(agent["initial_map_feature"], actual)

    def test_circle_default_preserves_existing_outputs_and_attention_halo(self):
        tokens = self.tokens([[30., 0.], [33., 0.], [31., 31.], [0., 1.]], types=[4, 0, 5, 0])
        changed = deepcopy(tokens)
        changed["light_type"][1] = 4
        for layers in (1, 2):
            with self.subTest(layers=layers), torch.no_grad():
                encoder = self.map_encoder(layers=layers, crop="circle")
                explicit = encoder(tokens, tokenized_agent=self.origin_agent())
                self.assertEqual(explicit["position"].tolist(), [[30., 0.]])
                altered = encoder(changed, tokenized_agent=self.origin_agent())
                self.assertGreater((explicit["pt_token"] - altered["pt_token"]).abs().max().item(), 1e-7)
                del encoder.token_processor.init_map_crop
                del encoder.token_processor.init_map_half_extent
                implicit = encoder(tokens, tokenized_agent=self.origin_agent())
                self.assert_feature_equal(implicit, explicit)
                # Legacy shared map encoding keeps world road edges globally.
                shared = encoder(tokens)
                torch.testing.assert_close(shared["position"], tokens["position"][[0, 2]])

    def test_shared_evaluation_uses_the_same_square_map_route_as_training(self):
        decoder = self.decoder(sep_map=False)
        model = fixtures.InitDiffusionSeparateMapTest.training_model(decoder)
        model.scenario_dreamer_init = True
        model.n_rollout_closed_val = 1
        model.n_vis_batch = 0
        model.challenge_type = None
        tokens, agent, _, _ = self.world_tokens()
        agent["type"] = torch.zeros(len(agent["batch"]), dtype=torch.long)
        count = len(agent["batch"])
        prediction = dict(
            pred_traj_10hz=torch.zeros(count, 1, 2), pred_head_10hz=torch.zeros(count, 1),
            pred_z_10hz=torch.zeros(count, 1), shape=torch.ones(count, 2),
        )
        with patch.object(decoder, "_get_map_feature", wraps=decoder._get_map_feature) as prepare, \
                patch.object(decoder, "inference", return_value=prediction), \
                patch.object(decoder.map_encoder, "forward", wraps=decoder.map_encoder.forward) as encode:
            SMART._rollouts(model, tokens, agent, {})
        prepare.assert_called_once_with(tokens, agent)
        self.assertIs(encode.call_args.kwargs["tokenized_agent"], agent)
        self.assertFalse(encode.call_args.kwargs["return_local"])
        self.assertEqual(agent["map_feature"]["batch"].tolist(), [0, 1])

    @staticmethod
    def public_processor(**options):
        def agent_library(processor, path):
            processor.agent_token_all_veh = torch.zeros(3, 1)

        def map_library(processor, path):
            processor.map_token_traj_src = torch.zeros(3, 22)

        with patch.object(TokenProcessor, "init_agent_token", agent_library), \
                patch.object(TokenProcessor, "init_map_token", map_library):
            values = dict(
                map_token_file="unused-map.pkl", agent_token_file="unused-agent.pkl",
                map_token_sampling=OmegaConf.create(dict(num_k=1, temp=1.)),
                agent_token_sampling=OmegaConf.create(dict(num_k=1, temp=1.)),
            )
            values.update(options)
            return TokenProcessor(**values)

    def test_public_processor_defaults_options_and_validation(self):
        default = self.public_processor()
        self.assertEqual(default.init_map_crop, "circle")
        self.assertEqual(default.init_map_half_extent, 32.)
        chosen = self.public_processor(init_map_crop="square", init_map_half_extent=16.)
        self.assertEqual(chosen.init_map_crop, "square")
        self.assertEqual(chosen.init_map_half_extent, 16.)
        self.assertEqual(validate_init_map_crop("square", 16), 16.)
        for crop in ("triangle", "Square", "", None, True, 0):
            with self.subTest(crop=crop), self.assertRaisesRegex(ValueError, "init_map_crop"):
                self.public_processor(init_map_crop=crop)
        for half_extent in (0., -1., float("nan"), float("inf"), -float("inf"), True, "32", None):
            for crop in ("circle", "square"):
                with self.subTest(crop=crop, half_extent=half_extent), \
                        self.assertRaisesRegex(ValueError, "init_map_half_extent"):
                    self.public_processor(init_map_crop=crop, init_map_half_extent=half_extent)

    def test_configuration_defaults_and_lane_training_evaluation_inheritance(self):
        root = Path(__file__).resolve().parents[1]
        generic = OmegaConf.load(root / "configs/model/smart.yaml").model_config.token_processor
        self.assertEqual(generic.init_map_crop, "circle")
        self.assertEqual(generic.init_map_half_extent, 32.)
        OmegaConf.register_new_resolver("sim_root", lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
            legacy = compose(config_name="run.yaml", overrides=["experiment=init_bc"])
            self.assertEqual(legacy.model.model_config.token_processor.init_map_crop, "circle")
            self.assertEqual(legacy.model.model_config.token_processor.init_map_half_extent, 32.)
            for experiment in ("init_diffusion_lane_conditioned", "init_diffusion_lane_conditioned_eval"):
                with self.subTest(experiment=experiment):
                    config = compose(config_name="run.yaml", overrides=[f"experiment={experiment}"])
                    options = config.model.model_config.token_processor
                    self.assertEqual(options.init_map_crop, "square")
                    self.assertEqual(options.init_map_half_extent, 32.)
                    processor = self.public_processor(**OmegaConf.to_container(options, resolve=True))
                    self.assertEqual(processor.init_map_crop, "square")
                    self.assertEqual(processor.init_map_half_extent, 32.)
                    selected = compose(config_name="run.yaml", overrides=[
                        f"experiment={experiment}",
                        "model.model_config.token_processor.init_map_crop=circle",
                        "model.model_config.token_processor.init_map_half_extent=24",
                    ])
                    self.assertEqual(selected.model.model_config.token_processor.init_map_crop, "circle")
                    self.assertEqual(selected.model.model_config.token_processor.init_map_half_extent, 24.)


if __name__ == "__main__":
    unittest.main()
