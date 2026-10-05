"""Independent initial-state map conditioning and legacy checkpoint migration."""

from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from lightning import LightningModule

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.model.smart import SMART
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.utils import transform_to_local


class TinyPolicy(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, agent, feature):
        return {"policy_logit": feature["pt_token"] * self.weight}


class TinyInitialDecoder(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(1, hidden_dim + 1)[None] / hidden_dim)
            self.projection.bias.fill_(.3)

    def forward(self, agent):
        return self.projection(agent["initial_map_feature"]["pt_token"]).square().mean()


class ProjectionFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.hidden_dim = 4
        self.model.lane_embed = nn.Linear(8, 4, bias=False)


class EqualWidthProjectionFlow(ProjectionFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.hidden_dim = 8
        self.model.lane_embed = nn.Linear(8, 8, bias=False)


class InitDiffusionSeparateMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def processor(*, pred_init=True, learn_init=True):
        return SimpleNamespace(
            pred_init=pred_init, learn_init=learn_init, init_map_range=50.,
            map_token_traj_src=torch.linspace(-1., 1., 66).reshape(3, 22),
        )

    def decoder(self, *, sep_map=True, learn_init=True, pred_init=True,
                num_map_layers=1, finetune=False, **options):
        settings = dict(
            hidden_dim=8, num_historical_steps=11, num_future_steps=80,
            pl2pl_radius=15., time_span=10, pl2a_radius=50., a2a_radius=50.,
            num_freq_bands=4, num_map_layers=num_map_layers, num_agent_layers=1,
            num_heads=2, head_dim=4, dropout=0., hist_drop_prob=0.,
            pt2pt_neighbor=8, pt2a_neighbor=8, a2a_neighbor=8, n_token_agent=3,
            dis_a2a_radius=0., dis_weight=0., dist_decay=1., reward_weight=0.,
            reward_decay=1., token_processor=self.processor(
                pred_init=pred_init, learn_init=learn_init),
            initial_scene_only=pred_init, finetune=finetune, sep_map=sep_map,
        )
        settings.update(options)
        with patch("src.smart.modules.smart_decoder.SMARTAgentDecoder", TinyPolicy), \
                patch("src.smart.modules.smart_decoder.InitDiffusion", TinyInitialDecoder):
            return SMARTDecoder(**settings)

    @staticmethod
    def agents():
        return dict(
            num_graphs=2, batch=torch.tensor([0, 0, 1, 1]),
            ego_mask=torch.tensor([True, False, True, False]),
            initial_pos=torch.tensor([[10., -4.], [12., -3.], [-8., 20.], [-6., 21.]]),
            initial_heading=torch.tensor([math.pi / 2., .2, -.4, -.1]),
        )

    @staticmethod
    def map_tokens(*, final_scene_empty=False, empty=False):
        result = dict(
            type=torch.tensor([4, 5, 0, 4]), batch=torch.tensor([0, 0, 0, 1]),
            position=torch.tensor([[12., -4.], [10., -1.], [11., -3.], [-7., 20.]]),
            orientation=torch.tensor([.3, -.2, .1, .6]),
            token_idx=torch.tensor([0, 1, 2, 1]), light_type=torch.zeros(4, dtype=torch.long),
        )
        keep = torch.zeros(4, dtype=torch.bool) if empty else (
            result["batch"] == 0 if final_scene_empty else torch.ones(4, dtype=torch.bool))
        return {key: value[keep] for key, value in result.items()}

    @staticmethod
    def training_model(decoder):
        model = SMART.__new__(SMART)
        LightningModule.__init__(model)
        model.encoder = decoder
        model.token_processor = decoder.token_processor
        model.training_rollout_len = 1
        return model

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0,
                                       msg=lambda message, name=name: name + ": " + message)

    def test_separate_map_starts_equal_but_has_independent_parameters_and_same_architecture(self):
        for layers in (1, 2):
            with self.subTest(layers=layers):
                decoder = self.decoder(num_map_layers=layers)
                self.assertTrue(decoder.sep_map)
                self.assertIsNot(decoder.init_map_encoder, decoder.map_encoder)
                self.assertEqual(decoder.init_map_encoder.num_layers, layers)
                self.assert_state_equal(decoder.init_map_encoder.state_dict(), decoder.map_encoder.state_dict())
                shared_ids = {id(parameter) for parameter in decoder.map_encoder.parameters()}
                self.assertTrue(all(id(parameter) not in shared_ids
                                    for parameter in decoder.init_map_encoder.parameters()))

    def test_default_shared_map_has_no_new_state_and_disabled_init_does_not_build_it(self):
        decoder = self.decoder(sep_map=False)
        self.assertFalse(decoder.sep_map)
        self.assertIsNone(decoder.init_map_encoder)
        self.assertFalse(any(key.startswith("init_map_encoder.") for key in decoder.state_dict()))
        no_initial = self.decoder(sep_map=False, pred_init=False)
        self.assertIsNone(no_initial.init_decoder)

    def test_sep_map_rejects_missing_initial_prediction_and_other_decoder_paths(self):
        for options in (dict(pred_init=False), dict(init_decoder="scenario_dreamer"),
                        dict(dis_a2a_radius=50.)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.decoder(**options)

    def test_old_shared_only_checkpoint_strictly_initializes_from_loaded_weights_in_nested_model(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                source = self.decoder(sep_map=False)
                with torch.no_grad():
                    for parameter in source.map_encoder.parameters():
                        parameter.add_(.25)
                target = self.decoder()
                if nested:
                    source_wrapper, target_wrapper = nn.Module(), nn.Module()
                    source_wrapper.encoder, target_wrapper.encoder = source, target
                    source_state = source_wrapper.state_dict()
                    incompatible = target_wrapper.load_state_dict(source_state, strict=True)
                else:
                    incompatible = target.load_state_dict(source.state_dict(), strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assert_state_equal(target.map_encoder.state_dict(), source.map_encoder.state_dict())
                self.assert_state_equal(target.init_map_encoder.state_dict(), source.map_encoder.state_dict())
                saved_initial = {name: tensor.clone() for name, tensor in target.init_map_encoder.state_dict().items()}
                with torch.no_grad():
                    next(target.map_encoder.parameters()).add_(1.)
                target.initialize_initial_map_from_shared()
                self.assert_state_equal(target.init_map_encoder.state_dict(), saved_initial)

    def test_existing_independent_map_checkpoint_and_explicit_reinitialization_preserve_trained_weights(self):
        source = self.decoder()
        with torch.no_grad():
            for parameter in source.init_map_encoder.parameters():
                parameter.add_(.75)
        expected = {name: tensor.clone() for name, tensor in source.init_map_encoder.state_dict().items()}
        target = self.decoder()
        target.load_state_dict(source.state_dict(), strict=True)
        target.initialize_initial_map_from_shared()
        self.assert_state_equal(target.init_map_encoder.state_dict(), expected)
        self.assertFalse(torch.equal(next(target.map_encoder.parameters()),
                                     next(target.init_map_encoder.parameters())))

    def test_legacy_migration_does_not_hide_unrelated_missing_checkpoint_weights(self):
        legacy = self.decoder(sep_map=False).state_dict()
        legacy.pop("map_encoder.type_pt_emb.weight")
        with self.assertRaisesRegex(RuntimeError, "map_encoder.type_pt_emb.weight"):
            self.decoder().load_state_dict(legacy, strict=True)

    def test_independent_map_trains_through_actual_map_attention_while_shared_map_stays_frozen(self):
        for finetune, layers, snapshot in ((False, 1, True), (True, 1, True),
                                          (False, 2, True), (False, 1, False)):
            with self.subTest(finetune=finetune, layers=layers, snapshot=snapshot):
                torch.manual_seed(12)
                decoder = self.decoder(finetune=finetune, num_map_layers=layers,
                                       initial_scene_only=snapshot)
                model = self.training_model(decoder)
                model._configure_finetuning(enabled=finetune)
                self.assertFalse(any(parameter.requires_grad for parameter in decoder.map_encoder.parameters()))
                self.assertFalse(any(parameter.requires_grad for parameter in decoder.agent_encoder.parameters()))
                self.assertTrue(all(parameter.requires_grad for parameter in decoder.init_map_encoder.parameters()))
                shared_before = {name: tensor.clone() for name, tensor in decoder.map_encoder.state_dict().items()}
                init_before = next(decoder.init_map_encoder.token_emb.parameters()).detach().clone()
                with patch.object(decoder.map_encoder, "forward", side_effect=AssertionError("shared map must not run")):
                    prediction = decoder(self.map_tokens(), self.agents())
                loss = prediction["initial_logit"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradient = next(decoder.init_map_encoder.token_emb.parameters()).grad
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0.)
                self.assertTrue(all(parameter.grad is None for parameter in decoder.map_encoder.parameters()))
                torch.optim.SGD((parameter for parameter in decoder.parameters() if parameter.requires_grad), lr=.01).step()
                self.assertFalse(torch.equal(init_before, next(decoder.init_map_encoder.token_emb.parameters())))
                self.assert_state_equal(shared_before, decoder.map_encoder.state_dict())

    def test_independent_map_is_frozen_when_initial_decoder_is_not_learned(self):
        for finetune in (False, True):
            with self.subTest(finetune=finetune):
                model = self.training_model(self.decoder(learn_init=False, finetune=finetune))
                model._configure_finetuning(enabled=finetune)
                self.assertFalse(any(parameter.requires_grad
                                     for parameter in model.encoder.init_map_encoder.parameters()))

    def test_both_optimizer_profiles_include_independent_map_and_exclude_shared_map(self):
        for profile in ("default", "scenario_dreamer"):
            with self.subTest(profile=profile):
                model = self.training_model(self.decoder())
                model._configure_finetuning(enabled=False)
                model.optimizer_profile = profile
                model.lr, model.lr_warmup_steps = 1e-4, 0
                model.lr_total_steps, model.lr_min_ratio = 64, .05
                optimizer = model.configure_optimizers()["optimizer"]
                actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
                expected = {id(parameter) for module in (model.encoder.init_map_encoder, model.encoder.init_decoder)
                            for parameter in module.parameters() if parameter.requires_grad}
                self.assertEqual(actual, expected)

    def test_independent_map_handles_empty_and_missing_final_scene_without_wrong_ego_assignment(self):
        decoder = self.decoder()
        for options in (dict(), dict(final_scene_empty=True), dict(empty=True)):
            with self.subTest(options=options):
                tokens, agent = self.map_tokens(**options), self.agents()
                feature = decoder.init_map_encoder(tokens, tokenized_agent=agent)
                road_edge = (tokens["type"] == 4) | (tokens["type"] == 5)
                batch = tokens["batch"][road_edge]
                expected_pos, expected_heading = transform_to_local(
                    tokens["position"][road_edge], tokens["orientation"][road_edge],
                    agent["initial_pos"][agent["ego_mask"]][batch],
                    agent["initial_heading"][agent["ego_mask"]][batch],
                )
                torch.testing.assert_close(feature["position"], expected_pos)
                torch.testing.assert_close(feature["orientation"], expected_heading)
                torch.testing.assert_close(feature["batch"], batch)
                self.assertEqual(feature["pt_token"].shape, (len(batch), 8))

    def test_flow_projects_separate_local_map_without_transforming_it_twice_and_ema_keeps_raw_features(self):
        decoder = self.decoder()
        tokens, agent = self.map_tokens(final_scene_empty=True), self.agents()
        decoder._prepare_initial_map_feature(tokens, agent, None)
        raw = dict(agent["initial_map_feature"])
        with patch("src.smart.diffusion.initial_diffusion.Flow", ProjectionFlow):
            initial = InitDiffusion(8, 2, 4, decoder.token_processor, False, use_ema=True)
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        with patch("src.smart.diffusion.initial_diffusion.transform_to_local",
                   side_effect=AssertionError("local map must not be transformed twice")):
            actual = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
            torch.testing.assert_close(actual["position"], raw["position"])
            torch.testing.assert_close(actual["orientation"], raw["orientation"])
            torch.testing.assert_close(actual["pt_token"], initial.G1.model.lane_embed(raw["pt_token"]))
            with torch.no_grad():
                initial.G1.model.lane_embed.weight.add_(.2)
            updated = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
            torch.testing.assert_close(updated["pt_token"], initial.G1.model.lane_embed(raw["pt_token"]))
        self.assertEqual(len(initial.ema.shadow_params), len(list(initial.G1.parameters())))
        torch.testing.assert_close(agent["_initial_map_raw_feature"]["pt_token"], raw["pt_token"])

    def test_equal_width_raw_map_still_projects_exactly_once_without_ema(self):
        decoder = self.decoder()
        agent = self.agents()
        decoder._prepare_initial_map_feature(self.map_tokens(), agent, None)
        raw_tokens = agent["initial_map_feature"]["pt_token"].clone()
        with patch("src.smart.diffusion.initial_diffusion.Flow", EqualWidthProjectionFlow):
            initial = InitDiffusion(8, 2, 4, decoder.token_processor, False, use_ema=False)
        scene_pos = agent["initial_pos"][agent["ego_mask"]]
        scene_heading = agent["initial_heading"][agent["ego_mask"]]
        with patch.object(initial.G1.model.lane_embed, "forward",
                          wraps=initial.G1.model.lane_embed.forward) as project:
            actual = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
            self.assertEqual(project.call_count, 1)
            expected = torch.nn.functional.linear(raw_tokens, initial.G1.model.lane_embed.weight)
            torch.testing.assert_close(actual["pt_token"], expected)
            repeated = initial._initial_map_feature(agent, scene_pos, scene_heading, 2)
            self.assertEqual(project.call_count, 1)
            torch.testing.assert_close(repeated["pt_token"], expected)


if __name__ == "__main__":
    unittest.main()
