"""VectorWorld uses SMART's supervised, optimizer and snapshot metric paths."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from lightning import LightningModule
from waymo_open_dataset.utils.sim_agents.submission_specs import ChallengeType

from src.smart.model.smart import SMART
from src.smart.model.smart_gail import SMART_GAIL
from src.smart.modules.smart_decoder import SMARTDecoder


class UnusedSceneEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, *args, **kwargs):
        raise AssertionError("Graph initialization must not encode the SMART map or policy")

    inference = forward


class TinyVectorWorld(nn.Module):
    loss_kind = "vectorworld"
    use_gan = False

    def __init__(self, token_processor, *, training_stage="ldm", **options):
        super().__init__()
        self.token_processor, self.options = token_processor, options
        self.training_stage = training_stage
        self.learn_autoencoder = training_stage == "autoencoder"
        self.autoencoder = nn.Linear(2, 2)
        self.diff_model = None if self.learn_autoencoder else nn.Linear(2, 2)
        if not self.learn_autoencoder:
            self.autoencoder.requires_grad_(False)
        self.ema_updates, self.ema_observed_weight = 0, None

    def autoencoder_loss(self, agent):
        loss = self.autoencoder(agent["initial_pos"]).square().mean()
        return {"loss": loss, "agent_reconstruction": loss}

    def forward(self, agent):
        if self.learn_autoencoder:
            return self.autoencoder_loss(agent)
        if self.training:
            loss = self.diff_model(agent["initial_pos"]).square().mean()
            return {"loss": loss, "agent_loss": loss}
        pos = agent["initial_pos"].new_tensor([[2., 1.], [3., 4.], [5., 6.]])
        agent["type"], agent["batch"] = torch.tensor([0, 1, 0]), torch.zeros(3, dtype=torch.long)
        agent["generated_map"] = {
            "coordinate_frame": "sd_local", "road_points": pos.new_zeros(1, 20, 2),
            "road_connection_types": torch.eye(6)[[0]],
            "edge_index_lane_to_lane": torch.zeros(2, 1, dtype=torch.long),
            "batch": torch.zeros(1, dtype=torch.long), "lg_type": torch.zeros(1, dtype=torch.long),
        }
        return (pos[:, None], pos.new_zeros(3, 1), torch.zeros(3, 1, dtype=torch.long),
                pos.new_tensor([[4., 2.]]).expand(3, 2), pos.new_tensor([[7., 2.]]).expand(3, 2))

    def update_ema(self):
        if self.diff_model is not None:
            self.ema_updates += 1
            self.ema_observed_weight = self.diff_model.weight.detach().clone()

    def report_options(self):
        return {"ldm_type": "flow", "mode": "initial_scene", "sampling_steps": 24}


class VectorWorldIntegrationTest(unittest.TestCase):
    @staticmethod
    def processor():
        return SimpleNamespace(pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @staticmethod
    def agent():
        return {"num_graphs": 1, "initial_pos": torch.tensor([[0., 0.], [1., 2.]]),
                "type": torch.zeros(2, dtype=torch.long), "batch": torch.zeros(2, dtype=torch.long),
                "initial_scene_only": True}

    def decoder(self, **overrides):
        options = dict(hidden_dim=8, num_historical_steps=11, num_future_steps=80,
                       pl2pl_radius=15., time_span=10, pl2a_radius=50., a2a_radius=50.,
                       num_freq_bands=4, num_map_layers=1, num_agent_layers=1,
                       num_heads=2, head_dim=4, dropout=0., hist_drop_prob=0.,
                       pt2pt_neighbor=8, pt2a_neighbor=8, a2a_neighbor=8, n_token_agent=3,
                       dis_a2a_radius=0., dis_weight=0., dist_decay=1., reward_weight=0.,
                       reward_decay=1., token_processor=self.processor(), init_decoder="vectorworld",
                       vectorworld={"training_mode": "lane_conditioned"})
        options.update(overrides)
        stub = ModuleType("src.smart.vectorworld.decoder")
        stub.VectorWorldInitDecoder = TinyVectorWorld
        with patch.dict("sys.modules", {stub.__name__: stub}), \
                patch("src.smart.modules.smart_decoder.SMARTMapDecoder", UnusedSceneEncoder), \
                patch("src.smart.modules.smart_decoder.SMARTAgentDecoder", UnusedSceneEncoder):
            return SMARTDecoder(**options)

    def model(self, cls=SMART_GAIL, *, stage="ldm"):
        model = cls.__new__(cls)
        LightningModule.__init__(model)
        model.encoder = self.decoder(vectorworld={"training_stage": stage})
        model.token_processor = model.encoder.token_processor
        model.optimizer_profile = "scenario_dreamer"
        model.lr, model.lr_warmup_steps = 1e-4, 0
        model.training_rollout_len, model.gail = 1, False
        model.scenario_dreamer_init = True
        model.n_rollout_closed_val, model.n_vis_batch = 1, 0
        model.challenge_type = ChallengeType.SCENARIO_GEN
        return model

    def test_decoder_selects_vectorworld_and_passes_options(self):
        decoder = self.decoder()
        self.assertIsInstance(decoder.init_decoder, TinyVectorWorld)
        self.assertEqual(decoder.init_decoder.options, {"training_mode": "lane_conditioned"})
        self.assertIs(decoder.init_decoder.token_processor, decoder.token_processor)

    def test_incompatible_gail_and_sep_map_are_rejected(self):
        for options in ({"dis_a2a_radius": 20.}, {"sep_map": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.decoder(**options)

    def test_training_routes_dictionary_loss_without_unused_encoders(self):
        model, agent, tokens = self.model(), self.agent(), {"sentinel": torch.ones(1)}
        with patch.object(model, "_log_train") as log:
            loss, log_prob, entropy = model.get_pred(tokens, agent)
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(agent["tokenized_map"], tokens)
        self.assertEqual(log_prob.numel(), 0)
        self.assertEqual(entropy.numel(), 0)
        self.assertIn("train/vectorworld/agent_loss", [call.args[0] for call in log.call_args_list])
        loss.backward()
        self.assertIsNotNone(model.encoder.init_decoder.diff_model.weight.grad)
        self.assertIsNone(model.encoder.map_encoder.weight.grad)
        self.assertIsNone(model.encoder.agent_encoder.weight.grad)

    def test_inference_returns_continuous_snapshot_and_generated_map(self):
        decoder, agent = self.decoder().eval(), self.agent()
        out = decoder.inference(agent)
        self.assertEqual(out["pred_traj_10hz"].shape, (3, 1, 2))
        self.assertEqual(out["pred_head_10hz"].shape, (3, 1))
        self.assertEqual(out["initial_local_vel"].shape, (3, 2))
        torch.testing.assert_close(out["initial_local_vel"], torch.tensor([[7., 2.]]).expand(3, 2))
        self.assertIs(out["generated_map"], agent["generated_map"])
        self.assertEqual(agent["type"].tolist(), [0, 1, 0])

    def test_rollouts_preserve_generated_counts_types_and_map_and_input(self):
        model, agent = self.model().eval(), self.agent()
        out = model._rollouts({}, agent, {})
        self.assertEqual(out["traj"].shape, (3, 1, 1, 2))
        self.assertEqual(out["vel"].shape, (3, 1, 2))
        self.assertEqual(out["size"].shape, (3, 1, 1, 2))
        self.assertEqual(out["generated_type"].flatten().tolist(), [0, 1, 0])
        self.assertEqual(out["generated_batch"].tolist(), [0, 0, 0])
        self.assertEqual(out["generated_map"]["road_points"].shape, (1, 20, 2))
        self.assertEqual(agent["type"].tolist(), [0, 0])
        self.assertNotIn("generated_map", agent)

    def test_generated_map_requires_one_rollout(self):
        model = self.model().eval()
        model.n_rollout_closed_val = 2
        with self.assertRaisesRegex(ValueError, "Generated-map evaluation requires exactly one"):
            model._rollouts({}, self.agent(), {})

    def test_generated_metadata_is_passed_to_cached_evaluator(self):
        model, agent = self.model().eval(), self.agent()
        model.use_sd_evaluator, model.sd_evaluator = True, Mock()
        data = {"scenario_dreamer_cache_file": "scene.pkl"}
        model._validate_closed_loop(data, {}, agent, 0)
        received_data, received_agent, out = model.sd_evaluator.update.call_args.args
        self.assertIs(received_data, data)
        self.assertEqual(received_agent["type"].tolist(), [0, 1, 0])
        self.assertEqual(received_agent["batch"].tolist(), [0, 0, 0])
        self.assertIn("generated_map", out)

    def test_ae_validation_and_test_use_vectorworld_names_and_no_sampling(self):
        model, agent = self.model(stage="autoencoder"), self.agent()
        model.token_processor = Mock(return_value=({}, agent))
        with patch.object(model, "log") as log, \
                patch.object(model, "_rollouts", side_effect=AssertionError("AE must not sample")):
            model.on_validation_epoch_start()
            val = model.validation_step({}, 0)
            model.on_validation_epoch_end()
            model.on_test_epoch_start()
            test = model.test_step({}, 0)
            model.on_test_epoch_end()
        self.assertTrue(torch.isfinite(val) and torch.isfinite(test))
        names = [call.args[0] for call in log.call_args_list]
        self.assertIn("val/vectorworld/autoencoder/loss", names)
        self.assertIn("test/vectorworld/autoencoder/loss", names)
        with patch.object(model, "_log_train") as log:
            result = model.encoder.init_decoder(agent)
            loss = model._initial_prediction_loss({"initial_logit": result}, agent, result["loss"])
        self.assertIs(loss, result["loss"])
        self.assertIn("train/vectorworld/autoencoder/loss", [call.args[0] for call in log.call_args_list])

    def test_optimizer_selects_stage_model_for_both_entry_points(self):
        for cls in (SMART, SMART_GAIL):
            for stage in ("autoencoder", "ldm"):
                with self.subTest(entry_point=cls.__name__, stage=stage):
                    model = self.model(cls, stage=stage)
                    optimizer = model.configure_optimizers()["optimizer"]
                    target = model.encoder.init_decoder
                    target = target.autoencoder if stage == "autoencoder" else target.diff_model
                    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
                    self.assertEqual(actual, {id(p) for p in target.parameters() if p.requires_grad})

    def test_scratch_freezes_unused_encoders_with_default_optimizer(self):
        model = self.model()
        model._configure_finetuning(False)
        for module in (model.encoder.map_encoder, model.encoder.agent_encoder):
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.encoder.init_decoder.diff_model.parameters()))

    def test_actual_optimizer_step_updates_ema_after_online_weights(self):
        model = self.model()
        optimizer = model.configure_optimizers()["optimizer"]
        decoder = model.encoder.init_decoder
        before = decoder.diff_model.weight.detach().clone()

        def closure():
            optimizer.zero_grad()
            loss = decoder(self.agent())["loss"]
            loss.backward()
            return loss

        model.optimizer_step(epoch=0, batch_idx=0, optimizer=optimizer, optimizer_closure=closure)
        self.assertEqual(decoder.ema_updates, 1)
        self.assertFalse(torch.equal(before, decoder.diff_model.weight))
        torch.testing.assert_close(decoder.ema_observed_weight, decoder.diff_model.weight)

    def test_metric_report_uses_vectorworld_options_without_sd_model_fields(self):
        model = self.model().eval()
        model.val_closed_loop, model.use_sd_evaluator = True, True
        model.wosac_submission = SimpleNamespace(is_active=False)
        model.sd_evaluator = SimpleNamespace(compute=lambda: {"collision_rate": 1.5}, report=lambda: {})
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "log") as log:
            model.video_dir = Path(directory) / "videos"
            model.on_validation_epoch_end()
            report = json.loads((Path(directory) / "sd_agent_metrics.json").read_text())
        self.assertEqual(report["decoder"], {"name": "vectorworld", "ldm_type": "flow",
                                             "mode": "initial_scene", "sampling_steps": 24})
        self.assertEqual(log.call_args.args[:2], ("collision_rate", 1.5))


if __name__ == "__main__":
    unittest.main()
