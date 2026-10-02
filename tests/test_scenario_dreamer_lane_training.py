"""Agent-only denoising with clean regular/partitioned lane latents."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from test_scenario_dreamer_init_decoder import make_checkpoints, official_scene


class ScenarioDreamerLaneTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.ae, cls.ldm = make_checkpoints(Path(cls.temp.name))
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def model(self, **options):
        args = dict(ae_checkpoint=self.ae, ldm_checkpoint=self.ldm,
                    map_source="exact", training_mode="lane_conditioned")
        args.update(options)
        return ScenarioDreamerInitDecoder(self.processor, **args)

    def inputs(self, kinds=(0, 0)):
        batch = Batch.from_data_list([
            HeteroData(adapt_preprocessed_scene(official_scene(kind), f"{i}.pkl"))
            for i, kind in enumerate(kinds)
        ])
        tokens, agent = self.processor(batch)
        agent["tokenized_map"] = tokens
        return agent

    def test_core_noises_only_agents_keeps_lanes_and_uses_shared_scene_timesteps(self):
        model = self.model().train()
        data = model._encode(self.inputs())[0]
        core = model.diff_model
        original = core.model.forward
        captured = {}

        def capture(x_agent, x_lane, graph, t_agent, t_lane):
            captured.update(agent=x_agent.clone(), lane=x_lane.clone(),
                            t_agent=t_agent.clone(), t_lane=t_lane.clone())
            return original(x_agent, x_lane, graph, t_agent, t_lane)

        with patch.object(core.model, "forward", side_effect=capture), \
                patch.object(core, "q_sample", wraps=core.q_sample) as q_sample, \
                patch("src.smart.scenario_dreamer.core.ldm.torch.randn_like", wraps=torch.randn_like) as noise:
            losses = core.loss(data, mode="lane_conditioned")
        self.assertEqual(q_sample.call_count, 1)
        self.assertEqual(noise.call_count, 1)
        torch.testing.assert_close(captured["lane"], data["lane"].latents.unsqueeze(1), atol=0, rtol=0)
        self.assertFalse(torch.equal(captured["agent"], data["agent"].latents.unsqueeze(1)))
        for scene in range(data.batch_size):
            agent_times = captured["t_agent"][data["agent"].batch == scene].unique()
            lane_times = captured["t_lane"][data["lane"].batch == scene].unique()
            self.assertEqual(agent_times.numel(), 1)
            torch.testing.assert_close(agent_times, lane_times, atol=0, rtol=0)
        self.assertEqual(losses["lane_loss"].item(), 0)
        self.assertFalse(losses["lane_loss"].requires_grad)
        torch.testing.assert_close(losses["loss"], losses["agent_loss"], atol=0, rtol=0)
        losses["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in core.model.pred_agent_noise.parameters()))
        self.assertTrue(all(p.grad is None for p in core.model.pred_lane_noise.parameters()))

    def test_decoder_training_objective_is_independent_of_generation_mode(self):
        model = self.model(generation_mode="initial_scene").train()
        with patch.object(model.diff_model, "loss", wraps=model.diff_model.loss) as loss:
            result = model(self.inputs())
        self.assertEqual(loss.call_args.kwargs["mode"], "lane_conditioned")
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["lane_loss"].item(), 0)
        result["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.autoencoder.parameters()))
        self.assertFalse(model.autoencoder.training)
        torch.optim.SGD(model.diff_model.parameters(), lr=.01).step()
        model.update_ema()
        restored = self.model()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(restored.get_extra_state()["training_mode"], "lane_conditioned")
        self.assertEqual(restored.ema.num_updates, model.ema.num_updates)
        for actual, expected in zip(restored.diff_model.parameters(), model.diff_model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_clean_lanes_retain_the_training_vae_posterior_sampling_convention(self):
        model = self.model().train()
        agent = self.inputs()
        graph = model._build_graph(agent)[0]
        with torch.no_grad():
            _, lane_mu, _, _ = model.autoencoder.forward_encoder(graph, return_stats=True)
        encoded = model._encode(agent)[0]["lane"].latents
        mean = (lane_mu - model.cfg.dataset.lane_latents_mean) / model.cfg.dataset.lane_latents_std
        self.assertFalse(torch.equal(encoded, mean))

    def test_training_rejects_unsupported_missing_metadata_and_token_fallback_before_encoding(self):
        cases = [
            ({}, (0, 1), lambda a: a["sd_map"].update(lg_type=torch.tensor([0, 2])), r"batch indices \[1\]"),
            ({}, (0, 0), lambda a: a["sd_map"].pop("lg_type"), "missing metadata"),
            ({}, (0, 0), lambda a: a["sd_map"].update(lg_type=torch.tensor([0])), "metadata count"),
            ({"map_source": "auto"}, (0, 0), lambda a: a.pop("sd_map"), "token-map fallback"),
            ({"map_source": "tokens"}, (0, 0), None, "token-map fallback"),
        ]
        for options, kinds, mutate, message in cases:
            for cached in (False, True):
                with self.subTest(options=options, kinds=kinds, cached=cached):
                    model = self.model(**options).train()
                    agent = self.inputs(kinds)
                    if mutate:
                        mutate(agent)
                    if cached:
                        agent["sd_cached_posterior"] = {}
                    with patch.object(model, "_encode") as encode, \
                            patch.object(model, "_check_cached_encoder") as cache:
                        with self.assertRaisesRegex(ValueError, message):
                            model(agent)
                    encode.assert_not_called()
                    cache.assert_not_called()

    def test_partitioned_agent_mask_keeps_prefix_clean_and_sets_zero_noise_targets(self):
        for kinds in ((0, 1), (1, 1)):
            with self.subTest(kinds=kinds):
                model = self.model().train()
                data = model._encode(self.inputs(kinds))[0]
                core = model.diff_model
                agent = data["agent"].latents.unsqueeze(1)
                lane = data["lane"].latents.unsqueeze(1)
                before = data["agent"].partition_mask.bool()
                self.assertTrue(before.any())
                self.assertTrue((~before).any())
                noise = torch.full_like(agent, .4)
                t_agent = torch.ones(agent.shape[0], dtype=torch.long)
                t_lane = torch.ones(lane.shape[0], dtype=torch.long)
                expected_noised = core.q_sample(agent, t_agent, noise)
                original_model = core.model.forward
                original_loss = core.agent_loss_fn.forward
                captured = {}

                def capture_model(x_agent, x_lane, *args):
                    captured.update(agent=x_agent.clone(), lane=x_lane.clone())
                    return original_model(x_agent, x_lane, *args)

                def capture_loss(prediction, target, *args):
                    captured["target"] = target.clone()
                    return original_loss(prediction, target, *args)

                with patch.object(core.model, "forward", side_effect=capture_model), \
                        patch.object(core.agent_loss_fn, "forward", side_effect=capture_loss), \
                        patch("src.smart.scenario_dreamer.core.ldm.torch.randn_like", return_value=noise.clone()):
                    loss, agent_loss, lane_loss = core.p_losses(
                        agent, lane, data, t_agent, t_lane, mode="lane_conditioned")
                torch.testing.assert_close(captured["agent"][before], agent[before], atol=0, rtol=0)
                torch.testing.assert_close(captured["agent"][~before], expected_noised[~before], atol=0, rtol=0)
                torch.testing.assert_close(captured["lane"], lane, atol=0, rtol=0)
                torch.testing.assert_close(captured["target"][before], torch.zeros_like(noise[before]), atol=0, rtol=0)
                torch.testing.assert_close(captured["target"][~before], noise[~before], atol=0, rtol=0)
                torch.testing.assert_close(loss, agent_loss, atol=0, rtol=0)
                self.assertFalse(lane_loss.any())
                loss.mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in core.model.pred_agent_noise.parameters()))
                self.assertTrue(all(p.grad is None for p in core.model.pred_lane_noise.parameters()))

    def test_mixed_and_partitioned_decoder_training_succeeds_but_evaluation_rejects_them(self):
        for kinds in ((0, 1), (1, 1)):
            with self.subTest(kinds=kinds):
                model = self.model(generation_mode="lane_conditioned").train()
                losses = model(self.inputs(kinds))
                self.assertTrue(torch.isfinite(losses["loss"]))
                self.assertEqual(losses["lane_loss"].item(), 0)
                losses["loss"].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.diff_model.parameters()))
                with patch.object(model, "_encode") as encode:
                    with self.assertRaisesRegex(ValueError, "full non-partitioned"):
                        model.eval()(self.inputs(kinds))
                encode.assert_not_called()

    def test_joint_objective_remains_default_and_accepts_partitioned_training(self):
        model = self.model(training_mode="joint", generation_mode="lane_conditioned").train()
        data = model._encode(self.inputs((0, 1)))[0]
        torch.manual_seed(37)
        default = model.diff_model.loss(data)
        torch.manual_seed(37)
        explicit = model.diff_model.loss(data, mode="joint")
        for key in default:
            torch.testing.assert_close(default[key], explicit[key], atol=0, rtol=0)
        self.assertGreater(default["lane_loss"].item(), 0)
        self.assertTrue(torch.isfinite(model(self.inputs((0, 1)))["loss"]))

    def test_resume_mode_metadata_is_backward_compatible_and_rejects_changed_objectives(self):
        joint = self.model(training_mode="joint")
        state = copy.deepcopy(joint.state_dict())
        self.assertEqual(state["_extra_state"].pop("training_mode"), "joint")
        joint.load_state_dict(state, strict=True)
        with self.assertRaisesRegex(ValueError, "same training_mode.*checkpoint=joint"):
            self.model().load_state_dict(state, strict=True)
        conditional = self.model()
        state = conditional.state_dict()
        with self.assertRaisesRegex(ValueError, "same training_mode.*checkpoint=lane_conditioned"):
            joint.load_state_dict(state, strict=True)
        # Official weights initialize either objective with the identical parameter schema.
        official = torch.load(self.ldm, weights_only=False)["state_dict"]
        for key, value in conditional.diff_model.state_dict().items():
            torch.testing.assert_close(value, official["diff_model." + key], atol=0, rtol=0)

    def test_invalid_training_modes_and_autoencoder_combination_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "training_mode must be"):
            self.model(training_mode="unknown")
        with self.assertRaisesRegex(ValueError, "requires training_stage=ldm"):
            self.model(training_stage="autoencoder", ldm_checkpoint=None)


if __name__ == "__main__":
    unittest.main()
