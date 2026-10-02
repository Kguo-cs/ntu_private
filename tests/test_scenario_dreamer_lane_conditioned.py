"""Lane-conditioned evaluation accepts only explicitly complete reference maps."""
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


class ScenarioDreamerLaneConditionedTest(unittest.TestCase):
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
                    map_source="exact", generation_mode="lane_conditioned")
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

    def assert_rejected_before_encoding(self, model, agent, message):
        with patch.object(model, "_encode") as encode, \
                patch.object(model, "_check_cached_encoder") as cache, \
                patch.object(model.autoencoder, "forward_encoder") as ae, \
                patch.object(model.diff_model, "forward") as diffusion:
            with self.assertRaisesRegex(ValueError, message):
                model.eval()(agent)
        encode.assert_not_called()
        cache.assert_not_called()
        ae.assert_not_called()
        diffusion.assert_not_called()

    def test_full_map_inference_keeps_reference_lanes_and_conditions_diffusion(self):
        for source in ("exact", "auto"):
            with self.subTest(map_source=source):
                model = self.model(map_source=source).eval()
                agent = self.inputs()
                before = {key: value.clone() for key, value in agent["sd_map"].items()}
                batch_before = agent["batch"].clone()
                agent["generated_map"] = {"stale": True}
                original_forward = model.diff_model.forward
                calls = []

                def capture(data, mode):
                    calls.append(mode)
                    expected_lanes = data["lane"].latents.clone()
                    result = original_forward(data, mode=mode)
                    torch.testing.assert_close(result[1], expected_lanes, atol=0, rtol=0)
                    return result

                with patch.object(model.diff_model, "forward", side_effect=capture):
                    output = model(agent)
                self.assertEqual(calls, ["lane_conditioned"])
                self.assertEqual(len(output), 5)
                self.assertEqual(output[0].shape, (6, 1, 2))
                self.assertNotIn("generated_map", agent)
                torch.testing.assert_close(agent["batch"], batch_before, atol=0, rtol=0)
                for key, expected in before.items():
                    torch.testing.assert_close(agent["sd_map"][key], expected, atol=0, rtol=0)

    def test_partitioned_and_mixed_maps_fail_before_online_or_cached_encoding(self):
        model = self.model()
        for kinds, indices in (((0, 1), r"\[1\]"), ((1, 1), r"\[0, 1\]")):
            for cached in (False, True):
                with self.subTest(kinds=kinds, cached=cached):
                    agent = self.inputs(kinds)
                    if cached:
                        agent["sd_cached_posterior"] = {}
                    self.assert_rejected_before_encoding(model, agent, "batch indices " + indices)

    def test_unknown_missing_and_wrong_count_metadata_fail_early(self):
        model = self.model()
        mutations = [
            (lambda agent: agent["sd_map"].pop("lg_type"), "missing metadata"),
            (lambda agent: agent["sd_map"].update(lg_type=None), "missing metadata"),
            (lambda agent: agent["sd_map"].update(lg_type=torch.tensor([0])), "metadata count"),
            (lambda agent: agent["sd_map"].update(lg_type=torch.tensor([0, 0, 0])), "metadata count"),
            (lambda agent: agent["sd_map"].update(lg_type=torch.tensor([0, 2])), r"batch indices \[1\]"),
            (lambda agent: agent["sd_map"].update(lg_type=torch.tensor([0., float("nan")])), r"batch indices \[1\]"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                agent = self.inputs()
                mutate(agent)
                agent["sd_cached_posterior"] = {}
                self.assert_rejected_before_encoding(model, agent, message)

    def test_token_map_fallback_cannot_claim_nonpartitioned_lanes(self):
        for source in ("exact", "auto", "tokens"):
            for keep_exact in (False, True):
                if keep_exact and source != "tokens":
                    continue
                with self.subTest(map_source=source, keep_exact=keep_exact):
                    model = self.model(map_source=source)
                    agent = self.inputs()
                    if not keep_exact:
                        agent.pop("sd_map")
                    self.assert_rejected_before_encoding(model, agent, r"token-map fallback.*batch indices \[0, 1\]")

    def test_partitioned_ldm_training_and_ae_training_validation_remain_allowed(self):
        model = self.model().train()
        losses = model(self.inputs((0, 1)))
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.diff_model.parameters()))
        ae = self.model(training_stage="autoencoder", ldm_checkpoint=None)
        for training in (True, False):
            with self.subTest(ae_training=training):
                losses = ae.train(training)(self.inputs((0, 1)))
                self.assertTrue(torch.isfinite(losses["loss"]))

    def test_official_prior_is_incompatible_with_lane_conditioning(self):
        with self.assertRaisesRegex(ValueError, "official_prior requires LDM initial_scene"):
            self.model(scene_count_source="official_prior")


if __name__ == "__main__":
    unittest.main()
