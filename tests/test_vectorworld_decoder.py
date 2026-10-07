"""Exercise real VectorWorld losses/sampling through the SMART decoder contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from src.smart.vectorworld.decoder import VectorWorldInitDecoder
from src.smart.vectorworld.checkpoints import public_config, validate_stats
from test_vectorworld_core import ae_config, gen_config
import test_vectorworld_data as data_fixtures

ROOT = Path(__file__).resolve().parents[1]

def configs(kind="flow"):
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "src/smart/vectorworld/waymo_flow.yaml"),
                          gen_config(kind, relational=kind != "meanflow"))
    cfg.dataset.agent_latents_mean = [0.] * 4
    cfg.dataset.agent_latents_std = [1.] * 4
    cfg.dataset.lane_latents_mean = [0.] * 4
    cfg.dataset.lane_latents_std = [1.] * 4
    ae = OmegaConf.create(dict(model=OmegaConf.to_container(ae_config()),
                              dataset=OmegaConf.to_container(cfg.dataset), train={}))
    return ae, cfg

class VectorWorldDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def processor(self):
        p = data_fixtures.VectorWorldDataTest.processor()
        p.shift = 5
        p.token_velocity_in_current_frame = lambda tokens, dt: torch.zeros(len(tokens), tokens.shape[1], 2)
        return p

    def agent(self, kinds=(0,), *, motion=True):
        fixture = data_fixtures.VectorWorldDataTest()
        agent = fixture.agents(kinds, motion=motion)[0]
        agent["tokenized_map"] = {}
        return agent

    def decoder(self, kind="flow", mode="joint", **kwargs):
        ae, cfg = configs(kind)
        from src.smart.vectorworld.core import AutoEncoder
        path = self.root / "ae.ckpt"
        if not path.exists():
            model = AutoEncoder(ae.model)
            torch.save({"state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                        "hyper_parameters": {"cfg": ae}}, path)
        return VectorWorldInitDecoder(self.processor(), ae_checkpoint=path,
            ldm_config=cfg, training_mode=mode, generation_mode=(
                "lane_conditioned" if mode == "lane_conditioned" else "initial_scene"), **kwargs)

    def test_vae_stage_real_gradient_and_validation_missing_motion(self):
        ae, _ = configs()
        decoder = VectorWorldInitDecoder(self.processor(), training_stage="autoencoder",
                                          ae_checkpoint=None, ae_config=ae)
        losses = decoder(self.agent((0, 1)))
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.autoencoder.parameters()))
        self.assertIsNone(decoder.diff_model)
        decoder.eval()
        with self.assertRaisesRegex(ValueError, "requires real trajectory"):
            decoder.autoencoder_loss(self.agent(motion=False))

    def test_all_generator_training_losses_and_frozen_ae(self):
        for kind in ("flow", "meanflow", "diffusion"):
            with self.subTest(kind=kind):
                decoder = self.decoder(kind)
                losses = decoder(self.agent((0, 1)))
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.diff_model.parameters()))
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in decoder.autoencoder.parameters()))
                self.assertFalse(decoder.autoencoder.training)

    def test_lane_conditioned_training_all_types_and_lane_weight_restored(self):
        for kind in ("flow", "meanflow", "diffusion"):
            with self.subTest(kind=kind):
                decoder = self.decoder(kind, "lane_conditioned")
                losses = decoder(self.agent((0, 1)))
                self.assertTrue(torch.isfinite(losses["loss"]))
                self.assertEqual(float(decoder.cfg.train.lane_weight), 10.)
                losses["loss"].backward()

    def test_lane_evaluation_full_only_and_five_tuple(self):
        decoder = self.decoder(mode="lane_conditioned").eval()
        agent = self.agent()
        result = decoder(agent)
        self.assertEqual(len(result), 5)
        self.assertEqual(tuple(result[0].shape), (3, 1, 2))
        self.assertEqual(tuple(result[4].shape), (3, 2))
        self.assertTrue(all(torch.isfinite(v).all() for v in result))
        self.assertNotIn("generated_map", agent)
        self.assertEqual(tuple(agent["generated_motion"].shape), (3, 12))
        with self.assertRaisesRegex(ValueError, "full non-partitioned lanes"):
            decoder(self.agent((1,)))

    def test_joint_inference_needs_no_motion_and_returns_generated_lanes(self):
        decoder = self.decoder().eval()
        agent = self.agent(motion=False)
        output = decoder(agent)
        self.assertEqual(len(output), 5)
        lanes = agent["generated_map"]
        self.assertEqual(lanes["coordinate_frame"], "sd_local")
        self.assertEqual(tuple(lanes["road_points"].shape), (2, 20, 2))
        self.assertEqual(tuple(lanes["road_connection_types"].shape), (4, 6))
        self.assertTrue(torch.isfinite(lanes["road_points"]).all())
        self.assertTrue((lanes["lg_type"] == 0).all())

    def test_missing_motion_rejected_for_ldm_training(self):
        decoder = self.decoder()
        with self.assertRaisesRegex(ValueError, "requires real trajectory"):
            decoder(self.agent(motion=False))

    def test_ema_update_and_strict_smart_checkpoint_roundtrip(self):
        decoder = self.decoder()
        first = next(decoder.diff_model.parameters())
        with torch.no_grad():
            first.add_(.01)
        decoder.update_ema()
        restored = self.decoder()
        restored.load_state_dict(decoder.state_dict(), strict=True)
        self.assertEqual(restored.ema.num_updates, decoder.ema.num_updates)
        for a, b in zip(restored.ema.shadow_params, decoder.ema.shadow_params):
            torch.testing.assert_close(a, b)
        self.assertEqual(restored.checkpoint_step, 1)

    def test_native_generator_uses_embedded_vae_config_weights_and_ema(self):
        decoder = self.decoder("meanflow")
        path = self.root / "native.ckpt"
        state = {"gen_model." + k: v for k, v in decoder.diff_model.state_dict().items()}
        state.update({"autoencoder.model." + k: v for k, v in decoder.autoencoder.state_dict().items()})
        torch.save({"state_dict": state, "hyper_parameters": {"cfg": decoder.cfg, "cfg_ae": decoder.ae_cfg},
                    "ema_state_dict": decoder.ema.state_dict(), "global_step": 123}, path)
        loaded = VectorWorldInitDecoder(self.processor(), ae_checkpoint=None, ldm_checkpoint=path)
        self.assertEqual(loaded.cfg.model.ldm_type, "meanflow")
        self.assertEqual(loaded.checkpoint_step, 123)
        self.assertEqual(loaded.report_options()["sampling_steps"], 1)
        for key, value in decoder.autoencoder.state_dict().items():
            torch.testing.assert_close(loaded.autoencoder.state_dict()[key], value)
        output = loaded.eval()(self.agent(motion=False))
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_smart_generator_checkpoint_as_pretrained_init(self):
        decoder = self.decoder()
        path = self.root / "smart.ckpt"
        torch.save({"state_dict": {"encoder.init_decoder." + k: v for k, v in decoder.state_dict().items()}}, path)
        loaded = VectorWorldInitDecoder(self.processor(), ae_checkpoint=None, ldm_checkpoint=path)
        self.assertEqual(loaded.cfg.model.ldm_type, "flow")
        torch.testing.assert_close(next(loaded.diff_model.parameters()), next(decoder.diff_model.parameters()))

    def test_independent_scene_counts_update_agent_rows(self):
        prior = self.root / "counts.npz"
        probabilities = np.zeros((2, 5, 5), dtype=np.float32)
        probabilities[:, 2, 2] = 1.
        np.savez(prior, probabilities=probabilities)
        decoder = self.decoder(scene_count_source="official_prior", count_prior_path=prior).eval()
        agent = self.agent(motion=False)
        output = decoder(agent)
        self.assertEqual(len(output[0]), 2)
        self.assertEqual(len(agent["batch"]), 2)
        self.assertEqual(int(agent["ego_mask"].sum()), 1)
        self.assertEqual(tuple(agent["generated_map"]["road_points"].shape), (2, 20, 2))
        self.assertEqual(decoder.sampling_report()["num_scenes"], 1)

    def test_public_config_does_not_retain_credentials(self):
        ae, cfg = configs()
        cfg.dataset.cos_secret_id = "private-id"
        cfg.dataset.cos_secret_key = "private-key"
        cfg.dataset.preprocess_dir = "/some/private/remote/path"
        cfg.model.autoencoder_path = "/external/path"
        saved = public_config(cfg)
        text = OmegaConf.to_yaml(saved)
        for value in ("private-id", "private-key", "/some/private/remote/path", "/external/path"):
            self.assertNotIn(value, text)

    def test_stats_dimension_finite_positive_validation(self):
        _, cfg = configs()
        validate_stats(cfg)
        cfg.dataset.agent_latents_std = [0.] * 4
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_stats(cfg)

    def test_experiment_configs_compose_without_external_runtime_paths(self):
        for name in ("vectorworld", "vectorworld_ae", "vectorworld_eval",
                     "vectorworld_meanflow", "vectorworld_diffusion",
                     "vectorworld_lane_conditioned", "vectorworld_lane_conditioned_eval"):
            with self.subTest(experiment=name), initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="run", overrides=[f"experiment={name}", "optimizer=vectorworld"])
                self.assertEqual(cfg.model.model_config.decoder.init_decoder, "vectorworld")
                self.assertEqual(cfg.model.model_config.optimizer, "scenario_dreamer")
                self.assertEqual(cfg.trainer.precision, "32-true")
                self.assertTrue(cfg.data.scenario_dreamer_preprocessed)
                self.assertNotIn("/home/ke/code/VectorWorld", OmegaConf.to_yaml(cfg))
                if "lane_conditioned" in name:
                    self.assertEqual(cfg.model.model_config.decoder.vectorworld.generation_mode, "lane_conditioned")
                    self.assertFalse(cfg.data.scenario_dreamer_train_non_partitioned_only)

if __name__ == "__main__":
    unittest.main()
