"""Native SMART decoder contracts with small real AE/LDM checkpoints."""
from pathlib import Path
import tempfile
import pickle
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.core.autoencoder import AutoEncoder
from src.smart.scenario_dreamer.core.ldm import LDM
from src.smart.scenario_dreamer.data import attach_model_map, build_graph
from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder, _checkpoint
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.scenario_dreamer.core.data_helpers import normalize_scene
from src.smart.tokens.token_processor import TokenProcessor


def make_checkpoints(root):
    dataset = dict(max_num_agents=30, max_num_lanes=100, num_points_per_lane=20,
                   num_map_ids=2, fov=64, min_speed=0, max_speed=114.088,
                   min_length=-.098, max_length=22.929, min_width=.096, max_width=12.527,
                   min_lane_x=-32, max_lane_x=32, min_lane_y=-32, max_lane_y=32,
                   agent_latents_mean=.0138, agent_latents_std=1.0088,
                   lane_latents_mean=.0123, lane_latents_std=1.0487)
    ae_config = dict(hidden_dim=32, agent_hidden_dim=16, num_encoder_blocks=1, num_decoder_blocks=1,
                     lane_attr=2, num_heads=4, dropout=0, dim_f=64, state_dim=7,
                     num_agent_types=3, lane_conn_attr=6, num_lane_types=0,
                     agent_num_heads=4, agent_dim_f=32, lane_conn_hidden_dim=8,
                     lane_latent_dim=24, agent_latent_dim=8, kl_weight=.01,
                     lane_weight=10, lane_conn_weight=10, cond_dis_weight=.1,
                     num_points_per_lane=20, max_num_lanes=100)
    ae = AutoEncoder(OmegaConf.create(ae_config))
    cfg = OmegaConf.create(dict(
        model=dict(hidden_dim=64, num_heads=4, agent_hidden_dim=32, agent_num_heads=4,
                   num_factorized_dit_blocks=1, lane_latent_dim=24, agent_latent_dim=8,
                   dropout=0, label_dropout=.1, num_l2l_blocks=1, n_diffusion_timesteps=4,
                   lane_sampling_temperature=.75, diffusion_clip=5),
        dataset=dataset, train=dict(loss_type="l2", lane_weight=10, guidance_scale=4, ema_decay=.99),
    ))
    ldm = LDM(cfg)
    ema = ExponentialMovingAverage(ldm.parameters(), decay=.99)
    ae_path, ldm_path = root / "ae.ckpt", root / "ldm.ckpt"
    torch.save(dict(hyper_parameters={"cfg": OmegaConf.create(dict(model=ae_config, dataset=dataset))},
                    state_dict={"model." + k: v for k, v in ae.state_dict().items()}), ae_path)
    state = {"diff_model." + k: v for k, v in ldm.state_dict().items()}
    state.update({"autoencoder.model." + k: v for k, v in ae.state_dict().items()})
    torch.save(dict(hyper_parameters={"cfg": cfg}, state_dict=state, ema_state_dict=ema.state_dict(), global_step=19), ldm_path)
    return ae_path, ldm_path


def inputs():
    batch = torch.tensor([0, 0, 1, 1, 1])
    lane = torch.stack((torch.zeros(20), torch.linspace(-20, 20, 20)), -1)
    lanes = torch.stack((lane, lane + torch.tensor([4., 0]), lane, lane + torch.tensor([4., 0])))
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]])
    agent = dict(batch=batch, num_graphs=2, ego_mask=torch.tensor([False, True, False, False, True]),
                 initial_pos=torch.tensor([[1., 3.], [0., 0.], [100., 0.], [101., 2.], [100., 1.]]),
                 initial_heading=torch.zeros(5), local_vel=torch.ones(5, 2),
                 shape=torch.tensor([[4.8, 2., 1.6]]).repeat(5, 1), type=torch.zeros(5, dtype=torch.long),
                 tokenized_map={})
    agent["sd_map"] = dict(lanes=lanes, batch=torch.tensor([0, 0, 1, 1]), edges=edges,
                           types=torch.tensor([5, 0, 0, 5, 5, 0, 0, 5]),
                           center=torch.tensor([[0., 0.], [100., 1.]], dtype=torch.float64),
                           angle=torch.tensor([torch.pi / 2, torch.pi / 2], dtype=torch.float64),
                           lg_type=torch.zeros(2, dtype=torch.long))
    return agent


def official_scene(kind=0):
    lane = torch.stack((torch.zeros(20), torch.linspace(-20, -1, 20)), -1)
    states = torch.tensor([[0., 0., 5., 0., 1., 4.5, 2.],
                           [3., -4., 2., 1., 0., 4., 1.8],
                           [-2., 5., 0., -1., 0., 4., 2.]], dtype=torch.float64)
    return dict(agent_states=states.numpy(), agent_types=torch.eye(3).numpy(),
                num_agents=3, num_lanes=2, lg_type=kind, scene_timestep=37,
                road_points=torch.stack((lane, -lane)).double().numpy(),
                edge_index_lane_to_lane=torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]]),
                road_connection_types=torch.nn.functional.one_hot(torch.tensor([5, 2, 1, 5]), 6).numpy())


class ScenarioDreamerInitDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.ae, cls.ldm = make_checkpoints(Path(cls.temp.name))
        cls.scratch_config = torch.load(cls.ldm, weights_only=False)["hyper_parameters"]["cfg"]
        cls.ae_scratch_config = torch.load(cls.ae, weights_only=False)["hyper_parameters"]["cfg"].model
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def decoder(self, **options):
        options.setdefault("ldm_checkpoint", self.ldm)
        options.setdefault("ae_checkpoint", self.ae)
        return ScenarioDreamerInitDecoder(self.processor, map_source="exact", **options)

    def scratch_decoder(self):
        # Override only architecture, leaving all other settings to the bundled preset.
        return self.decoder(ldm_checkpoint=None, ldm_config={"model": self.scratch_config.model})

    def ae_decoder(self):
        return self.decoder(training_stage="autoencoder", ldm_checkpoint=None,
                            ae_checkpoint=None, ae_config=self.ae_scratch_config)

    def test_ae_scratch_requires_no_checkpoint_or_ldm_allocation(self):
        with patch("src.smart.scenario_dreamer.decoder._checkpoint", side_effect=AssertionError("no weights")), \
             patch("src.smart.scenario_dreamer.decoder.LDM", side_effect=AssertionError("no LDM")):
            model = self.ae_decoder()
        self.assertIsNone(model.diff_model)
        self.assertIsNone(model.ema)
        self.assertEqual(model.checkpoint_step, 0)
        self.assertFalse(model.use_ema)
        self.assertTrue(all(p.requires_grad for p in model.autoencoder.parameters()))
        model.update_ema()  # SMART's optimizer hook is also valid for AE-only training.
        self.assertFalse(model.eval().autoencoder.training)
        self.assertTrue(model.train().autoencoder.training)

    def test_ae_training_updates_encoder_decoder_and_partition_count_head(self):
        model = self.ae_decoder().train()
        _, agent = self.official_inputs((0, 1))
        graph, _, _, _ = model._build_graph(agent)
        self.assertEqual(graph.num_lanes_after_origin.tolist(), [0, 1])
        before = [p.detach().clone() for p in model.autoencoder.parameters()]
        losses = model(agent)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        self.assertGreater(losses["lane_cond_dis_loss"].item(), 0)
        losses["loss"].backward()
        for module in (model.autoencoder.encoder, model.autoencoder.decoder,
                       model.autoencoder.encoder.pred_lane_cond_dis,
                       model.autoencoder.decoder.pred_lane_conn):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        torch.optim.AdamW(model.autoencoder.parameters(), lr=1e-3).step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.autoencoder.parameters())))

    def test_ae_nonpartitioned_validation_losses_are_finite(self):
        model = self.ae_decoder().eval()
        with torch.no_grad():
            losses = model(inputs())
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        self.assertEqual(losses["lane_cond_dis_loss"].item(), 0)
        self.assertEqual(losses["lane_cond_dis_acc"].item(), 0)
        self.assertIn("kl_loss", losses)
        self.assertIn("lane_conn_loss", losses)

    def test_ae_finetuning_loads_official_weights_without_freezing(self):
        with patch("src.smart.scenario_dreamer.decoder._checkpoint", wraps=_checkpoint) as load:
            model = self.decoder(training_stage="autoencoder", ldm_checkpoint=None)
        load.assert_called_once_with(self.ae)
        saved = torch.load(self.ae, weights_only=False)["state_dict"]
        for key, value in model.autoencoder.state_dict().items():
            torch.testing.assert_close(value, saved["model." + key])
        self.assertTrue(all(p.requires_grad for p in model.autoencoder.parameters()))

    def test_ae_resume_and_transfer_to_frozen_ldm_encoder(self):
        model = self.ae_decoder()
        model(inputs())["loss"].backward()
        torch.optim.SGD(model.autoencoder.parameters(), lr=.01).step()
        path = Path(self.temp.name) / "smart_ae.ckpt"
        torch.save({"state_dict": {"encoder.init_decoder." + k: v for k, v in model.state_dict().items()},
                    "hyper_parameters": {"model_config": {}}, "global_step": 3}, path)
        restored = self.ae_decoder()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertIsNone(restored.ema)
        from_smart = self.decoder(training_stage="autoencoder", ldm_checkpoint=None, ae_checkpoint=path)
        ldm = self.decoder(ldm_checkpoint=None, ldm_config={"model": self.scratch_config.model},
                           ae_checkpoint=path)
        self.assertEqual(from_smart.checkpoint_step, 3)
        self.assertEqual(ldm.checkpoint_step, 0)
        self.assertFalse(any(p.requires_grad for p in ldm.autoencoder.parameters()))
        for other in (restored, from_smart, ldm):
            for key, value in model.autoencoder.state_dict().items():
                torch.testing.assert_close(other.autoencoder.state_dict()[key], value)
        with self.assertRaisesRegex(ValueError, "same training_stage"):
            ldm.set_extra_state(model.get_extra_state())
        self.assertTrue(torch.isfinite(ldm(inputs())["loss"]))

    def test_ae_validation_test_and_training_use_existing_smart_loss_hooks(self):
        from src.smart.model.smart_gail import SMART_GAIL
        from lightning import LightningModule
        wrapper = SMART_GAIL.__new__(SMART_GAIL)
        LightningModule.__init__(wrapper)
        wrapper.encoder = torch.nn.Module()
        wrapper.encoder.init_decoder = self.ae_decoder().eval()
        wrapper.token_processor = self.processor
        batch, agent = self.official_inputs((0, 0))
        with patch.object(wrapper, "log") as log, \
             patch.object(wrapper, "_rollouts", side_effect=AssertionError("AE must not generate diffusion samples")):
            wrapper.on_validation_epoch_start()
            val_loss = wrapper.validation_step(batch, 0)
            wrapper.on_validation_epoch_end()
            wrapper.on_test_epoch_start()
            test_loss = wrapper.test_step(batch, 0)
            wrapper.on_test_epoch_end()
        self.assertTrue(torch.isfinite(val_loss))
        self.assertTrue(torch.isfinite(test_loss))
        names = [call.args[0] for call in log.call_args_list]
        self.assertIn("val/scenario_dreamer/autoencoder/loss", names)
        self.assertIn("test/scenario_dreamer/autoencoder/loss", names)
        model = wrapper.encoder.init_decoder.train()
        losses = model(agent)
        with patch.object(wrapper, "_log_train") as log_train:
            actual = wrapper._initial_prediction_loss({"initial_logit": losses}, agent, losses["loss"])
        self.assertIs(actual, losses["loss"])
        self.assertIn("train/scenario_dreamer/autoencoder/loss", [call.args[0] for call in log_train.call_args_list])

    def test_ae_stage_rejects_incompatible_configuration(self):
        for options, message in (
            ({"training_stage": "joint"}, "training_stage must be"),
            ({"training_stage": "autoencoder"}, "Autoencoder training requires"),
            ({"training_stage": "autoencoder", "ldm_checkpoint": None,
              "ldm_config": {"model": {"hidden_dim": 64}}}, "Autoencoder training requires"),
            ({"ldm_checkpoint": None, "ae_checkpoint": None}, "trained ae_checkpoint"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.decoder(**options)

    def test_scratch_loads_only_ae_and_initializes_fresh_ema(self):
        with patch("src.smart.scenario_dreamer.decoder._checkpoint", wraps=_checkpoint) as load:
            model = self.scratch_decoder()
        load.assert_called_once_with(self.ae)
        self.assertEqual(model.checkpoint_step, 0)
        self.assertEqual(model.ema.num_updates, 0)
        self.assertEqual(model.ema.decay, model.cfg.train.ema_decay)
        for shadow, parameter in zip(model.ema.shadow_params, model.diff_model.parameters()):
            torch.testing.assert_close(shadow, parameter)
        ae_state = torch.load(self.ae, weights_only=False)["state_dict"]
        for key, value in model.autoencoder.state_dict().items():
            torch.testing.assert_close(value, ae_state["model." + key])
        self.assertFalse(any(p.requires_grad for p in model.autoencoder.parameters()))
        pretrained = torch.load(self.ldm, weights_only=False)["state_dict"]
        self.assertTrue(any(not torch.equal(p, pretrained["diff_model." + key])
                            for key, p in model.diff_model.named_parameters()))

    def test_scratch_training_and_smart_state_resume_without_ldm_checkpoint(self):
        model = self.scratch_decoder().train()
        before = [p.detach().clone() for p in model.diff_model.parameters()]
        loss = model(inputs())["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.autoencoder.parameters()))
        self.assertFalse(model.autoencoder.training)
        torch.optim.SGD(model.diff_model.parameters(), lr=.01).step()
        model.update_ema()
        self.assertEqual(model.ema.num_updates, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.diff_model.parameters())))
        path = Path(self.temp.name) / "scratch_resume.pt"
        torch.save(model.state_dict(), path)
        restored = self.scratch_decoder()
        restored.load_state_dict(torch.load(path, weights_only=False), strict=True)
        self.assertEqual(restored.checkpoint_step, 0)
        self.assertEqual(restored.ema.num_updates, model.ema.num_updates)
        for a, b in zip(restored.diff_model.parameters(), model.diff_model.parameters()):
            torch.testing.assert_close(a, b)
        for a, b in zip(restored.ema.shadow_params, model.ema.shadow_params):
            torch.testing.assert_close(a, b)
        torch.manual_seed(123)
        expected = model.eval()(inputs())
        torch.manual_seed(123)
        actual = restored.eval()(inputs())
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)

    def test_explicit_checkpoint_restores_ldm_embedded_ae_and_ema(self):
        saved = torch.load(self.ldm, weights_only=False)
        # Distinguish the saved EMA and embedded AE from freshly initialized values.
        saved["ema_state_dict"]["shadow_params"][0].add_(2)
        ae_key = next(k for k in saved["state_dict"] if k.startswith("autoencoder.model."))
        saved["state_dict"][ae_key].add_(1)
        path = Path(self.temp.name) / "distinct_pretrained.ckpt"
        torch.save(saved, path)
        model = self.decoder(ldm_checkpoint=path)
        self.assertEqual(model.checkpoint_step, 19)
        for key, value in model.diff_model.state_dict().items():
            torch.testing.assert_close(value, saved["state_dict"]["diff_model." + key])
        for key, value in model.autoencoder.state_dict().items():
            torch.testing.assert_close(value, saved["state_dict"]["autoencoder.model." + key])
        for actual, expected in zip(model.ema.shadow_params, saved["ema_state_dict"]["shadow_params"]):
            torch.testing.assert_close(actual, expected)

    def test_explicit_missing_ldm_checkpoint_does_not_fall_back_to_scratch(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing Scenario Dreamer checkpoint"):
            self.decoder(ldm_checkpoint=Path(self.temp.name) / "missing.ckpt")

    def test_invalid_scratch_config_fails_before_building_ldm(self):
        with self.assertRaisesRegex(ValueError, "ldm_checkpoint=null"):
            self.decoder(ldm_config={"model": {"hidden_dim": 64}})
        with self.assertRaisesRegex(ValueError, "agent_latent_dim must match the AE"):
            self.decoder(ldm_checkpoint=None, ldm_config={"model": {"agent_latent_dim": 16}})

    def test_real_core_training_backward_and_frozen_ae(self):
        model = self.decoder().train()
        loss = model(inputs())
        self.assertEqual(set(loss), {"loss", "agent_loss", "lane_loss"})
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.diff_model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.autoencoder.parameters()))
        self.assertFalse(model.autoencoder.training)
        before = model.ema.num_updates
        torch.optim.SGD(model.diff_model.parameters(), lr=.01).step()
        model.update_ema()
        self.assertEqual(model.ema.num_updates, before + 1)

    def test_inference_uses_latent_shapes_and_smart_five_tuple(self):
        model = self.decoder().eval()
        agent = inputs()
        graph, _, _, _ = model._encode(agent)
        self.assertEqual(graph["agent"].x.shape, (5, 8))
        self.assertEqual(graph["lane"].x.shape, (4, 24))
        with torch.no_grad():
            result = model(agent)
        self.assertEqual([tuple(x.shape) for x in result], [(5, 1, 2), (5, 1), (5, 1), (5, 2), (5, 2)])
        self.assertTrue(all(torch.isfinite(value).all() for value in result))
        self.assertTrue(torch.isin(agent["type"], torch.tensor([0, 1, 2])).all())
        self.assertEqual(agent["token_traj_all"].shape[0], 5)

    def test_saved_ema_restored_by_normal_module_state_dict(self):
        model = self.decoder()
        with torch.no_grad():
            next(model.diff_model.parameters()).add_(.2)
        model.update_ema()
        path = Path(self.temp.name) / "resume.pt"
        torch.save(model.state_dict(), path)
        restored = self.decoder()
        restored.load_state_dict(torch.load(path, weights_only=False), strict=True)
        self.assertEqual(restored.ema.num_updates, model.ema.num_updates)
        for a, b in zip(restored.ema.shadow_params, model.ema.shadow_params):
            torch.testing.assert_close(a, b)

    def test_lane_conditioning_has_no_gt_agent_geometry_or_type_leak(self):
        model = self.decoder().eval()
        a, b = inputs(), inputs()
        b["initial_pos"] += 7
        b["type"].fill_(1)
        with torch.no_grad():
            graph_a = model._encode(a)[0]
            graph_b = model._encode(b)[0]
        torch.testing.assert_close(graph_a["lane"].latents, graph_b["lane"].latents)

    def test_ego_order_and_global_output_velocity_are_restored(self):
        model = self.decoder()
        agent = inputs()
        rows = torch.tensor([1, 0, 4, 2, 3])
        states = torch.tensor([[1., 2., 3., 1., 0., 4., 2.]]).repeat(5, 1)
        centers = torch.tensor([[10., 20.], [30., 40.]], dtype=torch.float64)
        angles = torch.tensor([0., torch.pi / 2], dtype=torch.float64)
        result = model._smart_output(states, torch.zeros(5, dtype=torch.long), rows,
                                     torch.tensor([0, 0, 1, 1, 1]), centers, angles, agent)
        torch.testing.assert_close(result[0][:2, 0], torch.tensor([[11., 22.], [11., 22.]]))
        torch.testing.assert_close(result[0][2:, 0], torch.tensor([[32., 39.]]).repeat(3, 1))
        torch.testing.assert_close(result[4][:2], torch.tensor([[3., 0.]]).repeat(2, 1))
        torch.testing.assert_close(result[4][2:], torch.tensor([[0., -3.]]).repeat(3, 1), atol=1e-6, rtol=0)

    def test_lane_relation_direction_priority_and_batch_offsets(self):
        # Lane 0 lists lane 1 as its predecessor: edge 1 -> 0 is "pred".
        adjacency = torch.tensor([[0, 1], [0, 0]])
        graph = dict(road_points=torch.zeros(2, 20, 2), pre_adj=adjacency,
                     suc_adj=adjacency.T, left_adj=adjacency, right_adj=adjacency.T)
        record = {}
        attach_model_map(record, dict(lg_type=0, graphs={"regular": graph}))
        data = HeteroData(record)
        torch.testing.assert_close(data["sd_lane", "to", "sd_lane"].type,
                                   torch.tensor([5, 2, 1, 5]))
        batch = Batch.from_data_list([data, data.clone()])
        edges = batch["sd_lane", "to", "sd_lane"].edge_index
        self.assertTrue(torch.equal(batch["sd_lane"].batch[edges[0]], batch["sd_lane"].batch[edges[1]]))
        self.assertEqual(int(edges.max()), 3)

    def official_inputs(self, kinds=(0, 0)):
        graphs = [HeteroData(adapt_preprocessed_scene(official_scene(k), f"scene-{i}.pkl"))
                  for i, k in enumerate(kinds)]
        data = Batch.from_data_list(graphs)
        tokens, agent = self.processor(data)
        agent["tokenized_map"] = tokens
        return data, agent

    def test_official_snapshot_preserves_features_edges_and_frame(self):
        data, agent = self.official_inputs()
        self.assertEqual(agent["ego_mask"].tolist(), [False, False, True, False, False, True])
        self.assertEqual(agent["type"].tolist(), [1, 2, 0, 1, 2, 0])
        self.assertTrue(agent["initial_scene_only"])
        self.assertNotIn("gt_z_raw", agent)
        self.assertEqual(data.scene_timestep.tolist(), [37, 37])
        model = self.decoder()
        graph, rows, centers, angles = build_graph(agent, {}, model.cfg.dataset, map_source="exact")
        keys = ("fov", "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
                "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y")
        expected, _ = normalize_scene(agent["sd_states"].numpy().copy(), official_scene()["road_points"].copy(),
                                      **{k: model.cfg.dataset[k] for k in keys})
        torch.testing.assert_close(graph["agent"].x[torch.argsort(rows)], torch.from_numpy(expected).float())
        self.assertEqual(int(torch.count_nonzero(centers)), 0)
        self.assertEqual(int(torch.count_nonzero(angles)), 0)
        # Explicit pickle edge order is intentionally not lexicographic.
        original = official_scene()
        torch.testing.assert_close(data["sd_lane", "to", "sd_lane"].edge_index[:, :4], original["edge_index_lane_to_lane"])

    def test_official_partitioned_training_preserves_conditioning_masks(self):
        _, agent = self.official_inputs((0, 1))
        model = self.decoder().train()
        graph, rows, _, _ = build_graph(agent, {}, model.cfg.dataset, map_source="exact")
        self.assertEqual(graph.lg_type.tolist(), [0, 1])
        self.assertEqual(graph["agent"].partition_mask.tolist(), [False, False, False, True, True, False])
        self.assertEqual(graph["lane"].partition_mask.tolist(), [False, False, True, False])
        for source, _, target in graph.edge_types:
            edge = graph[source, "to", target]
            expected = graph[source].partition_mask[edge.edge_index[0]] == graph[target].partition_mask[edge.edge_index[1]]
            torch.testing.assert_close(edge.encoder_mask, expected)
        encoded, _, _, _ = model._encode(agent)
        torch.testing.assert_close(encoded["agent"].partition_mask, graph["agent"].partition_mask)
        loss = model(agent)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.eval()
        model(agent)
        self.assertEqual(agent["generated_map"]["lg_type"].tolist(), [0, 1])

    def test_official_inference_returns_one_snapshot_without_ar_history(self):
        from src.smart.modules.smart_decoder import SMARTDecoder
        _, agent = self.official_inputs()
        wrapper = SMARTDecoder.__new__(SMARTDecoder)
        torch.nn.Module.__init__(wrapper)
        wrapper.init_decoder_name = "scenario_dreamer"
        wrapper.init_decoder = self.decoder().eval()
        out = wrapper.inference(agent)
        self.assertEqual(out["pred_traj_10hz"].shape, (6, 1, 2))
        self.assertEqual(out["pred_head_10hz"].shape, (6, 1))
        self.assertEqual(out["initial_local_vel"].shape, (6, 2))
        self.assertTrue(torch.isfinite(out["pred_traj_10hz"]).all())
        self.assertEqual(out["generated_map"]["road_points"].shape, (4, 20, 2))
        self.assertEqual(out["generated_map"]["road_connection_types"].shape, (8, 6))

    def test_official_dataset_uses_manifest_order_and_rejects_missing_files(self):
        from src.smart.datasets.scalable_dataset import MultiDataset
        from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.pkl", "b.pkl", "unused.pkl"):
                with (root / name).open("wb") as handle:
                    pickle.dump(official_scene(), handle)
            manifest = root / "selected.pickle"
            with manifest.open("wb") as handle:
                pickle.dump({"files": ["b.pkl", "a.pkl"]}, handle)
            dataset = MultiDataset(str(root), WaymoTargetBuilderVal(),
                                   scenario_dreamer_preprocessed=True, sample_list=str(manifest))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["scenario_dreamer_cache_file"], "b.pkl")
            self.assertEqual(dataset[1]["scenario_dreamer_cache_file"], "a.pkl")
            with manifest.open("wb") as handle:
                pickle.dump({"files": ["missing.pkl"]}, handle)
            with self.assertRaisesRegex(FileNotFoundError, "Missing 1"):
                MultiDataset(str(root), WaymoTargetBuilderVal(), sample_list=str(manifest))

    def test_joint_generation_ignores_gt_latents_and_keeps_batched_lane_graph(self):
        model = self.decoder().eval()
        a, b = inputs(), inputs()
        b["sd_map"]["lanes"] *= -2
        b["sd_map"]["types"].fill_(0)
        b["initial_pos"] += 7
        b["type"].fill_(1)
        with patch.object(model.autoencoder, "forward_encoder", side_effect=AssertionError("GT must not be encoded")):
            torch.manual_seed(71)
            out_a = model(a)
            torch.manual_seed(71)
            out_b = model(b)
        for left, right in zip(out_a, out_b):
            torch.testing.assert_close(left, right)
        for key in ("road_points", "road_connection_types", "edge_index_lane_to_lane", "batch"):
            torch.testing.assert_close(a["generated_map"][key], b["generated_map"][key])
        graph = a["generated_map"]
        self.assertEqual(graph["coordinate_frame"], "sd_local")
        edges = graph["edge_index_lane_to_lane"]
        torch.testing.assert_close(graph["batch"][edges[0]], graph["batch"][edges[1]])
        self.assertTrue(torch.all(graph["road_connection_types"].sum(-1) == 1))

    def test_lane_conditioned_mode_preserves_reference_map_contract(self):
        model = self.decoder(generation_mode="lane_conditioned").eval()
        agent = inputs()
        agent["generated_map"] = {"stale": True}
        result = model(agent)
        self.assertEqual(len(result), 5)
        self.assertNotIn("generated_map", agent)

    def test_exact_map_and_count_requirements(self):
        model = self.decoder()
        a = inputs(); del a["sd_map"]
        with self.assertRaisesRegex(ValueError, "map_source=exact"):
            build_graph(a, {}, model.cfg.dataset, map_source="exact")
        a = inputs(); a["ego_mask"].fill_(False)
        with self.assertRaisesRegex(ValueError, "one ego"):
            build_graph(a, {}, model.cfg.dataset)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device contract")
    def test_graph_metadata_moves_to_cuda_with_nodes(self):
        model = self.decoder()
        a = inputs()
        for key, value in a.items():
            if torch.is_tensor(value): a[key] = value.cuda()
        a["sd_map"] = {key: value.cuda() for key, value in a["sd_map"].items()}
        graph, _, _, _ = build_graph(a, {}, model.cfg.dataset)
        self.assertEqual(graph.map_id.device.type, "cuda")
        self.assertEqual(graph.num_agents.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
