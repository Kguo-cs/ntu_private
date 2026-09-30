"""Native SMART decoder contracts with small real AE/LDM checkpoints."""
from pathlib import Path
import tempfile
import pickle
import unittest

import torch
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.core.autoencoder import AutoEncoder
from src.smart.scenario_dreamer.core.ldm import LDM
from src.smart.scenario_dreamer.data import attach_model_map, build_graph
from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
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
        cls.processor = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl",
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       OmegaConf.create(dict(num_k=1, temp=1.)),
                                       pred_init=True, learn_init=True, scenario_dreamer_init=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def decoder(self):
        return ScenarioDreamerInitDecoder(self.processor, ae_checkpoint=self.ae,
                                          ldm_checkpoint=self.ldm, map_source="exact")

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
        with self.assertRaisesRegex(ValueError, "partitioned scenes are supported for training"):
            model(agent)

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
