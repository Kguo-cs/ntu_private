"""Small CPU regressions for the vendored VectorWorld numerical core."""
import unittest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from src.smart.vectorworld.core import AutoEncoder, FlowLDM, LDM, MeanFlowLDM
from src.smart.vectorworld.core.utils.data_helpers import normalize_latents, unnormalize_latents


def ae_config():
    return OmegaConf.create(dict(
        hidden_dim=16, agent_hidden_dim=16, num_encoder_blocks=1,
        num_decoder_blocks=1, num_heads=2, agent_num_heads=2,
        dim_f=32, agent_dim_f=32, lane_conn_hidden_dim=8, dropout=0.,
        lane_attr=2, state_dim=7, motion_dim=12, lane_latent_dim=4,
        agent_latent_dim=4, num_agent_types=3, lane_conn_attr=6,
        num_lane_types=0, num_points_per_lane=20, max_num_lanes=4,
        kl_weight=.006, lane_weight=10., lane_conn_weight=10.,
        cond_dis_weight=.08, lane_endpoint_weight=1.,
        motion_loss_weight=1., motion_smooth_weight=.01,
        motion_collision_weight=.1, collision_margin=.1,
        motion_static_weight=2., motion_static_eps=.03,
        static_xy_weight=3., static_other_weight=1.,
        motion_num_points=6, motion_x_range=12., motion_y_range=6.,
        latent_noise_std=.05, fov=64., min_length=-.098,
        max_length=22.929, min_width=.096, max_width=12.527,
    ))


def gen_config(kind, *, relational=True):
    return OmegaConf.create(dict(
        model=dict(
            ldm_type=kind, hidden_dim=32, agent_hidden_dim=32,
            num_heads=2, agent_num_heads=2, num_factorized_dit_blocks=1,
            num_l2l_blocks=1, lane_latent_dim=4, agent_latent_dim=4,
            dropout=0., label_dropout=.1, n_diffusion_timesteps=4,
            lane_sampling_temperature=.75, diffusion_clip=5.,
            flow_num_steps=2, flow_solver='heun',
            use_rel_bias=relational, use_cross_rel_bias=relational,
            use_rel_gate=relational, use_gcf=relational,
            lane_rel_dim=8, agent_rel_dim=8, edge_dim=8,
            qk_norm=True, attn_logit_clip=30., gcf_var_scale=.15,
            meanflow_use_two_times=True, meanflow_training_mode='identity',
            meanflow_enable_jvp=True, meanflow_num_steps_eval=1,
            meanflow_tr_ratio=.3, meanflow_loss_p=.8, meanflow_loss_c=1e-3,
        ),
        dataset=dict(num_map_ids=2, max_num_lanes=4, max_num_agents=4),
        train=dict(loss_type='l2', lane_weight=10., guidance_scale=4.),
    ))


def graph_batch(*, latent=False):
    """Two complete, separate scene graphs with static and motion features."""
    data = HeteroData()
    batch = torch.tensor([0, 0, 1])
    data['agent'].num_nodes = 3
    data['lane'].num_nodes = 3
    data['agent'].batch = batch.clone()
    data['lane'].batch = batch.clone()
    data['agent'].x = torch.tensor([
        [-.5, 0., -.5, 1., 0., -.6, -.7],
        [.5, .2, -.2, 0., 1., -.5, -.6],
        [0., 0., -.4, 1., 0., -.6, -.7],
    ])
    data['agent'].type = torch.eye(3)
    # Zero physical motion maps to (1, 0), rather than normalized zeros.
    data['agent'].motion = torch.tensor([[1., 0.] * 6] * 3)
    data['lane'].x = torch.linspace(-.9, .9, 120).reshape(3, 20, 2)
    edges = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
    for relation in [('agent', 'to', 'agent'), ('lane', 'to', 'lane'), ('lane', 'to', 'agent')]:
        data[relation].edge_index = edges.clone()
        data[relation].encoder_mask = torch.ones(5, dtype=torch.bool)
    data['lane', 'to', 'lane'].type = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 0, 0]), 6
    ).float()
    data['agent'].partition_mask = torch.tensor([True, False, False])
    data['lane'].partition_mask = torch.tensor([True, False, False])
    data.lg_type = torch.tensor([1, 0])
    data.map_id = torch.tensor([0, 1])
    data.num_agents = torch.tensor([2, 1])
    data.num_lanes = torch.tensor([2, 1])
    data.num_lanes_after_origin = torch.tensor([1, 0])
    data.batch_size = 2
    if latent:
        data['agent'].latents = torch.linspace(-1., 1., 12).reshape(3, 4)
        data['lane'].latents = torch.linspace(-.5, .5, 12).reshape(3, 4)
        # Native LDM graphs expose latent dimensions in .x for sampling shapes.
        data['agent'].x = data['agent'].latents.clone()
        data['lane'].x = data['lane'].latents.clone()
    return data


class VectorWorldCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_diffusion_training_and_conditioned_sampling(self):
        self.check_generator('diffusion', LDM)

    def test_flow_training_and_conditioned_sampling(self):
        self.check_generator('flow', FlowLDM)

    def test_meanflow_identity_jvp_training_and_conditioned_sampling(self):
        self.check_generator('meanflow', MeanFlowLDM)

    def test_motion_vae_loss_backprop_and_decoder_contract(self):
        torch.manual_seed(42)
        model = AutoEncoder(ae_config())
        data = graph_batch()
        loss = model.loss(data)
        assert torch.isfinite(loss['loss'])
        loss['loss'].backward()
        gate_grads = [p.grad for p in model.encoder.agent_gate_mlp.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gate_grads)
        assert any(g.abs().sum() > 0 for g in gate_grads)
        model.eval()
        with torch.no_grad():
            stats = model.forward_encoder(data, return_stats=True)
            assert len(stats) == 4
            assert all(x.shape == (3, 4) and torch.isfinite(x).all() for x in stats)
            agent, lane, prob = model.forward_encoder(data)
            assert prob.shape == (2, 5)
            torch.testing.assert_close(prob.sum(-1), torch.ones(2))
            full = model.forward_decoder_with_motion(agent, lane, data)
            static = model.forward_decoder(agent, lane, data)
            assert full[0].shape == (3, 19)
            assert full[1].shape == (3, 20, 2)
            assert full[2].shape == (3,)
            assert full[3] is None
            assert full[4].shape == (5, 6)
            torch.testing.assert_close(static[0], full[0][:, :7])
            assert model(data, return_lane_embeddings=True).shape == (3, 16)
        with torch.device('meta'):
            restored = AutoEncoder(ae_config())
        restored.load_state_dict(model.state_dict(), strict=True, assign=True)
        assert all(p.device.type == 'cpu' for p in restored.parameters())


    def check_generator(self, kind, model_cls):
        torch.manual_seed(7)
        # The published MeanFlow checkpoint predates relational/global-context blocks.
        model = model_cls(gen_config(kind, relational=kind != 'meanflow'))
        data = graph_batch(latent=True)
        loss = model.loss(data)
        assert torch.isfinite(loss['loss'])
        loss['loss'].backward()
        active = [p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert any(g.abs().sum() > 0 for g in active)
        model.eval()
        agent, lane = model(data, mode='lane_conditioned')
        assert agent.shape == (3, 4)
        assert torch.isfinite(agent).all()
        torch.testing.assert_close(lane, data['lane'].latents, rtol=0, atol=0)
        agent, lane = model(data, mode='train')
        torch.testing.assert_close(agent[0], data['agent'].latents[0], rtol=0, atol=0)
        torch.testing.assert_close(lane[0], data['lane'].latents[0], rtol=0, atol=0)
        with torch.device('meta'):
            restored = model_cls(gen_config(kind, relational=kind != 'meanflow'))
        restored.load_state_dict(model.state_dict(), strict=True, assign=True)
        assert all(p.device.type == 'cpu' for p in restored.parameters())
        assert all(b.device.type == 'cpu' for b in restored.buffers())


    def test_per_dimension_latent_statistics_roundtrip_and_invalid_dimension(self):
        a = torch.arange(12.).reshape(3, 4)
        l = torch.arange(16.).reshape(4, 4)
        statistics = ([1., 2., 3., 4.], [2., 3., 4., 5.], [0., 1., 2., 3.], [1., 2., 3., 4.])
        normalized = normalize_latents(a, l, *statistics)
        restored = unnormalize_latents(*normalized, *statistics)
        torch.testing.assert_close(restored[0], a)
        torch.testing.assert_close(restored[1], l)
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            normalize_latents(a, l, [0., 0.], [1., 1.], 0., 1.)

if __name__ == '__main__':
    unittest.main()
