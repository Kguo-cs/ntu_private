"""Posterior moments and reproducible scene selection for fresh VectorWorld AEs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import pickle
import sys
import unittest

import torch
from torch import nn
from torch_geometric.data import HeteroData, Batch
from omegaconf import OmegaConf

from src import vectorworld_latent_stats as stats
from src.smart.vectorworld.core import AutoEncoder
from src.smart.vectorworld.data import build_graph
from src.smart.vectorworld.checkpoints import validate_stats
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from test_vectorworld_core import ae_config
import test_vectorworld_data as data_fixtures

ROOT = Path(__file__).resolve().parents[1]


class FixtureProcessor(nn.Module):
    """Use actual native snapshot tokenization without loading token libraries."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.processor = data_fixtures.VectorWorldDataTest.processor()

    def forward(self, batch):
        return TokenProcessor.process_data(self.processor, batch)


class MomentAE:
    def __init__(self, owner):
        self.owner = owner

    def forward_encoder(self, graph, return_stats=True):
        self.owner.order.extend(graph.scene_timestep.tolist())
        a = graph['sd_agent'].x[:, 0].double()
        l = graph['sd_lane'].x[:, 0, 0].double()
        return (torch.stack((a, 2 * a), -1), torch.stack((l, 3 * l, -l), -1),
                torch.log(torch.tensor([4., 9.], dtype=torch.float64)).expand(len(a), -1),
                torch.log(torch.tensor([1., 4., 16.], dtype=torch.float64)).expand(len(l), -1))


class MomentDecoder:
    def __init__(self, *args, **kwargs):
        self.ae_config = OmegaConf.create(dict(agent_latent_dim=2, lane_latent_dim=3))
        self.ae_cfg = OmegaConf.create(dict(dataset=dict(
            max_num_agents=9, max_num_lanes=4, num_points_per_lane=20,
            num_map_ids=2, fov=80., min_speed=-2., max_speed=100.,
            min_length=.1, max_length=10., min_width=.2, max_width=4.,
            min_lane_x=-40., max_lane_x=40., min_lane_y=-40., max_lane_y=40.,
            motion=dict(enabled=True, dim=12),
            agent_latents_mean=[999.], agent_latents_std=[999.])))
        self.order = []
        self.autoencoder = MomentAE(self)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def _build_graph(self, agent):
        # Only aggregation is mocked; actual native states/motion batching is used.
        return agent['native_batch'], None, None, None


class AggregateProcessor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, batch):
        return {}, {'native_batch': batch}


class VectorWorldLatentStatsTest(unittest.TestCase):
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
        self.base = self.root / 'base.yaml'
        self.base_config = OmegaConf.load(ROOT / 'src/smart/vectorworld/waymo_flow.yaml')
        self.base_config.model.hidden_dim = 32
        self.base_config.model.flow_num_steps = 7
        OmegaConf.save(self.base_config, self.base)

    def write_scene(self, name, x_values, lane_values, timestep=0, *, motion=True):
        scene = data_fixtures.VectorWorldDataTest.scene(motion=motion)
        n, l = len(x_values), len(lane_values)
        scene['num_agents'], scene['num_lanes'] = n, l
        scene['agent_states'] = scene['agent_states'][:n].copy()
        scene['agent_types'] = scene['agent_types'][:n].copy()
        scene['agent_states'][:, 0] = x_values
        scene['road_points'] = scene['road_points'][:l].copy()
        scene['road_points'][:, :, 0] = torch.tensor(lane_values)[:, None].numpy()
        edge = torch.cartesian_prod(torch.arange(l), torch.arange(l)).T
        # torch.cartesian_prod(...).T has normal [2,1] shape for one lane too.
        scene['edge_index_lane_to_lane'] = edge
        labels = torch.where(edge[0] == edge[1], 5, 0)
        scene['road_connection_types'] = torch.nn.functional.one_hot(labels, 6).numpy()
        scene['scene_timestep'] = timestep
        if motion:
            scene['agent_motion_raw'] = scene['agent_motion_raw'][:n].copy()
            scene['agent_motion_is_static'] = scene['agent_motion_is_static'][:n].copy()
        path = self.root / name
        with path.open('wb') as handle:
            pickle.dump(scene, handle)
        return path

    def aggregate(self, **kwargs):
        decoder = MomentDecoder()
        with patch.object(stats, 'TokenProcessor', AggregateProcessor), \
                patch.object(stats, 'VectorWorldInitDecoder', return_value=decoder):
            cfg, report = stats.compute_stats(self.root / 'mock.ckpt', self.root,
                                              base_config=self.base, **kwargs)
        return cfg, report, decoder.order

    def test_analytic_posterior_moments_include_nonzero_variance_and_node_weighting(self):
        self.write_scene('a.pkl', [1., -2., 3.], [2., -1.], 11)
        self.write_scene('b.pkl', [10.], [4.], 22)
        cfg, report, order = self.aggregate(batch_size=1)
        a = torch.tensor([[1., 2.], [-2., -4.], [3., 6.], [10., 20.]], dtype=torch.float64)
        l = torch.tensor([[2., 6., -2.], [-1., -3., 1.], [4., 12., -4.]], dtype=torch.float64)
        for kind, mu, posterior_var in [('agent', a, torch.tensor([4., 9.])),
                                       ('lane', l, torch.tensor([1., 4., 16.]))]:
            expected_mean = mu.mean(0)
            expected_std = (mu.var(0, correction=0) + posterior_var).sqrt()
            torch.testing.assert_close(torch.tensor(list(cfg.dataset[f'{kind}_latents_mean']), dtype=torch.float64), expected_mean)
            torch.testing.assert_close(torch.tensor(list(cfg.dataset[f'{kind}_latents_std']), dtype=torch.float64), expected_std)
            self.assertFalse(torch.equal(expected_std, mu.std(0)))
        self.assertEqual(order, [11, 22])
        self.assertEqual((report['num_scenes'], report['agent_nodes'], report['lane_nodes']), (2, 4, 3))
        self.assertEqual(report['estimator'], 'per-dimension E[mu], E[mu^2+exp(logvar)]')
        validate_stats(cfg)

    def test_stats_are_batch_size_and_seed_independent_and_full_config_survives(self):
        self.write_scene('a.pkl', [1., -2., 3.], [2., -1.], 11)
        self.write_scene('b.pkl', [10.], [4.], 22)
        first, _, _ = self.aggregate(batch_size=1, seed=0)
        second, _, _ = self.aggregate(batch_size=8, seed=87)
        for kind in ('agent', 'lane'):
            for stat in ('mean', 'std'):
                self.assertEqual(first.dataset[f'{kind}_latents_{stat}'], second.dataset[f'{kind}_latents_{stat}'])
        self.assertEqual(first.model.agent_latent_dim, 2)
        self.assertEqual(first.model.lane_latent_dim, 3)
        self.assertEqual(first.model.hidden_dim, 32)
        self.assertEqual(first.model.flow_num_steps, 7)
        self.assertEqual(first.train, self.base_config.train)
        self.assertEqual(first.dataset.fov, 80.)
        self.assertEqual(first.dataset.max_num_agents, 9)
        self.assertEqual(first.dataset.max_num_lanes, 4)
        self.assertEqual(first.dataset.min_speed, -2.)
        self.assertEqual(first.dataset.motion.dim, 12)

    def test_sample_list_order_and_max_scene_selection_are_recorded(self):
        self.write_scene('a.pkl', [1.], [2.], 11)
        self.write_scene('b.pkl', [3.], [4.], 22)
        self.write_scene('c.pkl', [5.], [6.], 33)
        manifest = self.root / 'names.pkl'
        with manifest.open('wb') as handle:
            pickle.dump({'files': ['c.pkl', 'a.pkl', 'b.pkl']}, handle)
        _, report, order = self.aggregate(sample_list=manifest, max_scenes=2, batch_size=1)
        self.assertEqual(order, [33, 11])
        self.assertEqual(report['num_scenes'], 2)
        self.assertEqual(report['ordered_scene_names_sha256'], hashlib.sha256(b'c.pkl\na.pkl').hexdigest())
        self.assertEqual(report['sample_list'], str(manifest.resolve()))
        self.assertEqual(report['sample_list_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_invalid_manifest_and_missing_scene_are_rejected_before_model_loading(self):
        for names in (['../outside.pkl'], ['a.pkl', 'a.pkl'], [], 'a.pkl'):
            manifest = self.root / 'invalid.pkl'
            with manifest.open('wb') as handle:
                pickle.dump({'files': names}, handle)
            with self.subTest(names=names), patch.object(stats, 'TokenProcessor') as processor, \
                    self.assertRaisesRegex(ValueError, 'sample_list'):
                stats.compute_stats('missing.ckpt', self.root, sample_list=manifest)
            processor.assert_not_called()
        manifest = self.root / 'missing.pkl'
        with manifest.open('wb') as handle:
            pickle.dump({'files': ['absent.pkl']}, handle)
        with patch.object(stats, 'TokenProcessor') as processor, self.assertRaises(FileNotFoundError):
            stats.compute_stats('missing.ckpt', self.root, sample_list=manifest)
        processor.assert_not_called()

    def tiny_checkpoint(self):
        model_cfg = ae_config()
        model_cfg.agent_latent_dim, model_cfg.lane_latent_dim = 3, 5
        dataset = OmegaConf.merge(OmegaConf.load(ROOT / 'src/smart/vectorworld/waymo_vae.yaml').dataset,
                                  dict(max_num_agents=4, max_num_lanes=4))
        cfg = OmegaConf.create(dict(model=OmegaConf.to_container(model_cfg),
                                    dataset=OmegaConf.to_container(dataset), train={}))
        torch.manual_seed(37)
        model = AutoEncoder(model_cfg).eval()
        path = self.root / 'tiny.ckpt'
        torch.save({'state_dict': {'model.' + key: value for key, value in model.state_dict().items()},
                    'hyper_parameters': {'cfg': cfg}}, path)
        return path, model, cfg

    def test_real_tiny_native_motion_ae_checkpoint_matches_direct_posterior_moments(self):
        path, model, ae_cfg = self.tiny_checkpoint()
        scene_path = self.write_scene('scene.pkl', [0., 2., -1.], [2., -1.], 10)
        with scene_path.open('rb') as handle:
            scene = pickle.load(handle)
        batch = Batch.from_data_list([HeteroData(adapt_preprocessed_scene(scene, scene_path.name))])
        _, agent = FixtureProcessor()(batch)
        graph, _, _, _ = build_graph(agent, {}, ae_cfg.dataset, motion_dim=12)
        with torch.no_grad():
            am, lm, av, lv = model.forward_encoder(graph, return_stats=True)
        with patch.object(stats, 'TokenProcessor', FixtureProcessor):
            cfg, report = stats.compute_stats(path, self.root, max_scenes=1, batch_size=1, base_config=self.base)
        for kind, mu, logvar in [('agent', am, av), ('lane', lm, lv)]:
            mu, logvar = mu.double(), logvar.double()
            expected_mean = mu.mean(0)
            expected_std = ((mu.square() + logvar.exp()).mean(0) - expected_mean.square()).sqrt()
            torch.testing.assert_close(torch.tensor(list(cfg.dataset[f'{kind}_latents_mean']), dtype=torch.float64), expected_mean)
            torch.testing.assert_close(torch.tensor(list(cfg.dataset[f'{kind}_latents_std']), dtype=torch.float64), expected_std)
        self.assertEqual((cfg.model.agent_latent_dim, cfg.model.lane_latent_dim), (3, 5))
        self.assertEqual((report['agent_latent_dim'], report['lane_latent_dim']), (3, 5))
        self.assertEqual(cfg.dataset.max_num_lanes, 4)
        self.assertEqual(report['ae_checkpoint'], str(path.resolve()))
        validate_stats(cfg)

    def test_real_ae_statistics_refuse_snapshots_without_motion_supervision(self):
        path, _, _ = self.tiny_checkpoint()
        self.write_scene('no-motion.pkl', [0.], [2.], motion=False)
        with patch.object(stats, 'TokenProcessor', FixtureProcessor), self.assertRaisesRegex(ValueError, 'requires real trajectory'):
            stats.compute_stats(path, self.root, max_scenes=1)

    def test_cli_writes_full_ldm_yaml_and_metadata_json(self):
        self.write_scene('a.pkl', [1.], [2.])
        output = self.root / 'out' / 'new_ldm.yaml'
        decoder = MomentDecoder()
        with patch.object(stats, 'TokenProcessor', AggregateProcessor), \
                patch.object(stats, 'VectorWorldInitDecoder', return_value=decoder), \
                patch.object(sys, 'argv', ['stats', '--ae-checkpoint', str(self.root / 'mock.ckpt'),
                    '--data-dir', str(self.root), '--output', str(output), '--base-config', str(self.base), '--max-scenes', '1']):
            stats.main()
        cfg = OmegaConf.load(output)
        report = json.loads(output.with_suffix('.json').read_text())
        self.assertEqual(cfg.model.agent_latent_dim, 2)
        self.assertEqual(cfg.model.hidden_dim, 32)
        self.assertEqual(report['num_scenes'], 1)
        self.assertIn('estimator', report)
        validate_stats(cfg)

    def test_nonfinite_or_wrong_dimension_posteriors_fail_before_saving_normalization(self):
        self.write_scene('a.pkl', [1.], [2.])
        for wrong in ('nan', 'shape'):
            decoder = MomentDecoder()
            original = decoder.autoencoder.forward_encoder
            def bad_stats(graph, return_stats=True):
                am, lm, av, lv = original(graph, return_stats)
                if wrong == 'nan':
                    av.fill_(float('nan'))
                else:
                    am = am[:, :1]
                return am, lm, av, lv
            decoder.autoencoder.forward_encoder = bad_stats
            with self.subTest(wrong=wrong), patch.object(stats, 'TokenProcessor', AggregateProcessor), \
                    patch.object(stats, 'VectorWorldInitDecoder', return_value=decoder), \
                    self.assertRaisesRegex(ValueError, 'posterior'):
                stats.compute_stats(self.root / 'mock.ckpt', self.root, base_config=self.base)


if __name__ == '__main__':
    unittest.main()
