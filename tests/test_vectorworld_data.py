"""CPU graph/label contract tests for the VectorWorld initial-scene adapter."""
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.vectorworld.data import build_graph, build_generation_graph, compute_motion_code, normalize_motion
from src.smart.vectorworld.core.utils.data_container import get_features_with_motion, get_encoder_edge_indices


class VectorWorldDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def cfg():
        return SimpleNamespace(max_num_agents=30, max_num_lanes=100, fov=64.,
            min_speed=0., max_speed=114.088, min_length=-.098, max_length=22.929,
            min_width=.096, max_width=12.527)

    @staticmethod
    def scene(kind=0, *, motion=True):
        states = torch.tensor([[0., 0., 3., 0., 1., 4.5, 2.],
                               [2., 4., 5., 1., 0., 4., 1.8],
                               [-1., -4., 7., 0., -1., 1., .6]], dtype=torch.float64)
        lanes = torch.stack((torch.stack((torch.linspace(-2, 2, 20), torch.full((20,), 5.)), -1),
                             torch.stack((torch.linspace(-2, 2, 20), torch.full((20,), -5.)), -1)))
        edges = torch.cartesian_prod(torch.arange(2), torch.arange(2)).T
        result = dict(idx=0, lg_type=kind, scene_timestep=10, num_agents=3, num_lanes=2,
                      agent_states=states.numpy(), agent_types=torch.eye(3).numpy(),
                      road_points=lanes.numpy(), edge_index_lane_to_lane=edges,
                      road_connection_types=torch.nn.functional.one_hot(torch.tensor([5, 1, 2, 5]), 6).numpy())
        if motion:
            result['agent_motion_raw'] = np.tile(np.array([-1., 0.] * 6, dtype=np.float32), (3, 1))
            result['agent_motion_raw'][:, 0::2] *= np.arange(1, 4)[:, None]
            result['agent_motion_is_static'] = np.zeros(3, dtype=bool)
        return result

    @staticmethod
    def processor():
        p = SimpleNamespace(scenario_dreamer_init=True, pred_init=True, training=True)
        p._make_ego_mask = TokenProcessor._make_ego_mask
        p._get_agent_tokens = lambda types: (torch.zeros(len(types), 2), torch.zeros(len(types), 1, 3, 4, 2), torch.zeros(len(types), 1, 4, 2))
        p._attach_token_libraries = lambda agent: None
        p._load_map = lambda data: {}
        p.get_init = lambda agent, data: TokenProcessor.get_init(p, agent, data)
        return p

    def agents(self, kinds=(0,), *, motion=True):
        samples = [HeteroData(adapt_preprocessed_scene(self.scene(kind, motion=motion), f'case_{kind}_{i}.pkl'))
                   for i, kind in enumerate(kinds)]
        batch = Batch.from_data_list(samples)
        _, agent = TokenProcessor.process_data(self.processor(), batch)
        return agent, batch

    def test_native_motion_preserved_through_batch_tokenization_and_recursive_ordering(self):
        agent, _ = self.agents((0, 1))
        graph, rows, centers, angles = build_graph(agent, {}, self.cfg(), mode='lane_conditioned')
        torch.testing.assert_close(graph['agent'].motion, normalize_motion(agent['vectorworld_motion_raw'])[rows])
        self.assertEqual(graph.vectorworld_motion_source, 'native_raw')
        self.assertTrue(graph['agent'].motion_valid_mask.all())
        self.assertEqual(tuple(graph['agent'].x.shape), (6, 7))
        self.assertEqual(tuple(graph['lane'].x.shape), (4, 20, 2))
        self.assertEqual(tuple(get_features_with_motion(graph)[0].shape), (6, 22))
        torch.testing.assert_close(centers, torch.zeros(2, 2, dtype=torch.float64))
        torch.testing.assert_close(angles, torch.zeros(2, dtype=torch.float64))
        # Batch edges stay in-scene after offsets, while each map keeps directed labels.
        for source, target in [('agent', 'agent'), ('lane', 'lane'), ('lane', 'agent')]:
            store = graph[source, 'to', target]
            self.assertTrue(torch.equal(graph[source].batch[store.edge_index[0]], graph[target].batch[store.edge_index[1]]))
            expected = graph[source].partition_mask[store.edge_index[0]] == graph[target].partition_mask[store.edge_index[1]]
            self.assertTrue(torch.equal(store.encoder_mask, expected))
        labels = graph['lane', 'to', 'lane'].type.argmax(-1)
        self.assertEqual(sorted(labels.tolist()), [1, 1, 2, 2, 5, 5, 5, 5])
        get_encoder_edge_indices(graph)

    def test_before_partition_mask_is_true_in_native_boolean_convention(self):
        agent, _ = self.agents((1,))
        graph, rows, _, _ = build_graph(agent, {}, self.cfg())
        torch.testing.assert_close(graph['agent'].partition_mask, agent['initial_pos'][rows, 1] <= 0)
        torch.testing.assert_close(graph['lane'].partition_mask, graph['lane'].x[:, 9, 1] <= 0)
        torch.testing.assert_close(graph.num_agents_after_origin, torch.tensor([1]))

    def test_lane_conditioned_eval_requires_full_lanes_but_training_supports_both_types(self):
        agent, _ = self.agents((1,))
        build_graph(agent, {}, self.cfg(), mode='lane_conditioned', is_training=True)
        with self.assertRaisesRegex(ValueError, 'non-partitioned'):
            build_graph(agent, {}, self.cfg(), mode='lane_conditioned', is_training=False)
        agent, _ = self.agents((0,))
        build_graph(agent, {}, self.cfg(), mode='lane_conditioned', is_training=False)

    def test_missing_motion_never_becomes_training_supervision(self):
        agent, _ = self.agents(motion=False)
        with self.assertRaisesRegex(ValueError, 'real trajectory history'):
            build_graph(agent, {}, self.cfg())
        with self.assertRaisesRegex(ValueError, 'evaluation-only'):
            build_graph(agent, {}, self.cfg(), motion_missing='static_masked')
        with self.assertWarnsRegex(RuntimeWarning, 'stationary encoder placeholders'):
            graph, _, _, _ = build_graph(agent, {}, self.cfg(), is_training=False, motion_missing='static_masked')
        self.assertFalse(graph['agent'].motion_valid_mask.any())
        torch.testing.assert_close(graph['agent'].motion, torch.tensor([[1., 0.] * 6] * 3))

    def test_partial_missing_motion_is_rejected_or_explicitly_masked(self):
        agent, _ = self.agents()
        agent['vectorworld_motion_valid_mask'][1] = False
        with self.assertRaisesRegex(ValueError, '1/3'):
            build_graph(agent, {}, self.cfg())
        with self.assertWarns(RuntimeWarning):
            graph, rows, _, _ = build_graph(agent, {}, self.cfg(), is_training=False, motion_missing='static_masked')
        expected_mask = agent['vectorworld_motion_valid_mask'][rows]
        self.assertFalse(agent['vectorworld_motion_is_static'].any())
        self.assertTrue(torch.equal(graph['agent'].motion_valid_mask, expected_mask))
        torch.testing.assert_close(graph['agent'].motion[~expected_mask], torch.tensor([[1., 0.] * 6]))

    def test_motion_normalization_matches_native_physical_ranges_and_clipping(self):
        raw = torch.tensor([[0., 0., -12., -6., -24., 12.]])
        torch.testing.assert_close(normalize_motion(raw), torch.tensor([[1., 0., -1., -1., -1., 1.]]))
        for invalid in (torch.zeros(2, 3), torch.tensor([[float('nan'), 0.]])):
            with self.assertRaises(ValueError):
                normalize_motion(invalid)

    def test_real_history_resampling_is_body_frame_invariant_with_static_and_absent_labels(self):
        position = torch.zeros(3, 5, 2)
        position[0, :, 0] = torch.arange(5.)
        heading, velocity = torch.zeros(3, 5), torch.zeros(3, 5, 2)
        velocity[0, :, 0] = 10.
        valid = torch.ones(3, 5, dtype=torch.bool)
        valid[2, 4] = False
        raw, static, available = compute_motion_code(position, heading, velocity, valid, 4)
        torch.testing.assert_close(raw[0].reshape(6, 2)[:, 0], torch.linspace(-4, 0, 6))
        self.assertTrue(torch.equal(static, torch.tensor([False, True, True])))
        self.assertTrue(torch.equal(available, torch.tensor([True, True, False])))
        angle = .7
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        moved = position @ rotation.T + torch.tensor([100., -32.])
        transformed, _, _ = compute_motion_code(moved, heading + angle, velocity @ rotation.T, valid, 4)
        torch.testing.assert_close(raw, transformed, atol=1e-5, rtol=1e-5)

    def test_history_uses_valid_samples_only_and_native_arc_truncation(self):
        pos = torch.zeros(1, 8, 2)
        pos[0, :, 0] = torch.tensor([0., 3., 6., 9., 12., 15., 18., 21.])
        valid = torch.ones(1, 8, dtype=torch.bool)
        valid[0, 5] = False
        vel = torch.zeros_like(pos)
        vel[..., 0] = 30.
        raw, _, available = compute_motion_code(pos, torch.zeros(1, 8), vel, valid, 7)
        # Native uses searchsorted(total-12), then samples that retained arc.
        torch.testing.assert_close(raw.reshape(6, 2)[:, 0], torch.linspace(-12., 0., 6))
        self.assertTrue(available.all())

    def test_token_processor_retains_real_full_scene_history_without_computing_for_sd(self):
        pos = torch.zeros(3, 12, 3)
        raw = HeteroData(agent=dict(position=pos, heading=torch.zeros(3, 12),
            velocity=torch.ones(3, 12, 2), valid_mask=torch.ones(3, 12, dtype=torch.bool),
            shape=torch.ones(3, 3), type=torch.zeros(3, dtype=torch.long), num_nodes=3), scene_timestep=7)
        batch = Batch.from_data_list([raw])
        with patch('src.smart.vectorworld.data.compute_motion_code', side_effect=AssertionError('SD should not compute motion')):
            _, agent = TokenProcessor.process_data(self.processor(), batch)
        self.assertIs(agent['vectorworld_history']['position'], batch['agent'].position)
        torch.testing.assert_close(agent['vectorworld_history']['scene_timestep'], torch.full((3,), 7))

    def test_vectorworld_does_not_reuse_sd_latent_posteriors(self):
        agent, _ = self.agents()
        agent['sd_cached_posterior'] = {'agent_mu': torch.full((3, 99), float('nan'))}
        graph, _, _, _ = build_graph(agent, {}, self.cfg())
        self.assertNotIn('posterior_mu', graph['agent'])

    def test_prior_generation_requires_no_motion_or_reference_geometry(self):
        graph, rows, centers, angles = build_generation_graph([(0, 2, 3), (1, 1, 1)],
            agent_latent_dim=18, lane_latent_dim=24, device='cpu', dtype=torch.float32)
        self.assertEqual(tuple(graph['agent'].x.shape), (4, 18))
        self.assertFalse(graph['agent'].partition_mask.any())
        self.assertFalse(graph['lane'].partition_mask.any())
        torch.testing.assert_close(rows, torch.tensor([2, 0, 1, 3]))
        self.assertEqual(graph.vectorworld_motion_source, 'unconditioned_generation')
        torch.testing.assert_close(centers, torch.zeros(2, 2))
        torch.testing.assert_close(angles, torch.zeros(2))

    def test_native_scene_categories_override_config_per_scene_with_explicit_fallback(self):
        scenes = [self.scene(), self.scene(), self.scene()]
        scenes[0]['nocturne_compatible'] = 0
        scenes[2]['map_id'] = 1
        samples = [HeteroData(adapt_preprocessed_scene(scene, f'label_{i}.pkl'))
                   for i, scene in enumerate(scenes)]
        _, agent = TokenProcessor.process_data(self.processor(), Batch.from_data_list(samples))
        agent['sd_map_id'] = 1  # Missing metadata alone uses the configured category.
        torch.testing.assert_close(agent['vectorworld_map_valid_mask'], torch.tensor([True, False, True]))
        graph, _, _, _ = build_graph(agent, {}, self.cfg())
        torch.testing.assert_close(graph.map_id, torch.tensor([0, 1, 1]))
        self.assertEqual(graph.vectorworld_map_source, 'mixed_metadata_and_config')
        self.assertEqual(graph.vectorworld_map_sources,
                         ['metadata_nocturne_compatible', 'fallback_config1', 'metadata_map_id'])

    def test_absent_category_metadata_keeps_config_fallback_visible(self):
        agent, _ = self.agents((0, 1))
        graph, _, _, _ = build_graph(agent, {}, self.cfg())
        torch.testing.assert_close(graph.map_id, torch.tensor([0, 0]))
        self.assertEqual(graph.vectorworld_map_source, 'fallback_config0')
        self.assertEqual(graph.vectorworld_map_sources, ['fallback_config0', 'fallback_config0'])

    def test_full_scene_top_and_nested_category_metadata_survive_tokenization(self):
        for metadata in ({'nocturne_compatible': 1}, {'map_id': 1},
                         {'scenario_dreamer': {'nocturne_compatible': 1}},
                         {'scene_metadata': {'map_id': 1}},
                         {'metadata': {'nocturne_compatible': 1}, 'map_id': 0}):
            with self.subTest(metadata=metadata):
                raw = HeteroData(agent=dict(position=torch.zeros(3, 12, 3),
                    heading=torch.zeros(3, 12), velocity=torch.ones(3, 12, 2),
                    valid_mask=torch.ones(3, 12, dtype=torch.bool), shape=torch.ones(3, 3),
                    type=torch.zeros(3, dtype=torch.long), num_nodes=3), scene_timestep=7,
                    **metadata)
                _, agent = TokenProcessor.process_data(self.processor(), Batch.from_data_list([raw]))
                torch.testing.assert_close(agent['vectorworld_map_id'], torch.tensor([1]))
                self.assertTrue(agent['vectorworld_map_valid_mask'].all())

    def test_category_metadata_rejects_invalid_values_and_batch_count(self):
        for key in ('nocturne_compatible', 'map_id'):
            for value in (2, -.5, float('nan'), [0, 1]):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, '0/1 label'):
                    scene = self.scene()
                    scene[key] = value
                    adapt_preprocessed_scene(scene, 'bad_category.pkl')
        agent, _ = self.agents((0, 1))
        agent['vectorworld_map_id'] = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, 'per scene'):
            build_graph(agent, {}, self.cfg())

    def test_native_snapshot_rejects_malformed_motion(self):
        for field, value in [('agent_motion_raw', np.zeros((2, 12))),
                             ('agent_motion_valid_mask', np.zeros(2)),
                             ('agent_motion_raw', np.full((3, 12), np.nan))]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'motion'):
                scene = self.scene()
                scene[field] = value
                adapt_preprocessed_scene(scene, 'bad.pkl')


if __name__ == '__main__':
    unittest.main()
