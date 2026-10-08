"""Rebuilt SD map category labels survive batching and initial tokenization."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Batch, HeteroData

from src.smart.metrics.metadata_adapter import attach_sd_metric_metadata
from src.smart.scenario_dreamer.preprocessed import read_vectorworld_map_metadata
from src.smart.tokens.token_processor import TokenProcessor


class InitDiffusionMapMetadataTest(unittest.TestCase):
    @staticmethod
    def saved(**labels):
        info = dict(center_world=[12., -7.], rotation_angle=.37,
                    lane_adjacency=torch.zeros(3, 3))
        info.update(labels)
        return dict(scenario_dreamer=info,
                    scenario_dreamer_cache_file='scene_0_10.pkl', scene_timestep=10)

    @classmethod
    def rebuilt(cls, saved, n_agents=2):
        data = HeteroData(deepcopy(saved))
        cached = data['tokenized_agent']
        cached.num_nodes = n_agents
        cached.initial_pos = torch.zeros(n_agents, 2)
        cached.initial_heading = torch.zeros(n_agents)
        cached.local_vel = torch.ones(n_agents, 2)
        cached.type = torch.zeros(n_agents, dtype=torch.long)
        cached.shape = torch.ones(n_agents, 2)
        return attach_sd_metric_metadata(data, saved, generation_scene_timestep=10)

    @staticmethod
    def processor():
        processor = SimpleNamespace(scenario_dreamer_init=True, pred_init=True)
        processor._make_ego_mask = TokenProcessor._make_ego_mask
        processor._load_map = lambda data: {}
        processor._load_cached_initial_agent = lambda cached: TokenProcessor._load_cached_initial_agent(processor, cached)
        processor._attach_token_libraries = lambda agent: None
        return processor

    def test_nested_nocturne_label_is_kept_before_dense_graph_removal(self):
        saved = self.saved(nocturne_compatible=1)
        data = self.rebuilt(saved)
        self.assertNotIn('scenario_dreamer', data)
        self.assertNotIn('scenario_dreamer', data.node_types)
        self.assertEqual(data.vectorworld_map_id.tolist(), [1])
        self.assertEqual(data.vectorworld_map_id.dtype, torch.long)
        self.assertEqual(data.vectorworld_map_valid_mask.tolist(), [True])
        self.assertEqual(data.vectorworld_map_valid_mask.dtype, torch.bool)
        self.assertEqual(data.vectorworld_map_source, 'metadata_scenario_dreamer_nocturne_compatible')
        # Metric coordinate/frame fields are still attached exactly as before.
        torch.testing.assert_close(data.sd_center_world, torch.tensor([[12., -7.]], dtype=torch.float64))
        torch.testing.assert_close(data.sd_rotation_angle, torch.tensor([.37], dtype=torch.float64))
        self.assertEqual(data.scene_timestep, 10)
        self.assertEqual(data.generation_scene_timestep, 10)
        self.assertIn('scenario_dreamer', saved)

    def test_same_dictionary_pipeline_preserves_nested_generic_map_id(self):
        saved = self.saved(map_id=0)
        result = attach_sd_metric_metadata(saved, saved, generation_scene_timestep=10)
        self.assertIs(result, saved)
        self.assertNotIn('scenario_dreamer', result)
        self.assertEqual(result['vectorworld_map_id'].tolist(), [0])
        self.assertEqual(result['vectorworld_map_valid_mask'].tolist(), [True])
        self.assertEqual(result['vectorworld_map_source'], 'metadata_scenario_dreamer_map_id')

    def test_explicit_top_canonical_label_mask_and_source_keep_reader_precedence(self):
        saved = self.saved(nocturne_compatible=1, map_id=1)
        saved.update(vectorworld_map_id=torch.tensor([0]),
                     vectorworld_map_valid_mask=torch.tensor([False]),
                     vectorworld_map_source='explicit_unavailable')
        data = self.rebuilt(saved)
        self.assertEqual(data.vectorworld_map_id.tolist(), [0])
        self.assertEqual(data.vectorworld_map_valid_mask.tolist(), [False])
        self.assertEqual(data.vectorworld_map_source, 'explicit_unavailable')
        expected = read_vectorworld_map_metadata(saved, 1)
        actual = read_vectorworld_map_metadata(data, 1)
        for observed, wanted in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(observed, wanted, atol=0, rtol=0)
        self.assertEqual(actual[2], expected[2])

    def test_canonical_nested_label_outranks_top_generic_map_id(self):
        saved = self.saved(vectorworld_map_id=1, vectorworld_map_valid_mask=True,
                           vectorworld_map_source='nested_explicit')
        saved['map_id'] = 0
        data = self.rebuilt(saved)
        self.assertEqual(data.vectorworld_map_id.tolist(), [1])
        self.assertEqual(data.vectorworld_map_source, 'nested_explicit')

    def test_missing_category_zero_placeholder_is_never_marked_valid(self):
        data = self.rebuilt(self.saved())
        self.assertEqual(data.vectorworld_map_id.tolist(), [0])
        self.assertEqual(data.vectorworld_map_valid_mask.tolist(), [False])
        self.assertEqual(data.vectorworld_map_source, 'missing')
        ids, valid, sources = read_vectorworld_map_metadata(data, 1)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(valid.tolist(), [False])
        self.assertEqual(sources, ['missing'])

    def test_mixed_pyg_batch_preserves_scene_order_and_token_processor_labels(self):
        saved = [self.saved(nocturne_compatible=1), self.saved(), self.saved(map_id=0)]
        for scene, count in zip(saved, (2, 3, 1)):
            scene['scenario_dreamer']['lane_adjacency'] = torch.zeros(count, count)
        samples = [self.rebuilt(scene, count) for scene, count in zip(saved, (2, 3, 1))]
        # The removed nested metadata deliberately has unequal dense graph sizes.
        batch = Batch.from_data_list(samples)
        self.assertEqual(batch.num_graphs, 3)
        self.assertEqual(batch.vectorworld_map_id.tolist(), [1, 0, 0])
        self.assertEqual(batch.vectorworld_map_valid_mask.tolist(), [True, False, True])
        expected_sources = ['metadata_scenario_dreamer_nocturne_compatible', 'missing',
                            'metadata_scenario_dreamer_map_id']
        self.assertEqual(batch.vectorworld_map_source, expected_sources)
        _, agents = TokenProcessor.process_data(self.processor(), batch)
        self.assertEqual(agents['num_graphs'], 3)
        self.assertEqual(agents['vectorworld_map_id'].tolist(), [1, 0, 0])
        self.assertEqual(agents['vectorworld_map_valid_mask'].tolist(), [True, False, True])
        self.assertEqual(agents['vectorworld_map_source'], expected_sources)
        self.assertEqual(agents['batch'].tolist(), [0, 0, 1, 1, 1, 2])

    def test_generic_initial_diffusion_opt_in_forwards_labels_and_graph_types(self):
        scenes = [self.saved(nocturne_compatible=1), self.saved(map_id=0)]
        for scene, kind in zip(scenes, (0, 1)):
            scene['sd_lg_type'] = kind
        batch = Batch.from_data_list([self.rebuilt(scene) for scene in scenes])
        processor = self.processor()
        processor.scenario_dreamer_init = False
        processor.init_map_id_conditioning = True
        _, agents = TokenProcessor.process_data(processor, batch)
        self.assertEqual(agents['vectorworld_map_id'].tolist(), [1, 0])
        self.assertEqual(agents['vectorworld_map_valid_mask'].tolist(), [True, True])
        self.assertEqual(agents['lg_type'].tolist(), [0, 1])

    def test_generic_legacy_tokenization_does_not_add_optional_labels(self):
        batch = Batch.from_data_list([self.rebuilt(self.saved(nocturne_compatible=1))])
        processor = self.processor()
        processor.scenario_dreamer_init = False
        _, agents = TokenProcessor.process_data(processor, batch)
        self.assertNotIn('vectorworld_map_id', agents)
        self.assertNotIn('vectorworld_map_valid_mask', agents)
        self.assertNotIn('lg_type', agents)

    def test_invalid_categories_masks_or_sources_fail_shared_validation(self):
        invalid = [dict(nocturne_compatible=2), dict(nocturne_compatible=.5),
                   dict(map_id=float('nan')), dict(map_id=[0, 1]),
                   dict(vectorworld_map_id=1, vectorworld_map_valid_mask=2),
                   dict(vectorworld_map_id=1, vectorworld_map_valid_mask=[True, False]),
                   dict(vectorworld_map_id=1, vectorworld_map_source=['a', 'b']),
                   dict(vectorworld_map_id=1, vectorworld_map_source=[5])]
        for labels in invalid:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.rebuilt(self.saved(**labels))

    def test_metric_frame_alignment_guard_is_unchanged(self):
        saved = self.saved(map_id=1)
        with self.assertRaisesRegex(ValueError, 'reference frame'):
            attach_sd_metric_metadata({}, saved, generation_scene_timestep=11)
        with self.assertRaisesRegex(KeyError, 'save-scene-info'):
            attach_sd_metric_metadata({}, {'map_id': 1})


if __name__ == '__main__':
    unittest.main()
