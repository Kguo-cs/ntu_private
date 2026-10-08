"""Scene count conditioning follows SD node routing without changing cache state."""

import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.diffusion.scale_flow import Flow
from src.smart.scenario_dreamer.core.dit_layers import LabelEmbedder


class InitDiffusionCountEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(817)

    @staticmethod
    def args(**options):
        values = dict(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                      num_denoiser_layers=1, num_branch_steps=1, branch_steps=[0],
                      sampling_steps=4, use_rl=False, heading_noise='circular',
                      heading_objective='x0', velocity_representation='vector',
                      count_embedding_type='scenario_dreamer', count_lane_source='map_tokens',
                      count_max_num_agents=128, count_max_num_lanes=1024)
        values.update(options)
        return SimpleNamespace(**values)

    @staticmethod
    def processor(**options):
        values = dict(use_refiner=False, learn_init=True, init_map_range=50.)
        values.update(options)
        return SimpleNamespace(**values)

    def denoiser(self, **options):
        values = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                      num_layers=1, num_heads=2, dropout=0.,
                      count_embedding_type='scenario_dreamer', count_lane_source='map_tokens',
                      count_max_num_agents=128, count_max_num_lanes=1024)
        values.update(options)
        return InitDenoiser(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        values = dict(count_embedding_type='scenario_dreamer', count_lane_source='map_tokens',
                      count_max_num_agents=128, count_max_num_lanes=1024,
                      heading_noise='circular', heading_objective='angular_velocity',
                      velocity_representation='speed')
        values.update(options)
        with patch.object(InitDiffusion, '_make_args', return_value=self.args()):
            return InitDiffusion(32, 2, 4, self.processor(), False, **values)

    @staticmethod
    def inputs(representation='vector'):
        vector = torch.tensor([[2., 30., 1., 0., 4.5, 2., 3., 4.],
                               [-20., 30., 0., 1., 4., 1.8, -2., 0.],
                               [0., 0., 1., 0., 4.5, 2., 0., 0.],
                               [0., 0., 1., 0., 4.5, 2., 0., 0.],
                               [15., 25., .6, .8, 4.8, 2., 7., 0.],
                               [0., 0., 1., 0., 4.5, 2., 0., 0.]])
        clean = (torch.cat((vector[:, :6], vector[:, 6:8].norm(dim=-1, keepdim=True)), -1)
                 if representation == 'speed' else vector)
        batch = torch.tensor([0, 0, 0, 1, 2, 2])
        types = torch.tensor([0, 1, 0, 0, 2, 0])
        type_counts = torch.bincount(batch * 3 + types, minlength=9).reshape(3, 3).float()
        agent = dict(batch=batch, type=types, num_graphs=3,
                     ego_mask=torch.tensor([False, False, True, True, False, True]),
                     expert_input=clean.clone(), local_vel=vector[:, 6:8].clone(),
                     batch_ego_pos=torch.zeros(6, 2), batch_ego_heading=torch.zeros(6),
                     initial_pos=vector[:, :2].clone(),
                     initial_heading=torch.atan2(vector[:, 3], vector[:, 2]),
                     ego_feat=torch.cat((torch.zeros(3, 9), type_counts), -1),
                     sd_map=dict(batch=torch.tensor([0, 0, 1, 1, 1, 1, 2])))
        # The final scene deliberately has no prepared map tokens. Exact SD
        # lanes still exist there, so the two sources have distinct semantics.
        feature = dict(batch=torch.tensor([0, 1, 1, 1]),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.], [0., -4.]]),
                       orientation=torch.tensor([0., .3, -.4, 0.]), pt_token=torch.randn(4, 32))
        return clean, agent, feature

    def test_bundled_label_embedding_and_scene_counts_match_exactly(self):
        model = self.denoiser()
        _, agent, feature = self.inputs()
        for embedder, maximum in ((model.num_agents_embedder, 128),
                                  (model.num_lanes_embedder, 1024)):
            self.assertIsInstance(embedder, LabelEmbedder)
            self.assertEqual(embedder.dropout_prob, 0.)
            self.assertEqual(embedder.num_classes, maximum + 1)
            self.assertEqual(embedder.embedding_table.num_embeddings, maximum + 1)
        agents, lanes = model._embed_scene_counts(agent, feature)
        self.assertEqual(agents.shape, (3, 32))
        self.assertEqual(lanes.shape, (3, 32))
        torch.testing.assert_close(agents,
            model.num_agents_embedder(torch.tensor([3, 1, 2]), train=False), atol=0, rtol=0)
        torch.testing.assert_close(lanes,
            model.num_lanes_embedder(torch.tensor([1, 3, 0]), train=False), atol=0, rtol=0)
        ego_before = agent['ego_feat'].clone()
        model._embed_scene_counts(agent, feature)
        torch.testing.assert_close(agent['ego_feat'], ego_before, atol=0, rtol=0)

    def test_zero_and_last_vocabulary_entries_are_valid_without_extra_cfg_label(self):
        model = self.denoiser(count_max_num_agents=5, count_max_num_lanes=7)
        agent = dict(batch=torch.zeros(5, dtype=torch.long), num_graphs=2)
        feature = dict(batch=torch.zeros(7, dtype=torch.long))
        agents, lanes = model._embed_scene_counts(agent, feature)
        torch.testing.assert_close(agents,
            model.num_agents_embedder(torch.tensor([5, 0]), train=False), atol=0, rtol=0)
        torch.testing.assert_close(lanes,
            model.num_lanes_embedder(torch.tensor([7, 0]), train=False), atol=0, rtol=0)
        feature['batch'] = torch.empty(0, dtype=torch.long)
        _, empty = model._embed_scene_counts(agent, feature)
        torch.testing.assert_close(empty,
            model.num_lanes_embedder(torch.tensor([0, 0]), train=False), atol=0, rtol=0)

    def test_sources_are_explicit_and_identical_between_training_and_evaluation(self):
        _, agent, feature = self.inputs()
        for source, expected_counts in (('map_tokens', torch.tensor([1, 3, 0])),
                                        ('scenario_dreamer', torch.tensor([2, 4, 1]))):
            model = self.denoiser(count_lane_source=source)
            observed = []
            for training in (True, False):
                model.train(training)
                agents, lanes = model._embed_scene_counts(agent, feature)
                expected = model.num_lanes_embedder(expected_counts, train=training)
                torch.testing.assert_close(lanes, expected, atol=0, rtol=0)
                observed.append((agents, lanes))
            for actual, expected in zip(observed[0], observed[1]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        exact = self.denoiser(count_lane_source='scenario_dreamer')
        agent.pop('sd_map')
        for training in (True, False):
            exact.train(training)
            with self.subTest(training=training), self.assertRaises(ValueError):
                exact._embed_scene_counts(agent, feature)

    def test_routing_adds_each_count_only_to_its_node_type_and_never_mutates_cache(self):
        model = self.denoiser()
        clean, agent, feature = self.inputs()
        original_tokens = feature['pt_token'].clone()
        original_mapping = dict(feature)
        agent_emb, lane_emb = model._embed_scene_counts(agent, feature)
        time = torch.full((6, 1), .4)
        base = model._embed_agents(clean, time, agent['type'], agent['batch'], agent, 1)[0]
        captured = []

        def inspect_attention(**kwargs):
            captured.append(kwargs)
            return torch.zeros(len(kwargs['feat_a']), 8)

        with patch.object(model, '_apply_graph_attention', side_effect=inspect_attention):
            for _ in range(3):
                model(clean, time, agent, feature)
        for values in captured:
            torch.testing.assert_close(values['feat_a'], base + agent_emb[agent['batch']])
            torch.testing.assert_close(values['map_feature']['pt_token'],
                                       original_tokens + lane_emb[feature['batch']])
            self.assertIsNot(values['map_feature'], feature)
        torch.testing.assert_close(feature['pt_token'], original_tokens, atol=0, rtol=0)
        self.assertEqual(set(feature), set(original_mapping))
        for key in ('batch', 'position', 'orientation'):
            self.assertIs(feature[key], original_mapping[key])

    def test_eval_mask_retains_full_scene_agent_counts_including_ego(self):
        model = self.denoiser()
        clean, agent, feature = self.inputs()
        mask = torch.tensor([True, False, False, False, True, False])
        time = torch.full((6, 1), .4)
        selected_batch = agent['batch'][mask]
        base = model._embed_agents(clean[mask], time[mask], agent['type'][mask],
                                   selected_batch, agent, 1)[0]
        with patch.object(model, '_apply_graph_attention',
                          return_value=torch.zeros(2, 8)) as attention:
            result = model(clean, time, agent, feature, eval_mask=mask)
        self.assertEqual(result.shape, (2, 8))
        expected = model.num_agents_embedder(torch.tensor([3, 2]), train=model.training)
        torch.testing.assert_close(attention.call_args.kwargs['feat_a'], base + expected)
        torch.testing.assert_close(agent['batch'], torch.tensor([0, 0, 0, 1, 2, 2]))

    def test_count_overflows_raise_instead_of_clipping(self):
        _, agent, feature = self.inputs()
        for options in (dict(count_max_num_agents=2), dict(count_max_num_lanes=2),
                        dict(count_lane_source='scenario_dreamer', count_max_num_lanes=3)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.denoiser(**options)._embed_scene_counts(agent, feature)

    def test_invalid_options_are_rejected_at_public_entry_points(self):
        invalid_options = [dict(count_embedding_type='unknown'), dict(count_lane_source='unknown')]
        invalid_options += [{key: value} for key in ('count_max_num_agents', 'count_max_num_lanes')
                            for value in (0, -1, True, 1.5, float('nan'), float('inf'))]
        for options in invalid_options:
            for constructor in (lambda: self.denoiser(**options),
                                lambda: self.flow(**options),
                                lambda: self.wrapper(**options)):
                with self.subTest(options=options, constructor=constructor), self.assertRaises(ValueError):
                    constructor()

    def test_invalid_batch_ids_are_rejected_without_count_aliasing(self):
        model = self.denoiser()
        _, agent, feature = self.inputs()
        for field in ('agent', 'map'):
            for invalid in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([[0, 1]]),
                            torch.tensor([.5]), torch.tensor([True])):
                inputs, maps = copy.deepcopy(agent), copy.deepcopy(feature)
                (inputs if field == 'agent' else maps)['batch'] = invalid
                with self.subTest(field=field, shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                    model._embed_scene_counts(inputs, maps)

    def test_initialization_and_main_refiner_option_propagation(self):
        flow = Flow(self.args(heading_noise='gaussian', count_lane_source='scenario_dreamer',
                              count_max_num_agents=47, count_max_num_lanes=203),
                    self.processor(use_refiner=True), False)
        for model in (flow.model, flow.refine_model):
            self.assertEqual(model.count_embedding_type, 'scenario_dreamer')
            self.assertEqual(model.count_lane_source, 'scenario_dreamer')
            self.assertEqual(model.count_max_num_agents, 47)
            self.assertEqual(model.count_max_num_lanes, 203)
            for embedder, maximum in ((model.num_agents_embedder, 47),
                                      (model.num_lanes_embedder, 203)):
                weight = embedder.embedding_table.weight
                self.assertEqual(weight.shape, (maximum + 1, 32))
                self.assertAlmostEqual(weight.std(unbiased=False).item(), .02, delta=.002)
                self.assertLess(weight.mean().abs().item(), .002)

    def test_disabled_default_preserves_checkpoint_schema_rng_and_legacy_behavior(self):
        options = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                       num_layers=1, num_heads=2, dropout=0.)
        torch.manual_seed(817)
        default = InitDenoiser(**options)
        next_default = torch.randn(8)
        torch.manual_seed(817)
        explicit = InitDenoiser(**options, count_embedding_type='none')
        next_explicit = torch.randn(8)
        self.assertFalse(hasattr(default, 'num_agents_embedder'))
        self.assertFalse(hasattr(default, 'num_lanes_embedder'))
        self.assertFalse(any('num_agents_embedder' in key or 'num_lanes_embedder' in key
                             for key in default.state_dict()))
        torch.testing.assert_close(next_default, next_explicit, atol=0, rtol=0)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key, value in default.state_dict().items():
            torch.testing.assert_close(explicit.state_dict()[key], value, atol=0, rtol=0)
        loaded = explicit.load_state_dict(default.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        clean, agent, feature = self.inputs()
        default.eval()
        explicit.eval()
        torch.testing.assert_close(default(clean, torch.full((6, 1), .4), agent, feature),
                                   explicit(clean, torch.full((6, 1), .4), agent, feature))
        with self.assertRaises(RuntimeError):
            self.denoiser().load_state_dict(default.state_dict(), strict=True)

    def test_real_train_and_sampling_gradients_for_speed_vector_and_time_options(self):
        for representation in ('vector', 'speed'):
            for time_type in ('legacy', 'scenario_dreamer'):
                for objective in ('x0', 'angular_velocity'):
                    with self.subTest(representation=representation, time=time_type, objective=objective):
                        flow = self.flow(velocity_representation=representation,
                                         time_embedding_type=time_type, heading_objective=objective)
                        clean, agent, feature = self.inputs(representation)
                        before = feature['pt_token'].clone()
                        with patch.object(flow, '_sample_time', return_value=torch.full((6, 1), .4)):
                            losses = flow._supervised_loss(clean, agent, feature)
                        total = losses[0].mean() + losses[1]
                        self.assertTrue(torch.isfinite(total))
                        total.backward()
                        for embedder in (flow.model.num_agents_embedder, flow.model.num_lanes_embedder):
                            gradient = embedder.embedding_table.weight.grad
                            self.assertIsNotNone(gradient)
                            self.assertTrue(torch.isfinite(gradient).all())
                            self.assertGreater(gradient.abs().sum().item(), 0.)
                        flow.eval()
                        generated = flow.sample(agent, feature, steps=4)
                        self.assertEqual(generated.shape, clean.shape)
                        self.assertTrue(torch.isfinite(generated).all())
                        torch.testing.assert_close(generated[agent['ego_mask']], clean[agent['ego_mask']],
                                                   atol=0, rtol=0)
                        torch.testing.assert_close(feature['pt_token'], before, atol=0, rtol=0)

    def test_wrapper_ema_and_count_weights_survive_strict_checkpoint_roundtrip(self):
        source = self.wrapper(use_ema=True, ema_decay=.9, count_max_num_agents=41,
                              count_max_num_lanes=307, time_embedding_type='scenario_dreamer')
        self.assertEqual(source.count_embedding_type, 'scenario_dreamer')
        self.assertEqual(source.G1.model.count_max_num_agents, 41)
        self.assertEqual(source.G1.model.count_max_num_lanes, 307)
        names = source.get_extra_state()['ema_parameter_names']
        expected_names = ['model.num_agents_embedder.embedding_table.weight',
                          'model.num_lanes_embedder.embedding_table.weight']
        self.assertTrue(all(name in names for name in expected_names))
        with torch.no_grad():
            source.G1.model.num_agents_embedder.embedding_table.weight.add_(.05)
            source.G1.model.num_lanes_embedder.embedding_table.weight.add_(.05)
        source.update_ema()
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        target = self.wrapper(use_ema=True, ema_decay=.7, count_max_num_agents=41,
                              count_max_num_lanes=307, time_embedding_type='scenario_dreamer')
        loaded = target.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        self.assertEqual(target.ema.decay, .9)
        self.assertEqual(target.ema.num_updates, 1)
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        _, agent, feature = self.inputs('speed')
        online = copy.deepcopy(target.G1.state_dict())
        with source.ema.average_parameters(source.G1.parameters()):
            expected = source.G1.model._embed_scene_counts(agent, feature)
        with target.ema.average_parameters(target.G1.parameters()):
            actual = target.G1.model._embed_scene_counts(agent, feature)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
        for key, value in online.items():
            torch.testing.assert_close(target.G1.state_dict()[key], value, atol=0, rtol=0)

    def test_training_and_evaluation_configs_share_defaults_and_explicit_sources(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                default = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                    'model.model_config.decoder.init_diffusion.count_embedding_type=none'])
                options = default.model.model_config.decoder.init_diffusion
                self.assertEqual(options.count_embedding_type, 'none')
                self.assertEqual(options.count_lane_source, 'map_tokens')
                self.assertEqual(options.count_max_num_agents, 128)
                self.assertEqual(options.count_max_num_lanes, 1024)
                for source in ('map_tokens', 'scenario_dreamer'):
                    config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                        'model.model_config.decoder.init_diffusion.count_embedding_type=scenario_dreamer',
                        f'model.model_config.decoder.init_diffusion.count_lane_source={source}',
                        'model.model_config.decoder.init_diffusion.count_max_num_agents=30',
                        'model.model_config.decoder.init_diffusion.count_max_num_lanes=100'])
                    actual = config.model.model_config.decoder.init_diffusion
                    self.assertEqual(actual.count_embedding_type, 'scenario_dreamer')
                    self.assertEqual(actual.count_lane_source, source)
                    self.assertEqual(actual.count_max_num_agents, 30)
                    self.assertEqual(actual.count_max_num_lanes, 100)


if __name__ == '__main__':
    unittest.main()
