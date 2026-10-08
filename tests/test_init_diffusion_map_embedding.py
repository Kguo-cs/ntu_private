"""SD scene-type conditioning shares labels and dropout across agent/map nodes."""

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


class InitDiffusionMapEmbeddingTest(unittest.TestCase):
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
                      map_embedding_type='scenario_dreamer', map_id_source='fixed',
                      map_id=0, map_lg_type=0, map_label_dropout=.1)
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
                      map_embedding_type='scenario_dreamer', map_id_source='fixed',
                      map_id=0, map_lg_type=0, map_label_dropout=.1)
        values.update(options)
        return InitDenoiser(**values)

    def flow(self, **options):
        return Flow(self.args(**options), self.processor(), False)

    def wrapper(self, **options):
        values = dict(map_embedding_type='scenario_dreamer', map_id_source='fixed',
                      map_id=0, map_lg_type=0, map_label_dropout=.1,
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
                     vectorworld_map_id=torch.tensor([0, 1, 0]),
                     vectorworld_map_valid_mask=torch.ones(3, dtype=torch.bool),
                     vectorworld_map_source=['test_metadata'] * 3,
                     sd_map=dict(batch=torch.tensor([0, 0, 1, 1, 1, 1, 2]),
                                 lg_type=torch.tensor([0, 1, 0])))
        feature = dict(batch=torch.tensor([0, 1, 1, 1]),
                       position=torch.tensor([[0., -4.], [5., 4.], [-5., 4.], [0., -4.]]),
                       orientation=torch.tensor([0., .3, -.4, 0.]), pt_token=torch.randn(4, 32))
        return clean, agent, feature

    def test_bundled_module_all_four_scene_indices_and_null_row_match_sd(self):
        model = self.denoiser(map_id_source='metadata', map_lg_type=None).eval()
        labels = dict(batch=torch.arange(4), num_graphs=4,
                      vectorworld_map_id=torch.tensor([0, 1, 0, 1]),
                      vectorworld_map_valid_mask=torch.ones(4, dtype=torch.bool),
                      sd_map=dict(lg_type=torch.tensor([0, 0, 1, 1])))
        embedder = model.scene_type_embedder
        self.assertIsInstance(embedder, LabelEmbedder)
        self.assertEqual(embedder.num_classes, 4)
        self.assertEqual(embedder.dropout_prob, .1)
        self.assertEqual(embedder.embedding_table.num_embeddings, 5)
        expected = embedder(torch.tensor([0, 1, 2, 3]), train=False)
        actual = model._embed_scene_type(labels)
        self.assertEqual(actual.shape, (4, 32))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.device, labels['batch'].device)

    def test_fixed_category_ignores_missing_or_conflicting_metadata(self):
        model = self.denoiser(map_id=1, map_lg_type=0).eval()
        _, agent, _ = self.inputs()
        agent['vectorworld_map_id'] = torch.tensor([5, -1, .5])
        agent['vectorworld_map_valid_mask'] = torch.zeros(3, dtype=torch.bool)
        agent['sd_map']['lg_type'] = torch.tensor([1, 1, 1])
        expected = model.scene_type_embedder(torch.tensor([1, 1, 1]), train=False)
        torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)
        for key in ('vectorworld_map_id', 'vectorworld_map_valid_mask', 'sd_map'):
            agent.pop(key)
        torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)

    def test_metadata_labels_honor_validity_and_supported_source_fields(self):
        model = self.denoiser(map_id_source='metadata', map_label_dropout=0.).eval()
        for values in (dict(nocturne_compatible=torch.tensor([0, 1, 0])),
                       dict(map_id=torch.tensor([0, 1, 0])),
                       dict(vectorworld_map_id=torch.tensor([0, 1, 0]),
                            vectorworld_map_valid_mask=torch.ones(3, dtype=torch.bool)),
                       dict(metadata=dict(nocturne_compatible=torch.tensor([0, 1, 0])))):
            labels = dict(batch=torch.tensor([0, 0, 1, 2]), num_graphs=3, **values)
            expected = model.scene_type_embedder(torch.tensor([0, 1, 0]), train=False)
            torch.testing.assert_close(model._embed_scene_type(labels), expected, atol=0, rtol=0)
        _, agent, _ = self.inputs()
        invalid = copy.deepcopy(agent)
        invalid['vectorworld_map_valid_mask'][1] = False
        for labels in (invalid, dict(batch=agent['batch'], num_graphs=3)):
            for training in (True, False):
                model.train(training)
                with self.subTest(training=training), self.assertRaises(ValueError):
                    model._embed_scene_type(labels)

    def test_optional_graph_type_uses_explicit_metadata_and_never_filename_inference(self):
        model = self.denoiser(map_id=1, map_lg_type=None, map_label_dropout=0.).eval()
        _, agent, _ = self.inputs()
        expected = model.scene_type_embedder(torch.tensor([1, 3, 1]), train=False)
        torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)
        graph_type = agent.pop('sd_map')['lg_type']
        agent['lg_type'] = graph_type
        torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)
        agent.pop('lg_type')
        agent['scenario_dreamer_cache_file'] = ['training.tfrecord-00001-of-01000_8_1_9.pkl'] * 3
        with self.assertRaises(ValueError):
            model._embed_scene_type(agent)

    def test_invalid_options_and_metadata_raise_before_label_lookup(self):
        invalid_options = [dict(map_embedding_type='unknown'), dict(map_id_source='unknown')]
        invalid_options += [{key: value} for key in ('map_id', 'map_lg_type')
                            for value in (-1, 2, True, .5)]
        invalid_options += [dict(map_label_dropout=value)
                            for value in (-.1, 1.1, float('nan'), float('inf'))]
        for options in invalid_options:
            for constructor in (lambda: self.denoiser(**options),
                                lambda: self.flow(**options),
                                lambda: self.wrapper(**options)):
                with self.subTest(options=options, constructor=constructor), self.assertRaises(ValueError):
                    constructor()
        model = self.denoiser(map_id_source='metadata', map_lg_type=None).eval()
        _, agent, _ = self.inputs()
        for ids in (torch.tensor([0, 2, 1]), torch.tensor([0., .5, 1.]), torch.tensor([0, 1])):
            labels = copy.deepcopy(agent)
            labels['vectorworld_map_id'] = ids
            with self.subTest(ids=ids.tolist()), self.assertRaises(ValueError):
                model._embed_scene_type(labels)
        for kinds in (torch.tensor([0, 2, 1]), torch.tensor([0., .5, 1.]), torch.tensor([0, 1])):
            labels = copy.deepcopy(agent)
            labels['sd_map']['lg_type'] = kinds
            with self.subTest(kinds=kinds.tolist()), self.assertRaises(ValueError):
                model._embed_scene_type(labels)

    def test_shared_scene_dropout_is_sampled_once_and_routes_to_both_node_types(self):
        model = self.denoiser(map_id_source='metadata', map_lg_type=None).train()
        clean, agent, feature = self.inputs()
        original = feature['pt_token'].clone()
        time = torch.full((6, 1), .4)
        base = model._embed_agents(clean, time, agent['type'], agent['batch'], agent, 1)[0]
        embedder = model.scene_type_embedder
        with torch.no_grad():
            embedder.embedding_table.weight.copy_(torch.arange(5.)[:, None].expand(-1, 32))
        # Scene0 and2 drop to nullrow4; scene1 keeps 2*lg1+map1 =3.
        scene = embedder.embedding_table(torch.tensor([4, 3, 4]))
        with patch('src.smart.scenario_dreamer.core.dit_layers.torch.rand',
                   return_value=torch.tensor([0., 1., 0.])) as random_draw, \
                patch.object(embedder, 'forward', wraps=embedder.forward) as embed, \
                patch.object(model, '_apply_graph_attention',
                             return_value=torch.zeros(6, 8)) as attention:
            model(clean, time, agent, feature)
        self.assertEqual(random_draw.call_count, 1)
        self.assertEqual(embed.call_count, 1)
        torch.testing.assert_close(attention.call_args.kwargs['feat_a'], base + scene[agent['batch']])
        torch.testing.assert_close(attention.call_args.kwargs['map_feature']['pt_token'],
                                   original + scene[feature['batch']])
        torch.testing.assert_close(feature['pt_token'], original, atol=0, rtol=0)

    def test_dropout_probability_endpoints_and_evaluation_labels(self):
        _, agent, _ = self.inputs()
        for probability, rows in ((0., 4), (1., 5)):
            model = self.denoiser(map_id=1, map_label_dropout=probability).train()
            self.assertEqual(model.scene_type_embedder.embedding_table.num_embeddings, rows)
            expected_label = 4 if probability == 1. else 1
            expected = model.scene_type_embedder.embedding_table(torch.full((3,), expected_label))
            torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)
            model.eval()
            expected = model.scene_type_embedder.embedding_table(torch.full((3,), 1))
            torch.testing.assert_close(model._embed_scene_type(agent), expected, atol=0, rtol=0)

    def test_map_count_time_combination_and_eval_mask_do_not_accumulate_cached_features(self):
        model = self.denoiser(map_id_source='metadata', map_lg_type=None, map_label_dropout=0.,
                              count_embedding_type='scenario_dreamer', time_embedding_type='scenario_dreamer')
        clean, agent, feature = self.inputs()
        time = torch.full((6, 1), .4)
        mask = torch.tensor([True, False, False, False, True, False])
        batch = agent['batch'][mask]
        original = feature['pt_token'].clone()
        base = model._embed_agents(clean[mask], time[mask], agent['type'][mask], batch, agent, 1)[0]
        agent_count, lane_count = model._embed_scene_counts(agent, feature)
        scene = model._embed_scene_type(agent)
        captured = []

        def inspect_attention(**kwargs):
            captured.append(kwargs)
            return torch.zeros(2, 8)

        with patch.object(model, '_apply_graph_attention', side_effect=inspect_attention):
            for _ in range(3):
                result = model(clean, time, agent, feature, eval_mask=mask)
                self.assertEqual(result.shape, (2, 8))
        for values in captured:
            torch.testing.assert_close(values['feat_a'], base + agent_count[batch] + scene[batch])
            torch.testing.assert_close(values['map_feature']['pt_token'],
                                       original + lane_count[feature['batch']] + scene[feature['batch']])
            self.assertIsNot(values['map_feature'], feature)
        torch.testing.assert_close(feature['pt_token'], original, atol=0, rtol=0)

    def test_initialization_main_refiner_and_option_propagation(self):
        flow = Flow(self.args(heading_noise='gaussian', map_id_source='metadata', map_id=1,
                              map_lg_type=None, map_label_dropout=.25),
                    self.processor(use_refiner=True), False)
        for model in (flow.model, flow.refine_model):
            self.assertEqual(model.map_embedding_type, 'scenario_dreamer')
            self.assertEqual(model.map_id_source, 'metadata')
            self.assertEqual(model.map_id, 1)
            self.assertIsNone(model.map_lg_type)
            self.assertEqual(model.map_label_dropout, .25)
            self.assertEqual(model.scene_type_embedder.embedding_table.weight.shape, (5, 32))
            self.assertAlmostEqual(model.scene_type_embedder.embedding_table.weight.std(
                unbiased=False).item(), .02, delta=.004)

    def test_disabled_default_preserves_rng_checkpoint_and_forward_behavior(self):
        options = dict(token_processor=None, input_dim=8, hidden_dim=32, output_dim=8,
                       num_layers=1, num_heads=2, dropout=0.)
        torch.manual_seed(817)
        default = InitDenoiser(**options)
        next_default = torch.randn(8)
        torch.manual_seed(817)
        explicit = InitDenoiser(**options, map_embedding_type='none')
        next_explicit = torch.randn(8)
        self.assertFalse(hasattr(default, 'scene_type_embedder'))
        self.assertFalse(any('scene_type_embedder' in key for key in default.state_dict()))
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
        time = torch.full((6, 1), .4)
        torch.testing.assert_close(default(clean, time, agent, feature),
                                   explicit(clean, time, agent, feature), atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            self.denoiser().load_state_dict(default.state_dict(), strict=True)

    def test_real_training_sampling_and_gradients_for_speed_vector_and_condition_options(self):
        for representation in ('vector', 'speed'):
            for source in ('fixed', 'metadata'):
                for objective in ('x0', 'angular_velocity'):
                    with self.subTest(representation=representation, source=source, objective=objective):
                        flow = self.flow(velocity_representation=representation, map_id_source=source,
                                         map_lg_type=None if source == 'metadata' else 0,
                                         heading_objective=objective, map_label_dropout=0.,
                                         count_embedding_type='scenario_dreamer',
                                         time_embedding_type='scenario_dreamer')
                        clean, agent, feature = self.inputs(representation)
                        original = feature['pt_token'].clone()
                        optimizer = torch.optim.Adam(flow.parameters(), lr=1.e-3)
                        before = flow.model.scene_type_embedder.embedding_table.weight.detach().clone()
                        with patch.object(flow, '_sample_time', return_value=torch.full((6, 1), .4)):
                            losses = flow._supervised_loss(clean, agent, feature)
                        total = losses[0].mean() + losses[1]
                        self.assertTrue(torch.isfinite(total))
                        total.backward()
                        gradient = flow.model.scene_type_embedder.embedding_table.weight.grad
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0.)
                        optimizer.step()
                        self.assertFalse(torch.equal(before, flow.model.scene_type_embedder.embedding_table.weight))
                        flow.eval()
                        generated = flow.sample(agent, feature, steps=4)
                        self.assertEqual(generated.shape, clean.shape)
                        self.assertTrue(torch.isfinite(generated).all())
                        torch.testing.assert_close(generated[agent['ego_mask']], clean[agent['ego_mask']],
                                                   atol=0, rtol=0)
                        torch.testing.assert_close(feature['pt_token'], original, atol=0, rtol=0)

    def test_embedding_dtype_preserves_reference_values_and_gradients(self):
        _, agent, _ = self.inputs()
        for dtype in (torch.float64, torch.bfloat16):
            model = self.denoiser(map_id_source='metadata', map_lg_type=None,
                                  map_label_dropout=0.).to(dtype=dtype)
            actual = model._embed_scene_type(agent)
            expected = model.scene_type_embedder(torch.tensor([0, 3, 0]), train=True)
            self.assertEqual(actual.dtype, dtype)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.square().sum().backward()
            gradient = model.scene_type_embedder.embedding_table.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_wrapper_ema_scene_weights_survive_strict_checkpoint_roundtrip(self):
        source = self.wrapper(use_ema=True, ema_decay=.9, map_id_source='metadata',
                              map_lg_type=None, map_label_dropout=0.)
        self.assertEqual(source.map_embedding_type, 'scenario_dreamer')
        self.assertEqual(source.G1.model.map_id_source, 'metadata')
        self.assertIsNone(source.G1.model.map_lg_type)
        key = 'model.scene_type_embedder.embedding_table.weight'
        self.assertIn(key, source.get_extra_state()['ema_parameter_names'])
        with torch.no_grad():
            source.G1.model.scene_type_embedder.embedding_table.weight.add_(.05)
        source.update_ema()
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        target = self.wrapper(use_ema=True, ema_decay=.7, map_id_source='metadata',
                              map_lg_type=None, map_label_dropout=0.)
        loaded = target.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        self.assertEqual(target.ema.decay, .9)
        self.assertEqual(target.ema.num_updates, 1)
        for actual, expected in zip(target.ema.shadow_params, source.ema.shadow_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        _, agent, _ = self.inputs('speed')
        online = copy.deepcopy(target.G1.state_dict())
        with source.ema.average_parameters(source.G1.parameters()):
            expected = source.G1.model._embed_scene_type(agent)
        with target.ema.average_parameters(target.G1.parameters()):
            torch.testing.assert_close(target.G1.model._embed_scene_type(agent), expected, atol=0, rtol=0)
        for name, value in online.items():
            torch.testing.assert_close(target.G1.state_dict()[name], value, atol=0, rtol=0)

    def test_wrapper_enables_generic_metadata_forwarding_only_when_required(self):
        for options, expected in ((dict(map_id_source='metadata'), True),
                                  (dict(map_lg_type=None), True),
                                  (dict(map_id_source='fixed', map_lg_type=0), False),
                                  (dict(map_embedding_type='none', map_id_source='metadata'), False)):
            with self.subTest(options=options):
                model = self.wrapper(**options)
                self.assertEqual(getattr(model.token_processor, 'init_map_id_conditioning', False),
                                 expected)

    def test_model_defaults_and_train_eval_modes_are_explicit_in_hydra_config(self):
        root = Path(__file__).resolve().parents[1]
        generic = OmegaConf.load(root/'configs/model/smart.yaml').model_config.decoder.init_diffusion
        self.assertEqual(generic.map_embedding_type, 'none')
        self.assertEqual(generic.map_id_source, 'fixed')
        self.assertEqual(generic.map_id, 0)
        self.assertEqual(generic.map_lg_type, 0)
        self.assertEqual(generic.map_label_dropout, .1)
        OmegaConf.register_new_resolver('sim_root', lambda: str(root), replace=True)
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            for experiment in ('init_diffusion_lane_conditioned', 'init_diffusion_lane_conditioned_eval'):
                for source in ('fixed', 'metadata'):
                    config = compose(config_name='run.yaml', overrides=[f'experiment={experiment}',
                        'model.model_config.decoder.init_diffusion.map_embedding_type=scenario_dreamer',
                        f'model.model_config.decoder.init_diffusion.map_id_source={source}',
                        'model.model_config.decoder.init_diffusion.map_id=1',
                        'model.model_config.decoder.init_diffusion.map_lg_type=null',
                        'model.model_config.decoder.init_diffusion.map_label_dropout=0.25'])
                    actual = config.model.model_config.decoder.init_diffusion
                    self.assertEqual(actual.map_embedding_type, 'scenario_dreamer')
                    self.assertEqual(actual.map_id_source, source)
                    self.assertEqual(actual.map_id, 1)
                    self.assertIsNone(actual.map_lg_type)
                    self.assertEqual(actual.map_label_dropout, .25)


if __name__ == '__main__':
    unittest.main()
