"""CPU-only fixtures: no native best files, torch import or model construction."""
from __future__ import annotations

import copy
import contextlib
import hashlib
from functools import lru_cache
import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from omegaconf import OmegaConf

from tools import materialize_overnight_init_diffusion_best as best


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def source_schema():
    # Repo configs provide the schema; ignored historical logs are not required.
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver('sim_root', lambda: str(best.ROOT), replace=True)
    with initialize_config_dir(config_dir=str(best.ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='run', overrides=['experiment=init_diffusion_lane_conditioned_frozen_map',
            'paths.output_dir=/tmp/materializer-fixture-not-created', 'paths.log_dir=/tmp/materializer-fixture-not-created'])
    cfg = best.plain(cfg)
    cfg['action'] = 'fit'; cfg['ckpt_path'] = None
    cfg['data'].update(train_batch_size=512, val_batch_size=128, test_batch_size=128, shuffle=False,
                       num_workers=0, pin_memory=False, persistent_workers=False)
    init = cfg['model']['model_config']['decoder']['init_diffusion']
    init['normalizer_path'] = None
    for key in best.NEW_COMPATIBLE_OPTIONS: init.pop(key, None)  # explicit legacy constructor fixture
    return cfg


def source_fixture(seed=4817):
    cfg = copy.deepcopy(source_schema())
    cfg['seed'] = seed
    return cfg


def fixtures():
    cfg = source_fixture()
    model = cfg['model']['model_config']
    arm = {'name': 'control', 'batch': 512, 'recipe': copy.deepcopy(model['decoder']['init_diffusion']['component_loss']),
           'config': {}, 'map': model['initial_map_checkpoint'],
           'normalizer': str(best.ROOT / 'docs/experiments/init_diffusion_scratch_search_20261010/train_normalizer.json')}
    challenger = dict(copy.deepcopy(arm), name='novel', batch=256)
    planned = [arm, challenger]
    protocol = {'control': 'control', 'challenger': 'novel', 'train_seeds': list(best.TRAIN_SEEDS),
                'generation_seeds': list(best.GEN_SEEDS), 'full50000_no_reselection': True,
                'final_budget_B512': 4096, 'final_budget_B256': 8192,
                'actual_final_exposure_per_TRAIN': best.controller.actual_exposure(4096, 512)}
    rows = []
    for a in planned:
        budget = protocol['final_budget_B' + str(a['batch'])]
        for seed in best.TRAIN_SEEDS:
            for gen in best.GEN_SEEDS:
                rows.append({'variant': a['name'], 'seed': seed, 'eval_seed': gen, 'budget': budget,
                             'checkpoint': f'/fixture/{a["name"]}_{seed}/{a["name"]}/step_{budget}.ckpt',
                             'batch_size': a['batch'], 'seen_training_examples': protocol['actual_final_exposure_per_TRAIN'],
                             'evaluation_weights': 'ema', 'num_samples': 4096,
                             'map_frozen': True, 'initialization_scope': 'frozen_pretrained_map_fresh_G1',
                             'evaluation_only': False, 'resumed_from': None, 'resume_global_step': 0,
                             'optimizer_continued': False, 'actual_train_offset': 0, 'base_train_offset': 0,
                             'initial_generator_sha256': sha((a['name'], seed)), 'initial_map_sha256': sha('map'),
                             'train_order_sha256': sha(seed), 'train_manifest_sha256': sha('train'),
                             'calibration_sha256': sha('normalizer'), 'map_checkpoint_sha256': sha('map-file'),
                             'eval_manifest_sha256': sha('eval'), 'lr': model['lr'], 'gradient_clip_val': .5,
                             **{key: 1. if a['name'] == 'control' else .9 for key in best.controller.METRICS},
                             'generated_onroad_fraction': .8, 'num_generated_vehicles': 100., 'num_gt_vehicles': 100.})
    decision = {'complete_matrix': True, 'selected': 'novel', 'frozen_challenger': 'novel',
                'actual_final_exposure_per_TRAIN': protocol['actual_final_exposure_per_TRAIN'],
                'selected_arm': challenger, 'rows': rows}
    return decision, protocol, planned, cfg


def metadata_fixture(row, cfg):
    init = cfg['model']['model_config']['decoder']['init_diffusion']
    saved = {key: value for key, value in row.items() if key in (
        'seed', 'batch_size', 'base_train_offset', 'actual_train_offset', 'initialization_scope', 'resumed_from',
        'resume_global_step', 'optimizer_continued', 'map_frozen', 'calibration_sha256', 'map_checkpoint_sha256',
        'initial_generator_sha256', 'initial_map_sha256', 'train_manifest_sha256', 'train_order_sha256', 'eval_manifest_sha256')}
    extra = {'component_loss': copy.deepcopy(init['component_loss']), 'denoiser_options': dict(best.LEGACY_DENOISER_OPTIONS),
             'size_representation': init['size_representation'], 'ego_context_heading_encoding': init['ego_context_heading_encoding'],
             **{key: init[key] for key in ('generate_type', 'type_process', 'type_match_weight', 'type_noise_stats',
                                         'type_noise_std_floor', 'use_ego_embedding')},
             'ego_conditioning': {field: init.get('fix_ego_' + field) if init.get('fix_ego_' + field) is not None
                                 else init['fix_ego'] for field in best.EGO_FIELDS}}
    return {'global_step': row['budget'], 'ema_num_updates': row['budget'], 'ema_shadow_count': 12,
            'optimizer_step_min': row['budget'], 'optimizer_step_max': row['budget'],
            'scratch_sweep': dict(saved, saved_global_step=row['budget']), 'map_tensor_sha256': row['initial_map_sha256'],
            'normalizer_matches': True, 'cuda_initialized': False, 'shared_map_ema_absent': True, 'init_extra': extra}


class SelectionTest(unittest.TestCase):
    def test_complete_actual_matrix_selects_eligible_improvement(self):
        decision, protocol, arms, _ = fixtures()
        row = best.validate(decision, protocol, arms)
        self.assertEqual((row['variant'], row['seed'], row['eval_seed']), ('novel', 4817, 2817))

    def test_missing_duplicate_and_incomplete_rows_refused(self):
        for mutation in ('missing', 'duplicate', 'incomplete'):
            with self.subTest(mutation=mutation):
                d, p, a, _ = fixtures()
                if mutation == 'missing': d['rows'].pop()
                elif mutation == 'duplicate': d['rows'][-1] = copy.deepcopy(d['rows'][-2])
                else: d['complete_matrix'] = False
                with self.assertRaises(ValueError): best.validate(d, p, a)

    def test_primary_is_ratio_of_arm_means_not_mean_of_paired_ratios(self):
        d, _, _, _ = fixtures()
        for r in d['rows']:
            for k in best.controller.METRICS:
                r[k] = (1. if r['seed'] == 4817 else 3.) if r['variant'] == 'control' else (.5 if r['seed'] == 4817 else 3.)
        rows = d['rows']
        result = best.assessment([r for r in rows if r['variant'] == 'novel'], [r for r in rows if r['variant'] == 'control'])
        self.assertAlmostEqual(result['R'], .875)
        self.assertAlmostEqual(sum(r['R'] for r in result['per_training_seed'].values()) / 2, .75)
        self.assertEqual(len(result['per_training_seed']), 2)

    def test_pooled_pass_cannot_hide_independent_seed_gate_failure(self):
        d, p, a, _ = fixtures()
        for r in d['rows']:
            if r['variant'] == 'novel': r['speed_jsd'] = .8 if r['seed'] == 4817 else 1.2
        with self.assertRaisesRegex(ValueError, 'Declared selection'): best.validate(d, p, a)
        d.update(selected='control', selected_arm=a[0])
        self.assertEqual(best.validate(d, p, a)['variant'], 'control')

    def test_fallback_control_must_pass_coverage(self):
        d, p, a, _ = fixtures()
        for r in d['rows']: r['num_generated_vehicles'] = 130
        d.update(selected='control', selected_arm=a[0])
        with self.assertRaisesRegex(ValueError, 'fallback control'): best.validate(d, p, a)

    def test_budget_freshness_checkpoint_and_identity_fail_closed(self):
        mutations = {'batch_size': 512, 'budget': 8191, 'seen_training_examples': 1,
                     'evaluation_weights': 'raw', 'evaluation_only': True, 'resumed_from': '/old.ckpt',
                     'resume_global_step': 1, 'optimizer_continued': True, 'actual_train_offset': 1,
                     'base_train_offset': 1, 'map_frozen': False, 'checkpoint': '/fixture/step_8191.ckpt',
                     'initial_generator_sha256': None, 'train_order_sha256': sha('different')}
        for key, value in mutations.items():
            with self.subTest(key=key):
                d, p, a, _ = fixtures(); d['rows'][-1][key] = value
                with self.assertRaises(ValueError): best.validate(d, p, a)

    def test_actual_planned_arm_and_protocol_cannot_be_tampered(self):
        d, p, a, _ = fixtures(); d['selected_arm'] = dict(a[1], lr=.001)
        with self.assertRaisesRegex(ValueError, 'selected-arm'): best.validate(d, p, a)
        d, p, a, _ = fixtures(); p['actual_final_exposure_per_TRAIN'] += 1
        with self.assertRaisesRegex(ValueError, 'exposure'): best.validate(d, p, a)
        d, p, a, _ = fixtures(); a[1]['batch'] = 128
        with self.assertRaisesRegex(ValueError, 'batch'): best.validate(d, p, a)


class ConfigCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.d, self.p, self.arms, self.cfg = fixtures()
        self.row = self.d['rows'][0]

    def test_actual_source_exactly_inherits_base_optimizer_and_planned_options(self):
        best.validate_source_config(self.cfg, self.arms[0], self.row, self.cfg)
        base = copy.deepcopy(self.cfg)
        arm = copy.deepcopy(self.arms[0]); arm['config'] = {best.controller.PREFIX + 'speed_head_kind': 'detached',
                                                        best.controller.PREFIX + 'map_condition_types': 'lanes_and_edges'}
        changed = OmegaConf.create(self.cfg)
        for key, value in arm['config'].items(): OmegaConf.update(changed, key, value, force_add=True)
        best.validate_source_config(changed, arm, self.row, base)
        changed.model.model_config.optimizer = 'scenario_dreamer'
        with self.assertRaisesRegex(ValueError, 'Full actual model'): best.validate_source_config(changed, arm, self.row, base)

    def test_source_freshness_map_only_precision_and_loss(self):
        changes = [('ckpt_path', '/resume.ckpt'), ('model.model_config.finetune', False),
                   ('model.model_config.decoder.sep_map', True), ('trainer.precision', '16-mixed'),
                   ('model.model_config.decoder.init_diffusion.normalizer_path', '/other.json'),
                   ('model.model_config.decoder.init_diffusion.component_loss.components.velocity.weight', 99)]
        for key, value in changes:
            with self.subTest(key=key):
                cfg = OmegaConf.create(self.cfg); OmegaConf.update(cfg, key, value)
                with self.assertRaises(ValueError): best.validate_source_config(cfg, self.arms[0], self.row, self.cfg)

    def test_checkpoint_actual_metadata_and_legacy_options(self):
        m = metadata_fixture(self.row, self.cfg)
        best.validate_checkpoint_metadata(m, self.row, self.arms[0], self.cfg)
        m['init_extra'].pop('denoiser_options')
        best.validate_checkpoint_metadata(m, self.row, self.arms[0], self.cfg)
        # Match ScratchDiagnostics' real persisted subset, not invented fields.
        m['scratch_sweep'] = {key: value for key, value in m['scratch_sweep'].items()
                             if key in best.CHECKPOINT_PROVENANCE or key == 'saved_global_step'}
        best.validate_checkpoint_metadata(m, self.row, self.arms[0], self.cfg)
        sidecar = {key: self.row[key] for key in best.SIDECAR_PROVENANCE}
        best.validate_initialization_sidecar(sidecar, self.row)
        sidecar['actual_train_offset'] = 1
        with self.assertRaisesRegex(ValueError, 'sidecar'): best.validate_initialization_sidecar(sidecar, self.row)

    def test_actual_runner_last_checkpoint_name_is_allowed_but_budget_stays_strict(self):
        d, p, arms, _ = fixtures()
        for row in d['rows']: row['checkpoint'] = str(Path(row['checkpoint']).with_name('last.ckpt'))
        self.assertEqual(best.validate(d, p, arms)['budget'], 8192)

    def test_checkpoint_update_map_ema_normalizer_type_ego_and_architecture_drift_refused(self):
        changes = [('global_step', 1), ('ema_num_updates', 1), ('optimizer_step_max', 1), ('ema_shadow_count', 0),
                   ('scratch_sweep.initial_generator_sha256', sha('wrong')), ('map_tensor_sha256', sha('changed')),
                   ('normalizer_matches', False), ('cuda_initialized', True), ('shared_map_ema_absent', False),
                   ('init_extra.generate_type', False), ('init_extra.ego_conditioning.position', False),
                   ('init_extra.denoiser_options.denoiser_num_layers', 5),
                   ('init_extra.component_loss.components.velocity.weight', 99)]
        for key, value in changes:
            with self.subTest(key=key):
                m = metadata_fixture(self.row, self.cfg); cursor = m
                parts = key.split('.')
                for part in parts[:-1]: cursor = cursor[part]
                cursor[parts[-1]] = value
                with self.assertRaises(ValueError): best.validate_checkpoint_metadata(m, self.row, self.arms[0], self.cfg)

    def test_builder_fit_fresh_map_only_eval_clears_startup_and_preserves_new_features(self):
        cfg = copy.deepcopy(self.cfg)
        init = cfg['model']['model_config']['decoder']['init_diffusion']
        init.update(speed_head_kind='detached', state_feature_normalization='train', denoiser_hidden_dim=384,
                    denoiser_num_layers=5, map_condition_types='lanes_and_edges', time_sample_power=.5,
                    matching_weights={'velocity': 4.}, distribution_loss={'enabled': True})
        fit, test = best.build_configs(cfg, self.arms[0])
        actual = fit['model']['model_config']['decoder']['init_diffusion']
        for key in init:
            if key != 'normalizer_path': self.assertEqual(actual[key], init[key])
        self.assertIsNone(fit['ckpt_path']); self.assertTrue(fit['model']['model_config']['finetune'])
        self.assertIsNone(fit['data']['scenario_dreamer_train_sample_list'])
        self.assertIsNone(test['model']['model_config']['initial_map_checkpoint'])
        self.assertIsNone(test['model']['model_config']['decoder']['init_diffusion']['normalizer_path'])
        self.assertEqual(test['ckpt_path'], '${oc.env:' + best.CHECKPOINT_ENV + '}')
        self.assertEqual(fit['trainer']['max_epochs'], 128); self.assertEqual(fit['trainer']['max_steps'], -1)

    def test_real_hydra_compose_and_explicit_checkpoint_requirement(self):
        training, evaluation = best.build_configs(self.cfg, self.arms[0])
        fit, test = best.compose_configs(training, evaluation)
        self.assertTrue(fit['data']['shuffle'])
        self.assertEqual(fit['model']['model_config']['optimizer'], self.cfg['model']['model_config']['optimizer'])
        self.assertEqual(fit['model']['model_config']['lr'], self.cfg['model']['model_config']['lr'])
        self.assertEqual(test['model']['model_config']['sd_expected_scenes'], 50000)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception): best.plain(OmegaConf.create({'ckpt_path': evaluation['ckpt_path']}))
        with patch.dict(os.environ, {best.CHECKPOINT_ENV: '/saved.ckpt'}):
            self.assertEqual(best.plain(OmegaConf.create({'ckpt_path': evaluation['ckpt_path']}))['ckpt_path'], '/saved.ckpt')
        _, explicit = best.build_configs(self.cfg, self.arms[0], eval_checkpoint='/explicit.ckpt')
        self.assertEqual(explicit['ckpt_path'], '/explicit.ckpt')
        self.assertIn('optimum has not been established', best.config_text(training))

    def test_late_optimizer_override_and_condition_drift_caught_by_compose(self):
        training, evaluation = best.build_configs(self.cfg, self.arms[0])
        evaluation['model']['model_config']['decoder']['init_diffusion']['fix_ego_heading'] = False
        with self.assertRaisesRegex(ValueError, 'conditioning'): best.compose_configs(training, evaluation)


class EvidenceAndOutputTest(unittest.TestCase):
    def test_verify_all_four_actual_stages_before_materialization(self):
        d, p, arms, base = fixtures()
        with tempfile.TemporaryDirectory(prefix='materializer-evidence-fixture-') as tmp:
            root = Path(tmp); mapfile = root / 'map.ckpt'; normfile = root / 'norm.json'
            mapfile.write_text('dummy map'); normfile.write_text('{}')
            returns, configs = {}, {}
            for arm in arms:
                arm.update(map=str(mapfile), normalizer=str(normfile))
                for seed in best.TRAIN_SEEDS:
                    stage = root / f'{arm["name"]}_{seed}'; folder = stage / arm['name']; folder.mkdir(parents=True)
                    rows = [r for r in d['rows'] if r['variant'] == arm['name'] and r['seed'] == seed]
                    cfg = copy.deepcopy(base); cfg['seed'] = seed; cfg['data']['train_batch_size'] = arm['batch']
                    cfg['model']['model_config']['initial_map_checkpoint'] = str(mapfile)
                    checkpoint = folder / f'step_{rows[0]["budget"]}.ckpt'; checkpoint.write_text('fixture only')
                    manifests = [(stage / 'train_order.txt', 'train_order_sha256', str(seed)),
                                 (folder / 'train_files.txt', 'train_manifest_sha256', str(seed)),
                                 (stage / 'eval_files.txt', 'eval_manifest_sha256', 'evaluation')]
                    for path, _, content in manifests: path.write_text(content + '\n')
                    cfg['data']['scenario_dreamer_train_sample_list'] = str(folder / 'train_files.pkl')
                    cfg['data']['scenario_dreamer_eval_set'] = str(stage / 'eval_files.pkl')
                    for row in rows:
                        row.update(checkpoint=str(checkpoint), map_checkpoint_sha256=best.file_hash(mapfile), calibration_sha256=best.file_hash(normfile))
                        row.update({key: best.file_hash(path) for path, key, _ in manifests})
                    (folder / 'initialization.json').write_text(json.dumps({key: rows[0][key] for key in best.SIDECAR_PROVENANCE}))
                    (folder / 'config.yaml').write_text(yaml.safe_dump(cfg)); (stage / 'base_config.yaml').write_text(yaml.safe_dump(cfg))
                    (stage / 'protocol.json').write_text(json.dumps({'cyclic_gc_policy': 'synthetic fixture', 'young_gc_interval': 128}))
                    recipe, variants = stage / 'recipe.json', stage / 'variants.json'
                    recipe.write_text(json.dumps(arm['recipe']))
                    variant = {arm['name']: {'__config': arm['config'], '__lr': arm.get('lr', .0005)}}
                    variants.write_text(json.dumps(variant))
                    command = ['python', 'tools/overnight_init_diffusion_sweep.py', '--allow-multiple-epochs',
                               '--map-checkpoint', str(mapfile), '--normalizer-json', str(normfile),
                               '--batch-size', str(arm['batch']), '--scenes', '4096', '--seed', str(seed),
                               '--steps', str(rows[0]['budget']), '--variants', arm['name'],
                               '--evaluation-weights', 'ema', '--eval-seed', '2817', '--extra-eval-seed', '3817',
                               '--recipe', str(recipe), '--variant-file', str(variants)]
                    execution = {'status': 'success', 'returncode': 0,
                                 'arm': copy.deepcopy(arm), 'seed': seed, 'command': command,
                                 'fingerprints': {'design_inputs': {str(recipe): best.controller.digest(arm['recipe']),
                                                                 str(variants): best.controller.digest(variant)}},
                                 'completion_spec': {'name': arm['name'], 'seed': seed, 'steps': rows[0]['budget'],
                                                     'batch': arm['batch'], 'scenes': 4096, 'generations': list(best.GEN_SEEDS)},
                                 'artifacts': {'fixture': 'unchanged'}, 'source_hashes': {'configs/model/smart.yaml': sha('source')}}
                    (stage / 'execution.json').write_text(json.dumps(execution))
                    returns[str(stage)] = (rows, execution['artifacts']); configs[str(checkpoint)] = cfg
            d['selected_arm'] = arms[1]
            reader = lambda path, _: metadata_fixture(next(r for r in d['rows'] if r['checkpoint'] == str(path)), configs[str(path)])
            with patch.object(best, 'verify_completion', side_effect=lambda stage, *_: returns[str(stage)]) as verified, \
                 patch.object(best.controller, 'row_cohort', return_value={'physical': 'identical'}):
                row, cfg, audits = best.verify_evidence(d, p, arms, verify_current_source=False,
                    checkpoint_reader=reader)
                self.assertEqual(verified.call_count, 4); self.assertEqual(len(audits), 4)
                self.assertEqual(row['variant'], 'novel'); self.assertEqual(cfg.seed, 4817)
                execution_path = root / 'control_5817/execution.json'
                execution = json.loads(execution_path.read_text())
                incorrect = copy.deepcopy(execution); incorrect['command'][incorrect['command'].index('--steps') + 1] = '1'
                with self.assertRaisesRegex(ValueError, 'command differs'): best.validate_execution_plan(incorrect, arms[0],
                    next(r for r in d['rows'] if r['variant'] == 'control' and r['seed'] == 5817))
                execution['status'] = 'failed'
                execution_path.write_text(json.dumps(execution))
                with self.assertRaisesRegex(ValueError, 'completed stage'):
                    best.verify_evidence(d, p, arms, verify_current_source=False, checkpoint_reader=reader)

    def test_cli_dry_run_writes_no_real_config_and_incomplete_decision_exits_before_outputs(self):
        d, p, arms, cfg = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [directory / name for name in ('decision.json', 'frozen.json', 'search.json')]
            for path, value in zip(paths, (d, p, {'arms': arms})): path.write_text(json.dumps(value))
            args = ['--decision', str(paths[0]), '--frozen-protocol', str(paths[1]), '--search-protocol', str(paths[2]), '--dry-run']
            with patch.object(best, 'verify_evidence', return_value=(d['rows'][-2], OmegaConf.create(cfg), [])), \
                 patch.object(best, 'write_outputs') as output, contextlib.redirect_stdout(io.StringIO()):
                artifact = best.main(args)
                output.assert_not_called()
                self.assertFalse(artifact['fit_resumes_evidence_checkpoint'])
                self.assertFalse(artifact['full128epoch_optimum_proven'])
                self.assertEqual(artifact['independent_confirmations'], 1)
            d['complete_matrix'] = False; paths[0].write_text(json.dumps(d))
            with patch.object(best, 'write_outputs') as output, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result: best.main(args)
                self.assertEqual(result.exception.code, 2); output.assert_not_called()

    def test_output_preflight_is_non_destructive_and_refusal_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.write_text('existing')
            with self.assertRaises(FileExistsError): best.write_outputs({second: 'new', first: 'different'})
            self.assertFalse(second.exists()); self.assertEqual(first.read_text(), 'existing')
            best.write_outputs({first: 'different', second: 'new'}, overwrite=True)
            self.assertEqual(first.read_text(), 'different'); self.assertEqual(second.read_text(), 'new')


if __name__ == '__main__':
    unittest.main()
