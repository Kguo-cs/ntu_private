"""CPU fixtures for the separate native TEST sampler driver (no torch imports)."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import evaluate_init_diffusion_sampler as driver


class FakeFlow:
    training = False
    use_sde = False

    def __init__(self):
        self.t_eps = .05

    def sample(self, agent, lane, steps=20):
        return {'steps': steps, 'eps': self.t_eps}


def fixture_model(steps=40, updates=20480):
    flow = FakeFlow()
    decoder = NS(G1=flow, sampling_steps=steps, use_ema=True, ema=NS(num_updates=updates),
                 component_loss={'components': {'speed': {'kind': 'mse', 'time_power': 2}}})
    return NS(training=False, encoder=NS(init_decoder=decoder)), NS(state=NS(fn='TrainerFn.TESTING'),
                world_size=1, precision='32-true', datamodule=NS(test_batch_size=128))


def fixture_cell():
    return {'TRAIN_seed': 4817, 'GEN_seed': 2817, 'option': 'n40_eps0001', 'steps': 40, 'eps': 1e-4,
            'scenes': 4096, 'eval_files_sha256': 'a' * 64, 'checkpoint': '/explicit/actual.ckpt',
            'evaluation_batch_size': 128, 'main_endpoint_scenes': 4096,
            'checkpoint_sha256': 'b' * 64, 'source_config': '/actual/config.yaml',
            'source_config_sha256': 'c' * 64, 'trained_updates': 20480, 'source_model': 'separate_speed',
            'actual_training_updates_in_this_job': 0}


def fixture_report(cell):
    return {'reference_mode': 'matched', 'map_source': 'reference', 'num_samples': cell['scenes'],
            'num_gt_samples': cell['scenes'], 'unverified_generation_timestep_count': 0,
            'require_full_set': cell['scenes'] == 50000, 'full_membership': cell['scenes'] == 50000,
            'decoder': {'sampling_steps': cell['steps'], 'use_ema': True, 'ema_num_updates': cell['trained_updates']},
            'runtime': {'torch_initial_seed': cell['GEN_seed'], 'trainer_precision': '32-true',
                        'float32_matmul_precision': 'medium'},
            'agent_metrics': {k: .2 for k in driver.METRICS}, 'num_generated_vehicles': 23000,
            'num_gt_vehicles': 24000, 'feature_counts_generated': [23000, 18000],
            'feature_counts_gt': [24000, 19000]}


def fixture_audit(cell):
    return {'finished': True, 'sample_calls': 32, 'actual_sampling_steps': [cell['steps']],
            'effective_steps': cell['steps'], 'effective_eps': cell['eps'], 'actual_seed_after_reset': cell['GEN_seed'],
            'training_executed': False, 'training_loss_config_unchanged': True,
            'EMA_num_updates': cell['trained_updates'], 'completed_test_batches': (cell['scenes'] + 127) // 128,
            'actual_evaluation_batch_size': cell['evaluation_batch_size'],
            'runtime_loader_source_hashes': driver.runtime_loader_sources(),
            'loader_cache_at_test_end': {'classes_created': 1, 'class_reuses': cell['scenes'], 'cached_loader_kinds': ['_load']},
            'runtime_policy': driver.RUNTIME_POLICY}


class SamplerHookTests(unittest.TestCase):
    def test_post_load_test_override_seed_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, trainer = fixture_model()
            decoder = model.encoder.init_decoder
            before = copy.deepcopy(decoder.component_loss)
            order = []

            def reset(seed):
                self.assertEqual(decoder.G1.t_eps, 1e-4)
                order.append(('reset_after_checkpoint_restore', seed, decoder.ema.num_updates))

            hook = driver.InferenceSamplerHookCore(2817, 40, 1e-4, 20480, Path(temporary) / 'audit.json')
            with patch.object(driver, 'runtime_seed', reset), patch.object(driver, 'runtime_seed_value', return_value=2817):
                hook.on_test_start(trainer, model)
            self.assertEqual(order, [('reset_after_checkpoint_restore', 2817, 20480)])
            self.assertEqual(decoder.G1.sample(None, None, 40), {'steps': 40, 'eps': 1e-4})
            self.assertEqual(decoder.G1.sample(None, None, steps=40), {'steps': 40, 'eps': 1e-4})
            for i in range(32):
                hook.on_test_batch_end(trainer, model, None, None, i)
            hook.on_test_end(trainer, model)
            self.assertNotIn('sample', decoder.G1.__dict__)
            self.assertEqual(decoder.G1.t_eps, .05)
            self.assertEqual(decoder.component_loss, before)
            audit = driver.read(Path(temporary) / 'audit.json')
            self.assertTrue(audit['finished'])
            self.assertEqual(audit['actual_sampling_steps'], [40])
            self.assertEqual(audit['sample_calls'], 2)

    def test_training_and_validation_are_explicitly_forbidden(self):
        hook = driver.InferenceSamplerHookCore(2817, 20, .05, 10240, '/unused/audit.json')
        for name in ('on_fit_start', 'on_train_start', 'on_validation_start'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                getattr(hook, name)(None, None)

    def test_constructor_steps_must_survive_checkpoint_load(self):
        # A restored checkpoint may restore EMA but must never replace the
        # requested inference steps. A contrary loader is rejected before RNG.
        model, trainer = fixture_model(steps=20)
        hook = driver.InferenceSamplerHookCore(2817, 40, .05, 20480, '/unused/audit.json')
        with patch.object(driver, 'runtime_seed') as reset, self.assertRaisesRegex(ValueError, 'restoration'):
            hook.on_test_start(trainer, model)
        reset.assert_not_called()
        tree = ast.parse((driver.ROOT / 'src/smart/diffusion/initial_diffusion.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InitDiffusion')
        for name in ('get_extra_state', 'set_extra_state', '_load_from_state_dict'):
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertNotIn('sampling_steps', {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)})
            self.assertNotIn('sampling_steps', {n.value for n in ast.walk(fn)
                                              if isinstance(n, ast.Constant) and isinstance(n.value, str)})

    def test_wrong_ema_budget_or_non_test_mode_rejected(self):
        for field, value in (('ema_budget', 3), ('mode', 'TrainerFn.FITTING'), ('precision', '16-mixed')):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(field=field):
                model, trainer = fixture_model()
                if field == 'ema_budget':
                    model.encoder.init_decoder.ema.num_updates = value
                elif field == 'mode':
                    trainer.state.fn = value
                else:
                    trainer.precision = value
                hook = driver.InferenceSamplerHookCore(2817, 40, .05, 20480, Path(temporary) / 'audit.json')
                with patch.object(driver, 'runtime_seed') as reset, self.assertRaises(ValueError):
                    hook.on_test_start(trainer, model)
                reset.assert_not_called()

    def test_wrong_actual_steps_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, trainer = fixture_model()
            hook = driver.InferenceSamplerHookCore(2817, 40, .05, 20480, Path(temporary) / 'audit.json')
            with patch.object(driver, 'runtime_seed'), patch.object(driver, 'runtime_seed_value', return_value=2817):
                hook.on_test_start(trainer, model)
            with self.assertRaisesRegex(ValueError, 'sampler'):
                model.encoder.init_decoder.G1.sample(None, None, 20)

    def test_no_manual_gc_at_any_test_batch_boundary(self):
        hook = driver.InferenceSamplerHookCore(2817, 20, .05, 10240, '/unused/audit.json', defer_gc=True)
        with patch.object(driver.gc, 'collect') as collect, patch.object(driver.gc, 'freeze') as freeze:
            for i in range(129):
                hook.on_test_batch_end(None, None, None, None, i)
            collect.assert_not_called()
            freeze.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'batch ordering'):
            hook.on_test_batch_end(None, None, None, None, 4)


class ScopeAndCompletionTests(unittest.TestCase):
    def test_main_incomplete_is_rejected_before_checkpoint_payload_or_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            driver.save(path / 'queue_complete.json', {'status': 'PARTIAL_DEADLINE',
                'complete_final_matrix': False, 'complete_full50000_audit': False})
            with self.assertRaisesRegex(ValueError, 'not actually complete'):
                driver.verify_main(path)
        self.assertNotIn('torch', sys.modules)

    def test_exact_grid_plan_is_20_auxiliary_cells_and_manifest_order_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            driver.save(path / 'final_selection.json', {'fixture': 'no checkpoint payload'})
            names = [f'native_{i:05d}.pt' for i in range(4096)]
            driver.save(path / 'manifest.json', {'files': names})
            sources = {seed: {'row': {'budget': 20480, 'eval_manifest_sha256': driver.files_hash(names)},
                'rows': [{'eval_seed': gen, **{k: .2 for k in driver.METRICS}} for gen in driver.GEN_SEEDS],
                'cfg': {'data': {'test_raw_dir': '/CPU_fixture_native', 'test_batch_size': 128}}, 'checkpoint': '/actual.ckpt',
                'checkpoint_sha256': 'a' * 64, 'source_config': '/actual.yaml',
                'source_config_sha256': 'b' * 64, 'model_source_sha256': {'fixture.py': 'c' * 64}}
                for seed in driver.TRAIN_SEEDS}
            args = NS(main_art=path, stage='grid', manifest=path / 'manifest.json')
            with patch.object(driver, 'verify_main', return_value=({'selected': 'separate_speed'}, sources)), \
                 patch.object(Path, 'is_file', return_value=True):
                plan = driver.make_plan(args)
                self.assertEqual(len(plan['cells']), 20)
                self.assertEqual(plan['files'], names)
                self.assertEqual({(v['TRAIN_seed'], v['GEN_seed']) for v in plan['cells']},
                                 {(s, g) for s in driver.TRAIN_SEEDS for g in driver.GEN_SEEDS})
                self.assertTrue(plan['auxiliary_inference_cohort'])
                self.assertEqual(plan['independent_new_training_runs'], 0)
                self.assertEqual(plan['cells'][0]['main_endpoint_metrics']['speed_jsd'], .2)
                self.assertTrue(all(cell['option'] == 'n20_eps005' for cell in plan['cells'][:4]))
                self.assertEqual(plan['cells'][0]['evaluation_batch_size'], 128)
                driver.save(args.manifest, {'files': list(reversed(names))})
                with self.assertRaisesRegex(ValueError, 'scene order'):
                    driver.make_plan(args)

    def test_default20_seven_values_compare_in_native_units_without_sqrt(self):
        reference = {k: .04 for k in driver.METRICS}
        result = driver.compare_main_default20(reference, reference)
        self.assertTrue(result['exact_seven_metrics_identical'])
        self.assertTrue(result['within_absolute_tolerance'])
        actual = {**reference, 'speed_jsd': .09}
        result = driver.compare_main_default20(actual, reference)
        self.assertFalse(result['exact_seven_metrics_identical'])
        self.assertFalse(result['within_absolute_tolerance'])
        self.assertAlmostEqual(result['delta_native_scaled_units']['speed_jsd'], .05)
        self.assertEqual(len(result['delta_native_scaled_units']), 7)
        almost = {**reference, 'speed_jsd': .04 + 5e-7}
        self.assertTrue(driver.compare_main_default20(almost, reference)['within_absolute_tolerance'])

    def test_selected_control_source_keeps_shared_head_and_its_10240_budget(self):
        source = {'paths': {}, 'data': {}, 'trainer': {}, 'model': {'model_config': {
            'initial_map_checkpoint': '/map.ckpt', 'decoder': {'init_diffusion': {
                'use_ema': True, 'sampling_steps': 20, 'speed_head_kind': 'shared',
                'ego_context_heading_encoding': 'angle', 'size_representation': 'linear'}}}}}
        cell = fixture_cell(); cell.update(source_model='control', trained_updates=10240)
        cfg = driver.build_eval_config(source, cell, Path('/cell'), Path('/manifest.pkl'))
        init = cfg['model']['model_config']['decoder']['init_diffusion']
        self.assertEqual(init['speed_head_kind'], 'shared')
        self.assertEqual(init['ego_context_heading_encoding'], 'angle')
        self.assertEqual(cfg['callbacks']['sampler_audit']['expected_updates'], 10240)

    def test_full_stage_rejects_missing_freeze_and_partial_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            args = NS(main_art=path, stage='full', frozen_sampler=None, full_train_seeds=[4817])
            with patch.object(driver, 'verify_main', return_value=({'selected': 'separate_speed'}, {})):
                with self.assertRaisesRegex(ValueError, 'explicit frozen'):
                    driver.make_plan(args)
                grid = path / 'grid.json'
                driver.save(grid, {'complete': False, 'rows': []})
                args.frozen_sampler = path / 'frozen.json'
                driver.save(args.frozen_sampler, {'selection_is_frozen': True, 'selected_option': 'n40_eps0001',
                                                  'grid_summary': str(grid), 'grid_summary_sha256': driver.file_hash(grid)})
                with self.assertRaisesRegex(ValueError, 'actual complete inference grid'):
                    driver.make_plan(args)

    def test_builder_preserves_conditions_and_clears_startup_initializers(self):
        source = {'action': 'fit', 'ckpt_path': None, 'seed': 4817, 'paths': {'output_dir': '/old', 'log_dir': '/old'},
                  'model': {'_target_': 'src.smart.model.smart_gail.SMART_GAIL', 'model_config': {
                      'initial_map_checkpoint': '/only_map.ckpt', 'finetune': True, 'sd_reference_mode': 'matched',
                      'token_processor': {'init_map_range': 50}, 'decoder': {'init_decoder': 'flow', 'sep_map': False,
                      'init_diffusion': {'normalizer_path': '/TRAIN.json', 'use_ema': True, 'sampling_steps': 20,
                          'fix_ego_position': True, 'fix_ego_heading': True, 'fix_ego_shape': True,
                          'fix_ego_velocity': False, 'fix_ego_type': True, 'generate_type': True,
                          'speed_head_kind': 'separate', 'velocity_representation': 'speed',
                          'component_loss': {'speed': {'kind': 'l1', 'time_power': 0, 'weight': 4}}}}}},
                  'data': {'test_raw_dir': '/native/full', 'shuffle': False}, 'trainer': {'max_steps': 20480},
                  'callbacks': {'training_only': {'_target_': 'unused.callback'}}}
        before = copy.deepcopy(source)
        cfg = driver.build_eval_config(source, fixture_cell(), Path('/cell'), Path('/cell/manifest.pkl'))
        self.assertEqual(source, before)
        init = cfg['model']['model_config']['decoder']['init_diffusion']
        expected = copy.deepcopy(before['model']['model_config']['decoder']['init_diffusion'])
        expected.update(normalizer_path=None, sampling_steps=40)
        self.assertEqual(init, expected)
        self.assertEqual(cfg['action'], 'test')
        self.assertIsNone(cfg['model']['model_config']['initial_map_checkpoint'])
        self.assertEqual(cfg['trainer']['precision'], '32-true')
        self.assertEqual(cfg['callbacks']['sampler_audit']['seed'], 2817)
        self.assertNotIn('t_eps', init)  # Not a constructor argument or training intervention.

    def test_native_scope_certification_rejects_partial_and_wrong_precision(self):
        cell = fixture_cell()
        driver.validate_native_report(fixture_report(cell), fixture_audit(cell), cell)
        for key, value in (('num_samples', 256), ('reference_mode', 'full'), ('map_source', 'generated')):
            report = fixture_report(cell)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.validate_native_report(report, fixture_audit(cell), cell)
        audit = fixture_audit(cell)
        audit['finished'] = False
        with self.assertRaises(ValueError):
            driver.validate_native_report(fixture_report(cell), audit, cell)
        report = fixture_report(cell)
        report['runtime']['trainer_precision'] = '16-mixed'
        with self.assertRaises(ValueError):
            driver.validate_native_report(report, fixture_audit(cell), cell)

    def test_full_requires_full_native_membership(self):
        cell = fixture_cell(); cell['scenes'] = 50000
        driver.validate_native_report(fixture_report(cell), fixture_audit(cell), cell)
        report = fixture_report(cell); report['full_membership'] = False
        with self.assertRaises(ValueError):
            driver.validate_native_report(report, fixture_audit(cell), cell)

    def test_parent_requires_successful_exit_even_if_child_wrote_valid_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'cell'; folder.mkdir()
            cell = fixture_cell()
            plan = {'driver_sha256': 'd' * 64, 'native_runner_sha256': 'e' * 64,
                    'runtime_loader_source_hashes': driver.runtime_loader_sources(),
                    'main_decision_sha256': 'f' * 64}
            report = fixture_report(cell)
            driver.save(folder / 'sd_agent_metrics.json', report)
            driver.save(folder / 'sampler_audit.json', fixture_audit(cell))
            row = {**cell, **plan, **report['agent_metrics'], 'evaluation_only': True, 'evaluation_weights': 'ema',
                   'auxiliary_inference_cohort': True, 'runtime_policy': driver.RUNTIME_POLICY,
                   'metrics_path': str(folder / 'sd_agent_metrics.json'),
                   'report_sha256': driver.file_hash(folder / 'sd_agent_metrics.json'),
                   'sampler_audit_sha256': driver.file_hash(folder / 'sampler_audit.json')}
            driver.save(folder / 'cell_summary.json', row)
            log = folder.parent / 'cell.log'; log.write_text('DONE_INFERENCE_ONLY ' + str(folder) + '\n')
            execution = {'status': 'failed', 'returncode': -11, 'cell': cell, 'driver_sha256': plan['driver_sha256'],
                         'runtime_loader_source_hashes': driver.runtime_loader_sources(),
                         'log_sha256': driver.file_hash(log), 'artifacts': {str(p): driver.file_hash(p) for p in
                             (folder / 'cell_summary.json', folder / 'sd_agent_metrics.json', folder / 'sampler_audit.json')}}
            ex = folder.parent / 'cell.execution.json'; driver.save(ex, execution)
            with self.assertRaisesRegex(ValueError, 'successful child'):
                driver.validate_completed_cell(row, cell, folder, plan, require_execution=True)
            execution.update(status='success', returncode=0); driver.save(ex, execution)
            driver.validate_completed_cell(row, cell, folder, plan, require_execution=True)
            report['agent_metrics']['speed_jsd'] = .3
            driver.save(folder / 'sd_agent_metrics.json', report)
            with self.assertRaisesRegex(ValueError, 'artifacts changed'):
                driver.validate_completed_cell(row, cell, folder, plan, require_execution=True)

    def test_all_explicit_gc_is_deferred_including_generation_zero(self):
        with patch.dict(os.environ, {driver.GC_ENV: '1'}), \
             patch.object(driver.gc, 'disable') as disable, patch.object(driver.gc, 'collect'):
            driver.defer_runtime_gc()
            self.assertEqual(driver.gc.collect(), 0)
            self.assertEqual(driver.gc.collect(2), 0)
            self.assertEqual(driver.gc.collect(1), 0)
            self.assertEqual(driver.gc.collect(0), 0)
            disable.assert_called_once_with()

    def test_parent_timeout_terminates_own_process_group_and_keeps_actual_return(self):
        process = NS(pid=12345, returncode=-2)
        process.poll = lambda: None
        process.wait = unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired('fixture', .1), -2])
        with tempfile.TemporaryDirectory() as temporary, patch.object(driver.subprocess, 'Popen', return_value=process) as popen, \
             patch.object(driver.os, 'killpg') as kill:
            execution = {'status': 'running'}
            with self.assertRaises(subprocess.TimeoutExpired):
                driver.wait_isolated_child(['CPU_fixture_no_GPU'], None, execution, Path(temporary) / 'execution.json', 0)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['env'][driver.GC_ENV], '1')
            kill.assert_called_once_with(12345, driver.signal.SIGINT)
            self.assertEqual(execution['returncode'], -2)
            self.assertTrue(execution['child_terminated_after_parent_interruption_or_timeout'])

    def test_planning_module_is_torch_free_and_grid_is_two_by_two(self):
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(len(driver.TRAIN_SEEDS) * len(driver.GEN_SEEDS) * len(driver.OPTIONS), 20)
        self.assertEqual(driver.OPTIONS['n20_eps005'], (20, .05))


if __name__ == '__main__':
    unittest.main()
