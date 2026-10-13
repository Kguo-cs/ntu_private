"""Independent stdlib-only fixtures for future N20 TEST registration.

These tests never deserialize a checkpoint, instantiate the model or use GPU.
The source collector is replaced with its documented, already-verified return
shape. All file/snapshot hashes, registrations, guards and scores are real.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import evaluate_init_diffusion_n20_power_grids as adapter


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='n20_registration_tests_')
        cls.root = Path(cls.tmp.name).resolve()
        cls.art = cls.root / 'art'
        cls.runs = cls.root / 'runs'
        cls.data = cls.root / 'data'
        cls.data.mkdir()
        cls.files = [f'{i:05}_validation.tfrecord_0_0_0.pt' for i in range(4096)]
        for name in cls.files:
            (cls.data / name).write_bytes(b'opaque scene fixture')
        cls.manifest = cls.root / 'eval4096_files.json'
        write_json(cls.manifest, cls.files)
        cls.calibration = cls.root / 'calibration.json'
        write_json(cls.calibration, {'normal_mean': [1.] * 7, 'normal_scale': [1.] * 7})
        cls.map = cls.root / 'fixed_map.ckpt'
        cls.map.write_bytes(b'opaque frozen map fixture')
        cls.source = cls.root / 'production.py'
        cls.source.write_text('# frozen source fixture\n')
        protocol = {'sources': {'production.py': adapter.sha(cls.source)}, 'plan_sha256': 'plan fixture'}
        write_json(cls.art / 'protocol.json', protocol)
        cls.identity = {'metric_scope': 'Scenario Dreamer initial-scene vehicle metrics',
                        'generation_task': 'map-conditioned; generated agents on official reference maps',
                        'map_source': 'reference', 'reference_mode': 'matched',
                        'reference_source_manifest_sha256': 'reference_manifest_fixture',
                        'official_source_sha256': {'official.py': 'official_fixture'},
                        'backend_scope': 'CPU fixture'}
        cls.membership = {'full_membership': False, 'require_full_set': False,
                          'reference_mode': 'matched',
                          'reference_source_manifest_sha256': 'reference_manifest_fixture'}
        rows, stages = [], []
        for seed in (4817, 5817):
            folder = cls.runs / f'medium_train{seed}'
            model_folder = folder / 'candidate'
            model_folder.mkdir(parents=True)
            checkpoint = model_folder / 'last.ckpt'
            checkpoint.write_bytes(f'opaque immutable terminal checkpoint TRAIN {seed}'.encode())
            cfg = {'model': {'model_config': {'decoder': {'init_decoder': 'flow',
                    'initial_scene_only': True, 'init_diffusion': {
                        **adapter.REVIEWED_RECIPE, 'sampling_steps': 20, 'use_ema': True}},
                    'sd_reference_mode': 'matched', 'sd_r_reference': None, 'sd_require_full_set': False}},
                    'data': {'test_batch_size': 128, 'test_raw_dir': str(cls.data)},
                    'trainer': {'precision': '32-true'},
                    'paths': {'output_dir': str(folder), 'log_dir': str(folder)}}
            write_json(model_folder / 'config.yaml', cfg)
            spec = {'variant': 'candidate', 'seed': seed, 'steps': 4096, 'scenes': 4096,
                    'gens': list(adapter.GEN_SEEDS), 'checkpoints': [2048], 'evaluation': None}
            stage_rows = []
            for gen in adapter.GEN_SEEDS:
                metric_path = model_folder / f'gen{gen}' / 'sd_agent_metrics.json'
                native = {**cls.identity, **cls.membership, 'num_samples': 4096,
                          'num_gt_samples': 4096, 'unverified_generation_timestep_count': 0,
                          'runtime': {'torch_initial_seed': gen, 'trainer_precision': '32-true',
                                      'float32_matmul_precision': 'medium'},
                          'decoder': {'ema_num_updates': 4096, 'sampling_steps': 20, 'use_ema': True},
                          'agent_metrics': {name: 1. for name in adapter.METRICS},
                          'num_generated_vehicles': 100, 'num_gt_vehicles': 100}
                write_json(metric_path, native)
                row = {'source': 'main', 'variant': 'candidate', 'budget': 4096, 'seed': seed,
                       'eval_seed': gen, 'num_samples': 4096, 'checkpoint': str(checkpoint),
                       'evaluation_weights': 'ema', 'eval_manifest_sha256': adapter.names_sha(cls.files),
                       'actual_train_offset': 0, 'resumed_from': None, 'map_frozen': True,
                       'generator_num_parameters': 8083978, 'calibration_sha256': adapter.sha(cls.calibration),
                       'map_checkpoint': str(cls.map), 'map_checkpoint_sha256': adapter.sha(cls.map),
                       'stage': folder.name, 'native_metric_identity': cls.identity,
                       'native_membership': cls.membership, 'metrics_path': str(metric_path),
                       'metrics_sha256': adapter.sha(metric_path), **{name: 1. for name in adapter.METRICS},
                       'generated_onroad_fraction': .9, 'num_generated_vehicles': 100,
                       'num_gt_vehicles': 100}
                stage_rows.append(row)
            write_json(folder / 'summary.json', stage_rows)
            write_json(folder / 'edge_contract.json', {'sources': protocol['sources'],
                        'plan_sha256': protocol['plan_sha256'], 'spec': spec})
            marker = {'status': 'VERIFIED_COMPLETE', 'returncode': 0,
                      'summary_sha256': adapter.sha(folder / 'summary.json'),
                      'contract_sha256': adapter.sha(folder / 'edge_contract.json'),
                      'metrics_reports': [{'path': r['metrics_path'], 'sha256': r['metrics_sha256']}
                                          for r in stage_rows]}
            write_json(folder / 'edge_verified_completion.json', marker)
            write_json(folder / 'verified_completion.json', marker)
            stage_info = {'source': 'main', 'stage': folder.name, 'spec': spec,
                          'summary_sha256': adapter.sha(folder / 'summary.json'),
                          'contract_sha256': adapter.sha(folder / 'edge_contract.json'),
                          'marker_sha256': adapter.sha(folder / 'edge_verified_completion.json')}
            for row in stage_rows:
                row['summary_sha256'] = stage_info['summary_sha256']
            rows.extend(stage_rows)
            stages.append(stage_info)
        cls.collection = {'source': 'main', 'protocol': protocol,
                          'protocol_sha256': adapter.sha(cls.art / 'protocol.json'),
                          'rows': rows, 'verified_stages': stages, 'quarantined': []}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.collections = {'main': copy.deepcopy(self.collection), 'compact': {'rows': []}}
        self.output = self.root / f'output_{self._testMethodName}'
        self.patchers = [mock.patch.object(adapter, 'ROOT', self.root),
                         mock.patch.object(adapter, 'SOURCES', {'main': (self.art, self.runs),
                                                              'compact': (self.art, self.runs)}),
                         mock.patch.object(adapter, 'CALIBRATION', self.calibration),
                         mock.patch.object(adapter, 'collect_sources', side_effect=lambda: self.collections),
                         mock.patch.object(adapter, 'code_sources', side_effect=lambda: {str(self.source): adapter.sha(self.source)})]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(lambda: shutil.rmtree(self.output, ignore_errors=True))

    def build(self, **overrides):
        values = dict(source='main', variant='candidate', budget=4096, train_seeds=[4817, 5817],
                      manifest=self.manifest, output=self.output,
                      deadline_utc='2099-01-01T00:00:00+00:00', min_cell_seconds=120)
        values.update(overrides)
        return adapter.build_registration(**values)

    def frozen(self):
        plan = self.build()
        return plan, adapter.freeze_registration(plan, self.output)

    def mutate_file(self, path, data):
        original = Path(path).read_bytes()
        Path(path).write_bytes(data)
        self.addCleanup(lambda: Path(path).write_bytes(original))

    def test_build_readonly_exact_matrix_never_queries_gpu_or_loads_checkpoint(self):
        with mock.patch.object(adapter, 'gpu_compute_pids', side_effect=AssertionError('GPU query in CPU build')), \
             mock.patch.object(adapter, 'live_search_processes', side_effect=AssertionError('Process query in CPU build')):
            plan = self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(plan['cells']), 16)
        self.assertEqual({(c['TRAIN_seed'], c['GEN_seed'], c['arm']) for c in plan['cells']},
                         {(train, gen, arm['arm']) for train in (4817, 5817)
                          for gen in adapter.GEN_SEEDS for arm in adapter.ARMS})
        self.assertTrue(all(c['sampling_steps'] == 20 and c['scenes'] == 4096
                            and c['evaluation_batch_size'] == 128 for c in plan['cells']))

    def test_default_cli_only_plans_and_never_executes_or_freezes(self):
        with mock.patch.object(adapter, 'execution_guard', side_effect=AssertionError('execution from dry-run')), \
             mock.patch.object(adapter, 'freeze_registration', side_effect=AssertionError('freeze from dry-run')):
            plan = adapter.main(['--variant', 'candidate', '--budget', '4096', '--output', str(self.output),
                                 '--manifest', str(self.manifest), '--deadline-utc', '2099-01-01T00:00:00+00:00'])
        self.assertEqual(len(plan['cells']), 16)
        self.assertFalse(self.output.exists())

    def test_unregistered_cli_execution_is_rejected(self):
        with mock.patch.object(adapter, 'execute_registration', side_effect=AssertionError('unregistered GPU execution')):
            with self.assertRaises(ValueError):
                adapter.main(['--execute', '--variant', 'candidate', '--output', str(self.output),
                              '--deadline-utc', '2099-01-01T00:00:00+00:00'])

    def test_historical_sources_and_wrong_budget_or_train_seeds_rejected(self):
        for changes in ({'source': 'historical'}, {'budget': 2048}, {'budget': True},
                        {'train_seeds': [4817, 4817]}, {'train_seeds': [9999]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(**changes)

    def test_missing_or_duplicate_gen_pair_rejected(self):
        rows = self.collections['main']['rows']
        rows.pop()
        with self.assertRaises(ValueError): self.build()
        rows.append(copy.deepcopy(rows[-1]))
        with self.assertRaises(ValueError): self.build()

    def test_partial_intermediate_or_quarantined_stage_rejected(self):
        original = copy.deepcopy(self.collections)
        for mode in ('partial', 'intermediate', 'quarantine'):
            self.collections = copy.deepcopy(original)
            collection = self.collections['main']
            if mode == 'partial': collection['verified_stages'] = []
            elif mode == 'intermediate': collection['verified_stages'][0]['spec']['steps'] = 10240
            else: collection['quarantined'] = [{'stage': collection['rows'][0]['stage']}]
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.build()

    def test_wrong_ema_resumed_unfrozen_map_or_parameter_cap_rejected(self):
        for name, value in (('evaluation_weights', 'online'), ('actual_train_offset', 2),
                            ('resumed_from', 'old.ckpt'), ('map_frozen', False),
                            ('generator_num_parameters', 8100001)):
            self.collections = {'main': copy.deepcopy(self.collection), 'compact': {'rows': []}}
            self.collections['main']['rows'][0][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError): self.build()

    def test_unverified_rolling_checkpoint_path_rejected(self):
        self.collections['main']['rows'][0]['checkpoint'] = str(self.root / 'rolling' / 'last.ckpt')
        with self.assertRaises(ValueError): self.build()

    def test_reordered_manifest_rejected(self):
        manifest = self.root / 'reversed_manifest.json'
        write_json(manifest, list(reversed(self.files)))
        with self.assertRaises(ValueError): self.build(manifest=manifest)

    def test_native_reference_identity_mismatch_rejected(self):
        self.collections['main']['rows'][1]['native_metric_identity']['map_source'] = 'generated'
        with self.assertRaises(ValueError): self.build()

    def test_saved_config_unreviewed_representation_rejected(self):
        path = Path(self.collections['main']['rows'][0]['checkpoint']).parent / 'config.yaml'
        config = adapter.read_config(path)
        config['model']['model_config']['decoder']['init_diffusion']['type_process'] = 'joint'
        self.mutate_file(path, json.dumps(config).encode())
        with self.assertRaises(ValueError): self.build()

    def test_stale_collector_protocol_marker_or_metric_hash_rejected_before_freeze(self):
        first = self.collections['main']['rows'][0]
        paths = (self.art / 'protocol.json',
                 self.runs / first['stage'] / 'edge_verified_completion.json',
                 self.runs / first['stage'] / 'edge_contract.json',
                 self.runs / first['stage'] / 'summary.json',
                 Path(first['metrics_path']))
        for changed in paths:
            with self.subTest(path=changed):
                original = changed.read_bytes()
                try:
                    changed.write_bytes(original + b' ')
                    with self.assertRaises(ValueError): self.build()
                finally:
                    changed.write_bytes(original)

    def test_terminal_collector_membership_disappearing_after_freeze_rejected(self):
        plan, path = self.frozen()
        self.collections['main']['verified_stages'] = []
        with self.assertRaises(ValueError): adapter.verify_registration(path)

    def test_source_change_between_build_and_freeze_rejected(self):
        plan = self.build()
        source = Path(plan['endpoint_sources'][0]['source_checkpoint'])
        self.mutate_file(source, b'changed before snapshot')
        with self.assertRaises(ValueError): adapter.freeze_registration(plan, self.output)

    def test_freeze_snapshots_exact_bytes_and_verifies(self):
        plan, path = self.frozen()
        verified = adapter.verify_registration(path)
        self.assertEqual(verified, plan)
        self.assertNotEqual(plan['endpoint_sources'][0]['source_checkpoint'], plan['cells'][0]['checkpoint'])
        for endpoint in plan['endpoint_sources']:
            self.assertEqual(adapter.sha(endpoint['checkpoint']), endpoint['source_checkpoint_sha256'])
        with self.assertRaises(ValueError): adapter.freeze_registration(plan, self.output)

    def test_snapshot_checkpoint_mutation_rejected(self):
        plan, path = self.frozen()
        Path(plan['cells'][0]['checkpoint']).write_bytes(b'changed snapshot')
        with self.assertRaises(ValueError): adapter.verify_registration(path)

    def test_snapshot_config_mutation_rejected(self):
        plan, path = self.frozen()
        Path(plan['cells'][0]['source_config']).write_text('{}')
        with self.assertRaises(ValueError): adapter.verify_registration(path)

    def test_source_checkpoint_or_config_mutation_rejected(self):
        for key in ('source_checkpoint', 'source_config'):
            with self.subTest(key=key):
                plan, path = self.frozen()
                source = Path(plan['endpoint_sources'][0][key]); before = source.read_bytes()
                try:
                    source.write_bytes(b'changed source')
                    with self.assertRaises(ValueError): adapter.verify_registration(path)
                finally:
                    source.write_bytes(before); shutil.rmtree(self.output)

    def test_data_source_calibration_or_evidence_mutation_rejected(self):
        for changed in (self.data / self.files[0], self.source, self.calibration,
                        self.runs / 'medium_train4817' / 'edge_verified_completion.json'):
            with self.subTest(path=changed):
                plan, path = self.frozen(); original = changed.read_bytes()
                try:
                    changed.write_bytes(b'changed registered dependency')
                    with self.assertRaises(ValueError): adapter.verify_registration(path)
                finally:
                    changed.write_bytes(original); shutil.rmtree(self.output)

    def test_registration_hash_and_cells_cannot_be_changed(self):
        plan, path = self.frozen()
        plan['cells'].pop(); write_json(path, plan)
        with self.assertRaises(ValueError): adapter.verify_registration(path)
        marker_path = path.parent / 'registration_frozen.json'
        marker = adapter.read(marker_path); marker['registration_sha256'] = adapter.sha(path)
        write_json(marker_path, marker)
        with self.assertRaises(ValueError): adapter.verify_registration(path)


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'deadline_utc': datetime.fromtimestamp(1600, timezone.utc).isoformat(),
                     'min_cell_seconds': 120}

    def test_idle_guard_permits_only_owned_gpu_pid_and_sufficient_budget(self):
        with mock.patch.object(adapter, 'live_search_processes', return_value=[]), \
             mock.patch.object(adapter, 'gpu_compute_pids', return_value=[123]), \
             mock.patch.object(adapter.time, 'time', return_value=1000):
            self.assertEqual(adapter.execution_guard(self.plan, own_pid=123), 600)
            with self.assertRaises(ValueError): adapter.execution_guard(self.plan)

    def test_live_queue_always_rejected_before_gpu_query(self):
        with mock.patch.object(adapter, 'live_search_processes', return_value=[{'pid': 9, 'state': 'S'}]), \
             mock.patch.object(adapter, 'gpu_compute_pids') as gpu:
            with self.assertRaises(ValueError): adapter.execution_guard(self.plan)
            gpu.assert_not_called()

    def test_other_gpu_process_rejected(self):
        with mock.patch.object(adapter, 'live_search_processes', return_value=[]), \
             mock.patch.object(adapter, 'gpu_compute_pids', return_value=[123, 456]):
            with self.assertRaises(ValueError): adapter.execution_guard(self.plan, own_pid=123)

    def test_failed_gpu_query_is_not_treated_as_idle(self):
        with mock.patch.object(adapter, 'live_search_processes', return_value=[]), \
             mock.patch.object(adapter, 'gpu_compute_pids', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')):
            with self.assertRaises(subprocess.CalledProcessError): adapter.execution_guard(self.plan)

    def test_insufficient_deadline_and_naive_timestamp_rejected(self):
        with mock.patch.object(adapter, 'live_search_processes', return_value=[]), \
             mock.patch.object(adapter, 'gpu_compute_pids', return_value=[]), \
             mock.patch.object(adapter.time, 'time', return_value=1481):
            with self.assertRaises(ValueError): adapter.execution_guard(self.plan)
        with self.assertRaises(ValueError): adapter.parse_deadline('2026-10-13T02:00:00')

    def test_gpu_query_parser_fails_closed_on_unknown_output(self):
        for text in ('N/A\n', 'No running processes found\n', '12\nnot-a-pid\n'):
            with self.subTest(output=text), mock.patch.object(adapter.subprocess, 'run', return_value=SimpleNamespace(stdout=text)):
                with self.assertRaises(ValueError): adapter.gpu_compute_pids()
        with mock.patch.object(adapter.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
            self.assertEqual(adapter.gpu_compute_pids(), [])


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'output': '/tmp/unused_n20_scores', 'score_reference': 'same-checkpoint p1',
                     'TRAIN_seeds': [4817, 5817], 'cells': []}
        self.rows = []
        for arm in adapter.ARMS:
            for train in self.plan['TRAIN_seeds']:
                for gen in adapter.GEN_SEEDS:
                    cell = {**arm, 'TRAIN_seed': train, 'GEN_seed': gen}
                    self.plan['cells'].append(cell)
                    self.rows.append({**cell, **{name: 1. for name in adapter.METRICS},
                                      'generated_onroad_fraction': .9,
                                      'num_generated_vehicles': 100, 'num_gt_vehicles': 100})

    def test_incomplete_matrix_does_not_produce_scores(self):
        summary = adapter.summarize(self.plan, self.rows[:-1])
        self.assertFalse(summary['complete']); self.assertEqual(summary['scores'], [])

    def test_duplicate_cells_rejected(self):
        with self.assertRaises(ValueError): adapter.summarize(self.plan, self.rows + [self.rows[0]])

    def test_pooled_win_does_not_hide_losing_train_seed(self):
        for row in self.rows:
            if row['arm'] == 'p2_legacy_floor':
                for name in adapter.METRICS: row[name] = .7 if row['TRAIN_seed'] == 4817 else 1.1
        summary = adapter.summarize(self.plan, self.rows)
        score = next(r for r in summary['scores'] if r['arm'] == 'p2_legacy_floor')
        self.assertLess(score['sampler_R_vs_same_checkpoint_p1'], 1)
        self.assertFalse(score['all_TRAIN_improve_and_pass'])
        self.assertGreater(score['per_TRAIN']['5817']['R'], 1)
        self.assertFalse(score['model_vs_657_R_claimed'])

    def test_per_train_wins_do_not_imply_pooled_win(self):
        for row in self.rows:
            for index, name in enumerate(adapter.METRICS):
                high = (index < 4) if row['TRAIN_seed'] == 4817 else (index >= 4)
                baseline = 1000. if high else 1.
                if row['arm'] == 'p1_legacy_floor':
                    row[name] = baseline
                elif row['arm'] == 'p2_actual_time':
                    row[name] = baseline * (1.1 if high else .5)
                else:
                    row[name] = baseline
        score = next(r for r in adapter.summarize(self.plan, self.rows)['scores'] if r['arm'] == 'p2_actual_time')
        self.assertTrue(score['all_TRAIN_improve_and_pass'])
        self.assertGreater(score['sampler_R_vs_same_checkpoint_p1'], 1)
        self.assertFalse(score['robust_sampler_improvement'])

    def test_attribute_gate_failure_prevents_improvement_claim(self):
        for row in self.rows:
            if row['arm'] == 'p2_actual_time':
                for name in adapter.METRICS: row[name] = .7
                row['length_jsd'] = 1.2
        score = next(r for r in adapter.summarize(self.plan, self.rows)['scores'] if r['arm'] == 'p2_actual_time')
        self.assertLess(score['sampler_R_vs_same_checkpoint_p1'], 1)
        self.assertFalse(score['all_TRAIN_improve_and_pass'])

    def test_zero_denominator_rejected(self):
        for row in self.rows:
            if row['arm'] == 'p1_legacy_floor': row['width_jsd'] = 0
        with self.assertRaises(ValueError): adapter.summarize(self.plan, self.rows)

    def test_nonfinite_negative_metrics_or_populations_rejected(self):
        original = copy.deepcopy(self.rows)
        for key, value in (('speed_jsd', float('nan')), ('width_jsd', -1),
                           ('generated_onroad_fraction', float('inf')), ('num_gt_vehicles', 0)):
            self.rows = copy.deepcopy(original); self.rows[-1][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                adapter.summarize(self.plan, self.rows)


if __name__ == '__main__':
    unittest.main()
