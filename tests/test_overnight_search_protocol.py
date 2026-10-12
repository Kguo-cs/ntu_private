"""CPU fixtures for overnight selection, physical scope and restart safety."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import continue_overnight_init_diffusion as queue


def fixture_row(name='control', *, batch=512, budget=768, seed=4817, gen=2817, scenes=1024):
    return {'variant': name, 'seed': seed, 'eval_seed': gen, 'budget': budget,
            'eval_seconds': 1.,
            'batch_size': batch, 'seen_training_examples': queue.actual_exposure(budget, batch),
            'evaluation_weights': 'ema', 'num_samples': scenes,
            'map_frozen': True, 'initialization_scope': 'frozen_pretrained_map_fresh_G1',
            **{key: 1. for key in queue.METRICS}, 'generated_onroad_fraction': .8,
            'num_generated_vehicles': 100, 'num_gt_vehicles': 100}


class SelectionTest(unittest.TestCase):
    def test_actual_exposure_matches_partial_batches_and_equal_batch_budgets(self):
        for small, large, expected in ((1536, 768, 393216), (8192, 4096, 2097048),
                                       (24576, 12288, 6291144)):
            self.assertEqual(queue.actual_exposure(small, 256), expected)
            self.assertEqual(queue.actual_exposure(large, 512), expected)

    def test_planned_generation_cells_cannot_be_missing_or_duplicated(self):
        row = fixture_row()
        spec = dict(name='control', seed=4817, steps=768, scenes=1024, batch=512,
                    generations=[2817, 3817])
        with self.assertRaisesRegex(ValueError, 'missing planned'):
            queue.validate_rows([row], **spec)
        second = dict(row, eval_seed=3817)
        queue.validate_rows([row, second], **spec)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            queue.validate_rows([row, second, second], **spec)
        wrong = dict(second, seen_training_examples=row['seen_training_examples'] + 1)
        with self.assertRaisesRegex(ValueError, 'actual exposure'):
            queue.validate_rows([row, wrong], **spec)

    def test_seed_mean_is_equal_train_weight_not_generation_pseudoreplication(self):
        rows = [fixture_row(seed=4817, gen=2817), fixture_row(seed=4817, gen=3817),
                fixture_row(seed=5817, gen=2817), fixture_row(seed=5817, gen=3817)]
        for index, row in enumerate(rows):
            row['speed_jsd'] = (1., 3., 5., 7.)[index]
        self.assertEqual(queue.aggregate(rows)['speed_jsd'], 4.)
        with self.assertRaisesRegex(ValueError, 'Duplicate GEN'):
            queue.aggregate(rows + [rows[0]])

    def test_speed_shape_gate_is_independent_of_seven_metric_score(self):
        baseline = queue.aggregate([fixture_row()])
        candidate = dict(baseline, speed_jsd=1.16, nearest_dist_jsd=.1)
        result = queue.compare(candidate, baseline)
        self.assertLess(result['R'], 1.)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['gate_checks']['speed_jsd'])

    def test_screen_reserves_reference_only_when_fewer_two_eligible_novel(self):
        ranked = [{'name': name, 'eligible': eligible, 'R': score, 'R_attr': score}
                  for name, eligible, score in [('control', True, 1.), ('bad', False, .5),
                                               ('novel', True, .8), ('combo_reference', True, .9)]]
        promoted, exploratory = queue.promote_screen(ranked)
        self.assertEqual(promoted, ['control', 'novel', 'combo_reference'])
        self.assertFalse(exploratory)

    def test_all_failed_gates_is_explicit_exploration(self):
        ranked = [{'name': name, 'eligible': False, 'R': score, 'R_attr': score}
                  for name, score in [('control', 1.), ('first', .8), ('combo_reference', .9), ('last', 1.2)]]
        promoted, exploratory = queue.promote_screen(ranked)
        self.assertEqual(promoted, ['control', 'first', 'combo_reference'])
        self.assertTrue(exploratory)
        chosen, exploratory = queue.choose_medium(ranked)
        self.assertEqual(chosen['name'], 'first')
        self.assertTrue(exploratory)

    def test_deadline_budget_preserves_two_seeds_and_chooses_highest_feasible(self):
        control = fixture_row(budget=4096); control['train_seconds'] = 4096 * .4
        candidate = fixture_row(name='winner', batch=256, budget=8192); candidate['train_seconds'] = 8192 * .2
        medium = {'control': [control], 'winner': [candidate]}
        with patch.object(queue, 'DEADLINE', 22000):
            choice, meta = queue.choose_final_budget(medium, 'winner', now=0)
        self.assertEqual(choice['control_B512_updates'], 10240)
        self.assertFalse(meta['deadline_risk'])
        with patch.object(queue, 'DEADLINE', 10000):
            choice, meta = queue.choose_final_budget(medium, 'winner', now=0)
        self.assertEqual(choice['control_B512_updates'], 8192)
        self.assertTrue(meta['deadline_risk'])

    def test_slow_eval_reserves_actual_long_cells_and_full_scope(self):
        control = fixture_row(budget=4096, scenes=4096)
        candidate = fixture_row(name='winner', batch=256, budget=8192, scenes=4096)
        control.update(train_seconds=1638.4, eval_seconds=22.)
        candidate.update(train_seconds=1638.4, eval_seconds=110.)
        with patch.object(queue, 'DEADLINE', 100000):
            choice, meta = queue.choose_final_budget({'control': [control], 'winner': [candidate]},
                                                     'winner', now=0)
        # Two TRAINs x three budgets x two GENs, for each of the two arms.
        self.assertEqual(choice['long_evaluation_seconds'], 12 * (22 + 110))
        # Both arms x two GENs over50000, with25% allowance for full-set I/O.
        self.assertAlmostEqual(choice['full50k_reserved_seconds'], 2 * 50000 / 4096 * (22 + 110) * 1.25)
        self.assertGreater(choice['full50k_reserved_seconds'], 1200)
        candidate['eval_seconds'] = 0.
        with self.assertRaisesRegex(ValueError, 'evaluation timing'):
            queue.choose_final_budget({'control': [control], 'winner': [candidate]}, 'winner', now=0)


class StageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sim-overnight-cpu-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.art, self.runs, self.old = (self.root / name for name in ('art', 'runs', 'old'))
        for path in (self.art, self.runs, self.old):
            path.mkdir()
        (self.old / 'confirmation4096_files.json').write_text(json.dumps([f'{i}.pt' for i in range(4096)]))
        self.map = self.root / 'fixed-map.ckpt'; self.map.write_bytes(b'CPU fixture map')
        self.normalizer = self.root / 'normalizer.json'; self.normalizer.write_text('{}')
        self.arm = {'name': 'control', 'recipe': {'components': {}}, 'config': {},
                    'batch': 512, 'map': str(self.map), 'normalizer': str(self.normalizer)}
        for name, value in (('ART', self.art), ('RUNS', self.runs), ('OLD', self.old),
                            ('DEADLINE', float('inf'))):
            manager = patch.object(queue, name, value); manager.start(); self.addCleanup(manager.stop)
        manager = patch.object(queue, 'source_hashes', return_value={'model.py': 'unchanged'})
        manager.start(); self.addCleanup(manager.stop)

    def launch(self, argv, **kwargs):
        output = Path(argv[argv.index('--output') + 1])
        protocol = {'reference_mode': 'matched', 'scenes': 1024, 'seed': 4817,
                    'batch_size': 512, 'require_full_set': False}
        queue.save(output / 'protocol.json', protocol)
        row = fixture_row()
        report_path = output / 'control' / 'metrics.json'
        queue.save(report_path, {'reference_mode': 'matched', 'num_samples': 1024, 'num_gt_samples': 1024,
                   'agent_metrics': {key: row[key] for key in queue.METRICS}})
        row['metrics_path'] = str(report_path)
        queue.save(output / 'summary.json', [row])
        kwargs['stdout'].write('DONE ' + str(output) + '\n'); kwargs['stdout'].flush()

        class Process:
            pid = 12345
            returncode = 0

            def poll(self):
                return 0

        return Process()

    def execute(self, arm=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return queue.run('stage', self.arm if arm is None else arm, steps=768, scenes=1024)

    def complete(self):
        with patch.object(queue.subprocess, 'Popen', side_effect=self.launch) as process:
            rows = self.execute()
        self.assertEqual(process.call_count, 1)
        self.assertEqual(len(rows), 1)

    def test_identical_completed_stage_is_verified_without_replay(self):
        self.complete()
        with patch.object(queue.subprocess, 'Popen') as process:
            self.execute()
        process.assert_not_called()

    def test_same_argv_changed_recipe_refuses_stale_reuse(self):
        self.complete()
        arm = copy.deepcopy(self.arm); arm['recipe']['components']['velocity'] = {'weight': 9}
        with patch.object(queue.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'design/source changed'):
                self.execute(arm)
        process.assert_not_called()

    def test_mutated_summary_refuses_reuse(self):
        self.complete()
        path = self.runs / 'stage' / 'summary.json'
        rows = queue.read(path); rows[0]['speed_jsd'] = .9; queue.save(path, rows)
        with self.assertRaisesRegex(RuntimeError, 'artifacts.*(changed|invalid)'):
            self.execute()

    def test_mutated_report_refuses_reuse(self):
        self.complete()
        path = self.runs / 'stage' / 'control' / 'metrics.json'
        report = queue.read(path); report['num_gt_samples'] = 4096; queue.save(path, report)
        with self.assertRaisesRegex(RuntimeError, 'matched physical scope'):
            self.execute()

    def test_new_source_hash_refuses_stale_reuse(self):
        self.complete()
        with patch.object(queue, 'source_hashes', return_value={'model.py': 'changed'}):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                self.execute()

    def test_partial_output_refuses_replay(self):
        (self.runs / 'stage').mkdir()
        (self.runs / 'stage' / 'partial.ckpt').write_bytes(b'CPU fixture')
        with patch.object(queue.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'manual recovery'):
                self.execute()
        process.assert_not_called()

    def test_full50000_uses_full_scope_without_4096_manifest(self):
        command, inputs = queue.make_command('full', self.arm, steps=12288, scenes=50000,
                                             extra_seed=True, evaluation_checkpoint=self.map)
        self.assertIn('--require-full-set', command)
        self.assertNotIn('--eval-manifest', command)
        self.assertEqual(command[command.index('--scenes') + 1], '50000')
        self.assertEqual(command[command.index('--extra-eval-seed') + 1], '3817')
        self.assertIn('--evaluation-checkpoints', command)
        self.assertFalse(any('confirmation4096' in path for path in inputs))

    def test_no_old_completion_keeps_waiting_without_false_completion(self):
        self.assertIsNone(queue.gpu_handoff())
        self.assertFalse((self.old / 'queue_complete.json').exists())

    def test_partial_old_marker_alone_cannot_release_gpu(self):
        queue.save(self.old / 'queue_complete.json', {'status': 'PARTIAL_FAILED_RUNTIME'})
        self.assertIsNone(queue.gpu_handoff())

    def test_explicit_handoff_accepts_reaped_old_processes_and_keeps_old_partial(self):
        queue.save(self.old / 'queue_complete.json', {'status': 'PARTIAL_FAILED_RUNTIME'})
        marker = {'released': True, 'old_queue_terminal': True, 'gpu_processes_reaped': True,
                  'terminated_pids': [12345, 12346], 'reason': 'Three archived runtime failures; begin fresh search.'}
        queue.save(self.art / 'gpu_handoff.json', marker)
        with patch.object(queue, 'process_running', return_value=False):
            result = queue.gpu_handoff()
        self.assertEqual(result['mode'], 'explicit_terminal_old_queue_handoff')
        self.assertFalse(result['old_queue_completed'])
        self.assertEqual(queue.read(self.old / 'queue_complete.json')['status'], 'PARTIAL_FAILED_RUNTIME')
        self.assertEqual(len(result['sha256']), 64)

    def test_handoff_refuses_existing_prior_controller_or_child(self):
        queue.save(self.art / 'gpu_handoff.json', {'released': True, 'old_queue_terminal': True,
                   'gpu_processes_reaped': True, 'terminated_pids': [12345], 'reason': 'fixture'})
        with patch.object(queue, 'process_running', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'PIDs still exist'):
                queue.gpu_handoff()

    def test_handoff_requires_terminal_provenance_not_a_bare_permission_flag(self):
        queue.save(self.art / 'gpu_handoff.json', {'released': True})
        with self.assertRaisesRegex(RuntimeError, 'must certify'):
            queue.gpu_handoff()

    def test_actual_old_complete_has_a_separate_mode(self):
        queue.save(self.old / 'queue_complete.json', {'status': 'COMPLETE_ACTUAL_SEARCH'})
        self.assertEqual(queue.gpu_handoff()['mode'], 'verified_old_queue_completion')

    def test_failed_first_control_stops_before_unrankable_novel_arm(self):
        with patch.object(queue, 'design_arms', return_value=[self.arm, dict(self.arm, name='novel')]), \
             patch.object(queue, 'gpu_handoff', return_value={'mode': 'CPU fixture'}), \
             patch.object(queue, 'run', return_value=[]) as run, contextlib.redirect_stdout(io.StringIO()):
            queue.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], 'screen_control')
        result = queue.read(self.art / 'queue_complete.json')
        self.assertEqual(result['status'], 'FAILED_SCREEN_CONTROL')
        self.assertFalse(result['complete_final_matrix'])
        self.assertTrue(result['manual_fresh_recovery_required'])


if __name__ == '__main__':
    unittest.main()
