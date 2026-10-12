"""Synthetic CPU-only recovery protocol checks; never submits subprocesses."""
import copy
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

ROOT=pathlib.Path('/home/ke/code/sim')
sys.path.insert(0,str(ROOT))
from tools import recover_overnight_init_diffusion_runtime as recovery


class RecoveryTests(unittest.TestCase):
    def test_real_frozen_commands_and_inherited_success(self):
        p=recovery.base.read(recovery.ART/'frozen_long_protocol.json')
        arms={a['name']:a for a in recovery.base.read(recovery.ART/'search_protocol.json')['arms']}
        for name in ('control',p['challenger']):
            steps=p['final_budget_B512'] if name=='control' else p['final_budget_B256']
            stage=recovery.describe('long_seed4817_'+name,arms[name],steps,4817)
            diff=[(a,b) for a,b in zip(stage['command'],stage['original_command']) if a!=b]
            self.assertEqual(diff,[('0','128')])
            self.assertEqual(recovery.base.actual_exposure(steps,arms[name]['batch']),5242620)
        stage=recovery.describe('long_seed4817_control',arms['control'],10240,4817)
        self.assertEqual(len(recovery.existing_success(stage)),6)

    def test_exact_memory_bounds(self):
        unit=recovery.GIB_KIB
        self.assertIsNone(recovery.memory_limit_reason(18*unit,1.5*unit))
        self.assertEqual(recovery.memory_limit_reason(18*unit+1,2*unit),'child_RSS_exceeded_limit')
        self.assertEqual(recovery.memory_limit_reason(10*unit,1.5*unit-1),'host_MemAvailable_below_limit')

    def test_only_owned_process_signalled(self):
        child=mock.Mock(pid=123456)
        child.poll.return_value=None
        child.wait.side_effect=[recovery.subprocess.TimeoutExpired('fixture',30),0]
        with mock.patch.object(recovery.os,'killpg') as kill:
            recovery.stop_owned(child)
        self.assertEqual(kill.call_args_list,[mock.call(123456,recovery.signal.SIGTERM),mock.call(123456,recovery.signal.SIGKILL)])

    def test_parent_signals_raise_into_cleanup_handler(self):
        for signum in (recovery.signal.SIGTERM,recovery.signal.SIGINT):
            with self.assertRaises(KeyboardInterrupt):recovery.interrupted(signum,None)

    def fixture(self):
        proto={'challenger':'candidate','actual_final_exposure_per_TRAIN':5242620,'final_budget_rule':{}}
        final={name:[] for name in ('control','candidate')}
        for name in final:
            for seed in (4817,5817):
                for gen in (2817,3817):
                    row={k:1. for k in recovery.base.METRICS}
                    row.update(variant=name,seed=seed,eval_seed=gen,seen_training_examples=5242620,
                        generated_onroad_fraction=.9,num_generated_vehicles=100,num_gt_vehicles=100)
                    final[name].append(row)
        return proto,final,{'control':{},'candidate':{}}

    def test_partial_matrix_is_not_selected(self):
        proto,final,arms=self.fixture();final['candidate'].pop()
        result=recovery.select_final(final,proto,arms)
        self.assertFalse(result['complete_matrix']);self.assertNotIn('selected',result)

    def test_pooled_gate_cannot_hide_held_seed_speed_failure(self):
        proto,final,arms=self.fixture()
        for row in final['candidate']:
            for k in recovery.base.METRICS:row[k]=.7
            row['speed_jsd']=.8 if row['seed']==4817 else 1.3
        with mock.patch.object(recovery.base,'row_cohort',return_value={'fixture':'same'}):
            result=recovery.select_final(final,proto,arms)
        self.assertTrue(result['complete_matrix']);self.assertEqual(result['selected'],'control')
        candidate=next(r for r in result['ranking'] if r['name']=='candidate')
        self.assertTrue(candidate['aggregate_gates_pass'])
        self.assertFalse(candidate['per_training_seed']['5817']['eligible'])

    def test_both_seed_gate_improvement_selects_frozen_candidate(self):
        proto,final,arms=self.fixture()
        for row in final['candidate']:
            for k in recovery.base.METRICS:row[k]=.9
        with mock.patch.object(recovery.base,'row_cohort',return_value={'fixture':'same'}):
            result=recovery.select_final(final,proto,arms)
        self.assertEqual(result['selected'],'candidate')


if __name__=='__main__':unittest.main()
