"""CPU-only actual/synthetic completion validation tests; never invokes GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import continue_overnight_init_diffusion as base
from tools import runtime_completion as completion


class CompletionTests(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory);folder=root/'stage';folder.mkdir()
        loader=root/'loader.py';loader.write_text('# fixture')
        sources={str(loader):hashlib.sha256(loader.read_bytes()).hexdigest()}
        policy=dict(policy='public_pickle_module_class_cache_and_full_cyclic_gc_deferral',
            automatic_gc_enabled=False,manual_gc_collect='disabled',young_gc_interval=0,
            tensor_and_data_conversion=False,objective_or_model_change=False,
            only_explicit_weights_only_false_loads=True,custom_pickle_module_preserved=True,
            success_exit='flush_then_os._exit(0)',source_hashes=sources,
            class_cache={'classes_created':1,'class_reuses':100})
        spec=dict(name='fixture',seed=4817,steps=10,scenes=4096,batch=256,
                  generations=[2817,3817],budgets=[10])
        rows=[]
        for gen in spec['generations']:
            report_path=folder/f'report_{gen}.json'
            metrics={key:.1 for key in base.METRICS}
            report={'reference_mode':'matched','num_samples':4096,'num_gt_samples':4096,
                    'agent_metrics':metrics,'require_full_set':False,'full_membership':False,
                    'map_source':'reference','decoder':{'mode':'lane_conditioned'}}
            report_path.write_text(json.dumps(report))
            rows.append(dict(variant='fixture',seed=4817,eval_seed=gen,budget=10,
                evaluation_weights='ema',num_samples=4096,batch_size=256,
                seen_training_examples=base.actual_exposure(10,256),map_frozen=True,
                initialization_scope='frozen_pretrained_map_fresh_G1',resumed_from=None,
                optimizer_continued=False,resume_global_step=0,actual_train_offset=0,
                generated_onroad_fraction=.9,num_generated_vehicles=100,num_gt_vehicles=100,
                map_checkpoint_sha256='fixedmap',metrics_path=str(report_path),**metrics))
        protocol=dict(reference_mode='matched',scenes=4096,seed=4817,batch_size=256,
                      require_full_set=False,map_checkpoint_sha256='fixedmap',young_gc_interval=0,defer_cyclic_gc=True)
        for name,value in [('protocol.json',protocol),('summary.json',rows),('runtime_loader_policy.json',policy),
                           ('execution.json',dict(status='completed',returncode=0,runtime_recovery={'runtime_loader_source_hashes':sources}))]:
            (folder/name).write_text(json.dumps(value))
        log=root/'stage.log';log.write_text('TRAIN actual fixture\nDONE '+str(folder)+'\n'+completion.PREFIX+json.dumps(policy)+'\n')
        return folder,log,spec,policy

    def test_real_legacy_control_keeps_artifact_raw_bytes(self):
        folder=base.RUNS/'long_seed4817_control';execution=base.read(folder/'execution.json')
        rows,artifacts=completion.verify_completion(folder,base.RUNS/'long_seed4817_control.log',execution['completion_spec'])
        self.assertEqual(len(rows),6);self.assertEqual(artifacts,execution['artifacts'])

    def test_certified_suffix_keeps_original_raw_log_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder,log,spec,policy=self.fixture(directory)
            rows,artifacts=completion.verify_completion(folder,log,spec)
            self.assertEqual(len(rows),2)
            self.assertEqual(artifacts['log']['sha256'],hashlib.sha256(log.read_bytes()).hexdigest())
            self.assertEqual(artifacts['log']['path'],str(log))
            self.assertIn('runtime_loader_policy',artifacts)

    def assert_rejected(self, mutate):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);mutate(*fixture)
            with self.assertRaises(RuntimeError):completion.verify_completion(*fixture[:3])

    def test_missing_policy_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:(folder/'runtime_loader_policy.json').unlink())

    def test_changed_loader_bytes_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:Path(next(iter(policy['source_hashes']))).write_text('# changed'))

    def test_wrong_done_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:log.write_text(log.read_text().replace('DONE '+str(folder),'DONE wrongstage')))

    def test_duplicate_success_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:log.write_text(completion.PREFIX+json.dumps(policy)+'\n'+log.read_text()))

    def test_nonzero_exit_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:(folder/'execution.json').write_text(json.dumps({'status':'completed','returncode':-11})))

    def test_missing_actual_gen_rejected(self):
        self.assert_rejected(lambda folder,log,spec,policy:(folder/'summary.json').write_text(json.dumps(base.read(folder/'summary.json')[:1])))

    def test_runtime_objective_guard_rejected(self):
        def mutate(folder,log,spec,policy):
            policy['objective_or_model_change']=True
            (folder/'runtime_loader_policy.json').write_text(json.dumps(policy))
            log.write_text('DONE '+str(folder)+'\n'+completion.PREFIX+json.dumps(policy)+'\n')
        self.assert_rejected(mutate)

    def test_cached_stage_requires_suffix_even_if_done_present(self):
        self.assert_rejected(lambda folder,log,spec,policy:log.write_text('DONE '+str(folder)+'\n'))


if __name__=='__main__':unittest.main()
