"""Pure CPU source-ledger fixture tests; excluded from all search observations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tools import materialize_overnight_init_diffusion_best as best


class RuntimeProvenanceTests(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory);art=root/'art';art.mkdir();folder=root/'stage';folder.mkdir()
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value) if not isinstance(value,str) else value)
            return path
        model=write(root/'configs/model/smart.yaml','# immutable fixture')
        frozen={'configs/model/smart.yaml':best.file_hash(model)}
        freeze=write(art/'source_freeze.json',{'source_hashes':frozen})
        long=write(art/'frozen_long_protocol.json',{'fixture':'unchanged budget'})
        wrapper=write(root/'tools/recovery.py','# original recovery fixture')
        loader=write(root/'tools/loader.py','# loader fixture')
        helper=write(root/'tools/cache_helper.py','# helper fixture')
        sources={str(p):best.file_hash(p) for p in (loader,helper)}
        ledger=art/'runtime_recovery/protocol.json'
        recovery={'source_freeze_sha256':best.file_hash(freeze),'frozen_long_protocol_sha256':best.file_hash(long),
                  'source_hashes':frozen,'wrapper':str(wrapper),'wrapper_sha256':best.file_hash(wrapper),
                  'loader_wrapper':str(loader),'runtime_loader_source_hashes':sources}
        write(ledger,recovery)
        write(folder/'protocol.json',{'cyclic_gc_policy':'fixture GCoff','young_gc_interval':0,'defer_cyclic_gc':True})
        write(folder/'runtime_loader_policy.json',{'fixture':'policy bytes already validated by completion helper'})
        execution={'source_hashes':frozen,'command':['python',str(loader)],'completion_spec':{'budgets':[1,2,3]},
                   'runtime_recovery':{'ledger':str(ledger),'wrapper_sha256':best.file_hash(wrapper),
                                       'runtime_loader_source_hashes':sources}}
        return root,art,folder,execution,helper

    def evaluate(self,root,art,folder,execution):
        with mock.patch.object(best,'ROOT',root),mock.patch.object(best,'ART',art):
            return best.audit_runtime_sources(execution,folder)

    def test_loader_helper_and_raw_policy_artifact_are_captured(self):
        with tempfile.TemporaryDirectory() as directory:
            root,art,folder,execution,helper=self.fixture(directory)
            result=self.evaluate(root,art,folder,execution)
            self.assertEqual(result['young_gc_interval'],0)
            self.assertIn(str(helper),result['runtime_loader_sources'])
            self.assertEqual(result['actual_runtime_policy']['sha256'],best.file_hash(folder/'runtime_loader_policy.json'))

    def test_changed_runtime_helper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root,art,folder,execution,helper=self.fixture(directory);helper.write_text('# changed')
            with self.assertRaisesRegex(ValueError,'entry/helper'):self.evaluate(root,art,folder,execution)

    def test_changed_original_frozen_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root,art,folder,execution,helper=self.fixture(directory)
            (root/'configs/model/smart.yaml').write_text('# changed')
            with self.assertRaisesRegex(ValueError,'frozen source'):self.evaluate(root,art,folder,execution)

    def test_bridge_sources_and_completion_certificate_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root,art,folder,execution,helper=self.fixture(directory)
            bridge=root/'tools/bridge.py';bridge.write_text('# bridge fixture')
            ledger=art/'runtime_recovery/protocol.json'
            sources={str(bridge):best.file_hash(bridge)}
            (art/'runtime_recovery/verification_bridge.json').write_text(json.dumps(
                {'original_recovery_protocol_sha256':best.file_hash(ledger),'source_hashes':sources}))
            execution['completion_verification_bridge']={'source_hashes':sources,'actual_returncode':0,'actual_complete_rows':6}
            result=self.evaluate(root,art,folder,execution);self.assertIn('verification_bridge_sources',result)
            execution['completion_verification_bridge']['actual_complete_rows']=5
            with self.assertRaisesRegex(ValueError,'certification'):self.evaluate(root,art,folder,execution)


if __name__=='__main__':unittest.main()
