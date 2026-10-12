"""CPU-only certification tests with six actual synthetic cells; no subprocess."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import bridge_overnight_runtime_completion as bridge
from tools import test_runtime_completion as fixtures


class BridgeTests(unittest.TestCase):
    def fixture(self, directory):
        folder,log,spec,policy=fixtures.CompletionTests().fixture(directory)
        original=json.loads((folder/'summary.json').read_text());rows=[]
        spec['budgets']=[3,6,10]
        for budget in spec['budgets']:
            for source in original:
                row=copy.deepcopy(source);row['budget']=budget
                row['seen_training_examples']=bridge.v1.base.actual_exposure(budget,256);rows.append(row)
        (folder/'summary.json').write_text(json.dumps(rows))
        command=['python','certified_runtime_fixture.py'];arm={'name':'fixture'}
        execution=json.loads((folder/'execution.json').read_text())
        execution.update(command=command,arm=arm,completion_spec=spec,fingerprints={'same':'fixture'})
        execution['runtime_recovery']['wrapper_sha256']='first-controller'
        (folder/'execution.json').write_text(json.dumps(execution))
        stage={'label':folder.name,'arm':arm,'command':command,'spec':spec,'inputs':{},'checkpoint':None}
        contract={'wrapper_sha256':'first-controller','runtime_loader_source_hashes':policy['source_hashes']}
        return stage,folder,contract

    def evaluate(self,stage,folder,contract):
        original_read=bridge.v1.base.read
        def reader(path):
            return contract if Path(path)==bridge.v1.REC/'protocol.json' else original_read(path)
        with mock.patch.object(bridge.v1,'RUNS',folder.parent), \
             mock.patch.object(bridge.v1,'fingerprints',return_value={'same':'fixture'}), \
             mock.patch.object(bridge.v1.base,'read',side_effect=reader):
            return bridge.actual_completed(stage)

    def test_completed_zero_exit_six_cells_is_read_only_certifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,folder,contract=self.fixture(directory);before=(folder/'execution.json').read_bytes()
            result=self.evaluate(stage,folder,contract)
            self.assertEqual(len(result[1]),6);self.assertIn('runtime_loader_policy',result[2])
            self.assertEqual((folder/'execution.json').read_bytes(),before)

    def test_running_is_not_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,folder,contract=self.fixture(directory)
            ex=json.loads((folder/'execution.json').read_text());ex['status']='running'
            (folder/'execution.json').write_text(json.dumps(ex))
            self.assertIsNone(self.evaluate(stage,folder,contract))

    def test_failed_zero_or_completed_nonzero_not_certified(self):
        for status,code in [('failed',0),('completed',-11)]:
            with tempfile.TemporaryDirectory() as directory:
                stage,folder,contract=self.fixture(directory)
                ex=json.loads((folder/'execution.json').read_text());ex.update(status=status,returncode=code)
                (folder/'execution.json').write_text(json.dumps(ex))
                self.assertIsNone(self.evaluate(stage,folder,contract))

    def test_changed_command_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,folder,contract=self.fixture(directory);stage['command']=['changed']
            with self.assertRaises(RuntimeError):self.evaluate(stage,folder,contract)

    def test_missing_gen_cell_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,folder,contract=self.fixture(directory)
            rows=json.loads((folder/'summary.json').read_text());(folder/'summary.json').write_text(json.dumps(rows[:-1]))
            with self.assertRaises(RuntimeError):self.evaluate(stage,folder,contract)

    def test_wrong_recovery_source_contract_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,folder,contract=self.fixture(directory);contract['wrapper_sha256']='changed'
            with self.assertRaises(RuntimeError):self.evaluate(stage,folder,contract)


if __name__=='__main__':unittest.main()
