"""Stdlib/CPU tests for the native launcher supervisor, never train or use CUDA."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_init_diffusion_safe as native


class FakeProcess:
    pid = 13579
    def __init__(self, code=None):
        self.returncode = code
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        return self.returncode


LIMITS = {'max_rss_gib': 18., 'min_available_gib': 1.5, 'poll_seconds': .1}


class NativeGuardTests(unittest.TestCase):
    def test_default_limits_and_boundaries_match_main(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native.memory_limits(), {**LIMITS, 'poll_seconds': 2.})
        edge = {'child_rss_kib': 18*native.GIB_KIB, 'host_mem_available_kib': 1.5*native.GIB_KIB}
        self.assertIsNone(native.memory_stop_reason(edge, LIMITS))
        self.assertEqual(native.memory_stop_reason({**edge, 'child_rss_kib': edge['child_rss_kib']+1}, LIMITS), 'child_RSS_exceeded_limit')
        self.assertEqual(native.memory_stop_reason({**edge, 'host_mem_available_kib': edge['host_mem_available_kib']-1}, LIMITS), 'host_MemAvailable_below_limit')

    def test_invalid_limits_rejected(self):
        for key, value in [('SIM_NATIVE_MAX_RSS_GIB', '0'), ('SIM_NATIVE_MIN_AVAILABLE_GIB', '-1'),
                           ('SIM_NATIVE_MEMORY_POLL_SECONDS', 'nan'), ('SIM_NATIVE_MAX_RSS_GIB', 'inf')]:
            with patch.dict(os.environ, {key:value}, clear=True), self.assertRaises(ValueError):
                native.memory_limits()

    def test_success_forwards_exact_hydra_cli_and_validates_child_sidecar(self):
        process = FakeProcess()
        args = ['experiment=actual_native', '--cfg', 'job']
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'audit.json'
            def spawn(command, **kwargs):
                self.assertEqual(command[2:], args)
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(kwargs['env'][native.CHILD_ENV], '1')
                Path(kwargs['env']['SIM_NATIVE_RUNTIME_AUDIT']).write_text(json.dumps({'status':'success',
                    'automatic_gc_enabled':False, 'source_hashes':native.source_hashes()}))
                return process
            def finish(_):process.returncode=0
            with patch.object(native.subprocess, 'Popen', side_effect=spawn), patch.object(native.time, 'sleep', side_effect=finish), contextlib.redirect_stdout(io.StringIO()):
                code = native.run_guarded(args, out, LIMITS, memory_reader=lambda pid:{'child_rss_kib':1000,'host_mem_available_kib':20*native.GIB_KIB})
            record=json.loads(out.read_text())
            self.assertEqual(code,0);self.assertEqual(record['status'],'success')
            self.assertTrue(record['configuration_only']);self.assertEqual(record['memory_polls'],1)
            self.assertEqual(record['child_returncode'],0)
            self.assertFalse(record['checkpoint_auto_save_or_exact_resume_promised'])

    def test_memory_stop_not_success_even_if_child_returns_zero(self):
        process=FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'audit.json'
            def stop(p):self.assertIs(p,process);p.returncode=0;return ['SIGINT']
            with patch.object(native.subprocess,'Popen',return_value=process), patch.object(native,'stop_owned_child',side_effect=stop), contextlib.redirect_stdout(io.StringIO()):
                code=native.run_guarded(['experiment=fixture'],out,LIMITS,memory_reader=lambda pid:{'child_rss_kib':19*native.GIB_KIB,'host_mem_available_kib':20*native.GIB_KIB})
            record=json.loads(out.read_text())
            self.assertEqual(code,1);self.assertEqual(record['status'],'memory_limit_stopped')
            self.assertEqual(record['stop_reason'],'child_RSS_exceeded_limit')
            self.assertEqual(record['child_returncode'],0);self.assertEqual(record['stop_signals'],['SIGINT'])

    def test_real_owned_cpu_child_is_interrupted_on_memory_limit(self):
        real_popen=subprocess.Popen
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'audit.json';created=[]
            def spawn(unused_command,**kwargs):
                p=real_popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**kwargs)
                created.append(p);return p
            limits={'max_rss_gib':1e-5,'min_available_gib':0.,'poll_seconds':.1}
            with patch.object(native.subprocess,'Popen',side_effect=spawn),contextlib.redirect_stdout(io.StringIO()):
                code=native.run_guarded(['fixture_no_native_training'],out,limits)
            record=json.loads(out.read_text())
            self.assertEqual(code,1);self.assertEqual(record['status'],'memory_limit_stopped')
            self.assertEqual(record['stop_reason'],'child_RSS_exceeded_limit')
            self.assertEqual(record['child_pid'],created[0].pid);self.assertIsNotNone(created[0].poll())
            self.assertNotEqual(record['child_returncode'],0);self.assertIn('SIGINT',record['stop_signals'])

    def test_child_nonzero_and_missing_policy_remain_failure(self):
        for child_rc,expected_rc,status in [(3,3,'child_failed'),(0,1,'supervisor_failed')]:
            with self.subTest(child_rc=child_rc), tempfile.TemporaryDirectory() as temporary:
                out=Path(temporary)/'audit.json'
                with patch.object(native.subprocess,'Popen',return_value=FakeProcess(child_rc)),contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(native.run_guarded(['fixture'],out,LIMITS),expected_rc)
                record=json.loads(out.read_text());self.assertEqual(record['status'],status)
                self.assertEqual(record['child_returncode'],child_rc)

    def test_parent_interrupt_stops_only_its_child_and_records_signal(self):
        process=FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'audit.json'
            def stop(p):self.assertIs(p,process);p.returncode=-2;return ['SIGINT']
            with patch.object(native.subprocess,'Popen',return_value=process),patch.object(native,'stop_owned_child',side_effect=stop), \
                 patch.object(native.time,'sleep',side_effect=native.GuardInterrupted(15)),contextlib.redirect_stdout(io.StringIO()):
                code=native.run_guarded(['fixture'],out,LIMITS,memory_reader=lambda pid:{'child_rss_kib':1000,'host_mem_available_kib':20*native.GIB_KIB})
            record=json.loads(out.read_text());self.assertEqual(code,143)
            self.assertEqual(record['status'],'interrupted');self.assertEqual(record['interrupt_signal'],15)
            self.assertEqual(record['child_returncode'],-2);self.assertEqual(record['stop_signals'],['SIGINT'])

    def test_successful_existing_audit_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'audit.json';out.write_text('{}')
            with patch.object(native.subprocess,'Popen') as spawn,self.assertRaises(FileExistsError):
                native.run_guarded(['fixture'],out,LIMITS)
            spawn.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
