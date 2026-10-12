"""CPU-only loader tests; no models, CUDA initialization or checkpoints."""
import gc
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
from tools.torch_pickle_class_cache import _CachedUnpicklerMeta, _ProxyUnpickler, cache_stats, pickle_module_proxy
from tools import runtime_cached_torch_loader_sweep as entrypoint
from tools import run_init_diffusion_safe as native


class EntrypointContractTests(unittest.TestCase):
    def test_exact_cli_done_then_policy_success_and_durable_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fake_torch = types.SimpleNamespace(load=lambda *a, **k: None)
            def delegate():
                self.assertEqual(sys.argv[1:], ['--defer-cyclic-gc', '--young-gc-interval', '0', '--output', str(output)])
                self.assertFalse(gc.isenabled())
                self.assertEqual(gc.collect(0), 0)
                print('DONE ' + str(output), flush=True)
            fake_runner = types.SimpleNamespace(sweep=types.SimpleNamespace(main=delegate))
            import tools
            capture = io.StringIO()
            with patch.object(sys, 'argv', ['exact_cli_entry', '--defer-cyclic-gc', '--young-gc-interval', '0', '--output', str(output)]), \
                 patch.dict(sys.modules, {'torch': fake_torch}), patch.dict(os.environ, {}, clear=False), \
                 patch.object(tools, 'overnight_init_diffusion_sweep', fake_runner, create=True), \
                 patch.object(gc, 'collect'), patch.object(entrypoint.os, '_exit') as hard_exit, contextlib.redirect_stdout(capture):
                entrypoint.main()
            lines = capture.getvalue().splitlines()
            self.assertEqual(lines[-2], 'DONE ' + str(output))
            self.assertTrue(lines[-1].startswith('RUNTIME_LOADER_SUCCESS '))
            self.assertTrue((output/'runtime_loader_policy.json').is_file())
            hard_exit.assert_called_once_with(0)

    def test_cyclic_collection_request_and_missing_explicit_zero_rejected(self):
        for argv in (['x', '--defer-cyclic-gc', '--young-gc-interval', '128'], ['x', '--defer-cyclic-gc']):
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                entrypoint.main()

    def test_native_launcher_forwards_hydra_cli_unchanged_without_collections(self):
        capture = io.StringIO()
        fake_torch = types.SimpleNamespace(load=lambda *a, **k: None)
        args = ['experiment=actual_best', 'action=test', 'ckpt_path=/explicit/weights.ckpt']
        def delegated(path, run_name):
            self.assertEqual(path, str(ROOT/'src/run.py'))
            self.assertEqual(run_name, '__main__')
            self.assertEqual(sys.argv, [str(ROOT/'src/run.py'), *args])
            self.assertFalse(gc.isenabled())
            self.assertEqual(gc.collect(), 0)
            self.assertEqual(gc.collect(0), 0)
        with patch.object(sys, 'argv', ['native_safe', *args]), patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(gc, 'collect'), patch.dict(os.environ, {'SIM_NATIVE_RUNTIME_AUDIT': ''}), \
             patch.object(native.runpy, 'run_path', side_effect=delegated) as run_path, \
             patch.object(native.os, '_exit') as hard_exit, contextlib.redirect_stdout(capture):
            native.run_native_child()
        run_path.assert_called_once()
        self.assertTrue(capture.getvalue().splitlines()[-1].startswith('NATIVE_RUNTIME_SUCCESS '))
        hard_exit.assert_called_once_with(0)

    def test_native_failure_never_hard_exits_success(self):
        fake_torch = types.SimpleNamespace(load=lambda *a, **k: None)
        with patch.object(sys, 'argv', ['native_safe']), patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(gc, 'collect'), patch.dict(os.environ, {'SIM_NATIVE_RUNTIME_AUDIT': ''}), \
             patch.object(native.runpy, 'run_path', side_effect=SystemExit(3)), \
             patch.object(native.os, '_exit') as hard_exit, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            native.run_native_child()
        self.assertEqual(error.exception.code, 3)
        hard_exit.assert_not_called()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def original(*args, **kwargs):
            self.calls.append((args, kwargs))
            return 'original return'
        self.fake = types.SimpleNamespace(load=original)
        self.original = original
        self.assertIs(install_cached_loader(self.fake), original)

    def test_explicit_trusted_load_receives_public_proxy(self):
        self.assertEqual(self.fake.load('x', map_location='cpu', weights_only=False), 'original return')
        args, kwargs = self.calls[-1]
        self.assertEqual(args, ('x',))
        self.assertIs(kwargs['pickle_module'].Unpickler, _ProxyUnpickler)
        self.assertEqual(kwargs['map_location'], 'cpu')

    def test_safe_and_unspecified_loads_keep_original_behavior(self):
        for kwargs in ({}, {'weights_only': True}, {'weights_only': None}):
            self.fake.load('x', **kwargs)
            self.assertEqual(self.calls[-1][1], kwargs)

    def test_custom_keyword_pickle_module_preserved(self):
        custom = object()
        self.fake.load('x', weights_only=False, pickle_module=custom)
        self.assertIs(self.calls[-1][1]['pickle_module'], custom)

    def test_custom_positional_pickle_module_preserved(self):
        custom = object()
        self.fake.load('x', 'cpu', custom, weights_only=False)
        self.assertEqual(self.calls[-1], (('x', 'cpu', custom), {'weights_only': False}))

    def test_original_exceptions_propagate(self):
        def fail(*args, **kwargs):
            raise OSError('original error')
        fake = types.SimpleNamespace(load=fail)
        install_cached_loader(fake)
        with self.assertRaisesRegex(OSError, 'original error'):
            fake.load('x', weights_only=False)


class InstalledTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_zip_legacy_and_classcell_reuse(self):
        torch = self.torch
        payload = {'tensor': torch.tensor([0., -0., float('nan'), 2.]), 'metadata': [1, 'original']}
        proxy = pickle_module_proxy()
        for legacy in (False, True):
            buffer = io.BytesIO()
            torch.save(payload, buffer, _use_new_zipfile_serialization=not legacy)
            for _ in range(3):
                buffer.seek(0)
                actual = torch.load(buffer, map_location='cpu', weights_only=False, pickle_module=proxy)
                self.assertTrue(torch.equal(payload['tensor'].view(torch.uint8), actual['tensor'].view(torch.uint8)))
                self.assertEqual(payload['metadata'], actual['metadata'])
        stats = cache_stats()
        self.assertEqual(stats['classes_created'], 2)
        self.assertGreaterEqual(stats['class_reuses'], 4)
        self.assertEqual(stats['cached_loader_kinds'], ['_legacy_load', '_load'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_unknown_loader_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unknown Torch wrapper kind'):
            _CachedUnpicklerMeta('UnpicklerWrapper', (_ProxyUnpickler,), {'find_class': lambda self, a, b: None})

    def test_unknown_base_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unexpected Torch UnpicklerWrapper base'):
            _CachedUnpicklerMeta('UnpicklerWrapper', (object,), {})

    def test_unknown_method_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'no normal find_class'):
            _CachedUnpicklerMeta('UnpicklerWrapper', (_ProxyUnpickler,), {'find_class': None})


if __name__ == '__main__':
    # This environment has demonstrated native cyclic-GC SIGSEGVs. Mirror the
    # intended isolated runtime policy without treating hard-exit as a test pass.
    gc.disable()
    gc.collect = lambda *a, **k: 0
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if result.wasSuccessful() else 1)
