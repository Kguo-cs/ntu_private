import gc
import io
import unittest
from unittest.mock import patch

import torch

from src.smart.datasets.scalable_dataset import load_torch_scene


class SceneLoadGCTest(unittest.TestCase):
    def test_tensor_roundtrip_preserves_gc_and_rng(self):
        stream = io.BytesIO(); torch.save({'x': torch.arange(5)}, stream); stream.seek(0)
        before = torch.get_rng_state().clone(); enabled = gc.isenabled()
        torch.testing.assert_close(load_torch_scene(stream)['x'], torch.arange(5))
        self.assertEqual(gc.isenabled(), enabled)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_deserialization_disables_gc_and_restores_on_exception(self):
        def fail(*args, **kwargs):
            self.assertFalse(gc.isenabled())
            raise ValueError('bad cache')
        for enabled in (True, False):
            (gc.enable if enabled else gc.disable)()
            try:
                with patch('src.smart.datasets.scalable_dataset.torch.load', side_effect=fail):
                    with self.assertRaises(ValueError): load_torch_scene('ignored')
                self.assertEqual(gc.isenabled(), enabled)
            finally:
                gc.enable()


if __name__ == '__main__': unittest.main()
