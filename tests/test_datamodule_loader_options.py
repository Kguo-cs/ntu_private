"""Training loaders must honor configured worker and pinned-memory options."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.smart.datamodules.scalable_datamodule import MultiDataModule


class TinyGraphDataset(Dataset):
    """Pickleable CPU dataset exercising real PyG collation in worker processes."""

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return Data(
            x=torch.tensor([[float(index), 1.0]]),
            sample_id=torch.tensor([index]),
            worker_pid=torch.tensor([os.getpid()]),
            worker_strategy=torch.multiprocessing.get_sharing_strategy(),
        )


class DataModuleLoaderOptionsTest(unittest.TestCase):
    def setUp(self):
        strategy = torch.multiprocessing.get_sharing_strategy()
        self.addCleanup(torch.multiprocessing.set_sharing_strategy, strategy)

    def module(self, **options):
        arguments = dict(
            train_batch_size=3, val_batch_size=3, test_batch_size=3,
            train_raw_dir="unused", val_raw_dir="unused", test_raw_dir="unused",
            val_tfrecords_splitted=None, shuffle=False, num_workers=2,
            pin_memory=False, persistent_workers=True, train_max_num=30,
        )
        arguments.update(options)
        module = MultiDataModule(**arguments)
        # Bypass filesystem setup; exercise the real loaders and PyG collation.
        module.train_dataset = TinyGraphDataset()
        module.val_dataset = TinyGraphDataset()
        module.test_dataset = TinyGraphDataset()
        return module

    def iterator(self, loader):
        iterator = iter(loader)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            self.addCleanup(shutdown)
        return iterator

    def epoch(self, iterator):
        sample_ids, worker_pids, batch_sizes = [], set(), []
        for batch in iterator:
            sample_ids.extend(batch.sample_id.tolist())
            worker_pids.update(batch.worker_pid.tolist())
            batch_sizes.append(batch.num_graphs)
            self.assertEqual(batch.x.device.type, "cpu")
        self.assertEqual(sample_ids, list(range(7)))
        self.assertEqual(batch_sizes, [3, 3, 1])
        return worker_pids

    def test_two_workers_without_pin_thread_complete_two_epochs(self):
        loader = self.module().train_dataloader()
        self.assertEqual(loader.num_workers, 2)
        self.assertFalse(loader.pin_memory)
        self.assertTrue(loader.persistent_workers)
        first = self.iterator(loader)
        self.assertFalse(hasattr(first, "_pin_memory_thread"))
        first_pids = self.epoch(first)
        self.assertEqual(len(first_pids), 2)
        self.assertNotIn(os.getpid(), first_pids)
        second = self.iterator(loader)
        self.assertIs(first, second)
        self.assertFalse(hasattr(second, "_pin_memory_thread"))
        self.assertEqual(self.epoch(second), first_pids)

    def test_explicit_pin_memory_true_is_forwarded_without_starting_gpu_work(self):
        module = self.module(pin_memory=True)
        with patch("src.smart.datamodules.scalable_datamodule.DataLoader") as constructor:
            result = module.train_dataloader()
        self.assertIs(result, constructor.return_value)
        self.assertIs(constructor.call_args.kwargs["pin_memory"], True)
        self.assertEqual(constructor.call_args.kwargs["num_workers"], 2)
        self.assertIs(constructor.call_args.kwargs["persistent_workers"], True)

    def test_one_requested_worker_runs_in_a_worker_process(self):
        loader = self.module(num_workers=1).train_dataloader()
        self.assertEqual(loader.num_workers, 1)
        self.assertTrue(loader.persistent_workers)
        iterator = self.iterator(loader)
        self.assertFalse(hasattr(iterator, "_pin_memory_thread"))
        pids = self.epoch(iterator)
        self.assertEqual(len(pids), 1)
        self.assertNotIn(os.getpid(), pids)

    def test_zero_workers_disables_persistence_and_iterates_in_main_process(self):
        loader = self.module(num_workers=0, persistent_workers=True, prefetch_factor=3).train_dataloader()
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)
        self.assertFalse(loader.pin_memory)
        self.assertIsNone(loader.prefetch_factor)
        self.assertIsNone(loader.worker_init_fn)
        self.assertEqual(self.epoch(self.iterator(loader)), {os.getpid()})

    def test_train_val_and_test_workers_share_tensors_with_configured_prefetch(self):
        module = self.module(prefetch_factor=3)
        for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            with self.subTest(loader=name):
                loader = getattr(module, name)()
                self.assertEqual(loader.prefetch_factor, 3)
                batches = list(self.iterator(loader))
                self.assertEqual([i for b in batches for i in b.sample_id.tolist()], list(range(7)))
                self.assertEqual({s for b in batches for s in b.worker_strategy}, {"file_system"})
                self.assertNotIn(os.getpid(), {p for b in batches for p in b.worker_pid.tolist()})

    def test_spawn_workers_configure_file_system_in_the_new_interpreter(self):
        loader = self.module(num_workers=1, persistent_workers=False).train_dataloader()
        # Spawn imports torch afresh and does not inherit the parent's strategy.
        loader.multiprocessing_context = "spawn"
        batches = list(self.iterator(loader))
        self.assertEqual([i for b in batches for i in b.sample_id.tolist()], list(range(7)))
        self.assertEqual({s for b in batches for s in b.worker_strategy}, {"file_system"})
        self.assertNotIn(os.getpid(), {p for b in batches for p in b.worker_pid.tolist()})

    def test_custom_worker_callback_preserves_lightning_seed_opt_in_and_rank(self):
        for seed_workers in ("1", "0"):
            with self.subTest(seed_workers=seed_workers), patch.dict(os.environ, {"PL_SEED_WORKERS": seed_workers}):
                module = self.module()
                module.trainer = SimpleNamespace(global_rank=4)
                loader = module.train_dataloader()
                with patch("lightning.fabric.utilities.seed.pl_worker_init_function") as seed_worker:
                    loader.worker_init_fn(3)
                if seed_workers == "1":
                    seed_worker.assert_called_once_with(3, rank=4)
                else:
                    seed_worker.assert_not_called()

    def test_explicit_nonpersistent_workers_are_not_reused(self):
        loader = self.module(persistent_workers=False).train_dataloader()
        self.assertFalse(loader.persistent_workers)
        self.assertFalse(loader.pin_memory)
        first = self.iterator(loader)
        self.assertEqual(len(self.epoch(first)), 2)
        second = self.iterator(loader)
        self.assertIsNot(first, second)
        self.assertEqual(len(self.epoch(second)), 2)


if __name__ == "__main__":
    unittest.main()
