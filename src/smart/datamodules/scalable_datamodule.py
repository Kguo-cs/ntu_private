# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial
import os
from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader

from src.smart.datasets import MultiDataset

from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal
import torch


def _initialize_loader_worker(worker_id: int, *, sharing_strategy: str, seed_rank: Optional[int]) -> None:
    # Spawn/forkserver workers do not inherit the parent's PyTorch setting.
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    if seed_rank is not None:
        from lightning.fabric.utilities.seed import pl_worker_init_function
        pl_worker_init_function(worker_id, rank=seed_rank)


class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_tfrecords_splitted: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        scenario_dreamer_preprocessed: bool = False,
        scenario_dreamer_eval_set: Optional[str] = None,
        scenario_dreamer_train_latent_cache: Optional[str] = None,
        scenario_dreamer_val_latent_cache: Optional[str] = None,
        scenario_dreamer_test_latent_cache: Optional[str] = None,
        scenario_dreamer_train_non_partitioned_only: bool = False,
        scenario_dreamer_train_graph_type_index: Optional[str] = None,
        scenario_dreamer_train_sample_list: Optional[str] = None,
        sharing_strategy: str = "file_system",
        prefetch_factor: int = 2,
    ) -> None:
        super(MultiDataModule, self).__init__()
        if sharing_strategy not in torch.multiprocessing.get_all_sharing_strategies():
            raise ValueError(f"Unsupported tensor sharing strategy: {sharing_strategy}")
        if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int) or prefetch_factor < 1:
            raise ValueError("prefetch_factor must be a positive integer")
        self.sharing_strategy = sharing_strategy
        self.prefetch_factor = prefetch_factor
        self.dataset_options = {"scenario_dreamer_preprocessed": scenario_dreamer_preprocessed}
        self.eval_dataset_options = dict(self.dataset_options, sample_list=scenario_dreamer_eval_set)
        self.train_dataset_options = dict(
            self.dataset_options, scenario_dreamer_latent_cache=scenario_dreamer_train_latent_cache,
            scenario_dreamer_non_partitioned_only=scenario_dreamer_train_non_partitioned_only,
            scenario_dreamer_graph_type_index=scenario_dreamer_train_graph_type_index,
            sample_list=scenario_dreamer_train_sample_list, sample_list_check_exists=False)
        self.val_dataset_options = dict(self.eval_dataset_options, scenario_dreamer_latent_cache=scenario_dreamer_val_latent_cache)
        self.test_dataset_options = dict(self.eval_dataset_options, scenario_dreamer_latent_cache=scenario_dreamer_test_latent_cache)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted

        self.train_transform = WaymoTargetBuilderTrain(train_max_num)
        self.val_transform = WaymoTargetBuilderVal()
        self.test_transform = WaymoTargetBuilderVal()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MultiDataset(self.train_raw_dir, self.train_transform, **self.train_dataset_options)
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
                **self.val_dataset_options,
            )
        elif stage == "validate":
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
                **self.val_dataset_options,

            )
        elif stage == "test":
            self.test_dataset = MultiDataset(self.test_raw_dir, self.test_transform, **self.test_dataset_options)
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Honor Hydra's loader options, including the explicit pin_memory=False
        # default. Forcing pinning here starts an unwanted CUDA pin-memory thread.
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            **self._worker_options(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,  # False
            persistent_workers=False,
            drop_last=False,
            **self._worker_options(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 0
            pin_memory=False,  # False
            persistent_workers=False,
            drop_last=False,
            **self._worker_options(),
        )

    def _worker_options(self) -> dict:
        if self.num_workers == 0:
            # PyTorch rejects prefetch_factor without multiprocessing workers.
            return {}
        torch.multiprocessing.set_sharing_strategy(self.sharing_strategy)
        # Providing our callback prevents Lightning from inserting its seed
        # callback, so preserve seed_everything(..., workers=True) explicitly.
        seed_rank = None
        if os.environ.get("PL_SEED_WORKERS") == "1":
            seed_rank = getattr(getattr(self, "trainer", None), "global_rank", 0)
        return {
            "prefetch_factor": self.prefetch_factor,
            "worker_init_fn": partial(
                _initialize_loader_worker,
                sharing_strategy=self.sharing_strategy,
                seed_rank=seed_rank,
            ),
        }
