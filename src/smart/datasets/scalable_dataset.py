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

import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from torch_geometric.data import Dataset

from src.utils import RankedLogger
from random import shuffle
import os
import torch
log = RankedLogger(__name__, rank_zero_only=True)
working_dir = os.getcwd()
import torch

num_gpus = torch.cuda.device_count()
print("Total number of GPUs available:", num_gpus)

from pathlib import Path
from src.smart.metrics.metadata_adapter import attach_sd_metric_metadata
import torch

class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        self._raw_paths = [p.as_posix() for p in sorted(Path(raw_dir).glob("*"))]
        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self._num_samples = len(self._raw_paths)

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):

        idx = idx // num_gpus

        if '.pkl' in self.raw_paths[idx]:
            with open(self.raw_paths[idx], "rb") as handle:
                data = pickle.load(handle)
        else:
            data =torch.load(self.raw_paths[idx],map_location="cpu",weights_only=False)

        # ============================================================
        # Scenario Dreamer metric metadata
        # ============================================================
        if (
                 "scenario_dreamer" in data
                and "scenario_dreamer_cache_file" in data
        ):
            # IMPORTANT:
            # 只有当模型实际以这个 raw Waymo timestep 作为 initial scene
            # 时，才能这样设置。
            actual_generation_raw_t = int(data["scene_timestep"])

            data = attach_sd_metric_metadata(
                data,
                data,
                generation_scene_timestep=actual_generation_raw_t,
            )

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
