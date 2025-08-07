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


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        # self.val='val' in raw_dir

        raw_dir = Path(raw_dir)
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]  # [::1600]



       # shuffle(self._raw_paths)
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None


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
        #print(idx)
        # if  idx in self.cache_data.keys():
        #     data=self.cache_data[idx]
        # else:
        # if self.val:
        #     with open('./waymo_data/full/validation_map2/'+self.selected_files[idx//device_number], "rb") as handle:
        #         data = pickle.load(handle)
        # else:
        idx=idx//num_gpus

        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)


        # if 'keguo' in working_dir:
        #     self.cache_data[idx] = data

        # print(self.raw_paths[idx])

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
