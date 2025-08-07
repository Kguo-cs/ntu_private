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
log = RankedLogger(__name__, rank_zero_only=True)
working_dir = os.getcwd()


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]  # [::1600]

        if 'val' in raw_dir:
            self.selected_files = [
                "1000a444aa94927d.pkl",
                "1001824289d8eed3.pkl",
                "1001ebb6d3905d92.pkl",
                "1002fdc9826fc6d1.pkl",
                "10040e572b831a04.pkl",
                "10042b19381bfbcd.pkl",
                "10067cf7cc2506c7.pkl",
                "1006b706483b11f9.pkl",
                "10071ee58db4bd92.pkl",
                "10083669957ee5f8.pkl",
                "10089d1384111b08.pkl",
                "1008aa4114dbc237.pkl",
                "1008b7b63e2d60.pkl",
                "1008f05c233dd975.pkl",
                "100b939eefa4a0de.pkl",
                "100bbbc583f55cbd.pkl",
                "100cf5864d3bfbed.pkl",
                "100d033b60683a9f.pkl",
                "100f370df1797a88.pkl",
                "100f9b9f8af6036f.pkl",
                "1010cc7e3a91ebc5.pkl",
                "1015e9446e86cfa0.pkl",
                "1016c21f14ba11e2.pkl",
                "10195df1c4a2c3ad.pkl",
                "101a844960d63c3f.pkl",
                "101aa4d1dc71df5e.pkl",
                "101acf02f749093f.pkl",
                "101b00dd28e01037.pkl",
                "101ba4c98d705f0.pkl",
                "101c25888a0fcf63.pkl",
                "101d7af08d9b56ae.pkl",
                "101f37bb58da79c.pkl"
            ]


       # shuffle(self._raw_paths)
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None


        #self.cache_data={}

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
        if 'val' in self._raw_paths[idx]:
            with open('./waymo_data/full/validation_map2/'+self.selected_files[idx], "rb") as handle:
                data = pickle.load(handle)
        else:
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
