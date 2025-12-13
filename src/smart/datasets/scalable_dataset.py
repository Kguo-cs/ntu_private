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


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        # self.val='val' in raw_dir
        self.bird = 'bird' in raw_dir

        # if self.brid:
        #     self.pos_data = torch.load("/home/ke/code/catk/src/waymo_data/bird_data1/pos.pt")
        #
        #     frame=torch.unique(self.pos_data[:,0])
        #
        #     self.frame_len=20
        #
        #     frame_fut=frame+self.frame_len-1
        #
        #     mask = torch.isin(frame_fut,frame)
        #     frame_keep = frame[mask]
        #
        #     if 'train' in raw_dir:
        #         self._raw_paths=frame_keep[frame_keep%1000!=0]
        #     else:
        #         self._raw_paths=frame_keep[frame_keep//1000==0]
        #
        #
        #
        # else:

        self._raw_paths = [p.as_posix() for p in sorted(Path(raw_dir).glob("*"))]
        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None

        # if self.bird and tfrecord_dir is not None:
        #     # random_idx= np.random.choice(self._num_samples, size=64*5, replace=False)
        #     np.random.shuffle(self._raw_paths)
            #self._num_samples=64*5
            # print(self._raw_paths)

        # shuffle(self._raw_paths)
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

    def build_window_batch(self,pos_data, start_frame, frame_len: int, device="cpu"):

        # --- Parse dimensionality ---
        D = 3

        # --- Select agents: union of ids at start_frame+1 and start_frame+2 (as per your code) ---
        cur_ids = pos_data[pos_data[:, 0] == (start_frame + 2), 1]
        prev_ids = pos_data[pos_data[:, 0] == (start_frame + 1), 1]
        # Use numpy union (torch has no union op for tensors like this)
        track_id = np.union1d(cur_ids.numpy().astype(np.int64),
                              prev_ids.numpy().astype(np.int64))
        A = len(track_id)
        T = frame_len

        # --- Allocate outputs (torch) ---
        pos = torch.zeros((A, T, 3), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((A, T), dtype=torch.bool, device=device)

        # map id -> row index
        id2row = {int(tid): i for i, tid in enumerate(track_id.tolist())}

        window_frames = np.arange(start_frame, start_frame + T, dtype=np.int64)


        # --- Fill pos & valid_mask ---
        for t, f in enumerate(window_frames):
            fd = pos_data[pos_data[:, 0]== f]

            ids_f =  fd[:, 1]
            for j, tid in enumerate(ids_f):
                tid=int(tid)
                if tid not in id2row:
                    continue
                r = id2row[tid]
                pos[r, t, :3] =fd[j, 2:]
                valid_mask[r, t] = True

        return pos,valid_mask

    def fill_pos_fast_torch(self,pos_data: torch.Tensor, start_frame: int, frame_len: int, device=None):
        """
        pos_data: torch.float32 tensor with columns [frame, id, x, y, (z?)]
        Returns:
            track_id:   (A,) long
            pos:        (A, T, 3) float32
            valid_mask: (A, T) bool
        """
        if device is None:
            device = pos_data.device

        # Split columns
        frames_all = pos_data[:, 0].long()
        ids_all = pos_data[:, 1].long()
        coords_all = pos_data[:, 2:]  # (N, D)
        D = min(3, coords_all.shape[1])
        T = frame_len

        # --- select agent ids ---
        cur_ids = ids_all[frames_all == (start_frame + 3)]
        prev_ids = ids_all[frames_all == (start_frame + 2)]
        track_id = torch.unique(torch.cat([cur_ids, prev_ids]))
        track_id, _ = torch.sort(track_id)
        A = track_id.numel()

        # --- select frames in window ---
        win_mask = (frames_all >= start_frame) & (frames_all < start_frame + T)
        frames_w = frames_all[win_mask]
        ids_w = ids_all[win_mask]
        coords_w = coords_all[win_mask, :D].to(device)
        t_idx = (frames_w - start_frame).long()

        # --- map ids to agent rows ---
        pos_in_sorted = torch.searchsorted(track_id, ids_w)
        in_range = pos_in_sorted < A

        # clamp to avoid OOB, then compare
        pis_clamped = pos_in_sorted.clamp_max(A - 1)
        match = in_range & (track_id[pis_clamped] == ids_w)

        a_idx = pos_in_sorted[match]
        t_idx = t_idx[match]
        coords_w = coords_w[match]

        # --- scatter ---
        pos = torch.zeros((A, T, 3), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((A, T), dtype=torch.bool, device=device)

        # Use advanced indexing with an explicit dimension selector
        for d in range(D):
            pos.index_put_((a_idx, t_idx, torch.full_like(a_idx, d)), coords_w[:, d])
        valid_mask.index_put_((a_idx, t_idx), torch.ones_like(a_idx, dtype=torch.bool, device=device))

        return  pos, valid_mask

    def get(self, idx: int):
        #print(idx)
        # if  idx in self.cache_data.keys():
        #     data=self.cache_data[idx]
        # else:
        # if self.val:
        #     with open('./waymo_data/full/validation_map2/'+self.selected_files[idx//device_number], "rb") as handle:
        #         data = pickle.load(handle)
        # else:

        idx = idx // num_gpus

        # if self.brid:
        #     start_frame=self._raw_paths[idx]
        #
        #     #pos,valid_mask=self.build_window_batch(self.pos_data,start_frame=start_frame,frame_len=self.frame_len)
        #     pos,valid_mask=self.fill_pos_fast_torch(self.pos_data,start_frame,self.frame_len)
        #     #print(idx)
        #
        #     #print(torch.all(valid_mask==valid_mask1),torch.all(pos==pos1))
        #     data={"agent":{"valid_mask":valid_mask,"position":pos,"num_nodes":len(valid_mask)}}
        #
        # else:
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)


        # if 'keguo' in working_dir:
        #     self.cache_data[idx] = data

        # print(self.raw_paths[idx])

        if self._tfrecord_dir is not None and not self.bird:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
