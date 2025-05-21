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

import multiprocessing
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os
import time

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from src.data_preprocess import decode_tracks_from_proto,get_agent_features



def decode_map_features_from_proto(map_features):
    map_infos = {}
    polylines = []
    #lane_headings = []
    point_cnt = 0
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
        if feature_data_type is None:
            continue

        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if len(feature.polyline) > 1:

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, len(polylines)])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                polyline_heading = np.arctan2(
                    cur_polyline[1:, 1] - cur_polyline[:-1, 1],
                    cur_polyline[1:, 0] - cur_polyline[:-1, 0],
                )

                polyline_heading=np.concatenate([polyline_heading[:1], polyline_heading])[:,None]

                cur_polyline=np.concatenate([cur_polyline, polyline_heading],axis=-1)

                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        #lane_headings=np.stack(lane_headings)
    except:
        polylines = np.zeros((0, 4), dtype=np.float32)
       # lane_headings=np.zeros((0,), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = torch.FloatTensor(polylines)
    #map_infos["lane_headings"] = torch.FloatTensor(lane_headings)

    return map_infos



def get_agent_routes(agent,map_infos):
    positions=agent['position'][:,1::5,:2]
    heading=agent["heading"][:,1::5]
    valid_mask=agent['valid_mask'][:,1::5]
    # last_velocity=agent['velocity'][:,-1]
    #
    # extended_goal_pos=positions[:,-1] + last_velocity * 0.5
    #
    # positions=torch.cat([positions,extended_goal_pos[:,None]],dim=1)
    # heading=torch.cat([heading,heading[:,-1:]],dim=1)
    # valid_mask=torch.cat([valid_mask,valid_mask[:,-1:]],dim=1)

    all_polylines=map_infos["all_polylines"]
    polyline_start_heading=map_infos["all_polylines"]

    lane_pos=all_polylines[:,:2]

    lane_dir=all_polylines[:,-1]

    ln_id=all_polylines[:,2]

    dist = torch.linalg.norm(lane_pos[None,None] - positions[:,:,None], dim=-1)
    rot = torch.einsum('i,mj->mji', lane_dir, heading)
    dist_rot=5*(rot<0)+dist
    routing_idx = torch.argmin(dist_rot,dim=-1)

    cur_lane_dir=lane_dir[routing_idx]

    route=ln_id[routing_idx]

    route[~valid_mask]=-1

    B, T = route.shape

    # Create a (T, T) upper triangle mask (i < j)
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)

    # Expand for batch
    route_exp = route[:, :, None]              # (B, T, 1)
    future_exp = route[:, None, :]             # (B, 1, T)

    # Compare: where future != current and time is in the future
    change_mask = (route_exp != future_exp) & mask[None, :, :]  # (B, T, T)

    # For each position, get first True index in time (axis=2)
    first_change_idx = change_mask.to(torch.int).argmax(axis=2)  # (B, T)


    print(1)

    # # Where no change is found, .argmax returns 0 — fix it:
    # no_change_mask = ~change_mask.to(torch.bool).any(axis=2)
    # first_change_idx[no_change_mask] = -1
    #
    # # Use advanced indexing to gather the next different value
    # batch_idx = torch.arange(B)[:, None]
    # next_route = torch.where(first_change_idx != -1,
    #                          route[batch_idx, first_change_idx],
    #                          -1)
    #
    # route_pair=torch.stack([route, next_route], dim=-1).reshape(-1,2)

    # unique_route_pair = torch.unique(route_pair, dim=0)
    #
    # avail_pair=unique_route_pair[(route_pair[:,0]!=-1)&(route_pair[:,1]!=-1)]
    #
    # lane_dir1=lane_dir[avail_pair[:,0]]

    #print("route pair shape:", route_pair.shape)




def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:
       # time1=time.time()

        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        track_infos = decode_tracks_from_proto(scenario)
        map_infos = decode_map_features_from_proto(scenario.map_features)

        current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id

        agent = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
        )
        data=get_agent_routes(agent, map_infos)


        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f)



def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = output_dir / "validation_tfrecords_splitted"
        output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir / (split+'route')
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])#[:1]
    # func = partial(
    #     wm2argo,
    #     split=split,
    #     output_dir=output_dir,
    #     output_dir_tfrecords_splitted=output_dir_tfrecords_splitted,
    # )
    #
    # with multiprocessing.Pool(num_workers) as p:
    #     r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))
    print(len(packages))
    for file_path in tqdm(packages):
        wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/ke/Windows/waymo_data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/ke/code/catk/src/waymo_data/full"
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )
