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

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from src.data_preprocess import decode_tracks_from_proto,decode_map_features_from_proto,decode_dynamic_map_states_from_proto,process_dynamic_map,get_map_features,get_agent_features,_polygon_types,_polygon_light_type
import matplotlib.pyplot as plt

def get_agent_routes(data,map_infos):
    positions=data["agent"]['position'][:,5::5,:2]
    heading=data["agent"]["heading"][:,5::5]
    valid_mask=data["agent"]['valid_mask'][:,5::5]
    last_velocity=data["agent"]['velocity'][:,-1]
    # type=data["agent"]["type"]

    # veh_mask=type==0
    # ped_mask=type==1
    # cyc_mask=type==2

    # veh_pos=positions[veh_mask]
    # ped_pos=positions[ped_mask]
    # cyc_pos=positions[cyc_mask]
    #
    # veh_dir=heading[veh_mask]
    # ped_dir=heading[ped_mask]
    # cyc_dir=heading[cyc_mask]

    extended_goal_pos=positions[:,-1] + last_velocity * 1

    positions=torch.cat([positions,extended_goal_pos[:,None]],dim=1)
    heading=torch.cat([heading,heading[:,-1:]],dim=1)
    valid_mask=torch.cat([valid_mask,valid_mask[:,-1:]],dim=1)

    map_data=data["map_save"]

    map_pos=map_data['traj_pos']
    map_dir=map_data['traj_theta']

    pl_type=data['pt_token']['type']
    ln_id=data['pt_token']['ln_id']

    lane_mask=torch.isin(pl_type,torch.tensor([0, 1, 2,3]))
    veh_lane_pos = map_pos[lane_mask][:, 1, :2].reshape(-1, 2)
    veh_lane_dir = map_dir[lane_mask]

    dist = torch.linalg.norm(veh_lane_pos[None,None] - positions[:,:,None], dim=-1)
    rot = torch.einsum('i,mj->mji', veh_lane_dir, heading)
    dist_rot=5*(rot<0)+dist
    routing_idx = torch.argmin(dist_rot,dim=-1)

    route_id=ln_id[routing_idx]

    route_id[~valid_mask]=-1

    dest_map_ids=route_id[:,-1]
    map_edge=map_infos["mp_edge"]

    inter_routes=[]

    for dest_map_id in dest_map_ids:

        next_list=[]

        next_map_id = dest_map_id
        max_len=4
        while next_map_id!=-1 and len(next_list)<max_len:
            next_edges = np.where(map_edge[:, 0] == next_map_id)[0]        # t_mask=valid_mask[veh_mask]
            dest_map_id, next_map_id = map_edge[np.random.choice(next_edges)]        # routing_dist = torch.amin(dist_rot,dim=-1)*t_mask
            next_list.append(next_map_id)

        next_list=next_list+[-1]*(max_len-len(next_list))

        inter_routes.append(next_list)

    route=torch.cat([route_id,torch.tensor(inter_routes)],dim=1)
    B, L = route.shape

    next_route=torch.zeros_like(route[:,:L-5])

    for t in range(L-5):
        fut_route=route[:,t+1:]

        cur_route=route[:,t]

        mask=fut_route!=cur_route[:,None]

        next_idx=torch.argmax(mask.to(torch.int),dim=1)+t+1

        next_route[:,t]=route[torch.arange(len(next_idx)),next_idx]

    # # Create tensors for time indices (i and j)
    # i_idx = torch.arange(L, device=route.device).view(1, L, 1)  # shape (1, L, 1)
    # j_idx = torch.arange(L, device=route.device).view(1, 1, L)  # shape (1, 1, L)
    #
    # # Expand route for broadcasting comparisons
    # route_i = route.unsqueeze(2)  # shape (B, L, 1)
    # route_j = route.unsqueeze(1)  # shape (B, 1, L)
    #
    # # Create a mask that is True when:
    # # 1. j > i (only later time steps)
    # # 2. route_j is different from route_i.
    # valid_mask = (j_idx > i_idx) & (route_j != route_i)  # shape: (B, L, L)
    #
    # # Build a tensor of candidate j indices, and set invalid ones to L (an out-of-bound index)
    # j_indices = j_idx.expand(B, L, L)
    # masked_j = torch.where(valid_mask, j_indices, torch.tensor(L, device=route.device))
    #
    # # For each (B, i), find the smallest j that is valid (i.e. the first later index with a different value)
    # min_j, _ = masked_j.min(dim=2)  # shape: (B, L); if no valid j exists, min_j will be L
    #
    # # To safely gather, clamp min_j to be within [0, L-1]
    # min_j_clamped = min_j.clamp(max=L - 1)
    # candidate = torch.gather(route, 1, min_j_clamped)
    #
    # # For positions where min_j == L, we use the original route value
    # next_routes = torch.where(min_j < L, candidate, route)
    #
    # print(next_routes)

    data["agent"]["route"]=route
    data["agent"]["next_route"]=next_route


def process_light(data,map_infos,tf_lights,tf_current_light):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]

    current_light_ids=torch.tensor(tf_current_light["lane_id"].values)

    if len(current_light_ids):
        in_lane=torch.isin(current_light_ids,torch.tensor(polygon_ids))

        current_light_ids=current_light_ids[in_lane]


    ln_id=data['pt_token']["ln_id"]

    light_edge=[]

    for i, current_light_id in enumerate(current_light_ids):
        light_ln_id=np.where(ln_id==polygon_ids.index(current_light_id))[0]

        light_edge.append(np.stack([i+np.zeros_like(light_ln_id),light_ln_id],axis=-1))

    if len(light_edge):
        light_edge=np.concatenate(light_edge).astype(int)
    else:
        light_edge=np.zeros([0,2]).astype(int)

    light=torch.zeros([len(current_light_ids),18]).to(torch.int64)

    for t in range(18):
        current_time_index=5+t*5
        light_t=tf_lights.loc[tf_lights["time_step"] == current_time_index]
        for i,light_id in enumerate(current_light_ids):
            res = light_t[light_t["lane_id"] == light_id]
            if len(res):
                light[i][t]=_polygon_light_type.index(res["state"].item())

    data["agent"]["light"] =light
    data["pt_token"]["light_edge"] =light_edge

def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:
        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        track_infos = decode_tracks_from_proto(scenario)
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(
            scenario.dynamic_map_states
        )

        current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id
        tf_lights = process_dynamic_map(dynamic_map_infos)
        tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
        map_data = get_map_features(map_infos, tf_current_light)

        data = preprocess_map(map_data)

        data["agent"] = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
        )

        process_light(data,map_infos,tf_lights,tf_current_light)

        get_agent_routes(data ,map_infos)#the routing is interpolated from the future trajectory until out of map
        # map_pos=data["map_save"]["traj_pos"].reshape(-1,2)
        #
        # xmin = map_pos[:, 0].min()
        # ymin = map_pos[:, 1].min()
        # xmax = map_pos[:, 0].max()
        # ymax = map_pos[:, 1].max()
        # data["map_boundary"]= np.array([xmin, xmax, ymin, ymax])



        data["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f)

        if output_dir_tfrecords_splitted is not None:
            file_name = output_dir_tfrecords_splitted / f"{scenario_id}.tfrecords"
            with tf.io.TFRecordWriter(file_name.as_posix()) as file_writer:
                file_writer.write(tf_data)


def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = output_dir / "validation_tfrecords_splitted"
        output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir / split
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
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )
