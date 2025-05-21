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
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from src.data_preprocess import decode_tracks_from_proto,decode_map_features_from_proto,decode_dynamic_map_states_from_proto,process_dynamic_map,get_map_features,get_agent_features,_polygon_types,_polygon_light_type

def get_agent_routes(data,map_infos):
    positions=data["agent"]['position'][:,5::5,:2]
    heading=data["agent"]["heading"][:,5::5]
    valid_mask=data["agent"]['valid_mask'][:,5::5]
    last_velocity=data["agent"]['velocity'][:,-1]

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

    data["agent"]["route"]=route
    data["agent"]["next_route"]=next_route



def process_light(map_infos,tf_lights,tf_current_light):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]#189
    polyline_index = [x["polyline_index"] for k in _polygon_types for x in map_infos[k]]#189
    all_polylines = map_infos["all_polylines"][:,:2]

    current_light_ids=torch.tensor(tf_current_light["lane_id"].values)
    if len(current_light_ids):#consider only the light position is known
        in_lane=torch.isin(current_light_ids,torch.tensor(polygon_ids))

        current_light_ids=current_light_ids[in_lane]

    light_pos=np.zeros([len(current_light_ids),2])
    light_polyline=np.zeros([len(current_light_ids),2,2])
    light_all = torch.zeros((len(current_light_ids), 90), dtype=torch.int8)

    for i, current_light_id in enumerate(current_light_ids):
        light_idx=polygon_ids.index(current_light_id)

        polyline_range=polyline_index[light_idx]

        polyline=all_polylines[polyline_range[0]:polyline_range[1]]

        start_pos=polyline[0]
        light_pos[i]=start_pos

        light_polyline[i][0]=polyline[len(polyline)//2]-start_pos
        light_polyline[i][1]=polyline[-1]-start_pos

    # light_all1 = torch.zeros([len(current_light_ids), 90],dtype=torch.int8)
    #
    # for t in range(1,91):
    #     light_t = tf_lights.loc[tf_lights["time_step"] == t]
    #     for i, light_id in enumerate(current_light_ids):
    #         res = light_t[light_t["lane_id"] == light_id.item()]
    #         if len(res):
    #             light_all1[i][t-1] = _polygon_light_type.index(res["state"].item())
    if len(current_light_ids):
        # Create a mapping from lane_id to index in light_all
        lane_id_to_index = {lid.item(): idx for idx, lid in enumerate(current_light_ids)}

        # Filter to only relevant lane_ids
        tf_lights_filtered = tf_lights[tf_lights["lane_id"].isin(lane_id_to_index)]

        # Also filter to time_step in 1–90
        tf_lights_filtered = tf_lights_filtered[
            (tf_lights_filtered["time_step"] >= 1) & (tf_lights_filtered["time_step"] <= 90)]

        # Map lane_id to row index
        tf_lights_filtered = tf_lights_filtered.copy()  # to avoid SettingWithCopyWarning
        tf_lights_filtered["row_idx"] = tf_lights_filtered["lane_id"].map(lane_id_to_index)

        # Map state to its index
        state_to_index = {state: idx for idx, state in enumerate(_polygon_light_type)}
        tf_lights_filtered["state_idx"] = tf_lights_filtered["state"].map(state_to_index)


        # Use .itertuples() for faster iteration and assign to tensor
        for row in tf_lights_filtered.itertuples(index=False):
            i = row.row_idx
            t = row.time_step - 1  # convert to 0-based index
            s = row.state_idx
            if pd.notna(i) and pd.notna(s):
                light_all[i, t] = s

    light_all=light_all.reshape(-1,18,5)


    light={
        "type": light_all,
        "pos": torch.FloatTensor(light_pos),
        "light_polyline": torch.FloatTensor(light_polyline).reshape(-1,4),
        "num_nodes": light_all.shape[0]
    }
    return light

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

        data={}

        data["agent"] = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
        )
        get_agent_routes(data, map_infos)


        data["scenario_id"] = scenario_id
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
