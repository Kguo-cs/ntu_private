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

    data["agent"]["route"]=route
    data["agent"]["next_route"]=next_route



def interpolate_polyline(polyline: torch.Tensor, num_points: int = 10) -> torch.Tensor:
    # Step 1: Compute segment lengths
    segment_vectors = polyline[1:] - polyline[:-1]  # (N-1, 2)
    segment_lengths = torch.norm(segment_vectors, dim=1)  # (N-1,)

    total_length = segment_lengths.sum()
    cumulative_lengths = torch.cat([torch.tensor([0.0], device=polyline.device), segment_lengths.cumsum(0)])  # (N,)

    # Step 2: Interpolate positions along the total length
    target_lengths = torch.linspace(0, total_length, steps=num_points)  # (num_points,)

    # Step 3: For each target length, find which segment it falls in
    points = []
    for t_len in target_lengths:
        seg_idx = torch.searchsorted(cumulative_lengths, t_len, right=True) - 1
        seg_idx = torch.clamp(seg_idx, 0, len(segment_vectors) - 1)

        seg_start = polyline[seg_idx]
        seg_vec = segment_vectors[seg_idx]
        seg_len = segment_lengths[seg_idx]

        # Avoid division by zero
        if seg_len > 0:
            alpha = (t_len - cumulative_lengths[seg_idx]) / seg_len
        else:
            alpha = 0.0

        point = seg_start + alpha * seg_vec
        points.append(point)

    return torch.stack(points)  # (num_points, 2)


def process_light(map_infos,tf_lights,tf_current_light):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]#189
    polyline_index = [x["polyline_index"] for k in _polygon_types for x in map_infos[k]]#189
    all_polylines = map_infos["all_polylines"][:,:2]

    current_light_ids=torch.tensor(tf_current_light["lane_id"].values)
    if len(current_light_ids):#consider only the light position is known
        in_lane=torch.isin(current_light_ids,torch.tensor(polygon_ids))

        current_light_ids=current_light_ids[in_lane]

    light_pos=np.zeros([len(current_light_ids),2])
    light_polyline=np.zeros([len(current_light_ids),10,2])
    light_all = torch.zeros((len(current_light_ids), 90), dtype=torch.int8)

    for i, current_light_id in enumerate(current_light_ids):
        light_idx=polygon_ids.index(current_light_id)

        polyline_range=polyline_index[light_idx]

        polylines=all_polylines[polyline_range[0]:polyline_range[1]]
        euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
        euclidean_dists = np.concatenate([[0], euclidean_dists])

        dist_along_path=np.cumsum(euclidean_dists)

        fxy = interp1d(dist_along_path, polylines, axis=0)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], dist_along_path[-1]/10)
        new_dist_along_path = np.concatenate(
            [new_dist_along_path[1:], dist_along_path[[-1]]]
        )
        new_polylines = fxy(new_dist_along_path)

        start_pos=polylines[0]
        light_pos[i]=start_pos

        light_polyline[i]=new_polylines-start_pos[None]

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

    light_idx=light_all[:,4::5]

    light={
        "light_idx": light_idx,
        "light_pos": torch.FloatTensor(light_pos),
        "light_orient": torch.FloatTensor(light_polyline).reshape(-1,20),
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
        dynamic_map_infos = decode_dynamic_map_states_from_proto(
            scenario.dynamic_map_states
        )
       # print(time.time()-time1)

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

        data={}

        data["light"]=process_light(map_infos,tf_lights,tf_current_light)

        # data = HeteroData(data).cuda()
        #
        # tokenized_map, tokenized_agent = token_processor(data)
        #
        # tokenized_agent.pop('gt_pos_raw', None)
        # tokenized_agent.pop("gt_head_raw", None)
        # tokenized_agent.pop("gt_valid_raw", None)
        # tokenized_agent.pop('gt_z_raw', None)
        # tokenized_agent.pop('gt_idx', None)
        # tokenized_agent.pop('gt_heading', None)
        # tokenized_agent.pop('gt_pos', None)
        #
        # for key in tokenized_map.keys():
        #     tokenized_map[key] = tokenized_map[key].cpu()
        #
        # for key in tokenized_agent.keys():
        #     tokenized_agent[key] = tokenized_agent[key].cpu()
        #
        # tokenized_map["token_idx"] = tokenized_map["token_idx"].to(torch.int16)
        # tokenized_agent["sampled_idx"] = tokenized_agent["sampled_idx"].to(torch.int16)
        #
        # tokenized_map["num_nodes"] = len(tokenized_map["position"])
        # tokenized_agent["num_nodes"] = len(tokenized_agent["sampled_pos"])
        #
        # tokenized_light={}
        # tokenized_light["light_token"] = light_token
        # tokenized_light["light_pos"] = light_pos
        # tokenized_light["light_polyline"] = light_polyline
        # tokenized_light["num_nodes"] = len(light_token)
        #
        # data = {"tokenized_map": tokenized_map, "tokenized_agent": tokenized_agent,"tokenized_light":tokenized_light}

        data["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f)

        # if output_dir_tfrecords_splitted is not None:
        #     file_name = output_dir_tfrecords_splitted / f"{scenario_id}.tfrecords"
        #     with tf.io.TFRecordWriter(file_name.as_posix()) as file_writer:
        #         file_writer.write(tf_data)
        # print(time.time()-time1)

        #print(1/0)


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
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )
    #
    # files = os.listdir(data_directory)
    #
    # for file in tqdm(files):
    #     process_file(file)

# # Set paths
# token_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_light/"
# data_directory = "/home/ke/code/catk/src/waymo_data/full/training/"

# # Worker function
# def process_file(filename):
#     input_path = os.path.join(data_directory, filename)
#     output_path = os.path.join(token_data_directory, filename)
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)

#     data= HeteroData(data).cuda()

#     tokenized_map, tokenized_agent = token_processor(data)

#     tokenized_agent.pop('gt_pos_raw', None)
#     tokenized_agent.pop("gt_head_raw", None)
#     tokenized_agent.pop("gt_valid_raw", None)
#     tokenized_agent.pop('gt_z_raw', None)
#     tokenized_agent.pop('gt_idx', None)
#     tokenized_agent.pop('gt_heading', None)
#     tokenized_agent.pop('gt_pos', None)
#     tokenized_map["token_idx"]=  tokenized_map["token_idx"].to(torch.int16)
#     tokenized_agent["light_token"]=data["light_token"]
#     tokenized_agent["light_pos"]= data["light_pos"]
#     tokenized_agent["light_polyline"]=data["light_polyline"]
#     tokenized_agent["sampled_idx"]=  tokenized_agent["sampled_idx"].to(torch.int16)

#     for key in tokenized_map.keys():
#         tokenized_map[key]=tokenized_map[key].cpu()


#     for key in tokenized_agent.keys():
#         tokenized_agent[key]=tokenized_agent[key].cpu()

#     tokenized_map["num_nodes"] = len(tokenized_map["position"])
#     tokenized_agent["num_nodes"] = len(tokenized_agent["sampled_pos"])

#     data_dict = {"tokenized_map": tokenized_map, "tokenized_agent": tokenized_agent}

#     # Save the tokenized data
#     with open(output_path, "wb") as f:
#         pickle.dump(data_dict, f)
