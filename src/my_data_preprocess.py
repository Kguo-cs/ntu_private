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
    light_idx = torch.zeros((len(current_light_ids), 91), dtype=torch.int8)

    for i, current_light_id in enumerate(current_light_ids):
        current_light_idx=polygon_ids.index(current_light_id)

        polyline_range=polyline_index[current_light_idx]

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

        light_polyline[i]=new_polylines#-start_pos[None]

    if len(current_light_ids):
        # Create a mapping from lane_id to index in light_all
        lane_id_to_index = {lid.item(): idx for idx, lid in enumerate(current_light_ids)}

        # Filter to only relevant lane_ids
        tf_lights_filtered = tf_lights[tf_lights["lane_id"].isin(lane_id_to_index)]

        # Also filter to time_step in 1–90
        # tf_lights_filtered = tf_lights_filtered[tf_lights_filtered["time_step"]%5==0]
            # (tf_lights_filtered["time_step"] >= 0) & (tf_lights_filtered["time_step"] <= 90)]

        # Map lane_id to row index
        tf_lights_filtered = tf_lights_filtered.copy()  # to avoid SettingWithCopyWarning
        tf_lights_filtered["row_idx"] = tf_lights_filtered["lane_id"].map(lane_id_to_index)

        # Map state to its index
        state_to_index = {state: idx for idx, state in enumerate(_polygon_light_type)}
        tf_lights_filtered["state_idx"] = tf_lights_filtered["state"].map(state_to_index)

        # Use .itertuples() for faster iteration and assign to tensor
        for row in tf_lights_filtered.itertuples(index=False):
            i = row.row_idx
            t = row.time_step  # convert to 0-based index
            s = row.state_idx
            if pd.notna(i) and pd.notna(s):
                light_idx[i, t] = s

    map_tensor=torch.tensor([3,4,0,1,2])
    light_idx = map_tensor[light_idx[:,5::5].long()]

    relative_pos=light_polyline[:,-1]-light_pos

    light_orient=np.arctan2(relative_pos[:, 1], relative_pos[:, 0])

    light={
        "light_idx": light_idx.to(torch.int8),
        "light_pos": torch.FloatTensor(light_pos),
        "light_orient": torch.FloatTensor(light_orient),
        "light_polyline":torch.FloatTensor(light_polyline),
        "num_nodes": light_idx.shape[0]
    }
    return light


def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0,
                                      num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                   axis=-1)  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = \
    (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    return ret_polylines, ret_polylines_mask


def process_map(polylines_list):

    split_polyline_pos = []
    split_polyline_type=[]

    for polylines in polylines_list:
        cur_type=torch.from_numpy(polylines[:,-2])
        polylines=polylines[:, :2]

        euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
        euclidean_dists = np.concatenate([[0], euclidean_dists])
        breakpoints = np.where(euclidean_dists > 3)[0]
        breakpoints = np.concatenate([[0], breakpoints, [polylines.shape[0]]])
        dist_along_path_list = []

        polylines_list=[]

        for i in range(1, breakpoints.shape[0]):
            start = breakpoints[i - 1]
            end = breakpoints[i]
            dist_along_path_list.append(
                np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start]
            )
            polylines_list.append(polylines[start:end])

        #multi_polylines_list = []
        #num_points = 10
        for idx in range(len(dist_along_path_list)):
            if len(dist_along_path_list[idx]) < 2 or dist_along_path_list[idx][-1]<1:
                continue

            dist_along_path = dist_along_path_list[idx]
            polylines_cur = polylines_list[idx]
            # Create interpolation functions for x and y coordinates
            fxy = interp1d(dist_along_path, polylines_cur, axis=0)

            num_points=int(dist_along_path[-1]//50+1)*10

            # Create an array of distances at which to interpolate
            new_dist_along_path = np.linspace(0, dist_along_path[-1], num_points) #[:num_points]

            # Combine the new x and y coordinates into a single array
            new_polylines = fxy(new_dist_along_path)
            new_polylines = torch.from_numpy(new_polylines)
            new_heading = torch.atan2(
                new_polylines[1:, 1] - new_polylines[:-1, 1],
                new_polylines[1:, 0] - new_polylines[:-1, 0],
            )
            new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
            new_polylines = torch.cat([new_polylines, new_heading], -1)

            new_polylines=new_polylines.reshape(-1,10,3)

            split_polyline_pos.append(new_polylines)
            split_polyline_type.append(cur_type[0].repeat(new_polylines.shape[0]))

    data = {}
    if len(split_polyline_pos) == 0:  # add dummy empty map
        data["tokenized_map"] = {
            # 6e4 such that it's within the range of float16.
            "traj_pos": torch.zeros([1, 3, 2], dtype=torch.float32) + 6e4,
            "traj_theta": torch.zeros([1], dtype=torch.float32),
            "type": torch.tensor([0], dtype=torch.uint8),
            "num_nodes": 1,
        }
    else:
        pos= torch.cat(split_polyline_pos, dim=0)
        data["tokenized_map"] = {
            "traj_pos": pos.to(torch.float32),  # [num_nodes, 3, 2]
            "type": torch.cat(split_polyline_type, dim=0).to(torch.int8),  # [num_nodes], uint8
            "num_nodes": pos.shape[0]
        }

    return data



def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:

        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        #track_infos = decode_tracks_from_proto(scenario)
        map_infos = decode_map_features_from_proto(scenario.map_features)
        # dynamic_map_infos = decode_dynamic_map_states_from_proto(
        #     scenario.dynamic_map_states
        # )## scenario.dynamic_map_states has stop_point

        # current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id
        # tf_lights = process_dynamic_map(dynamic_map_infos)
        # tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
        # map_data = get_map_features(map_infos, tf_current_light)
        # polylines = torch.from_numpy(map_infos['all_polylines_list'].copy())
        map_data = get_map_features(map_infos, [])
        data = preprocess_map(map_data)

        del data['pt_token']['light_type']
        del data['pt_token']['pl_type']

        #data= process_map(map_infos['all_polylines_list'])

        # data = preprocess_map(map_data)
        #
        # data["agent"] = get_agent_features(
        #     track_infos,
        #     split=split,
        #     num_historical_steps=current_time_index + 1,
        #     num_steps=91,
        # )
        #
        # data["light"]=process_light(map_infos,tf_lights,tf_current_light)

        #data["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data, f)

        # if output_dir_tfrecords_splitted is not None:
        #     file_name = output_dir_tfrecords_splitted / f"{scenario_id}.tfrecords"
        #     with tf.io.TFRecordWriter(file_name.as_posix()) as file_writer:
        #         file_writer.write(tf_data)

def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = output_dir / "validation_tfrecords_splitted"
        output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])#[1:]
    func = partial(
        wm2argo,
        split=split,
        output_dir=output_dir,
        output_dir_tfrecords_splitted=output_dir_tfrecords_splitted,
    )

    with multiprocessing.Pool(num_workers) as p:
        r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))
    # print(len(packages))
    # for file_path in tqdm(packages):
    #     wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/ke/Windows/waymo_data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/ke/code/catk/src/waymo_data/new"
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_workers", type=int, default=2)
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
