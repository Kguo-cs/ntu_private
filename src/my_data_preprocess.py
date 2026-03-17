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

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')
sys.path.append('/home/users/ntu/shanhelo/scratch/keguo_projects/sim')
sys.path.append('/mnt/d/code/sim')
sys.path.append('/home/ke/keguo/sim')
sys.path.append('/home/guoke/sim')

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from src.data_preprocess import decode_tracks_from_proto,decode_map_features_from_proto,decode_dynamic_map_states_from_proto,process_dynamic_map,get_map_features,get_agent_features,_polygon_types,_polygon_light_type

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


def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:

        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        scenario_id = scenario.scenario_id
        current_time_index = scenario.current_time_index
        #if scenario_id=='4d82fec943ddaa44':
        # map_infos = decode_map_features_from_proto(scenario.map_features)
        # dynamic_map_infos = decode_dynamic_map_states_from_proto(
        #  scenario.dynamic_map_states
        # )## scenario.dynamic_map_states has stop_point
        #
        # current_time_index = scenario.current_time_index
        #
        # tf_lights = process_dynamic_map(dynamic_map_infos)
        # tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
        # map_data = get_map_features(map_infos,tf_current_light,remove_last=False)
        #  # polylines = torch.from_numpy(map_infos['all_polylines_list'].copy())
        #  # map_data = get_map_features(map_infos, [])
        #data = preprocess_map(map_data)


    #  del data['pt_token']['light_type']
    #  del data['pt_token']['pl_type']

    # data={"edge":map_infos['road_edge_list']}

    # data= process_map(map_infos['all_polylines_list'])
        track_infos = decode_tracks_from_proto(scenario)

        agent = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
            all_agent=True
        )

        agent["position"]=agent["position"][:,:,:2]
        agent["shape"]=agent["shape"][:,:2]

        data={}

        data["agent"]=agent

        #del agent["id"]

        #data={"agent":agent["real_valid_mask"][:,10]}#"shape":agent["shape"]

        #
        # data["light"]=process_light(map_infos,tf_lights,tf_current_light)

        #data["scenario_id"] = scenario_id
        # with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
        #     pickle.dump(data, f)

        #print(1)

        torch.save(data, os.path.join(output_dir, f"{scenario_id}.pt"))

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
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    func = partial(
        wm2argo,
        split=split,
        output_dir=output_dir,
        output_dir_tfrecords_splitted=output_dir_tfrecords_splitted,
    )

    # with multiprocessing.Pool(num_workers) as p:
    #     r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))
    # print(len(packages))
    for file_path in tqdm(packages):
        wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/ke/keguo/waymo",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./waymo_data/all_agent"
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )

    # args.split='testing'
    #
    # batch_process9s_transformer(
    #     args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    # )
    #
    # args.split='training'
    #
    # batch_process9s_transformer(
    #     args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    # )
    #
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
