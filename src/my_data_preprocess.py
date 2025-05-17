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
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.tokens.my_token_processor import TokenProcessor

torch.set_float32_matmul_precision("high")

light_cluster=np.load("./initial_tokenizer/light_cluster.npy")#261
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
)


def process_light(data,map_infos,tf_lights,tf_current_light):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]#189
    polyline_index = [x["polyline_index"] for k in _polygon_types for x in map_infos[k]]#189
    all_polylines = map_infos["all_polylines"][:,:2]

    current_light_ids=torch.tensor(tf_current_light["lane_id"].values)
    if len(current_light_ids):#consider only the light position is known
        in_lane=torch.isin(current_light_ids,torch.tensor(polygon_ids))

        current_light_ids=current_light_ids[in_lane]

    light_pos=np.zeros([len(current_light_ids),2])
    light_polyline=np.zeros([len(current_light_ids),3,2])

    for i, current_light_id in enumerate(current_light_ids):
        light_idx=polygon_ids.index(current_light_id)

        polyline_range=polyline_index[light_idx]

        polyline=all_polylines[polyline_range[0]:polyline_range[1]]

        light_pos[i]=polyline[len(polyline)//2]

        light_polyline[i][0]=polyline[0]
        light_polyline[i][1]=polyline[len(polyline)//2]
        light_polyline[i][2]=polyline[-1]

    light_all = torch.zeros([len(current_light_ids), 90],dtype=torch.int8)

    for t in range(1,91):
        light_t = tf_lights.loc[tf_lights["time_step"] == t]
        for i, light_id in enumerate(current_light_ids):
            res = light_t[light_t["lane_id"] == light_id.item()]
            if len(res):
                light_all[i][t-1] = _polygon_light_type.index(res["state"].item())

    light_all=light_all.reshape(-1,18,5)

    light_match=torch.all(light_all[None]-light_cluster[:,None,None],axis=-1)

    light_token=torch.argmax(light_match.to(torch.int),dim=0).to(torch.int16)

    return light_token,torch.FloatTensor(light_pos),torch.FloatTensor(light_polyline).reshape(-1,6)

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


        tokenized_map, tokenized_agent = token_processor(data)

        light_token,light_pos,light_polyline=process_light(data,map_infos,tf_lights,tf_current_light)

        # Remove unnecessary keys
        tokenized_agent.pop('gt_pos_raw', None)
        tokenized_agent.pop("gt_head_raw", None)
        tokenized_agent.pop("gt_valid_raw", None)
        tokenized_agent.pop('gt_z_raw', None)
        tokenized_agent.pop('gt_idx', None)
        tokenized_agent.pop('gt_heading', None)
        tokenized_agent.pop('gt_pos', None)

        tokenized_agent["sampled_idx"] = tokenized_agent["sampled_idx"].to(torch.int16)
        tokenized_agent["light_token"] = light_token
        tokenized_agent["light_pos"] = light_pos
        tokenized_agent["light_polyline"] = light_polyline

        tokenized_agent["num_nodes"] = len(tokenized_agent["sampled_pos"])

        tokenized_map["token_idx"] = tokenized_map["token_idx"].to(torch.int16)
        tokenized_map["num_nodes"] = len(tokenized_map["position"])


        data = {"tokenized_map": tokenized_map, "tokenized_agent": tokenized_agent}

        # data["scenario_id"] = scenario_id
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
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
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
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )