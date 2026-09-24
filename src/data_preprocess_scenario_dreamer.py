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

"""SMART preprocessing with the uploaded Scenario Dreamer single-scene rules.

Copy this file, scenario_dreamer_filter.py and sd_reference.py into the same
project directory. The SMART map/trajectory formats are retained.

For dataset-level reproduction, replay an explicit frame manifest. A seed alone
cannot reproduce the official processing order or 50k evaluation membership.
"""
import json
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from functools import partial
import multiprocessing

from scenario_dreamer_filter import (
    get_agent_features,
    get_get_agent_features,
    decode_tracks_from_proto,
)

_polygon_types = ["lane", "road_edge", "road_line", "crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]



def get_map_features(map_infos, tf_current_light,dim=2, remove_last=False):
    polygon_ids = [x["id"] for k in _polygon_types for x in map_infos[k]]
    num_polygons = len(polygon_ids)

    # initialization
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    # point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    for _key in _polygon_types:
        for _seg in map_infos[_key]:
            _idx = polygon_ids.index(_seg["id"])
            centerline = map_infos["all_polylines"][
                _seg["polyline_index"][0] : _seg["polyline_index"][1]
            ]
            centerline = torch.from_numpy(centerline).float()
            polygon_type[_idx] = _polygon_types.index(_key)

            if remove_last:
                point_position[_idx] = centerline[:-1, :dim]
                centerline = centerline[1:] - centerline[:-1]
            else:
                point_position[_idx] = centerline[:, :dim]
            # point_orientation[_idx] = torch.cat(
            #     [torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0
            # )
            point_type[_idx] = torch.full(
                (len(centerline),), _seg["type"], dtype=torch.uint8
            )

            if _key == "lane" and len(tf_current_light):
                res = tf_current_light[tf_current_light["lane_id"] == _seg["id"]]
                if len(res) != 0:
                    polygon_light_type[_idx] = _polygon_light_type.index(
                        res["state"].item()
                    )

    num_points = torch.tensor(
        [point.size(0) for point in point_position], dtype=torch.long
    )
    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )

    map_data = {
        "map_polygon": {},
        "map_point": {},
        ("map_point", "to", "map_polygon"): {},
    }
    map_data["map_polygon"]["num_nodes"] = num_polygons
    map_data["map_polygon"]["type"] = polygon_type
    map_data["map_polygon"]["light_type"] = polygon_light_type
    if len(num_points) == 0:
        map_data["map_point"]["num_nodes"] = 0
        map_data["map_point"]["position"] = torch.tensor([], dtype=torch.float)
        # map_data["map_point"]["orientation"] = torch.tensor([], dtype=torch.float)
        map_data["map_point"]["type"] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data["map_point"]["num_nodes"] = num_points.sum().item()
        map_data["map_point"]["position"] = torch.cat(point_position, dim=0)
        # map_data["map_point"]["orientation"] = wrap_angle(
        #     torch.cat(point_orientation, dim=0)
        # )
        map_data["map_point"]["type"] = torch.cat(point_type, dim=0)
    map_data["map_point", "to", "map_polygon"][
        "edge_index"
    ] = point_to_polygon_edge_index
    return map_data


def process_dynamic_map(dynamic_map_infos):
    lane_ids = dynamic_map_infos["lane_id"]
    tf_lights = []
    for t in range(len(lane_ids)):
        lane_id = lane_ids[t]
        time = np.ones_like(lane_id) * t
        state = dynamic_map_infos["state"][t]
        tf_light = np.concatenate([lane_id, time, state], axis=0)
        tf_lights.append(tf_light)
    tf_lights = np.concatenate(tf_lights, axis=1).transpose(1, 0)
    tf_lights = pd.DataFrame(data=tf_lights, columns=["lane_id", "time_step", "state"])
    tf_lights["time_step"] = tf_lights["time_step"].astype("int")
    tf_lights["lane_id"] = tf_lights["lane_id"].astype("int")
    tf_lights["state"] = tf_lights["state"].astype("str")
    tf_lights.loc[tf_lights["state"].str.contains("STOP"), ["state"]] = (
        "LANE_STATE_STOP"
    )
    tf_lights.loc[tf_lights["state"].str.contains("GO"), ["state"]] = "LANE_STATE_GO"
    tf_lights.loc[tf_lights["state"].str.contains("CAUTION"), ["state"]] = (
        "LANE_STATE_CAUTION"
    )
    tf_lights.loc[tf_lights["state"].str.contains("UNKNOWN"), ["state"]] = (
        "LANE_STATE_UNKNOWN"
    )
    return tf_lights


def decode_map_features_from_proto(map_features,remove_mapid=[]):
    from src.smart.utils.preprocess import get_polylines_from_polygon
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    point_cnt = 0
    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        # pip install waymo-open-dataset-tf-2-6-0==1.4.9, not updated, should be driveway
        if feature_data_type is None:
            continue

        if mf.id in remove_mapid:
            continue

        feature = getattr(mf, feature_data_type)
        if feature_data_type == "lane":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                if feature.type == 0:  # UNDEFINED
                    cur_info["type"] = 1
                elif feature.type == 1:  # FREEWAY
                    cur_info["type"] = 0
                elif feature.type == 2:  # SURFACE_STREET
                    cur_info["type"] = 1
                elif feature.type == 3:  # BIKE_LANE
                    cur_info["type"] = 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["lane"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_edge":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # assert feature.type > 0
                cur_info["type"] = feature.type + 3

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_edge"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type == "road_line":
            if len(feature.polyline) > 1:
                cur_info = {"id": mf.id}
                # there is no UNKNOWN = 0
                # BROKEN_SINGLE_WHITE = 1
                # SOLID_SINGLE_WHITE = 2
                # SOLID_DOUBLE_WHITE = 3
                # BROKEN_SINGLE_YELLOW = 4
                # BROKEN_DOUBLE_YELLOW = 5
                # SOLID_SINGLE_YELLOW = 6
                # SOLID_DOUBLE_YELLOW = 7
                # PASSING_DOUBLE_YELLOW = 8
                # assert feature.type > 0  # no UNKNOWN = 0
                if feature.type in [1, 4, 5]:
                    cur_info["type"] = 6  # BROKEN
                elif feature.type in [2, 6]:
                    cur_info["type"] = 7  # SOLID_SINGLE
                else:
                    cur_info["type"] = 8  # DOUBLE

                cur_polyline = np.stack(
                    [
                        np.array([p.x, p.y, p.z, cur_info["type"], cur_info["id"]])
                        for p in feature.polyline
                    ],
                    axis=0,
                )

                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                map_infos["road_line"].append(cur_info)
                polylines.append(cur_polyline)
                point_cnt += len(cur_polyline)

        elif feature_data_type in ["speed_bump", "driveway", "crosswalk"]:
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
            pl_polygon = get_polylines_from_polygon(xyz[polygon_idx])
            cur_info = {"id": mf.id, "type": 9}

            cur_polyline = np.stack(
                [
                    np.array([p[0], p[1], p[2], cur_info["type"], cur_info["id"]])
                    for p in pl_polygon
                ],
                axis=0,
            )

            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            map_infos["crosswalk"].append(cur_info)
            polylines.append(cur_polyline)
            point_cnt += len(cur_polyline)

    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")
        if feature_data_type == "stop_sign":
            feature = mf.stop_sign
            for l_id in feature.lane:
                # override FREEWAY/SURFACE_STREET with stop sign lane
                # BIKE_LANE remains unchanged
                is_found = False
                for _i in range(len(map_infos["lane"])):
                    if map_infos["lane"][_i]["id"] == l_id:
                        is_found = True
                        if map_infos["lane"][_i]["type"] < 2:
                            map_infos["lane"][_i]["type"] = 2
                # not necessary found, some stop sign lanes are for lane with length 1
                # assert is_found
    #map_infos["all_polylines_list"] = polylines
    #map_infos["road_edge_list"]=road_edge_list

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    signal_state = {
        0: "LANE_STATE_UNKNOWN",
        #  States for traffic signals with arrows.
        1: "LANE_STATE_ARROW_STOP",
        2: "LANE_STATE_ARROW_CAUTION",
        3: "LANE_STATE_ARROW_GO",
        #  Standard round traffic signals.
        4: "LANE_STATE_STOP",
        5: "LANE_STATE_CAUTION",
        6: "LANE_STATE_GO",
        #  Flashing light signals.
        7: "LANE_STATE_FLASHING_STOP",
        8: "LANE_STATE_FLASHING_CAUTION",
    }

    dynamic_map_infos = {"lane_id": [], "state": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state = [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))

    return dynamic_map_infos


def _parse_frame(value):
    if value in (None, "random"):
        return None
    if value == "current":
        return "current"
    return int(value)


def load_frame_manifest(path):
    """JSONL rows: scenario_id, scene_timestep, optional sample_name/lg_type.

    One scenario may have several reference frames. Replay is independent of
    TFRecord traversal order because each entry specifies a physical frame.
    It is NOT valid to infer scenario_id from an official raw-pickle basename.
    Use an explicit mapping when converting an official eval list.
    """
    if path is None:
        return None
    by_scenario, names = {}, set()
    with Path(path).open(encoding="utf-8") as file:
        for line_no, text in enumerate(file, 1):
            if not text.strip():
                continue
            entry = json.loads(text)
            sid = str(entry["scenario_id"])
            t = entry["scene_timestep"]
            if isinstance(t, bool) or not isinstance(t, int) or t < 0:
                raise ValueError(f"Manifest line {line_no}: timestep must be a nonnegative integer")
            lg_type = entry.get("lg_type", 0)
            if lg_type not in (0, 1):
                raise ValueError(f"Manifest line {line_no}: lg_type must be 0 or 1")
            name = entry.get("sample_name", f"{sid}_{lg_type}_{t}.pt")
            if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name or name in (".", ".."):
                raise ValueError(f"Manifest line {line_no}: sample_name must be a plain filename")
            name = str(Path(name).with_suffix(".pt"))
            if name in names:
                raise ValueError(f"Manifest line {line_no}: duplicate output name {name}")
            names.add(name)
            by_scenario.setdefault(sid, []).append({
                "scene_timestep": t, "sample_name": name, "lg_type": lg_type,
            })
    if not by_scenario:
        raise ValueError("Frame manifest is empty")
    return by_scenario


def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted,
            *, rng=None, scene_timestep=None, frame_manifest=None,
            save_scene_info=False, manifest_writer=None, written_names=None):
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
   # from src.smart.utils.preprocess import preprocess_map

    output_dir = Path(output_dir)
    seen = set()
    count = 0
    dataset = tf.data.TFRecordDataset(file_path, compression_type="")
    for tf_data in dataset:
        raw_bytes = bytes(tf_data.numpy())
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_bytes)
        sid = scenario.scenario_id
        if frame_manifest is not None and sid not in frame_manifest:
            continue
        seen.add(sid)
        requests = frame_manifest[sid] if frame_manifest is not None else [{
            "scene_timestep": scene_timestep, "lg_type": 0,
        }]
        track_infos = decode_tracks_from_proto(scenario)
        #map_infos = decode_map_features_from_proto(scenario.map_features)
        #dynamic = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)
        # if len(dynamic["lane_id"]):
        #     lights = process_dynamic_map(dynamic)
        # else:
        #     lights = pd.DataFrame(columns=["lane_id", "time_step", "state"])

        wrote_scenario = False
        for request in requests:
            agents, scene = get_agent_features(
                track_infos, split=split,
                num_historical_steps=scenario.current_time_index + 1,
                num_steps=max(91, track_infos["states"].shape[1]),
                scenario=scenario, scene_timestep=request["scene_timestep"],
                rng=rng, return_scene_info=True,
            )
            if not scene["valid_scene"]:
                if frame_manifest is not None:
                    raise ValueError(f"Requested reference sample {sid}: {scene['reason']}")
                continue
            #t = scene["scene_timestep"]
            #lg_type = request["lg_type"]
            #graph = scene["graphs"]["regular" if lg_type == 0 else "partitioned"]
            #scene["lg_type"] = lg_type
            #scene["road_points"] = graph["road_points"]
            #scene["num_lanes"] = graph["num_lanes"]
            name = request.get("sample_name", f"{sid}.pt")
            if written_names is not None:
                if name in written_names:
                    raise ValueError(f"Duplicate output sample {name}; use an explicit, unique manifest")
                written_names.add(name)

            #current_lights = lights.loc[lights["time_step"] == t]
            #data = preprocess_map(get_map_features(map_infos, current_lights))
            data={}
            data["agent"] = agents
            data["scenario_id"] = sid
            # Always keep the physical reference time, even without scene_info.
          #  data["scene_timestep"] = t
            #data["agent_source_index"] = scene["source_index"]
            # if save_scene_info:
            #     data["scenario_dreamer"] = scene
            torch.save(data, output_dir / name)
            count += 1
            wrote_scenario = True
            # if manifest_writer is not None:
            #     record = {
            #         "scenario_id": sid, "scene_timestep": t, "lg_type": lg_type,
            #         "sample_name": name, "source_tfrecord": Path(file_path).name,
            #         "selected_track_ids": agents["id"].tolist(),
            #     }
            #     manifest_writer.write(json.dumps(record) + "\n")
            #     manifest_writer.flush()
        if wrote_scenario and output_dir_tfrecords_splitted is not None:
            out_record = Path(output_dir_tfrecords_splitted) / f"{sid}.tfrecords"
            with tf.io.TFRecordWriter(str(out_record)) as writer:
                writer.write(raw_bytes)
    return seen, count


def batch_process9s_transformer(input_dir, output_dir, split, num_workers=1,
                                seed=10, scene_timestep=None, *, frame_manifest=None,
                                save_scene_info=False):
    """Sequential processing; no unannounced RNG reordering from worker pools."""
    if num_workers != 1:
        warnings.warn("Processing sequentially to keep RNG order fixed; num_workers is ignored.")
    input_dir = Path(input_dir) / split
    root = Path(output_dir)
    target = root / split
    target.mkdir(parents=True, exist_ok=True)
    record_dir = root / "validation_tfrecords_splitted" if split == "validation" else None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
    packages = sorted(p for p in input_dir.iterdir() if p.is_file() and "tfrecord" in p.name)
    if not packages:
        raise FileNotFoundError(f"No TFRecord files under {input_dir}")
    replay = load_frame_manifest(frame_manifest)
    generator = random.Random(seed)
    written, seen, total = set(), set(), 0
    log_path = target / "sample_manifest.jsonl"
    # Do not overwrite the very manifest being replayed.
    if frame_manifest is not None and Path(frame_manifest).resolve() == log_path.resolve():
        log_path = target / "sample_manifest_replayed.jsonl"
    # with log_path.open("w", encoding="utf-8") as writer:
    #     for path in tqdm(packages):
    #         found, count = wm2argo(
    #             str(path), split, target, record_dir, rng=generator,
    #             scene_timestep=scene_timestep, frame_manifest=replay,
    #             save_scene_info=save_scene_info, manifest_writer=writer,
    #             written_names=written,
    #         )
    #         seen.update(found)
    #         total += count

    func = partial(
        wm2argo,
        split=split,
        output_dir=output_dir,
        output_dir_tfrecords_splitted=None,
    )

    with multiprocessing.Pool(num_workers) as p:
        r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))


    if replay is not None:
        missing = set(replay) - seen
        if missing:
            raise FileNotFoundError(f"{len(missing)} requested scenarios not found; examples: {sorted(missing)[:5]}")
    print(f"Saved {total} samples; replay manifest: {log_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default='/home/ke/keguo/waymo', help="Directory containing training/validation/testing")
    parser.add_argument("--output_dir", default='./waymo_data/scenario_dreamer_data')
    parser.add_argument("--split", default="training", choices=["training", "validation", "testing"])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--scene_timestep", default="random", help="random, current, or raw-frame integer")
    parser.add_argument("--frame_manifest", help="JSONL with explicit scenario_id / scene_timestep entries")
    parser.add_argument("--save_scene_info", action="store_true", help="Save exact-precision local features and both map variants")
    args = parser.parse_args()
    try:
        frame = _parse_frame(args.scene_timestep)
    except (ValueError, TypeError):
        parser.error("--scene_timestep must be random, current, or an integer")
    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, args.num_workers,
        args.seed, frame, frame_manifest=args.frame_manifest,
        save_scene_info=args.save_scene_info,
    )
