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
import sys

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/zs/code/sim')
sys.path.append('/mnt/d/code/sim')
sys.path.append('/home/ke/keguo/sim')
sys.path.append('/home/guoke/sim')

from data_preprocess import decode_dynamic_map_states_from_proto,decode_tracks_from_proto,decode_map_features_from_proto,process_dynamic_map,get_map_features

from scenario_dreamer_filter import (
    get_agent_features,
    get_get_agent_features,
    decode_tracks_from_proto,
)

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
    from src.smart.utils.preprocess import preprocess_map

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
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)
        if len(dynamic["lane_id"]):
            lights = process_dynamic_map(dynamic)
        else:
            lights = pd.DataFrame(columns=["lane_id", "time_step", "state"])

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
            t = scene["scene_timestep"]
            lg_type = request["lg_type"]
            graph = scene["graphs"]["regular" if lg_type == 0 else "partitioned"]
            scene["lg_type"] = lg_type
            scene["road_points"] = graph["road_points"]
            scene["num_lanes"] = graph["num_lanes"]
            name = request.get("sample_name", f"{sid}_{lg_type}_{t}.pt")
            if written_names is not None:
                if name in written_names:
                    raise ValueError(f"Duplicate output sample {name}; use an explicit, unique manifest")
                written_names.add(name)

            current_lights = lights.loc[lights["time_step"] == t]
            data = preprocess_map(get_map_features(map_infos, current_lights))
            data["agent"] = agents
            data["scenario_id"] = sid
            # Always keep the physical reference time, even without scene_info.
            data["scene_timestep"] = t
            data["agent_source_index"] = scene["source_index"]
            if save_scene_info:
                data["scenario_dreamer"] = scene
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
    packages = sorted(p for p in input_dir.iterdir() if p.is_file() and "tfrecord" in p.name)#[470:475]
    if not packages:
        raise FileNotFoundError(f"No TFRecord files under {input_dir}")
    replay = load_frame_manifest(frame_manifest)
    generator = random.Random(seed)
    written, seen, total = set(), set(), 0
    log_path = root / "sample_manifest.jsonl"
    # Do not overwrite the very manifest being replayed.
    if frame_manifest is not None and Path(frame_manifest).resolve() == log_path.resolve():
        log_path = root / "sample_manifest_replayed.jsonl"
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
    # print(len(packages))
    for file_path in tqdm(packages):
        wm2argo(file_path, split, target, None)

    # func = partial(
    #     wm2argo,
    #     split=split,
    #     output_dir=target,
    #     output_dir_tfrecords_splitted=None,
    # )
    #
    # with multiprocessing.Pool(num_workers) as p:
    #     r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))


    if replay is not None:
        missing = set(replay) - seen
        if missing:
            raise FileNotFoundError(f"{len(missing)} requested scenarios not found; examples: {sorted(missing)[:5]}")
    print(f"Saved {total} samples; replay manifest: {log_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default='./waymo_data/waymo110', help="Directory containing training/validation/testing")
    parser.add_argument("--output_dir", default='./waymo_data/full/training_sd')
    parser.add_argument("--split", default="training", choices=["training", "validation", "testing"])
    parser.add_argument("--num_workers", type=int, default=3)
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
