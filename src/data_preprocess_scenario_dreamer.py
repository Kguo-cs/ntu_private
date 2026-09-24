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
import random
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
import os
from src.smart.utils.geometry import wrap_angle
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map

# agent_types = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
# agent_roles = {0: "ego_vehicle", 1: "interest", 2: "predict"}
# polyline_type = {
#     # for lane
#     "TYPE_FREEWAY": 0,
#     "TYPE_SURFACE_STREET": 1,
#     "TYPE_STOP_SIGN": 2,
#     "TYPE_BIKE_LANE": 3,
#     # for roadedge
#     "TYPE_ROAD_EDGE_BOUNDARY": 4,
#     "TYPE_ROAD_EDGE_MEDIAN": 5,
#     # for roadline
#     "BROKEN": 6,
#     "SOLID_SINGLE": 7,
#     "DOUBLE": 8,
#     # for crosswalk, speed bump and drive way
#     "TYPE_CROSSWALK": 9,
# }
_polygon_types = ["lane", "road_edge", "road_line", "crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]


def _sd_rotate(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ matrix.T


def _sd_wrap(angles):
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def _sd_resample(points: np.ndarray, count: int) -> np.ndarray:
    """Arc-length interpolation, including the upstream one-point-lane case."""
    points = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(targets, cumulative, points[:, axis]) for axis in range(2)
    ])


def get_scenario_dreamer_compact_lanes(scenario) -> Dict[int, np.ndarray]:
    """Build the compact centerlines needed by the SD offroad filter.

    Uses raw proto lane types (1=freeway, 2=surface street). Do not use the
    uploaded map_infos type labels: that decoder relabels undefined lanes.
    Successive degree-two lanes are concatenated, omitting the next lane's
    first point, as in SD. Lateral edges are unnecessary for this operation.

    A malformed reference to an excluded lane is removed completely; unlike
    upstream's list mutation during iteration, this cannot leave stale IDs.
    This safety difference only concerns malformed/dangling graph references.
    """
    raw = {
        int(feature.id): feature.lane
        for feature in scenario.map_features
        if feature.HasField("lane")
    }
    # Upstream constructs connectivity only between lanes with >=2 points.
    connected_ids = {i for i, lane in raw.items() if len(lane.polyline) >= 2}
    lanes = {
        i: np.array([(p.x, p.y) for p in lane.polyline], dtype=np.float64)
        for i, lane in raw.items()
        if int(lane.type) in (1, 2) and len(lane.polyline) > 0
    }
    if not lanes:
        return {}

    predecessors, successors = {}, {}
    for lane_id in lanes:
        lane = raw[lane_id]
        predecessors[lane_id] = [
            int(i) for i in lane.entry_lanes
            if lane_id in connected_ids and int(i) in connected_ids and int(i) in lanes
        ]
        successors[lane_id] = [
            int(i) for i in lane.exit_lanes
            if lane_id in connected_ids and int(i) in connected_ids and int(i) in lanes
        ]

    starts = {
        i for i in set(predecessors) | set(successors)
        if not (len(predecessors[i]) == 1
                and len(successors[predecessors[i][0]]) == 1)
    }
    visited, groups = set(), []

    def walk(start):
        chain = []
        current = start
        while current not in visited:
            visited.add(current)
            chain.append(current)
            if len(successors[current]) != 1:
                break
            nxt = successors[current][0]
            if len(predecessors[nxt]) != 1:
                break
            current = nxt
        if chain:
            groups.append(chain)

    for lane_id in starts:
        walk(lane_id)
    # Fully closed degree-two cycles have no starting lane.
    while len(visited) < len(lanes):
        walk(list(set(predecessors) - visited)[0])

    return {
        group_id: np.concatenate([
            lanes[lane_id] if offset == 0 else lanes[lane_id][1:]
            for offset, lane_id in enumerate(chain)
        ], axis=0)
        for group_id, chain in enumerate(groups)
    }


def _sd_lanes_in_view(
    compact_lanes: Mapping[int, np.ndarray],
    center: np.ndarray,
    rotation: float,
    fov: float,
    upsample_points: int,
) -> Dict[int, np.ndarray]:
    """Crop stored vertices, THEN resample each retained compact lane."""
    result = {}
    for lane_id, points in compact_lanes.items():
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("compact_lanes values must have shape [P, 2].")
        if not np.isfinite(points).all():
            raise ValueError("Lane coordinates must be finite.")
        local = _sd_rotate(points - center, rotation)
        inside = (np.abs(local) < fov / 2).all(axis=1)
        if inside.any():
            result[lane_id] = _sd_resample(local[inside], upsample_points)
    return result


def select_scenario_dreamer_agents(
    positions_ego: np.ndarray,
    object_types: np.ndarray,
    valid: np.ndarray,
    lane_points_ego: np.ndarray,
    *,
    fov: float = 64.0,
    max_num_agents: int = 30,
    offroad_threshold: float = 1.5,
    generate_only_vehicles: bool = False,
    remove_offroad_agents: bool = True,
) -> np.ndarray:
    """Return ORIGINAL track-row indices in SD's distance-sorted order.

    Types use the user's encoding: vehicle=0, pedestrian=1, cyclist=2.
    Order matters: valid/type -> nearest 30 -> square FOV -> offroad removal.
    There is NO refill, lane-heading test, velocity-heading test or collision
    test. The first distance-sorted row is exempt from offroad removal, exactly
    as in SD (normally the ego at the origin). Equal-position ties retain
    NumPy's argsort semantics rather than introducing a new ego tie-breaker.

    lane_points_ego must contain the 1000-point, cropped COMPACT lanes, not the
    later 20-point lanes used to serialize the scene or compute JSD.
    """
    xy = np.asarray(positions_ego, dtype=np.float64)
    types = np.asarray(object_types)
    valid = np.asarray(valid, dtype=bool)
    road = np.asarray(lane_points_ego, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions_ego must have shape [N, 2].")
    if valid.shape != (len(xy),) or types.shape != (len(xy),):
        raise ValueError("valid and object_types must have shape [N].")
    if road.ndim != 2 or road.shape[1] != 2:
        raise ValueError("lane_points_ego must have shape [P, 2].")
    if fov <= 0 or max_num_agents < 1 or offroad_threshold < 0:
        raise ValueError("Require fov > 0, max_num_agents >= 1, threshold >= 0.")
    if not np.isfinite(road).all():
        raise ValueError("Lane coordinates must be finite.")

    accepted_types = (0,) if generate_only_vehicles else (0, 1, 2)
    candidates = np.flatnonzero(valid & np.isin(types, accepted_types))
    if len(candidates) == 0 or len(road) == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(xy[candidates]).all():
        raise ValueError("Valid agent coordinates must be finite.")

    distance = np.linalg.norm(xy[candidates], axis=1)
    selected = candidates[np.argsort(distance)[:max_num_agents]]
    inside = (np.abs(xy[selected]) < fov / 2).all(axis=1)
    selected = selected[inside]

    if remove_offroad_agents and len(selected) > 1:
        keep = np.ones(len(selected), dtype=bool)
        # <=30 agents: loop over agents to avoid allocating [N, all_lane_pts, 2].
        for row, original_idx in enumerate(selected[1:], start=1):
            if types[original_idx] == 0:
                distance = np.linalg.norm(road - xy[original_idx], axis=1).min()
                keep[row] = distance <= offroad_threshold
        selected = selected[keep]
    return selected.astype(np.int64, copy=False)


def get_agent_features(
    track_infos: Dict[str, np.ndarray],
    split,
    num_historical_steps: int,
    num_steps: int,
    all_agent: bool = False,
    *,
    scenario=None,
    compact_lanes: Optional[Mapping[int, np.ndarray]] = None,
    scene_timestep=None,
    rng: Optional[random.Random] = None,
    fov: float = 64.0,
    max_num_agents: int = 30,
    offroad_threshold: float = 1.5,
    generate_only_vehicles: bool = False,
    remove_offroad_agents: bool = True,
    return_scene_info: bool = False,
):
    """SD initial-scene agent selection with the original SMART output fields.

    Required map input: raw `scenario`, or OFFICIAL compact_lanes in world XY.
    scene_timestep=None: uniformly sample an ego-valid raw frame (SD behavior).
    scene_timestep="current": use num_historical_steps-1 (explicit adaptation).
    An integer fixes the raw frame, e.g. the official eval cache's timestep.
    `split` is kept for API compatibility and does not change the selection.

    No gap interpolation. Invalid and padded trajectory slots stay zero/False.
    All trajectories stay in WORLD coordinates and original time indices.
    shape uses the last raw valid frame (SD uses its length/width; height is
    retained here only to preserve your [N,3] shape schema).

    When return_scene_info=True, return (agent_dict, scene_info). The latter
    holds the selected original indices, sampled timestep, and unnormalised
    SD [x,y,speed,cos_heading,sin_heading,length,width] features in EGO +Y frame.
    Empty/invalid scenes return num_nodes=0 and valid_scene=False.
    """
    if all_agent:
        raise ValueError("all_agent=True bypasses SD selection; use False.")
    if scenario is None and compact_lanes is None:
        raise ValueError("Pass scenario=... or official compact_lanes=...; the SD filter needs lanes.")

    states = np.asarray(track_infos["states"])
    valid = np.asarray(track_infos["valid"], dtype=bool)
    types = np.asarray(track_infos["object_type"])
    roles = np.asarray(track_infos["role"], dtype=bool)
    ids = np.asarray(track_infos["object_id"], dtype=np.int64)
    if states.ndim != 3 or states.shape[2] != 9:
        raise ValueError("states must be [N,T,9]: x,y,z,l,w,h,heading,vx,vy.")
    n, total_steps, _ = states.shape
    if valid.shape != (n, total_steps) or roles.shape != (n, 3):
        raise ValueError("valid or role has the wrong shape.")
    if types.shape != (n,) or ids.shape != (n,):
        raise ValueError("object_type and object_id must have shape [N].")
    if num_steps < total_steps:
        raise ValueError("num_steps must cover the raw sequence; do not silently truncate SD input.")
    if fov <= 0 or max_num_agents < 1 or offroad_threshold < 0:
        raise ValueError("Invalid FOV, agent limit or offroad threshold.")

    ego_rows = np.flatnonzero(roles[:, 0])
    if len(ego_rows) != 1:
        raise ValueError("Exactly one SDC must be identified by track_infos['role'][:,0].")
    ego = int(ego_rows[0])
    eligible_times = np.flatnonzero(valid[ego])
    scene_info = {
        "valid_scene": False, "scene_timestep": -1, "reason": "no_valid_ego_frame",
        "source_index": torch.empty(0, dtype=torch.int64), "ego_index": -1,
        "agent_states": torch.empty((0, 7), dtype=torch.float32),
        "agent_types": torch.empty((0, 3), dtype=torch.float32),
        "road_points": torch.empty((0, 20, 2), dtype=torch.float32),
        "coordinate_frame": "ego_y_forward", "normalized": False,
    }
    selected = np.empty(0, dtype=np.int64)
    local_xy, local_lanes, center, rotation = None, {}, None, None

    if len(eligible_times):
        if scene_timestep is None:
            source_rng = random if rng is None else rng
            timestep = int(eligible_times[source_rng.randrange(len(eligible_times))])
        elif isinstance(scene_timestep, str) and scene_timestep == "current":
            timestep = num_historical_steps - 1
        elif isinstance(scene_timestep, (int, np.integer)):
            timestep = int(scene_timestep)
        else:
            raise ValueError("scene_timestep must be None, 'current', or an integer.")
        if not 0 <= timestep < total_steps:
            raise ValueError(f"scene_timestep={timestep} outside raw sequence length {total_steps}.")
        scene_info["scene_timestep"] = timestep
        scene_info["reason"] = "ego_invalid_at_selected_frame"
        if valid[ego, timestep]:
            frame = np.asarray(states[:, timestep], dtype=np.float64)
            if not np.isfinite(frame[valid[:, timestep]]).all():
                raise ValueError("A raw valid state contains nonfinite values.")
            center = frame[ego, :2].copy()
            ego_heading = float(_sd_wrap(frame[ego, 6]))
            rotation = np.pi / 2 - ego_heading
            local_xy = _sd_rotate(frame[:, :2] - center, rotation)
            if compact_lanes is None:
                compact_lanes = get_scenario_dreamer_compact_lanes(scenario)
            local_lanes = _sd_lanes_in_view(compact_lanes, center, rotation, fov, 1000)
            scene_info["reason"] = "no_lane_vertices_in_fov"
            if local_lanes:
                road = np.concatenate(list(local_lanes.values()), axis=0)
                selected = select_scenario_dreamer_agents(
                    local_xy, types, valid[:, timestep], road,
                    fov=fov, max_num_agents=max_num_agents,
                    offroad_threshold=offroad_threshold,
                    generate_only_vehicles=generate_only_vehicles,
                    remove_offroad_agents=remove_offroad_agents,
                )
                scene_info["reason"] = "no_selected_agents" if len(selected) == 0 else ""

    # Same keys, shapes, dtypes and WORLD coordinate frame as the uploaded code.
    count = len(selected)
    out = {
        "num_nodes": count,
        "valid_mask": torch.zeros((count, num_steps), dtype=torch.bool),
        "role": torch.from_numpy(roles[selected].copy()),
        "id": torch.from_numpy(ids[selected].copy()),
        "type": torch.from_numpy(types[selected].astype(np.uint8)),
        "position": torch.zeros((count, num_steps, 3), dtype=torch.float32),
        "heading": torch.zeros((count, num_steps), dtype=torch.float32),
        "velocity": torch.zeros((count, num_steps, 2), dtype=torch.float32),
        "shape": torch.zeros((count, 3), dtype=torch.float32),
    }
    for row, source_idx in enumerate(selected):
        times = np.flatnonzero(valid[source_idx])
        source = states[source_idx, times]
        if not np.isfinite(source).all():
            raise ValueError(f"Nonfinite valid state for track {ids[source_idx]}.")
        out["valid_mask"][row, times] = True
        out["position"][row, times] = torch.as_tensor(source[:, :3], dtype=torch.float32)
        out["velocity"][row, times] = torch.as_tensor(source[:, 7:9], dtype=torch.float32)
        out["heading"][row, times] = torch.as_tensor(
            _sd_wrap(source[:, 6].astype(np.float64)), dtype=torch.float32
        )
        out["shape"][row] = torch.as_tensor(states[source_idx, times[-1], 3:6], dtype=torch.float32)

    if count:
        sampled = states[selected, scene_info["scene_timestep"]].astype(np.float64)
        local_velocity = _sd_rotate(sampled[:, 7:9], rotation)
        local_heading = _sd_wrap(_sd_wrap(sampled[:, 6]) + rotation)
        features = np.column_stack([
            local_xy[selected], np.linalg.norm(local_velocity, axis=1),
            np.cos(local_heading), np.sin(local_heading), out["shape"][:, :2].numpy(),
        ])
        # SD serializes the regular graph after downsampling and nearest-100 cap.
        lanes20 = np.stack([_sd_resample(p, 20) for p in local_lanes.values()])
        lane_order = np.argsort(np.linalg.norm(lanes20, axis=2).min(axis=1))[:100]
        ego_selected = np.flatnonzero(selected == ego)
        scene_info.update({
            "valid_scene": True,
            "source_index": torch.from_numpy(selected.copy()),
            "ego_index": int(ego_selected[0]) if len(ego_selected) else -1,
            "center_world": torch.tensor(center, dtype=torch.float64),
            "rotation_angle": float(rotation),
            "agent_states": torch.tensor(features, dtype=torch.float32),
            "agent_types": torch.from_numpy(np.eye(3, dtype=np.float32)[types[selected].astype(int)]),
            "road_points": torch.tensor(lanes20[lane_order], dtype=torch.float32),
        })
    return (out, scene_info) if return_scene_info else out


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


def decode_tracks_from_proto(scenario):
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = [i.track_index for i in scenario.tracks_to_predict]
    object_id_interest = [i for i in scenario.objects_of_interest]

    track_infos = {
        "object_id": [],
        "object_type": [],
        "states": [],
        "valid": [],
        "role": [],
    }
    for i, cur_data in enumerate(scenario.tracks):  # number of objects

        step_state = []
        step_valid = []
        for s in cur_data.states:
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                ]
            )
            step_valid.append(s.valid)
            # This angle is normalized to [-pi, pi). The velocity vector in m/s

        track_infos["object_id"].append(cur_data.id)
        track_infos["object_type"].append(cur_data.object_type - 1)
        track_infos["states"].append(np.array(step_state, dtype=np.float32))
        track_infos["valid"].append(np.array(step_valid))

        track_infos["role"].append([False, False, False])
        if i in track_index_predict:
            track_infos["role"][-1][2] = True  # predict=2
        if cur_data.id in object_id_interest:
            track_infos["role"][-1][1] = True  # interest=1
        if i == sdc_track_index:  # ego_vehicle=0
            track_infos["role"][-1][0] = True

    track_infos["states"] = np.array(track_infos["states"], dtype=np.float32)
    track_infos["valid"] = np.array(track_infos["valid"], dtype=bool)
    track_infos["role"] = np.array(track_infos["role"], dtype=bool)
    track_infos["object_id"] = np.array(track_infos["object_id"], dtype=np.int64)
    # Keep raw UNSET=-1 representable until the type filter removes it.
    track_infos["object_type"] = np.array(track_infos["object_type"], dtype=np.int16)
    return track_infos

def decode_map_features_from_proto(map_features,remove_mapid=[]):
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


def wm2argo(file_path, split, output_dir, output_dir_tfrecords_splitted, *, rng=None, scene_timestep=None):
    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )
    for tf_data in dataset:
        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        track_infos = decode_tracks_from_proto(scenario)
        # map_infos = decode_map_features_from_proto(scenario.map_features)
        # dynamic_map_infos = decode_dynamic_map_states_from_proto(
        #     scenario.dynamic_map_states
        # )
        #
        current_time_index = scenario.current_time_index
        scenario_id = scenario.scenario_id
        agent_data, scene_info = get_agent_features(
            track_infos,
            split=split,
            num_historical_steps=current_time_index + 1,
            num_steps=91,
            scenario=scenario,
            scene_timestep=scene_timestep,
            rng=rng,
            return_scene_info=True,
        )
        if not scene_info["valid_scene"]:
            continue

        # Use the same raw timestep for scene metadata and traffic signals.
        # Agent trajectories and SMART map points both remain in world coordinates.
        # tf_lights = process_dynamic_map(dynamic_map_infos)
        # tf_current_light = tf_lights.loc[
        #     tf_lights["time_step"] == scene_info["scene_timestep"]
        # ]
        # map_data = get_map_features(map_infos, tf_current_light)
        # data = preprocess_map(map_data)
        data={}
        data["agent"] = agent_data
        data["scenario_dreamer"] = scene_info

        data["scenario_id"] = scenario_id
        torch.save(data, os.path.join(output_dir, f"{scenario_id}.pt"))

        if output_dir_tfrecords_splitted is not None:
            file_name = output_dir_tfrecords_splitted / f"{scenario_id}.tfrecords"
            with tf.io.TFRecordWriter(file_name.as_posix()) as file_writer:
                file_writer.write(tf_data)


def batch_process9s_transformer(input_dir, output_dir, split, num_workers, seed=42, scene_timestep=None):
    output_dir = Path(output_dir)
    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = output_dir / "validation_tfrecords_splitted"
        output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])#[6:]
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
    rng = random.Random(seed)
    for file_path in tqdm(packages):
        wm2argo(
            file_path, split, output_dir, output_dir_tfrecords_splitted,
            rng=rng, scene_timestep=scene_timestep,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/ke/code/sim/src/waymo_data/waymo131",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/ke/code/sim/src/waymo_data/scenario_dreamer"
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene_timestep", default="random",
        help="random (SD sampling), current (original current frame), or an integer raw frame index",
    )
    args = parser.parse_args()
    if args.scene_timestep == "random":
        scene_timestep = None
    elif args.scene_timestep == "current":
        scene_timestep = "current"
    else:
        try:
            scene_timestep = int(args.scene_timestep)
        except ValueError:
            parser.error("--scene_timestep must be random, current, or an integer")

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers,
        seed=args.seed, scene_timestep=scene_timestep,
    )
