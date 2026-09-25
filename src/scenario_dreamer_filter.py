"""Scenario Dreamer single-scene preprocessing + a SMART output adapter.

The numerical rules live in sd_reference.py (frozen uploaded source).
Use raw Scenario Dreamer pickles for the most direct reference comparison.
For TFRecords, the bridge deliberately starts with MessageToDict, just like
Scenario Dreamer's extractor. It does NOT filter using float32 SMART states.

Same raw input + same timestep + same configuration + same numerical runtime
is the scope of equivalence. This module does not claim to reconstruct an
unknown dataset split, filename mapping, or random-number history.
"""
from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sd_reference import (
    ReferenceSceneOps,
    extract_raw_waymo_data,
    get_compact_lane_graph,
    modify_agent_states,
)

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

@dataclass(frozen=True)
class SceneConfig:
    """Defaults of Scenario Dreamer's Waymo initial-scene preprocessing."""
    fov: float = 64.0
    max_num_agents: int = 30
    max_num_lanes: int = 100
    upsample_lane_num_points: int = 1000
    num_points_per_lane: int = 20
    offroad_threshold: float = 1.5
    generate_only_vehicles: bool = False
    remove_offroad_agents: bool = True

    def __post_init__(self):
        if not np.isfinite(self.fov) or self.fov <= 0:
            raise ValueError("fov must be finite and positive")
        if not np.isfinite(self.offroad_threshold) or self.offroad_threshold < 0:
            raise ValueError("offroad_threshold must be finite and nonnegative")
        for name in ("max_num_agents", "max_num_lanes", "upsample_lane_num_points", "num_points_per_lane"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def scenario_to_raw_data(scenario) -> tuple[dict, np.ndarray]:
    """Adapt a Waymo proto (or its MessageToDict result) to SD's raw pickle.

    Returns (raw_data, raw_object_row_to_original_track_row). This mapping is
    needed because the reference extractor drops tracks lacking a width field
    at their final valid state. No map-reference cleanup is done here.

    The bridge keeps the source extractor's sparse connectivity dictionaries,
    insertion order, class filtering stage, degree conversion, and field
    presence semantics. Non-lane map features are not needed by the numerical
    core and are not copied into this auxiliary dictionary.
    """
    if isinstance(scenario, Mapping):
        data = scenario
    else:
        from google.protobuf.json_format import MessageToDict
        data = MessageToDict(scenario)

    labels = {
        "TYPE_UNSET": "unset", "TYPE_VEHICLE": "vehicle",
        "TYPE_PEDESTRIAN": "pedestrian", "TYPE_CYCLIST": "cyclist",
        "TYPE_OTHER": "other",
    }
    objects, rows = [], []
    av_index = -1
    for row, track in enumerate(data["tracks"]):
        if row == data["sdcTrackIndex"]:
            av_index = len(objects)  # Same placement as official get_objects.
        states = track["states"]
        final = 0
        for t, state in enumerate(states):
            if state["valid"]:
                final = t
        if "width" not in states[final]:
            continue
        objects.append({
            "position": [
                {"x": s["centerX"], "y": s["centerY"]}
                if s["valid"] else {"x": -1e4, "y": -1e4} for s in states
            ],
            "velocity": [
                {"x": s["velocityX"], "y": s["velocityY"]}
                if s["valid"] else {"x": -1e4, "y": -1e4} for s in states
            ],
            "heading": [math.degrees(s["heading"]) if s["valid"] else -1e4 for s in states],
            "length": states[final]["length"],
            "width": states[final]["width"],
            "valid": [s["valid"] for s in states],
            "type": labels[track["objectType"]],
        })
        rows.append(row)
    assert av_index != -1

    # get_lane_pairs operates on ALL >=2-point lanes before lane-type removal.
    # IMPORTANT: do not initialise empty entries, sort IDs, or clean excluded
    # references here: those changes alter the subsequent compact graph order.
    if "mapFeatures" not in data:
        print('no lane')
    lane_info = {}
    if "mapFeatures"  in data:
        for feature in data["mapFeatures"]:
            if "lane" in feature:
                lane_info[feature["id"]] = feature["lane"]
    engaged = {key: lane for key, lane in lane_info.items() if len(lane["polyline"]) >= 2}
    relations = {name: {} for name in ("pre_pairs", "suc_pairs", "left_pairs", "right_pairs")}
    for lane_id, lane in engaged.items():
        for field, relation in (("entryLanes", "pre_pairs"), ("exitLanes", "suc_pairs"),
                                ("leftNeighbors", "left_pairs"), ("rightNeighbors", "right_pairs")):
            for value in lane.get(field, []):
                other = value["featureId"] if field.endswith("Neighbors") else value
                if other in engaged:
                    relations[relation].setdefault(int(lane_id), []).append(int(other))

    graph = {name: {} for name in ("lanes", "pre_pairs", "suc_pairs", "left_pairs", "right_pairs")}
    for lane_id, lane in lane_info.items():
        if lane["type"] in ("TYPE_UNDEFINED", "TYPE_BIKE_LANE"):
            continue
        key = int(lane_id)
        xyz = np.array([[p["x"], p["y"], p["z"]] for p in lane["polyline"]])
        graph["lanes"][key] = xyz[:, :2]
        for relation, pairs in relations.items():
            if key in pairs:
                graph[relation][key] = pairs[key]
    return {"objects": objects, "av_idx": av_index, "lane_graph": graph}, np.asarray(rows, dtype=np.int64)


def build_scene(
    raw_data: Mapping[str, Any],
    *,
    scene_timestep: int | None = None,
    rng: random.Random | None = None,
    config: SceneConfig | None = None,
) -> dict[str, Any]:
    """Reference-equivalent initial scene from an OFFICIAL raw pickle dictionary.

    Output agent_states/road_points are unnormalised NumPy arrays; their native
    precision is preserved. Both regular and partitioned map geometries and
    adjacency matrices are included. Full PyG training graphs are not built.
    scene_timestep=None consumes one random.randrange draw after compaction,
    matching the uploaded slow path. Explicit timesteps consume no RNG draws.
    """
    cfg = config or SceneConfig()
    ops = ReferenceSceneOps()
    ops.cfg = cfg
    all_states, all_types = extract_raw_waymo_data(raw_data["objects"])
    ego = int(raw_data["av_idx"])
    compact = get_compact_lane_graph(copy.deepcopy(raw_data))

    if scene_timestep is None:
        valid_times = np.where(all_states[ego, :, -1] == 1)[0]
        generator = random if rng is None else rng
        t = int(valid_times[generator.randrange(len(valid_times))])
    else:
        if isinstance(scene_timestep, bool) or not isinstance(scene_timestep, (int, np.integer)):
            raise TypeError("scene_timestep must be None or an integer raw-frame index")
        t = int(scene_timestep)
        if t < 0 or t >= all_states.shape[1]:
            raise IndexError(f"scene_timestep={t} outside [0, {all_states.shape[1]})")
    invalid = {"valid_scene": False, "scene_timestep": t, "source_index": np.empty(0, dtype=np.int64)}
    if not all_states[ego, t, -1]:
        return {**invalid, "reason": "ego_invalid_at_selected_frame"}

    normalizer = {"center": all_states[ego, t, :2].copy(), "yaw": all_states[ego, t, 4].copy()}
    graph = ops.normalize_compact_lane_graph(copy.deepcopy(compact), normalizer)
    graph = ops.get_lane_graph_within_fov(graph)
    if len(graph["lanes"]) == 0:
        return {**invalid, "reason": "no_lane_vertices_in_fov"}
    partitioned = ops.partition_compact_lane_graph(copy.deepcopy(graph))

    exists = copy.deepcopy(all_states[:, t, -1]).astype(bool)
    if cfg.generate_only_vehicles:
        type_mask = copy.deepcopy(all_types[:, 1]).astype(bool)
    else:
        type_mask = copy.deepcopy(all_types[:, 1] + all_types[:, 2] + all_types[:, 3]).astype(bool)
    exists = exists * type_mask
    agents = copy.deepcopy(all_states[exists, t])

    # Carry row IDs in an EXTRA TYPE column, not in the state matrix. This leaves
    # state layout, strides, arithmetic and all original numerical methods alone.
    # Both reference filtering methods pass type columns through unchanged.
    types_with_ids = np.column_stack([copy.deepcopy(all_types[exists]), np.flatnonzero(exists)])
    agents, types_with_ids = ops.get_agents_within_fov(agents, types_with_ids, normalizer)
    if cfg.remove_offroad_agents:
        agents, types_with_ids = ops.remove_offroad_agents(agents, types_with_ids, graph["lanes"])
    agents = modify_agent_states(agents)
    if len(agents) == 0:
        return {**invalid, "reason": "no_selected_agents"}

    variants = {}
    for lg_type, local_graph in enumerate((graph, partitioned)):
        points, pre, suc, left, right, n = ops.get_road_points_adj(local_graph)
        variants["regular" if lg_type == 0 else "partitioned"] = {
            "lg_type": lg_type, "road_points": points, "num_lanes": n,
            "pre_adj": pre, "suc_adj": suc, "left_adj": left, "right_adj": right,
        }
    rows = types_with_ids[:, -1].astype(np.int64)
    ego_rows = np.flatnonzero(rows == ego)
    return {
        "valid_scene": True, "reason": "", "scene_timestep": t,
        "source_index": rows, "ego_index": int(ego_rows[0]) if len(ego_rows) else -1,
        "agent_states": agents[:, :-1], "agent_types": types_with_ids[:, 1:4],
        "num_agents": len(agents), "num_lanes": variants["regular"]["num_lanes"],
        "road_points": variants["regular"]["road_points"], "lg_type": 0,
        "graphs": variants, "center_world": normalizer["center"],
        "rotation_angle": (np.pi / 2) + np.sign(-normalizer["yaw"]) * np.abs(normalizer["yaw"]),
        "coordinate_frame": "ego_y_forward", "normalized": False,
    }


def _to_torch(value):
    """Copy arrays without float32 downcasting; keep the reference precision."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_agent_features(
    track_infos: dict,
    split,
    num_historical_steps: int,
    num_steps: int,
    all_agent: bool = False,
    *,
    scenario=None,
    raw_data: Mapping | None = None,
    raw_to_track_index: np.ndarray | None = None,
    scene_timestep=None,
    rng: random.Random | None = None,
    fov: float = 64.0,
    max_num_agents: int = 30,
    offroad_threshold: float = 1.5,
    generate_only_vehicles: bool = False,
    remove_offroad_agents: bool = True,
    return_scene_info: bool = False,
):
    """Select with Scenario Dreamer, return the original SMART agent dictionary.

    - With scenario=..., selection uses the original proto -> MessageToDict
      pipeline, not potentially rounded track_infos['states'].
    - With raw_data=..., supply raw_to_track_index explicitly; do not assume
      object rows and original track rows match after omitted objects.
    - scene_timestep=None matches random valid-ego-frame selection.
      'current' is an EXPLICIT fixed-frame adaptation, not the random protocol.
    - out trajectories stay in WORLD coordinates and ORIGINAL time slots.
      Invalid slots remain zero/False. No temporal interpolation is performed.
    - SMART tensors remain float32; optional scene_info keeps the reference
      float64 cache values. Use those local features for exact cache comparisons.
    """
    del split  # Retained for source-compatible calls; it does not alter the mask.
    if all_agent:
        raise ValueError("all_agent=True bypasses Scenario Dreamer filtering; use False")
    if (scenario is None) == (raw_data is None):
        raise ValueError("Provide exactly one of scenario=... or raw_data=...")
    if scenario is not None:
        if raw_to_track_index is not None:
            raise ValueError("raw_to_track_index is generated automatically with scenario=...")
        raw_data, raw_to_track_index = scenario_to_raw_data(scenario)
    elif raw_to_track_index is None:
        raise ValueError("With raw_data, supply raw_to_track_index in the original track_infos order")

    states = np.asarray(track_infos["states"])
    valid = np.asarray(track_infos["valid"], dtype=bool)
    types = np.asarray(track_infos["object_type"])
    roles = np.asarray(track_infos["role"], dtype=bool)
    ids = np.asarray(track_infos["object_id"], dtype=np.int64)
    if states.ndim != 3 or states.shape[2] != 9:
        raise ValueError("states must be [N,T,9]: x,y,z,length,width,height,heading,vx,vy")
    n, T, _ = states.shape
    if valid.shape != (n, T) or roles.shape != (n, 3) or types.shape != (n,) or ids.shape != (n,):
        raise ValueError("track_infos array shapes do not agree")
    if num_steps < T:
        raise ValueError("num_steps must cover all raw time slots")
    mapping = np.asarray(raw_to_track_index)
    if mapping.dtype.kind not in "iu" or mapping.shape != (len(raw_data["objects"]),):
        raise ValueError("raw_to_track_index must be an integer array with one row per raw object")
    if len(np.unique(mapping)) != len(mapping) or np.any(mapping < 0) or np.any(mapping >= n):
        raise ValueError("raw_to_track_index contains duplicate or out-of-range indices")
    raw_valid = np.asarray([obj["valid"] for obj in raw_data["objects"]], dtype=bool)
    if raw_valid.shape != valid[mapping].shape or not np.array_equal(raw_valid, valid[mapping]):
        raise ValueError("Raw data and track_infos do not refer to identical valid masks/time slots")
    ego = int(raw_data["av_idx"])
    if not (0 <= ego < len(mapping)) or not roles[mapping[ego], 0] or roles[:, 0].sum() != 1:
        raise ValueError("SD ego mapping does not match track_infos role[:,0]")
    labels = {"unset": -1, "vehicle": 0, "pedestrian": 1, "cyclist": 2, "other": 3}
    raw_types = np.array([labels[obj["type"]] for obj in raw_data["objects"]])
    if not np.array_equal(raw_types, types[mapping]):
        raise ValueError("Raw data and track_infos type/order mismatch (model vehicle type must be 0)")

    if isinstance(scene_timestep, str):
        if scene_timestep != "current":
            raise ValueError("scene_timestep must be None, 'current', or an integer")
        scene_timestep = num_historical_steps - 1
    cfg = SceneConfig(fov=fov, max_num_agents=max_num_agents, offroad_threshold=offroad_threshold,
                      generate_only_vehicles=generate_only_vehicles, remove_offroad_agents=remove_offroad_agents)
    scene = build_scene(raw_data, scene_timestep=scene_timestep, rng=rng, config=cfg)
    # `scene["source_index"]` is in Scenario Dreamer's official selected order
    # (normally ego first). Keep that order untouched inside `scene_info`, because
    # scene["agent_states"] / scene["agent_types"] are reference SD local features.
    sd_raw_selected = np.asarray(scene["source_index"], dtype=np.int64).copy()
    sd_selected = mapping[sd_raw_selected].astype(np.int64)
    count = len(sd_selected)

    # SMART convention requested here: non-ego agents first, ego ALWAYS last.
    # Reordering happens only AFTER Scenario Dreamer's filtering, so it cannot
    # affect FOV/off-road selection or the max-agent truncation.
    if count:
        ego_rows = np.flatnonzero(roles[sd_selected, 0])
        if len(ego_rows) != 1:
            raise ValueError("Exactly one selected ego is required before moving ego to the last row")
        ego_row = int(ego_rows[0])
        output_order = np.concatenate([
            np.arange(count, dtype=np.int64)[np.arange(count) != ego_row],
            np.asarray([ego_row], dtype=np.int64),
        ])
        raw_selected = sd_raw_selected[output_order]
        selected = sd_selected[output_order]
    else:
        raw_selected = sd_raw_selected
        selected = sd_selected

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

    # IMPORTANT: the SMART output below is entirely in the ORIGINAL Waymo/world
    # frame. No ego translation/rotation is applied to position, velocity, or
    # heading. Invalid time slots remain zero and are identified by valid_mask.
    for row, original in enumerate(selected):
        times = np.flatnonzero(valid[original])
        values = states[original, times]
        out["valid_mask"][row, times] = True
        out["position"][row, times] = torch.as_tensor(values[:, :3], dtype=torch.float32)
        out["velocity"][row, times] = torch.as_tensor(values[:, 7:9], dtype=torch.float32)
        out["heading"][row, times] = torch.as_tensor(values[:, 6], dtype=torch.float32)
        obj = raw_data["objects"][raw_selected[row]]
        out["shape"][row] = torch.tensor(
            [obj["length"], obj["width"], states[original, times[-1], 5]],
            dtype=torch.float32,
        )

    # Sanity check for downstream SMART code: if agents exist, the last row is ego.
    if count and not bool(out["role"][-1, 0]):
        raise RuntimeError("Internal error: ego was not moved to the last output row")

    if not return_scene_info:
        return out

    # Keep SD reference order/coordinate frame intact in scene_info.
    # Add explicit output mapping for consumers that need to align scene_info
    # with the ego-last SMART out_dict.
    scene["raw_source_index"] = sd_raw_selected
    scene["source_index"] = sd_selected
    scene["output_raw_source_index"] = raw_selected.copy()
    scene["output_source_index"] = selected.copy()
    scene["output_ego_index"] = count - 1 if count else -1
    scene["output_coordinate_frame"] = "world"
    return out, _to_torch(scene)


# Accept the spelling used in the original request.
get_get_agent_features = get_agent_features


# Float64 before filtering; signed types preserve UNSET=-1.
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
        track_infos["states"].append(np.array(step_state, dtype=np.float64))
        track_infos["valid"].append(np.array(step_valid))

        track_infos["role"].append([False, False, False])
        if i in track_index_predict:
            track_infos["role"][-1][2] = True  # predict=2
        if cur_data.id in object_id_interest:
            track_infos["role"][-1][1] = True  # interest=1
        if i == sdc_track_index:  # ego_vehicle=0
            track_infos["role"][-1][0] = True

    track_infos["states"] = np.array(track_infos["states"], dtype=np.float64)
    track_infos["valid"] = np.array(track_infos["valid"], dtype=bool)
    track_infos["role"] = np.array(track_infos["role"], dtype=bool)
    track_infos["object_id"] = np.array(track_infos["object_id"], dtype=np.int64)
    track_infos["object_type"] = np.array(track_infos["object_type"], dtype=np.int16)
    return track_infos
