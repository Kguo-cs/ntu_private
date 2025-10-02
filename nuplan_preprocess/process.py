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

from typing import Any, Dict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d
import pandas as pd
import warnings

# Convert all RuntimeWarnings to errors
warnings.simplefilter("error", RuntimeWarning)

_polygon_types = ["lane","crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]



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

def get_map_features(map_infos, tf_current_light, dim=2):
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

            point_position[_idx] = centerline[:-1, :dim]
            center_vectors = centerline[1:] - centerline[:-1]
            # point_orientation[_idx] = torch.cat(
            #     [torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0
            # )
            point_type[_idx] = torch.full(
                (len(center_vectors),), _seg["type"], dtype=torch.uint8
            )

            if _key == "lane":
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

    # mp_edge = []
    #
    # for edge in map_infos['mp_edge']:
    #     if edge[1] in polygon_ids:
    #         mp_edge.append((polygon_ids.index(edge[0]), polygon_ids.index(edge[1])))
    #     else:
    #         mp_edge.append((polygon_ids.index(edge[0]), -1))
    #
    # mp_edge = np.array(mp_edge)

    # map_infos["mp_edge"]=mp_edge

    return map_data


def get_map_features(map_infos, tf_current_light, dim=2):
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

            point_position[_idx] = centerline[:-1, :dim]
            center_vectors = centerline[1:] - centerline[:-1]
            # point_orientation[_idx] = torch.cat(
            #     [torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0
            # )
            point_type[_idx] = torch.full(
                (len(center_vectors),), _seg["type"], dtype=torch.uint8
            )

            if _key == "lane":
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

    # mp_edge = []
    #
    # for edge in map_infos['mp_edge']:
    #     if edge[1] in polygon_ids:
    #         mp_edge.append((polygon_ids.index(edge[0]), polygon_ids.index(edge[1])))
    #     else:
    #         mp_edge.append((polygon_ids.index(edge[0]), -1))
    #
    # mp_edge = np.array(mp_edge)

    # map_infos["mp_edge"]=mp_edge

    return map_data


def get_polylines_from_polygon(polygon: np.ndarray) -> np.ndarray:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0,1,2,3,4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return np.array(pl)

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return np.concatenate([pl1, pl1[::-1], pl2, pl2[::-1]], axis=0)


def _interplating_polyline(polylines, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = []
    polylines_list = []
    euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
    euclidean_dists = np.concatenate([[0], euclidean_dists])
    breakpoints = np.where(euclidean_dists > 3)[0]
    breakpoints = np.concatenate([[0], breakpoints, [polylines.shape[0]]])
    for i in range(1, breakpoints.shape[0]):
        start = breakpoints[i - 1]
        end = breakpoints[i]
        dist_along_path_list.append(
            np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start]
        )
        polylines_list.append(polylines[start:end])

    if dist_along_path_list[0][1]==0:
        print(1)

    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = dist_along_path_list[idx]
        polylines_cur = polylines_list[idx]
        # Create interpolation functions for x and y coordinates
        fxy = interp1d(dist_along_path, polylines_cur, axis=0)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate(
            [new_dist_along_path, dist_along_path[[-1]]]
        )


        # Combine the new x and y coordinates into a single array
        new_polylines = fxy(new_dist_along_path)
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) % polyline_size
            final_index = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0],
        )
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(
                dimension=0, size=polyline_size + 1, step=polyline_size
            )
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size :]
            last_polyline = last_polyline[
                torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()
            ]
            if multi_polylines is not None:
                multi_polylines = torch.cat(
                    [multi_polylines, last_polyline.unsqueeze(0)], dim=0
                )
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0).to(torch.float32)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def preprocess_map(map_data: Dict[str, Any]) -> Dict[str, Any]:
    pt2pl = map_data[("map_point", "to", "map_polygon")]["edge_index"]
    split_polyline_type = []
    split_polyline_pos = []
    split_polyline_theta = []
    split_polygon_type = []
    split_light_type = []
    split_polyline_id=[]

    for i in sorted(torch.unique(pt2pl[1])):
        index = pt2pl[0, pt2pl[1] == i]
        if len(index) <= 2:
            continue

        polygon_type = map_data["map_polygon"]["type"][i]
        light_type = map_data["map_polygon"]["light_type"][i]
        cur_type = map_data["map_point"]["type"][index]
        cur_pos = map_data["map_point"]["position"][index, :2]

        # assert len(np.unique(cur_type)) == 1

        split_polyline = _interplating_polyline(cur_pos.numpy())
        if split_polyline is None:
            continue
        split_polyline_pos.append(split_polyline[..., :2])
        split_polyline_theta.append(split_polyline[..., 2])
        split_polyline_type.append(cur_type[0].repeat(split_polyline.shape[0]))
        split_polygon_type.append(polygon_type.repeat(split_polyline.shape[0]))
        split_light_type.append(light_type.repeat(split_polyline.shape[0]))
        split_polyline_id.append(torch.zeros([split_polyline.shape[0]],dtype=torch.int64)+i)

    data = {}
    if len(split_polyline_pos) == 0:  # add dummy empty map
        data["map_save"] = {
            # 6e4 such that it's within the range of float16.
            "traj_pos": torch.zeros([1, 3, 2], dtype=torch.float32) + 6e4,
            "traj_theta": torch.zeros([1], dtype=torch.float32),
        }
        data["pt_token"] = {
            "type": torch.tensor([0], dtype=torch.uint8),
            "pl_type": torch.tensor([0], dtype=torch.uint8),
            "light_type": torch.tensor([0], dtype=torch.uint8),
            "ln_id": torch.tensor([0], dtype=torch.uint8),
            "num_nodes": 1,
        }
    else:
        data["map_save"] = {
            "traj_pos": torch.cat(split_polyline_pos, dim=0),  # [num_nodes, 3, 2]
            "traj_theta": torch.cat(split_polyline_theta, dim=0)[:, 0],  # [num_nodes]
        }
        data["pt_token"] = {
            "type": torch.cat(split_polyline_type, dim=0),  # [num_nodes], uint8
            "pl_type": torch.cat(split_polygon_type, dim=0),  # [num_nodes], uint8
            "light_type": torch.cat(split_light_type, dim=0),  # [num_nodes], uint8
            "ln_id": torch.cat(split_polyline_id, dim=0),
            "num_nodes": data["map_save"]["traj_pos"].shape[0]
        }
    return data


def get_agent_features(
        track_infos, num_historical_steps, num_steps
) :
    """
    track_infos:
    object_id (100,) int64
    object_type (100,) uint8
    states (100, 91, 9) float32
    valid (100, 91) bool
    role (100, 3) bool
    """

    idx_agents_to_add = []
    for i in range(len(track_infos["object_id"])):
        add_agent = track_infos["valid"][i, num_historical_steps - 1]

        if add_agent:
            idx_agents_to_add.append(i)

    num_agents = len(idx_agents_to_add)
    out_dict = {
        "num_nodes": num_agents,
        "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
        "role": torch.zeros([num_agents, 3], dtype=torch.bool),
        "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
        "type": torch.zeros(num_agents, dtype=torch.uint8),
        "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
        "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
        "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
        "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
    }

    for i, idx in enumerate(idx_agents_to_add):

        out_dict["role"][i] = torch.from_numpy(track_infos["role"][idx])
        out_dict["id"][i] = track_infos["object_id"][idx]
        out_dict["type"][i] = track_infos["object_type"][idx]

        valid = track_infos["valid"][idx]  # [n_step]
        states = track_infos["states"][idx]

        object_shape = states[:, 3:6]  # [n_step, 3], length, width, height
        object_shape = object_shape[valid].mean(axis=0)  # [3]
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        valid_steps = np.where(valid)[0]
        position = states[:, :3]  # [n_step, dim], x, y, z
        velocity = states[:, 7:9]  # [n_step, 2], vx, vy
        heading = states[:, 6]  # [n_step], heading
        if valid.sum() > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            f_pos = interp1d(valid_steps, position[valid], axis=0)
            f_vel = interp1d(valid_steps, velocity[valid], axis=0)
            f_yaw = interp1d(valid_steps, np.unwrap(heading[valid], axis=0), axis=0)
            t_in = np.arange(t_start, t_end + 1)
            out_dict["valid_mask"][i, t_start: t_end + 1] = True
            out_dict["position"][i, t_start: t_end + 1] = torch.from_numpy(f_pos(t_in))
            out_dict["velocity"][i, t_start: t_end + 1] = torch.from_numpy(f_vel(t_in))
            out_dict["heading"][i, t_start: t_end + 1] = torch.from_numpy(f_yaw(t_in))
        else:
            t = valid_steps[0]
            out_dict["valid_mask"][i, t] = True
            out_dict["position"][i, t] = torch.from_numpy(position[t])
            out_dict["velocity"][i, t] = torch.from_numpy(velocity[t])
            out_dict["heading"][i, t] = torch.tensor(heading[t])

    return out_dict
