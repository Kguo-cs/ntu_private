"""Fast TrafficGen initial-state agent selection utilities.

This module replaces the slow `_get_trafficgen_data()` path that only returned
`ret["PZH_TRACK_NAMES"][ret["agent_mask"]]` after building a large intermediate
TrafficGen-style sample.  The fast path computes the same first-frame selection
mask directly from the Waymo scenario proto:

    valid agent
    non-zero object type
    within ego-centered 50m range
    close to a center lane
    velocity direction roughly aligned with heading
    heading roughly aligned with the nearest center-lane segment

It avoids extracting all timesteps, dynamic map states, unsampled lanes,
boundaries, crosswalks, rest map tokens, vector-based representations, and
padding/case-list construction.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


DEFAULT_CURRENT_T = 10
DEFAULT_LANE_RANGE = 50.0
DEFAULT_LANE_DIST_THRES = 5.0
DEFAULT_LANE_SAMPLE_NUM = 10
DEFAULT_MAX_CENTER_VECTORS = 384


CENTER_LANE_TYPES = {1, 2, 3}


def rotate(x, y, angle):
    """Rotate x/y by `angle` and return [..., 2].

    This keeps the old function name for backward compatibility, but the module
    no longer depends on torch.  `x`, `y`, and `angle` can be scalars or NumPy
    arrays that broadcast together.
    """
    return np.stack(
        [
            np.cos(angle) * x - np.sin(angle) * y,
            np.sin(angle) * x + np.cos(angle) * y,
        ],
        axis=-1,
    )


def cal_rel_dir(dir1, dir2):
    """Vectorized signed angular difference `dir1 - dir2` in [-pi, pi)."""
    return (dir1 - dir2 + np.pi) % (2.0 * np.pi) - np.pi


def _sample_polyline_xy(polyline: Iterable, sample_num: int) -> np.ndarray:
    """TrafficGen-style polyline downsampling for Waymo proto points."""
    points = np.asarray([[p.x, p.y] for p in polyline], dtype=np.float32)
    if points.shape[0] == 0:
        return points.reshape(0, 2)
    if points.shape[0] < sample_num:
        return points
    return points[::sample_num]


def _sdc_first_track_order(num_tracks: int, sdc_index: int) -> list[int]:
    """Return local track indices after swapping SDC to index 0."""
    if sdc_index < 0 or sdc_index >= num_tracks:
        raise IndexError(f"Invalid sdc_track_index={sdc_index} for {num_tracks} tracks")
    return [sdc_index] + [idx for idx in range(num_tracks) if idx != sdc_index]


def _extract_current_agents_and_names(scenario, current_t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one timestep of agents with SDC placed at index 0.

    Returns:
        agent: [N, 6], columns are x, y, vx, vy, heading, object_type.
        valid: [N], bool valid mask at current_t.
        names: [N], object array of track ids in the same SDC-first order.
    """
    tracks = scenario.tracks
    order = _sdc_first_track_order(len(tracks), int(scenario.sdc_track_index))

    agent = np.zeros((len(order), 6), dtype=np.float32)
    valid = np.zeros((len(order),), dtype=bool)
    names = np.empty((len(order),), dtype=object)

    for out_idx, track_idx in enumerate(order):
        track = tracks[track_idx]
        names[out_idx] = track.id
        agent[out_idx, 5] = track.object_type

        if current_t < 0 or current_t >= len(track.states):
            continue

        state = track.states[current_t]
        agent[out_idx, 0] = state.center_x
        agent[out_idx, 1] = state.center_y
        agent[out_idx, 2] = state.velocity_x
        agent[out_idx, 3] = state.velocity_y
        agent[out_idx, 4] = state.heading
        valid[out_idx] = bool(state.valid)

    return agent, valid, names


def _extract_center_lane_vectors_in_ego(
    scenario,
    ego_xy: np.ndarray,
    ego_heading: float,
    lane_range: float = DEFAULT_LANE_RANGE,
    sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> np.ndarray:
    """Extract center-lane vectors in ego coordinates.

    This is the only map component needed for the TrafficGen start-agent mask.
    It matches the old path's center lane filter (`lane.type in {1,2,3}`),
    endpoint-in-range mask, sorting by distance to ego, and truncation to 384
    center vectors.

    Returns:
        vectors: [K, 4], columns are x1, y1, x2, y2 in ego coordinates.
    """
    vectors = []

    for map_feature in scenario.map_features:
        if not map_feature.HasField("lane"):
            continue

        lane = map_feature.lane
        if int(lane.type) not in CENTER_LANE_TYPES:
            continue

        points = _sample_polyline_xy(lane.polyline, sample_num=sample_num)
        if points.shape[0] < 2:
            continue

        points = points - ego_xy[None, :]
        points = rotate(points[:, 0], points[:, 1], -ego_heading)

        point_mask = (np.abs(points[:, 0]) < lane_range) & (np.abs(points[:, 1]) < lane_range)
        segment_mask = point_mask[:-1] & point_mask[1:]
        if not np.any(segment_mask):
            continue

        segment = np.concatenate([points[:-1], points[1:]], axis=-1)[segment_mask]
        vectors.append(segment.astype(np.float32, copy=False))

    if not vectors:
        return np.zeros((0, 4), dtype=np.float32)

    vectors = np.concatenate(vectors, axis=0)

    # Match process_lane(): sort by first endpoint distance and keep first 384.
    order = np.argsort(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    vectors = vectors[order]
    return vectors[:max_center_vectors]


def get_trafficgen_select_index_fast(
    scenario,
    current_t: int = DEFAULT_CURRENT_T,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    lane_sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TrafficGen-selected agent indices and track names quickly.

    Args:
        scenario: Waymo scenario proto.
        current_t: Initial timestep used by the old code. The uploaded file used
            `data['all_agent'] = data['all_agent'][10:11]`, so the default is 10.
        lane_range: Ego-frame square range in meters.
        lane_dist_thres: Maximum distance to nearest center-lane segment midpoint.
        lane_sample_num: Downsampling stride for lane polylines.
        max_center_vectors: Number of nearest center vectors kept, matching the
            old `process_map(..., center_num=384)` behavior.

    Returns:
        selected_index:
            `np.ndarray[int64]` of indices in the SDC-first local agent order.
            These can be used directly as `trafficgen_select_index`.
        selected_names:
            `np.ndarray[object]` of track ids in selected order.
    """
    agent, valid, names = _extract_current_agents_and_names(scenario, current_t=current_t)

    if agent.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.asarray([], dtype=object)

    ego_xy = agent[0, :2].copy()
    ego_heading = float(agent[0, 4])

    rel_xy = rotate(agent[:, 0] - ego_xy[0], agent[:, 1] - ego_xy[1], -ego_heading)
    rel_vel = rotate(agent[:, 2], agent[:, 3], -ego_heading)
    rel_heading = agent[:, 4] - ego_heading

    # Old mask: agent_mask * agent_type_mask, then range mask at first frame.
    agent_mask = valid & (agent[:, 5] != 0)
    agent_mask &= (np.abs(rel_xy[:, 0]) < lane_range) & (np.abs(rel_xy[:, 1]) < lane_range)

    lane_vectors = _extract_center_lane_vectors_in_ego(
        scenario=scenario,
        ego_xy=ego_xy,
        ego_heading=ego_heading,
        lane_range=lane_range,
        sample_num=lane_sample_num,
        max_center_vectors=max_center_vectors,
    )

    if lane_vectors.shape[0] == 0:
        selected_mask = agent_mask.copy()
        selected_mask[0] = True
        selected_index = np.flatnonzero(selected_mask).astype(np.int64)
        return selected_index, names[selected_index]

    lane_midpoints = 0.5 * (lane_vectors[:, :2] + lane_vectors[:, 2:4])
    dist = np.linalg.norm(rel_xy[:, None, :] - lane_midpoints[None, :, :], axis=-1)
    nearest_vec_index = np.argmin(dist, axis=-1)
    min_dist_to_lane = dist[np.arange(dist.shape[0]), nearest_vec_index]
    min_dist_mask = min_dist_to_lane < lane_dist_thres

    selected_vec = lane_vectors[nearest_vec_index]
    lane_dir = np.arctan2(
        selected_vec[:, 3] - selected_vec[:, 1],
        selected_vec[:, 2] - selected_vec[:, 0],
    )

    speed = np.linalg.norm(rel_vel, axis=-1)
    vel_dir = np.arctan2(rel_vel[:, 1], rel_vel[:, 0])

    vel_relative_dir = cal_rel_dir(vel_dir, rel_heading)
    vel_relative_dir[speed < 0.1] = 0.0

    heading_relative_dir = cal_rel_dir(rel_heading, lane_dir)

    selected_mask = (
        agent_mask
        & min_dist_mask
        & (np.abs(vel_relative_dir) < np.pi / 6.0)
        & (np.abs(heading_relative_dir) < np.pi / 4.0)
    )

    # TrafficGen always keeps the SDC.
    selected_mask[0] = True

    selected_index = np.flatnonzero(selected_mask).astype(np.int64)
    return selected_index, names[selected_index]


def get_trafficgen_track_names_fast(
    scenario,
    current_t: int = DEFAULT_CURRENT_T,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    lane_sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> np.ndarray:
    """Return TrafficGen-selected track ids only."""
    _, selected_names = get_trafficgen_select_index_fast(
        scenario=scenario,
        current_t=current_t,
        lane_range=lane_range,
        lane_dist_thres=lane_dist_thres,
        lane_sample_num=lane_sample_num,
        max_center_vectors=max_center_vectors,
    )
    return selected_names


def _get_trafficgen_data(scenario):
    """Backward-compatible wrapper for the old API.

    Old behavior returned:
        ret["PZH_TRACK_NAMES"][ret["agent_mask"]]

    New behavior computes the same selected names directly without building
    TrafficGen's full intermediate `ret` dictionary.
    """
    return get_trafficgen_track_names_fast(scenario, current_t=DEFAULT_CURRENT_T)
