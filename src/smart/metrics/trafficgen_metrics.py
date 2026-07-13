"""Fast and explicit TrafficGen-style agent selection.

The public API separates three concerns:
1. extract one Waymo frame;
2. extract center-lane vectors in the SDC frame;
3. select valid agents and return integer indices.

`get_select_index` always returns ``np.int64`` indices, never a boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_CURRENT_T = 10
DEFAULT_LANE_RANGE = 50.0
DEFAULT_LANE_DIST_THRES = 5.0
DEFAULT_LANE_SAMPLE_NUM = 10
DEFAULT_MAX_CENTER_VECTORS = 384

CENTER_LANE_TYPES = {1, 2, 3}


@dataclass(frozen=True)
class TrafficGenSelection:
    """Selection result in SDC-first agent order."""

    lane_vectors: np.ndarray
    selected_index: np.ndarray
    selected_track_ids: np.ndarray


def rotate_xy(xy: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D vectors counter-clockwise by ``angle`` radians."""
    xy = np.asarray(xy, dtype=np.float32)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack(
        (cos_a * x - sin_a * y, sin_a * x + cos_a * y),
        axis=-1,
    )


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# Backward-compatible aliases used by older code.
def rotate(x, y, angle):
    return rotate_xy(np.stack((x, y), axis=-1), angle)


def cal_rel_dir(dir1, dir2):
    return wrap_angle(np.asarray(dir1) - np.asarray(dir2))


def _sample_polyline_xy(polyline: Iterable, stride: int) -> np.ndarray:
    """Read proto points and downsample them with a fixed stride."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    points = np.asarray([(point.x, point.y) for point in polyline], dtype=np.float32)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) < stride:
        return points
    return points[::stride]


def _sdc_first_track_order(num_tracks: int, sdc_index: int) -> np.ndarray:
    """Return scenario-track indices with the SDC at local index 0."""
    if not 0 <= sdc_index < num_tracks:
        raise IndexError(
            f"Invalid sdc_track_index={sdc_index} for {num_tracks} tracks"
        )

    other = np.arange(num_tracks, dtype=np.int64)
    other = other[other != sdc_index]
    return np.concatenate((np.asarray([sdc_index], dtype=np.int64), other))


def _extract_current_agents_and_names(
    scenario,
    current_t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one frame in SDC-first order.

    Returns:
        positions: ``[N, 2]``
        velocities: ``[N, 2]``
        headings: ``[N]``
        object_types: ``[N]``; Waymo type 0 means unset/invalid
        valid: ``[N]``
        track_ids: ``[N]``
    """
    order = _sdc_first_track_order(
        num_tracks=len(scenario.tracks),
        sdc_index=int(scenario.sdc_track_index),
    )

    num_agents = len(order)
    positions = np.zeros((num_agents, 2), dtype=np.float32)
    velocities = np.zeros((num_agents, 2), dtype=np.float32)
    headings = np.zeros(num_agents, dtype=np.float32)
    object_types = np.zeros(num_agents, dtype=np.int32)
    valid = np.zeros(num_agents, dtype=bool)
    track_ids = np.empty(num_agents, dtype=object)

    for local_index, track_index in enumerate(order):
        track = scenario.tracks[int(track_index)]
        track_ids[local_index] = track.id
        object_types[local_index] = int(track.object_type)

        if not 0 <= current_t < len(track.states):
            continue

        state = track.states[current_t]
        positions[local_index] = (state.center_x, state.center_y)
        velocities[local_index] = (state.velocity_x, state.velocity_y)
        headings[local_index] = state.heading
        valid[local_index] = bool(state.valid)

    return positions, velocities, headings, object_types, valid, track_ids


def _extract_center_lane_vectors_in_ego(
    scenario,
    ego_xy: np.ndarray,
    ego_heading: float,
    lane_range: float = DEFAULT_LANE_RANGE,
    sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> np.ndarray:
    """Extract lane segments as ``[x1, y1, x2, y2]`` in the ego frame."""
    all_segments: list[np.ndarray] = []

    for feature in scenario.map_features:
        if not feature.HasField("lane"):
            continue

        lane = feature.lane
        if int(lane.type) not in CENTER_LANE_TYPES:
            continue

        points = _sample_polyline_xy(lane.polyline, stride=sample_num)
        if len(points) < 2:
            continue

        ego_points = rotate_xy(points - np.asarray(ego_xy)[None, :], -ego_heading)
        point_in_range = np.all(np.abs(ego_points) < lane_range, axis=-1)
        segment_in_range = point_in_range[:-1] & point_in_range[1:]
        if not np.any(segment_in_range):
            continue

        segments = np.concatenate((ego_points[:-1], ego_points[1:]), axis=-1)
        all_segments.append(segments[segment_in_range])

    if not all_segments:
        return np.empty((0, 4), dtype=np.float32)

    lane_vectors = np.concatenate(all_segments, axis=0).astype(np.float32, copy=False)

    # Preserve TrafficGen's ordering: nearest first endpoint first, then truncate.
    distance_sq = np.sum(lane_vectors[:, :2] ** 2, axis=-1)
    nearest_first = np.argsort(distance_sq)
    return lane_vectors[nearest_first[:max_center_vectors]]


def get_select_index(
    lane_vectors: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    valid: np.ndarray | None = None,
    object_types: np.ndarray | None = None,
    *,
    ego_index: int = 0,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    min_speed: float = 0.1,
    max_velocity_heading_error: float = np.pi / 6.0,
    max_lane_heading_error: float = np.pi / 4.0,
) -> np.ndarray:
    """Return selected agent indices.

    All lane vectors must already be expressed in the ego coordinate frame.
    Agent positions, velocities, and headings are provided in the world frame.

    ``object_types`` is optional because model-side type 0 may mean "vehicle",
    while Waymo proto type 0 means "unset".  When supplied, type 0 is removed.
    The ego is always retained.
    """
    lane_vectors = np.asarray(lane_vectors, dtype=np.float32).reshape(-1, 4)
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    headings = np.asarray(headings, dtype=np.float32).reshape(-1)

    num_agents = len(positions)
    if positions.shape != (num_agents, 2):
        raise ValueError(f"positions must have shape [N, 2], got {positions.shape}")
    if velocities.shape != (num_agents, 2):
        raise ValueError(f"velocities must have shape [N, 2], got {velocities.shape}")
    if headings.shape != (num_agents,):
        raise ValueError(f"headings must have shape [N], got {headings.shape}")
    if num_agents == 0:
        return np.empty(0, dtype=np.int64)

    ego_index = int(ego_index) % num_agents

    if valid is None:
        candidate = np.ones(num_agents, dtype=bool)
    else:
        candidate = np.asarray(valid, dtype=bool).reshape(-1).copy()
        if candidate.shape != (num_agents,):
            raise ValueError(f"valid must have shape [N], got {candidate.shape}")

    if object_types is not None:
        # object_types = np.asarray(object_types).reshape(-1)
        # if object_types.shape != (num_agents,):
        #     raise ValueError(
        #         f"object_types must have shape [N], got {object_types.shape}"
        #     )
        candidate &= object_types == 0

    ego_xy = positions[ego_index]
    ego_heading = float(headings[ego_index])

    relative_xy = rotate_xy(positions - ego_xy, -ego_heading)
    relative_velocity = rotate_xy(velocities, -ego_heading)
    relative_heading = wrap_angle(headings - ego_heading)

    candidate &= np.all(np.abs(relative_xy) < lane_range, axis=-1)

    if len(lane_vectors) == 0:
        candidate[ego_index] = True
        return np.flatnonzero(candidate).astype(np.int64, copy=False)

    # Point-to-segment distance.  Using only segment midpoints incorrectly
    # rejects agents near the ends of long lane vectors.
    lane_start = lane_vectors[:, :2]
    lane_delta = lane_vectors[:, 2:] - lane_start
    lane_length_sq = np.sum(lane_delta**2, axis=-1)

    agent_to_start = relative_xy[:, None, :] - lane_start[None, :, :]
    projection = np.sum(agent_to_start * lane_delta[None, :, :], axis=-1)
    projection /= np.maximum(lane_length_sq[None, :], 1e-8)
    projection = np.clip(projection, 0.0, 1.0)

    closest_point = (
        lane_start[None, :, :]
        + projection[..., None] * lane_delta[None, :, :]
    )
    distance = np.linalg.norm(relative_xy[:, None, :] - closest_point, axis=-1)
    nearest_lane_index = np.argmin(distance, axis=1)
    close_to_lane = distance[np.arange(num_agents), nearest_lane_index] < lane_dist_thres

    nearest_lane = lane_vectors[nearest_lane_index]
    lane_heading = np.arctan2(
        nearest_lane[:, 3] - nearest_lane[:, 1],
        nearest_lane[:, 2] - nearest_lane[:, 0],
    )

    speed = np.linalg.norm(relative_velocity, axis=-1)
    velocity_heading = np.arctan2(relative_velocity[:, 1], relative_velocity[:, 0])
    velocity_heading_error = wrap_angle(velocity_heading - relative_heading)
    velocity_heading_error[speed < min_speed] = 0.0

    lane_heading_error = wrap_angle(relative_heading - lane_heading)

    selected = (
        candidate
        & close_to_lane
        & (np.abs(velocity_heading_error) < max_velocity_heading_error)
        & (np.abs(lane_heading_error) < max_lane_heading_error)
    )
    selected[ego_index] = True

    return np.flatnonzero(selected).astype(np.int64, copy=False)


def get_trafficgen_selection(
    scenario,
    current_t: int = DEFAULT_CURRENT_T,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    lane_sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> TrafficGenSelection:
    """Compute lane vectors, selected indices, and track IDs from a scenario."""
    (
        positions,
        velocities,
        headings,
        object_types,
        valid,
        track_ids,
    ) = _extract_current_agents_and_names(scenario, current_t=current_t)

    # `_extract_current_agents_and_names` explicitly puts SDC at index 0.
    ego_index = 0
    lane_vectors = _extract_center_lane_vectors_in_ego(
        scenario=scenario,
        ego_xy=positions[ego_index],
        ego_heading=float(headings[ego_index]),
        lane_range=lane_range,
        sample_num=lane_sample_num,
        max_center_vectors=max_center_vectors,
    )

    selected_index = get_select_index(
        lane_vectors=lane_vectors,
        positions=positions,
        velocities=velocities,
        headings=headings,
        valid=valid,
        object_types=object_types,
        ego_index=ego_index,
        lane_range=lane_range,
        lane_dist_thres=lane_dist_thres,
    )

    return TrafficGenSelection(
        lane_vectors=lane_vectors,
        selected_index=selected_index,
        selected_track_ids=track_ids[selected_index],
    )


def get_trafficgen_select_index_fast(
    scenario,
    current_t: int = DEFAULT_CURRENT_T,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    lane_sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility API: return integer indices and selected track IDs."""
    result = get_trafficgen_selection(
        scenario=scenario,
        current_t=current_t,
        lane_range=lane_range,
        lane_dist_thres=lane_dist_thres,
        lane_sample_num=lane_sample_num,
        max_center_vectors=max_center_vectors,
    )
    return result.selected_index, result.selected_track_ids


def get_trafficgen_track_names_fast(
    scenario,
    current_t: int = DEFAULT_CURRENT_T,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    lane_sample_num: int = DEFAULT_LANE_SAMPLE_NUM,
    max_center_vectors: int = DEFAULT_MAX_CENTER_VECTORS,
) -> np.ndarray:
    """Return selected track IDs only."""
    return get_trafficgen_selection(
        scenario=scenario,
        current_t=current_t,
        lane_range=lane_range,
        lane_dist_thres=lane_dist_thres,
        lane_sample_num=lane_sample_num,
        max_center_vectors=max_center_vectors,
    ).selected_track_ids


def _get_trafficgen_data(scenario, current_t: int = DEFAULT_CURRENT_T) -> np.ndarray:
    """Backward-compatible replacement for the old expensive extraction path."""
    return get_trafficgen_track_names_fast(scenario, current_t=current_t)