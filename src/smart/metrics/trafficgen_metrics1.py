"""Exact single-frame TrafficGen agent filtering with explicit array layouts.

The public API separates three concerns:
1. extract one Waymo frame;
2. extract center-lane vectors in the SDC frame;
3. reproduce TrafficGen ``process_agent`` + ``get_vec_rep`` and return
   original-order integer indices.

``get_select_index`` preserves TrafficGen's midpoint lane association, raster
ordering, optional random permutation, angular thresholds, forced SDC, and
32-agent truncation.
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


def _trafficgen_initial_order(
    relative_positions: np.ndarray,
    initial_mask: np.ndarray,
    *,
    raster_resolution: float,
    randomize_agents: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Reproduce TrafficGen ``process_agent`` ordering for one scene.

    Input agents must already be SDC-first.  TrafficGen first raster-sorts all
    non-SDC agents by descending raster-y and ascending raster-x.  In the
    dataset path it calls ``process_agent(..., sort_agent=False)``, which then
    randomly permutes the valid non-SDC prefix.

    Returns:
        Indices into the SDC-first input array, in TrafficGen order.
    """
    num_agents = len(relative_positions)
    if num_agents <= 1:
        return np.arange(num_agents, dtype=np.int64)

    xy = relative_positions[1:].copy()
    mask = initial_mask[1:]

    # Exact sentinels used by TrafficGen to move invalid agents to the end.
    xy[~mask, 0] = 10e8
    xy[~mask, 1] = -10e8

    raster_xy = np.floor(xy / raster_resolution)
    local_index = np.arange(1, num_agents, dtype=np.int64)

    # Exact two-stage sorting used in process_agent().
    y_order = np.argsort(-raster_xy[:, 1])
    raster_xy = raster_xy[y_order]
    local_index = local_index[y_order]

    for raster_y in np.unique(raster_xy[:, 1])[::-1]:
        row = np.flatnonzero(raster_xy[:, 1] == raster_y)
        x_order = np.argsort(raster_xy[row, 0])
        raster_xy[row] = raster_xy[row][x_order]
        local_index[row] = local_index[row][x_order]

    order = np.concatenate((np.asarray([0], dtype=np.int64), local_index))

    # process_agent(..., sort_agent=False) randomly permutes only the valid
    # non-SDC prefix.  Keep the exact global-numpy behavior when rng is None.
    if randomize_agents:
        sorted_mask = initial_mask[order].copy()
        sorted_mask[0] = True
        valid_num = int(sorted_mask.sum())

        if valid_num > 1:
            if rng is None:
                permutation = np.random.permutation(np.arange(1, valid_num)) - 1
            else:
                permutation = rng.permutation(np.arange(1, valid_num)) - 1
            order[1:valid_num] = order[1:valid_num][permutation]

    return order


def get_select_index(
    lane_vectors: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    valid: np.ndarray,
    object_types: np.ndarray,
    *,
    ego_index: int = 0,
    vehicle_type: int = 1,
    lane_range: float = DEFAULT_LANE_RANGE,
    lane_dist_thres: float = DEFAULT_LANE_DIST_THRES,
    max_agent_num: int = 32,
    raster_resolution: float = 0.25,
    min_speed: float = 0.1,
    max_velocity_heading_error: float = np.pi / 6.0,
    max_lane_heading_error: float = np.pi / 4.0,
    randomize_agents: bool = True,
    rng: np.random.Generator | None = None,
    center_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return agent indices using the exact TrafficGen filter semantics.

    This combines the single-frame behavior of TrafficGen's
    ``process_agent`` and ``get_vec_rep``:

    1. put the SDC at local index 0;
    2. transform agents to the SDC coordinate frame;
    3. keep valid vehicles inside the 50 m square;
    4. apply TrafficGen's raster ordering and optional random permutation;
    5. associate each agent with the nearest lane-vector *midpoint*;
    6. require lane distance < 5 m, velocity-heading error < 30 degrees,
       and heading-lane error < 45 degrees;
    7. always retain the SDC and keep at most 32 agents.

    ``vehicle_type`` adapts the original Waymo convention (vehicle == 1) to
    this codebase's model convention (vehicle == 0) without changing the
    filtering logic.

    The returned indices refer to the original input order, but their order is
    exactly the order produced by TrafficGen after sorting/permutation/filtering.
    Lane vectors must already be in the SDC coordinate frame.
    """
    lane_vectors = np.asarray(lane_vectors, dtype=np.float32).reshape(-1, 4)
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    headings = np.asarray(headings, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    object_types = np.asarray(object_types).reshape(-1)

    num_agents = len(positions)
    expected_2d = (num_agents, 2)
    expected_1d = (num_agents,)

    if positions.shape != expected_2d:
        raise ValueError(f"positions must have shape {expected_2d}, got {positions.shape}")
    if velocities.shape != expected_2d:
        raise ValueError(f"velocities must have shape {expected_2d}, got {velocities.shape}")
    if headings.shape != expected_1d:
        raise ValueError(f"headings must have shape {expected_1d}, got {headings.shape}")
    if valid.shape != expected_1d:
        raise ValueError(f"valid must have shape {expected_1d}, got {valid.shape}")
    if object_types.shape != expected_1d:
        raise ValueError(
            f"object_types must have shape {expected_1d}, got {object_types.shape}"
        )
    if num_agents == 0:
        return np.empty(0, dtype=np.int64)
    if len(lane_vectors) == 0:
        # Original TrafficGen reaches argmin() on an empty axis here.  Raise a
        # clearer error while preserving that the filter is undefined without
        # center-lane vectors.
        raise ValueError("TrafficGen filtering requires at least one lane vector")
    if max_agent_num <= 0:
        raise ValueError(f"max_agent_num must be positive, got {max_agent_num}")

    ego_index = int(ego_index) % num_agents

    # TrafficGen's internal representation is always SDC-first.
    original_index = np.arange(num_agents, dtype=np.int64)
    sdc_first = np.concatenate(
        (
            np.asarray([ego_index], dtype=np.int64),
            original_index[original_index != ego_index],
        )
    )

    positions = positions[sdc_first]
    velocities = velocities[sdc_first]
    headings = headings[sdc_first]
    valid = valid[sdc_first]
    object_types = object_types[sdc_first]
    original_index = original_index[sdc_first]

    ego_xy = positions[0].copy()
    ego_heading = float(headings[0])

    relative_positions = rotate_xy(positions - ego_xy, -ego_heading)
    relative_velocities = rotate_xy(velocities, -ego_heading)
    relative_headings = headings - ego_heading

    # Exact process_agent mask: valid * vehicle * square-range.
    initial_mask = (
        valid
        & (object_types == vehicle_type)
        & (np.abs(relative_positions[:, 0]) < lane_range)
        & (np.abs(relative_positions[:, 1]) < lane_range)
    )

    trafficgen_order = _trafficgen_initial_order(
        relative_positions=relative_positions,
        initial_mask=initial_mask,
        raster_resolution=raster_resolution,
        randomize_agents=randomize_agents,
        rng=rng,
    )

    relative_positions = relative_positions[trafficgen_order]
    relative_velocities = relative_velocities[trafficgen_order]
    relative_headings = relative_headings[trafficgen_order]
    initial_mask = initial_mask[trafficgen_order]
    original_index = original_index[trafficgen_order]

    # TrafficGen always marks the SDC slot valid after process_agent.
    initial_mask = initial_mask.copy()
    initial_mask[0] = True

    # Exact get_vec_rep association: distance to lane-vector midpoint, not
    # geometric point-to-segment distance.
    lane_midpoints = 0.5 * (lane_vectors[:, :2] + lane_vectors[:, 2:4])
    distance = np.linalg.norm(
        relative_positions[:, None, :] - lane_midpoints[None, :, :],
        axis=-1,
    )

    if center_mask is not None:
        center_mask = np.asarray(center_mask, dtype=bool).reshape(-1)
        if center_mask.shape != (len(lane_vectors),):
            raise ValueError(
                "center_mask must have shape "
                f"({len(lane_vectors)},), got {center_mask.shape}"
            )
        distance[:, ~center_mask] = 10e5

    nearest_lane_index = np.argmin(distance, axis=-1)
    min_dist_to_lane = np.min(distance, axis=-1)
    min_dist_mask = min_dist_to_lane < lane_dist_thres

    selected_lane = lane_vectors[nearest_lane_index]
    lane_heading = np.arctan2(
        selected_lane[:, 3] - selected_lane[:, 1],
        selected_lane[:, 2] - selected_lane[:, 0],
    )

    speed = np.linalg.norm(relative_velocities, axis=-1)
    velocity_heading = np.arctan2(
        relative_velocities[:, 1], relative_velocities[:, 0]
    )

    velocity_heading_error = cal_rel_dir(velocity_heading, relative_headings)
    velocity_heading_error[speed < min_speed] = 0.0
    lane_heading_error = cal_rel_dir(relative_headings, lane_heading)

    selected_mask = (
        min_dist_mask
        & initial_mask
        & (np.abs(velocity_heading_error) < max_velocity_heading_error)
        & (np.abs(lane_heading_error) < max_lane_heading_error)
    )

    # Exact TrafficGen behavior: always keep SDC, then truncate in current order.
    selected_mask[0] = True
    selected_position = np.flatnonzero(selected_mask)[:max_agent_num]

    return original_index[selected_position].astype(np.int64, copy=False)

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
        vehicle_type=1,  # Waymo proto: TYPE_VEHICLE == 1
        lane_range=lane_range,
        lane_dist_thres=lane_dist_thres,
        max_agent_num=32,
        randomize_agents=True,  # matches process_agent(..., sort_agent=False)
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
