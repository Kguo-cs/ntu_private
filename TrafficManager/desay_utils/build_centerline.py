"""
boundary_to_centerlines_xyz.py

Generate 3D (xyz) lane centerlines using ONLY boundary polylines.
- Pairs boundaries into corridors (left/right relative to a reference boundary)
- Computes a local XY width profile and derives lane counts that may vary along s
- Emits one or more centerline polylines per corridor segment, with full 3D coords
- Applies orientation gating: omit points where local orientation mismatch is large
- Optionally emits only wide-enough segments

Dependencies:
    pip install numpy scipy shapely
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


# ------------------------ Utilities (3D-aware) ------------------------ #

def _to_xyz(arr: Iterable) -> np.ndarray:
    """Accept [N,3], [N,2], or [3,N] and return np.ndarray [N,3]."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("Input must be 2D array-like")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), dtype=float)])
    if a.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3] or [N,2] or [3,N]")
    return a


def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _densify_xyz(xyz: np.ndarray, max_seg_len: float = 0.5) -> np.ndarray:
    """Densify by XY segment length; linearly interpolate Z as well."""
    xyz = _to_xyz(xyz)
    if xyz.shape[0] < 2:
        return xyz
    out = [xyz[0]]
    for a, b in zip(xyz[:-1], xyz[1:]):
        seg = np.linalg.norm(b[:2] - a[:2])
        if seg <= max_seg_len or seg == 0:
            out.append(b)
            continue
        n = int(np.ceil(seg / max_seg_len))
        ts = np.linspace(0.0, 1.0, n + 1)[1:]
        pts = a + (b - a) * ts[:, None]
        out.extend(pts)
    return np.asarray(out, dtype=float)


def _resample_by_arclength_xyz(xyz: np.ndarray, n: int = 200) -> np.ndarray:
    xyz = _to_xyz(xyz)
    if xyz.shape[0] == 1:
        return np.repeat(xyz, n, axis=0)
    s = _arclength2d(xyz[:, :2])
    if s[-1] == 0:
        return np.repeat(xyz[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    x = np.interp(t, s, xyz[:, 0]); y = np.interp(t, s, xyz[:, 1]); z = np.interp(t, s, xyz[:, 2])
    return np.stack([x, y, z], axis=1)


def _resample_at_s_xyz(xyz: np.ndarray, s_query: float, s_table: np.ndarray) -> np.ndarray:
    """Interpolate xyz at XY arclength s_query with table s_table from same densified polyline."""
    x = np.interp(s_query, s_table, xyz[:, 0])
    y = np.interp(s_query, s_table, xyz[:, 1])
    z = np.interp(s_query, s_table, xyz[:, 2])
    return np.array([x, y, z], dtype=float)


def _tangents2d(xy: np.ndarray, smooth_window: int = 15, smooth_poly: int = 2) -> np.ndarray:
    v = np.zeros_like(xy, dtype=float)
    v[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] >= smooth_window and smooth_window >= 3:
        if smooth_window % 2 == 0:
            smooth_window += 1
        v = savgol_filter(v, smooth_window, smooth_poly, axis=0, mode="interp")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n


def _mean_parallelism_xy(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    A2 = _resample_by_arclength_xyz(np.column_stack([A_xy, np.zeros(len(A_xy))]), 150)[:, :2]
    B2 = _resample_by_arclength_xyz(np.column_stack([B_xy, np.zeros(len(B_xy))]), 150)[:, :2]
    tA = _tangents2d(A2); tB = _tangents2d(B2)
    return float(np.nanmean(np.abs(np.sum(tA * tB, axis=1))))


def _smooth_polyline_nd(arr: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if window < 3 or arr.shape[0] < window:
        return arr
    if window % 2 == 0:
        window += 1
    out = []
    for d in range(arr.shape[1]):
        out.append(savgol_filter(arr[:, d], window, poly, mode="interp"))
    return np.stack(out, axis=1)


def _closest_xyz_and_tangent_from_record(rec, p_xy: np.ndarray, ds: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    rec: (geom2d, id, xyz_d, s_table)
    returns: (q_xyz [3], tangent_xy [2], s_on_geom)
    """
    geom2d, _bid, xyz_d, s_tab = rec
    s = geom2d.project(Point(float(p_xy[0]), float(p_xy[1])))
    L = geom2d.length
    s0 = max(0.0, s - ds)
    s1 = min(L,    s + ds)
    q0 = geom2d.interpolate(s0); q1 = geom2d.interpolate(s1)
    t_xy = np.array([q1.x - q0.x, q1.y - q0.y], dtype=float)
    n = np.linalg.norm(t_xy) + 1e-9
    t_xy = t_xy / n
    q_xyz = _resample_at_s_xyz(xyz_d, s, s_tab)
    return q_xyz, t_xy, float(s)


# ------------------------ Data structure ------------------------ #

@dataclass
class CenterlineResult:
    boundary_a_id: int
    boundary_b_id: int
    side: str                   # 'left' or 'right'
    parallelism: float
    mean_gap_m: float
    lane_index: int
    lane_count: int
    alpha: float
    centerline: np.ndarray      # [M,3]
    group_key: int


# ------------------------ Core algorithm (from scratch, 3D, variable lanes) ------------------------ #

def build_centerlines_from_boundaries_xyz(
    boundary_dict: Dict[int, Iterable],
    *,
    densify_step: float = 0.5,
    search_buffer: float = 20.0,
    parallelism_threshold: float = 0.6,
    side_confidence_eps: float = 0.05,
    desired_lane_width_m: float = 3.6,
    split_tolerance_m: float = 0.6,
    max_lanes: int = 6,
    min_pair_mean_dist: float = 2.0,
    max_pair_mean_dist: Optional[float] = 25.0,
    smoothing_window: int = 21,
    smoothing_poly: int = 3,
    segment_wide_only: bool = False,
    small_gap_skip_m: float = 1.2,
    min_segment_len_m: float = 8.0,
    max_orient_diff_deg: float = 35.0,
    orient_ds: float = 0.5,
    segment_on_orientation: bool = True,
    merge_short_segments_m: float = 12.0,
) -> List[CenterlineResult]:
    """
    Build lane centerlines (3D) using only boundary polylines.

    Returns list of CenterlineResult; centerline is [M,3].
    """

    # Build densified records: (geom2d, id, xyz_densified, s_table)
    records = []
    for bid, xyz in boundary_dict.items():
        xyz_d = _densify_xyz(xyz, max_seg_len=densify_step)
        if xyz_d.shape[0] < 2:
            continue
        geom2d = LineString(xyz_d[:, :2])
        s_tab = _arclength2d(xyz_d[:, :2])
        records.append((geom2d, int(bid), xyz_d, s_tab))
    if not records:
        return []

    geoms = [r[0] for r in records]
    tree = STRtree(geoms)

    out: List[CenterlineResult] = []
    seen_pairs = set()
    cos_thresh = float(np.cos(np.deg2rad(max_orient_diff_deg)))

    # Decide lane count at a local width
    def _decide_count(rep_w: float) -> int:
        if rep_w <= desired_lane_width_m - split_tolerance_m:
            return 1
        n = int(np.floor((rep_w + split_tolerance_m) / max(1e-6, desired_lane_width_m)))
        return int(np.clip(n, 1, max_lanes))

    group_key = 0

    # Loop over each boundary as reference A
    for a_id, a_xyz0 in boundary_dict.items():
        a_xyz = _densify_xyz(a_xyz0, max_seg_len=densify_step)
        if a_xyz.shape[0] < 2:
            continue
        a_geom2d = LineString(a_xyz[:, :2])
        a_rs_xyz = _resample_by_arclength_xyz(a_xyz, 200)    # [200,3]
        a_rs_xy  = a_rs_xyz[:, :2]
        a_t      = _tangents2d(a_rs_xy)

        # Candidate nearby boundaries
        cand_idx = tree.query(a_geom2d.buffer(search_buffer))
        if not isinstance(cand_idx, (list, tuple, np.ndarray)):
            cand_idx = [cand_idx]

        best = {'left': None, 'right': None}  # (score, rec_idx, par, mean_dist)
        nprobe = min(60, len(a_rs_xy))
        probe_idx = np.linspace(0, len(a_rs_xy)-1, nprobe).astype(int)

        # pick best partner per side using parallelism + distance + side sign
        for idx in cand_idx:
            geom2d_b, b_id, xyz_d_b, s_tab_b = records[idx]
            if b_id == a_id:
                continue
            par = _mean_parallelism_xy(a_xyz[:, :2], xyz_d_b[:, :2])
            if par < parallelism_threshold:
                continue
            dists = []
            signs = []
            for i in probe_idx:
                p_xy = a_rs_xy[i]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(records[idx], p_xy, ds=orient_ds)
                d_xy = q_xyz[:2] - p_xy
                tA = a_t[i]
                cross_z = tA[0]*d_xy[1] - tA[1]*d_xy[0]
                signs.append(np.sign(cross_z))
                dists.append(np.linalg.norm(d_xy))
            mean_sign = float(np.mean(signs))
            mean_dist = float(np.mean(dists))
            if mean_dist < min_pair_mean_dist:
                continue
            if (max_pair_mean_dist is not None) and (mean_dist > max_pair_mean_dist):
                continue
            if abs(mean_sign) <= side_confidence_eps:
                continue
            side = 'left' if mean_sign > 0 else 'right'
            score = mean_dist / max(1e-6, par)
            cur = best[side]
            if (cur is None) or (score < cur[0]):
                best[side] = (score, idx, par, mean_dist)

        # For each side, compute per-sample closest, orientation check, width, lane_count_series, then segment
        for side in ('left','right'):
            if best[side] is None:
                continue
            _, idx, par, mean_dist = best[side]
            geom2d_b, b_id, xyz_d_b, s_tab_b = records[idx]

            pair_key = frozenset({int(a_id), int(b_id)})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # compute for each resampled sample along A: partner xyz, partner tangent, sB
            nS = len(a_rs_xy)
            b_pts_xyz = np.empty((nS, 3), dtype=float)
            b_tan_xy  = np.empty((nS, 2), dtype=float)
            b_s       = np.empty(nS, dtype=float)
            for i, p_xy in enumerate(a_rs_xy):
                q_xyz, tB, sB = _closest_xyz_and_tangent_from_record(records[idx], p_xy, ds=orient_ds)
                b_pts_xyz[i] = q_xyz
                b_tan_xy[i]  = tB
                b_s[i]       = sB

            band_vec_xyz = b_pts_xyz - a_rs_xyz
            width_series = np.linalg.norm(band_vec_xyz[:, :2], axis=1)
            rep_width = float(np.median(width_series))

            # orientation cos series
            cos_series = np.abs(np.sum(a_t * b_tan_xy, axis=1))
            orient_mask = cos_series >= cos_thresh

            # width mask
            width_mask = width_series >= small_gap_skip_m

            # decide keep_mask combining orientation and width rules
            if segment_wide_only:
                keep_mask = width_mask & orient_mask
            else:
                if rep_width < small_gap_skip_m:
                    continue
                keep_mask = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)

            # compute local lane count series per sample
            lane_count_series = np.array([_decide_count(w) for w in width_series], dtype=int)

            # optionally merge lane counts spatially (avoid flicker)
            # create segments of constant lane count and merge short ones
            segments: List[Tuple[int,int,int]] = []
            cur_val = lane_count_series[0]; start = 0
            for i in range(1, nS):
                if lane_count_series[i] != cur_val:
                    segments.append((start, i, int(cur_val)))
                    start = i; cur_val = lane_count_series[i]
            segments.append((start, nS, int(cur_val)))

            # Filter segments by keep_mask runs (orientation/width), and merge tiny segments
            s_table = _arclength2d(a_rs_xy)
            filtered_segments: List[Tuple[int,int,int]] = []
            for (i0, i1, cnt) in segments:
                # get mask inside this segment
                seg_mask = keep_mask[i0:i1]
                if not np.any(seg_mask):
                    continue
                runs = []
                # contiguous true runs inside [i0,i1)
                idxs = np.flatnonzero(np.diff(np.concatenate(([False], seg_mask, [False]))))
                runs = list(zip(idxs[0::2], idxs[1::2]))  # relative indices
                # convert to absolute indices, remove tiny segments
                for (ra, rb) in runs:
                    a_abs = i0 + ra
                    b_abs = i0 + rb
                    seg_len = s_table[b_abs - 1] - s_table[a_abs]
                    if seg_len >= min_segment_len_m:
                        filtered_segments.append((a_abs, b_abs, cnt))
            # Merge very short adjacent segments (based on merge_short_segments_m)
            if filtered_segments:
                merged = []
                prev_a, prev_b, prev_cnt = filtered_segments[0]
                for a_abs, b_abs, cnt in filtered_segments[1:]:
                    prev_len = s_table[prev_b - 1] - s_table[prev_a]
                    if prev_len < merge_short_segments_m:
                        # merge into next for stability
                        prev_b = b_abs
                        prev_cnt = cnt
                    else:
                        merged.append((prev_a, prev_b, prev_cnt))
                        prev_a, prev_b, prev_cnt = a_abs, b_abs, cnt
                merged.append((prev_a, prev_b, prev_cnt))
                filtered_segments = merged

            if not filtered_segments:
                continue


            # Emit centerlines per segment and per lane index
            for (a_idx, b_idx, n_lanes) in filtered_segments:
                # recompute rep width for segment to decide exact n_lanes (in case small variation)
                seg_rep_width = float(np.median(width_series[a_idx:b_idx]))
                n_lanes = _decide_count(seg_rep_width)
                for k in range(n_lanes):
                    alpha = (k + 0.5) / n_lanes
                    seg_center_xyz = a_rs_xyz[a_idx:b_idx] + alpha * band_vec_xyz[a_idx:b_idx]
                    # smooth (per channel)
                    w = min(smoothing_window, (len(seg_center_xyz) // 2) * 2 - 1)
                    if w >= 3:
                        seg_center_xyz = _smooth_polyline_nd(seg_center_xyz, window=w, poly=smoothing_poly)
                    out.append(CenterlineResult(
                        boundary_a_id=int(a_id),
                        boundary_b_id=int(b_id),
                        side=side,
                        parallelism=float(par),
                        mean_gap_m=float(np.median(width_series[a_idx:b_idx])),
                        lane_index=k,
                        lane_count=n_lanes,
                        alpha=float(alpha),
                        centerline=seg_center_xyz,
                        group_key=group_key
                    ))

                group_key=group_key+1

    return out


# ------------------------ Parsing helper (your schema) ------------------------ #

def build_boundary_dict_xyz(
    map_features: Iterable[Dict[str, Any]],
    remove_mapid: Optional[Iterable[int]] = None
) -> Dict[int, np.ndarray]:
    """
    Convert map_features into {boundary_id: np.ndarray [N,3]}.
    Accepts feature['xyz'] in forms: [3,N], [N,3], or [N,2] (z=0).
    Only includes features with class == 'boundary'.
    """
    out: Dict[int, np.ndarray] = {}
    rm = set(remove_mapid or [])
    for mf in map_features:
        fid = int(mf.get('global_id'))
        if fid in rm:
            continue
        if mf.get('class') != 'boundary':
            continue
        xyz = _to_xyz(mf['xyz'])
        out[fid] = xyz
    return out


# # ------------------------ Example usage ------------------------ #
#
# if __name__ == "__main__":
#     # Example (user must supply map_features):
#     # map_features = [...]
#     # boundary_dict = build_boundary_dict_xyz(map_features)
#     # centerlines = build_centerlines_from_boundaries_xyz(boundary_dict)
#     # print(f"Generated {len(centerlines)} centerlines.")
#     pass
