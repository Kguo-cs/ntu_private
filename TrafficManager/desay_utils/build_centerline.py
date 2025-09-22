"""
boundary_to_centerlines_xyz_fixed.py

Full pipeline (3D) for generating lane centerlines from boundaries only.
Includes fixes so savgol_filter is always called with valid window lengths.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Any, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


# ------------------------ Data structures ------------------------ #

@dataclass
class CenterlineResult:
    boundary_a_id: int
    boundary_b_id: int
    side: str
    parallelism: float
    mean_gap_m: float
    lane_index: int
    lane_count: int
    alpha: float
    centerline: np.ndarray  # [M,3]


# ------------------------ Utility helpers ------------------------ #

def _to_xyz(arr: Iterable) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("Input must be 2D array-like")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), dtype=float)])
    if a.shape[1] != 3:
        raise ValueError("xyz must be shape [N,3] or [N,2] or [3,N]")
    return a

def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _safe_savgol(arr: np.ndarray, window: int, poly: int):
    """
    Safely apply savgol_filter to 1D arr. If window is invalid relative to len(arr),
    adjust to the largest odd window <= len(arr) (and >=3). If still invalid, return arr.
    """
    n = len(arr)
    if n < 3:
        return arr
    w = int(min(window, n))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr
    if poly >= w:
        poly = max(1, w-1)
    return savgol_filter(arr, w, poly, mode="interp")

def _safe_savgol_nd(mat: np.ndarray, window: int, poly: int):
    """Apply safe savgol per column on mat (shape [N, D])."""
    mat = np.asarray(mat)
    if mat.ndim == 1:
        return _safe_savgol(mat, window, poly)
    out = np.zeros_like(mat)
    for d in range(mat.shape[1]):
        out[:, d] = _safe_savgol(mat[:, d], window, poly)
    return out

def _densify_xyz(xyz: np.ndarray, max_seg_len: float = 0.5) -> np.ndarray:
    xyz = _to_xyz(xyz)
    if xyz.shape[0] < 2:
        return xyz
    out = [xyz[0]]
    for a, b in zip(xyz[:-1], xyz[1:]):
        seg = np.linalg.norm(b[:2] - a[:2])
        if seg <= max_seg_len or seg == 0:
            out.append(b); continue
        n = int(np.ceil(seg / max_seg_len))
        ts = np.linspace(0.0, 1.0, n+1)[1:]
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

def _tangents2d(xy: np.ndarray, smooth_window: int = 15, smooth_poly: int = 2) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    # apply safe savgol per column to v
    v = _safe_savgol_nd(v, smooth_window, smooth_poly)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _mean_parallelism_xy(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    A2 = _resample_by_arclength_xyz(np.column_stack([A_xy, np.zeros(len(A_xy))]), 150)[:, :2]
    B2 = _resample_by_arclength_xyz(np.column_stack([B_xy, np.zeros(len(B_xy))]), 150)[:, :2]
    tA = _tangents2d(A2); tB = _tangents2d(B2)
    return float(np.nanmean(np.abs(np.sum(tA * tB, axis=1))))

def _closest_xyz_and_tangent_from_record(rec, p_xy: np.ndarray, ds: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    geom2d, _bid, xyz_d, s_tab = rec
    s = geom2d.project(Point(float(p_xy[0]), float(p_xy[1])))
    L = geom2d.length
    s0 = max(0.0, s - ds)
    s1 = min(L, s + ds)
    q0 = geom2d.interpolate(s0); q1 = geom2d.interpolate(s1)
    t_xy = np.array([q1.x - q0.x, q1.y - q0.y], dtype=float)
    nrm = np.linalg.norm(t_xy) + 1e-9
    t_xy = t_xy / nrm
    q_xyz = np.array([np.interp(s, s_tab, xyz_d[:, 0]),
                      np.interp(s, s_tab, xyz_d[:, 1]),
                      np.interp(s, s_tab, xyz_d[:, 2])], dtype=float)
    return q_xyz, t_xy, float(s)

def _smooth_polyline_nd(arr: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] < 3:
        return arr
    return _safe_savgol_nd(arr, window, poly)


def straighten_centerline(
    center_xyz: np.ndarray,
    width_series: Optional[np.ndarray] = None,
    *,
    base_smooth: float = 0.0,
    strength: float = 1.0,
    blend_factor_max: float = 0.9,
    min_points: int = 6,
    spline_k: int = 3
) -> np.ndarray:
    center_xyz = np.asarray(center_xyz, dtype=float)
    if center_xyz.shape[0] < min_points:
        return center_xyz.copy()
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center_xyz[:, :2], axis=0), axis=1))]
    if width_series is not None:
        w = np.asarray(width_series, dtype=float)
        if len(w) != len(center_xyz):
            w = np.interp(s, np.linspace(0, s[-1], len(w)), w)
        rel_var = np.std(w) / (np.mean(w) + 1e-9)
    else:
        rel_var = 0.0
    s_spline = float(base_smooth + strength * rel_var * len(center_xyz))
    k = min(spline_k, max(1, len(center_xyz)-1))
    splx = UnivariateSpline(s, center_xyz[:, 0], s=s_spline, k=k)
    sply = UnivariateSpline(s, center_xyz[:, 1], s=s_spline, k=k)
    splz = UnivariateSpline(s, center_xyz[:, 2], s=s_spline, k=k)
    spline_xyz = np.stack([splx(s), sply(s), splz(s)], axis=1)
    baseline = np.outer(1.0 - (s/s[-1]), center_xyz[0]) + np.outer(s/s[-1], center_xyz[-1])
    c = 6.0
    blend = np.clip((1.0 - np.exp(-c * rel_var)) * blend_factor_max, 0.0, blend_factor_max)
    return (1.0 - blend) * spline_xyz + blend * baseline


# ------------------------ Parsing helpers ------------------------ #

def build_boundary_dict_xyz(map_features: Iterable[Dict[str, Any]], remove_mapid: Optional[Iterable[int]] = None) -> Dict[int, np.ndarray]:
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


# ------------------------ Core pipeline ------------------------ #

def build_centerlines_from_boundaries_xyz(
    boundary_dict: Dict[int, np.ndarray],
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

    def _decide_count(rep_w: float) -> int:
        if rep_w <= desired_lane_width_m - split_tolerance_m:
            return 1
        n = int(np.floor((rep_w + split_tolerance_m) / max(1e-6, desired_lane_width_m)))
        return int(np.clip(n, 1, max_lanes))

    for a_id, a_xyz0 in boundary_dict.items():
        a_xyz = _densify_xyz(a_xyz0, max_seg_len=densify_step)
        if a_xyz.shape[0] < 2:
            continue
        a_geom2d = LineString(a_xyz[:, :2])
        a_rs_xyz = _resample_by_arclength_xyz(a_xyz, 200)
        a_rs_xy = a_rs_xyz[:, :2]
        a_t = _tangents2d(a_rs_xy)

        cand_idx = tree.query(a_geom2d.buffer(search_buffer))
        if not isinstance(cand_idx, (list, tuple, np.ndarray)):
            cand_idx = [cand_idx]

        best = {'left': None, 'right': None}
        nprobe = min(60, len(a_rs_xy))
        probe_idx = np.linspace(0, len(a_rs_xy) - 1, nprobe).astype(int)

        for idx in cand_idx:
            geom2d_b, b_id, xyz_d_b, s_tab_b = records[idx]
            if b_id == a_id:
                continue
            par = _mean_parallelism_xy(a_xyz[:, :2], xyz_d_b[:, :2])
            if par < parallelism_threshold:
                continue
            dists, signs = [], []
            for i in probe_idx:
                p_xy = a_rs_xy[i]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(records[idx], p_xy, ds=orient_ds)
                d_xy = q_xyz[:2] - p_xy
                tA = a_t[i]
                cross_z = tA[0] * d_xy[1] - tA[1] * d_xy[0]
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

        for side in ('left', 'right'):
            if best[side] is None:
                continue
            _, idx, par, mean_dist = best[side]
            geom2d_b, b_id, xyz_d_b, s_tab_b = records[idx]
            pair_key = frozenset({int(a_id), int(b_id)})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            nS = len(a_rs_xy)
            b_pts_xyz = np.empty((nS, 3), dtype=float)
            b_tan_xy = np.empty((nS, 2), dtype=float)
            b_s = np.empty(nS, dtype=float)
            for i, p_xy in enumerate(a_rs_xy):
                q_xyz, tB, sB = _closest_xyz_and_tangent_from_record(records[idx], p_xy, ds=orient_ds)
                b_pts_xyz[i] = q_xyz
                b_tan_xy[i] = tB
                b_s[i] = sB

            band_vec_xyz = b_pts_xyz - a_rs_xyz
            width_series = np.linalg.norm(band_vec_xyz[:, :2], axis=1)
            rep_width = float(np.median(width_series))
            cos_series = np.abs(np.sum(a_t * b_tan_xy, axis=1))
            orient_mask = cos_series >= cos_thresh
            width_mask = width_series >= small_gap_skip_m

            if segment_wide_only:
                keep_mask = width_mask & orient_mask
            else:
                if rep_width < small_gap_skip_m:
                    continue
                keep_mask = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)

            if not np.any(keep_mask):
                continue

            lane_count_series = np.array([_decide_count(w) for w in width_series], dtype=int)

            # segment runs of constant lane_count
            segments: List[Tuple[int, int, int]] = []
            start = 0
            cur_val = int(lane_count_series[0])
            for i in range(1, nS):
                if lane_count_series[i] != cur_val:
                    segments.append((start, i, cur_val))
                    start = i; cur_val = int(lane_count_series[i])
            segments.append((start, nS, cur_val))

            # filter segments by keep_mask and length
            s_table = _arclength2d(a_rs_xy)
            filtered: List[Tuple[int, int, int]] = []
            for (i0, i1, cnt) in segments:
                if not np.any(keep_mask[i0:i1]):
                    continue
                seg_len = s_table[i1 - 1] - s_table[i0]
                if seg_len < min_segment_len_m:
                    continue
                filtered.append((i0, i1, cnt))

            # merge very short filtered segments
            if filtered:
                merged = []
                pa, pb, pc = filtered[0]
                for a, b, c in filtered[1:]:
                    prev_len = s_table[pb - 1] - s_table[pa]
                    if prev_len < merge_short_segments_m:
                        pb = b; pc = c
                    else:
                        merged.append((pa, pb, pc))
                        pa, pb, pc = a, b, c
                merged.append((pa, pb, pc))
                filtered = merged

            if not filtered:
                continue

            # emit centerlines per filtered segment
            for (a_idx, b_idx, n_lanes) in filtered:
                seg_widths = width_series[a_idx:b_idx]
                seg_band = band_vec_xyz[a_idx:b_idx]
                seg_a_rs_xyz = a_rs_xyz[a_idx:b_idx]
                rep_seg_width = float(np.median(seg_widths))
                n_lanes = _decide_count(rep_seg_width)
                for k in range(n_lanes):
                    alpha = (k + 0.5) / n_lanes
                    seg_center_xyz = seg_a_rs_xyz + alpha * seg_band
                    # smoothing (safe)
                    seg_center_xyz = _smooth_polyline_nd(seg_center_xyz, window=min(smoothing_window, max(3, len(seg_center_xyz))), poly=smoothing_poly)
                    seg_center_xyz = straighten_centerline(seg_center_xyz, width_series=seg_widths)
                    out.append(CenterlineResult(
                        boundary_a_id=int(a_id),
                        boundary_b_id=int(b_id),
                        side=side,
                        parallelism=float(par),
                        mean_gap_m=float(np.median(seg_widths)),
                        lane_index=k,
                        lane_count=n_lanes,
                        alpha=float(alpha),
                        centerline=seg_center_xyz,
                    ))

    return out


# ------------------------ Example usage ------------------------ #

# if __name__ == "__main__":
#     # Example:
#     # map_features = [...]
#     # bd = build_boundary_dict_xyz(map_features)
#     # cls = build_centerlines_from_boundaries_xyz(bd)
#     # print(len(cls))
#     pass
