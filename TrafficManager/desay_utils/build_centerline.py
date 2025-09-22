"""
build_centerlines_with_dash_fallback.py

Usage:
  - Provide map_features (list of dicts). Each feature must include:
      * 'global_id' : int
      * 'class' : 'lane_line' or 'boundary'
      * 'xyz' : [3,N] or [N,3] or [N,2]
      * lane lines have 'attrs' with 'laneline_type' == 'solid'|'dot'
  - Call `build_line_and_boundary_dicts(map_features)`, then
    `centerlines = build_centerlines(line_dict, boundary_dict)`.
Dependencies:
    pip install numpy scipy shapely
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


# ---------------- Data classes ----------------

@dataclass
class CenterlineResult:
    src_type: str                # 'dash_vs_boundary' or 'boundary_pair'
    src_id: int                  # id of dashed line if src_type==dash_vs_boundary, else left boundary id
    partner_id: int              # partner boundary id
    side: str                    # 'left' or 'right' relative to src
    lane_index: int
    lane_count: int
    alpha: float
    centerline: np.ndarray       # [M,3]


# ---------------- safe savgol helpers ----------------

def _safe_window(n: int, window: int) -> int:
    """Return largest odd window <= window and <= n and >=3, or 0 if impossible."""
    if n < 3 or window < 3:
        return 0
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return 0
    return w

def _safe_savgol_1d(arr: np.ndarray, window: int, poly: int):
    n = len(arr)
    w = _safe_window(n, window)
    if w == 0:
        return arr
    if poly >= w:
        poly = max(1, w-1)
    return savgol_filter(arr, w, poly, mode='interp')

def _safe_savgol_nd(mat: np.ndarray, window: int, poly: int):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        return _safe_savgol_1d(mat, window, poly)
    out = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        out[:, i] = _safe_savgol_1d(mat[:, i], window, poly)
    return out


# ---------------- geometry helpers (3D-aware) ----------------

def _to_xyz(arr: Iterable) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("xyz input must be 2-D array-like")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), dtype=float)])
    if a.shape[1] != 3:
        raise ValueError("xyz shape must be [N,3] or [N,2] or [3,N]")
    return a

def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _densify_xyz(xyz: np.ndarray, max_seg_len: float = 0.5) -> np.ndarray:
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
    v = _safe_savgol_nd(v, smooth_window, smooth_poly)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _mean_parallelism_xy(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    A2 = _resample_by_arclength_xyz(np.column_stack([A_xy, np.zeros(len(A_xy))]), 150)[:, :2]
    B2 = _resample_by_arclength_xyz(np.column_stack([B_xy, np.zeros(len(B_xy))]), 150)[:, :2]
    tA = _tangents2d(A2); tB = _tangents2d(B2)
    return float(np.nanmean(np.abs(np.sum(tA * tB, axis=1))))

def _closest_xyz_and_tangent_from_record(rec, p_xy: np.ndarray, ds: float = 0.5):
    geom2d, _bid, xyz_d, s_tab = rec
    s = geom2d.project(Point(float(p_xy[0]), float(p_xy[1])))
    L = geom2d.length
    s0 = max(0.0, s - ds)
    s1 = min(L, s + ds)
    q0 = geom2d.interpolate(s0); q1 = geom2d.interpolate(s1)
    t_xy = np.array([q1.x - q0.x, q1.y - q0.y], dtype=float)
    t_xy /= (np.linalg.norm(t_xy) + 1e-9)
    q_xyz = np.array([np.interp(s, s_tab, xyz_d[:, 0]),
                      np.interp(s, s_tab, xyz_d[:, 1]),
                      np.interp(s, s_tab, xyz_d[:, 2])], dtype=float)
    return q_xyz, t_xy, float(s)


# ---------------- straightening helper ----------------

def _smooth_polyline_nd(arr: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] < 3:
        return arr
    w = _safe_window(len(arr), window)
    if w == 0:
        return arr
    return _safe_savgol_nd(arr, w, poly)

def straighten_centerline(center_xyz: np.ndarray, width_series: Optional[np.ndarray] = None,
                          base_smooth: float = 0.0, strength: float = 1.0, blend_factor_max: float = 0.9,
                          min_points: int = 6, spline_k: int = 3) -> np.ndarray:
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


# ---------------- parsing helper ----------------

def build_line_and_boundary_dicts(map_features: Iterable[Dict[str, Any]], remove_mapid: Optional[Iterable[int]] = None):
    line_dict: Dict[int, Tuple[np.ndarray, str]] = {}   # id -> (xyz [N,3], type 'solid'|'dot')
    boundary_dict: Dict[int, np.ndarray] = {}
    rm = set(remove_mapid or [])
    for mf in map_features:
        fid = int(mf['global_id'])
        if fid in rm:
            continue
        cls = mf.get('class')
        xyz_raw = np.array(mf['xyz'])
        if xyz_raw.ndim == 2 and xyz_raw.shape[0] == 3:
            xyz = xyz_raw.T[:, :3]
        elif xyz_raw.ndim == 2 and xyz_raw.shape[1] >= 2:
            # [N,2] or [N,3]
            if xyz_raw.shape[1] == 2:
                xyz = np.column_stack([xyz_raw, np.zeros(len(xyz_raw))])
            else:
                xyz = xyz_raw[:, :3]
        else:
            continue

        if cls == 'lane_line':
            line_type = mf.get('attrs', {}).get('laneline_type', 'dot')
            # store both dotted and solid, but we'll prefer 'dot' when building centerlines
            line_dict[fid] = (xyz, line_type)
        elif cls == 'boundary':
            boundary_dict[fid] = xyz
        else:
            continue
    return line_dict, boundary_dict


# ---------------- core combining function ----------------

def build_centerlines(line_dict: Dict[int, Tuple[np.ndarray, str]],
                      boundary_dict: Dict[int, np.ndarray],
                      *,
                      densify_step: float = 0.5,
                      search_buffer: float = 20.0,
                      parallelism_threshold: float = 0.6,
                      orient_ds: float = 0.5,
                      max_orient_diff_deg: float = 35.0,
                      small_gap_skip_m: float = 1.2,
                      desired_lane_width_m: float = 3.6,
                      smoothing_window: int = 21,
                      smoothing_poly: int = 3,
                      min_segment_len_m: float = 8.0,
                      segment_wide_only: bool = False,
                      segment_on_orientation: bool = True,
                      max_lanes: int = 6
                      ) -> List[CenterlineResult]:
    """
    Build centerlines using dashed lane lines preferentially; otherwise fallback to boundary pairs.
    Returns list of CenterlineResult (with centerline [M,3]).
    """

    # 1) Prepare densified records for boundaries and an STRtree
    boundary_records = []
    for bid, xyz in boundary_dict.items():
        xyzd = _densify_xyz(xyz, max_seg_len=densify_step)
        if xyzd.shape[0] < 2:
            continue
        geom2d = LineString(xyd := xyzd[:, :2])
        s_tab = _arclength2d(xyd)
        boundary_records.append((geom2d, int(bid), xyzd, s_tab))
    if not boundary_records:
        return []

    bgeoms = [r[0] for r in boundary_records]
    btree = STRtree(bgeoms)

    results: List[CenterlineResult] = []
    used_boundary_pairs = set()
    cos_thresh = float(np.cos(np.deg2rad(max_orient_diff_deg)))

    # Helper to choose lane count from width
    def _decide_count(width: float) -> int:
        if width <= desired_lane_width_m - 0.6:
            return 1
        n = int(np.floor((width + 0.6) / desired_lane_width_m))
        return int(np.clip(n, 1, max_lanes))

    # 2) First: build centerlines using dotted lane lines when present
    for lid, lxyz_raw in list(line_dict.items()):
        # process only dotted (dash) lines as preferred; if you want to allow solids too, include them
        # if ltype != 'dot':
        #     continue
        lxyz = _densify_xyz(lxyz_raw, max_seg_len=densify_step)
        if lxyz.shape[0] < 2:
            continue
        l_geom2d = LineString(lxyz[:, :2])
        l_rs_xyz = _resample_by_arclength_xyz(lxyz, 200)
        l_rs_xy = l_rs_xyz[:, :2]
        l_t = _tangents2d(l_rs_xy)

        # find nearby boundaries
        cand_idx = btree.query(l_geom2d.buffer(search_buffer))
        if not isinstance(cand_idx, (list, tuple, np.ndarray)):
            cand_idx = [cand_idx]

        best = None  # (score, rec_idx, par, mean_dist, side_sign)
        for idx in cand_idx:
            geom2d_b, bid, xyzd_b, s_tab_b = boundary_records[idx]
            par = _mean_parallelism_xy(lxyz[:, :2], xyzd_b[:, :2])
            if par < parallelism_threshold:
                continue
            # compute mean distance and side sign
            # sample ~60 points along l_rs
            nprobe = min(60, len(l_rs_xy))
            pidx = np.linspace(0, len(l_rs_xy)-1, nprobe).astype(int)
            dists = []
            signs = []
            for i in pidx:
                p_xy = l_rs_xy[i]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx], p_xy, ds=orient_ds)
                d = q_xyz[:2] - p_xy
                cross_z = l_t[i][0] * d[1] - l_t[i][1] * d[0]
                signs.append(np.sign(cross_z))
                dists.append(np.linalg.norm(d))
            mean_dist = float(np.mean(dists))
            mean_sign = float(np.mean(signs))
            if mean_dist < 1.0:
                continue
            if abs(mean_sign) < 0.05:
                continue
            side = 'left' if mean_sign > 0 else 'right'
            score = mean_dist / max(1e-6, par)
            if (best is None) or (score < best[0]):
                best = (score, idx, par, mean_dist, mean_sign, side)

        if best is None:
            continue

        _, idx, par, mean_dist, mean_sign, side = best
        geom2d_b, bid, xyzd_b, s_tab_b = boundary_records[idx]

        # produce centerlines between the dashed line (l_rs_xyz) and this boundary
        # compute per-sample closest points and tangents for orientation gating
        nS = len(l_rs_xy)
        b_pts_xyz = np.empty((nS, 3), dtype=float)
        b_tan_xy = np.empty((nS, 2), dtype=float)
        for i, p_xy in enumerate(l_rs_xy):
            q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx], p_xy, ds=orient_ds)
            b_pts_xyz[i] = q_xyz
            b_tan_xy[i] = tB

        band_vec = b_pts_xyz - l_rs_xyz
        width_series = np.linalg.norm(band_vec[:, :2], axis=1)
        cos_series = np.abs(np.sum(l_t * b_tan_xy, axis=1))
        orient_mask = cos_series >= cos_thresh
        width_mask = width_series >= small_gap_skip_m

        if segment_wide_only:
            keep_mask = width_mask & orient_mask
        else:
            if np.median(width_series) < small_gap_skip_m:
                continue
            keep_mask = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)

        if not np.any(keep_mask):
            continue

        # lane count series
        lane_count_series = np.array([_decide_count(w) for w in width_series], dtype=int)
        # segments
        segments = []
        start = 0; cur_cnt = int(lane_count_series[0])
        for i in range(1, nS):
            if int(lane_count_series[i]) != cur_cnt:
                segments.append((start, i, cur_cnt))
                start = i; cur_cnt = int(lane_count_series[i])
        segments.append((start, nS, cur_cnt))

        # filter segments by keep_mask and min length
        s_table = _arclength2d(l_rs_xy)
        filtered = []
        for i0, i1, cnt in segments:
            if not np.any(keep_mask[i0:i1]):
                continue
            if (s_table[i1 - 1] - s_table[i0]) < min_segment_len_m:
                continue
            filtered.append((i0, i1, cnt))

        # emit centerlines
        for a_idx, b_idx, n_lanes in filtered:
            seg_widths = width_series[a_idx:b_idx]
            seg_a = l_rs_xyz[a_idx:b_idx]
            seg_band = band_vec[a_idx:b_idx]
            rep_width = float(np.median(seg_widths))
            n_lanes = _decide_count(rep_width)
            for k in range(n_lanes):
                alpha = (k + 0.5) / n_lanes
                seg_center = seg_a + alpha * seg_band
                seg_center = _smooth_polyline_nd(seg_center, window=min(smoothing_window, max(3, len(seg_center))), poly=smoothing_poly)
                seg_center = straighten_centerline(seg_center, width_series=seg_widths)
                results.append(CenterlineResult(
                    src_type='dash_vs_boundary',
                    src_id=int(lid),
                    partner_id=int(bid),
                    side=side,
                    lane_index=k,
                    lane_count=n_lanes,
                    alpha=float(alpha),
                    centerline=seg_center
                ))

        # mark pair as used to avoid duplicate boundary-boundary later
        used_boundary_pairs.add(tuple(sorted((int(bid), int(lid)))))

    # 3) Second pass: for boundaries not covered by dash lines, do boundary↔boundary fallback
    # We'll pair boundaries with each other as before, but skip pairs already 'covered' by dash above.
    # Build quick map from boundary id to its record index
    bid_to_idx = {rec[1]: i for i, rec in enumerate(boundary_records)}

    # iterate boundaries and try to pair with best opposite boundary
    for i, (geom2d_a, a_id, a_xyz_d, a_s_tab) in enumerate(boundary_records):
        # already potentially used? we still try because dash pairing might only cover some segments
        a_rs_xyz = _resample_by_arclength_xyz(a_xyz_d, 200)
        a_rs_xy = a_rs_xyz[:, :2]
        a_t = _tangents2d(a_rs_xy)

        cand_idx = btree.query(geom2d_a.buffer(search_buffer))
        if not isinstance(cand_idx, (list, tuple, np.ndarray)):
            cand_idx = [cand_idx]

        best = {'left': None, 'right': None}
        nprobe = min(60, len(a_rs_xy))
        probe_idx = np.linspace(0, len(a_rs_xy) - 1, nprobe).astype(int)

        for idx in cand_idx:
            geom2d_b, b_id, xyzd_b, s_tab_b = boundary_records[idx]
            if b_id == a_id:
                continue
            # compute parallelism
            #par = _mean_parallelism_xy(a_xyz_d := a_xyz_d if False else a_xyz_d, xyzd_b[:, :2])  # small inline tweak to keep signature
            par = _mean_parallelism_xy(a_xyz_d[:, :2] if False else a_xyz_d[:, :2], xyzd_b[:, :2]) if False else _mean_parallelism_xy(a_rs_xy, xyzd_b[:, :2])
            if par < parallelism_threshold:
                continue
            dists = []; signs = []
            for ii in probe_idx:
                p_xy = a_rs_xy[ii]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx], p_xy, ds=orient_ds)
                d_xy = q_xyz[:2] - p_xy
                tA = a_t[ii]
                cross_z = tA[0] * d_xy[1] - tA[1] * d_xy[0]
                signs.append(np.sign(cross_z))
                dists.append(np.linalg.norm(d_xy))
            mean_dist = float(np.mean(dists))
            mean_sign = float(np.mean(signs))
            if mean_dist < 2.0:
                continue
            if abs(mean_sign) < 0.05:
                continue
            side = 'left' if mean_sign > 0 else 'right'
            score = mean_dist / max(1e-6, par)
            cur = best[side]
            if (cur is None) or (score < cur[0]):
                best[side] = (score, idx, par, mean_dist)

        for side in ('left', 'right'):
            if best[side] is None:
                continue
            _, idx_b, par, mean_dist = best[side]
            geom2d_b, b_id, xyzd_b, s_tab_b = boundary_records[idx_b]

            # avoid repeating pairs if already covered by any dash pairing (simple check)
            if tuple(sorted((int(a_id), int(b_id)))) in used_boundary_pairs:
                # still might need to cover other segments, but skip to keep simple
                continue

            # compute closest points per sample
            nS = len(a_rs_xy)
            b_pts_xyz = np.empty((nS, 3), dtype=float)
            b_tan_xy = np.empty((nS, 2), dtype=float)
            for ii, p_xy in enumerate(a_rs_xy):
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx_b], p_xy, ds=orient_ds)
                b_pts_xyz[ii] = q_xyz
                b_tan_xy[ii] = tB

            band_vec = b_pts_xyz - a_rs_xyz
            width_series = np.linalg.norm(band_vec[:, :2], axis=1)
            cos_series = np.abs(np.sum(a_t * b_tan_xy, axis=1))
            orient_mask = cos_series >= cos_thresh
            width_mask = width_series >= small_gap_skip_m

            if segment_wide_only:
                keep_mask = width_mask & orient_mask
            else:
                if float(np.median(width_series)) < small_gap_skip_m:
                    continue
                keep_mask = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)

            if not np.any(keep_mask):
                continue

            lane_count_series = np.array([_decide_count(w := width_series[i]) for i in range(len(width_series))], dtype=int)
            # build runs of constant lane_count, filter by keep_mask and min length
            runs = []
            nS = len(lane_count_series)
            start = 0; cur_val = int(lane_count_series[0])
            for ii in range(1, nS):
                if int(lane_count_series[ii]) != cur_val:
                    runs.append((start, ii, cur_val))
                    start = ii; cur_val = int(lane_count_series[ii])
            runs.append((start, nS, cur_val))

            s_table = _arclength2d(a_rs_xy)
            filtered = []
            for (i0, i1, cnt) in runs:
                if not np.any(keep_mask[i0:i1]):
                    continue
                if (s_table[i1 - 1] - s_table[i0]) < min_segment_len_m:
                    continue
                filtered.append((i0, i1, cnt))
            if not filtered:
                continue

            # emit centerlines per segment
            for (a_idx, b_idx, n_lanes) in filtered:
                seg_widths = width_series[a_idx:b_idx]
                seg_band = band_vec[a_idx:b_idx]
                seg_a = a_rs_xyz[a_idx:b_idx]
                repw = float(np.median(seg_widths))
                n_lanes = _decide_count(repw)
                for k in range(n_lanes):
                    alpha = (k + 0.5) / n_lanes
                    seg_center = seg_a + alpha * seg_band
                    seg_center = _smooth_polyline_nd(seg_center, window=min(smoothing_window, max(3, len(seg_center))), poly=smoothing_poly)
                    seg_center = straighten_centerline(seg_center, width_series=seg_widths)
                    results.append(CenterlineResult(
                        src_type='boundary_pair',
                        src_id=int(a_id),
                        partner_id=int(b_id),
                        side=side,
                        lane_index=k,
                        lane_count=n_lanes,
                        alpha=float(alpha),
                        centerline=seg_center
                    ))

    return results


# # ---------------- Example usage ----------------
#
# if __name__ == "__main__":
#     # Example usage sketch
#     # map_features = ...  # your list
#     # line_dict, boundary_dict = build_line_and_boundary_dicts(map_features)
#     # cls = build_centerlines(line_dict, boundary_dict)
#     # for c in cls[:5]:
#     #     print(c.src_type, c.src_id, c.partner_id, c.lane_index, c.lane_count, c.centerline.shape)
#     pass
