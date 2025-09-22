# centerline_from_dicts.py
# Requires: pip install numpy scipy shapely

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from collections import defaultdict

@dataclass
class CenterlineResult:
    src_type: str
    src_id: Optional[int]
    partner_id: Optional[int]
    side: Optional[str]
    lane_index: int
    lane_count: int
    alpha: float
    centerline: np.ndarray   # [M,3]

# ----------------- safe savgol helpers -----------------
def _safe_window(n:int, window:int)->int:
    if n < 3 or window < 3: return 0
    w = min(window, n)
    if w % 2 == 0: w -= 1
    if w < 3: return 0
    return w

def _safe_savgol_1d(arr: np.ndarray, window:int, poly:int) -> np.ndarray:
    n = len(arr)
    w = _safe_window(n, window)
    if w == 0:
        return arr
    if poly >= w:
        poly = max(1, w-1)
    return savgol_filter(arr, w, poly, mode='interp')

def _safe_savgol_nd(mat: np.ndarray, window:int, poly:int) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim == 1:
        return _safe_savgol_1d(mat, window, poly)
    out = np.zeros_like(mat)
    for d in range(mat.shape[1]):
        out[:, d] = _safe_savgol_1d(mat[:, d], window, poly)
    return out

# ----------------- geometry helpers (3D aware) -----------------
def _to_xyz(arr: Iterable) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("Input must be 2D array-like")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a),dtype=float)])
    if a.shape[1] != 3:
        raise ValueError("xyz must be [N,3] or [N,2] or [3,N]")
    return a

def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _densify_xyz(xyz: np.ndarray, max_seg_len: float = 0.5) -> np.ndarray:
    xyz = _to_xyz(xyz)
    if xyz.shape[0] < 2:
        return xyz
    out = [xyz[0]]
    for a,b in zip(xyz[:-1], xyz[1:]):
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
    if xyz.shape[0] == 1: return np.repeat(xyz, n, axis=0)
    s = _arclength2d(xyz[:, :2])
    if s[-1] == 0: return np.repeat(xyz[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    x = np.interp(t, s, xyz[:, 0]); y = np.interp(t, s, xyz[:, 1]); z = np.interp(t, s, xyz[:, 2])
    return np.stack([x,y,z], axis=1)

def _tangents2d(xy: np.ndarray, smooth_window:int=15, smooth_poly:int=2) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    v = _safe_savgol_nd(v, smooth_window, smooth_poly)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _mean_parallelism_xy(A_xy: np.ndarray, B_xy: np.ndarray)->float:
    A2 = _resample_by_arclength_xyz(np.column_stack([A_xy, np.zeros(len(A_xy))]), 150)[:, :2]
    B2 = _resample_by_arclength_xyz(np.column_stack([B_xy, np.zeros(len(B_xy))]), 150)[:, :2]
    tA = _tangents2d(A2); tB = _tangents2d(B2)
    return float(np.nanmean(np.abs(np.sum(tA * tB, axis=1))))

def _closest_xyz_and_tangent_from_record(rec, p_xy: np.ndarray, ds: float = 0.5):
    geom2d, _bid, xyz_d, s_tab = rec
    s = geom2d.project(Point(float(p_xy[0]), float(p_xy[1])))
    L = geom2d.length
    s0 = max(0.0, s-ds); s1 = min(L, s+ds)
    q0 = geom2d.interpolate(s0); q1 = geom2d.interpolate(s1)
    t_xy = np.array([q1.x - q0.x, q1.y - q0.y], dtype=float)
    t_xy /= (np.linalg.norm(t_xy) + 1e-9)
    q_xyz = np.array([np.interp(s, s_tab, xyz_d[:,0]),
                      np.interp(s, s_tab, xyz_d[:,1]),
                      np.interp(s, s_tab, xyz_d[:,2])], dtype=float)
    return q_xyz, t_xy, float(s)

def _smooth_polyline_nd(arr: np.ndarray, window:int=21, poly:int=3)->np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] < 3: return arr
    w = _safe_window(len(arr), window)
    if w == 0: return arr
    return _safe_savgol_nd(arr, w, poly)

def straighten_centerline(center_xyz: np.ndarray, width_series: Optional[np.ndarray]=None,
                          base_smooth:float=0.0, strength:float=1.0, blend_factor_max:float=0.9,
                          min_points:int=6, spline_k:int=3) -> np.ndarray:
    center_xyz = np.asarray(center_xyz, dtype=float)
    if center_xyz.shape[0] < min_points: return center_xyz.copy()
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
    splx = UnivariateSpline(s, center_xyz[:,0], s=s_spline, k=k)
    sply = UnivariateSpline(s, center_xyz[:,1], s=s_spline, k=k)
    splz = UnivariateSpline(s, center_xyz[:,2], s=s_spline, k=k)
    spline_xyz = np.stack([splx(s), sply(s), splz(s)], axis=1)
    baseline = np.outer(1.0 - (s/s[-1]), center_xyz[0]) + np.outer(s/s[-1], center_xyz[-1])
    c = 6.0
    blend = np.clip((1.0 - np.exp(-c * rel_var)) * blend_factor_max, 0.0, blend_factor_max)
    return (1.0 - blend) * spline_xyz + blend * baseline

# ----------------- dedupe & merge utilities -----------------
def _resample_polyline_xyzt(p: np.ndarray, n:int=100)->np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape[0] <= 1: return np.repeat(p, n, axis=0)
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, s[-1], n)
    x = np.interp(t, s, p[:,0]); y = np.interp(t, s, p[:,1]); z = np.interp(t, s, p[:,2])
    return np.stack([x,y,z], axis=1)

def _clipped_mean(v: np.ndarray, lo_p=20, hi_p=85):
    lo = np.percentile(v, lo_p); hi = np.percentile(v, hi_p)
    m = v[(v >= lo) & (v <= hi)]; return float(np.mean(m)) if m.size else float(np.mean(v))

def _mean_bidirectional_distance(a: np.ndarray, b: np.ndarray)->float:
    if a.size == 0 or b.size == 0: return float('inf')
    n = max(len(a), len(b))
    a_r = _resample_polyline_xyzt(a, n); b_r = _resample_polyline_xyzt(b, n)
    da = np.linalg.norm(a_r - b_r, axis=1); db = np.linalg.norm(b_r - a_r, axis=1)
    return 0.5 * (_clipped_mean(da) + _clipped_mean(db))

def dedupe_centerlines(results: List[CenterlineResult], distance_tol: float = 0.35, resample_n: int = 80) -> List[CenterlineResult]:
    if not results: return []
    ps = [np.asarray(r.centerline, dtype=float) for r in results]
    p_res = [_resample_polyline_xyzt(p, resample_n) for p in ps]
    used = [False] * len(results)
    reps = []
    groups = defaultdict(list)
    for i, r in enumerate(results):
        groups[(r.src_type, r.src_id, r.partner_id, r.side, r.lane_index)].append(i)
    for _, idxs in groups.items():
        for i in idxs:
            if used[i]: continue
            cluster = [i]; used[i] = True
            for j in idxs:
                if used[j] or j == i: continue
                d = _mean_bidirectional_distance(p_res[i], p_res[j])
                if d <= distance_tol:
                    used[j] = True; cluster.append(j)
            best = max(cluster, key=lambda k: (len(ps[k]), -k))
            reps.append(results[best])
    return reps

def _endpoints_close(a: np.ndarray, b: np.ndarray, tol: float = 0.6) -> bool:
    if a.size == 0 or b.size == 0: return False
    if np.linalg.norm(a[-1,:2] - b[0,:2]) <= tol: return True
    if np.linalg.norm(b[-1,:2] - a[0,:2]) <= tol: return True
    return False

def merge_adjacent_segments(results: List[CenterlineResult], gap_tol: float = 0.6)->List[CenterlineResult]:
    if not results: return []
    def same_key(r): return (r.src_type, r.src_id, r.partner_id, r.side, r.lane_index, r.lane_count)
    groups = defaultdict(list)
    for r in results: groups[same_key(r)].append(r)
    merged_results = []
    for k, group in groups.items():
        group_sorted = sorted(group, key=lambda r: float(np.mean(np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(r.centerline[:, :2], axis=0), axis=1)]))))
        used = [False]*len(group_sorted)
        for i, base in enumerate(group_sorted):
            if used[i]: continue
            used[i] = True
            segs = [np.asarray(base.centerline)]
            for j in range(i+1, len(group_sorted)):
                if used[j]: continue
                cand = np.asarray(group_sorted[j].centerline)
                if _endpoints_close(segs[-1], cand, tol=gap_tol):
                    segs.append(cand); used[j] = True
            merged_poly = segs[0] if len(segs)==1 else np.vstack([segs[0]] + [s[1:] for s in segs[1:]])
            new = CenterlineResult(src_type=base.src_type, src_id=base.src_id, partner_id=base.partner_id,
                                   side=base.side, lane_index=base.lane_index, lane_count=base.lane_count,
                                   alpha=base.alpha, centerline=merged_poly)
            merged_results.append(new)
    return merged_results

# ----------------- main builder (line_dict + line_type_dict + boundary_dict) -----------------
def build_centerlines_from_dicts(line_dict: Dict[int, np.ndarray],
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

    # prepare boundary records for spatial queries
    boundary_records = []
    for bid, xyz in boundary_dict.items():
        xyzd = _densify_xyz(xyz, max_seg_len=densify_step)
        if xyzd.shape[0] < 2: continue
        boundary_records.append((LineString(xyzd[:, :2]), int(bid), xyzd, _arclength2d(xyzd[:, :2])))
    if not boundary_records:
        return []
    bgeoms = [r[0] for r in boundary_records]
    btree = STRtree(bgeoms)

    results: List[CenterlineResult] = []
    used_pairs = set()
    cos_thresh = float(np.cos(np.deg2rad(max_orient_diff_deg)))

    def _decide_count(width: float)->int:
        if width <= desired_lane_width_m - 0.6: return 1
        n = int(np.floor((width + 0.6) / desired_lane_width_m))
        return int(np.clip(n, 1, max_lanes))

    # 1) prefer dashed lane lines (line_type='dot') to pair with nearest boundary
    for lid, lxyz_raw in line_dict.items():
        # ltype = line_type_dict.get(lid, 'dot')
        # if ltype != 'dot':
        #     continue
        lxyz = _densify_xyz(lxyz_raw, max_seg_len=densify_step)
        if lxyz.shape[0] < 2: continue
        l_geom2d = LineString(lxyz[:, :2])
        l_rs = _resample_by_arclength_xyz(lxyz, 200)
        l_xy = l_rs[:, :2]
        l_t = _tangents2d(l_xy)

        cand = btree.query(l_geom2d.buffer(search_buffer))
        if not isinstance(cand, (list, tuple, np.ndarray)): cand = [cand]
        best = None
        for idx in cand:
            geom2d_b, bid, xyzdb, s_tab_b = boundary_records[idx]
            par = _mean_parallelism_xy(lxyz[:, :2], xyzdb[:, :2])
            if par < parallelism_threshold: continue
            # sample distances & sign
            nprobe = min(60, len(l_xy))
            pidx = np.linspace(0, len(l_xy)-1, nprobe).astype(int)
            dists=[]; signs=[]
            for i in pidx:
                p_xy = l_xy[i]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx], p_xy, ds=orient_ds)
                d = q_xyz[:2] - p_xy
                cross = l_t[i,0]*d[1] - l_t[i,1]*d[0]
                signs.append(np.sign(cross)); dists.append(np.linalg.norm(d))
            mean_dist=float(np.mean(dists)); mean_sign=float(np.mean(signs))
            if mean_dist < 1.0 or abs(mean_sign) < 0.05: continue
            side = 'left' if mean_sign>0 else 'right'
            score = mean_dist / max(1e-6, par)
            if (best is None) or (score < best[0]): best=(score, idx, par, mean_dist, side)
        if best is None: continue
        _, idxb, par, mean_dist, side = best
        geom2d_b, bid, xyzdb, s_tab_b = boundary_records[idxb]

        # build band and per-sample checks
        nS = len(l_xy)
        b_pts = np.empty((nS,3)); b_tan = np.empty((nS,2))
        for i,p_xy in enumerate(l_xy):
            q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idxb], p_xy, ds=orient_ds)
            b_pts[i]=q_xyz; b_tan[i]=tB
        band = b_pts - l_rs
        widths = np.linalg.norm(band[:, :2], axis=1)
        cos_series = np.abs(np.sum(l_t * b_tan, axis=1))
        orient_mask = cos_series >= cos_thresh
        width_mask = widths >= small_gap_skip_m
        if segment_wide_only:
            keep = orient_mask & width_mask
        else:
            if np.median(widths) < small_gap_skip_m: continue
            keep = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)
        if not np.any(keep): continue

        lane_count_series = np.array([_decide_count(w) for w in widths], dtype=int)
        # segment by constant lane count
        segments=[]; start=0; cur=lane_count_series[0]
        for i in range(1, nS):
            if lane_count_series[i] != cur:
                segments.append((start, i, int(cur))); start=i; cur=lane_count_series[i]
        segments.append((start, nS, int(cur)))
        s_table = _arclength2d(l_xy)
        filtered=[]
        for i0,i1,cnt in segments:
            if not np.any(keep[i0:i1]): continue
            if (s_table[i1-1] - s_table[i0]) < min_segment_len_m: continue
            filtered.append((i0,i1,cnt))
        for a_idx,b_idx,n_lanes in filtered:
            seg_widths = widths[a_idx:b_idx]
            seg_a = l_rs[a_idx:b_idx]; seg_band = band[a_idx:b_idx]
            repw = float(np.median(seg_widths)); n_lanes = _decide_count(repw)
            for k in range(n_lanes):
                alpha = (k+0.5)/n_lanes
                seg_center = seg_a + alpha * seg_band
                seg_center = _smooth_polyline_nd(seg_center, window=min(smoothing_window, max(3,len(seg_center))), poly=smoothing_poly)
                seg_center = straighten_centerline(seg_center, width_series=seg_widths)
                results.append(CenterlineResult(src_type='dash_vs_boundary', src_id=int(lid), partner_id=int(bid),
                                               side=side, lane_index=k, lane_count=n_lanes, alpha=float(alpha), centerline=seg_center))
        used_pairs.add(tuple(sorted((int(bid), int(lid)))))

    # 2) fallback: boundary <-> boundary pairing
    if not boundary_records: return results
    for i, (geom2d_a, a_id, xyzda, s_tab_a) in enumerate(boundary_records):
        a_rs = _resample_by_arclength_xyz(xyzda, 200); a_xy = a_rs[:, :2]; a_t = _tangents2d(a_xy)
        cand = btree.query(geom2d_a.buffer(search_buffer))
        if not isinstance(cand, (list, tuple, np.ndarray)): cand=[cand]
        best={'left':None, 'right':None}
        nprobe = min(60, len(a_xy))
        pidx = np.linspace(0, len(a_xy)-1, nprobe).astype(int)
        for idx in cand:
            geom2d_b, b_id, xyzdb, s_tab_b = boundary_records[idx]
            if b_id == a_id: continue
            par = _mean_parallelism_xy(a_xy, xyzdb[:, :2])
            if par < parallelism_threshold: continue
            dists=[]; signs=[]
            for ii in pidx:
                p_xy = a_xy[ii]
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idx], p_xy, ds=orient_ds)
                d_xy = q_xyz[:2] - p_xy
                cross = a_t[ii,0]*d_xy[1] - a_t[ii,1]*d_xy[0]
                signs.append(np.sign(cross)); dists.append(np.linalg.norm(d_xy))
            mean_dist=float(np.mean(dists)); mean_sign=float(np.mean(signs))
            if mean_dist < 2.0 or abs(mean_sign) < 0.05: continue
            side = 'left' if mean_sign>0 else 'right'
            score = mean_dist / max(1e-6, par)
            cur = best[side]
            if (cur is None) or (score < cur[0]): best[side] = (score, idx, par, mean_dist)
        for side in ('left','right'):
            if best[side] is None: continue
            _, idxb, par, mean_dist = best[side]
            geom2d_b, b_id, xyzdb, s_tab_b = boundary_records[idxb]
            if tuple(sorted((int(a_id), int(b_id)))) in used_pairs: continue
            nS = len(a_xy)
            b_pts = np.empty((nS,3)); b_tan = np.empty((nS,2))
            for ii, p_xy in enumerate(a_xy):
                q_xyz, tB, _s = _closest_xyz_and_tangent_from_record(boundary_records[idxb], p_xy, ds=orient_ds)
                b_pts[ii]=q_xyz; b_tan[ii]=tB
            band = b_pts - a_rs
            widths = np.linalg.norm(band[:, :2], axis=1)
            cos_series = np.abs(np.sum(a_t * b_tan, axis=1))
            orient_mask = cos_series >= cos_thresh
            width_mask = widths >= small_gap_skip_m
            if segment_wide_only:
                keep = orient_mask & width_mask
            else:
                if np.median(widths) < small_gap_skip_m: continue
                keep = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)
            if not np.any(keep): continue
            lane_count_series = np.array([_decide_count(w) for w in widths], dtype=int)
            runs=[]; start=0; cur=lane_count_series[0]
            for ii in range(1, nS):
                if lane_count_series[ii] != cur:
                    runs.append((start, ii, int(cur))); start=ii; cur=lane_count_series[ii]
            runs.append((start, nS, int(cur)))
            s_table = _arclength2d(a_xy)
            filtered=[]
            for i0,i1,cnt in runs:
                if not np.any(keep[i0:i1]): continue
                if (s_table[i1-1] - s_table[i0]) < min_segment_len_m: continue
                filtered.append((i0,i1,cnt))
            for a_idx,b_idx,n_lanes in filtered:
                seg_widths = widths[a_idx:b_idx]; seg_a = a_rs[a_idx:b_idx]; seg_band = band[a_idx:b_idx]
                repw = float(np.median(seg_widths)); n_lanes = _decide_count(repw)
                for k in range(n_lanes):
                    alpha = (k+0.5)/n_lanes
                    seg_center = seg_a + alpha * seg_band
                    seg_center = _smooth_polyline_nd(seg_center, window=min(smoothing_window, max(3,len(seg_center))), poly=smoothing_poly)
                    seg_center = straighten_centerline(seg_center, width_series=seg_widths)
                    results.append(CenterlineResult(src_type='boundary_pair', src_id=int(a_id), partner_id=int(b_id),
                                                   side=side, lane_index=k, lane_count=n_lanes, alpha=float(alpha),
                                                   centerline=seg_center))
    # post-process: merge adjacent segments then dedupe
    merged = merge_adjacent_segments(results, gap_tol=0.6)
    deduped = dedupe_centerlines(merged, distance_tol=0.35, resample_n=80)
    return deduped

# --------------------- Example ---------------------
# Example usage:
# line_dict = {id: xyz_np, ...}          # from your code where dot lines added
# line_type_dict = {id: 'dot' or 'solid', ...}
# boundary_dict = {id: xyz_np, ...}
# centerlines = build_centerlines_from_dicts(line_dict, line_type_dict, boundary_dict)
# Each element in centerlines is a CenterlineResult with centerline (np.ndarray [M,3]).

