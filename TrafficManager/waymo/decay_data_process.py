import numpy as np
import matplotlib.pyplot as plt

# centerlines_from_boundaries.py
# Requires: pip install shapely scipy numpy
import numpy as np
from shapely.geometry import LineString, Point

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from .desay_lane_graph import build_lane_graph,plot_lane_graph

# ------------------------ Utilities (3D-aware) ------------------------ #

def _to_xyz(arr) -> np.ndarray:
    """Ensure shape [N,3]. If [N,2], pad z=0; if [3,N], transpose."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("xyz must be 2D array-like")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), dtype=float)])
    if a.shape[1] != 3:
        raise ValueError("xyz must be of shape [N,3] or [3,N] or [N,2]")
    return a


def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _densify_xyz(xyz: np.ndarray, max_seg_len: float = 0.5) -> np.ndarray:
    """
    Densify a 3D polyline so consecutive samples are <= max_seg_len apart in XY.
    XYZ is linearly interpolated across inserted points.
    """
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
        ts = np.linspace(0.0, 1.0, n + 1)[1:]  # exclude a, include b
        pts = a + (b - a) * ts[:, None]
        out.extend(pts)
    return np.asarray(out, dtype=float)


def _resample_by_arclength_xyz(xyz: np.ndarray, n: int = 200) -> np.ndarray:
    """Resample a 3D polyline uniformly in XY arclength to n samples (x,y,z all interpolated)."""
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
    """Interpolate xyz at a given XY arclength s_query using table s_table from same densified polyline."""
    x = np.interp(s_query, s_table, xyz[:, 0])
    y = np.interp(s_query, s_table, xyz[:, 1])
    z = np.interp(s_query, s_table, xyz[:, 2])
    return np.array([x, y, z], dtype=float)


def _tangents2d(xy: np.ndarray, smooth_window: int = 15, smooth_poly: int = 2) -> np.ndarray:
    """Unit tangents along a 2D polyline (optionally smoothed)."""
    v = np.zeros_like(xy, dtype=float)
    v[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] >= smooth_window and smooth_window >= 3:
        if smooth_window % 2 == 0:
            smooth_window += 1
        v = savgol_filter(v, smooth_window, smooth_poly, axis=0, mode="interp")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n


def _mean_parallelism_xy(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    """Mean |cos(theta)| between 2D tangents after resampling to 150 pts each."""
    A_rs = _resample_by_arclength_xyz(np.column_stack([A_xy, np.zeros(len(A_xy))]), 150)[:, :2]
    B_rs = _resample_by_arclength_xyz(np.column_stack([B_xy, np.zeros(len(B_xy))]), 150)[:, :2]
    tA = _tangents2d(A_rs)
    tB = _tangents2d(B_rs)
    return float(np.nanmean(np.abs(np.sum(tA * tB, axis=1))))


def _smooth_polyline_nd(arr: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing per channel (works for 2D or 3D)."""
    arr = np.asarray(arr, dtype=float)
    if window < 3 or arr.shape[0] < window:
        return arr
    if window % 2 == 0:
        window += 1
    out = []
    for d in range(arr.shape[1]):
        out.append(savgol_filter(arr[:, d], window, poly, mode="interp"))
    return np.stack(out, axis=1)


# ------------------------ Data structures ------------------------ #

@dataclass
class CenterlineResult:
    boundary_a_id: int              # reference boundary id
    boundary_b_id: int              # paired/partner boundary id
    side: str                       # 'left' or 'right' relative to boundary A orientation (in XY)
    parallelism: float
    mean_gap_m: float               # median XY gap across corridor
    lane_index: int                 # 0..lane_count-1
    lane_count: int
    alpha: float                    # fraction from boundary A -> partner boundary
    centerline: np.ndarray          # [M,3]


# ------------------------ Core algorithm (boundary-only, 3D) ------------------------ #

def build_centerlines_from_boundaries_xyz(
    boundary_dict: Dict[int, np.ndarray],      # {boundary_id: [N,3] or [3,N] or [N,2]}
    *,
    densify_step: float = 0.5,                 # m
    search_buffer: float = 20.0,               # m neighborhood to look for partner boundaries
    parallelism_threshold: float = 0.6,        # mean |cos θ| in XY
    side_confidence_eps: float = 0.05,         # avg sign magnitude must exceed this
    # lane splitting across the corridor:
    desired_lane_width_m: float = 3.6,
    split_tolerance_m: float = 0.6,
    max_lanes: int = 6,
    # corridor distance sanity:
    min_pair_mean_dist: float = 2.0,           # ignore super-narrow shoulders
    max_pair_mean_dist: Optional[float] = 25.0,# cap to avoid across-median pairing
    # smoothing:
    smoothing_window: int = 21,
    smoothing_poly: int = 3,
    # narrow sections handling:
    segment_wide_only: bool = False,           # emit only segments >= small_gap_skip_m
    small_gap_skip_m: float = 1.2,             # threshold for "wide" when segmenting
    min_segment_len_m: float = 8.0,            # ignore tiny segments when segmenting
    # orientation gating:
    max_orient_diff_deg: float = 35.0,         # omit samples with angle diff > this
    orient_ds: float = 0.5,                    # arc-length window for tangent estimation (m)
    segment_on_orientation: bool = True,       # split into segments when orientation fails
) -> List[CenterlineResult]:
    """
    Pair each boundary (3D) with the best opposite boundary (left/right) using XY geometry,
    then emit lane centerlines across the corridor. Centerlines are returned as [M,3] arrays.
    """

    # Build candidate geometries and parameterizations
    # records: (geom2d, id, xyz_densified, s_table)
    records: List[Tuple[LineString, int, np.ndarray, np.ndarray]] = []
    for bid, xyz in boundary_dict.items():
        xyz_d = _densify_xyz(xyz, max_seg_len=densify_step)     # [Nd,3]
        if xyz_d.shape[0] >= 2:
            xy_d = xyz_d[:, :2]
            geom2d = LineString(xy_d)                           # XY only for Shapely ops
            s_tab = _arclength2d(xy_d)                          # XY arclength parameterization
            records.append((geom2d, int(bid), xyz_d, s_tab))
    if not records:
        return []

    geoms = [g for g, *_ in records]
    tree = STRtree(geoms)

    out: List[CenterlineResult] = []
    seen_pairs = set()
    cos_thresh = float(np.cos(np.deg2rad(max_orient_diff_deg)))

    # Helper: closest XYZ point and 2D tangent on partner at XY arclength s
    def _closest_xyz_and_tangent(rec, p_xy: np.ndarray, ds: float):
        geom2d, _bid, xyz_d, s_tab = rec
        s = geom2d.project(Point(float(p_xy[0]), float(p_xy[1])))
        L = geom2d.length
        s0 = max(0.0, s - ds)
        s1 = min(L,    s + ds)
        q0 = geom2d.interpolate(s0); q1 = geom2d.interpolate(s1)
        t_xy = np.array([q1.x - q0.x, q1.y - q0.y], dtype=float)
        n = np.linalg.norm(t_xy) + 1e-9
        t_xy = t_xy / n
        q_xyz = _resample_at_s_xyz(xyz_d, s, s_tab)             # (x,y,z) at arclength s
        return q_xyz, t_xy, float(s)

    for a_id, a_xyz0 in boundary_dict.items():
        a_xyz = _densify_xyz(a_xyz0, max_seg_len=densify_step)
        if a_xyz.shape[0] < 2:
            continue

        a_geom2d = LineString(a_xyz[:, :2])
        a_rs_xyz = _resample_by_arclength_xyz(a_xyz, 200)       # [200,3]
        a_rs_xy  = a_rs_xyz[:, :2]
        a_t      = _tangents2d(a_rs_xy)                         # [200,2]

        # spatial candidates (boundaries near A)
        cand_idx = tree.query(a_geom2d.buffer(search_buffer))
        if not isinstance(cand_idx, (list, tuple, np.ndarray)):
            cand_idx = [cand_idx]

        # choose best partner on each side
        best = {'left': None, 'right': None}  # (score, rec_idx, par, mean_dist)
        nprobe = min(60, len(a_rs_xy))
        probe_idx = np.linspace(0, len(a_rs_xy) - 1, nprobe).astype(int)

        for idx in cand_idx:
            rec = records[idx]
            geom2d_b, b_id, xyz_d_b, s_tab_b = rec
            if b_id == a_id:
                continue

            par = _mean_parallelism_xy(a_xyz[:, :2], xyz_d_b[:, :2])
            if par < parallelism_threshold:
                continue

            dists = []
            signs = []
            for i in probe_idx:
                p_xy = a_rs_xy[i]
                q_xyz, _tB, _sB = _closest_xyz_and_tangent(rec, p_xy, ds=orient_ds)
                d_xy = q_xyz[:2] - p_xy
                tA   = a_t[i]
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

        # Build centerlines for each valid side; dedupe boundary pairs
        for side in ('left', 'right'):
            if best[side] is None:
                continue
            _, b_idx, par, mean_dist = best[side]
            rec_b = records[b_idx]
            geom2d_b, b_id, xyz_d_b, s_tab_b = rec_b

            pair_key = frozenset({int(a_id), int(b_id)})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # per-sample closest partner point and tangent, with z via s interpolation
            b_pts_xyz = np.empty_like(a_rs_xyz)                 # [200,3]
            b_tan_xy  = np.empty_like(a_rs_xy)                  # [200,2]
            b_s       = np.empty(len(a_rs_xy), dtype=float)
            for i, p_xy in enumerate(a_rs_xy):
                q_xyz, tB_xy, sB = _closest_xyz_and_tangent(rec_b, p_xy, ds=orient_ds)
                b_pts_xyz[i] = q_xyz
                b_tan_xy[i]  = tB_xy
                b_s[i]       = sB

            band_vec_xyz = b_pts_xyz - a_rs_xyz                 # 3D vector across corridor
            width_series = np.linalg.norm(band_vec_xyz[:, :2], axis=1)  # XY width
            rep_width    = float(np.median(width_series))

            # Orientation mask: |cos(theta)| >= cos_thresh
            cos_series = np.abs(np.sum(a_t * b_tan_xy, axis=1))
            orient_mask = cos_series >= cos_thresh

            # Width + orientation gating → contiguous runs
            runs: List[Tuple[int, int]] = []
            if segment_wide_only:
                width_mask = width_series >= small_gap_skip_m
                keep_mask  = width_mask & orient_mask
            else:
                if rep_width < small_gap_skip_m:
                    continue
                keep_mask = orient_mask if segment_on_orientation else np.ones_like(orient_mask, dtype=bool)

            # Convert mask → runs and drop tiny segments
            if np.any(keep_mask == False):
                idx = np.flatnonzero(np.diff(np.concatenate(([False], keep_mask, [False]))))
                runs = list(zip(idx[0::2], idx[1::2]))
            else:
                runs = [(0, len(a_rs_xyz))]

            # filter by geometric length in XY
            if runs:
                s_a = _arclength2d(a_rs_xy)
                runs = [(i0, i1) for (i0, i1) in runs if (s_a[i1 - 1] - s_a[i0]) >= min_segment_len_m]
            if not runs:
                continue

            # decide lane count and emit [M,3] centerlines at alpha=(k+0.5)/n
            def _decide_lane_count(width_m: float) -> int:
                if width_m <= desired_lane_width_m - split_tolerance_m:
                    return 1
                n = int(np.floor((width_m + split_tolerance_m) / max(1e-6, desired_lane_width_m)))
                return int(np.clip(n, 1, max_lanes))

            n_lanes = _decide_lane_count(rep_width)

            for k in range(n_lanes):
                alpha = (k + 0.5) / n_lanes
                for (i0, i1) in runs:
                    seg_center_xyz = a_rs_xyz[i0:i1] + alpha * band_vec_xyz[i0:i1]
                    # smooth with a window that fits the segment (apply per channel)
                    w = min(smoothing_window, (len(seg_center_xyz) // 2) * 2 - 1)
                    if w >= 3:
                        seg_center_xyz = _smooth_polyline_nd(seg_center_xyz, window=w, poly=min(smoothing_poly, 3))
                    out.append(CenterlineResult(
                        boundary_a_id=int(a_id),
                        boundary_b_id=int(b_id),
                        side=side,
                        parallelism=par,
                        mean_gap_m=rep_width,
                        lane_index=k,
                        lane_count=n_lanes,
                        alpha=float(alpha),
                        centerline=seg_center_xyz,   # [M,3]
                    ))

    return out


def decode_map_features_from_json(annotation,remove_mapid=[]):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    # other_id=[]
    point_cnt = 0

    map_features=annotation['lines']+annotation["traffic_elements"]
    line_dict={}
    boundary_dict={}

    max_id=0

    for mf in map_features:
        id=mf['global_id']

        if id in remove_mapid:
            continue

        feature_data_type=mf['class']
        xyz=np.array(mf['xyz']).T
        cur_info = {"id": id}
        max_id=max(max_id,id)

        if feature_data_type=="lane_line":
            line_type = mf['attrs']["laneline_type"]
            if line_type=="solid":
                cur_info["type"] = 7
                # plt.plot(xyz[:, 0], xyz[:, 1], color='r')
                # plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')

            else:#dot
                cur_info["type"] = 6
                # print(line_type)
                # plt.plot(xyz[:, 0], xyz[:, 1], color='g')
                # plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')
            line_dict[id]=xyz

        elif feature_data_type=="boundary":
            cur_info["type"] = 4
            plt.plot(xyz[:, 0], xyz[:, 1], color='y')
            plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')

            boundary_dict[id]=xyz

        elif feature_data_type == "speed_bump" or feature_data_type=="crosswalk":
            cur_info["type"] = 9
        # elif feature_data_type=="arrow":
        #     continue
        else:
            continue

        cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)

        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        if feature_data_type=="lane_line":
            map_infos["road_line"].append(cur_info)
        elif feature_data_type == "boundary":
            map_infos["road_edge"].append(cur_info)
        elif feature_data_type == "speed_bump":
            map_infos["crosswalk"].append(cur_info)


    centerlines=build_centerlines_from_boundaries_xyz(boundary_dict)

    for centerline in centerlines:

        center=centerline.centerline

        plt.plot(center[:, 0], center[:, 1], color='grey')
        plt.plot(center[:2, 0], center[:2, 1], color='red')

    plt.show()


    # # print(len(polylines))
    #

    centerline_list=[]

    for i,centerline in enumerate(centerlines):
        cur_info = {"id": max_id+1+i}

        cur_info["type"] = 1
        xyz = centerline.centerline

        centerline_list.append(xyz)

        cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        map_infos["lane"].append(cur_info)

    G=build_lane_graph(centerline_list)

    plot_lane_graph(G)

    # print(len(line_dict.keys()))

    #plt.show()
    # centerline_list=[]
    # for i,group in enumerate(annotation['lane_line_groups']):
    #
    #     lane1=line_dict[group['lane_line_ids'][0]]
    #     lane2=line_dict[group['lane_line_ids'][1]]
    #
    #     xyz=centerline(lane1, lane2)
    #     cur_info = {"id": max_id+1+i}
    #
    #     cur_info["type"] = 1
    #
    #     centerline_list.append(xyz[:,:2])
    #
    #     cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)
    #     cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
    #     polylines.append(cur_polyline)
    #     point_cnt += len(cur_polyline)
    #
    #     map_infos["lane"].append(cur_info)

        # plt.plot(lane1[:,0],lane1[:,1],color='r')
        # plt.plot(lane2[:,0],lane2[:,1],color='b')
        # plt.plot(xyz[:,0],xyz[:,1],color='y')
        # plt.plot(xyz[-2:,0],xyz[-2:,1],color='g')
        # plt.show()
        #
        # print(1)

    map_infos["all_polylines_list"] = polylines
    map_infos["centerline_list"]=centerline_list

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos