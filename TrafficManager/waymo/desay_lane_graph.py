# lane_graph.py
# Requires: pip install shapely scipy numpy networkx

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


# --------- small geometry helpers --------- #

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("centerline must be [N,3] or [N,2] or [3,N]")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3:
        raise ValueError("centerline must be shape [N,3]")
    return a

def _tangents2d(xy: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] >= window and window >= 3:
        if window % 2 == 0:
            window += 1
        v = savgol_filter(v, window, poly, axis=0, mode="interp")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _heading_at_endpoints(xy: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit tangents at start and end."""
    k = max(1, min(k, len(xy) - 1))
    t_start = xy[min(k, len(xy)-1)] - xy[0]
    t_end   = xy[-1] - xy[max(0, len(xy)-1-k)]
    for t in (t_start, t_end):
        n = np.linalg.norm(t)
        if n > 0:
            t /= n
    return t_start, t_end

def _closest_point_on_ls(ls: LineString, p: np.ndarray) -> Tuple[np.ndarray, float]:
    """Closest point on LineString to p (2D). Returns (point_xy, arclength_s)."""
    s = ls.project(Point(float(p[0]), float(p[1])))
    q = ls.interpolate(s)
    return np.array([q.x, q.y]), float(s)

def _polyline_length2d(xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0

def _angle_ok(t1: np.ndarray, t2: np.ndarray, max_diff_deg: float) -> bool:
    cos = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
    return cos >= np.cos(np.deg2rad(max_diff_deg))

def _signed_lateral(p2: np.ndarray, p1: np.ndarray, t1: np.ndarray) -> float:
    """Signed lateral offset from p1 (on lane A) to p2 (on lane B) using A's tangent t1.
       + means to the left of A, - to the right."""
    d = p2 - p1
    return float(t1[0]*d[1] - t1[1]*d[0])

def _resample_by_count(xyz: np.ndarray, n: int = 200) -> np.ndarray:
    """Uniform arclength resampling in XY to n samples; z interpolated."""
    xyz = _to_xyz(xyz)
    if len(xyz) == 1:
        return np.repeat(xyz, n, axis=0)
    # arclength in XY
    d = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] == 0:
        return np.repeat(xyz[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    x = np.interp(u, s, xyz[:, 0]); y = np.interp(u, s, xyz[:, 1]); z = np.interp(u, s, xyz[:, 2])
    return np.stack([x, y, z], axis=1)


# --------- data container (optional) --------- #

@dataclass
class LaneNode:
    lane_id: Any
    xyz: np.ndarray           # [N,3]
    length_m: float
    start_xy: np.ndarray
    end_xy: np.ndarray
    t_start: np.ndarray       # 2D unit tangent at start
    t_end: np.ndarray         # 2D unit tangent at end
    left_neighbors: List[Any]
    right_neighbors: List[Any]


# --------- main graph builder --------- #

def build_lane_graph(
    centerlines_xyz: Dict[Any, np.ndarray] | List[np.ndarray],
    *,
    # resampling for consistent tangents/orientation tests
    resample_n: int = 200,
    # successor (forward) linking
    successor_radius_m: float = 6.0,            # search radius from A.end to B.start
    successor_max_join_angle_deg: float = 35.0, # allowed heading diff A.end → B.start
    successor_project_tol_m: float = 4.0,       # allow attach to B within this arclength from its start
    # lateral neighbor linking
    lateral_search_radius_m: float = 4.0,       # for discovering left/right neighbors
    lateral_min_m: float = 2.0,                 # typical min lane spacing
    lateral_max_m: float = 6.0,                 # typical max lane spacing
    lateral_min_overlap_frac: float = 0.3,      # min fraction of A covered by valid lateral relation
    lateral_min_orient_cos: float = 0.85,       # require |cos(theta)| >= this along overlap
    # smoothing of stats
    tangent_smooth_window: int = 9,
) -> nx.DiGraph:
    """
    Build a directed lane graph from 3D centerlines.

    Nodes (lane_id) carry:
      - xyz [N,3], length_m, start_xy, end_xy, t_start, t_end
      - left_neighbors, right_neighbors (lists of lane_ids)

    Edges:
      - ('type'='successor', 'cost'=distance, 'angle_diff_deg'=..)
      - ('type'='lateral', 'side'='left'|'right', 'median_gap_m'=.., 'overlap_frac'=..)

    centerlines_xyz: either {lane_id: xyz} or list of xyz (ids auto-assigned 0..N-1)
    """
    # normalize input dict
    if isinstance(centerlines_xyz, dict):
        lane_items = list(centerlines_xyz.items())
    else:
        lane_items = list(enumerate(centerlines_xyz))

    # Precompute per-lane attributes
    nodes: Dict[Any, LaneNode] = {}
    geoms: Dict[Any, LineString] = {}
    starts, ends = [], []
    start_ids, end_ids = [], []

    for lane_id, xyz in lane_items:
        xyz = _to_xyz(xyz)
        if len(xyz) < 2:
            continue
        # resample to stabilize headings/orientation checks
        xyz_rs = _resample_by_count(xyz, resample_n)
        xy = xyz_rs[:, :2]
        t_start, t_end = _heading_at_endpoints(xy, k=max(3, resample_n//50))
        length_m = _polyline_length2d(xy)
        node = LaneNode(
            lane_id=lane_id,
            xyz=xyz_rs,
            length_m=length_m,
            start_xy=xy[0].copy(),
            end_xy=xy[-1].copy(),
            t_start=t_start.copy(),
            t_end=t_end.copy(),
            left_neighbors=[],
            right_neighbors=[],
        )
        nodes[lane_id] = node
        geoms[lane_id] = LineString(xy)
        starts.append(xy[0]); start_ids.append(lane_id)
        ends.append(xy[-1]);  end_ids.append(lane_id)

    # Build quick spatial indices for starts/ends and whole lanes
    start_tree = cKDTree(np.array(starts)) if starts else None
    end_tree   = cKDTree(np.array(ends)) if ends else None
    lane_list  = [geoms[i] for i in nodes.keys()]
    lane_ids   = list(nodes.keys())
    lane_tree  = STRtree(lane_list)

    G = nx.DiGraph()

    # Add nodes
    for lane_id, node in nodes.items():
        G.add_node(lane_id,
                   xyz=node.xyz,
                   length_m=node.length_m,
                   start_xy=node.start_xy,
                   end_xy=node.end_xy,
                   t_start=node.t_start,
                   t_end=node.t_end,
                   left_neighbors=[],
                   right_neighbors=[])

    # ---------- 1) Successor linking (forward connectivity) ----------
    if end_tree is not None and start_tree is not None:
        end_points = np.array(ends)
        for i, lane_id in enumerate(end_ids):
            nodeA = nodes[lane_id]
            # nearby starts within radius
            idxs = start_tree.query_ball_point(nodeA.end_xy, r=successor_radius_m)
            for j in idxs:
                succ_id = start_ids[j]
                if succ_id == lane_id:
                    continue
                nodeB = nodes[succ_id]
                # heading agreement
                if not _angle_ok(nodeA.t_end, nodeB.t_start, successor_max_join_angle_deg):
                    # allow loose attach to early arclength of B (e.g., if B doesn't start exactly at join)
                    # project A.end to B and require it's near B's start along arclength
                    p = nodeA.end_xy
                    q_xy, sB = _closest_point_on_ls(geoms[succ_id], p)
                    if sB > successor_project_tol_m:
                        continue
                    # also check tangent at that early arc
                    # approximate B start tangent
                    if not _angle_ok(nodeA.t_end, nodeB.t_start, successor_max_join_angle_deg + 10.0):
                        continue
                # cost = euclidean gap
                gap = float(np.linalg.norm(nodeB.start_xy - nodeA.end_xy))
                G.add_edge(lane_id, succ_id,
                           type="successor",
                           cost=gap,
                           angle_diff_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(nodeA.t_end, nodeB.t_start), -1, 1)))))
    # ---------- 2) Lateral neighbor linking (left/right) ----------
    # For each lane, probe points along it and search nearby lanes to evaluate lateral relation.
    probe_n = 60
    for lane_id, nodeA in nodes.items():
        xyA = nodeA.xyz[:, :2]
        tsA = _tangents2d(xyA, window=tangent_smooth_window)
        probes_idx = np.linspace(0, len(xyA)-1, probe_n).astype(int)
        pA = xyA[probes_idx]
        tA = tsA[probes_idx]

        # collect candidate lanes within search radius by STRtree
        # use the whole lines for a coarse spatial filter
        envelope = LineString(xyA).buffer(lateral_search_radius_m)
        cand_idxs = lane_tree.query(envelope)
        if not isinstance(cand_idxs, (list, tuple, np.ndarray)):
            cand_idxs = [cand_idxs]

        best_left = None   # (score, laneB_id, stats)
        best_right = None

        for idx in cand_idxs:
            laneB_geom = lane_list[idx]
            laneB_id   = lane_ids[idx]
            if laneB_id == lane_id:
                continue

            xyB = nodes[laneB_id].xyz[:, :2]
            # quick global orientation check (start/end tangents)
            orient_ok_global = (abs(np.dot(nodeA.t_start, nodes[laneB_id].t_start)) >= lateral_min_orient_cos) or \
                               (abs(np.dot(nodeA.t_end,   nodes[laneB_id].t_end))   >= lateral_min_orient_cos)
            if not orient_ok_global:
                continue

            # per-sample relation
            lateral_vals = []
            orient_vals  = []
            inside_vals  = []

            for p, t in zip(pA, tA):
                q, sB = _closest_point_on_ls(laneB_geom, p)
                # is q "near" p?
                if np.linalg.norm(q - p) > lateral_search_radius_m:
                    continue
                # signed lateral offset (+ left, - right)
                lat = _signed_lateral(q, p, t)
                # local orientation via dot of tangents at A sample and B local segment (approx via end tangent)
                # use B's nearest end tangent as proxy; cheap and okay if resampled
                tB_local = nodes[laneB_id].t_start if sB < laneB_geom.length/2 else nodes[laneB_id].t_end
                orient = abs(float(np.dot(t, tB_local)))
                lateral_vals.append(lat)
                orient_vals.append(orient)
                # consider "inside" if projection is not near B ends too extremely
                inside_vals.append(1.0 if (sB >= 0.0 and sB <= laneB_geom.length) else 0.0)

            if len(lateral_vals) < max(5, probe_n*0.15):
                continue

            lateral_vals = np.array(lateral_vals)
            orient_vals  = np.array(orient_vals)
            inside_vals  = np.array(inside_vals)

            # valid mask by lateral bounds + orientation
            valid_mask = (np.abs(lateral_vals) >= lateral_min_m) & (np.abs(lateral_vals) <= lateral_max_m) & \
                         (orient_vals >= lateral_min_orient_cos)

            overlap_frac = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
            if overlap_frac < lateral_min_overlap_frac:
                continue

            # choose side by sign of median lateral offset
            med_lat = float(np.median(lateral_vals[valid_mask]))
            side = 'left' if med_lat > 0 else 'right'
            score = abs(med_lat)  # prefer closest lateral neighbor

            stats = dict(
                median_gap_m=abs(med_lat),
                overlap_frac=overlap_frac,
                orient_med=float(np.median(orient_vals[valid_mask])) if np.any(valid_mask) else 0.0
            )

            if side == 'left':
                if (best_left is None) or (score < best_left[0]):
                    best_left = (score, laneB_id, stats)
            else:
                if (best_right is None) or (score < best_right[0]):
                    best_right = (score, laneB_id, stats)

        # commit best left/right
        if best_left:
            _, lid, stats = best_left
            nodes[lane_id].left_neighbors.append(lid)
            G.nodes[lane_id]['left_neighbors'].append(lid)
            G.add_edge(lane_id, lid, type="lateral", side="left",
                       median_gap_m=stats['median_gap_m'],
                       overlap_frac=stats['overlap_frac'])
            # make it reciprocal
            nodes[lid].right_neighbors.append(lane_id)
            G.nodes[lid]['right_neighbors'].append(lane_id)
            G.add_edge(lid, lane_id, type="lateral", side="right",
                       median_gap_m=stats['median_gap_m'],
                       overlap_frac=stats['overlap_frac'])

        if best_right:
            _, rid, stats = best_right
            nodes[lane_id].right_neighbors.append(rid)
            G.nodes[lane_id]['right_neighbors'].append(rid)
            G.add_edge(lane_id, rid, type="lateral", side="right",
                       median_gap_m=stats['median_gap_m'],
                       overlap_frac=stats['overlap_frac'])
            # reciprocal
            nodes[rid].left_neighbors.append(lane_id)
            G.nodes[rid]['left_neighbors'].append(lane_id)
            G.add_edge(rid, lane_id, type="lateral", side="left",
                       median_gap_m=stats['median_gap_m'],
                       overlap_frac=stats['overlap_frac'])

    return G


# --------- convenience: nearest lane & routing --------- #

def nearest_lane(center_graph: nx.DiGraph, xy: np.ndarray) -> Any:
    """Return lane_id of the node whose polyline is closest (in XY) to point xy."""
    xy = np.asarray(xy, dtype=float)
    best = None
    for lane_id, data in center_graph.nodes(data=True):
        ls = LineString(data['xyz'][:, :2])
        d = ls.distance(Point(float(xy[0]), float(xy[1])))
        if (best is None) or (d < best[0]):
            best = (d, lane_id)
    return best[1] if best else None

def route_lane_ids(center_graph: nx.DiGraph, start_xy: np.ndarray, goal_xy: np.ndarray) -> List[Any]:
    """
    Simple routing: pick nearest start/goal lanes, then shortest path over 'successor' edges.
    Edge weights use 'cost' for successors; lateral edges are disallowed in routing here.
    """
    s_lane = nearest_lane(center_graph, start_xy)
    g_lane = nearest_lane(center_graph, goal_xy)
    if s_lane is None or g_lane is None:
        return []

    # Build a view with only successor edges and weight=cost
    H = nx.DiGraph()
    for u, v, d in center_graph.edges(data=True):
        if d.get('type') == 'successor':
            H.add_edge(u, v, weight=float(d.get('cost', 1.0)))
    if s_lane not in H or g_lane not in H:
        # fall back to original graph if needed
        try:
            path = nx.shortest_path(center_graph, s_lane, g_lane)
            return path
        except Exception:
            return []

    try:
        path = nx.shortest_path(H, s_lane, g_lane, weight='weight')
        return path
    except Exception:
        return []

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _to_xy(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("xyz must be 2D array-like")
    if a.shape[1] == 3:
        return a[:, :2]
    if a.shape[1] == 2:
        return a
    if a.shape[0] == 3 and a.shape[1] != 3:
        return a.T[:, :2]
    raise ValueError("expected shape [N,3] or [N,2] or [3,N]")

def _mid_xy(xyz):
    xy = _to_xy(xyz)
    return xy[len(xy)//2]

def plot_lane_graph(
    G,
    *,
    show_ids: bool = True,
    draw_lateral: bool = True,
    draw_successor: bool = True,
    figsize=(10, 10),
    save_path: str | None = None,
    ax=None,
):
    """
    Plot lane centerlines + graph edges.
    - Nodes must have 'xyz' (polyline [N,3] or [N,2]).
    - Successor edges: edge attr 'type' == 'successor'
      (drawn with arrows from end of u to start of v).
    - Lateral edges:   edge attr 'type' == 'lateral'
      (drawn dashed between lane midpoints; drawn once per pair).

    Returns the matplotlib Axes.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # 1) Lane centerlines as a LineCollection (fast)
    lines = []
    for _, data in G.nodes(data=True):
        xy = _to_xy(data["xyz"])
        lines.append(xy)
    lc = LineCollection(lines, linewidths=2, alpha=0.9)
    ax.add_collection(lc)

    print(G.edges)

    # 2) Successor arrows (end(u) -> start(v))
    if draw_successor:
        for u, v, d in G.edges(data=True):
            if d.get("type") != "successor":
                continue
            su = np.asarray(G.nodes[u]["end_xy"], dtype=float)
            sv = np.asarray(G.nodes[v]["start_xy"], dtype=float)
            # Use annotate for arrowheads
            ax.annotate(
                "",
                xy=(sv[0], sv[1]),
                xytext=(su[0], su[1]),
                arrowprops=dict(arrowstyle="->", lw=1, alpha=0.8),
            )

    # 3) Lateral connectors (midpoint-to-midpoint, dashed; draw once per pair)
    if draw_lateral:
        seen = set()
        for u, v, d in G.edges(data=True):
            if d.get("type") != "lateral":
                continue
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            mu = _mid_xy(G.nodes[u]["xyz"])
            mv = _mid_xy(G.nodes[v]["xyz"])
            ax.plot([mu[0], mv[0]], [mu[1], mv[1]], linestyle="--", linewidth=1, alpha=0.6)

    # 4) Optional lane-id labels at each lane midpoint
    if show_ids:
        for nid, data in G.nodes(data=True):
            c = _mid_xy(data["xyz"])
            ax.text(c[0], c[1], str(nid), fontsize=8)

    # Ax cosmetics
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.margins(0.05)
    plt.show()
    # if created_fig:
    #     fig.tight_layout()
    #     if save_path:
    #         fig.savefig(save_path, dpi=200)
    return ax