# lane_graph_with_connectors.py
# Requires: pip install numpy shapely networkx scipy

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Any
import numpy as np
import networkx as nx
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy.signal import savgol_filter


# ----------------------------- Helpers ----------------------------- #

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("need [N,3]/[N,2]/[3,N]")
    return a

def _smooth_polyline_nd(arr: np.ndarray, window=21, poly=3) -> np.ndarray:
    arr = np.asarray(arr, float)
    if window < 3 or len(arr) < window: return arr
    if window % 2 == 0: window += 1
    out = []
    for d in range(arr.shape[1]):
        out.append(savgol_filter(arr[:, d], window, poly, mode="interp"))
    return np.stack(out, axis=1)

def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _tangents2d(xy: np.ndarray, window=9, poly=2) -> np.ndarray:
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    if len(xy) >= window and window >= 3:
        if window % 2 == 0: window += 1
        v = savgol_filter(v, window, poly, axis=0, mode="interp")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _poly_sample_at_s(xyz: np.ndarray, s_query: float) -> np.ndarray:
    s = _arclength2d(xyz[:, :2])
    x = np.interp(s_query, s, xyz[:, 0])
    y = np.interp(s_query, s, xyz[:, 1])
    z = np.interp(s_query, s, xyz[:, 2])
    return np.array([x, y, z], float)

def _heading_vec_end(xyz: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit tangent at start and end (2D projected)"""
    xy = xyz[:, :2]
    k = max(1, min(k, len(xy)-1))
    t0 = xy[min(k, len(xy)-1)] - xy[0]
    t1 = xy[-1] - xy[max(len(xy)-1-k, 0)]
    for t in (t0, t1):
        n = np.linalg.norm(t) + 1e-9
        t /= n
    return t0, t1

def _angle_diff_deg(u: np.ndarray, v: np.ndarray) -> float:
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

def _bezier_cubic(p0, p1, t0, t1, len0=8.0, len1=8.0, n=30) -> np.ndarray:
    """Cubic Bézier by endpoint tangents; returns [n,3]"""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    t0 = np.asarray(t0, float); t1 = np.asarray(t1, float)
    t0_3 = np.array([t0[0], t0[1], 0.0]); t1_3 = np.array([t1[0], t1[1], 0.0])
    c0 = p0 + len0 * t0_3
    c1 = p1 - len1 * t1_3
    ts = np.linspace(0.0, 1.0, n)
    B = (1-ts)[:,None]**3 * p0 + \
        3*(1-ts)[:,None]**2*ts[:,None]*c0 + \
        3*(1-ts)[:,None]*ts[:,None]**2*c1 + \
        ts[:,None]**3 * p1
    return B

def _as_index_list(idxs) -> List[int]:
    """Normalize STRtree.query result to a flat list of Python ints."""
    import numpy as _np
    if idxs is None:
        return []
    if isinstance(idxs, _np.ndarray):
        return [int(i) for i in idxs.ravel().tolist()]
    if isinstance(idxs, (list, tuple)):
        return [int(i) for i in idxs]
    return [int(idxs)]


# -------------------- Input dataclass expected -------------------- #

@dataclass
class CenterlineInput:
    """Minimal info needed from your centerline builder (3D)."""
    boundary_a_id: int
    boundary_b_id: int
    side: str                    # 'left' | 'right' (relative to A)
    lane_index: int              # 0..lane_count-1 within its corridor/segment
    lane_count: int
    centerline: np.ndarray       # [N,3] polyline (already reasonably smooth)
    mean_gap_m: float            # median XY corridor width (for metadata)
    parallelism: float           # |cos| mean for reference
    group_key: Tuple[int,int,str,int] | None = None  # (A,B,side,lane_count)


# ------------------------ Graph builder ------------------------ #

@dataclass
class LaneEdge:
    edge_id: str
    lane_uid: str
    geom: np.ndarray        # [N,3]
    kind: str               # 'lane' | 'connector'
    from_node: str
    to_node: str
    attrs: Dict[str, Any]

def build_lane_graph_with_connectors(
    centerlines: List[CenterlineInput],
    *,
    # smoothing
    smooth_window: int = 17,
    smooth_poly: int = 3,
    # longitudinal linking thresholds
    max_longitudinal_snap_m: float = 40.0,
    max_longitudinal_heading_diff_deg: float = 35.0,
    forward_only: bool = True,
    forward_min_m: float = 0.0,
    forward_lateral_tol_m: float = 20.0,
    # lateral (lane-change) connectors: ALWAYS connect adjacent lanes within group
    enable_lateral: bool = True,
    lateral_anchor_count: int = 3,            # number of connectors along overlap
    lateral_anchor_margin_frac: float = 0.1,  # keep anchors within (10%, 90%) of overlap
    lateral_curve_len_m: float = 12.0,        # Bézier handle length
    # turning connectors
    enable_turning: bool = True,
    turn_snap_radius_m: float = 18.0,
    turn_bezier_len_m: float = 10.0,
    turn_min_angle_deg: float = 45.0,
    turn_max_angle_deg: float = 145.0,
    allow_uturn: bool = False,
) -> nx.DiGraph:
    """
    Build a directed lane graph with:
      - lane edges per centerline segment
      - longitudinal connectors between consecutive segments of same lane (forward-only)
      - **lateral connectors between every adjacent lane pair in the same group**
      - turning connectors between different groups at junctions (left/right classification)

    Returns a NetworkX DiGraph with node 'xyz' and edge attributes including:
      'kind' in {'lane','connector'}, 'subtype' for connectors, and 'geom' polyline [M,3].
    """
    G = nx.DiGraph()

    # 1) Normalize & smooth centerlines; create lane edges
    lane_edges: List[LaneEdge] = []
    for i, c in enumerate(centerlines):
        geom = _to_xyz(c.centerline)
        if smooth_window and len(geom) >= smooth_window:
            geom = _smooth_polyline_nd(geom, window=smooth_window, poly=smooth_poly)

        # Use lane_count in group key to avoid crossing tapers (4->3, etc.)
        gkey = c.group_key #or (c.boundary_a_id, c.boundary_b_id, c.side, int(c.lane_count))
        lane_uid = f"corridor:{c.boundary_a_id}-{c.boundary_b_id}:{c.side}:lane{c.lane_index:02d}"
        edge_id = f"lane_{i:05d}"
        n0 = f"node_{edge_id}_start"
        n1 = f"node_{edge_id}_end"

        G.add_node(n0, xyz=geom[0])
        G.add_node(n1, xyz=geom[-1])

        lane_edges.append(LaneEdge(
            edge_id=edge_id,
            lane_uid=lane_uid,
            geom=geom,
            kind="lane",
            from_node=n0,
            to_node=n1,
            attrs=dict(
                type="lane",
                boundary_a_id=c.boundary_a_id,
                boundary_b_id=c.boundary_b_id,
                side=c.side,
                lane_index=c.lane_index,
                lane_count=c.lane_count,
                mean_gap_m=c.mean_gap_m,
                parallelism=c.parallelism,
                group_key=gkey,
            )
        ))

    # Add lane edges to graph
    for e in lane_edges:
        G.add_edge(e.from_node, e.to_node, id=e.edge_id, kind=e.kind, geom=e.geom, **e.attrs)

    # Spatial indices for starts/ends
    start_points = [(e.geom[0, :2], e) for e in lane_edges]
    end_points   = [(e.geom[-1, :2], e) for e in lane_edges]
    tree_start = STRtree([Point(p) for p, _ in start_points])
    tree_end   = STRtree([Point(p) for p, _ in end_points])

    # 2) Longitudinal connectors (forward-only continuation)
    for e in lane_edges:
        _, t_end = _heading_vec_end(e.geom)
        p_end_xy = e.geom[-1, :2]
        p_end_geom = Point(p_end_xy)

        idxs = tree_start.query(p_end_geom.buffer(max_longitudinal_snap_m))
        for idx in _as_index_list(idxs):
            p_start_xy, e2 = start_points[idx]
            if e2.edge_id == e.edge_id:
                continue

            d = np.asarray(p_start_xy, float) - np.asarray(p_end_xy, float)
            along = float(d[0]*t_end[0] + d[1]*t_end[1])
            lateral = float(d[0]*(-t_end[1]) + d[1]*t_end[0])

            if forward_only:
                if not (along >= forward_min_m and along <= max_longitudinal_snap_m):
                    continue
                if abs(lateral) > forward_lateral_tol_m:
                    continue
            else:
                if np.hypot(*d) > max_longitudinal_snap_m:
                    continue

            u = t_end
            v, _ = _heading_vec_end(e2.geom)  # heading at start of e2
            ang = _angle_diff_deg(u, v)
            if ang > max_longitudinal_heading_diff_deg:
                continue

            p0 = e.geom[-1]
            p1 = e2.geom[0]
            bez = _bezier_cubic(p0, p1, u, v, len0=6.0, len1=6.0, n=20)
            cn_id = f"conn_long_{e.edge_id}_to_{e2.edge_id}"
            G.add_edge(e.to_node, e2.from_node,
                       id=cn_id, kind="connector", geom=bez,
                       type="connector", subtype="longitudinal",
                       from_lane=e.lane_uid, to_lane=e2.lane_uid,
                       forward_along_m=along, forward_lateral_m=lateral)

    # 3) Lateral lane-change connectors: **connect ALL adjacent lanes in the same group**
    if enable_lateral:
        by_group: Dict[Tuple[int,int,str,int], List[LaneEdge]] = {}
        for e in lane_edges:
            by_group.setdefault(e.attrs['group_key'], []).append(e)

        for group_key, edges in by_group.items():
            if len(edges) < 2:
                continue
            # sort inside group by lane_index and connect neighbors (index diff == 1)
            edges_sorted = sorted(edges, key=lambda x: x.attrs['lane_index'])

            for eL, eR in zip(edges_sorted, edges_sorted[1:]):
                # Only adjacent lane indices
                if abs(int(eL.attrs['lane_index']) - int(eR.attrs['lane_index'])) != 1:
                    continue

                a_xy = eL.geom[:, :2]
                b_xy = eR.geom[:, :2]
                sA = _arclength2d(a_xy)
                sB = _arclength2d(b_xy)
                overlap = min(sA[-1], sB[-1])

                # Choose evenly spaced anchors within (margin, 1-margin) of the overlap
                if overlap <= 0:
                    continue
                m0 = max(0.0, float(lateral_anchor_margin_frac))
                m1 = 1.0 - m0
                n_anchors = max(1, int(lateral_anchor_count))
                anchors = np.linspace(m0*overlap, m1*overlap, n_anchors)

                # tangents for orientation along lanes
                tA = _tangents2d(a_xy)
                tB = _tangents2d(b_xy)

                for ak in anchors:
                    pA = _poly_sample_at_s(eL.geom, ak)
                    pB = _poly_sample_at_s(eR.geom, ak)
                    # local tangents by nearest index
                    iA = max(0, np.searchsorted(sA, ak)-1)
                    iB = max(0, np.searchsorted(sB, ak)-1)
                    u = tA[min(iA, len(tA)-1)]
                    v = tB[min(iB, len(tB)-1)]

                    # build short, smooth Bézier; no filtering — always connect
                    bez = _bezier_cubic(
                        pA, pB, u, v,
                        len0=lateral_curve_len_m,
                        len1=lateral_curve_len_m,
                        n=25
                    )
                    # add both directions
                    cn_id = f"conn_lat_{eL.edge_id}_to_{eR.edge_id}_s{int(round(ak))}"
                    G.add_edge(eL.from_node, eR.from_node,
                               id=cn_id, kind="connector", geom=bez,
                               type="connector", subtype="lateral",
                               from_lane=eL.lane_uid, to_lane=eR.lane_uid,
                               anchor_s=float(ak))

                    cn_id2 = f"conn_lat_{eR.edge_id}_to_{eL.edge_id}_s{int(round(ak))}"
                    G.add_edge(eR.from_node, eL.from_node,
                               id=cn_id2, kind="connector", geom=bez[::-1].copy(),
                               type="connector", subtype="lateral",
                               from_lane=eR.lane_uid, to_lane=eL.lane_uid,
                               anchor_s=float(ak))

    # 4) Turning connectors (left/right) between different groups
    if enable_turning:
        min_ang = 0.0 if allow_uturn else turn_min_angle_deg
        max_ang = 180.0 if allow_uturn else turn_max_angle_deg

        for e in lane_edges:
            p_end = e.geom[-1]
            u_end = _heading_vec_end(e.geom)[1]
            p_end_xy = p_end[:2]
            neighborhood = tree_start.query(Point(p_end_xy).buffer(turn_snap_radius_m))

            for idx in _as_index_list(neighborhood):
                p_start_xy, tgt = start_points[idx]
                if tgt.edge_id == e.edge_id:
                    continue
                # only treat as "turn" if target is a different group (incl. different lane_count)
                if tgt.attrs['group_key'] == e.attrs['group_key']:
                    continue

                d = np.asarray(p_start_xy, float) - np.asarray(p_end_xy, float)
                dist = float(np.hypot(d[0], d[1]))
                if dist <= 0.1 or dist > turn_snap_radius_m:
                    continue

                v_start, _ = _heading_vec_end(tgt.geom)
                ang = _angle_diff_deg(u_end, v_start)
                if not (min_ang <= ang <= max_ang):
                    continue

                cross_z = u_end[0]*d[1] - u_end[1]*d[0]
                turn_type = 'turn_left' if cross_z > 0 else 'turn_right'

                bez = _bezier_cubic(
                    p_end, tgt.geom[0], u_end, v_start,
                    len0=turn_bezier_len_m, len1=turn_bezier_len_m, n=30
                )
                cn_id = f"conn_turn_{turn_type}_{e.edge_id}_to_{tgt.edge_id}"
                G.add_edge(e.to_node, tgt.from_node,
                           id=cn_id, kind="connector", geom=bez,
                           type="connector", subtype=turn_type,
                           from_lane=e.lane_uid, to_lane=tgt.lane_uid,
                           turn_angle_deg=float(ang), turn_dist_m=float(dist))

    return G


# ----------------------------- Simple plot ----------------------------- #
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

def plot_lane_graph(G: nx.DiGraph, show_nodes=True, figsize=(10,10)):
    """Top-down XY plot of lanes and connectors."""
    fig, ax = plt.subplots(figsize=figsize)

    for u, v, data in G.edges(data=True):
        geom = data.get('geom')
        if geom is None:
            continue
        xy = np.asarray(geom)[:, :2]
        kind = data.get('kind', 'lane')
        if kind == 'lane':
            ax.plot(xy[:,0], xy[:,1], color='blue', linewidth=2, alpha=0.75, zorder=1)
        else:
            subtype = data.get('subtype', 'connector')
            if subtype == 'longitudinal':
                ax.plot(xy[:,0], xy[:,1], color='green', linestyle='--', linewidth=1.6, alpha=0.9, zorder=2)
            elif subtype == 'lateral':
                ax.plot(xy[:,0], xy[:,1], color='orange', linestyle='--', linewidth=1.6, alpha=0.9, zorder=2)
            elif subtype in ('turn_left','turn_right'):
                ax.plot(xy[:,0], xy[:,1], color='purple', linestyle='-.', linewidth=1.8, alpha=0.95, zorder=3)
            else:
                ax.plot(xy[:,0], xy[:,1], color='gray', linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)

    if show_nodes:
        for n, data in G.nodes(data=True):
            p = data.get('xyz')
            if p is not None:
                ax.scatter(p[0], p[1], color='black', s=9, zorder=4)

    ax.set_aspect('equal')
    ax.set_title("Lane Graph with Longitudinal, Lateral, and Turning Connectors")
    plt.show()