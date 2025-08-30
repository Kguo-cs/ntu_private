# edge_graph_topo_merge.py
# Merge junction nodes using boundary coordinates & lane connections (not distance),
# then build a SUMO-like edge graph with mutual lane-change grouping.
# Requires: numpy, networkx, shapely, scipy

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

# ---------- small array helpers ----------
def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("array must be [N,3],[N,2],or[3,N]")
    return a

def _s_arclen_xy(xyz: np.ndarray) -> np.ndarray:
    xy = xyz[:, :2]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _closest_on_ls_s(ls: LineString, p_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    s = ls.project(Point(float(p_xy[0]), float(p_xy[1])))
    q = ls.interpolate(s)
    return np.array([q.x, q.y], dtype=float), float(s)

# ---------- boundary cache & projection ----------
@dataclass
class BoundaryRec:
    id: int
    geom: LineString
    s_len: float

def _build_boundary_cache(boundary_dict: Dict[int, np.ndarray]) -> Tuple[Dict[int, BoundaryRec], STRtree]:
    recs: List[BoundaryRec] = []
    for bid, arr in boundary_dict.items():
        xy = _to_xyz(arr)[:, :2]
        if len(xy) < 2: continue
        ls = LineString(xy)
        recs.append(BoundaryRec(int(bid), ls, ls.length))
    tree = STRtree([r.geom for r in recs])
    # id->rec (preserve mapping by same index order used to build STRtree)
    id2rec = {rec.id: rec for rec in recs}
    return id2rec, tree

def _boundary_sbar_for_endpoint(p_xy: np.ndarray, pair: Tuple[int,int], id2rec: Dict[int, BoundaryRec]) -> float:
    """Return mean corridor coordinate s̄ = 0.5*(sA+sB) for endpoint p on boundary pair."""
    a, b = pair
    recA = id2rec.get(a); recB = id2rec.get(b)
    if (recA is None) or (recB is None):
        return float("nan")
    _, sA = _closest_on_ls_s(recA.geom, p_xy)
    _, sB = _closest_on_ls_s(recB.geom, p_xy)
    return 0.5 * (sA + sB)

# ---------- DSU ----------
class DSU:
    def __init__(self, n:int):
        self.p=list(range(n)); self.r=[0]*n
    def find(self,x:int)->int:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a:int,b:int):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: ra, rb = rb, ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1

# ---------- lane snap (for routing later) ----------
@dataclass
class LaneGeom:
    lane_id: Any
    xyz: np.ndarray
    s: np.ndarray
    ls: LineString
    length: float

class LaneSnapIndex:
    def __init__(self, G_lane: nx.DiGraph):
        self.lanes: Dict[Any, LaneGeom] = {}
        for lid, data in G_lane.nodes(data=True):
            xyz = _to_xyz(data["xyz"])
            s = _s_arclen_xy(xyz)
            self.lanes[lid] = LaneGeom(lid, xyz, s, LineString(xyz[:, :2]), float(s[-1]))

    def nearest_lane(self, xy: np.ndarray) -> Tuple[Any, float, np.ndarray]:
        xy = np.asarray(xy, float)
        best = None
        for lid, L in self.lanes.items():
            _, s_on = _closest_on_ls_s(L.ls, xy)
            q = np.array([np.interp(s_on, L.s, L.xyz[:,0]),
                          np.interp(s_on, L.s, L.xyz[:,1]),
                          np.interp(s_on, L.s, L.xyz[:,2])])
            d = np.linalg.norm(q[:2] - xy[:2])
            if (best is None) or (d < best[0]):
                best = (d, lid, s_on, q)
        _, lid, s_on, q = best
        return lid, s_on, q

# ---------- build: boundary/lane-connection merged nodes + mutual-lane-change edges ----------
@dataclass
class BuildResult:
    EG: nx.DiGraph
    lane_to_edge: Dict[Any, str]
    node_positions: Dict[str, np.ndarray]

def build_edge_graph_from_lane_graph_topo(
    G_lane: nx.DiGraph,
    boundary_dict: Dict[int, np.ndarray],
    *,
    s_merge_tol_m: float = 2.0,             # along-corridor tolerance on s̄ for merging
    lateral_require_bidir: bool = True,     # mutual lane-change to be in same edge
    lateral_min_overlap_frac: float = 0.25, # if present on edge, enforce minimum
    representative: str = "median",         # representative lane per edge
    edge_id_prefix: str = "e",
) -> BuildResult:
    """
    Build a SUMO-like edge graph where:
      - Junction nodes are merged using boundary arclength coordinates (s̄) and successor links.
      - Inside each (u->v) junction pair, lanes are grouped into edges by mutual lane-changing.
    """
    # Boundary cache
    id2rec, _ = _build_boundary_cache(boundary_dict)

    # Collect endpoints with boundary-pair curvilinear coordinate
    endpoints: List[Tuple[Any, str, np.ndarray, Tuple[int,int], float]] = []
    # format: (lane_id, 'start'|'end', xy, boundary_pair, sbar)
    for lid, data in G_lane.nodes(data=True):
        bp = data.get("boundary_pair")
        if not bp or len(bp)!=2:   # skip if no boundary pair
            continue
        bp = tuple(sorted(tuple(bp)))
        sxy = np.asarray(data["start_xy"], float)
        exy = np.asarray(data["end_xy"], float)
        sbar_s = _boundary_sbar_for_endpoint(sxy, bp, id2rec)
        sbar_e = _boundary_sbar_for_endpoint(exy, bp, id2rec)
        endpoints.append((lid, "start", sxy, bp, sbar_s))
        endpoints.append((lid, "end",   exy, bp, sbar_e))

    if not endpoints:
        return BuildResult(nx.DiGraph(), {}, {})

    # Build DSU over endpoints (index in 'endpoints' list)
    dsu = DSU(len(endpoints))

    # Rule 1: merge by boundary curvilinear coordinate s̄ (same boundary_pair)
    # Group indices by boundary_pair
    by_bp: Dict[Tuple[int,int], List[int]] = {}
    for i, (_lid, _typ, _xy, bp, sbar) in enumerate(endpoints):
        by_bp.setdefault(bp, []).append(i)
    for bp, inds in by_bp.items():
        # Sort by s̄; union neighbors within tolerance
        inds_sorted = sorted(inds, key=lambda i: endpoints[i][4])
        for i0, i1 in zip(inds_sorted[:-1], inds_sorted[1:]):
            s0 = endpoints[i0][4]; s1 = endpoints[i1][4]
            if (np.isfinite(s0) and np.isfinite(s1)) and (abs(s1 - s0) <= s_merge_tol_m):
                dsu.union(i0, i1)

    # Rule 2: force-merge lane connections (successor)
    # For any successor u->v, union end(u) with start(v)
    # Build quick maps from lid to its endpoint indices
    start_idx: Dict[Any, int] = {}
    end_idx: Dict[Any, int] = {}
    for idx, (lid, typ, *_rest) in enumerate(endpoints):
        if typ == "start": start_idx[lid] = idx
        else:              end_idx[lid]   = idx
    for u, v, d in G_lane.edges(data=True):
        if d.get("type") != "successor": continue
        iu = end_idx.get(u); iv = start_idx.get(v)
        if iu is not None and iv is not None:
            dsu.union(iu, iv)

    # Turn DSU clusters into junction nodes with positions (average of members)
    clusters: Dict[int, List[int]] = {}
    for i in range(len(endpoints)):
        clusters.setdefault(dsu.find(i), []).append(i)

    node_positions: Dict[str, np.ndarray] = {}
    idx2nid: Dict[int, str] = {}
    for comp in clusters.values():
        pts = np.array([endpoints[i][2] for i in comp], dtype=float)
        nid = f"J{len(node_positions)}"
        node_positions[nid] = pts.mean(axis=0)
        for i in comp: idx2nid[i] = nid

    def node_of(lid, which):
        # find endpoint tuple index and map via idx2nid
        for idx, (ll, typ, *_rest) in enumerate(endpoints):
            if ll == lid and typ == which:
                return idx2nid[idx]
        raise KeyError("endpoint not found")

    # Assign each lane to a directed (u,v) node pair
    lanes_by_uv: Dict[Tuple[str,str], List[Any]] = {}
    for lid, _data in G_lane.nodes(data=True):
        # skip lanes that were not in endpoints (e.g., missing boundary_pair)
        if lid not in start_idx or lid not in end_idx:
            continue
        u = node_of(lid, "start"); v = node_of(lid, "end")
        lanes_by_uv.setdefault((u,v), []).append(lid)

    # Build EG
    EG = nx.DiGraph()
    for nid, xy in node_positions.items():
        EG.add_node(nid, xy=np.asarray(xy, float))

    lane_to_edge: Dict[Any, str] = {}
    comp_counter_by_uv: Dict[Tuple[str,str], int] = {}

    # Inside each (u,v), group lanes by mutual lane-change (bidirectional lateral)
    def _mutual_ok(i, j) -> bool:
        eij = G_lane.get_edge_data(i, j); eji = G_lane.get_edge_data(j, i)
        if eij is None or eji is None:
            return False if lateral_require_bidir else (eij is not None or eji is not None)
        if eij.get("type") != "lateral" or eji.get("type") != "lateral":
            return False
        # optional overlap gating if present
        oi = eij.get("overlap_frac"); oj = eji.get("overlap_frac")
        if (oi is not None and oi < lateral_min_overlap_frac): return False
        if (oj is not None and oj < lateral_min_overlap_frac): return False
        return True

    for (u, v), lane_ids in lanes_by_uv.items():
        if not lane_ids: continue
        H = nx.Graph(); H.add_nodes_from(lane_ids)
        for i in range(len(lane_ids)):
            for j in range(i+1, len(lane_ids)):
                a, b = lane_ids[i], lane_ids[j]
                if _mutual_ok(a, b):
                    H.add_edge(a, b)
        comps = list(nx.connected_components(H)) if H.number_of_edges()>0 else [{lid} for lid in lane_ids]

        for comp in comps:
            comp = sorted(list(comp), key=lambda L: G_lane.nodes[L].get("lane_index", 0))
            # representative lane shape
            rep_lane = comp[0]
            if representative == "median":
                if all(G_lane.nodes[L].get("lane_index") is not None for L in comp):
                    rep_lane = comp[len(comp)//2]
            shape_xyz = _to_xyz(G_lane.nodes[rep_lane]["xyz"])
            # edge length = mean of member lane lengths
            lengths = []
            for lid in comp:
                s = _s_arclen_xy(_to_xyz(G_lane.nodes[lid]["xyz"]))
                lengths.append(float(s[-1]))
            length_m = float(np.mean(lengths)) if lengths else float(_s_arclen_xy(shape_xyz)[-1])

            k = comp_counter_by_uv.get((u, v), 0)
            eid = f"{edge_id_prefix}_{u}_{v}_{k}"
            comp_counter_by_uv[(u, v)] = k + 1

            EG.add_edge(u, v,
                        id=eid,
                        lanes=tuple(comp),
                        num_lanes=len(comp),
                        length_m=length_m,
                        shape_xyz=shape_xyz,
                        weight=length_m)
            for lid in comp:
                lane_to_edge[lid] = eid

    # Allowed edge turns: derive from lane successors
    allowed_edge_turns = set()
    for lu, lv, d in G_lane.edges(data=True):
        if d.get("type") != "successor": continue
        eu = lane_to_edge.get(lu); ev = lane_to_edge.get(lv)
        if eu and ev:
            allowed_edge_turns.add((eu, ev))
    EG.graph["allowed_edge_turns"] = allowed_edge_turns

    return BuildResult(EG=EG, lane_to_edge=lane_to_edge, node_positions=node_positions)

# ---------- Router on the edge graph (unchanged API) ----------
@dataclass
class EdgeRoute:
    edge_ids: List[str]
    distance_m: float
    route_xyz: Optional[np.ndarray]
    start_edge: str
    goal_edge: str

def route_on_edge_graph(
    EG: nx.DiGraph,
    G_lane: nx.DiGraph,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    weight_attr: str = "weight",
    build_geometry: bool = True,
) -> EdgeRoute:
    shape_of_e, length_of_e, lane_to_edge = {}, {}, {}
    id2uv = {}
    for u, v, ed in EG.edges(data=True):
        eid = ed["id"]
        id2uv[eid] = (u, v)
        shape_of_e[eid] = _to_xyz(ed["shape_xyz"])
        length_of_e[eid] = float(ed.get("length_m", _s_arclen_xy(shape_of_e[eid])[-1]))
        for lid in ed["lanes"]:
            lane_to_edge[lid] = eid

    # snap start/goal to nearest lane → edge
    snap = LaneSnapIndex(G_lane)
    s_lane, _, _ = snap.nearest_lane(np.asarray(start_xy, float))
    g_lane, _, _ = snap.nearest_lane(np.asarray(goal_xy, float))
    s_edge = lane_to_edge.get(s_lane); g_edge = lane_to_edge.get(g_lane)
    if s_edge is None or g_edge is None:
        raise RuntimeError("Could not map snapped lanes to edges.")

    # edge-ID graph using allowed turns
    EID = nx.DiGraph()
    for eid in shape_of_e.keys(): EID.add_node(eid)
    allowed = EG.graph.get("allowed_edge_turns", None)
    if allowed:
        for e1, e2 in allowed:
            if (e1 in shape_of_e) and (e2 in shape_of_e):
                # cost = weight of the next edge
                w = None
                for u, v, ed in EG.edges(data=True):
                    if ed["id"] == e2:
                        w = float(ed.get(weight_attr, ed.get("weight"))); break
                if w is None: w = float(length_of_e[e2])
                EID.add_edge(e1, e2, weight=w)
    else:
        # fallback: connect edges sharing a junction
        for e1, (u1, v1) in id2uv.items():
            for e2, (u2, _) in id2uv.items():
                if v1 == u2:
                    w = None
                    for u, v, ed in EG.edges(data=True):
                        if ed["id"] == e2:
                            w = float(ed.get(weight_attr, ed.get("weight"))); break
                    if w is None: w = float(length_of_e[e2])
                    EID.add_edge(e1, e2, weight=w)

    edge_path = nx.shortest_path(EID, s_edge, g_edge, weight="weight")

    dist = float(sum(length_of_e[eid] for eid in edge_path))
    route_xyz = None
    if build_geometry:
        parts = []
        for i, eid in enumerate(edge_path):
            P = shape_of_e[eid]
            if i>0 and len(parts) and np.allclose(parts[-1][-1], P[0], atol=1e-6):
                parts[-1] = parts[-1][:-1]
            parts.append(P)
        route_xyz = np.vstack(parts)

    return EdgeRoute(edge_ids=edge_path, distance_m=dist, route_xyz=route_xyz,
                     start_edge=s_edge, goal_edge=g_edge)

# plot_edge_graph.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _to_xy(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[1] == 3: return a[:, :2]
    if a.shape[1] == 2: return a
    if a.shape[0] == 3 and a.shape[1] != 3: return a.T[:, :2]
    raise ValueError("expected [N,3] or [N,2] or [3,N]")

def _midpoint(xy):
    return xy[len(xy)//2]

def plot_edge_graph(
    EG,
    *,
    route_edge_ids=None,          # list like ["e0","e3","e9"] to overlay a route (optional)
    show_edge_ids: bool=True,
    show_junctions: bool=True,
    show_junction_ids: bool=True,
    base_lw: float=1.6,           # base linewidth
    lanes_gain: float=0.5,        # extra lw per lane beyond 1
    arrows: bool=True,            # tiny arrowheads to indicate edge direction
    figsize=(11,11),
    save_path: str|None=None,
    ax=None,
):
    """Plot a SUMO-like edge graph built by `build_edge_graph_from_lane_graph`."""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # 1) Draw edges (LineCollection for speed)
    segs, lws, cols, mids, labels = [], [], [], [], []
    for u, v, ed in EG.edges(data=True):
        xy = _to_xy(ed["shape_xyz"])
        segs.append(xy)
        nlanes = int(ed.get("num_lanes", 1))
        lws.append(base_lw + lanes_gain * max(0, nlanes - 1))
        cols.append("C0")  # single hue; keep clean. (Change if you want per-edge colors.)
        mids.append(_midpoint(xy))
        labels.append(ed.get("id", f"{u}->{v}"))

    if segs:
        lc = LineCollection(segs, linewidths=lws, colors=cols, alpha=0.95, zorder=2)
        ax.add_collection(lc)

    # 2) Direction arrowheads (small, at 60% along each edge)
    if arrows:
        for xy in segs:
            if len(xy) < 2: continue
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            if s[-1] <= 0: continue
            t = 0.6 * s[-1]
            # linear interpolation to position and local tangent
            px = np.interp(t, s, xy[:,0]); py = np.interp(t, s, xy[:,1])
            # tangent via finite diff
            dt = min(max(0.02*s[-1], 1e-3), s[-1]/3)
            p0x = np.interp(max(0,t-dt), s, xy[:,0]); p0y = np.interp(max(0,t-dt), s, xy[:,1])
            p1x = np.interp(min(s[-1],t+dt), s, xy[:,0]); p1y = np.interp(min(s[-1],t+dt), s, xy[:,1])
            ax.annotate("", xy=(p1x,p1y), xytext=(p0x,p0y),
                        arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.75), zorder=3)

    # 3) Edge ID labels (at midpoints)
    if show_edge_ids:
        for m, lab in zip(mids, labels):
            ax.text(m[0], m[1], lab, fontsize=8, color="k", alpha=0.85)

    # 4) Junctions
    if show_junctions:
        xs, ys, nids = [], [], []
        for nid, nd in EG.nodes(data=True):
            if "xy" not in nd: continue
            p = np.asarray(nd["xy"], float)
            xs.append(p[0]); ys.append(p[1]); nids.append(nid)
        if xs:
            ax.scatter(xs, ys, s=10, c="k", alpha=0.8, zorder=4)
            if show_junction_ids:
                for x, y, nid in zip(xs, ys, nids):
                    ax.text(x, y, str(nid), fontsize=7, color="k", ha="left", va="bottom")

    # 5) Optional route overlay (thicker)
    if route_edge_ids:
        # build a polyline by concatenating edge shapes in sequence
        parts = []
        for i, eid in enumerate(route_edge_ids):
            # find the EG edge with this id
            found = False
            for u, v, ed in EG.edges(data=True):
                if ed.get("id") == eid:
                    P = _to_xy(ed["shape_xyz"])
                    if i>0 and len(parts) and np.allclose(parts[-1][-1], P[0], atol=1e-6):
                        parts[-1] = parts[-1][:-1]
                    parts.append(P); found = True; break
            if not found:
                # silently skip unknown id
                continue
        if parts:
            R = np.vstack(parts)
            ax.plot(R[:,0], R[:,1], linewidth=3.0, alpha=0.95, zorder=5)

    # Ax cosmetics
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.margins(0.05)
    plt.show()
    # if created:
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=220)
    return ax

# # --- tiny demo ---
# if __name__ == "__main__":
#     import networkx as nx
#     EG = nx.DiGraph()
#     # nodes with positions
#     EG.add_node("J0", xy=np.array([0,0])); EG.add_node("J1", xy=np.array([50,0])); EG.add_node("J2", xy=np.array([50,30]))
#     # edges with shapes and lane counts
#     EG.add_edge("J0","J1", id="e0", num_lanes=2, shape_xyz=np.array([[0,0,0],[50,0,0]]))
#     EG.add_edge("J1","J2", id="e1", num_lanes=1, shape_xyz=np.array([[50,0,0],[50,30,0]]))
#     plot_edge_graph(EG, route_edge_ids=["e0","e1"], show_edge_ids=True, show_junctions=True, show_junction_ids=True)
#     plt.show()
