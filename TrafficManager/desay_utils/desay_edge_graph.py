# desay_edge_graph.py
# Build a SUMO-like edge graph from a lane graph:
# - Merge nodes by connector relations (topology).
# - Also merge the set of lane endpoints that belong to the same group_key.
# - Merge lanes into edges by group_key (one edge per (uJ->vJ, group_key)).

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import networkx as nx


# ---------- helpers ----------

def _to_xyz(arr) -> np.ndarray:
    """Normalize to [N,3]. Accepts [N,3], [N,2], [3,N], [x,y,z], [x,y]."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        if a.shape[0] == 2: return np.array([[a[0], a[1], 0.0]], float)
        if a.shape[0] == 3: return a.reshape(1, 3)
        raise ValueError("1D xyz must have length 2 or 3")
    if a.ndim != 2:
        raise ValueError("xyz must be 2D after normalization")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), float)])
    if a.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    return a

def _lane_group_key_of_edge(data: Dict[str, Any]):
    return data['group_key']
    # # Prefer precomputed group_key (e.g., (A,B,side,lane_count))
    # if 'group_key' in data and isinstance(data['group_key'], (tuple, list)):
    #     return tuple(data['group_key'])
    # return (int(data['boundary_a_id']), int(data['boundary_b_id']), str(data['side']))


# ---------- result dataclass ----------

@dataclass
class BuildResult:
    edge_graph: nx.DiGraph
    junction_map: Dict[str, List[str]]        # merged_junction_id -> list(original lane node ids)
    edge_members: Dict[str, List[str]]        # new edge id -> list(original lane edge ids)
    node_xyz: Dict[str, np.ndarray]           # merged_junction_id -> representative xyz


# ---------- union-find ----------

class DSU:
    def __init__(self):
        self.p = {}
        self.r = {}
    def add(self, x):
        if x not in self.p:
            self.p[x] = x
            self.r[x] = 0
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ---------- main: topology merge + group-endpoint merge + group-based edges ----------

def build_edge_graph_from_lane_graph_topo(
    G_lane: nx.DiGraph,
    *,
    representative: str = "median",   # how to pick representative lane geom per edge
    edge_id_prefix: str = "e",
    # which connector subtypes trigger node merging (all True by default)
    merge_longitudinal: bool = True,
    merge_lateral: bool = True,
    merge_turning: bool = True,
    # NEW: also merge lane endpoints that share the same group_key
    merge_group_endpoints: bool = True,     # merge all to_nodes per group_key
    merge_group_startpoints: bool = True,  # optionally, merge all from_nodes per group_key
) -> BuildResult:
    """
    1) Merge nodes by connector relations:
       - For every connector edge (kind='connector'), union its (u,v) endpoints.

    2) (Optional) Merge lane endpoints by group:
       - If merge_group_endpoints: union all 'to_node' of lane edges sharing the same group_key.
       - If merge_group_startpoints: union all 'from_node' likewise.

       NOTE: this collapses each group's lane fan-out/fan-in at a cross-section into one junction,
       independent of spatial proximity.

    3) Build edges by group information:
       - For each merged (uJ -> vJ) and group_key, aggregate *all lanes* into ONE edge.
    """
    # Cache node xyz where available; fallback to lane endpoints
    node_xyz_raw: Dict[str, np.ndarray] = {}
    for n, nd in G_lane.nodes(data=True):
        if 'xyz' in nd:
            try:
                node_xyz_raw[n] = _to_xyz(nd['xyz'])
            except Exception:
                pass
    for u, v, ed in G_lane.edges(data=True):
        if ed.get('kind') != 'lane': continue
        geom = ed.get('geom')
        if geom is None or len(geom) < 1: continue
        g = _to_xyz(geom)
        node_xyz_raw.setdefault(u, g[:1])
        node_xyz_raw.setdefault(v, g[-1:].copy())

    # Collect lane edges for later bucketing and for group endpoint merges
    lane_edges: List[Tuple[str, str, Dict[str, Any]]] = []
    by_group_starts: Dict[tuple, List[str]] = {}
    by_group_ends: Dict[tuple, List[str]] = {}
    for u, v, ed in G_lane.edges(data=True):
        if ed.get('kind') != 'lane':
            continue
        lane_edges.append((u, v, ed))
        gk = _lane_group_key_of_edge(ed)
        by_group_starts.setdefault(gk, []).append(u)
        by_group_ends.setdefault(gk, []).append(v)

    # Node merging
    dsu = DSU()
    for n in G_lane.nodes: dsu.add(n)

    # (1) Merge by connectors (topology-only)
    def _merge_allowed(subtype: str) -> bool:
        if subtype == 'longitudinal': return merge_longitudinal
        if subtype == 'lateral':      return merge_lateral
        if subtype in ('turn_left','turn_right'): return merge_turning
        return True

    for u, v, ed in G_lane.edges(data=True):
        if ed.get('kind') != 'connector': continue
        st = ed.get('subtype', 'connector')
        if _merge_allowed(st):
            dsu.union(u, v)

    # (2) ALSO merge lane endpoints by group_key (as requested)
    if merge_group_endpoints:
        for gk, nodes in by_group_ends.items():
            if not nodes: continue
            root = nodes[0]
            for nid in nodes[1:]:
                dsu.union(root, nid)

    if merge_group_startpoints:
        for gk, nodes in by_group_starts.items():
            if not nodes: continue
            root = nodes[0]
            for nid in nodes[1:]:
                dsu.union(root, nid)

    # Collapse DSU -> junctions, compute representative xyz
    groups: Dict[str, List[str]] = {}
    for nid in G_lane.nodes:
        rid = dsu.find(nid)
        groups.setdefault(rid, []).append(nid)

    junction_id_of: Dict[str, str] = {}
    junction_xyz: Dict[str, np.ndarray] = {}
    for jidx, (rep, members) in enumerate(groups.items()):
        jid = f"j{jidx:06d}"
        for nid in members:
            junction_id_of[nid] = jid
        pts = []
        for nid in members:
            if nid in node_xyz_raw:
                p = node_xyz_raw[nid][0] if node_xyz_raw[nid].ndim == 2 else node_xyz_raw[nid]
                pts.append(p)
        junction_xyz[jid] = np.mean(np.stack(pts, axis=0), axis=0) if pts else None

    # Bucket lanes by (uJ, vJ, group_key) and build one edge per bucket
    from collections import defaultdict
    pair_to_lanes: Dict[Tuple[str, str, tuple], List[Dict[str, Any]]] = defaultdict(list)

    for u, v, data in lane_edges:
        uJ = junction_id_of.get(u); vJ = junction_id_of.get(v)
        if uJ is None or vJ is None or uJ == vJ:
            continue
        gk = _lane_group_key_of_edge(data)
        rec = dict(
            lane_edge_id=data.get('id', data.get('edge_id', f"{u}->{v}")),
            lane_uid=f"corridor:{data['boundary_a_id']}-{data['boundary_b_id']}:{data['side']}:lane{int(data['lane_index']):02d}",
            lane_index=int(data['lane_index']),
            group_key=gk,
            geom=np.asarray(data.get('geom'), float) if 'geom' in data else None,
        )
        pair_to_lanes[(uJ, vJ, gk)].append(rec)

    def _pick_representative(lanes: List[Dict[str, Any]], mode: str = "median") -> Dict[str, Any]:
        if not lanes: raise ValueError("lanes list is empty")
        if mode == "median":
            idxs = sorted((int(ld['lane_index']), j) for j, ld in enumerate(lanes))
            return lanes[idxs[len(idxs)//2][1]]
        if mode in ("leftmost","min","min_index"):
            return min(lanes, key=lambda d: int(d['lane_index']))
        if mode in ("rightmost","max","max_index"):
            return max(lanes, key=lambda d: int(d['lane_index']))
        return lanes[len(lanes)//2]

    EdgeG = nx.DiGraph()
    for (uJ, vJ, gk), lanes in pair_to_lanes.items():
        if not lanes: continue
        lanes_sorted = sorted(lanes, key=lambda d: d['lane_index'])
        rep = _pick_representative(lanes_sorted, representative)

        eid = f"{edge_id_prefix}_{uJ}_{vJ}"
        if uJ not in EdgeG: EdgeG.add_node(uJ, xyz=junction_xyz.get(uJ))
        if vJ not in EdgeG: EdgeG.add_node(vJ, xyz=junction_xyz.get(vJ))

        EdgeG.add_edge(
            uJ, vJ,
            id=eid,
            lanes=[le['lane_edge_id'] for le in lanes_sorted],
            lane_indices=[int(le['lane_index']) for le in lanes_sorted],
            group_key=gk,
            representative_lane=rep['lane_edge_id'],
            representative_uid=rep['lane_uid'],
            representative_index=int(rep['lane_index']),
            geom=rep.get('geom', None),   # representative geometry for drawing
            type="edge",
            lanes_count=len(lanes_sorted),
        )

    # Outputs
    junction_map: Dict[str, List[str]] = {}
    for nid, jid in junction_id_of.items():
        junction_map.setdefault(jid, []).append(nid)

    edge_members: Dict[str, List[str]] = {}
    for _, _, data in EdgeG.edges(data=True):
        edge_members[data['id']] = list(data.get('lanes', []))

    return BuildResult(
        edge_graph=EdgeG,
        junction_map=junction_map,
        edge_members=edge_members,
        node_xyz={j: EdgeG.nodes[j].get('xyz') for j in EdgeG.nodes},
    )


# ---------- quick plot ----------

def plot_edge_graph(edge_result: BuildResult, show_nodes=True, show_labels=False, figsize=(10,10)):
    import matplotlib.pyplot as plt
    G = edge_result.edge_graph
    node_xyz = edge_result.node_xyz

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for u, v, data in G.edges(data=True):
        geom = data.get('geom')
        if geom is not None and len(geom) >= 2:
            xy = np.asarray(geom)[:, :2]
            ax.plot(xy[:, 0], xy[:, 1], linewidth=2, alpha=0.85)
            if show_labels:
                mid = len(xy)//2
                ax.text(xy[mid,0], xy[mid,1], data.get('id',''), fontsize=8, color='purple')
        else:
            pu = node_xyz.get(u); pv = node_xyz.get(v)
            if pu is not None and pv is not None:
                ax.plot([pu[0], pv[0]], [pu[1], pv[1]], linestyle='--', alpha=0.5)

    # Draw nodes
    if show_nodes:
        for nid, p in node_xyz.items():
            if p is None: continue
            ax.scatter(p[0], p[1], color='black', s=20)
            if show_labels:
                ax.text(p[0], p[1], nid, fontsize=7, color='red')

    ax.set_aspect('equal')
    ax.set_title("Edge Graph (connectors merged + group endpoint merged)")
    plt.show()
