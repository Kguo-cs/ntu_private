# desay_edge_graph.py
# Build a SUMO-like edge graph from a lane graph with robust node handling.
# Requires: numpy, shapely, networkx, scipy (already used in your pipeline)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Iterable
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point


# ---------- helpers ----------

def _to_xyz(arr) -> np.ndarray:
    """
    Normalize input to shape [N,3].
    Accepts [N,3], [N,2], [3,N], [x,y,z], [x,y].
    """
    a = np.asarray(arr, dtype=float)

    # 1D vectors: [x,y] or [x,y,z]
    if a.ndim == 1:
        if a.shape[0] == 2:
            a = np.array([[a[0], a[1], 0.0]], dtype=float)
        elif a.shape[0] == 3:
            a = a.reshape(1, 3)
        else:
            raise ValueError("1D xyz must have length 2 or 3")

    if a.ndim != 2:
        raise ValueError("xyz must be 2D after normalization")

    # [3,N] -> transpose (but not a 3x3 matrix)
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T

    # [N,2] -> pad z=0
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a), dtype=float)])

    if a.shape[1] != 3:
        raise ValueError("xyz must be shape [N,3] (after normalization)")

    return a


def _arclength2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([])
    return np.concatenate([[0.0], np.cumsum(d)]) if d.size else np.array([0.0])


def _lane_group_key_of_edge(data: Dict[str, Any]) -> Tuple[int, int, str]:
    # expected format from previous builders
    if 'group_key' in data and isinstance(data['group_key'], (tuple, list)):
        gk = tuple(data['group_key'])
        return (int(gk[0]), int(gk[1]), str(gk[2]))
    return (int(data['boundary_a_id']), int(data['boundary_b_id']), str(data['side']))


def _pick_representative(lanes: List[Dict[str, Any]], mode: str = "median") -> Dict[str, Any]:
    if not lanes:
        raise ValueError("lanes list is empty")
    if mode == "median":
        idxs = sorted((int(ld['lane_index']), j) for j, ld in enumerate(lanes))
        mid = idxs[len(idxs)//2][1]
        return lanes[mid]
    if mode in ("leftmost", "min", "min_index"):
        j = min(range(len(lanes)), key=lambda j: int(lanes[j]['lane_index']))
        return lanes[j]
    if mode in ("rightmost", "max", "max_index"):
        j = max(range(len(lanes)), key=lambda j: int(lanes[j]['lane_index']))
        return lanes[j]
    return lanes[len(lanes)//2]


# ---------- result dataclass ----------

@dataclass
class BuildResult:
    edge_graph: nx.DiGraph
    junction_map: Dict[str, List[str]]        # merged_junction_id -> list(original lane node ids)
    edge_members: Dict[str, List[str]]        # new edge id -> list(original lane edge ids)
    node_xyz: Dict[str, np.ndarray]           # merged_junction_id -> representative xyz


# ---------- main function ----------

def build_edge_graph_from_lane_graph_topo(
    G_lane: nx.DiGraph,
    boundary_dict: Dict[int, np.ndarray],
    *,
    s_merge_tol_m: float = 10.0,                 # arclength tolerance (meters) for merging
    lateral_require_bidir: bool = True,         # mutual lane-change to be in same edge
    lateral_min_overlap_frac: float = 0.25,     # best-effort check using anchors (optional)
    representative: str = "median",             # representative lane per edge
    edge_id_prefix: str = "e",
) -> BuildResult:
    """
    Build a SUMO-like edge graph where:
      - Junction nodes are merged using boundary arclength coordinates (s) and successor links.
      - Inside each (u->v) junction pair, lanes are grouped into edges by mutual lane-changing.
    """

    # 0) Precompute reference boundary geometries (LineString)
    bgeom: Dict[int, LineString] = {}
    for bid, xyz in boundary_dict.items():
        xyz = _to_xyz(xyz)
        bgeom[int(bid)] = LineString(xyz[:, :2])

    # 1) Seed node_xyz from lane edges and/or node attributes (robust to missing/1D xyz)
    node_xyz: Dict[str, np.ndarray] = {}

    for u, v, data in G_lane.edges(data=True):
        if data.get('kind') != 'lane':
            continue
        # prefer graph node xyz if present and valid
        if 'xyz' in G_lane.nodes[u]:
            try:
                node_xyz[u] = _to_xyz(G_lane.nodes[u]['xyz'])
            except Exception:
                pass
        if 'xyz' in G_lane.nodes[v]:
            try:
                node_xyz[v] = _to_xyz(G_lane.nodes[v]['xyz'])
            except Exception:
                pass

    # fallback to edge endpoints for any missing nodes
    for u, v, data in G_lane.edges(data=True):
        if data.get('kind') != 'lane':
            continue
        geom = data.get('geom', None)
        if geom is None or len(geom) < 1:
            continue
        geom = _to_xyz(geom)
        if u not in node_xyz:
            node_xyz[u] = geom[:1]
        if v not in node_xyz:
            node_xyz[v] = geom[-1:].copy()

    # 2) Collect lane edges and project their endpoints to boundary A arclength
    node_s_by_group: Dict[str, Dict[Tuple[int, int, str], float]] = {}
    lane_edges: List[Tuple[str, str, Dict[str, Any]]] = []

    for u, v, data in G_lane.edges(data=True):
        if data.get('kind') != 'lane':
            continue
        lane_edges.append((u, v, data))
        gk = _lane_group_key_of_edge(data)
        bA = int(data['boundary_a_id'])
        lsA = bgeom.get(bA)
        if lsA is None:
            continue

        # project start & end onto boundary A (if we have node positions)
        if u in node_xyz:
            pu = node_xyz[u][0, :2]
            su = float(lsA.project(Point(float(pu[0]), float(pu[1]))))
            node_s_by_group.setdefault(u, {})[gk] = su
        if v in node_xyz:
            pv = node_xyz[v][0, :2]
            sv = float(lsA.project(Point(float(pv[0]), float(pv[1]))))
            node_s_by_group.setdefault(v, {})[gk] = sv

        # store s on the lane edge too (useful later)
        data['_s_start_m'] = node_s_by_group.get(u, {}).get(gk, None)
        data['_s_end_m']   = node_s_by_group.get(v, {}).get(gk, None)
        data['_group_key'] = gk

    # 3) Merge junctions per group_key by clustering nodes with similar s within tolerance
    merged_id_map: Dict[Tuple[str, Tuple[int, int, str]], str] = {}  # (node_id, gk) -> merged_id
    junction_members: Dict[str, List[str]] = {}                      # merged_id -> list(node_ids)
    junction_xyz: Dict[str, np.ndarray] = {}                         # merged_id -> mean xyz
    next_jid = 0

    from collections import defaultdict
    by_group_nodes = defaultdict(list)  # gk -> list[(node_id, s, xyz)]
    for nid, per in node_s_by_group.items():
        for gk, s in per.items():
            xyz = node_xyz.get(nid)
            if xyz is None:
                continue
            p = xyz[0] if xyz.ndim == 2 else xyz
            by_group_nodes[gk].append((nid, float(s), np.asarray(p, float)))

    def _flush_cluster(cluster: List[Tuple[str, float, np.ndarray]], gk):
        nonlocal next_jid
        if not cluster:
            return
        jid = f"j{next_jid:06d}"
        next_jid += 1
        members = [nid for (nid, _, _) in cluster]
        for nid, _, _ in cluster:
            merged_id_map[(nid, gk)] = jid
        xyzs = np.stack([p for (_, _, p) in cluster], axis=0)
        junction_members[jid] = members
        junction_xyz[jid] = np.mean(xyzs, axis=0)

    for gk, lst in by_group_nodes.items():
        lst.sort(key=lambda x: x[1])  # sort by s
        cluster = []
        if not lst:
            continue
        s_anchor = lst[0][1]
        for item in lst:
            nid, s, p = item
            if not cluster:
                cluster = [item]
                s_anchor = s
            else:
                if abs(s - s_anchor) <= s_merge_tol_m:
                    cluster.append(item)
                    s_anchor = min(s_anchor, s)
                else:
                    _flush_cluster(cluster, gk)
                    cluster = [item]
                    s_anchor = s
        _flush_cluster(cluster, gk)

    # 4) Bucket lane edges by merged junction pair (uJ -> vJ) and group_key
    pair_to_lanes: Dict[Tuple[str, str, Tuple[int, int, str]], List[Dict[str, Any]]] = defaultdict(list)

    for u, v, data in lane_edges:
        gk = data['_group_key']
        uJ = merged_id_map.get((u, gk))
        vJ = merged_id_map.get((v, gk))
        if uJ is None or vJ is None or uJ == vJ:
            continue
        rec = dict(
            lane_edge_id=data.get('id', data.get('edge_id', f"{u}->{v}")),
            lane_uid=f"corridor:{data['boundary_a_id']}-{data['boundary_b_id']}:{data['side']}:lane{int(data['lane_index']):02d}",
            lane_index=int(data['lane_index']),
            group_key=gk,
            geom=np.asarray(data.get('geom'), float) if 'geom' in data else None,
            _s_start_m=data.get('_s_start_m'),
            _s_end_m=data.get('_s_end_m'),
            _u=u, _v=v,
        )
        pair_to_lanes[(uJ, vJ, gk)].append(rec)

    # 5) Build lateral connectivity between lane_uids from lateral connectors in G_lane
    from collections import defaultdict as dd2
    lateral_dir: Dict[Tuple[str, str], List[float]] = dd2(list)  # (from_uid, to_uid) -> [anchor_s...]
    lateral_pairs: Dict[frozenset, Dict[str, bool]] = {}

    for x, y, ed in G_lane.edges(data=True):
        if ed.get('kind') != 'connector' or ed.get('subtype') != 'lateral':
            continue
        uid_from = ed.get('from_lane')
        uid_to   = ed.get('to_lane')
        if not uid_from or not uid_to:
            continue
        anchor_s = float(ed.get('anchor_s', 0.0))
        lateral_dir[(uid_from, uid_to)].append(anchor_s)
        key = frozenset([uid_from, uid_to])
        lateral_pairs.setdefault(key, {'ab': False, 'ba': False})

    # finalize mutual flags
    for key in list(lateral_pairs.keys()):
        a, b = tuple(key)
        ab = (a, b) in lateral_dir and len(lateral_dir[(a, b)]) > 0
        ba = (b, a) in lateral_dir and len(lateral_dir[(b, a)]) > 0
        lateral_pairs[key]['ab'] = ab
        lateral_pairs[key]['ba'] = ba

    def _lanes_connected(uidA: str, uidB: str) -> bool:
        key = frozenset([uidA, uidB])
        if key not in lateral_pairs:
            return False
        st = lateral_pairs[key]
        return (st.get('ab', False) and st.get('ba', False)) if lateral_require_bidir \
               else (st.get('ab', False) or st.get('ba', False))

    def _approx_seg_len(seg: Dict[str, Any]) -> float:
        geom = seg.get('geom')
        if geom is None or len(geom) < 2:
            return 1.0
        s = _arclength2d(np.asarray(geom)[:, :2])
        return float(s[-1])

    def _passes_overlap(uidA: str, uidB: str, approx_len: float) -> bool:
        if lateral_min_overlap_frac <= 0:
            return True
        anchors = len(lateral_dir.get((uidA, uidB), [])) + len(lateral_dir.get((uidB, uidA), []))
        if anchors == 0:
            return False
        typical_spacing = 30.0  # conservative if you place lateral anchors ~35m
        est_cover = anchors * typical_spacing
        return (est_cover / max(1.0, approx_len)) >= lateral_min_overlap_frac

    # 6) Build edge graph
    EdgeG = nx.DiGraph()

    for (uJ, vJ, gk), lanes in pair_to_lanes.items():
        if not lanes:
            continue
        # sort by lane_index; contiguous indices may be grouped if laterally connected
        lanes_sorted = sorted(lanes, key=lambda d: d['lane_index'])
        groups: List[List[Dict[str, Any]]] = []
        cur = [lanes_sorted[0]]

        for i in range(1, len(lanes_sorted)):
            A = lanes_sorted[i-1]
            B = lanes_sorted[i]
            contig = (abs(A['lane_index'] - B['lane_index']) == 1)
            if not contig:
                groups.append(cur); cur = [B]; continue
            connected = _lanes_connected(A['lane_uid'], B['lane_uid'])
            if connected:
                seg_len = min(_approx_seg_len(A), _approx_seg_len(B))
                if _passes_overlap(A['lane_uid'], B['lane_uid'], seg_len):
                    cur.append(B)
                else:
                    groups.append(cur); cur = [B]
            else:
                groups.append(cur); cur = [B]
        groups.append(cur)

        # add one EdgeG edge per lane group
        for gidx, group in enumerate(groups):
            rep = _pick_representative(group, representative)
            eid = f"{edge_id_prefix}_{uJ}_{vJ}_{gidx:02d}"

            if uJ not in EdgeG:
                EdgeG.add_node(uJ, xyz=None)  # fill below
            if vJ not in EdgeG:
                EdgeG.add_node(vJ, xyz=None)

            EdgeG.add_edge(
                uJ, vJ,
                id=eid,
                lanes=[le['lane_edge_id'] for le in group],
                lane_indices=[int(le['lane_index']) for le in group],
                group_key=gk,
                representative_lane=rep['lane_edge_id'],
                representative_uid=rep['lane_uid'],
                representative_index=int(rep['lane_index']),
                geom=rep.get('geom', None),
                type="edge",
                lanes_count=len(group),
            )

    # assign node xyz from merged clusters (mean of members)
    # (junction_xyz is computed below)
    # 7) Build junction maps / xyz
    junction_map: Dict[str, List[str]] = {}
    node_xyz_merged: Dict[str, np.ndarray] = {}

    # reconstruct from merged_id_map: inverse mapping
    from collections import defaultdict as dd
    inv_map = dd(list)
    for (nid, gk), jid in merged_id_map.items():
        inv_map[jid].append(nid)

    for jid, members in inv_map.items():
        pts = []
        for nid in members:
            if nid in node_xyz:
                p = node_xyz[nid][0] if node_xyz[nid].ndim == 2 else node_xyz[nid]
                pts.append(p)
        if pts:
            xyz = np.mean(np.stack(pts, axis=0), axis=0)
            node_xyz_merged[jid] = xyz
        else:
            node_xyz_merged[jid] = None
        junction_map[jid] = members

    # ensure EdgeG nodes get xyz
    for n in EdgeG.nodes():
        if EdgeG.nodes[n].get('xyz') is None:
            EdgeG.nodes[n]['xyz'] = node_xyz_merged.get(n)

    # 8) Build edge_members dict
    edge_members: Dict[str, List[str]] = {}
    for uJ, vJ, data in EdgeG.edges(data=True):
        eid = data['id']
        edge_members[eid] = list(data.get('lanes', []))

    return BuildResult(
        edge_graph=EdgeG,
        junction_map=junction_map,
        edge_members=edge_members,
        node_xyz=node_xyz_merged,
    )


# ---------- optional: quick plot ----------
#
# def plot_edge_graph(edge_result: BuildResult, show_nodes=True, show_labels=False, figsize=(10,10)):
#     import matplotlib.pyplot as plt
#     G = edge_result.edge_graph
#     node_xyz = edge_result.node_xyz
#
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # Draw edges
#     for u, v, data in G.edges(data=True):
#         geom = data.get('geom')
#         if geom is not None and len(geom) >= 2:
#             xy = np.asarray(geom)[:, :2]
#             ax.plot(xy[:, 0], xy[:, 1], color='blue', linewidth=2, alpha=0.7, zorder=1)
#             if show_labels:
#                 mid = len(xy)//2
#                 ax.text(xy[mid,0], xy[mid,1], data.get('id',''), fontsize=8, color='purple')
#         else:
#             pu = node_xyz.get(u); pv = node_xyz.get(v)
#             if pu is not None and pv is not None:
#                 ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color='blue', linestyle='--', alpha=0.5)
#
#     # Draw nodes
#     if show_nodes:
#         for nid, p in node_xyz.items():
#             if p is None:
#                 continue
#             ax.scatter(p[0], p[1], color='black', s=20, zorder=2)
#             if show_labels:
#                 ax.text(p[0], p[1], nid, fontsize=7, color='red')
#
#     ax.set_aspect('equal')
#     ax.set_title("Edge Graph (SUMO-like)")
#     plt.show()
#
#
# # ---------- example ----------
#
# if __name__ == "__main__":
#     # Example usage (requires you to supply G_lane and boundary_dict):
#     # result = build_edge_graph_from_lane_graph_topo(G_lane, boundary_dict)
#     # plot_edge_graph(result, show_nodes=True, show_labels=True)
#     pass
