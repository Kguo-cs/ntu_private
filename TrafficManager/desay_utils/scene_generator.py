# -*- coding: utf-8 -*-
# TrafficGenerator with pedestrians following boundaries; vehicles on lanes/edges (lane-aware spacing)
from __future__ import annotations
from typing import Iterable, Set, List, Optional, Dict, Any, Tuple
import numpy as np
import networkx as nx
import torch.linalg
from shapely.geometry import LineString, Point

# ------------------ utilities ------------------
# 1) Build an "ego" agent dict compatible with TrafficGenerator output
def make_ego_agent(ego_route_xyz: np.ndarray, ego_start_xy: np.ndarray,ego_edge_ids, *,
                   ego_id="ego", cls="car", avg_speed_mps=14.0, size_lwh=(4.5, 1.85, 1.6)) -> dict:
    # route_xyz is already a continuous polyline along EG path
    start_xyz = np.array([ego_start_xy[0], ego_start_xy[1], 0.0], dtype=float)
    return dict(
        agent_id=ego_id,
        cls=cls,
        size_lwh_m=tuple(size_lwh),
        avg_speed_mps=float(avg_speed_mps),
        start_xyz=start_xyz,  # simulator will snap to the nearest lane
        route_xyz=np.asarray(ego_route_xyz, float),
        edge_ids=list(ego_edge_ids),  # optional; not required by the torch simulator
        start_lane_id=None,  # let the simulator pick nearest lane
        goal_xyz=ego_route_xyz[-1]
    )


def _key_id(x):
    if isinstance(x, np.ndarray):
        return tuple(np.asarray(x).ravel().tolist())
    if isinstance(x, (np.bool_, np.number)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return tuple(_key_id(t) for t in x)
    return x

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("expect [N,3]/[N,2]/[3,N]")
    return a

def _arclen2d(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2: return np.array([0.0])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _interp_xyz_at_s(xyz: np.ndarray, s_tab: np.ndarray, s: float) -> np.ndarray:
    x = np.interp(s, s_tab, xyz[:,0]); y = np.interp(s, s_tab, xyz[:,1]); z = np.interp(s, s_tab, xyz[:,2])
    return np.array([x, y, z], float)

def _heading_at_s_dir(xyz: np.ndarray, s_tab: np.ndarray, s: float, dir_sign: int, eps: float = 0.8) -> float:
    """Heading at arclength s following direction sign (+1 forward, -1 backward)."""
    if s_tab[-1] <= 0: return 0.0
    if dir_sign >= 0:
        s0 = max(0.0, s - eps); s1 = min(s_tab[-1], s + eps)
    else:
        s0 = min(s_tab[-1], s + eps); s1 = max(0.0, s - eps)  # swap to go "backwards"
    p0 = _interp_xyz_at_s(xyz, s_tab, s0)
    p1 = _interp_xyz_at_s(xyz, s_tab, s1)
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    if dx == 0 and dy == 0: return 0.0
    return float(np.arctan2(dy, dx))

def _slice_from_s_to_s(xyz: np.ndarray, s_tab: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """Slice polyline between arclength s0→s1 (handles forward/backward)."""
    s0 = float(np.clip(s0, 0.0, s_tab[-1] if len(s_tab) else 0.0))
    s1 = float(np.clip(s1, 0.0, s_tab[-1] if len(s_tab) else 0.0))
    if s1 == s0:
        P = _interp_xyz_at_s(xyz, s_tab, s0)
        return np.vstack([P, P])
    # forward
    if s1 > s0:
        mask = (s_tab > s0) & (s_tab < s1)
        pts = [_interp_xyz_at_s(xyz, s_tab, s0)]
        if np.any(mask): pts.extend(list(xyz[mask]))
        pts.append(_interp_xyz_at_s(xyz, s_tab, s1))
        return np.asarray(pts)
    # backward
    else:
        mask = (s_tab < s0) & (s_tab > s1)
        pts = [_interp_xyz_at_s(xyz, s_tab, s0)]
        if np.any(mask): pts.extend(list(xyz[mask][::-1]))
        pts.append(_interp_xyz_at_s(xyz, s_tab, s1))
        return np.asarray(pts)

def _sample_point_on_shape_with_s(shape_xyz: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    P = _to_xyz(shape_xyz)
    s = _arclen2d(P[:, :2])
    if s[-1] <= 0: return P[0], 0.0
    u = float(rng.uniform(0.0, s[-1]))
    return _interp_xyz_at_s(P, s, u), u

def build_edge_id_graph(EG: nx.DiGraph, weight_attr: str = "weight") -> nx.DiGraph:
    id2uv = {}; w_of = {}
    for u, v, ed in EG.edges(data=True):
        eid = _key_id(ed["id"])
        id2uv[eid] = (_key_id(u), _key_id(v))
        w_of[eid] = float(ed.get(weight_attr, ed.get("weight", 1.0)))
    EID = nx.DiGraph()
    for eid in id2uv.keys(): EID.add_node(eid)
    allowed = EG.graph.get("allowed_edge_turns", None)
    if allowed:
        for e1, e2 in allowed:
            e1k, e2k = _key_id(e1), _key_id(e2)
            if e1k in id2uv and e2k in id2uv:
                EID.add_edge(e1k, e2k, weight=w_of[e2k])
    else:
        for e1, (u1, v1) in id2uv.items():
            for e2, (u2, _v2) in id2uv.items():
                if v1 == u2:
                    EID.add_edge(e1, e2, weight=w_of[e2])
    return EID

# ------------------ generator ------------------

class TrafficGenerator:
    def __init__(
        self,
        EG: nx.DiGraph,
        G_lane: nx.DiGraph,
        router_func=None,
        *,
        boundary_xyz: Optional[Dict[Any, np.ndarray] | List[np.ndarray]] = None,
        vehicle_classes: Optional[Set[str]] = None,
        pedestrian_classes: Optional[Set[str]] = None
    ):
        """
        EG (edge graph):
          - edges must have: 'id'
          - preferred: 'geom' ([N,3]) for representative geometry; fallback to 'shape_xyz'
          - optional: 'length_m' (else computed)
          - optional: 'lanes' -> list of lane edge IDs (from G_lane)

        G_lane (lane graph):
          - read lane centerlines from edges with kind == 'lane' -> 'id', 'geom'
        """
        self.EG = EG
        self.G_lane = G_lane
        self.router = router_func
        self._EID: Optional[nx.DiGraph] = None

        # cache edge shapes & lengths (prefer 'geom' from edge graph)
        self.edge_shapes: Dict[Any, np.ndarray] = {}
        self.edge_lengths: Dict[Any, float] = {}
        self.edge_member_lanes: Dict[Any, List[Any]] = {}

        for u, v, ed in EG.edges(data=True):
            eid = _key_id(ed["id"])
            geom = ed.get("geom", ed.get("shape_xyz", None))
            if geom is None:
                pu = EG.nodes[_key_id(u)].get("xyz", None)
                pv = EG.nodes[_key_id(v)].get("xyz", None)
                if pu is not None and pv is not None:
                    geom = np.vstack([_to_xyz(pu)[0], _to_xyz(pv)[0]])
                else:
                    raise ValueError(f"Edge {eid} missing 'geom'/'shape_xyz' and node xyz fallback failed.")
            P = _to_xyz(geom)
            self.edge_shapes[eid] = P
            self.edge_lengths[eid] = float(ed.get("length_m", _arclen2d(P[:, :2])[-1]))
            lanes = ed.get("lanes", [])
            self.edge_member_lanes[eid] = [_key_id(l) for l in lanes] if lanes is not None else []

        # lane centerlines from lane graph (edges with kind == 'lane')
        self.lane_xyz: Dict[Any, np.ndarray] = {}
        for u, v, ed in G_lane.edges(data=True):
            if ed.get("kind") == "lane":
                lid = _key_id(ed.get("id", ed.get("edge_id", f"{u}->{v}")))
                geom = ed.get("geom", None)
                if geom is None:
                    pu = G_lane.nodes[u].get("xyz"); pv = G_lane.nodes[v].get("xyz")
                    if pu is not None and pv is not None:
                        geom = np.vstack([_to_xyz(pu)[0], _to_xyz(pv)[0]])
                if geom is not None:
                    self.lane_xyz[lid] = _to_xyz(geom)

        # boundaries list + lengths
        if boundary_xyz is None:
            self.boundaries: List[np.ndarray] = []
        elif isinstance(boundary_xyz, dict):
            self.boundaries = [_to_xyz(arr) for arr in boundary_xyz.values()]
        else:
            self.boundaries = [_to_xyz(arr) for arr in boundary_xyz]
        self.boundary_lengths: List[float] = [float(_arclen2d(B[:, :2])[-1]) for B in self.boundaries]

        # class sets
        self.vehicle_classes = set(vehicle_classes) if vehicle_classes is not None else {"car", "truck", "bicycle"}
        self.pedestrian_classes = set(pedestrian_classes) if pedestrian_classes is not None else {"pedestrian"}

    def _build_edge_id_graph(self, weight_attr: str = "weight") -> nx.DiGraph:
        if hasattr(self, "_EID") and self._EID is not None: return self._EID
        self._EID = build_edge_id_graph(self.EG, weight_attr=weight_attr)
        return self._EID

    def _edges_connected_to_ego_edges(self, ego_edge_ids: Iterable[Any], *, hops: Optional[int]=1,
                                      mode: str="both") -> Set[Any]:
        from collections import deque
        EID = self._build_edge_id_graph()
        ego = [_key_id(e) for e in ego_edge_ids if _key_id(e) in EID]
        if not ego: return set()
        if mode in ("strong","weak"):
            if mode=="strong":
                out=set()
                for comp in nx.strongly_connected_components(EID):
                    if any(e in comp for e in ego): out |= set(comp)
                return out
            U = EID.to_undirected(); out=set(); seen=set()
            for e in ego:
                if e in seen: continue
                comp = nx.node_connected_component(U, e)
                out |= set(comp); seen |= set(comp)
            return out
        def bfs(srcs: List[Any], forward: bool, max_hops: Optional[int]) -> Set[Any]:
            Gdir = EID if forward else EID.reverse()
            if max_hops is None:
                out = set(srcs)
                for s in srcs: out |= nx.descendants(Gdir, s)
                return out
            visited=set(srcs); dq=deque((s,0) for s in srcs)
            while dq:
                u,d = dq.popleft()
                if d==max_hops: continue
                for v in Gdir.successors(u):
                    if v not in visited:
                        visited.add(v); dq.append((v,d+1))
            return visited
        if mode=="forward":  return bfs(ego, True, hops)
        if mode=="backward": return bfs(ego, False, hops)
        return bfs(ego, True, hops) | bfs(ego, False, hops)

    # ---- spawners ----
    def _spawn_on_boundary(self, b_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
        B = self.boundaries[b_idx]; sB = _arclen2d(B[:, :2]); L = sB[-1]
        if L <= 0: return B[0], 0.0, 0.0
        s0 = float(rng.uniform(0.0, L))
        heading = _heading_at_s_dir(B, sB, s0, dir_sign=+1)
        xyz = _interp_xyz_at_s(B, sB, s0)
        return xyz, heading, s0

    def sample_agent(self,lane_avail_lengths,lane_avail_s,class_ratio,speeds,size_tab,ego_start_lanes):
        agent_list=[]

        while True:
            for type in ['car','truck',"bicycle"]:
                for agent_idx in range(class_ratio[type]):
                    lanes = list(lane_avail_lengths.keys())
                    size_lwh=size_tab[type]

                    avg_speed=speeds[type]

                    heading_gap=avg_speed*1.5+size_lwh[0]+2  #minimum gap 2 m, 1.5 s heading

                    weights = np.array(list(lane_avail_lengths.values()), dtype=float)-heading_gap

                    if ego_start_lanes is not None:
                        for i,id in enumerate(lane_avail_lengths.keys()):
                            if id not in ego_start_lanes:
                                weights[i]=0

                    weights=np.clip(weights,a_min=0,a_max=1000)

                    if np.sum(weights)==0:
                        return lane_avail_lengths,lane_avail_s,agent_list

                    # sample one lane
                    sampled_lane = np.random.choice(lanes, p=weights / weights.sum() )

                    new_gaps=[]

                    for lane_s in lane_avail_s[sampled_lane]:
                        new_gap=lane_s[1]-lane_s[0]-heading_gap
                        new_gaps.append(new_gap)

                    weights=np.array(new_gaps)

                    weights=np.clip(weights,a_min=0,a_max=1000)

                    if np.sum(weights)==0:
                        return lane_avail_lengths,lane_avail_s,agent_list

                    sampled_seg=np.random.choice(len(new_gaps),p=weights/np.sum(weights))

                    sampled_pos=np.random.rand()*new_gaps[sampled_seg]

                    lane_avail_lengths[sampled_lane]=lane_avail_lengths[sampled_lane]-heading_gap

                    original_seg=lane_avail_s[sampled_lane][sampled_seg]

                    new_seg1=np.array([original_seg[0],original_seg[0]+sampled_pos])

                    new_seg2=np.array([original_seg[0]+sampled_pos+heading_gap,original_seg[1]])

                    del lane_avail_s[sampled_lane][sampled_seg]

                    lane_avail_s[sampled_lane].append(new_seg1)
                    lane_avail_s[sampled_lane].append(new_seg2)

                    u=original_seg[0]+sampled_pos

                    L = self.lane_xyz[sampled_lane]
                    sL = self.sL[sampled_lane]

                    xyz = _interp_xyz_at_s(L, sL, u)
                    heading = _heading_at_s_dir(L, sL, u, dir_sign=+1)

                    agent=dict(
                            agent_id=f"A{len(agent_list)}",
                            cls=type,
                            size_lwh_m=size_lwh,
                            avg_speed_mps=avg_speed,
                            start_xyz=xyz,
                            start_heading_rad=heading,
                        )

                    agent_list.append(agent)

                    if ego_start_lanes is not None:
                        return lane_avail_lengths,lane_avail_s,agent_list


        return lane_avail_lengths,lane_avail_s,agent_list


    # ---- ego route (unchanged API, uses seeded rng) ----
    def random_ego_edge_route(self, *, seed: Optional[int]=0, min_len_m: float=30.0, max_len_m: float=5000.0,
                              attempts: int=200, weight_attr: str="weight",
                              sample_start_on_edge: bool=True, end_at_last_point: bool=True
                              ) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        EID = self._build_edge_id_graph(weight_attr)
        all_ids = list(EID.nodes())
        if not all_ids: raise RuntimeError("Edge-ID graph is empty.")
        lengths = np.array([self.edge_lengths.get(e, 1.0) for e in all_ids], float)

        lengths=np.clip(lengths-20,a_min=0,a_max=1000)

        probs = lengths / (lengths.sum() if lengths.sum() > 0 else 1.0)
        for _ in range(attempts):
            s_eid = _key_id(rng.choice(all_ids, p=probs))
            g_eid = _key_id(rng.choice(all_ids, p=probs))
            try:
                edge_path = nx.shortest_path(EID, s_eid, g_eid, weight="weight")
            except nx.NetworkXNoPath:
                continue
            parts=[]
            for i,eid in enumerate(edge_path):
                P = self.edge_shapes[eid]
                if i==0 and sample_start_on_edge:
                    start_xyz, s0 = _sample_point_on_shape_with_s(P, rng)
                    ego_start_xy = start_xyz[:2]
                    parts.append(_slice_from_s_to_s(P, _arclen2d(P[:, :2]), s0, _arclen2d(P[:, :2])[-1]))
                else:
                    if i and parts and np.allclose(parts[-1][-1,:2], P[0,:2], atol=1e-6):
                        parts[-1] = parts[-1][:-1]
                    parts.append(P)
            if not parts: continue
            ego_route_xyz = np.vstack(parts)
            last_P = self.edge_shapes[edge_path[-1]]
            ego_goal_xy = last_P[-1,:2] if end_at_last_point else ego_route_xyz[-1,:2]
            dsum = float(np.sum(np.linalg.norm(np.diff(ego_route_xyz[:, :2], axis=0), axis=1)))
            if min_len_m <= dsum <= max_len_m:
                return edge_path, ego_route_xyz, ego_start_xy, ego_goal_xy
        raise RuntimeError("Failed to sample a random ego route within length bounds.")

    def _route(self,  s_xy: np.ndarray, g_xy: np.ndarray):
        EID = self._build_edge_id_graph("weight")
        """
        Route using edge-IDs and self.edge_shapes; distance accounts for start/goal offsets.
        Returns: path_eids, dist_m, route_xyz, start_eid, goal_eid
        """
        s_pt = Point(float(s_xy[0]), float(s_xy[1]))
        g_pt = Point(float(g_xy[0]), float(g_xy[1]))

        # --- snap start & goal to the best edge-id by geometric distance

        best_s_list = []  # (dist, eid, s_on_edge)
        best_g_list = []  # (dist, eid, s_on_edge)

        for eid, P in self.edge_shapes.items():
            P = np.asarray(P, float)
            if P.shape[0] < 2:
                continue
            ls = LineString(P[:, :2])

            ss = float(ls.project(s_pt))
            sg = float(s_pt.distance(ls))

            gg = float(ls.project(g_pt))
            dg = float(g_pt.distance(ls))

            if sg < 40:
                best_s_list.append((sg, eid, ss))
            if dg < 40:
                best_g_list.append((dg, eid, gg))

        if len(best_s_list)==0:
            raise RuntimeError("start place more than 40 m from road edge.")

        if len(best_g_list)==0:
            raise RuntimeError("goal place more than 40 m from road edge")

        best_sum_dist=100
        best_path=None

        for best_s in best_s_list:
            for best_g in best_g_list:

                start_gap,start_eid, s_off = best_s
                end_gap,goal_eid, g_off = best_g

                # --- shortest path in the edge-id graph
                try:
                    path = nx.shortest_path(EID, start_eid, goal_eid, weight="weight")
                    best_dist=start_gap+end_gap
                    if best_dist<best_sum_dist:
                        best_sum_dist=best_dist
                        best_path=path,s_off,g_off
                except nx.NetworkXNoPath:
                    continue

        if best_path is None:
            raise RuntimeError(f"No path from {s_xy} to {g_xy}")

        path,s_off,g_off=best_path

        # --- build trimmed geometry and true distance
        parts = []
        dist_m = 0.0

        if len(path) == 1:
            # start and goal on the same edge
            P = np.asarray(self.edge_shapes[path[0]], float)
            s_tab = _arclen2d(P[:, :2])
            s0, s1 = float(min(s_off, g_off)), float(max(s_off, g_off))
            parts.append(_slice_from_s_to_s(P, s_tab, s0, s1))
            dist_m += (s1 - s0)
        else:
            for i, eid in enumerate(path):
                P = np.asarray(self.edge_shapes[eid], float)
                s_tab = _arclen2d(P[:, :2])
                if s_tab[-1] <= 0.0:
                    continue

                if i == 0:
                    # first edge: trim from start offset to end
                    seg = _slice_from_s_to_s(P, s_tab, float(s_off), float(s_tab[-1]))
                    parts.append(seg)
                    dist_m += (float(s_tab[-1]) - float(s_off))
                elif i == len(path) - 1:
                    # last edge: trim from 0 to goal offset
                    seg = _slice_from_s_to_s(P, s_tab, 0.0, float(g_off))
                    # stitch: remove duplicate first point if same as previous end
                    if parts and np.allclose(parts[-1][-1, :2], seg[0, :2], atol=1e-6):
                        seg = seg[1:]
                    parts.append(seg)
                    dist_m += float(g_off)
                else:
                    # middle edges: full length
                    seg = P
                    if parts and np.allclose(parts[-1][-1, :2], seg[0, :2], atol=1e-6):
                        seg = seg[1:]
                    parts.append(seg)
                    dist_m += float(s_tab[-1])

        route_xyz = np.vstack(parts) if parts else np.zeros((0, 3))
        return path, dist_m, route_xyz, start_eid, goal_eid

    def generate_batch(
            self,
            density01: float,
            class_ratio: Dict[str, float],
            ego_edge_ids: Optional[List[Any]] = None,
            ego_route_xyz: Optional[np.ndarray] = None,
            neighbor_hops: int = 3,
            neighbor_mode: str = "both",
            seed: Optional[int] = 42,
            size_table: Optional[Dict[str, Tuple[float,float,float]]] = None,
            avg_speed_override: Optional[Dict[str, float]] = None,
            ped_min_len_m: float = 20.0,
            ped_max_len_m: float = 200.0,
            ped_forward_prob: float = 0.7
    ) -> List[Dict[str, Any]]:

        # single RNG for the whole generation → deterministic
        rng = np.random.default_rng(seed)

        # hop-connected edges for vehicles
        candidate_edges = sorted(self._edges_connected_to_ego_edges(
            ego_edge_ids, hops=neighbor_hops, mode=neighbor_mode
        ))
        if not candidate_edges:
            candidate_edges = [ _key_id(ed["id"]) for _, _, ed in self.EG.edges(data=True) ]

        # defaults
        DEFAULT_CLASS_SIZES ={"pedestrian":(0.5,0.5,1.7), "bicycle":(1.8,0.6,1.6), "car":(4.4,1.8,1.6), "truck":(12.0,2.5,3.6)}
        DEFAULT_CLASS_SPEED_MPS = {"pedestrian":1.4, "bicycle":4.5*(1.1-density01), "car":13.9*(1.1-density01), "truck":11.1*(1.1-density01)}
        #STYLE_SPEED_SCALE = {"conservative":0.85, "normal":1.0, "aggressive":1.15}
        size_tab = size_table or DEFAULT_CLASS_SIZES
        #styles   = style_table or {}
        speeds   = dict(DEFAULT_CLASS_SPEED_MPS)
        if avg_speed_override: speeds.update(avg_speed_override)

        lane_avail_lengths={}
        lane_avail_s = {}
        self.sL={}

        for eid in candidate_edges:
            lanes=self.edge_member_lanes.get(eid)
            for id in lanes:
                lane=self.lane_xyz[id]

                lane_length=_arclen2d(lane[:, :2])
                lane_avail_lengths[id] =lane_length[-1]
                lane_avail_s[id]= [np.array([0,lane_length[-1]])]

                self.sL[id]=lane_length

        ego_start_lanes=self.edge_member_lanes.get(ego_edge_ids[0])

        lane_avail_lengths,lane_avail_s,agent_list=self.sample_agent(lane_avail_lengths,lane_avail_s,class_ratio,speeds,size_tab,ego_start_lanes)

        lane_avail_lengths,lane_avail_s,agent_list=self.sample_agent(lane_avail_lengths,lane_avail_s,class_ratio,speeds,size_tab,None)

        all_agent_num=len(agent_list)

        ped_num=int(all_agent_num/(class_ratio['car']+class_ratio['bicycle']+class_ratio['truck']) *class_ratio['pedestrian'])

        for i in range(ped_num):
            b_idx = int(rng.integers(0, len(self.boundaries)))
            B = self.boundaries[b_idx]; sB = _arclen2d(B[:, :2]); Lb = sB[-1]
            if Lb <= 0: continue
            s0 = float(rng.uniform(0.0, Lb))
            dir_sign = +1 if rng.random() < ped_forward_prob else -1
            seg_len = float(rng.uniform(ped_min_len_m, ped_max_len_m))
            s1 = min(Lb, s0 + seg_len) if dir_sign > 0 else max(0.0, s0 - seg_len)
            route_xyz = _slice_from_s_to_s(B, sB, s0, s1)
            start_xyz = route_xyz[0]
            start_heading = _heading_at_s_dir(B, sB, s0, dir_sign=dir_sign)
            agent = dict(
                agent_id=f"A{len(agent_list)}",
                cls='pedestrian',
                size_lwh_m=size_tab['pedestrian'],
                avg_speed_mps=DEFAULT_CLASS_SPEED_MPS["pedestrian"],
                start_xyz=start_xyz,
                start_heading_rad=start_heading,
            )

            agent_list.append(agent)

        return agent_list
