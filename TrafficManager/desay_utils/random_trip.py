# -*- coding: utf-8 -*-
# TrafficGenerator with pedestrians following boundaries; vehicles on lanes/edges
from __future__ import annotations
from typing import Iterable, Set, List, Optional, Dict, Any, Tuple
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point

# ------------------ utilities ------------------

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

        for _, _, ed in EG.edges(data=True):
            eid = _key_id(ed["id"])
            geom = ed.get("geom", ed.get("shape_xyz", None))
            if geom is None:
                # last resort: straight line between node xyz if available
                pu = EG.nodes[_key_id(_[0])].get("xyz") if _ else None  # not reliable in this scope
                pv = EG.nodes[_key_id(_[1])].get("xyz") if _ else None
                if pu is not None and pv is not None:
                    geom = np.vstack([_to_xyz(pu)[0], _to_xyz(pv)[0]])
                else:
                    raise ValueError(f"Edge {eid} missing 'geom'/'shape_xyz' and node xyz fallback failed.")
            P = _to_xyz(geom)
            self.edge_shapes[eid] = P
            self.edge_lengths[eid] = float(ed.get("length_m", _arclen2d(P[:, :2])[-1]))
            # member lane IDs (keep as keys)
            lanes = ed.get("lanes", [])
            self.edge_member_lanes[eid] = [_key_id(l) for l in lanes] if lanes is not None else []

        # lane centerlines from lane graph EDGES (kind == 'lane')
        self.lane_xyz: Dict[Any, np.ndarray] = {}
        for u, v, ed in G_lane.edges(data=True):
            if ed.get("kind") == "lane":
                lid = _key_id(ed.get("id", ed.get("edge_id", f"{u}->{v}")))
                geom = ed.get("geom", None)
                if geom is None:
                    # fallback from u/v node xyz if present
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

    # ---- ego route (unchanged API, uses new edge_shapes) ----
    def random_ego_edge_route(self, *, seed: Optional[int]=0, min_len_m: float=30.0, max_len_m: float=5000.0,
                              attempts: int=200, weight_attr: str="weight",
                              sample_start_on_edge: bool=True, end_at_last_point: bool=True
                              ) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        EID = self._build_edge_id_graph(weight_attr)
        all_ids = list(EID.nodes())
        if not all_ids: raise RuntimeError("Edge-ID graph is empty.")
        lengths = np.array([self.edge_lengths.get(e, 1.0) for e in all_ids], float)
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

    # ---- helper: dedup ----
    def _start_conflict(self, new_xy: np.ndarray, new_edge: Any, new_s: float,
                        taken_xy: List[np.ndarray], taken_per_edge_s: Dict[Any, List[float]],
                        min_dist: float, min_same_edge_s: float) -> bool:
        for p in taken_xy:
            if np.linalg.norm(new_xy[:2] - p[:2]) < min_dist: return True
        k = _key_id(new_edge)
        if k in taken_per_edge_s:
            for s in taken_per_edge_s[k]:
                if abs(s - new_s) < min_same_edge_s: return True
        return False

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
        """Return start_xyz, heading_rad (forward), and s0 on boundary b_idx."""
        B = self.boundaries[b_idx]; sB = _arclen2d(B[:, :2]); L = sB[-1]
        if L <= 0: return B[0], 0.0, 0.0
        s0 = float(rng.uniform(0.0, L))
        heading = _heading_at_s_dir(B, sB, s0, dir_sign=+1)
        xyz = _interp_xyz_at_s(B, sB, s0)
        return xyz, heading, s0

    def _spawn_on_random_lane_of_edge(self, start_edge_id: Any, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
        """Return start_xyz, heading_rad, and s (projected onto edge shape for de-dup)."""
        eid = _key_id(start_edge_id)
        # gather lane members from EG (lane edge IDs from G_lane)
        member_lane_ids = self.edge_member_lanes.get(eid, [])
        chosen_geom = None
        if member_lane_ids:
            # filter to those we actually have in lane_xyz dict
            avail = [lid for lid in member_lane_ids if lid in self.lane_xyz]
            if avail:
                lid = _key_id(rng.choice(avail))
                L = self.lane_xyz[lid]
                if len(L) >= 2:
                    sL = _arclen2d(L[:, :2])
                    u = float(rng.uniform(0.0, sL[-1] if sL[-1] > 0 else 0.0))
                    xyz = _interp_xyz_at_s(L, sL, u)
                    heading = _heading_at_s_dir(L, sL, u, dir_sign=+1)
                    # project sample onto the representative edge geometry for s bookkeeping
                    ls_edge = LineString(self.edge_shapes[eid][:, :2])
                    s_on = float(ls_edge.project(Point(float(xyz[0]), float(xyz[1]))))
                    return xyz, heading, s_on
                chosen_geom = L
        # fallback to representative edge geometry
        P = self.edge_shapes[eid]; sP = _arclen2d(P[:, :2])
        u = float(rng.uniform(0.0, sP[-1] if sP[-1] > 0 else 0.0))
        xyz = _interp_xyz_at_s(P, sP, u); heading = _heading_at_s_dir(P, sP, u, dir_sign=+1)
        return xyz, heading, u

    # ------------------ main: batch ------------------

    def generate_batch(
        self,
        density01: float,
        class_ratio: Dict[str, float],
        *,
        ego_edge_ids: Optional[List[Any]] = None,
        ego_route_xyz: Optional[np.ndarray] = None,
        ego_auto_min_len_m: float = 300.0,
        ego_auto_max_len_m: float = 5000.0,
        ego_seed: Optional[int] = 0,
        neighbor_hops: int = 3,
        neighbor_mode: str = "both",
        seed: Optional[int] = 42,
        min_route_m: float = 50.0,
        max_route_m: float = 5000.0,
        size_table: Optional[Dict[str, Tuple[float,float,float]]] = None,
        style_table: Optional[Dict[str, str]] = None,
        avg_speed_override: Optional[Dict[str, float]] = None,
        lift_to_lane_ids: bool = False,
        max_attempts_per_agent: int = 120,
        min_start_spacing_m: float = 4.0,
        min_same_edge_s_m: float = 12.0,
        # --- NEW pedestrian params ---
        ped_min_len_m: float = 20.0,
        ped_max_len_m: float = 200.0,
        ped_forward_prob: float = 0.7     # chance to move increasing arclength
    ) -> List[Dict[str, Any]]:

        rng = np.random.default_rng(seed)

        # ego route (for vehicle candidate edges)
        if ego_edge_ids is None and ego_route_xyz is None:
            ego_edge_ids, ego_route_xyz, _, _ = self.random_ego_edge_route(
                seed=ego_seed, min_len_m=ego_auto_min_len_m, max_len_m=ego_auto_max_len_m
            )
        if not ego_edge_ids:
            ego_edge_ids = [ _key_id(ed["id"]) for _, _, ed in self.EG.edges(data=True) ]

        # hop-connected edges for vehicles
        candidate_edges = sorted(self._edges_connected_to_ego_edges(
            ego_edge_ids, hops=neighbor_hops, mode=neighbor_mode
        ))
        if not candidate_edges:
            candidate_edges = [ _key_id(ed["id"]) for _, _, ed in self.EG.edges(data=True) ]

        # capacity & allocation
        HEADWAY_M = {"pedestrian":2.0, "bicycle":5.0, "car":12.0, "truck":20.0}
        L = float(sum(self.edge_lengths[e] for e in candidate_edges))
        base_headway = HEADWAY_M["car"]
        N_base = int(np.floor(np.clip(density01, 0, 1) * (L / base_headway)))

        keys = list(class_ratio.keys())
        vals = np.array([max(0.0, float(class_ratio[k])) for k in keys], float)
        if vals.sum() <= 0: vals = np.ones_like(vals)
        probs_class = vals / vals.sum()
        alloc = {k: int(np.floor(N_base * p)) for k, p in zip(keys, probs_class)}
        rem = N_base - sum(alloc.values()); i = 0
        while rem > 0 and keys:
            alloc[keys[i % len(keys)]] += 1; rem -= 1; i += 1

        # vehicle edge sampling weights
        cand_edges = np.array(candidate_edges, dtype=object)
        w_edges = np.array([self.edge_lengths[e] for e in cand_edges], float)
        probs_edges = w_edges / (w_edges.sum() if w_edges.sum() > 0 else 1.0)

        # defaults
        DEFAULT_CLASS_SIZES = {"pedestrian":(0.5,0.5,1.7), "bicycle":(1.8,0.6,1.6), "car":(4.4,1.8,1.6), "truck":(12.0,2.5,3.6)}
        DEFAULT_CLASS_SPEED_MPS = {"pedestrian":1.4, "bicycle":4.5, "car":13.9, "truck":11.1}
        STYLE_SPEED_SCALE = {"conservative":0.85, "normal":1.0, "aggressive":1.15}
        size_tab = size_table or DEFAULT_CLASS_SIZES
        styles   = style_table or {}
        speeds   = dict(DEFAULT_CLASS_SPEED_MPS)
        if avg_speed_override: speeds.update(avg_speed_override)

        # dedup caches
        taken_starts_xy: List[np.ndarray] = []
        taken_s_per_key: Dict[Any, List[float]] = {}

        # lightweight router for vehicles
        def _route(EG: nx.DiGraph, s_xy: np.ndarray, g_xy: np.ndarray):
            EID = self._build_edge_id_graph("weight")
            best_s = best_g = None
            s_pt = Point(float(s_xy[0]), float(s_xy[1])); g_pt = Point(float(g_xy[0]), float(g_xy[1]))
            for _, _, ed in EG.edges(data=True):
                eid = _key_id(ed["id"])
                P = self.edge_shapes[eid]
                ls = LineString(P[:, :2])
                ss = ls.project(s_pt); s_q = ls.interpolate(ss)
                sg = float(s_pt.distance(s_q))
                gg = ls.project(g_pt); g_q = ls.interpolate(gg)
                dg = float(g_pt.distance(g_q))
                if (best_s is None) or (sg < best_s[0]): best_s = (sg, eid)
                if (best_g is None) or (dg < best_g[0]): best_g = (dg, eid)
            if best_s is None or best_g is None: raise RuntimeError("snap failed")
            s_eid, g_eid = best_s[1], best_g[1]
            path = nx.shortest_path(EID, s_eid, g_eid, weight="weight")
            parts: List[np.ndarray] = []; dist = 0.0
            for i, e in enumerate(path):
                P = self.edge_shapes[e]
                if i and parts and np.allclose(parts[-1][-1,:2], P[0,:2], atol=1e-6):
                    parts[-1] = parts[-1][:-1]
                parts.append(P); dist += self.edge_lengths[e]
            return path, dist, (np.vstack(parts) if parts else np.zeros((0,3))), s_eid, g_eid

        agents: List[Dict[str, Any]] = []
        id_counter = 0

        # --- generate ---
        for cls, n in alloc.items():
            if n <= 0: continue
            size_lwh = size_tab.get(cls, (4.0, 1.8, 1.6))
            sf = STYLE_SPEED_SCALE.get(styles.get(cls, "normal"), 1.0)
            avg_speed = float(speeds.get(cls, 10.0) * sf)

            tries = 0; made = 0
            while made < n and tries < n * max_attempts_per_agent:
                tries += 1

                # VEHICLES: hop-connected start/goal on edges
                if cls in self.vehicle_classes:
                    s_eid = _key_id(np.random.default_rng().choice(cand_edges, p=probs_edges))
                    g_eid = _key_id(np.random.default_rng().choice(cand_edges, p=probs_edges))
                    # spawn on a lane of start edge (heading from lane), de-dup uses s on start edge
                    Ps, heading_rad, s_on = self._spawn_on_random_lane_of_edge(s_eid, rng)
                    Pg, _ = _sample_point_on_shape_with_s(self.edge_shapes[g_eid], rng)
                    # conflict check (key = start edge)
                    if self._start_conflict(Ps, s_eid, s_on, taken_starts_xy, taken_s_per_key,
                                            min_start_spacing_m, min_same_edge_s_m):
                        continue
                    # route over edge graph
                    try:
                        if self.router is not None:
                            edge_path, dist_m, route_xyz, se, ge = self.router(self.EG, Ps[:2], Pg[:2], self.G_lane, "weight")
                        else:
                            edge_path, dist_m, route_xyz, se, ge = _route(self.EG, Ps[:2], Pg[:2])
                    except Exception:
                        continue
                    if not (min_route_m <= dist_m <= max_route_m): continue
                    taken_starts_xy.append(Ps.copy())
                    taken_s_per_key.setdefault(_key_id(s_eid), []).append(float(s_on))
                    agents.append(dict(
                        agent_id=f"A{id_counter}", cls=cls,
                        size_lwh_m=size_lwh, avg_speed_mps=avg_speed,
                        start_xyz=Ps, start_heading_rad=heading_rad,
                        goal_xyz=Pg,
                        edge_ids=[_key_id(e) for e in edge_path],
                        route_xyz=route_xyz,
                        lane_ids=self.edge_member_lanes.get(_key_id(s_eid), None) if lift_to_lane_ids else None,
                        spawn_source="lane_or_edge", path_type="edge_graph"
                    ))
                    id_counter += 1; made += 1
                    continue

                # PEDESTRIANS: move along a boundary
                if cls in self.pedestrian_classes and self.boundaries:
                    # choose a boundary
                    b_idx = int(rng.integers(0, len(self.boundaries)))
                    B = self.boundaries[b_idx]; sB = _arclen2d(B[:, :2]); Lb = sB[-1]
                    if Lb <= 0: continue
                    # spawn point s0 and direction
                    s0 = float(rng.uniform(0.0, Lb))
                    dir_sign = +1 if rng.random() < ped_forward_prob else -1
                    # choose travel length
                    seg_len = float(rng.uniform(ped_min_len_m, ped_max_len_m))
                    s1 = min(Lb, s0 + seg_len) if dir_sign > 0 else max(0.0, s0 - seg_len)
                    # slice boundary & heading at start in travel direction
                    route_xyz = _slice_from_s_to_s(B, sB, s0, s1)
                    start_xyz = route_xyz[0]
                    start_heading = _heading_at_s_dir(B, sB, s0, dir_sign=dir_sign)
                    goal_xyz = route_xyz[-1]
                    # de-dup: key by ("B", b_idx) and s0
                    if self._start_conflict(start_xyz, ("B", b_idx), s0, taken_starts_xy, taken_s_per_key,
                                            min_start_spacing_m, min_same_edge_s_m):
                        continue
                    # length check against ped limits
                    dsum = float(np.sum(np.linalg.norm(np.diff(route_xyz[:, :2], axis=0), axis=1)))
                    if not (ped_min_len_m <= dsum <= ped_max_len_m): continue
                    taken_starts_xy.append(start_xyz.copy())
                    taken_s_per_key.setdefault(("B", b_idx), []).append(float(s0))
                    agents.append(dict(
                        agent_id=f"A{id_counter}", cls=cls,
                        size_lwh_m=size_lwh, avg_speed_mps=avg_speed,
                        start_xyz=start_xyz, start_heading_rad=start_heading,
                        goal_xyz=goal_xyz,
                        edge_ids=[],            # pedestrians don't use edge routing here
                        route_xyz=route_xyz,    # boundary-following path
                        lane_ids=None,
                        boundary_id=b_idx,
                        spawn_source="boundary", path_type="boundary"
                    ))
                    id_counter += 1; made += 1
                    continue

                # Fallback for other classes: use vehicle logic
                s_eid = _key_id(np.random.default_rng().choice(cand_edges, p=probs_edges))
                g_eid = _key_id(np.random.default_rng().choice(cand_edges, p=probs_edges))
                Ps, heading_rad, s_on = self._spawn_on_random_lane_of_edge(s_eid, rng)
                Pg, _ = _sample_point_on_shape_with_s(self.edge_shapes[g_eid], rng)
                if self._start_conflict(Ps, s_eid, s_on, taken_starts_xy, taken_s_per_key,
                                        min_start_spacing_m, min_same_edge_s_m):
                    continue
                try:
                    if self.router is not None:
                        edge_path, dist_m, route_xyz, se, ge = self.router(self.EG, Ps[:2], Pg[:2], self.G_lane, "weight")
                    else:
                        edge_path, dist_m, route_xyz, se, ge = _route(self.EG, Ps[:2], Pg[:2])
                except Exception:
                    continue
                if not (min_route_m <= dist_m <= max_route_m): continue
                taken_starts_xy.append(Ps.copy())
                taken_s_per_key.setdefault(_key_id(s_eid), []).append(float(s_on))
                agents.append(dict(
                    agent_id=f"A{id_counter}", cls=cls,
                    size_lwh_m=size_lwh, avg_speed_mps=avg_speed,
                    start_xyz=Ps, start_heading_rad=heading_rad,
                    goal_xyz=Pg,
                    edge_ids=[_key_id(e) for e in edge_path],
                    route_xyz=route_xyz,
                    lane_ids=self.edge_member_lanes.get(_key_id(s_eid), None) if lift_to_lane_ids else None,
                    spawn_source="lane_or_edge", path_type="edge_graph"
                ))
                id_counter += 1; made += 1

        return agents
