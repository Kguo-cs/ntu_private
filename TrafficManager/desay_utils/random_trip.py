# --- 新增/替换的 TrafficGenerator（支持随机 ego 路径 + 起点去重） ---
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import numpy as np
import networkx as nx

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("expect [N,3]/[N,2]/[3,N]")
    return a

# ---- paste to replace TrafficGenerator.random_ego_edge_route ----
def _arclen2d(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _interp_xyz_at_s(xyz: np.ndarray, s_tab: np.ndarray, s: float) -> np.ndarray:
    x = np.interp(s, s_tab, xyz[:,0]); y = np.interp(s, s_tab, xyz[:,1]); z = np.interp(s, s_tab, xyz[:,2])
    return np.array([x, y, z], float)

def _slice_from_s_to_end(P: np.ndarray, s0: float) -> np.ndarray:
    """从 polyline 的弧长 s0 切到终点（含 s0/终点插值）"""
    P = np.asarray(P, float)
    s = _arclen2d(P[:, :2])
    s0 = float(np.clip(s0, 0.0, s[-1] if len(s) else 0.0))
    # 构造分段：s0 -> end
    if len(P) < 2 or s[-1] <= 0:
        return P.copy()
    pts = [_interp_xyz_at_s(P, s, s0)]
    mask = (s > s0) & (s < s[-1])
    if np.any(mask):
        pts.extend(list(P[mask]))
    pts.append(P[-1])
    return np.asarray(pts)

def _sample_point_on_shape_with_s(shape_xyz: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    P = np.asarray(shape_xyz, float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    s = _arclen2d(P[:, :2])
    if s[-1] <= 0:
        return P[0], 0.0
    u = float(rng.uniform(0.0, s[-1]))
    return _interp_xyz_at_s(P, s, u), u

def build_edge_id_graph(EG: nx.DiGraph, weight_attr: str = "weight") -> nx.DiGraph:
    id2uv = {}; w_of = {}
    for u, v, ed in EG.edges(data=True):
        eid = ed["id"]; id2uv[eid] = (u, v)
        w_of[eid] = float(ed.get(weight_attr, ed.get("weight", 1.0)))
    EID = nx.DiGraph()
    for eid in id2uv.keys(): EID.add_node(eid)
    allowed = EG.graph.get("allowed_edge_turns", None)
    if allowed:
        for e1, e2 in allowed:
            if e1 in id2uv and e2 in id2uv:
                EID.add_edge(e1, e2, weight=w_of[e2])
    else:
        for e1, (u1, v1) in id2uv.items():
            for e2, (u2, _) in id2uv.items():
                if v1 == u2:
                    EID.add_edge(e1, e2, weight=w_of[e2])
    return EID

class TrafficGenerator:
    def __init__(self, EG: nx.DiGraph, G_lane: nx.DiGraph, router_func=None):
        self.EG = EG
        self.G_lane = G_lane
        self.router = router_func  # 可传你自己的；不传则用内部轻量路由
        # 边形状&长度缓存
        self.edge_shapes = {}
        self.edge_lengths = {}
        for _, _, ed in EG.edges(data=True):
            eid = ed["id"]
            P = _to_xyz(ed["shape_xyz"])
            self.edge_shapes[eid] = P
            self.edge_lengths[eid] = float(ed.get("length_m", _arclen2d(P[:, :2])[-1]))

    # ========== 新增：随机生成 ego 的 edge 路径 ==========

    def random_ego_edge_route(
            self,
            *,
            seed: int | None = 0,
            min_len_m: float = 30.0,
            max_len_m: float = 5000.0,
            attempts: int = 200,
            weight_attr: str = "weight",
            restrict_to_largest_scc: bool = False,
            sample_start_on_edge: bool = True,  # ✅ 新增：起点随机采样在起始 edge 上
            end_at_last_point: bool = True  # ✅ 新增：终点为末尾 edge 的最后一个点
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        返回:
          - ego_edge_ids: List[str]
          - ego_route_xyz: np.ndarray  [M,3]，已按“起点随机采样 & 末尾 edge 终点”裁剪
          - ego_start_xy: np.ndarray   [2]  起点 XY
          - ego_goal_xy:  np.ndarray   [2]  终点 XY（末尾 edge 最末点）
        """
        rng = np.random.default_rng(seed)
        # 1) Edge-ID 图与候选集合
        EID = build_edge_id_graph(self.EG, weight_attr=weight_attr)
        all_ids = list(EID.nodes())
        if not all_ids:
            raise RuntimeError("Edge-ID graph is empty.")
        if restrict_to_largest_scc:
            sccs = list(nx.strongly_connected_components(EID))
            cand = list(max(sccs, key=len)) if sccs else all_ids
        else:
            cand = all_ids

        # 按长度加权抽样起终边
        lengths = np.array([self.edge_lengths.get(e, 1.0) for e in cand], float)
        probs = lengths / (lengths.sum() if lengths.sum() > 0 else 1.0)

        for _ in range(attempts):
            s_eid = rng.choice(cand, p=probs)
            g_eid = rng.choice(cand, p=probs)
            # if len(cand) > 1 and g_eid == s_eid:
            #     continue
            # 2) 最短路（edge-id 图）
            try:
                edge_path = nx.shortest_path(EID, s_eid, g_eid, weight="weight")
            except nx.NetworkXNoPath:
                continue

            # 3) 组装几何：默认整段，随后按需“起点随机 & 末尾终点”
            parts = []
            total = 0.0
            for i, eid in enumerate(edge_path):
                P = self.edge_shapes.get(eid)
                if P is None: break
                # 起始 edge: 若启用起点随机，则先采样弧长 s0 后切片
                if i == 0 and sample_start_on_edge:
                    start_xyz, s0 = _sample_point_on_shape_with_s(P, rng)
                    first = _slice_from_s_to_end(P, s0)
                    ego_start_xy = start_xyz[:2]
                    parts.append(first)
                else:
                    # 中间/尾段按全段（尾段“全段”+ end_at_last_point == True 等价于末点作为终点）
                    if i > 0 and parts and np.allclose(parts[-1][-1, :2], P[0, :2], atol=1e-6):
                        parts[-1] = parts[-1][:-1]
                    parts.append(P)

                total += float(self.edge_lengths.get(eid, _arclen2d(P[:, :2])[-1]))

            if not parts:
                continue

            ego_route_xyz = np.vstack(parts)

            # 末尾终点：取最后一个点（默认已满足 end_at_last_point）
            last_P = self.edge_shapes[edge_path[-1]]
            ego_goal_xy = last_P[-1, :2] if end_at_last_point else ego_route_xyz[-1, :2]

            # 路径总长检查（注意：起点切片后总长度会变短，更贴近真实）
            # 估长用几何再算一次：
            seg = ego_route_xyz[:, :2]
            dsum = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
            if min_len_m <= dsum <= max_len_m:
                return edge_path, ego_route_xyz, ego_start_xy, ego_goal_xy

        raise RuntimeError(
            "Failed to sample a random ego route within length bounds (with start-on-edge & end-at-last).")

    # ========== 新增：起点冲突检查（去重） ==========
    def _start_conflict(self,
                        new_xy: np.ndarray,
                        new_edge: str,
                        new_s_along_edge: float,
                        taken_xy: list[np.ndarray],
                        taken_per_edge_s: dict[str, list[float]],
                        min_start_spacing_m: float,
                        min_same_edge_s_m: float) -> bool:
        # 全局欧氏距离
        for p in taken_xy:
            if np.linalg.norm(new_xy[:2] - p[:2]) < min_start_spacing_m:
                return True
        # 同一 Edge 的弧长最小间距
        if new_edge in taken_per_edge_s:
            for s in taken_per_edge_s[new_edge]:
                if abs(s - new_s_along_edge) < min_same_edge_s_m:
                    return True
        return False

    def _edges_within_distance_of_ego(
            self,
            ego_edge_ids: list[str],
            *,
            max_dist_m: float = 100.0,
            mode: str = "to_ego",  # 'to_ego' | 'from_ego' | 'either'
            allowed_turns_only: bool = True,  # use allowed-turn graph; else junction-topology
            include_self: bool = True,
    ) -> list[str]:
        """
        Return edge IDs whose shortest-path distance (sum of edge lengths) to/from
        the ego edge set is <= max_dist_m, following turn legality.

        mode:
          - 'to_ego'   : can move TO any ego edge within max_dist_m (directed).
          - 'from_ego' : can move FROM ego edges to them within max_dist_m.
          - 'either'   : union of both directions.

        Note: cost is per-edge length; we ignore partial lengths within the first/last edge.
        """
        if not ego_edge_ids:
            return []

        # Build edge-ID graph
        if allowed_turns_only:
            G = build_edge_id_graph(self.EG, weight_attr="weight")  # DiGraph
        else:
            # looser: edges connect if they share a junction (undirected)
            id2uv = {}
            G = nx.DiGraph()
            for u, v, ed in self.EG.edges(data=True):
                eid = ed["id"];
                id2uv[eid] = (u, v);
                G.add_node(eid)
            # connect v1==u2
            for e1, (u1, v1) in id2uv.items():
                for e2, (u2, _) in id2uv.items():
                    if v1 == u2:
                        w = float(self.edge_lengths.get(e2, 1.0))
                        G.add_edge(e1, e2, weight=w)

        seeds = [e for e in ego_edge_ids if e in G]
        if not seeds:
            return []

        def _within(Gdir):
            dist = nx.multi_source_dijkstra_path_length(Gdir, sources=seeds,
                                                        cutoff=float(max_dist_m), weight="weight")
            out = set(dist.keys())
            if include_self:
                out |= set(seeds)
            return out

        if mode == "to_ego":
            Gdir = G.reverse(copy=False)  # reverse to measure "distance to ego"
            out = _within(Gdir)
        elif mode == "from_ego":
            out = _within(G)  # forward distances from ego
        elif mode == "either":
            out = _within(G) | _within(G.reverse(copy=False))
        else:
            raise ValueError("mode must be 'to_ego', 'from_ego', or 'either'.")

        return sorted(out)

    # ========== 批量生成（支持随机 ego + 起点去重） ==========
    def generate_batch(
        self,
        density01: float,
        class_ratio: dict[str, float],
        *,
        ego_edge_ids: list[str] | None = None,
        ego_route_xyz: np.ndarray | None = None,
        use_distance_to_ego: bool = True,  # ✅ turn on distance-based selection
        move_to_ego_max_m: float = 100.0,  # ✅ 100 m by your spec
        distance_mode: str = "to_ego",  # 'to_ego' | 'from_ego' | 'either'
        distance_allowed_turns: bool = True,
        ego_auto_min_len_m: float = 300.0,
        ego_auto_max_len_m: float = 5000.0,
        ego_seed: int | None = 0,
        seed: int | None = 42,
        min_route_m: float = 50.0,
        max_route_m: float = 5000.0,
        size_table: dict[str, tuple[float,float,float]] | None = None,
        style_table: dict[str, str] | None = None,
        avg_speed_override: dict[str, float] | None = None,
        lift_to_lane_ids: bool = False,
        corridor_radius_m: float = 60.0,
        max_attempts_per_agent: int = 120,
        # 起点去重参数：
        min_start_spacing_m: float = 4.0,        # 任意两起点的最小欧氏距离
        min_same_edge_s_m: float = 12.0          # 同一 Edge 上弧长方向的最小间距
    ) -> list:
        """
        若未提供 ego_edge_ids / ego_route_xyz，会自动随机生成一条 ego 路径，
        并将生成范围限制到其走廊（±corridor_radius_m）。
        同时对所有参与者进行起点冲突检查，不满足间距则重采样。
        """
        from shapely.geometry import LineString
        rng = np.random.default_rng(seed)

        # 1) 确定 ego 路径（若未提供则随机生成）
        if ego_edge_ids is None and ego_route_xyz is None:
            ego_edge_ids, ego_route_xyz = self.random_ego_edge_route(
                seed=ego_seed,
                min_len_m=ego_auto_min_len_m,
                max_len_m=ego_auto_max_len_m
            )

        # === Candidate edges selection ===
        if ego_edge_ids and use_distance_to_ego:
            candidate_edges = self._edges_within_distance_of_ego(
                ego_edge_ids,
                max_dist_m=float(move_to_ego_max_m),
                mode=distance_mode,
                allowed_turns_only=distance_allowed_turns,
                include_self=True
            )
        elif ego_edge_ids or (ego_route_xyz is not None and len(ego_route_xyz) >= 2):
            # (optional) fallback to corridor geometry if you still want it available
            candidate_edges = self._edges_connected_to_ego_edges(ego_edge_ids, hops=1)
        else:
            candidate_edges = [ed["id"] for _, _, ed in self.EG.edges(data=True)]

        if not candidate_edges:
            return []

        # 3) 容量估算 + 类别分配（沿用你之前的逻辑）
        HEADWAY_M = {"pedestrian":2.0,"bicycle":5.0,"car":12.0,"truck":20.0}
        L = float(sum(self.edge_lengths[e] for e in candidate_edges))
        base_headway = HEADWAY_M["car"]
        N_base = int(np.floor(np.clip(density01,0,1) * (L / base_headway)))

        keys = list(class_ratio.keys())
        vals = np.array([max(0.0, float(class_ratio[k])) for k in keys], float)
        if vals.sum() <= 0: vals = np.ones_like(vals)
        probs_class = vals / vals.sum()
        alloc = {k: int(np.floor(N_base * p)) for k, p in zip(keys, probs_class)}
        # 轻微修正到位
        gap = N_base - sum(alloc.values())
        i=0
        while gap>0 and keys:
            alloc[keys[i % len(keys)]] += 1; gap -= 1; i += 1

        # 4) 采样权重（按边长度）
        cand = np.array(candidate_edges)
        w_edges = np.array([self.edge_lengths[e] for e in cand], float)
        probs_edges = w_edges / (w_edges.sum() if w_edges.sum() > 0 else 1.0)

        # 5) 尺寸/速度表
        DEFAULT_CLASS_SIZES = {"pedestrian":(0.5,0.5,1.7),"bicycle":(1.8,0.6,1.6),"car":(4.4,1.8,1.6),"truck":(12.0,2.5,3.6)}
        DEFAULT_CLASS_SPEED_MPS = {"pedestrian":1.4,"bicycle":4.5,"car":13.9,"truck":11.1}
        STYLE_SPEED_SCALE = {"conservative":0.85,"normal":1.0,"aggressive":1.15}
        size_tab = size_table or DEFAULT_CLASS_SIZES
        styles = style_table or {}
        speeds = dict(DEFAULT_CLASS_SPEED_MPS)
        if avg_speed_override: speeds.update(avg_speed_override)

        # 6) 起点冲突缓存
        taken_starts_xy: list[np.ndarray] = []
        taken_s_per_edge: dict[str, list[float]] = {}

        # 7) 生成
        agents = []
        id_counter = 0

        # 简单内置路由（若未注入外部 router_func）
        def _route(EG, s_xy, g_xy):
            # 用 edge-id 图做最短路并拼几何（同我们之前的轻量实现）
            EID = build_edge_id_graph(EG, "weight")
            # 将 s/g 吸到走廊主干上（若有），增强相关性
            if lines:
                union = unary_union(lines)
                for xy in (s_xy, g_xy):
                    q = union.interpolate(union.project(Point(xy[0], xy[1])))
                    xy[:] = [q.x, q.y]
            # 找最近 edge（按代表形状）
            best_s = best_g = None
            for u,v,ed in EG.edges(data=True):
                eid = ed["id"]; P = _to_xyz(ed["shape_xyz"]); ls = LineString(P[:, :2])
                for tag, pt in (("s", s_xy), ("g", g_xy)):
                    ss = ls.project(Point(float(pt[0]), float(pt[1])))
                    q = ls.interpolate(ss)
                    d = float(Point(pt[0], pt[1]).distance(q))
                    rec = (d, eid)
                    if tag=="s":
                        if (best_s is None) or (d < best_s[0]): best_s = rec
                    else:
                        if (best_g is None) or (d < best_g[0]): best_g = rec
            if best_s is None or best_g is None: raise RuntimeError("snap failed")
            s_eid, g_eid = best_s[1], best_g[1]
            path = nx.shortest_path(EID, s_eid, g_eid, weight="weight")
            parts=[]; dist=0.0
            for i,e in enumerate(path):
                P = self.edge_shapes[e]
                if i and len(parts) and np.allclose(parts[-1][-1,:2], P[0,:2], atol=1e-6):
                    parts[-1]=parts[-1][:-1]
                parts.append(P); dist += self.edge_lengths[e]
            return path, dist, (np.vstack(parts) if parts else np.zeros((0,3))), s_eid, g_eid

        for cls, n in alloc.items():
            if n <= 0: continue
            size_lwh = size_tab.get(cls, (4.0,1.8,1.6))
            sf = STYLE_SPEED_SCALE.get(styles.get(cls, "normal"), 1.0)
            avg_speed = float(speeds.get(cls, 10.0) * sf)

            attempts = 0
            made = 0
            while made < n and attempts < n * max_attempts_per_agent:
                attempts += 1
                s_eid = rng.choice(cand, p=probs_edges)
                g_eid = rng.choice(cand, p=probs_edges)
                # if len(cand) > 1 and g_eid == s_eid:
                #     continue

                # 采样起点/终点以及起点在该 edge 上的弧长 s
                Ps, s_on_s = _sample_point_on_shape_with_s(self.edge_shapes[s_eid], rng)
                Pg, _       = _sample_point_on_shape_with_s(self.edge_shapes[g_eid], rng)

                # 起点冲突检查（欧氏 + 同边弧长）
                if self._start_conflict(Ps, s_eid, s_on_s, taken_starts_xy, taken_s_per_edge,
                                        min_start_spacing_m, min_same_edge_s_m):
                    continue  # 冲突则重采样

                # 路由（用外部 router 或内置轻量 router）
                try:
                    if self.router is not None:
                        edge_path, dist_m, route_xyz, se, ge = self.router(self.EG, Ps[:2], Pg[:2], self.G_lane, "weight")
                    else:
                        edge_path, dist_m, route_xyz, se, ge = _route(self.EG, Ps[:2].copy(), Pg[:2].copy())
                except Exception:
                    continue

                if not (min_route_m <= dist_m <= max_route_m):
                    continue

                # 通过冲突检查后，登记占位
                taken_starts_xy.append(Ps.copy())
                taken_s_per_edge.setdefault(s_eid, []).append(float(s_on_s))

                # 如需 lane 级别
                lane_ids = None
                if lift_to_lane_ids:
                    # 轻量抬升：优先找可达 successor 的下一条 lane
                    edge_lanes = {ed["id"]: list(ed.get("lanes", [])) for _,_,ed in self.EG.edges(data=True)}
                    lanes_out = []
                    if edge_path:
                        first = edge_lanes.get(edge_path[0], [])
                        lanes_out.append((rng.choice(first) if first else None))
                        for e1,e2 in zip(edge_path[:-1], edge_path[1:]):
                            prev = edge_lanes.get(e1, []); nxt = edge_lanes.get(e2, [])
                            chosen = None
                            if lanes_out[-1] is not None:
                                cur = lanes_out[-1]
                                for b in nxt:
                                    ed = self.G_lane.get_edge_data(cur, b)
                                    if ed and ed.get("type")=="successor":
                                        chosen=b; break
                            if chosen is None:
                                chosen = (rng.choice(nxt) if nxt else None)
                            lanes_out.append(chosen)
                    lane_ids = lanes_out

                agents.append(dict(
                    agent_id=f"A{id_counter}",
                    cls=cls,
                    size_lwh_m=size_lwh,
                    avg_speed_mps=avg_speed,
                    start_xyz=Ps,
                    goal_xyz=Pg,
                    edge_ids=edge_path,
                    route_xyz=route_xyz,
                    lane_ids=lane_ids
                ))
                id_counter += 1
                made += 1

        return agents
