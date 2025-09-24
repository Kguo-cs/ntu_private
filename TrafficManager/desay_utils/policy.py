# torch_idm_mobil_dualrate_smoothlc_connectors_smoothheading.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import torch
import networkx as nx
# plot_agents_on_map(agents, map_infos, show_headings=True)
# 3) Build lane adjacency from your lane_graph (neighbors = same group, lane_index +/- 1)
import networkx as nx


def build_lane_adjacency_from_groups(G_lane: nx.DiGraph) -> dict:
    """
    Returns {lane_id: [adjacent_lane_ids]} using:
      - same 'group_key' (or same boundary_a_id/boundary_b_id/side if group_key missing)
      - |lane_index_i - lane_index_j| == 1
    """
    # collect lanes with metadata
    lanes = []
    for u, v, ed in G_lane.edges(data=True):
        if ed.get("kind") != "lane":
            continue
        lid = ed.get("id", ed.get("edge_id", f"{u}->{v}"))
        gi = ed.get("group_key", None)
        if gi is None:
            gi = (ed.get("boundary_a_id"), ed.get("boundary_b_id"), ed.get("side"))
        lanes.append((
            lid,
            gi,
            int(ed.get("lane_index", 0)),
            int(ed.get("lane_count", 1))
        ))

    # group by corridor group_key AND lane_count (so only same lane-count corridors connect)
    from collections import defaultdict
    groups = defaultdict(list)
    for lid, gk, idx, lcnt in lanes:
        groups[(gk, lcnt)].append((lid, idx))

    # build adjacency
    adj = {lid: [] for lid, *_ in lanes}
    for (gk, lcnt), items in groups.items():
        # sort by index
        items.sort(key=lambda t: t[1])
        for (lid_i, idx_i), (lid_j, idx_j) in zip(items, items[1:]):
            if abs(idx_i - idx_j) == 1:
                adj[lid_i].append(lid_j)
                adj[lid_j].append(lid_i)
    return adj


# -------------------- helpers --------------------

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("need [N,3]/[N,2]/[3,N]")
    return a

def _arclen2d(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2: return np.array([0.0])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2*math.pi) - math.pi

def _angle_diff_signed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # returns smallest signed angle a-b in (-pi,pi]
    return _angle_wrap(a - b)

def _smootherstep_quintic(a: torch.Tensor) -> torch.Tensor:
    # a in [0,1] -> 10a^3 - 15a^4 + 6a^5  (C2-continuous)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    return 10*a3 - 15*a4 + 6*a5

# -------------------- parameters --------------------

@dataclass
class IDMParams:
    T: float = 1.2
    a_max: float = 1.4
    b_comf: float = 2.0
    s0: float = 2.0
    delta: float = 4.0

@dataclass
class MOBILParams:
    politeness: float = 0.3
    a_thr: float = 0.1
    b_safe: float = 3.5
    min_gap_lane_change: float = 1.0
    min_time_in_lane: float = 1.2
    # Smooth lane-change kinematics
    lat_speed_mps: float = 1.2
    lc_min_s: float = 0.9
    lc_max_s: float = 3.0

# -------------------- simulator --------------------

class TorchIDMSimulator:
    """
    CUDA-ready IDM/MOBIL with:
      - Lanes + connectors (both traversable)
      - Smooth LC using quintic blend of position *and* tangents
      - EMA-smoothed heading (unwrap-aware)
      - Route following (lane/connector IDs)
    """

    def __init__(
        self,
        G_lane: nx.DiGraph,
        *,
        dt_out: float = 0.1,
        dt_phys: float = 0.5,
        ds_grid: float = 0.5,
        idm: IDMParams = IDMParams(),
        mobil: MOBILParams = MOBILParams(),
        lane_adjacency: Optional[Dict[Any, List[Any]]] = None,
        speed_limits: Optional[Dict[Any, float]] = None,
        goal_tol: float = 10.0,
        # heading smoothing (EMA): tau in seconds; set 0 to disable
        heading_smooth_tau: float = 3,
        device: Optional[torch.device] = None
    ):
        self.G = G_lane
        self.dt_out = float(dt_out)
        self.dt_phys = float(dt_phys)
        self.idm = idm
        self.mobil = mobil
        self.goal_tol = float(goal_tol)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.speed_limits_override = speed_limits or {}
        self.heading_tau = float(max(0.0, heading_smooth_tau))

        # ------------- collect tracks -------------
        track_ids: List[Any] = []
        track_xyz: List[np.ndarray] = []
        track_kind: List[str] = []
        track_speed: List[float] = []

        for u, v, ed in G_lane.edges(data=True):
            k = ed.get("kind", None)
            if k not in ("lane", "connector"):
                continue
            tid = ed.get("id", ed.get("edge_id", f"{u}->{v}"))
            P = _to_xyz(ed["geom"])
            track_ids.append(tid)
            track_xyz.append(P)
            track_kind.append(k)
            vmax = float(ed.get("speed_mps", float("inf")))
            if tid in self.speed_limits_override:
                vmax = float(self.speed_limits_override[tid])
            track_speed.append(vmax)

        if not track_ids:
            raise ValueError("No 'lane' or 'connector' edges found in G_lane.")

        self.tid2idx = {tid: i for i, tid in enumerate(track_ids)}
        self.idx2tid = track_ids
        self.is_lane = np.array([k == "lane" for k in track_kind], dtype=bool)
        self.is_connector = ~self.is_lane

        # ------------- resample tracks -------------
        L_list: List[float] = []
        Xs: List[np.ndarray] = []
        Ys: List[np.ndarray] = []
        Ns: List[int] = []
        for P in track_xyz:
            s = _arclen2d(P[:, :2]); L = float(s[-1]); L_list.append(L)
            if L <= 1e-6:
                xs = np.array([P[0,0]]); ys = np.array([P[0,1]])
            else:
                n = max(2, int(math.ceil(L / ds_grid)) + 1)
                su = np.linspace(0.0, L, n)
                xs = np.interp(su, s, P[:,0]); ys = np.interp(su, s, P[:,1])
            Xs.append(xs); Ys.append(ys); Ns.append(len(xs))

        maxN = int(max(Ns))
        pad = lambda arr: np.pad(arr, (0, maxN - len(arr)), mode='edge')
        X = np.vstack([pad(a) for a in Xs])
        Y = np.vstack([pad(a) for a in Ys])
        N = np.array(Ns, dtype=np.int32)
        Larr = np.array(L_list, dtype=np.float32)
        vmax_arr = np.array(track_speed, dtype=np.float32)

        dev = self.device
        self.track_X = torch.from_numpy(X).to(dev).float()     # [nT, maxN]
        self.track_Y = torch.from_numpy(Y).to(dev).float()
        self.track_N = torch.from_numpy(N).to(dev).long()      # [nT]
        self.track_L = torch.from_numpy(Larr).to(dev).float()  # [nT]
        self.track_vmax = torch.from_numpy(vmax_arr).to(dev).float()
        self.maxN = int(maxN)
        self.nT = int(len(track_ids))

        # ------------- successors -------------
        self.successors: Dict[int, List[int]] = {}
        for u, v, ed in G_lane.edges(data=True):
            if ed.get("kind") not in ("lane", "connector"): continue
            i = self.tid2idx[ed.get("id", ed.get("edge_id", f"{u}->{v}"))]
            self.successors.setdefault(i, [])
            for _, v2, ed2 in G_lane.out_edges(v, data=True):
                if ed2.get("kind") in ("lane", "connector"):
                    t2 = ed2.get("id", ed2.get("edge_id", f"{v}->{v2}"))
                    if t2 in self.tid2idx:
                        self.successors[i].append(self.tid2idx[t2])

        # ------------- lateral adjacency (lanes only) -------------
        lane_adjacency = lane_adjacency or {}
        self.adj: Dict[int, List[int]] = {}
        for k, neigh in lane_adjacency.items():
            if k not in self.tid2idx: continue
            i = self.tid2idx[k]
            if not self.is_lane[i]: continue
            self.adj[i] = [self.tid2idx[t] for t in neigh if t in self.tid2idx and self.is_lane[self.tid2idx[t]]]
        for i in range(self.nT):
            if self.is_lane[i]:
                self.adj.setdefault(i, [])

        # ------------- dynamic state -------------
        self.N: int = 0
        self.agent_ids: List[str] = []
        self.tr_idx = torch.empty(0, dtype=torch.long, device=dev)
        self.s = torch.empty(0, dtype=torch.float32, device=dev)
        self.v = torch.empty(0, dtype=torch.float32, device=dev)
        self.v0 = torch.empty(0, dtype=torch.float32, device=dev)
        self.lenL = torch.empty(0, dtype=torch.float32, device=dev)
        self.timer = torch.empty(0, dtype=torch.float32, device=dev)
        self.goal_xy = torch.empty((0,2), dtype=torch.float32, device=dev)

        self.a_hold = torch.empty(0, dtype=torch.float32, device=dev)
        self.time_since_phys = 0.0

        # LC state
        self.lc_active = torch.empty(0, dtype=torch.bool, device=dev)
        self.lc_from = torch.empty(0, dtype=torch.long, device=dev)
        self.lc_to = torch.empty(0, dtype=torch.long, device=dev)
        self.lc_alpha = torch.empty(0, dtype=torch.float32, device=dev)
        self.lc_T = torch.empty(0, dtype=torch.float32, device=dev)
        self.lc_s_from0 = torch.empty(0, dtype=torch.float32, device=dev)
        self.lc_s_to0 = torch.empty(0, dtype=torch.float32, device=dev)

        # route state
        self.route_seq: List[List[int]] = []
        self.route_ptr = torch.empty(0, dtype=torch.long, device=dev)
        self.route_active = torch.empty(0, dtype=torch.bool, device=dev)

        # heading visual (EMA)
        self.h_vis = torch.empty(0, dtype=torch.float32, device=dev)
        self.h_init = torch.empty(0, dtype=torch.bool, device=dev)

    # ------------------ public API ------------------

    def init_agents_from_batch(self, agents: List[Dict[str, Any]], v0_fraction: float = 0.9):
        lane_idx = []; s_list = []; v_list = []; v0_list = []; Lveh_list = []; timers = []
        goals: List[np.ndarray] = []
        self.agent_ids = []
        self.route_seq = []

        for a in agents:
            start_tid = a.get("start_lane_id", None) or a.get("start_track_id", None)
            if start_tid is not None and start_tid not in self.tid2idx:
                start_tid = None
            if start_tid is None:
                start_xyz = np.asarray(a.get("start_xyz", [0,0,0]), float)
                if start_xyz.shape[0] >= 2:
                    start_tid = self._nearest_lane_id_cpu(start_xyz[:2])
            if start_tid is None:
                continue

            ti = self.tid2idx[start_tid]
            s0 = self._project_s_cpu(ti, np.asarray(a.get("start_xyz", [0,0,0]), float)[:2])

            v0 = float(a.get("avg_speed_mps", 15.0) * v0_fraction)
            v_init = min(v0, 0.5*v0 + 1.0)
            Lveh = float(a.get("size_lwh_m", (4.5,1.8,1.6))[0])
            gxy = np.asarray(a.get("goal_xyz", [0.0, 0.0]), float)[:2]

            seq_ids = a.get("lane_ids", None) or a.get("track_ids", None) or []
            route_seq_i = [self.tid2idx[r] for r in seq_ids if r in self.tid2idx]

            lane_idx.append(ti); s_list.append(s0); v_list.append(v_init); v0_list.append(v0)
            Lveh_list.append(Lveh); timers.append(0.0); goals.append(gxy)
            self.agent_ids.append(a.get("agent_id", f"A{len(self.agent_ids)}"))
            self.route_seq.append(route_seq_i)

        self.N = len(lane_idx)
        if self.N == 0:
            raise ValueError("No agents could be initialized on tracks.")

        dev = self.device
        self.tr_idx = torch.tensor(lane_idx, device=dev, dtype=torch.long)
        self.s = torch.tensor(s_list, device=dev, dtype=torch.float32)
        self.v = torch.tensor(v_list, device=dev, dtype=torch.float32)
        self.v0 = torch.tensor(v0_list, device=dev, dtype=torch.float32)
        self.lenL = torch.tensor(Lveh_list, device=dev, dtype=torch.float32)
        self.timer = torch.tensor(timers, device=dev, dtype=torch.float32)
        self.goal_xy = torch.tensor(np.asarray(goals, float), device=dev, dtype=torch.float32)

        # LC state
        self.lc_active = torch.zeros((self.N,), dtype=torch.bool, device=dev)
        self.lc_from = self.tr_idx.clone()
        self.lc_to = self.tr_idx.clone()
        self.lc_alpha = torch.zeros((self.N,), dtype=torch.float32, device=dev)
        self.lc_T = torch.ones((self.N,), dtype=torch.float32, device=dev)
        self.lc_s_from0 = self.s.clone()
        self.lc_s_to0 = self.s.clone()

        # route state
        self.route_active = torch.tensor([len(seq) > 0 for seq in self.route_seq],
                                         dtype=torch.bool, device=dev)
        ptr_list = []
        for i, seq in enumerate(self.route_seq):
            if not seq:
                ptr_list.append(0)
            else:
                ti = int(self.tr_idx[i].item())
                ptr_list.append(seq.index(ti) if ti in seq else 0)
        self.route_ptr = torch.tensor(ptr_list, dtype=torch.long, device=dev)

        # heading visual buffers
        self.h_vis = None #torch.zeros((self.N,), dtype=torch.float32, device=dev)
        self.h_init = torch.zeros((self.N,), dtype=torch.bool, device=dev)

        # first physics tick & first heading
        self._physics_tick()
        self.time_since_phys = 0.0
        self._update_heading_visual()  # initialize EMA

    def step(self):
        dt = self.dt_out
        if self.time_since_phys <= 1e-9:
            self._physics_tick()

        vmax_track = self.track_vmax[self.tr_idx]
        vmax = torch.minimum(self.v0, vmax_track)
        a = self.a_hold
        v_prev = self.v
        self.v = torch.clamp(self.v + a * dt, min=torch.zeros_like(vmax), max=vmax)
        self.s = self.s + v_prev * dt + 0.5 * a * dt * dt
        self.timer = self.timer + dt

        self._track_end_transitions()
        self._update_lane_change_progress(dt)

        # goal removal before heading smoothing
        pos_xy = self._current_xy_all()
        dist = torch.norm(pos_xy - self.goal_xy, dim=1)
        alive_mask = dist > self.goal_tol
        if not alive_mask.all():
            self._remove_agents(~alive_mask)

        # update smoothed heading (EMA)
        self._update_heading_visual()

        self.time_since_phys += dt
        if self.time_since_phys + 1e-9 >= self.dt_phys:
            self.time_since_phys = 0.0

    def get_positions(self) -> Dict[str, Tuple[float,float,float]]:
        out: Dict[str, Tuple[float,float,float]] = {}
        xy = self._current_xy_all()
        h = self.h_vis if self.h_vis.numel() == self.N else self._current_heading_all()
        for i, aid in enumerate(self.agent_ids):
            out[aid] = (float(xy[i,0].item()), float(xy[i,1].item()), float(h[i].item()))
        return out

    # ------------------ physics tick ------------------

    def _physics_tick(self):
        owners = torch.where(self.lc_active & (self.lc_alpha >= 0.5), self.lc_to, self.lc_from)
        owners = torch.where(self.lc_active, owners, self.tr_idx)
        leaders, followers, track_sorted = self._build_track_orderings(owners)
        a_now = self._idm_accel_vector(leaders)

        # target_lane = self._mobil_decisions(owners, leaders, followers, track_sorted, a_now)
        #
        # to_start = (target_lane >= 0) & (~self.lc_active)
        # if to_start.any():
        #     idxs = torch.nonzero(to_start, as_tuple=False).squeeze(1).tolist()
        #     for i in idxs:
        #         ti_from = int(self.tr_idx[i].item())
        #         ti_to = int(target_lane[i].item())
        #         if not (self.is_lane[ti_from] and self.is_lane[ti_to]):
        #             continue
        #         xy_from = self._current_xy(i)
        #         s_to0 = self._project_s_torch(ti_to, xy_from)
        #         xy_to = self._xy_of(ti_to, s_to0)
        #         lat_dist = float(torch.norm(xy_to - xy_from).item())
        #         v_lat = max(0.1, float(self.mobil.lat_speed_mps))
        #         T = float(np.clip(lat_dist / v_lat, self.mobil.lc_min_s, self.mobil.lc_max_s))
        #
        #         self.lc_active[i] = True
        #         self.lc_from[i] = ti_from
        #         self.lc_to[i] = ti_to
        #         self.lc_alpha[i] = 0.0
        #         self.lc_T[i] = T
        #         self.lc_s_from0[i] = self.s[i]
        #         self.lc_s_to0[i] = s_to0
        #         self.timer[i] = 0.0

        self.a_hold = a_now

    # ------------------ lane-change progress ------------------

    def _update_lane_change_progress(self, dt: float):
        if not self.lc_active.any(): return
        act = torch.nonzero(self.lc_active, as_tuple=False).squeeze(1).tolist()
        for i in act:
            T = float(self.lc_T[i].item())
            if T <= 1e-6:
                self.lc_alpha[i] = 1.0
            else:
                self.lc_alpha[i] = torch.clamp(self.lc_alpha[i] + dt / T, 0.0, 1.0)

            s_from = self.lc_s_from0[i] + (self.s[i] - self.lc_s_from0[i])
            s_to   = self.lc_s_to0[i]   + (self.s[i] - self.lc_s_from0[i])

            ti_from = int(self.lc_from[i].item()); ti_to = int(self.lc_to[i].item())
            s_from = torch.clamp(s_from, 0.0, self.track_L[ti_from])
            s_to   = torch.clamp(s_to,   0.0, self.track_L[ti_to])

            if self.lc_alpha[i] >= 0.5 and self.tr_idx[i] != self.lc_to[i]:
                self.tr_idx[i] = self.lc_to[i]
                self.s[i] = s_to

            if self.lc_alpha[i] >= 1.0 - 1e-6:
                self.tr_idx[i] = self.lc_to[i]
                self.s[i] = s_to
                self.lc_active[i] = False
                self.lc_alpha[i] = 0.0
                self.lc_T[i] = 1.0
                self.lc_from[i] = self.tr_idx[i]
                self.lc_to[i] = self.tr_idx[i]
                self.lc_s_from0[i] = self.s[i]
                self.lc_s_to0[i] = self.s[i]
                self._advance_route_if_matched(i, int(self.tr_idx[i].item()))

    # ------------------ IDM + MOBIL ------------------

    def _idm_accel_vector(self, leaders: torch.Tensor) -> torch.Tensor:
        p = self.idm
        amax = float(p.a_max); b = float(p.b_comf); T = float(p.T); s0 = float(p.s0); delta = float(p.delta)

        v = torch.clamp(self.v, min=0.0)
        v_lead = torch.zeros_like(v)
        maskL = leaders >= 0
        if maskL.any():
            v_lead[maskL] = self.v[leaders[maskL]]
        dv = v - v_lead

        s_lead = torch.full_like(self.s, 1e9)
        L_lead = torch.zeros_like(self.lenL)
        if maskL.any():
            s_lead[maskL] = self.s[leaders[maskL]]
            L_lead[maskL] = self.lenL[leaders[maskL]]
        s_rel = torch.clamp((s_lead - 0.5*L_lead) - (self.s + 0.5*self.lenL), min=0.1)

        denom = 2.0 * math.sqrt(max(1e-6, amax * b))
        s_star = s0 + torch.clamp(v*T + v*dv/denom, min=0.0)

        vmax_track = self.track_vmax[self.tr_idx]
        v0_eff = torch.minimum(self.v0, vmax_track)
        acc = amax * (1.0 - (v/torch.clamp(v0_eff, min=0.1))**delta - (s_star/s_rel)**2)
        return torch.clamp(acc, min=-b*2.5, max=amax)

    def _mobil_decisions(self, owners: torch.Tensor, leaders, followers, track_sorted, a_now_vec) -> torch.Tensor:
        N = self.N
        target = torch.full((N,), -1, dtype=torch.long, device=self.device)
        can_change = (self.timer >= self.mobil.min_time_in_lane) & (~self.lc_active)
        if not can_change.any(): return target

        sorted_s_of = {ti: track_sorted[ti]["s"] for ti in track_sorted}
        sorted_idx_of = {ti: track_sorted[ti]["idx"] for ti in track_sorted}

        for i in torch.nonzero(can_change, as_tuple=False).squeeze(1).tolist():
            ti = int(owners[i].item())
            if not self.is_lane[ti]:
                continue

            cand_list = self.adj.get(ti, [])
            if not cand_list:
                continue

            desired_cur = self._route_current_track(i)
            desired_next = self._route_next_track(i)
            route_cands = []
            if desired_cur is not None: route_cands.append(desired_cur)
            if desired_next is not None: route_cands.append(desired_next)
            route_cands = [c for c in cand_list if c in route_cands]
            cand_use = route_cands if route_cands else cand_list

            a_now = float(a_now_vec[i].item())
            xy = self._current_xy(i)
            best_gain = self.mobil.a_thr
            best_lane = -1

            for tj in cand_use:
                if tj < 0 or not self.is_lane[tj]: continue
                s_cand = self._project_s_torch(tj, xy)
                s_sorted = sorted_s_of.get(tj, torch.empty(0, device=self.device))
                idx_sorted = sorted_idx_of.get(tj, torch.empty(0, dtype=torch.long, device=self.device))
                pos = torch.searchsorted(s_sorted, s_cand)
                leader_idx = int(idx_sorted[pos].item()) if pos < s_sorted.numel() else -1
                follower_idx = int(idx_sorted[pos-1].item()) if pos > 0 else -1

                a_self_new = self._idm_accel_single(i, leader_idx, tj, s_self=s_cand)

                a_ft_now, a_ft_new = 0.0, 0.0
                if follower_idx >= 0:
                    lead_ft_now = self._leader_of_idx(idx_sorted, follower_idx)
                    a_ft_now = self._idm_accel_single(follower_idx, lead_ft_now, tj)
                    a_ft_new = self._idm_accel_single(follower_idx, i, tj)
                a_fc_now, a_fc_new = 0.0, 0.0
                foll_cur = int(followers[i].item())
                lead_cur = int(leaders[i].item())
                if foll_cur >= 0:
                    a_fc_now = self._idm_accel_single(foll_cur, i, ti)
                    a_fc_new = self._idm_accel_single(foll_cur, lead_cur, ti)

                if (follower_idx >= 0) and (a_ft_new < -self.mobil.b_safe):
                    continue
                if (foll_cur >= 0) and (a_fc_new < -self.mobil.b_safe):
                    continue

                incentive = (a_self_new - a_now) + self.mobil.politeness * ((a_ft_new - a_ft_now) + (a_fc_new - a_fc_now))
                if desired_next is not None and tj == desired_next:
                    incentive += 0.25

                if incentive > best_gain:
                    if self._has_min_gap_for_change(i, tj, s_cand, s_sorted, idx_sorted):
                        best_gain = incentive
                        best_lane = tj

            if best_lane >= 0:
                target[i] = best_lane

        return target

    def _idm_accel_single(self, idx: int, leader_idx: int, tr_idx: int, s_self: Optional[torch.Tensor]=None) -> float:
        if idx < 0: return 0.0
        p = self.idm
        v = float(self.v[idx].item())
        v0 = float(self.v0[idx].item())
        vmax_tr = float(self.track_vmax[tr_idx].item())
        v0_eff = min(v0, vmax_tr)
        amax = float(p.a_max); b = float(p.b_comf); T = float(p.T); s0 = float(p.s0); d = float(p.delta)
        Lself = float(self.lenL[idx].item())
        s_self_val = float((s_self if s_self is not None else self.s[idx]).item())
        if leader_idx >= 0:
            sL = float(self.s[leader_idx].item()); Llead = float(self.lenL[leader_idx].item())
            s_rel = max(0.1, (sL - 0.5*Llead) - (s_self_val + 0.5*Lself))
            dv = v - float(self.v[leader_idx].item())
        else:
            s_rel = 1e6; dv = 0.0
        denom = 2.0 * math.sqrt(max(1e-6, amax * b))
        s_star = s0 + max(0.0, v*T + v*dv/denom)
        acc = amax * (1.0 - (v/max(0.1, v0_eff))**d - (s_star/s_rel)**2)
        return float(np.clip(acc, -b*2.5, amax))

    def _has_min_gap_for_change(self, agent_idx: int, lane_j: int, s_cand: torch.Tensor,
                                s_sorted: torch.Tensor, idx_sorted: torch.Tensor) -> bool:
        p = self.mobil
        pos = torch.searchsorted(s_sorted, s_cand)
        leader_idx = int(idx_sorted[pos].item()) if pos < s_sorted.numel() else -1
        follower_idx = int(idx_sorted[pos-1].item()) if pos > 0 else -1
        L_new = float(self.lenL[agent_idx].item())
        ok_front, ok_back = True, True
        if leader_idx >= 0:
            gap_front = (float(self.s[leader_idx].item()) - 0.5*float(self.lenL[leader_idx].item())) \
                        - (float(s_cand.item()) + 0.5*L_new)
            ok_front = (gap_front >= p.min_gap_lane_change)
        if follower_idx >= 0:
            gap_back = (float(s_cand.item()) - 0.5*L_new) \
                       - (float(self.s[follower_idx].item()) + 0.5*float(self.lenL[follower_idx].item()))
            ok_back = (gap_back >= p.min_gap_lane_change)
        return bool(ok_front and ok_back)

    # ------------------ ordering & transitions ------------------

    def _build_track_orderings(self, owner_tr_idx: torch.Tensor):
        N = self.N
        leaders = torch.full((N,), -1, dtype=torch.long, device=self.device)
        followers = torch.full((N,), -1, dtype=torch.long, device=self.device)
        track_sorted: Dict[int, Dict[str, torch.Tensor]] = {}

        uniq = torch.unique(owner_tr_idx).tolist()
        for ti in uniq:
            mask = (owner_tr_idx == ti)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                track_sorted[ti] = {"s": torch.empty(0, device=self.device),
                                    "idx": torch.empty(0, dtype=torch.long, device=self.device)}
                continue
            s_sorted, order = torch.sort(self.s[idx])
            idx_sorted = idx[order]
            track_sorted[ti] = {"s": s_sorted, "idx": idx_sorted}
            leaders[idx_sorted[:-1]] = idx_sorted[1:]
            followers[idx_sorted[1:]] = idx_sorted[:-1]
        return leaders, followers, track_sorted

    def _track_end_transitions(self):
        over = self.s > (self.track_L[self.tr_idx] - 1e-6)
        if not over.any(): return
        idxs = torch.nonzero(over, as_tuple=False).squeeze(1).tolist()
        for i in idxs:
            ti = int(self.tr_idx[i].item())
            succ = self.successors.get(ti, [])
            if not succ:
                self.s[i] = self.track_L[ti]
                self.v[i] = torch.minimum(self.v[i], torch.tensor(0.0, device=self.device))
                continue

            desired_next = self._route_next_track(i)
            best = desired_next if (desired_next is not None and desired_next in succ) else None
            if best is None:
                h_now = self._heading_of(ti, self.track_L[ti])
                best_head, best_d = succ[0], 1e9
                for cand in succ:
                    h_c = self._heading_of(cand, torch.tensor(0.02, device=self.device))
                    d = float(torch.abs(((h_c - h_now + math.pi) % (2*math.pi)) - math.pi))
                    if d < best_d: best_head, best_d = cand, d
                best = best_head

            extra = self.s[i] - self.track_L[ti]
            self.tr_idx[i] = best
            self.s[i] = torch.clamp(extra, 0.0, self.track_L[best])
            self._advance_route_if_matched(i, best)

            if self.lc_active[i]:
                self.lc_from[i] = self.tr_idx[i]
                self.lc_to[i] = self.tr_idx[i]
                self.lc_active[i] = False
                self.lc_alpha[i] = 0.0
                self.lc_T[i] = 1.0
                self.lc_s_from0[i] = self.s[i]; self.lc_s_to0[i] = self.s[i]

    # ------------------ route helpers ------------------

    def _route_current_track(self, i: int) -> Optional[int]:
        if not bool(self.route_active[i].item()): return None
        seq = self.route_seq[i]
        if not seq: return None
        k = int(self.route_ptr[i].item())
        return seq[max(0, min(k, len(seq)-1))]

    def _route_next_track(self, i: int) -> Optional[int]:
        if not bool(self.route_active[i].item()): return None
        seq = self.route_seq[i]; k = int(self.route_ptr[i].item())
        if k+1 < len(seq): return seq[k+1]
        return None

    def _advance_route_if_matched(self, i: int, new_ti: int):
        if not bool(self.route_active[i].item()): return
        seq = self.route_seq[i]
        if not seq: return
        k = int(self.route_ptr[i].item())
        if new_ti in seq:
            j = seq.index(new_ti)
            if j >= k:
                self.route_ptr[i] = torch.tensor(j, device=self.device, dtype=torch.long)

    # ------------------ geometry & pose ------------------

    def _s_to_idx(self, ti: int, s_val: torch.Tensor) -> torch.Tensor:
        n = self.track_N[ti]; L = self.track_L[ti]
        if L <= 1e-6:
            return torch.zeros_like(s_val, dtype=torch.long, device=self.device)
        t = torch.clamp(s_val / torch.clamp(L, min=1e-6), 0.0, 1.0)
        idx = torch.round(t * (n - 1)).long()
        return torch.clamp(idx, 0, n - 1)

    def _xy_of(self, ti: int, s_val: torch.Tensor) -> torch.Tensor:
        idx = self._s_to_idx(ti, s_val)
        return torch.stack([self.track_X[ti, idx], self.track_Y[ti, idx]], dim=0)

    def _tangent_vec(self, ti: int, s_val: torch.Tensor) -> torch.Tensor:
        idx = self._s_to_idx(ti, s_val)
        i0 = torch.clamp(idx - 2, 0, self.track_N[ti]-1)
        i1 = torch.clamp(idx + 2, 0, self.track_N[ti]-1)
        dx = self.track_X[ti, i1] - self.track_X[ti, i0]
        dy = self.track_Y[ti, i1] - self.track_Y[ti, i0]
        n = torch.sqrt(dx*dx + dy*dy) + 1e-6
        return torch.stack([dx/n, dy/n], dim=0)

    def _heading_of(self, ti: int, s_val: torch.Tensor) -> torch.Tensor:
        t = self._tangent_vec(ti, s_val)
        return torch.atan2(t[1], t[0])

    def _current_xy(self, i: int) -> torch.Tensor:
        """Quintic blend of XY during lane change."""
        if not bool(self.lc_active[i].item()):
            return self._xy_of(int(self.tr_idx[i].item()), self.s[i])
        a = _smootherstep_quintic(self.lc_alpha[i].unsqueeze(0))[0]  # scalar
        ti_from = int(self.lc_from[i].item()); ti_to = int(self.lc_to[i].item())
        s_from = self.lc_s_from0[i] + (self.s[i] - self.lc_s_from0[i])
        s_to   = self.lc_s_to0[i]   + (self.s[i] - self.lc_s_from0[i])
        xy_from = self._xy_of(ti_from, s_from)
        xy_to   = self._xy_of(ti_to,   s_to)
        return (1.0 - a) * xy_from + a * xy_to

    def _current_xy_all(self) -> torch.Tensor:
        N = self.N
        out = torch.empty((N,2), dtype=torch.float32, device=self.device)
        for i in range(N):
            xy = self._current_xy(i)
            out[i,0] = xy[0]; out[i,1] = xy[1]
        return out

    def _current_heading_all(self) -> torch.Tensor:
        """Blend **tangents** with quintic alpha; normalize before atan2 (smooth heading)."""
        N = self.N
        out = torch.empty((N,), dtype=torch.float32, device=self.device)
        for i in range(N):
            if not bool(self.lc_active[i].item()):
                out[i] = self._heading_of(int(self.tr_idx[i].item()), self.s[i])
            else:
                a = _smootherstep_quintic(self.lc_alpha[i].unsqueeze(0))[0]
                ti_from = int(self.lc_from[i].item()); ti_to = int(self.lc_to[i].item())
                s_from = self.lc_s_from0[i] + (self.s[i] - self.lc_s_from0[i])
                s_to   = self.lc_s_to0[i]   + (self.s[i] - self.lc_s_from0[i])
                t_from = self._tangent_vec(ti_from, s_from)
                t_to   = self._tangent_vec(ti_to,   s_to)
                t_blend = (1.0 - a) * t_from + a * t_to
                n = torch.sqrt(t_blend[0]*t_blend[0] + t_blend[1]*t_blend[1]) + 1e-9
                out[i] = torch.atan2(t_blend[1]/n, t_blend[0]/n)
        return out

    # ------------- heading EMA (unwrap-aware) -------------

    def _update_heading_visual(self):
        raw = self._current_heading_all()
        if self.h_vis is None:
            self.h_vis = raw
            return

        if self.h_vis.numel() != self.N:
            self.h_vis = raw.clone()
            self.h_init = torch.ones((self.N,), dtype=torch.bool, device=self.device)
            return
        if self.heading_tau <= 1e-6:
            self.h_vis = raw
            return
        # EMA coefficient from time constant
        alpha = 1.0 - math.exp(-self.dt_out / max(1e-6, self.heading_tau))
        # unwrap-aware update: h_vis += alpha * wrap(raw - h_vis)
        delta = _angle_diff_signed(raw, self.h_vis)
        self.h_vis = _angle_wrap(self.h_vis + alpha * delta)

    # ------------------ removal ------------------

    def _remove_agents(self, mask_remove: torch.Tensor):
        keep = (~mask_remove)
        if keep.sum() == keep.numel(): return
        self.N = int(keep.sum().item())
        self.agent_ids = [aid for aid, k in zip(self.agent_ids, keep.tolist()) if k]
        self.tr_idx = self.tr_idx[keep]
        self.s = self.s[keep]
        self.v = self.v[keep]
        self.v0 = self.v0[keep]
        self.lenL = self.lenL[keep]
        self.timer = self.timer[keep]
        self.a_hold = self.a_hold[keep]
        self.goal_xy = self.goal_xy[keep]
        self.lc_active = self.lc_active[keep]
        self.lc_from = self.lc_from[keep]
        self.lc_to = self.lc_to[keep]
        self.lc_alpha = self.lc_alpha[keep]
        self.lc_T = self.lc_T[keep]
        self.lc_s_from0 = self.lc_s_from0[keep]
        self.lc_s_to0 = self.lc_s_to0[keep]
        self.h_vis = self.h_vis[keep]
        self.h_init = self.h_init[keep]
        # prune routes
        new_seq: List[List[int]] = []
        new_ptr: List[int] = []
        new_act: List[bool] = []
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1).tolist()
        for j in keep_idx:
            new_seq.append(self.route_seq[j])
            new_ptr.append(int(self.route_ptr[j].item()))
            new_act.append(bool(self.route_active[j].item()))
        self.route_seq = new_seq
        dev = self.device
        self.route_ptr = torch.tensor(new_ptr, dtype=torch.long, device=dev)
        self.route_active = torch.tensor(new_act, dtype=torch.bool, device=dev)

    # ------------------ CPU helpers ------------------

    def _nearest_lane_id_cpu(self, xy: np.ndarray) -> Optional[Any]:
        best_i, best_d = None, 1e18
        for ti in range(self.nT):
            if not self.is_lane[ti]:
                continue
            n = int(self.track_N[ti].item())
            X = self.track_X[ti, :n].detach().cpu().numpy()
            Y = self.track_Y[ti, :n].detach().cpu().numpy()
            d2 = (X - xy[0])**2 + (Y - xy[1])**2
            k = int(np.argmin(d2)); d = float(np.sqrt(d2[k]))
            if d < best_d: best_d, best_i = d, ti
        return self.idx2tid[best_i] if best_i is not None else None

    def _project_s_cpu(self, ti: int, xy: np.ndarray) -> float:
        n = int(self.track_N[ti].item())
        if n <= 1: return 0.0
        X = self.track_X[ti, :n].detach().cpu().numpy()
        Y = self.track_Y[ti, :n].detach().cpu().numpy()
        d2 = (X - xy[0])**2 + (Y - xy[1])**2
        k = int(np.argmin(d2))
        L = float(self.track_L[ti].item())
        return (k / max(1, n - 1)) * L

    def _project_s_torch(self, ti: int, xy: torch.Tensor) -> torch.Tensor:
        n = self.track_N[ti]
        if n <= 1:
            return torch.tensor(0.0, device=self.device)
        d2 = (self.track_X[ti, :n] - xy[0])**2 + (self.track_Y[ti, :n] - xy[1])**2
        k = int(torch.argmin(d2).item())
        if self.track_L[ti] <= 1e-6:
            return torch.tensor(0.0, device=self.device)
        return (k / (n - 1)) * self.track_L[ti]

    @staticmethod
    def _leader_of_idx(idx_sorted: torch.Tensor, idx: int) -> int:
        if idx < 0 or idx_sorted.numel() == 0: return -1
        pos = (idx_sorted == idx).nonzero(as_tuple=False)
        if pos.numel() == 0: return -1
        k = int(pos[0,0].item())
        return int(idx_sorted[k+1].item()) if (k+1) < idx_sorted.numel() else -1
