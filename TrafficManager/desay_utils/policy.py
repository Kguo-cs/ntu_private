# torch_idm_mobil_dualrate_smoothlc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import torch
import networkx as nx

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
    lat_speed_mps: float = 1.2   # lateral traverse speed (m/s) for duration estimate
    lc_min_s: float = 0.9        # minimum lane change duration
    lc_max_s: float = 3.0        # maximum lane change duration

# -------------------- simulator --------------------

class TorchIDMSimulator:
    """
    Dual-rate simulator (CUDA-ready) with smooth lane changing and route following:
      - Physics tick (IDM + MOBIL) every dt_phys seconds -> refresh accelerations & lane-change decisions.
      - Output/integration tick every dt_out seconds -> integrates motion with held accelerations.
      - Lane change is a continuous lateral blend between from-lane and to-lane paths over a duration.
      - Agents are removed when within goal_tol meters from their goal_xy.
      - If an agent dict has lane_ids=[...], it will follow that planned lane sequence:
          * prefer successors matching the next planned lane,
          * restrict/bias lane-change candidates to current/next planned lanes when possible.
    G_lane edges must have: kind='lane', id (or edge_id), geom [N,3].
    """

    def __init__(
        self,
        G_lane: nx.DiGraph,
        *,
        dt_out: float = 0.1,               # output / integration step (e.g., 10 Hz)
        dt_phys: float = 0.5,              # physics recompute step (IDM+MOBIL)
        ds_grid: float = 0.5,              # lane resample resolution (m)
        idm: IDMParams = IDMParams(),
        mobil: MOBILParams = MOBILParams(),
        lane_adjacency: Optional[Dict[Any, List[Any]]] = None,  # {lane_id: [adjacent lane_ids]}
        speed_limits: Optional[Dict[Any, float]] = None,        # {lane_id: vmax m/s}
        goal_tol: float = 10.0,
        device: Optional[torch.device] = None
    ):
        self.G = G_lane
        self.dt_out = float(dt_out)
        self.dt_phys = float(dt_phys)
        self.idm = idm
        self.mobil = mobil
        self.goal_tol = float(goal_tol)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.speed_limits = speed_limits or {}

        # ---- collect & resample lanes
        lane_ids: List[Any] = []
        lane_xyz: List[np.ndarray] = []
        for u, v, ed in G_lane.edges(data=True):
            if ed.get("kind") == "lane":
                lid = ed.get("id", ed.get("edge_id", f"{u}->{v}"))
                P = _to_xyz(ed["geom"])
                lane_ids.append(lid)
                lane_xyz.append(P)
        if not lane_ids:
            raise ValueError("No lane edges (kind='lane') found in G_lane.")

        self.lid2idx = {lid: i for i, lid in enumerate(lane_ids)}
        self.idx2lid = lane_ids

        L_list: List[float] = []
        lanes_X: List[np.ndarray] = []
        lanes_Y: List[np.ndarray] = []
        lanes_N: List[int] = []
        for P in lane_xyz:
            s = _arclen2d(P[:, :2]); L = float(s[-1]); L_list.append(L)
            if L <= 1e-6:
                xs = np.array([P[0,0]]); ys = np.array([P[0,1]])
            else:
                n = max(2, int(math.ceil(L / ds_grid)) + 1)
                s_u = np.linspace(0.0, L, n)
                xs = np.interp(s_u, s, P[:,0]); ys = np.interp(s_u, s, P[:,1])
            lanes_X.append(xs); lanes_Y.append(ys); lanes_N.append(len(xs))

        maxN = int(max(lanes_N))
        pad = lambda arr: np.pad(arr, (0, maxN - len(arr)), mode='edge')
        X = np.vstack([pad(a) for a in lanes_X])
        Y = np.vstack([pad(a) for a in lanes_Y])
        N = np.array(lanes_N, dtype=np.int32)
        Larr = np.array(L_list, dtype=np.float32)

        self.lane_X = torch.from_numpy(X).to(self.device).float()       # [nL, maxN]
        self.lane_Y = torch.from_numpy(Y).to(self.device).float()
        self.lane_N = torch.from_numpy(N).to(self.device).long()        # [nL]
        self.lane_L = torch.from_numpy(Larr).to(self.device).float()    # [nL]
        self.maxN = int(maxN)

        # adjacency
        lane_adjacency = lane_adjacency or {}
        self.adj = {self.lid2idx.get(k, -1): [self.lid2idx.get(t, -1) for t in v] for k, v in lane_adjacency.items()}
        for i in range(len(lane_ids)):
            self.adj.setdefault(i, [])

        # speed limits
        self.vmax_of = torch.full((len(lane_ids),), float("inf"), device=self.device)
        for lid, vmax in self.speed_limits.items():
            if lid in self.lid2idx:
                self.vmax_of[self.lid2idx[lid]] = float(vmax)

        # successors (by topology)
        self.successors: Dict[int, List[int]] = {}
        for u, v, ed in G_lane.edges(data=True):
            if ed.get("kind") != "lane": continue
            li = self.lid2idx[ed.get("id", ed.get("edge_id", f"{u}->{v}"))]
            self.successors.setdefault(li, [])
            for _, v2, ed2 in G_lane.out_edges(v, data=True):
                if ed2.get("kind") == "lane":
                    lid2 = ed2.get("id", ed2.get("edge_id", f"{v}->{v2}"))
                    if lid2 in self.lid2idx:
                        self.successors[li].append(self.lid2idx[lid2])

        # dynamic state
        self.N: int = 0
        self.agent_ids: List[str] = []
        self.lane_idx = torch.empty(0, dtype=torch.long, device=self.device)
        self.s = torch.empty(0, dtype=torch.float32, device=self.device)
        self.v = torch.empty(0, dtype=torch.float32, device=self.device)
        self.v0 = torch.empty(0, dtype=torch.float32, device=self.device)
        self.lenL = torch.empty(0, dtype=torch.float32, device=self.device)
        self.timer = torch.empty(0, dtype=torch.float32, device=self.device)
        self.goal_xy = torch.empty((0,2), dtype=torch.float32, device=self.device)

        self.a_hold = torch.empty(0, dtype=torch.float32, device=self.device)
        self.time_since_phys = 0.0

        # smooth lane-change state (per agent)
        self.lc_active = torch.empty(0, dtype=torch.bool, device=self.device)
        self.lc_from = torch.empty(0, dtype=torch.long, device=self.device)
        self.lc_to = torch.empty(0, dtype=torch.long, device=self.device)
        self.lc_alpha = torch.empty(0, dtype=torch.float32, device=self.device)   # 0..1
        self.lc_T = torch.empty(0, dtype=torch.float32, device=self.device)       # duration (s)
        self.lc_s_from0 = torch.empty(0, dtype=torch.float32, device=self.device) # s at start on from-lane
        self.lc_s_to0 = torch.empty(0, dtype=torch.float32, device=self.device)   # projected s at start on to-lane

        # --- route-following state (per-agent, variable-length kept as Python lists) ---
        # route_seq[i] is a list of lane indices (ints) for agent i; can be empty if no route.
        self.route_seq: List[List[int]] = []
        # pointer to current position inside the route (index into route_seq[i])
        self.route_ptr = torch.empty(0, dtype=torch.long, device=self.device)
        # whether this agent uses a route (len(route_seq[i]) >= 1)
        self.route_active = torch.empty(0, dtype=torch.bool, device=self.device)

    # ------------------ public API ------------------

    def init_agents_from_batch(self, agents: List[Dict[str, Any]], v0_fraction: float = 0.9):
        lane_idx = []; s_list = []; v_list = []; v0_list = []; Lveh_list = []; timers = []
        goals: List[np.ndarray] = []
        self.agent_ids = []
        self.route_seq = []  # reset routes

        for a in agents:
            # --- choose start lane ---
            lid = a.get("start_lane_id", None)
            if lid is None or lid not in self.lid2idx:
                start_xyz = np.asarray(a.get("start_xyz", [0,0,0]), float)
                if start_xyz.shape[0] >= 2:
                    lid = self._nearest_lane_id_cpu(start_xyz[:2])
                if lid is None or lid not in self.lid2idx:
                    continue

            li = self.lid2idx[lid]
            s0 = self._project_s_cpu(li, np.asarray(a.get("start_xyz", [0,0,0]), float)[:2])
            v0 = float(a.get("avg_speed_mps", 15.0) * v0_fraction)
            v_init = min(v0, 0.5*v0 + 1.0)
            Lveh = float(a.get("size_lwh_m", (4.5,1.8,1.6))[0])
            gxy = np.asarray(a.get("goal_xyz", [0.0, 0.0]), float)[:2]

            # --- map optional lane_ids route to lane indices ---
            route_lane_ids = a.get("lane_ids", None)
            route_seq_i: List[int] = []
            if route_lane_ids:
                for rid in route_lane_ids:
                    if rid in self.lid2idx:
                        route_seq_i.append(self.lid2idx[rid])

            lane_idx.append(li); s_list.append(s0); v_list.append(v_init); v0_list.append(v0)
            Lveh_list.append(Lveh); timers.append(0.0); goals.append(gxy)
            self.agent_ids.append(a.get("agent_id", f"A{len(self.agent_ids)}"))
            self.route_seq.append(route_seq_i)

        self.N = len(lane_idx)
        if self.N == 0:
            raise ValueError("No agents could be initialized on lanes.")

        dev = self.device
        self.lane_idx = torch.tensor(lane_idx, device=dev, dtype=torch.long)
        self.s = torch.tensor(s_list, device=dev, dtype=torch.float32)
        self.v = torch.tensor(v_list, device=dev, dtype=torch.float32)
        self.v0 = torch.tensor(v0_list, device=dev, dtype=torch.float32)
        self.lenL = torch.tensor(Lveh_list, device=dev, dtype=torch.float32)
        self.timer = torch.tensor(timers, device=dev, dtype=torch.float32)
        self.goal_xy = torch.tensor(np.asarray(goals, float), device=dev, dtype=torch.float32)

        # init lane-change state
        self.lc_active = torch.zeros((self.N,), dtype=torch.bool, device=dev)
        self.lc_from = self.lane_idx.clone()
        self.lc_to = self.lane_idx.clone()
        self.lc_alpha = torch.zeros((self.N,), dtype=torch.float32, device=dev)
        self.lc_T = torch.ones((self.N,), dtype=torch.float32, device=dev)
        self.lc_s_from0 = self.s.clone()
        self.lc_s_to0 = self.s.clone()

        # route flags and pointers
        self.route_active = torch.tensor([len(seq) > 0 for seq in self.route_seq],
                                         dtype=torch.bool, device=dev)
        # pointer = nearest route lane to current lane (robust default)
        ptr_list = []
        for i, seq in enumerate(self.route_seq):
            if not seq:
                ptr_list.append(0)
            else:
                li = int(self.lane_idx[i].item())
                # nearest in index space as a simple heuristic
                k = int(np.argmin([abs(li - q) for q in seq])) if seq else 0
                ptr_list.append(k)
        self.route_ptr = torch.tensor(ptr_list, dtype=torch.long, device=dev)

        # initial physics tick
        self._physics_tick()
        self.time_since_phys = 0.0

    def step(self):
        """Advance by dt_out; physics updates each dt_phys; remove agents at goal."""
        dt = self.dt_out
        if self.time_since_phys <= 1e-9:
            self._physics_tick()

        # integrate with held acceleration (exact kinematics)
        vmax = torch.minimum(self.v0, self.vmax_of[self.lane_idx])
        a = self.a_hold
        v_prev = self.v
        self.v = torch.clamp(self.v + a * dt, min=torch.zeros_like(vmax), max=vmax)
        self.s = self.s + v_prev * dt + 0.5 * a * dt * dt
        self.timer = self.timer + dt

        # lane-end transitions for "host" lane (current ownership)
        self._lane_end_transitions()

        # update lane-change progress (smooth blending)
        self._update_lane_change_progress(dt)

        # goal removal
        pos_xy = self._current_xy_all()       # [N,2] blended if changing
        dist = torch.norm(pos_xy - self.goal_xy, dim=1)
        alive_mask = dist > self.goal_tol
        if not alive_mask.all():
            self._remove_agents(~alive_mask)

        # physics tick accumulator
        self.time_since_phys += dt
        if self.time_since_phys + 1e-9 >= self.dt_phys:
            self.time_since_phys = 0.0

    def get_positions(self) -> Dict[str, Tuple[float,float,float]]:
        out: Dict[str, Tuple[float,float,float]] = {}
        xy = self._current_xy_all()
        h = self._current_heading_all()
        for i, aid in enumerate(self.agent_ids):
            out[aid] = (float(xy[i,0].item()), float(xy[i,1].item()), float(h[i].item()))
        return out

    def get_state(self):
        xy = self._current_xy_all(); h = self._current_heading_all()
        return {
            "ids": list(self.agent_ids),
            "lane_idx": self.lane_idx.clone(),
            "s": self.s.clone(),
            "v": self.v.clone(),
            "a": self.a_hold.clone(),
            "xyh": {aid: (float(xy[i,0]), float(xy[i,1]), float(h[i])) for i, aid in enumerate(self.agent_ids)},
            "lc_active": self.lc_active.clone(),
            "lc_alpha": self.lc_alpha.clone(),
            "lc_from": self.lc_from.clone(),
            "lc_to": self.lc_to.clone(),
            "route_ptr": self.route_ptr.clone(),
            "route_active": self.route_active.clone(),
        }

    # ------------------ physics tick ------------------

    def _physics_tick(self):
        # build ordering on host lane (ownership during change: from-lane if alpha<0.5 else to-lane)
        owners = torch.where(self.lc_active & (self.lc_alpha >= 0.5), self.lc_to, self.lc_from)
        owners = torch.where(self.lc_active, owners, self.lane_idx)

        leaders, followers, lane_sorted = self._build_lane_orderings(owners)

        # IDM accel
        a_now = self._idm_accel_vector(leaders)

        # MOBIL decisions (start new lane changes only if not already changing)
        target_lane = self._mobil_decisions(owners, leaders, followers, lane_sorted, a_now)

        # start lane change where requested
        to_start = (target_lane >= 0) & (~self.lc_active)
        if to_start.any():
            idxs = torch.nonzero(to_start, as_tuple=False).squeeze(1).tolist()
            for i in idxs:
                li_from = int(self.lane_idx[i].item())
                li_to = int(target_lane[i].item())
                # determine durations from lateral gap
                xy_from = self._current_xy(i)  # blended position is smoother
                s_to0 = self._project_s_torch(li_to, xy_from)  # align along-arc
                xy_to = self._xy_of(li_to, s_to0)
                lat_dist = float(torch.norm(xy_to - xy_from).item())
                v_lat = max(0.1, float(self.mobil.lat_speed_mps))
                T = float(np.clip(lat_dist / v_lat, self.mobil.lc_min_s, self.mobil.lc_max_s))

                self.lc_active[i] = True
                self.lc_from[i] = li_from
                self.lc_to[i] = li_to
                self.lc_alpha[i] = 0.0
                self.lc_T[i] = T
                self.lc_s_from0[i] = self.s[i]
                self.lc_s_to0[i] = s_to0
                # reset timer to avoid immediate flip-flop
                self.timer[i] = 0.0

        # hold acceleration until next physics tick
        self.a_hold = a_now

    # ------------------ update smooth LC progress ------------------

    def _update_lane_change_progress(self, dt: float):
        if not self.lc_active.any(): return
        # advance alpha
        active_idx = torch.nonzero(self.lc_active, as_tuple=False).squeeze(1).tolist()
        for i in active_idx:
            T = float(self.lc_T[i].item())
            if T <= 1e-6:
                self.lc_alpha[i] = 1.0
            else:
                self.lc_alpha[i] = torch.clamp(self.lc_alpha[i] + dt / T, 0.0, 1.0)

            # update along-arc mapping: keep delta_s identical on from/to
            s_from = self.lc_s_from0[i] + (self.s[i] - self.lc_s_from0[i])
            s_to   = self.lc_s_to0[i]   + (self.s[i] - self.lc_s_from0[i])
            # clamp to lengths
            li_from = int(self.lc_from[i].item()); li_to = int(self.lc_to[i].item())
            s_from = torch.clamp(s_from, 0.0, self.lane_L[li_from])
            s_to   = torch.clamp(s_to,   0.0, self.lane_L[li_to])
            # if more than half complete, switch logical lane_idx to to-lane for ownership
            if self.lc_alpha[i] >= 0.5 and self.lane_idx[i] != self.lc_to[i]:
                # rebase s to the to-lane value to avoid jump when finishing
                self.lane_idx[i] = self.lc_to[i]
                self.s[i] = s_to

            # finish
            if self.lc_alpha[i] >= 1.0 - 1e-6:
                # finalize on to-lane at mapped s
                self.lane_idx[i] = self.lc_to[i]
                self.s[i] = s_to
                self.lc_active[i] = False
                self.lc_alpha[i] = 0.0
                self.lc_T[i] = 1.0
                self.lc_from[i] = self.lane_idx[i]
                self.lc_to[i] = self.lane_idx[i]
                self.lc_s_from0[i] = self.s[i]
                self.lc_s_to0[i] = self.s[i]
                # route pointer may advance if this lane is on route
                self._advance_route_if_matched(i, int(self.lane_idx[i].item()))

    def _project_s_torch(self, li: int, xy: torch.Tensor) -> torch.Tensor:
        """Nearest-sample projection: argmin ||(X(li)-x, Y(li)-y)|| -> s."""
        n = self.lane_N[li]
        if n <= 1:
            return torch.tensor(0.0, device=self.device)
        d2 = (self.lane_X[li, :n] - xy[0])**2 + (self.lane_Y[li, :n] - xy[1])**2
        k = int(torch.argmin(d2).item())
        if self.lane_L[li] <= 1e-6:
            return torch.tensor(0.0, device=self.device)
        return (k / (n - 1)) * self.lane_L[li]

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

        acc = amax * (1.0 - (v/torch.clamp(self.v0, min=0.1))**delta - (s_star/s_rel)**2)
        return torch.clamp(acc, min=-b*2.5, max=amax)

    def _mobil_decisions(self, owners: torch.Tensor, leaders, followers, lane_sorted, a_now_vec) -> torch.Tensor:
        """
        Choose target lane for new lane changes. Agents already changing are skipped.
        'owners' gives the lane for ordering this tick (from-lane until half done).
        Route-aware: restrict/bias candidates to current/next planned lanes when possible.
        """
        N = self.N
        target_lane = torch.full((N,), -1, dtype=torch.long, device=self.device)
        can_change = (self.timer >= self.mobil.min_time_in_lane) & (~self.lc_active)
        if not can_change.any(): return target_lane

        sorted_s_of = {li: lane_sorted[li]["s"] for li in lane_sorted}
        sorted_idx_of = {li: lane_sorted[li]["idx"] for li in lane_sorted}

        for i in torch.nonzero(can_change, as_tuple=False).squeeze(1).tolist():
            li = int(owners[i].item())
            cand_list = self.adj.get(li, [])
            if not cand_list: continue

            # --- route restriction / bias ---
            desired_cur = self._route_current_lane(i)
            desired_next = self._route_next_lane(i)

            route_cands = []
            if desired_cur is not None:
                route_cands.append(desired_cur)
            if desired_next is not None:
                route_cands.append(desired_next)
            route_cands = [c for c in cand_list if c in route_cands]
            cand_use = route_cands if route_cands else cand_list

            a_now = float(a_now_vec[i].item())
            xy = self._current_xy(i)

            best_gain = self.mobil.a_thr
            best_lane = -1

            for lj in cand_use:
                if lj < 0: continue
                s_cand = self._project_s_torch(lj, xy)

                s_sorted = sorted_s_of.get(lj, torch.empty(0, device=self.device))
                idx_sorted = sorted_idx_of.get(lj, torch.empty(0, dtype=torch.long, device=self.device))
                pos = torch.searchsorted(s_sorted, s_cand)
                leader_idx = int(idx_sorted[pos].item()) if pos < s_sorted.numel() else -1
                follower_idx = int(idx_sorted[pos-1].item()) if pos > 0 else -1

                a_self_new = self._idm_accel_single(i, leader_idx, lj, s_self=s_cand)

                a_ft_now, a_ft_new = 0.0, 0.0
                if follower_idx >= 0:
                    lead_ft_now = self._leader_of_idx(idx_sorted, follower_idx)
                    a_ft_now = self._idm_accel_single(follower_idx, lead_ft_now, lj)
                    a_ft_new = self._idm_accel_single(follower_idx, i, lj)
                a_fc_now, a_fc_new = 0.0, 0.0
                foll_cur = int(followers[i].item())
                lead_cur = int(leaders[i].item())
                if foll_cur >= 0:
                    a_fc_now = self._idm_accel_single(foll_cur, i, li)
                    a_fc_new = self._idm_accel_single(foll_cur, lead_cur, li)

                # safety constraints
                if (follower_idx >= 0) and (a_ft_new < -self.mobil.b_safe):
                    continue
                if (foll_cur >= 0) and (a_fc_new < -self.mobil.b_safe):
                    continue

                incentive = (a_self_new - a_now) + self.mobil.politeness * ((a_ft_new - a_ft_now) + (a_fc_new - a_fc_now))

                # soft bias toward the planned next lane
                if desired_next is not None and lj == desired_next:
                    incentive += 0.25

                if incentive > best_gain:
                    if self._has_min_gap_for_change(i, lj, s_cand, s_sorted, idx_sorted):
                        best_gain = incentive
                        best_lane = lj
