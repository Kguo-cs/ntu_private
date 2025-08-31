# batch_idm_mobil.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx

# ---------- tiny helpers (reuse-friendly) ----------

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
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _interp_xyz_at_s(xyz: np.ndarray, s_tab: np.ndarray, s: float) -> np.ndarray:
    x = np.interp(s, s_tab, xyz[:,0]); y = np.interp(s, s_tab, xyz[:,1]); z = np.interp(s, s_tab, xyz[:,2])
    return np.array([x, y, z], float)

def _heading_at_s(xyz: np.ndarray, s_tab: np.ndarray, s: float, eps: float = 0.6) -> float:
    if s_tab[-1] <= 0: return 0.0
    s0 = max(0.0, s - eps); s1 = min(s_tab[-1], s + eps)
    p0 = _interp_xyz_at_s(xyz, s_tab, s0)
    p1 = _interp_xyz_at_s(xyz, s_tab, s1)
    dx, dy = (p1[0]-p0[0]), (p1[1]-p0[1])
    return float(np.arctan2(dy, dx)) if (dx*dx+dy*dy)>0 else 0.0

# ---------- lane atlas for fast access ----------

@dataclass
class LaneGeom:
    lane_id: Any
    xyz: np.ndarray
    s: np.ndarray
    length: float
    successors: List[Any]       # lane_ids
    laterals: List[Any]         # neighbor lane_ids (mutual recommended)

class LaneAtlas:
    def __init__(self, G_lane: nx.DiGraph):
        self.G = G_lane
        self.lanes: Dict[Any, LaneGeom] = {}
        for lid, nd in G_lane.nodes(data=True):
            k = _key_id(lid)
            xyz = _to_xyz(nd["xyz"])
            s = _arclen2d(xyz[:, :2])
            succ = []
            lat = []
            for _, v, ed in G_lane.out_edges(lid, data=True):
                if ed.get("type") == "successor":
                    succ.append(_key_id(v))
                elif ed.get("type") == "lateral":
                    # only accept if mutual lateral to be safe
                    if G_lane.has_edge(v, lid) and G_lane.get_edge_data(v, lid).get("type") == "lateral":
                        lat.append(_key_id(v))
            self.lanes[k] = LaneGeom(k, xyz, s, float(s[-1]), succ, lat)

# ---------- IDM + MOBIL core ----------

@dataclass
class IDMParams:
    v0: float = 14.0        # desired speed [m/s]
    T: float = 1.2          # time headway [s]
    a_max: float = 1.2      # max accel [m/s^2]
    b_comf: float = 2.0     # comfortable braking [m/s^2]
    s0: float = 2.0         # jam distance [m]
    delta: float = 4.0      # exponent

@dataclass
class MOBILParams:
    politeness: float = 0.3
    a_thr: float = 0.1          # incentive threshold [m/s^2]
    b_safe: float = 3.0         # max allowed braking for new follower [m/s^2]
    lc_cooldown_s: float = 2.0  # seconds after a lane change
    min_progress_m: float = 5.0 # don't change lanes when too close to lane end
    min_gap_m: float = 1.0      # minimal geometric gap considered

class BatchIDM:
    """
    Controls:
      • vehicles on lane centerlines (IDM + MOBIL lane-changing)
      • pedestrians along their route polylines (constant-speed)

    Expected agent fields (from your generator):
      - agent_id, cls, size_lwh_m, avg_speed_mps
      - start_xyz, route_xyz (peds) OR lane_ids/edge_ids (optional for vehicles)
    """
    def __init__(self, G_lane: nx.DiGraph,
                 idm: IDMParams = IDMParams(),
                 mobil: MOBILParams = MOBILParams()):
        self.atlas = LaneAtlas(G_lane)
        self.idm = idm
        self.mobil = mobil

        # runtime state
        self.veh_idx: Dict[str, int] = {}
        self.ped_idx: Dict[str, int] = {}

        self.ids: List[str] = []
        self.is_vehicle: np.ndarray = np.zeros(0, bool)

        # veh state
        self.veh_lane: List[Any] = []
        self.veh_s: np.ndarray = np.zeros(0, float)
        self.veh_v: np.ndarray = np.zeros(0, float)
        self.veh_len: np.ndarray = np.zeros(0, float)
        self.veh_route: List[List[Any]] = []   # optional lane path if provided
        self.veh_cooldown: np.ndarray = np.zeros(0, float)

        # ped state
        self.ped_xyz: List[np.ndarray] = []
        self.ped_s: np.ndarray = np.zeros(0, float)
        self.ped_v: np.ndarray = np.zeros(0, float)
        self.ped_route: List[np.ndarray] = []
        self.ped_route_s: List[np.ndarray] = []

    # ---------- attach agents ----------

    def attach_agents(self, agents: List[Dict[str, Any]], default_v0_by_class: Dict[str, float] | None = None):
        """
        Initialize internal state from generator agents.
        If a vehicle doesn't carry lane_ids, we snap to the nearest lane by geometry.
        """
        default_v0_by_class = default_v0_by_class or {"car": 14.0, "truck": 11.0, "bicycle": 5.0}

        self.ids = [a["agent_id"] for a in agents]
        n = len(agents)
        self.is_vehicle = np.array([a["cls"] != "pedestrian" for a in agents], bool)

        # slot indices
        self.veh_idx.clear(); self.ped_idx.clear()
        for i, a in enumerate(agents):
            if self.is_vehicle[i]:
                self.veh_idx[a["agent_id"]] = len(self.veh_lane)
            else:
                self.ped_idx[a["agent_id"]] = len(self.ped_route)

        # vehicles
        v_lane: List[Any] = []
        v_s: List[float] = []
        v_v: List[float] = []
        v_len: List[float] = []
        v_route: List[List[Any]] = []
        v_cooldown: List[float] = []

        # quick lane snap (brutal but robust)
        def _snap_lane(xy: np.ndarray) -> Tuple[Any, float]:
            best = None
            for L in self.atlas.lanes.values():
                sproj = min(max(0.0, float(np.argmin((L.xyz[:,0]-xy[0])**2 + (L.xyz[:,1]-xy[1])**2))), len(L.s)-1)
                # refine by projecting onto polyline length, cheap: walk nearest vertex arclen
                s_on = L.s[int(sproj)]
                # fallback: mid-step
                d = np.linalg.norm(L.xyz[int(sproj), :2] - xy[:2])
                if (best is None) or (d < best[0]): best = (d, L.lane_id, s_on)
            return best[1], best[2]

        for i, a in enumerate(agents):
            if not self.is_vehicle[i]:
                continue
            length = float(a.get("size_lwh_m", (4.0,1.8,1.6))[0])
            v0 = float(a.get("avg_speed_mps", default_v0_by_class.get(a["cls"], 12.0)))

            # choose lane: prefer provided lane_ids[0], else snap
            if a.get("lane_ids"):
                lane0 = _key_id(a["lane_ids"][0])
                if lane0 not in self.atlas.lanes:
                    # fallback snap
                    lane0, s0 = _snap_lane(np.asarray(a["start_xyz"], float))
                else:
                    # project start onto lane arclen (nearest by arclen grid)
                    L = self.atlas.lanes[lane0]
                    s0 = float(L.s[min(range(len(L.s)), key=lambda k: np.linalg.norm(L.xyz[k,:2]-np.asarray(a["start_xyz"][:2])))])
            else:
                lane0, s0 = _snap_lane(np.asarray(a["start_xyz"], float))

            v_lane.append(lane0)
            v_s.append(s0)
            v_v.append(max(0.0, 0.5*v0))  # start conservatively
            v_len.append(length)
            v_route.append([_key_id(l) for l in a.get("lane_ids", [])])
            v_cooldown.append(0.0)

        self.veh_lane = v_lane
        self.veh_s = np.array(v_s, float)
        self.veh_v = np.array(v_v, float)
        self.veh_len = np.array(v_len, float)
        self.veh_route = v_route
        self.veh_cooldown = np.array(v_cooldown, float)

        # pedestrians
        p_xyz: List[np.ndarray] = []
        p_s: List[float] = []
        p_v: List[float] = []
        p_route: List[np.ndarray] = []
        p_route_s: List[np.ndarray] = []
        for i, a in enumerate(agents):
            if self.is_vehicle[i]:
                continue
            P = _to_xyz(a["route_xyz"])
            s = _arclen2d(P[:, :2])
            p_route.append(P); p_route_s.append(s)
            # set starting s to 0 (route is already sliced in your generator)
            p_xyz.append(P[0].copy())
            p_s.append(0.0)
            p_v.append(float(a.get("avg_speed_mps", 1.4)))

        self.ped_xyz = p_xyz
        self.ped_s = np.array(p_s, float)
        self.ped_v = np.array(p_v, float)
        self.ped_route = p_route
        self.ped_route_s = p_route_s

    # ---------- physics ----------

    def _idm_acc(self, v, gap, dv):
        """IDM acceleration (dv = v - v_lead; positive if approaching)."""
        p = self.idm
        s_star = p.s0 + v*p.T + v*dv/(2.0*np.sqrt(p.a_max*p.b_comf) + 1e-6)
        term_free = (v / max(p.v0, 1e-3))**p.delta
        term_int  = (s_star / max(gap, 1e-3))**2
        return p.a_max * (1.0 - term_free - term_int)

    # ---------- step ----------

    def step(self, dt: float) -> Dict[str, Dict[str, Any]]:
        """
        Advance all agents by dt seconds. Returns a dict keyed by agent_id with:
          - xyz, heading_rad, speed_mps
          - (vehicles) lane_id
        """
        out: Dict[str, Dict[str, Any]] = {}

        # --- pedestrians: integrate along their route ---
        for aid, idx in self.ped_idx.items():
            s_tab = self.ped_route_s[idx]; P = self.ped_route[idx]
            self.ped_s[idx] = min(s_tab[-1], self.ped_s[idx] + self.ped_v[idx]*dt)
            xyz = _interp_xyz_at_s(P, s_tab, self.ped_s[idx])
            hdg = _heading_at_s(P, s_tab, self.ped_s[idx])
            self.ped_xyz[idx] = xyz
            out[aid] = {"xyz": xyz, "heading_rad": hdg, "speed_mps": float(self.ped_v[idx])}

        # --- vehicles: per-lane ordering and leaders ---
        # build lane → list of indices
        lane_to_inds: Dict[Any, List[int]] = {}
        for i, lane in enumerate(self.veh_lane):
            lane_to_inds.setdefault(lane, []).append(i)

        # prepare longitudinal decisions
        a_cmd = np.zeros_like(self.veh_v)

        # lane-change proposals (i -> target lane)
        lc_proposal: Dict[int, Any] = {}

        # cooldown decay
        self.veh_cooldown = np.maximum(0.0, self.veh_cooldown - dt)

        # evaluate per-lane
        for lane, inds in lane_to_inds.items():
            L = self.atlas.lanes[lane]
            # order by s
            inds.sort(key=lambda i: self.veh_s[i])
            # find leaders
            for k, i in enumerate(inds):
                v = self.veh_v[i]
                if k < len(inds)-1:
                    j = inds[k+1]
                    gap = max(self.mobil.min_gap_m, (self.veh_s[j] - self.veh_s[i] - 0.5*(self.veh_len[i]+self.veh_len[j])))
                    dv  = v - self.veh_v[j]
                    a_cmd[i] = self._idm_acc(v, gap, dv)
                else:
                    # virtual leader near lane end
                    dist_to_end = max(0.0, L.length - self.veh_s[i])
                    vlead = self.idm.v0
                    gap = max(self.mobil.min_gap_m, dist_to_end)
                    dv  = v - vlead
                    a_cmd[i] = self._idm_acc(v, gap, dv)

            # lane-change candidates (MOBIL) for vehicles on this lane
            for i in inds:
                if self.veh_cooldown[i] > 1e-6:
                    continue
                # avoid LC very close to lane end
                if (L.length - self.veh_s[i]) < self.mobil.min_progress_m:
                    continue
                # neighbors
                for tgt in L.laterals:
                    if tgt not in lane_to_inds and tgt not in self.atlas.lanes:
                        continue
                    TL = self.atlas.lanes[tgt]

                    # compute local neighbors on target lane (ahead/behind)
                    ti = lane_to_inds.get(tgt, [])
                    # add sentinel ends
                    ahead_gap = TL.length - self.veh_s[i]
                    ahead_v   = self.idm.v0
                    behind_gap = self.veh_s[i]
                    behind_v   = self.idm.v0
                    # find closest ahead/behind by s on target
                    if ti:
                        # choose by nearest s greater/smaller than our s (approximate mapping)
                        s_i = self.veh_s[i]
                        # sort once
                        ti_sorted = sorted(ti, key=lambda m: self.veh_s[m])
                        # ahead
                        ahead = [m for m in ti_sorted if self.veh_s[m] > s_i]
                        if ahead:
                            m = ahead[0]
                            ahead_gap = max(self.mobil.min_gap_m, self.veh_s[m]-s_i - 0.5*(self.veh_len[i]+self.veh_len[m]))
                            ahead_v   = self.veh_v[m]
                        # behind
                        behind = [m for m in ti_sorted if self.veh_s[m] < s_i]
                        if behind:
                            m = behind[-1]
                            behind_gap = max(self.mobil.min_gap_m, s_i - self.veh_s[m] - 0.5*(self.veh_len[i]+self.veh_len[m]))
                            behind_v   = self.veh_v[m]

                    # current lane leader for "before" acceleration
                    # (reuse a_cmd[i] which is current-lane IDM)
                    a_old_me = a_cmd[i]

                    # after-LC acceleration for ego on target lane (IDM vs its new leader)
                    v_me = self.veh_v[i]
                    dv_ahead = v_me - ahead_v
                    a_new_me = self._idm_acc(v_me, ahead_gap, dv_ahead)

                    # effect on the new follower (behind) on target lane:
                    # before: that follower saw its own leader; we approximate by free-road accel 0
                    # after: it sees us as leader -> recompute
                    a_old_foll = 0.0
                    a_new_foll = self._idm_acc(behind_v, behind_gap, behind_v - v_me)

                    # incentive: ego gain + politeness*(follower loss)
                    incentive = (a_new_me - a_old_me) + self.mobil.politeness * (a_new_foll - a_old_foll)

                    # safety: the new follower must not brake harder than -b_safe
                    safe = (a_new_foll >= -self.mobil.b_safe)

                    if safe and (incentive > self.mobil.a_thr):
                        # store best target by highest incentive
                        prev = lc_proposal.get(i, (None, -1e9))
                        if incentive > prev[1]:
                            lc_proposal[i] = (tgt, incentive)

        # resolve simultaneous swaps (simple priority by biggest incentive then agent index)
        if lc_proposal:
            # target occupancy check: allow multiple as long as local gaps remain (we'll be conservative: allow all)
            order = sorted([(i, tgt, inc) for i,(tgt,inc) in lc_proposal.items()],
                           key=lambda t: (-t[2], t[0]))
            for i, tgt, _ in order:
                self.veh_lane[i] = tgt
                self.veh_cooldown[i] = max(self.veh_cooldown[i], self.mobil.lc_cooldown_s)

        # after possible LC, recompute longitudinal accel quickly (again, same-lane leaders)
        lane_to_inds.clear()
        for i, lane in enumerate(self.veh_lane):
            lane_to_inds.setdefault(lane, []).append(i)
        a_cmd[:] = 0.0
        for lane, inds in lane_to_inds.items():
            L = self.atlas.lanes[lane]
            inds.sort(key=lambda i: self.veh_s[i])
            for k, i in enumerate(inds):
                v = self.veh_v[i]
                if k < len(inds)-1:
                    j = inds[k+1]
                    gap = max(self.mobil.min_gap_m, (self.veh_s[j] - self.veh_s[i] - 0.5*(self.veh_len[i]+self.veh_len[j])))
                    dv  = v - self.veh_v[j]
                    a_cmd[i] = self._idm_acc(v, gap, dv)
                else:
                    dist_to_end = max(0.0, L.length - self.veh_s[i])
                    vlead = self.idm.v0
                    gap = max(self.mobil.min_gap_m, dist_to_end)
                    dv  = v - vlead
                    a_cmd[i] = self._idm_acc(v, gap, dv)

        # integrate vehicles (semi-implicit Euler)
        self.veh_v = np.clip(self.veh_v + a_cmd*dt, 0.0, 100.0)
        self.veh_s = self.veh_s + self.veh_v*dt

        # lane end transitions
        for i, lane in enumerate(list(self.veh_lane)):
            L = self.atlas.lanes[lane]
            if self.veh_s[i] <= L.length - 1e-6:
                continue
            # need successor
            next_lane = None
            # prefer route-follow if provided
            if self.veh_route and len(self.veh_route[i]) >= 2:
                cur = _key_id(self.veh_route[i][0])
                nxt = _key_id(self.veh_route[i][1])
                if cur == lane and nxt in L.successors:
                    next_lane = nxt
                    # pop front
                    self.veh_route[i].pop(0)
            if next_lane is None:
                # generic: pick any successor, prefer one that exists
                if L.successors:
                    next_lane = L.successors[0]
            if next_lane is None:
                # dead-end: clamp to end of lane
                self.veh_s[i] = L.length
                self.veh_v[i] = 0.0
            else:
                NL = self.atlas.lanes[next_lane]
                self.veh_lane[i] = next_lane
                # carry overflow s beyond end (simple continuity)
                overflow = self.veh_s[i] - L.length
                self.veh_s[i] = min(NL.length, overflow)
                # small cooldown to avoid immediate LC at merges
                self.veh_cooldown[i] = max(self.veh_cooldown[i], 0.5)

        # write vehicle outputs
        for aid, idx in self.veh_idx.items():
            L = self.atlas.lanes[self.veh_lane[idx]]
            s = float(np.clip(self.veh_s[idx], 0.0, L.length))
            xyz = _interp_xyz_at_s(L.xyz, L.s, s)
            hdg = _heading_at_s(L.xyz, L.s, s)
            out[aid] = {
                "xyz": xyz,
                "heading_rad": hdg,
                "speed_mps": float(self.veh_v[idx]),
                "lane_id": self.veh_lane[idx],
            }

        return out
