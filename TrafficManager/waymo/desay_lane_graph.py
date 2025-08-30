# lane_graph_quintic_connectors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# ---------- helpers ----------
def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2: raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3: raise ValueError("array must be [N,3],[N,2],or[3,N]")
    return a

def _arclen2d(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

def _resample_xyz_count(xyz: np.ndarray, n: int = 200) -> np.ndarray:
    xyz = _to_xyz(xyz)
    if len(xyz) == 1: return np.repeat(xyz, n, axis=0)
    s = _arclen2d(xyz[:, :2])
    if s[-1] == 0: return np.repeat(xyz[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    x = np.interp(u, s, xyz[:,0]); y = np.interp(u, s, xyz[:,1]); z = np.interp(u, s, xyz[:,2])
    return np.stack([x,y,z], axis=1)

def _tangent2d(xy: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    v = np.zeros_like(xy); v[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] >= window and window >= 3:
        if window % 2 == 0: window += 1
        v = savgol_filter(v, window, poly, axis=0, mode="interp")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _end_tangents(xyz: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    xy = _to_xyz(xyz)[:, :2]
    k = max(1, min(k, len(xy)-1))
    t0 = xy[min(k, len(xy)-1)] - xy[0]
    t1 = xy[-1] - xy[max(0, len(xy)-1-k)]
    for t in (t0, t1):
        n = np.linalg.norm(t)
        if n > 0: t /= n
    return t0, t1

def _closest_on_ls(ls: LineString, p_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    s = ls.project(Point(float(p_xy[0]), float(p_xy[1])))
    q = ls.interpolate(s)
    return np.array([q.x, q.y]), float(s)

def _smooth_xyz(xyz: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    if window < 3 or len(xyz) < window: return xyz
    if window % 2 == 0: window += 1
    out = [savgol_filter(xyz[:,d], window, poly, mode="interp") for d in range(3)]
    return np.stack(out, axis=1)

# -------- curvature & slope at endpoints --------
def _rotate_left(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])

def _endpoint_curvature(xy: np.ndarray, end: str = "end", m: int = 6) -> Tuple[float, np.ndarray]:
    """
    Estimate signed curvature kappa (1/m) and left normal at the chosen end.
    Uses 3-point osculating circle on a short window near the end.
    """
    xy = np.asarray(xy, dtype=float)
    m = max(2, min(m, len(xy)-1))
    if end == "start":
        p0, p1, p2 = xy[0], xy[m//2], xy[m]
        t = xy[min(m, len(xy)-1)] - xy[0]
    else:
        p0, p1, p2 = xy[-1], xy[-1-m//2], xy[-1-m]
        t = xy[-1] - xy[max(0, len(xy)-1-m)]
    # triangle area (signed)
    a = p1 - p0; b = p2 - p1; c = p2 - p0
    area2 = a[0]*b[1] - a[1]*b[0]
    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    if la*lb*lc == 0:
        return 0.0, _rotate_left(t / (np.linalg.norm(t)+1e-9))
    kappa = 2*area2 / (la*lb*lc + 1e-9)   # signed
    T = t / (np.linalg.norm(t)+1e-9)
    N_left = _rotate_left(T)              # unit
    return float(kappa), N_left

def _endpoint_z_slope(xyz: np.ndarray, end: str = "end", k: int = 6) -> float:
    """Estimate dz/ds at the endpoint."""
    xyz = _to_xyz(xyz)
    k = max(1, min(k, len(xyz)-1))
    if end == "start":
        ds = np.linalg.norm(xyz[k,:2] - xyz[0,:2]) + 1e-9
        return float((xyz[k,2] - xyz[0,2]) / ds)
    else:
        ds = np.linalg.norm(xyz[-1,:2] - xyz[-1-k,:2]) + 1e-9
        return float((xyz[-1,2] - xyz[-1-k,2]) / ds)

# -------- quintic Hermite (C2) connector --------
def _quintic_hermite_connector_3d(
    P0: np.ndarray, T0_xy: np.ndarray, K0: float, N0_xy: np.ndarray, z_slope0: float,
    P1: np.ndarray, T1_xy: np.ndarray, K1: float, N1_xy: np.ndarray, z_slope1: float,
    n_samples: int = 65, scale: Optional[float] = None, accel_clamp: float = 0.2
) -> np.ndarray:
    """
    Position P, velocity V, acceleration A constraints at both ends.
    V0 = S*T0, V1 = S*T1 ; A0 = S^2 * K0 * N0 ; A1 = S^2 * K1 * N1
    S defaults to chord length unless 'scale' provided.
    """
    P0 = _to_xyz(P0.reshape(1,-1))[0]; P1 = _to_xyz(P1.reshape(1,-1))[0]
    chord = np.linalg.norm(P1[:2] - P0[:2]) + 1e-9
    S = float(scale if scale is not None else chord)
    # velocities
    V0_xy = T0_xy / (np.linalg.norm(T0_xy)+1e-9) * S
    V1_xy = T1_xy / (np.linalg.norm(T1_xy)+1e-9) * S
    # accelerations (limit magnitude to avoid overshoot)
    A0_xy = K0 * (S**2) * (N0_xy / (np.linalg.norm(N0_xy)+1e-9))
    A1_xy = K1 * (S**2) * (N1_xy / (np.linalg.norm(N1_xy)+1e-9))
    # clamp accelerations to a fraction of S to prevent wild swings
    Amax = accel_clamp * S
    if np.linalg.norm(A0_xy) > Amax: A0_xy = A0_xy / (np.linalg.norm(A0_xy)+1e-9) * Amax
    if np.linalg.norm(A1_xy) > Amax: A1_xy = A1_xy / (np.linalg.norm(A1_xy)+1e-9) * Amax

    # 1D z: use same quintic with slopes and zero accel at ends (robust)
    V0_z = z_slope0 * S
    V1_z = z_slope1 * S
    A0_z = 0.0
    A1_z = 0.0

    t = np.linspace(0, 1, n_samples)
    # quintic Hermite basis (pos/vel/acc at t=0,1)
    b0 = 1 - 10*t**3 + 15*t**4 - 6*t**5
    b1 = t - 6*t**3 + 8*t**4 - 3*t**5
    b2 = 0.5*(t**2 - 3*t**3 + 3*t**4 - t**5)
    b3 = 10*t**3 - 15*t**4 + 6*t**5
    b4 = -4*t**3 + 7*t**4 - 3*t**5
    b5 = 0.5*(t**3 - 2*t**4 + t**5)

    X = (b0[:,None]*P0[:2] + b1[:,None]*V0_xy + b2[:,None]*A0_xy +
         b3[:,None]*P1[:2] + b4[:,None]*V1_xy + b5[:,None]*A1_xy)
    Z = b0*P0[2] + b1*V0_z + b2*A0_z + b3*P1[2] + b4*V1_z + b5*A1_z
    return np.column_stack([X, Z])

# ---------- boundaries + clamping ----------
@dataclass
class BoundaryRec:
    id: int
    xyz: np.ndarray
    s: np.ndarray
    geom2d: LineString

def _densify_xyz(xyz: np.ndarray, step: float = 0.5) -> np.ndarray:
    xyz = _to_xyz(xyz)
    if len(xyz) < 2: return xyz
    out = [xyz[0]]
    for a,b in zip(xyz[:-1], xyz[1:]):
        L = np.linalg.norm(b[:2]-a[:2])
        if L <= step or L == 0: out.append(b); continue
        n = int(np.ceil(L/step)); ts = np.linspace(0,1,n+1)[1:]
        out.extend(a + (b-a)*ts[:,None])
    return np.asarray(out)

def _build_boundary_cache(boundary_dict: Dict[int, np.ndarray], densify_step=0.5):
    recs = []
    for bid, arr in boundary_dict.items():
        xyz = _densify_xyz(arr, densify_step)
        if len(xyz) < 2: continue
        s = _arclen2d(xyz[:, :2])
        recs.append(BoundaryRec(int(bid), xyz, s, LineString(xyz[:, :2])))
    tree = STRtree([r.geom2d for r in recs])
    return recs, tree

def _interp_xyz_at_s(rec: BoundaryRec, s: float) -> np.ndarray:
    x = np.interp(s, rec.s, rec.xyz[:,0])
    y = np.interp(s, rec.s, rec.xyz[:,1])
    z = np.interp(s, rec.s, rec.xyz[:,2])
    return np.array([x,y,z], dtype=float)

def _nearest_LR_boundaries(p_xy: np.ndarray, t_xy: np.ndarray,
                           recs: List[BoundaryRec], tree: STRtree,
                           search_radius: float = 25.0):
    env = Point(float(p_xy[0]), float(p_xy[1])).buffer(search_radius)
    idxs = tree.query(env)
    if not isinstance(idxs, (list, tuple, np.ndarray)): idxs = [idxs]
    bestL = None; bestR = None
    for idx in idxs:
        rec = recs[idx]
        q_xy, s = _closest_on_ls(rec.geom2d, p_xy)
        d = np.linalg.norm(q_xy - p_xy); d = d if d>0 else 1e-9
        cross = t_xy[0]*(q_xy[1]-p_xy[1]) - t_xy[1]*(q_xy[0]-p_xy[0])
        if cross > 0:
            if (bestL is None) or (d < bestL[0]): bestL = (d, rec, q_xy, s)
        else:
            if (bestR is None) or (d < bestR[0]): bestR = (d, rec, q_xy, s)
    return bestL, bestR

def _clamp_point_between_LR(p: np.ndarray, t_xy: np.ndarray, recL, recR) -> np.ndarray:
    _, rL, qL_xy, sL = recL; _, rR, qR_xy, sR = recR
    v = qL_xy - qR_xy; v2 = float(np.dot(v, v)) + 1e-9
    alpha = float(np.clip(np.dot(p[:2] - qR_xy, v) / v2, 0.0, 1.0))
    qL_xyz = _interp_xyz_at_s(rL, sL); qR_xyz = _interp_xyz_at_s(rR, sR)
    p_xy  = qR_xy + alpha * v
    z     = (1.0 - alpha) * qR_xyz[2] + alpha * qL_xyz[2]
    return np.array([p_xy[0], p_xy[1], z], dtype=float)

# ---------- your input ----------
@dataclass
class CenterlineResult:
    boundary_a_id: int
    boundary_b_id: int
    side: str
    parallelism: float
    mean_gap_m: float
    lane_index: int
    lane_count: int
    alpha: float
    centerline: np.ndarray   # [M,3]

# ---------- MAIN: smoother connectors ----------
def build_lane_graph_with_connectors(
    centerlines: List[CenterlineResult],
    boundary_dict: Dict[int, np.ndarray],
    *,
    resample_n: int = 200,
    successor_radius_m: float = 20.0,
    forward_max_angle_deg: float = 40.0,
    turn_max_angle_deg: float = 120.0,
    allow_u_turn: bool = False,
    # Quintic connector & smoothing controls:
    connector_samples: int = 65,
    accel_clamp_frac: float = 0.25,   # clamp accel vs S to avoid overshoot
    proj_smooth_iters: int = 2,       # projected smoothing passes
    chaikin_alpha: float = 0.33,      # Chaikin fraction (0.25..0.4 usually good)
    sg_window: int = 9,               # Savitzky–Golay window for final denoise
    clamp_search_radius_m: float = 25.0,
    lateral_within_same_pair_only: bool = True,
    lateral_min_m: float = 2.4, lateral_max_m: float = 5.0,
    lateral_min_overlap_frac: float = 0.35, lateral_min_orient_cos: float = 0.9,
) -> nx.DiGraph:

    def lane_key(c: CenterlineResult):
        return (tuple(sorted((c.boundary_a_id, c.boundary_b_id))), c.side, c.lane_index)
    lanes: Dict[Any, CenterlineResult] = {lane_key(c): c for c in centerlines}

    G = nx.DiGraph()
    for lid, c in lanes.items():
        xyz_rs = _resample_xyz_count(c.centerline, resample_n)
        G.add_node(lid,
                   xyz=xyz_rs,
                   start_xy=xyz_rs[0,:2],
                   end_xy=xyz_rs[-1,:2],
                   boundary_pair=tuple(sorted((c.boundary_a_id, c.boundary_b_id))),
                   side=c.side, lane_index=c.lane_index, lane_count=c.lane_count)

    starts = np.array([G.nodes[i]['start_xy'] for i in G.nodes]); start_ids = list(G.nodes)
    start_tree = cKDTree(starts) if len(starts) else None

    brecs, btree = _build_boundary_cache(boundary_dict, densify_step=0.5)

    def endpoint_frames(xyz: np.ndarray):
        xy = xyz[:, :2]
        T0, T1 = _end_tangents(xyz)
        k0, N0 = _endpoint_curvature(xy, "start", m=max(6, len(xy)//20))
        k1, N1 = _endpoint_curvature(xy, "end",   m=max(6, len(xy)//20))
        z0 = _endpoint_z_slope(xyz, "start"); z1 = _endpoint_z_slope(xyz, "end")
        return T0, T1, k0, N0, k1, N1, z0, z1

    # ---- successors with smooth (quintic) connectors ----
    for u in G.nodes:
        Xu = G.nodes[u]['xyz']
        Pu0, Pu1 = Xu[0], Xu[-1]
        Tu0, Tu1, ku0, Nu0, ku1, Nu1, dz0, dz1 = endpoint_frames(Xu)
        if start_tree is None: break
        idxs = start_tree.query_ball_point(Pu1[:2], r=successor_radius_m)
        for j in idxs:
            v = start_ids[j]
            if v == u: continue
            Xv = G.nodes[v]['xyz']
            Pv0, Pv1 = Xv[0], Xv[-1]
            Tv0, Tv1, kv0, Nv0, kv1, Nv1, dzv0, dzv1 = endpoint_frames(Xv)

            cosang = float(np.clip(np.dot(Tu1, Tv0), -1.0, 1.0))
            ang_deg = float(np.degrees(np.arccos(cosang)))
            is_forward = ang_deg <= forward_max_angle_deg
            is_turn    = (forward_max_angle_deg < ang_deg <= turn_max_angle_deg)
            is_uturn   = (ang_deg > 150.0)
            if is_uturn and not allow_u_turn: continue
            if not (is_forward or is_turn or is_uturn): continue

            # Quintic connector (C2)
            conn = _quintic_hermite_connector_3d(
                P0=Pu1, T0_xy=Tu1, K0=ku1, N0_xy=Nu1, z_slope0=dz1,
                P1=Pv0, T1_xy=Tv0, K1=kv0, N1_xy=Nv0, z_slope1=dzv0,
                n_samples=connector_samples,
                scale=np.linalg.norm(Pv0[:2]-Pu1[:2]),
                accel_clamp=accel_clamp_frac
            )

            # Projected smoothing: Chaikin corner cutting + SG, with re-clamp
            def chaikin(P, a=chaikin_alpha):
                if len(P) < 3: return P
                Q = [P[0]]
                for i in range(len(P)-1):
                    Q.append((1-a)*P[i] + a*P[i+1])
                    Q.append(a*P[i] + (1-a)*P[i+1])
                Q.append(P[-1])
                return np.asarray(Q)

            # dynamic clamp for every sample using nearest L/R boundaries
            def clamp_curve(curve):
                Tconn = _tangent2d(curve[:, :2], window=max(5, len(curve)//10))
                out = []
                for p, t in zip(curve, Tconn):
                    L, R = _nearest_LR_boundaries(p[:2], t, brecs, btree, search_radius=clamp_search_radius_m)
                    if L is not None and R is not None:
                        out.append(_clamp_point_between_LR(p, t, L, R))
                    else:
                        out.append(p)
                C = np.asarray(out)
                return _smooth_xyz(C, window=min(sg_window, (len(C)//2)*2 - 1), poly=2)
            # plt.plot(conn[:,0],conn[:,1],'r')
            # plt.show()
            #
            #for _ in range(max(0, proj_smooth_iters)):
            #     conn = chaikin(conn)
             #   conn = clamp_curve(conn)
            #
            # # final light denoise (keeps it silky)
            conn = _smooth_xyz(conn, window=min(sg_window, (len(conn)//2)*2 - 1), poly=2)

            cost = float(np.sum(np.linalg.norm(np.diff(conn[:, :2], axis=0), axis=1)))
            lateral_sign = Tu1[0]*(Pv0[:2][1]-Pu1[:2][1]) - Tu1[1]*(Pv0[:2][0]-Pu1[:2][0])
            turn_type = "forward" if is_forward else ("left" if lateral_sign > 0 else "right") if is_turn else "u-turn"
            G.add_edge(u, v, type="successor", turn=turn_type, angle_deg=ang_deg, cost=cost, connector_xyz=conn)

            # plt.plot(conn[:,0],conn[:,1],'r')
            # plt.show()

    # ---- lateral (optional, same-corridor only) ----
    if lateral_within_same_pair_only:
        by_pair: Dict[Tuple[int,int], List[Any]] = {}
        for lid, data in G.nodes(data=True):
            by_pair.setdefault(data['boundary_pair'], []).append(lid)
        for pair, ids in by_pair.items():
            for i in ids:
                xi = G.nodes[i]['xyz'][:, :2]; ti = _tangent2d(xi)
                probes = np.linspace(0, len(xi)-1, 50).astype(int)
                bestL = None; bestR = None
                for j in ids:
                    if i == j: continue
                    xj = G.nodes[j]['xyz'][:, :2]; geomj = LineString(xj)
                    gaps = []
                    for idx in probes:
                        p = xi[idx]; t = ti[idx]
                        q,_ = _closest_on_ls(geomj, p)
                        lat = t[0]*(q[1]-p[1]) - t[1]*(q[0]-p[0])
                        gaps.append(lat)
                    if len(gaps) < 8: continue
                    gaps = np.array(gaps)
                    valid = (np.abs(gaps) >= lateral_min_m) & (np.abs(gaps) <= lateral_max_m)
                    if float(np.mean(valid)) < lateral_min_overlap_frac: continue
                    med = float(np.median(gaps[valid]))
                    if med > 0:
                        if (bestL is None) or (abs(med) < bestL[0]): bestL = (abs(med), j)
                    else:
                        if (bestR is None) or (abs(med) < bestR[0]): bestR = (abs(med), j)
                if bestL:
                    G.add_edge(i, bestL[1], type="lateral", side="left", median_gap_m=bestL[0])
                    G.add_edge(bestL[1], i, type="lateral", side="right", median_gap_m=bestL[0])
                if bestR:
                    G.add_edge(i, bestR[1], type="lateral", side="right", median_gap_m=bestR[0])
                    G.add_edge(bestR[1], i, type="lateral", side="left", median_gap_m=bestR[0])

    return G


# plot_lane_graph_connectors.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _to_xy(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError("array must be 2D")
    if a.shape[1] == 3: return a[:, :2]
    if a.shape[1] == 2: return a
    if a.shape[0] == 3 and a.shape[1] != 3: return a.T[:, :2]
    raise ValueError("expected shape [N,3] or [N,2] or [3,N]")

def _mid_xy(xyz):
    xy = _to_xy(xyz)
    return xy[len(xy)//2]

def _color_for_pair(pair):
    # stable color per boundary pair
    rng = (hash(pair) % 9973) / 9973.0
    return plt.cm.tab20(int(rng*20) % 20)

def plot_lane_graph(
    G,
    *,
    boundaries: dict[int, np.ndarray] | None = None,
    show_ids: bool = False,
    draw_lateral: bool = True,         # you said: “lateral not across boundary”—default off
    draw_successor: bool = True,
    draw_connectors: bool = True,
    lane_lw: float = 2.0,
    connector_lw: float = 2.5,
    boundary_lw: float = 1.0,
    figsize=(11, 11),
    save_path: str | None = None,
    ax=None,
):
    """Plot lane graph with optional boundaries + smooth successor connectors."""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # 0) Boundaries (underlay)
    if boundaries:
        b_lines = [ _to_xy(v) for v in boundaries.values() if len(v) >= 2 ]
        if b_lines:
            blc = LineCollection(b_lines, linewidths=boundary_lw, alpha=0.6, linestyle='-', color='k')
            ax.add_collection(blc)

    # 1) Lane centerlines (color by boundary_pair)
    segs = []
    cols = []
    for nid, data in G.nodes(data=True):
        xy = _to_xy(data['xyz'])
        segs.append(xy)
        pair = data.get('boundary_pair', ('?', '?'))
        cols.append(_color_for_pair(pair))
    if segs:
        lc = LineCollection(segs, linewidths=lane_lw, colors=cols, alpha=0.95)
        ax.add_collection(lc)

    # 2) Successor edges (prefer connector geometry)
    if draw_successor:
        for u, v, d in G.edges(data=True):
            if d.get('type') != 'successor':
                continue
            if draw_connectors and ('connector_xyz' in d) and (len(d['connector_xyz']) >= 2):
                c = d['connector_xyz']
                ax.plot(c[:,0], c[:,1], linewidth=connector_lw, alpha=0.9)#, color='C3'
            else:
                su = np.asarray(G.nodes[u]['end_xy'], dtype=float)
                sv = np.asarray(G.nodes[v]['start_xy'], dtype=float)
                ax.annotate("", xy=(sv[0], sv[1]), xytext=(su[0], su[1]),
                            arrowprops=dict(arrowstyle="->", lw=1.3, alpha=0.8))#, color='C3'

    # 3) Lateral edges (optional; dashed mid→mid)
    if draw_lateral:
        seen = set()
        for u, v, d in G.edges(data=True):
            if d.get('type') != 'lateral':
                continue
            key = tuple(sorted((u, v)))
            if key in seen: continue
            seen.add(key)
            mu = _mid_xy(G.nodes[u]['xyz'])
            mv = _mid_xy(G.nodes[v]['xyz'])
            ax.plot([mu[0], mv[0]], [mu[1], mv[1]], linestyle="--", linewidth=1.1, alpha=0.6, color='C7')

    # 4) Optional labels
    if show_ids:
        for nid, data in G.nodes(data=True):
            m = _mid_xy(data['xyz'])
            ax.text(m[0], m[1], str(nid), fontsize=8, color='k')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.margins(0.05)
    plt.show()
    # if created:
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=220)
    return ax

