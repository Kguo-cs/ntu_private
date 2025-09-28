# static_elements_from_raw_v4.py
# Adds per-line spacing occupancy so placements never overlap prior objects' spacings.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


# ----------------- small geom utils -----------------

def _to_xyz(arr) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.ndim != 2:
        raise ValueError("array must be 2D")
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a))])
    if a.shape[1] != 3:
        raise ValueError("need [N,3]/[N,2]/[3,N]")
    return a

def _arclen2d(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return np.array([0.0])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _resample_polyline(xyz: np.ndarray, ds: float = 1.0, smooth: Optional[Tuple[int,int]] = None) -> np.ndarray:
    P = _to_xyz(xyz)
    s = _arclen2d(P[:, :2])
    L = float(s[-1])
    if L <= 1e-6:
        return P[:1]
    n = max(2, int(math.ceil(L / ds)) + 1)
    su = np.linspace(0.0, L, n)
    X = np.interp(su, s, P[:, 0])
    Y = np.interp(su, s, P[:, 1])
    Z = np.interp(su, s, P[:, 2])
    Q = np.stack([X, Y, Z], axis=1)
    if smooth is not None and _HAS_SAVGOL and n >= smooth[0] >= 3:
        w, poly = smooth
        if w % 2 == 0: w += 1
        for d in range(3):
            Q[:, d] = savgol_filter(Q[:, d], w, poly, mode="interp")
    return Q

def _tangent_heading_at(P: np.ndarray, idx: int) -> float:
    i0 = max(0, idx - 2); i1 = min(len(P)-1, idx + 2)
    dx = P[i1, 0] - P[i0, 0]
    dy = P[i1, 1] - P[i0, 1]
    if dx == 0 and dy == 0: return 0.0
    return math.atan2(dy, dx)

def _polyline_capacity_len(lines: List[np.ndarray]) -> float:
    total = 0.0
    for L in lines:
        total += float(_arclen2d(_to_xyz(L)[:, :2])[-1])
    return total


# ----------------- parameters -----------------

@dataclass
class StaticSpec:
    density01: float = 0.6
    ratios: Dict[str, float] = None                  # {"cone":1, "water_barrier":2, "hydrant":1}
    sizes_lwh_m: Dict[str, Tuple[float,float,float]] = None
    seed: int = 1234

    # belong-to filtering around ego edges
    belong_tol_m: float = 10.0
    ego_max_dist_m: float = 10.0

    # spacing baselines (the "spacing footprint" per object)
    spacing_m: Dict[str, float] = None               # cone: ~5m, barrier: ~2m, hydrant: ~25m

    # continuous run controls
    cone_run_min: int = 6
    cone_run_max: int = 15
    barrier_run_min: int = 6
    barrier_run_max: int = 15

    # resample params
    ds_resample_m: float = 1.0
    smooth: Optional[Tuple[int,int]] = (9, 2)

    # placement attempts
    max_run_shifts: int = 25       # how many s0 shifts we try when conflicts
    max_point_jitter_trials: int = 3  # after s picked, try small s jitter if still conflict

    def __post_init__(self):
        if self.ratios is None:
            self.ratios = {"cone":1, "water_barrier":2, "hydrant":1}
        if self.sizes_lwh_m is None:
            self.sizes_lwh_m = {
                "cone": (0.5, 0.5, 0.7),
                "water_barrier": (4.4, 0.5, 0.9),
                "hydrant": (0.6, 0.6, 0.9),
            }
        if self.spacing_m is None:
            self.spacing_m = {"cone":5.0, "water_barrier":5.0, "hydrant":25.0}


# ----------------- support-line collection (EGO-ONLY) -----------------

def _collect_support_lines_ego_only(
    EG,
    ego_edge_ids: List[Any],
    boundary_dict: Dict[int, np.ndarray],
    lane_dict: Dict[int, np.ndarray],
    spec: StaticSpec
) -> Tuple[List[np.ndarray], List[np.ndarray], object]:
    from shapely.ops import unary_union
    ego_edge_ids = set(ego_edge_ids or [])
    if not ego_edge_ids:
        return [], [], None

    ego_lines = []
    for u, v, ed in EG.edges(data=True):
        if ed.get("id") in ego_edge_ids:
            geom = ed.get("geom", ed.get("shape_xyz", None))
            if geom is not None:
                P = _resample_polyline(geom, ds=1.0)
                if len(P) >= 2:
                    ego_lines.append(LineString(P[:, :2]))
    if not ego_lines:
        return [], [], None
    ego_union = unary_union(ego_lines)

    explicit_boundary_ids: set[int] = set()
    explicit_divider_ids: set[int] = set()
    for _, _, ed in EG.edges(data=True):
        if ed.get("id") not in ego_edge_ids:
            continue
        for key in ("boundary_ids", "boundary_id_list"):
            if key in ed and isinstance(ed[key], (list, tuple)):
                explicit_boundary_ids.update(int(b) for b in ed[key])
        for key in ("boundary_a_id", "boundary_b_id"):
            if key in ed:
                try: explicit_boundary_ids.add(int(ed[key]))
                except Exception: pass
        for key in ("lane_divider_ids", "divider_ids", "lane_line_ids"):
            if key in ed and isinstance(ed[key], (list, tuple)):
                try:
                    explicit_divider_ids.update(int(x) for x in ed[key])
                except Exception:
                    pass

    belong_band = ego_union.buffer(spec.belong_tol_m)

    boundary_lines: List[np.ndarray] = []
    if explicit_boundary_ids:
        for bid in explicit_boundary_ids:
            B = boundary_dict.get(bid, None)
            if B is None: continue
            P = _resample_polyline(B, ds=spec.ds_resample_m, smooth=spec.smooth)
            if belong_band.intersects(LineString(P[:, :2])):
                boundary_lines.append(P)
    else:
        for B in boundary_dict.values():
            if B is None: continue
            P = _resample_polyline(B, ds=spec.ds_resample_m, smooth=spec.smooth)
            if belong_band.intersects(LineString(P[:, :2])):
                boundary_lines.append(P)

    divider_lines: List[np.ndarray] = []
    if explicit_divider_ids:
        for did in explicit_divider_ids:
            L = lane_dict.get(did, None)
            if L is None: continue
            P = _resample_polyline(L, ds=spec.ds_resample_m, smooth=spec.smooth)
            if belong_band.intersects(LineString(P[:, :2])):
                divider_lines.append(P)
    else:
        for L in lane_dict.values():
            if L is None: continue
            P = _resample_polyline(L, ds=spec.ds_resample_m, smooth=spec.smooth)
            if belong_band.intersects(LineString(P[:, :2])):
                divider_lines.append(P)

    return divider_lines, boundary_lines, ego_union

# ----------------- main generator -----------------

def generate_static_elements_from_raw(
    *,
    boundary_dict: Dict[int, np.ndarray],
    lane_dict: Dict[int, np.ndarray],        # your lane_line (solid/dot) act as lane dividers
    EG: Any,                                  # edge graph (networkx.DiGraph)
    ego_edge_ids: List[Any],                  # REQUIRED here to restrict
    ego_route_xyz: Optional[np.ndarray] = None,   # optional; hydrant relevance
    lane_graph: Optional[List[Polygon | MultiPolygon]] = None,
    spec: StaticSpec = StaticSpec(),
) -> List[Dict[str, Any]]:
    """
    Returns a list of static objects with size-aware spacing and continuous placement,
    **restricted to lane dividers / boundaries that BELONG to ego_edge_ids**.
    """
    rng = np.random.default_rng(spec.seed)

    # ---- strict support lines (ego-only) ----
    divider_lines, boundary_lines ,ego_union = _collect_support_lines_ego_only(
        EG, ego_edge_ids, boundary_dict, lane_dict, spec
    )
    support_lines = divider_lines + boundary_lines
    if not support_lines:
        return []

    spacing=5

    # ---- capacity & allocation ----
    support_len = _polyline_capacity_len(support_lines)

    total_cap= support_len / spacing*spec.density01

    alloc={}

    alloc["cone"] = int(total_cap*spec.ratios["cone"]/(spec.ratios["cone"]+spec.ratios["water_barrier"]))
    alloc["water_barrier"] =int(total_cap*spec.ratios["water_barrier"]/(spec.ratios["cone"]+spec.ratios["water_barrier"]))
    alloc["hydrant"] = int(alloc["cone"]*spec.ratios["hydrant"]/spec.ratios["cone"])

    line_occ: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(support_lines))}

    def _conflicts_on_line(i_line: int, s_center: float, S_new: float) -> bool:
        occ = line_occ[i_line]
        half_new = 0.5 * S_new
        for s_i, S_i in occ:
            if abs(s_center - s_i) < 0.5 * (S_new + S_i) - 1e-6:
                return True
        return False

    def _reserve_on_line(i_line: int, s_center: float, S_new: float):
        line_occ[i_line].append((s_center, S_new))

    # ------------ helpers for placement ------------
    def _place_runs_on_lines(
        cls: str,
        N_target: int,
        base_spacing: float,
        length_m: float,
        run_min: int,
        run_max: int,
        jitter_xy: float,
        jitter_heading: float,
        align_heading: bool = True,
    ) -> List[Tuple[np.ndarray, float]]:
        if N_target <= 0: return []
        # how many runs
        if run_min > run_max: run_max = run_min
        runs: List[int] = []
        remain = N_target
        while remain > 0:
            r = int(rng.integers(run_min, run_max + 1))
            r = min(r, remain)
            runs.append(r)
            remain -= r

        out: List[Tuple[np.ndarray, float]] = []
        if not support_lines:
            return out

        lens = np.array([float(_arclen2d(L[:, :2])[-1]) for L in support_lines], float)
        probs = lens / (lens.sum() if lens.sum() > 0 else 1.0)

        # per-class footprint (spacing footprint used in occupancy)
        S_fp = max(base_spacing, length_m)

        for r in runs:
            i_line = int(rng.choice(np.arange(len(support_lines)), p=probs))
            P = support_lines[i_line]
            s_arr = _arclen2d(P[:, :2]); Ltot = float(s_arr[-1])
            if Ltot <= 1e-6:
                continue

            step = max(base_spacing, length_m)
            run_len = (r - 1) * step
            if run_len > Ltot:
                r_fit = max(2, int(Ltot // max(1e-3, step)) + 1)
                if r_fit < 2:
                    continue
                r = r_fit
                run_len = (r - 1) * step

            # Try multiple s0 shifts to avoid occupancy conflicts
            success_run = False
            for _try in range(spec.max_run_shifts):
                s0 = float(rng.uniform(0.0, max(1e-6, Ltot - run_len)))
                # quick check: all centers OK?
                centers = [s0 + k * step for k in range(r)]
                if any(_conflicts_on_line(i_line, c, S_fp) for c in centers):
                    continue

                # All k points placeable → reserve & emit
                for k, u in enumerate(centers):
                    # tiny extra s jitter if needed to break edge cases
                    u_try = u
                    placed = False
                    for _jt in range(spec.max_point_jitter_trials):
                        if not _conflicts_on_line(i_line, u_try, S_fp):
                            # reserve the spacing footprint
                            _reserve_on_line(i_line, u_try, S_fp)
                            # compute pose
                            x = np.interp(u_try, s_arr, P[:, 0])
                            y = np.interp(u_try, s_arr, P[:, 1])
                            z = np.interp(u_try, s_arr, P[:, 2])
                            # heading along tangent
                            if len(P) > 1:
                                idxp = int(np.clip(round(u_try / max(1e-6, Ltot) * (len(P) - 1)), 0, len(P) - 1))
                            else:
                                idxp = 0
                            hd = _tangent_heading_at(P, idxp) if align_heading else float(rng.uniform(-math.pi, math.pi))
                            # position/heading jitter
                            jpar = float(rng.normal(0.0, jitter_xy * 0.15))
                            jlat = float(rng.normal(0.0, jitter_xy))
                            dx = jpar * math.cos(hd) - jlat * math.sin(hd)
                            dy = jpar * math.sin(hd) + jlat * math.cos(hd)
                            xy = np.array([x + dx, y + dy, z], float)
                            h = float(hd + rng.normal(0.0, jitter_heading))
                            out.append((xy, h))
                            placed = True
                            break
                        # small forward jitter (keeps order)
                        u_try = min(Ltot, u_try + 0.15 * step)

                    if not placed:
                        # If any point in the run fails, roll back reservations for this run and retry a new s0
                        # Rollback: remove just-reserved of this run
                        for c_prev in centers[:k]:
                            # erase last matching (c_prev, S_fp) from occ
                            occ = line_occ[i_line]
                            for j in range(len(occ)-1, -1, -1):
                                if abs(occ[j][0] - c_prev) < 1e-9 and abs(occ[j][1] - S_fp) < 1e-9:
                                    occ.pop(j); break
                        break  # try a new s0

                else:
                    # loop didn't break: whole run placed
                    success_run = True
                    break

            # if run placement failed after attempts, skip this run silently

        return out

    # cones
    L_cone = spec.sizes_lwh_m["cone"][0]
    cones = _place_runs_on_lines(
        "cone",
        N_target=alloc["cone"],
        base_spacing=spec.spacing_m["cone"],
        length_m=L_cone,
        run_min=spec.cone_run_min,
        run_max=spec.cone_run_max,
        jitter_xy=0.25,
        jitter_heading=0.05,
        align_heading=True
    )

    # water barriers
    L_bar = spec.sizes_lwh_m["water_barrier"][0]
    bars = _place_runs_on_lines(
        "water_barrier",
        N_target=alloc["water_barrier"],
        base_spacing=max(spec.spacing_m["water_barrier"], L_bar),
        length_m=L_bar,
        run_min=spec.barrier_run_min,
        run_max=spec.barrier_run_max,
        jitter_xy=0.08,
        jitter_heading=0.02,
        align_heading=True
    )

    # hydrants (kept simple; not tied to support lines; spacing handled implicitly by random sampling region)
    hydrants: List[Tuple[np.ndarray, float]] = []
    N_h = alloc.get("hydrant", 0)
    if N_h > 0:
        corridor = None
        if ego_union is not None:
            corridor = ego_union.buffer(spec.ego_max_dist_m)
        elif ego_route_xyz is not None:
            R = _to_xyz(ego_route_xyz)
            if len(R) >= 2:
                corridor = LineString(R[:, :2]).buffer(spec.ego_max_dist_m)

        outside_region = None
        if outside_region is None:
            lines=[]
            for u, v, data in lane_graph.edges(data=True):
                geom = data.get('geom')
                xy = np.asarray(geom)[:, :2]
                lines.append(LineString(xy))
            if lines:
                geom = unary_union(lines).buffer(10.0)
                minx, miny, maxx, maxy = geom.bounds
                big = Polygon([(minx-200,miny-200),(minx-200,maxy+200),(maxx+200,maxy+200),(maxx+200,miny-200)])
                outside_region = big.difference(geom.buffer(2.0))

        if outside_region and (not outside_region.is_empty):
            minx, miny, maxx, maxy = outside_region.bounds
            tries = 0
            while len(hydrants) < N_h and tries < N_h * 80:
                tries += 1
                x = float(rng.uniform(minx, maxx))
                y = float(rng.uniform(miny, maxy))
                p = Point(x, y)
                if not outside_region.contains(p):
                    continue
                if corridor is not None and (not corridor.contains(p)):
                    continue
                hd = float(rng.uniform(-math.pi, math.pi))
                hydrants.append((np.array([x, y, 0.0], float), hd))


    # ---- assemble output ----
    out: List[Dict[str, Any]] = []
    counter = 0

    def _emit(objs, cls):
        nonlocal counter, out
        LWH = spec.sizes_lwh_m.get(cls, (1.0,1.0,1.0))
        for (xyz, hd) in objs:
            out.append(dict(
                id=f"{cls}_{counter:06d}",
                cls=cls,
                size_lwh_m=LWH,
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                heading_rad=float(hd),
                meta={}
            ))
            counter += 1

    _emit(cones, "cone")
    _emit(bars, "water_barrier")
    _emit(hydrants, "hydrant")

    print("cone:",len(cones),"water_barrier",len(bars),"hydrant",len(hydrants))

    return out


# ----------------- quick plot helper (optional) -----------------

def plot_static_on_map(ax, static_objs: List[Dict[str,Any]], color_map: Optional[Dict[str,str]] = None):
    import math as _math
    from matplotlib.patches import Rectangle, Circle

    if color_map is None:
        color_map = {"cone":"orange", "water_barrier":"tab:blue", "hydrant":"red"}

    for o in static_objs:
        x, y = o["x"], o["y"]
        hd = o["heading_rad"]
        cls = o["cls"]
        c = color_map.get(cls, "k")

        L, W, H = o.get("size_lwh_m", (1.0, 1.0, 1.0))  # (length, width, height)

        if cls == "cone":
            # draw as circle footprint
            circ = Circle((x, y), radius=W*0.5, color=c, alpha=0.6)
            ax.add_patch(circ)

        elif cls == "water_barrier":
            # draw as oriented rectangle
            dx, dy = _math.cos(hd), _math.sin(hd)
            # corner of rectangle (centered on (x,y))
            corner_x = x - 0.5*L*dx + 0.5*W*dy
            corner_y = y - 0.5*L*dy - 0.5*W*dx
            rect = Rectangle(
                (corner_x, corner_y),
                L, W,
                angle=_math.degrees(hd),
                facecolor=c, alpha=0.5, edgecolor="k"
            )
            ax.add_patch(rect)

        elif cls == "hydrant":
            circ = Circle((x, y), radius=max(W, L)*0.5, color=c, alpha=0.8)
            ax.add_patch(circ)

        else:
            ax.plot(x, y, marker="o", ms=4, color=c, alpha=0.9)

        # heading arrow (scaled with length)
        ax.arrow(x, y, 0.5*L*_math.cos(hd), 0.5*L*_math.sin(hd),
                 head_width=0.3*W, head_length=0.4*L, color=c, alpha=0.6)

    ax.set_aspect("equal", "box")
