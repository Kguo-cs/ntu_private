# static_elements_from_raw_v3.py
# Only place cones / water barriers / hydrants on lane/boundary that BELONG to ego_edge_ids.
# Requires: numpy, shapely (optional: scipy for smoothing)

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
    belong_tol_m: float = 6.0                        # strict small band to claim “belongs to”
    # (Hydrants still consider a broader ego corridor for relevance)
    ego_max_dist_m: float = 120.0

    # spacing baselines
    spacing_m: Dict[str, float] = None               # cone: ~5m, barrier: ~2m, hydrant: ~25m

    # continuous run controls
    cone_run_min: int = 6
    cone_run_max: int = 15
    barrier_run_min: int = 3
    barrier_run_max: int = 10

    # resample params
    ds_resample_m: float = 1.0
    smooth: Optional[Tuple[int,int]] = (9, 2)       # (window, poly); set None to disable

    def __post_init__(self):
        if self.ratios is None:
            self.ratios = {"cone":1, "water_barrier":2, "hydrant":1}
        if self.sizes_lwh_m is None:
            self.sizes_lwh_m = {
                "cone": (0.5, 0.5, 0.7),
                "water_barrier": (4.8, 0.5, 0.9),
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
    """
    Returns (divider_lines, boundary_lines, ego_union_geom)
    Only lines that BELONG to ego edges (by id if available, else by spatial proximity).
    """
    ego_edge_ids = set(ego_edge_ids or [])
    if not ego_edge_ids:
        return [], [], None

    # Build ego geometry
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

    # 1) Gather boundary IDs explicitly referenced by ego edges
    explicit_boundary_ids: set[int] = set()
    explicit_divider_ids: set[int] = set()
    for _, _, ed in EG.edges(data=True):
        if ed.get("id") not in ego_edge_ids:
            continue
        # boundaries (typical fields from your pipeline)
        for key in ("boundary_ids", "boundary_id_list"):
            if key in ed and isinstance(ed[key], (list, tuple)):
                explicit_boundary_ids.update(int(b) for b in ed[key])
        for key in ("boundary_a_id", "boundary_b_id"):
            if key in ed:
                try: explicit_boundary_ids.add(int(ed[key]))
                except Exception: pass
        # lane dividers
        for key in ("lane_divider_ids", "divider_ids", "lane_line_ids"):
            if key in ed and isinstance(ed[key], (list, tuple)):
                try:
                    explicit_divider_ids.update(int(x) for x in ed[key])
                except Exception:
                    pass

    # 2) Build strict belong band
    belong_band = ego_union.buffer(spec.belong_tol_m)

    # 3) Select boundaries:
    boundary_lines: List[np.ndarray] = []
    if explicit_boundary_ids:
        for bid in explicit_boundary_ids:
            B = boundary_dict.get(bid, None)
            if B is None: continue
            P = _resample_polyline(B, ds=spec.ds_resample_m, smooth=spec.smooth)
            ls = LineString(P[:, :2])
            # must lie inside belong band
            if belong_band.intersects(ls):
                boundary_lines.append(P)
    else:
        # spatial fallback: only boundaries within belong band
        for B in boundary_dict.values():
            if B is None: continue
            P = _resample_polyline(B, ds=spec.ds_resample_m, smooth=spec.smooth)
            ls = LineString(P[:, :2])
            if belong_band.intersects(ls):
                boundary_lines.append(P)

    # 4) Select lane dividers (lane_dict)
    divider_lines: List[np.ndarray] = []
    if explicit_divider_ids:
        for did in explicit_divider_ids:
            L = lane_dict.get(did, None)
            if L is None: continue
            P = _resample_polyline(L, ds=spec.ds_resample_m, smooth=spec.smooth)
            ls = LineString(P[:, :2])
            if belong_band.intersects(ls):
                divider_lines.append(P)
    else:
        # spatial fallback
        for L in lane_dict.values():
            if L is None: continue
            P = _resample_polyline(L, ds=spec.ds_resample_m, smooth=spec.smooth)
            ls = LineString(P[:, :2])
            if belong_band.intersects(ls):
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
    drivable_area: Optional[List[Polygon | MultiPolygon]] = None,
    spec: StaticSpec = StaticSpec(),
) -> List[Dict[str, Any]]:
    """
    Returns a list of static objects with size-aware spacing and continuous placement,
    **restricted to lane dividers / boundaries that BELONG to ego_edge_ids**.
    """
    rng = np.random.default_rng(spec.seed)

    # ---- strict support lines (ego-only) ----
    divider_lines, boundary_lines, ego_union = _collect_support_lines_ego_only(
        EG, ego_edge_ids, boundary_dict, lane_dict, spec
    )
    support_lines = divider_lines + boundary_lines
    if not support_lines:
        return []

    # ---- capacity & allocation ----
    support_len = _polyline_capacity_len(support_lines)

    total_cap= support_len / 5*spec.density01

    alloc={}

    alloc["cone"] = int(total_cap*spec.ratios["cone"]/(spec.ratios["cone"]+spec.ratios["water_barrier"]))
    alloc["water_barrier"] =int(total_cap*spec.ratios["water_barrier"]/(spec.ratios["cone"]+spec.ratios["water_barrier"]))
    alloc["hydrant"] = int(alloc["cone"]*spec.ratios["hydrant"]/spec.ratios["cone"])

    # ---- placement utilities ----
    def _place_runs_on_lines(
        cls: str,
        N_target: int,
        base_spacing: float,
        length_m: float,   # use length dimension for spacing packing
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

        for r in runs:
            idx_line = int(rng.choice(np.arange(len(support_lines)), p=probs))
            P = support_lines[idx_line]
            s = _arclen2d(P[:, :2]); Ltot = float(s[-1])
            if Ltot <= 1e-6:
                continue

            step = max(base_spacing, length_m)  # size-aware spacing
            run_len = (r - 1) * step
            if run_len > Ltot:
                r_fit = max(2, int(Ltot // max(1e-3, step)) + 1)
                if r_fit < 2:
                    continue
                r = r_fit
                run_len = (r - 1) * step

            s0 = float(rng.uniform(0.0, max(1e-6, Ltot - run_len)))

            for k in range(r):
                u = s0 + k * step
                x = np.interp(u, s, P[:, 0]); y = np.interp(u, s, P[:, 1]); z = np.interp(u, s, P[:, 2])
                # heading
                if len(P) > 1:
                    idxp = int(np.clip(round(u / max(1e-6, Ltot) * (len(P) - 1)), 0, len(P) - 1))
                else:
                    idxp = 0
                hd = _tangent_heading_at(P, idxp) if align_heading else float(rng.uniform(-math.pi, math.pi))
                # small jitter
                jpar = float(rng.normal(0.0, jitter_xy * 0.15))
                jlat = float(rng.normal(0.0, jitter_xy))
                dx = jpar * math.cos(hd) - jlat * math.sin(hd)
                dy = jpar * math.sin(hd) + jlat * math.cos(hd)
                xy = np.array([x + dx, y + dy, z], float)
                h = float(hd + rng.normal(0.0, jitter_heading))
                out.append((xy, h))

        return out

    # cones (continuous runs with jitter)
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

    # water barriers (tight chained runs)
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

    # ---- hydrants: off drivable area, but still near ego corridor if provided ----
    hydrants: List[Tuple[np.ndarray, float]] = []
    N_h = alloc.get("hydrant", 0)
    if N_h > 0:
        # ego corridor for relevance
        corridor = None
        if ego_union is not None:
            corridor = ego_union.buffer(spec.ego_max_dist_m)
        elif ego_route_xyz is not None:
            R = _to_xyz(ego_route_xyz)
            if len(R) >= 2:
                corridor = LineString(R[:, :2]).buffer(spec.ego_max_dist_m)

        # off drivable area region
        outside_region = None
        if drivable_area:
            union_poly = unary_union(drivable_area)
            # bounds around support lines
            all_pts = []
            for L in support_lines:
                all_pts.append(_to_xyz(L))
            if all_pts:
                P = np.vstack(all_pts)
                minx, miny = float(P[:,0].min() - 200), float(P[:,1].min() - 200)
                maxx, maxy = float(P[:,0].max() + 200), float(P[:,1].max() + 200)
                big = Polygon([(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)])
                outside_region = big.difference(union_poly.buffer(0.01))
        if outside_region is None:
            # approximate road ribbon by buffering supports
            lines = [LineString(L[:, :2]) for L in support_lines]
            if lines:
                geom = unary_union(lines).buffer(10.0)  # ~10m road band
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

    # ---- assemble deterministic output ----
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

    return out


# ----------------- quick plot helper (optional) -----------------

def plot_static_on_map(ax, static_objs: List[Dict[str,Any]], color_map: Optional[Dict[str,str]] = None):
    import math as _math
    if color_map is None:
        color_map = {"cone":"orange", "water_barrier":"tab:blue", "hydrant":"red"}
    for o in static_objs:
        x,y = o["x"], o["y"]
        c = color_map.get(o["cls"], "k")
        ax.plot(x, y, marker="o", ms=4, color=c, alpha=0.9)
        # heading arrow
        hd = o["heading_rad"]
        ax.arrow(x, y, 1.0*_math.cos(hd), 1.0*_math.sin(hd),
                 head_width=0.35, head_length=0.55, color=c, alpha=0.6)
