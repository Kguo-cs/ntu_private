# static_elements_from_raw.py
# Generate cones, water barriers and hydrants using boundary_dict + lane_dict (lane_line as dividers).
# Requires: numpy, shapely, (optional) scipy for smoothing

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


# ----------------- data types -----------------

@dataclass
class StaticSpec:
    density01: float = 0.5                          # overall density [0..1]
    ratios: Dict[str, float] = None                  # e.g., {"cone":1, "water_barrier":2, "hydrant":1}
    sizes_lwh_m: Dict[str, Tuple[float,float,float]] = None
    seed: int = 1234
    # relevance to ego planning
    ego_max_dist_m: float = 120.0                   # limit to ego-route neighborhood
    # sampling spacing baselines (max capacity = length/spacing)
    spacing_m: Dict[str, float] = None              # e.g., cone: 5m, water_barrier: 8m, hydrant: 25m
    # resample params
    ds_resample_m: float = 1.0
    smooth: Optional[Tuple[int,int]] = (9, 2)       # (window, poly) for Savitzky-Golay; set None to disable

    def __post_init__(self):
        if self.ratios is None:
            self.ratios = {"cone":1, "water_barrier":2, "hydrant":1}
        if self.sizes_lwh_m is None:
            self.sizes_lwh_m = {
                "cone": (0.3, 0.3, 0.7),
                "water_barrier": (1.6, 0.5, 0.9),
                "hydrant": (0.6, 0.6, 0.9),
            }
        if self.spacing_m is None:
            self.spacing_m = {"cone":5.0, "water_barrier":8.0, "hydrant":25.0}


# ----------------- main API -----------------

def generate_static_elements_from_raw(
    *,
    boundary_dict: Dict[int, np.ndarray],
    lane_dict: Dict[int, np.ndarray],  # from your parsing: all lane_line (solid/dot) are treated as lane dividers
    ego_route_xyz: Optional[np.ndarray] = None,
    drivable_area: Optional[List[Polygon | MultiPolygon]] = None,
    spec: StaticSpec = StaticSpec(),
) -> List[Dict[str, Any]]:
    """
    Returns a list of static objects:
      {id, cls, size_lwh_m, x,y,z, heading_rad, source, meta}
    - cones / water_barrier: along lane_dict (lane_line -> used as dividers) + boundary_dict
    - hydrant: outside drivable area (if provided), else outside a buffered road ribbon
    Deterministic under the given seed.
    """
    rng = np.random.default_rng(spec.seed)

    # ---- 1) Collect candidate lines: boundaries + lane dividers (from lane_dict) ----
    boundary_lines: List[np.ndarray] = []
    for B in (boundary_dict or {}).values():
        if B is None: continue
        boundary_lines.append(_resample_polyline(B, ds=spec.ds_resample_m, smooth=spec.smooth))

    divider_lines: List[np.ndarray] = []
    for L in (lane_dict or {}).values():
        if L is None: continue
        divider_lines.append(_resample_polyline(L, ds=spec.ds_resample_m, smooth=spec.smooth))

    # ---- 2) Relevance filter to ego route ----
    def _keep_near_ego(lines: List[np.ndarray]) -> List[np.ndarray]:
        if ego_route_xyz is None: return lines
        route = _resample_polyline(ego_route_xyz, ds=1.0)
        ls_route = LineString(route[:, :2])
        out = []
        for L in lines:
            ls = LineString(L[:, :2])
            if ls_route.distance(ls) <= spec.ego_max_dist_m:
                out.append(L)
        return out

    boundary_lines = _keep_near_ego(boundary_lines)
    divider_lines  = _keep_near_ego(divider_lines)

    # ---- 3) Capacity & allocation ----
    # cones / water_barrier share the same candidate supports (boundary + dividers)
    support_lines = divider_lines + boundary_lines
    support_len = _polyline_capacity_len(support_lines)

    cone_cap = int(math.floor(support_len / max(0.1, spec.spacing_m["cone"])))
    wb_cap   = int(math.floor(support_len / max(0.1, spec.spacing_m["water_barrier"])))
    hyd_cap  = 2000  # rough cap; refined later by region/ego distance

    total_max = cone_cap + wb_cap + hyd_cap
    if total_max <= 0:
        return []

    N_total = max(0, int(round(np.clip(spec.density01, 0.0, 1.0) * total_max)))

    keys = list(spec.ratios.keys())
    vals = np.array([max(0.0, float(spec.ratios[k])) for k in keys], float)
    if vals.sum() <= 0: vals = np.ones_like(vals)
    probs = vals / vals.sum()

    alloc = {k: int(math.floor(N_total * p)) for k, p in zip(keys, probs)}
    rem = N_total - sum(alloc.values()); i = 0
    while rem > 0 and keys:
        k = keys[i % len(keys)]; alloc[k] += 1; rem -= 1; i += 1

    alloc["cone"] = min(alloc.get("cone", 0), cone_cap)
    alloc["water_barrier"] = min(alloc.get("water_barrier", 0), wb_cap)
    alloc["hydrant"] = max(0, N_total - alloc["cone"] - alloc["water_barrier"])

    # ---- 4) Sampling helpers ----
    def _sample_points_on_lines(lines: List[np.ndarray], N: int) -> List[Tuple[np.ndarray, float, Dict]]:
        out = []
        if N <= 0 or not lines:
            return out
        lens = np.array([float(_arclen2d(L[:, :2])[-1]) for L in lines], float)
        if lens.sum() <= 1e-6:
            return out
        probs = lens / lens.sum()
        # choose which line per sample, then sample uniform on that line-length
        idxs = rng.choice(np.arange(len(lines)), size=N, p=probs, replace=True)
        for idx in idxs:
            P = lines[idx]
            s = _arclen2d(P[:, :2]); L = float(s[-1])
            if L <= 1e-6:
                k = 0; Q = P[k]; hd = 0.0
            else:
                u = float(rng.uniform(0.0, L))
                x = np.interp(u, s, P[:,0]); y = np.interp(u, s, P[:,1]); z = np.interp(u, s, P[:,2])
                Q = np.array([x, y, z], float)
                k = int(np.clip(round(u / max(1e-6, L) * (len(P)-1)), 0, len(P)-1))
                hd = _tangent_heading_at(P, k)
            out.append((Q, hd, {"line_index": int(idx)}))
        return out

    N_cone = alloc.get("cone", 0)
    N_wb   = alloc.get("water_barrier", 0)
    pts_cone = _sample_points_on_lines(support_lines, N_cone)
    pts_wb   = _sample_points_on_lines(support_lines, N_wb)

    # hydrants: sample outside drivable area (if provided), else outside a buffered road ribbon
    hydrants: List[Tuple[np.ndarray, float, Dict]] = []
    N_h = alloc.get("hydrant", 0)
    if N_h > 0:
        outside_region = None
        if drivable_area:
            union_poly = unary_union(drivable_area)
            # build big bbox around supports
            all_pts = []
            for L in support_lines:
                all_pts.append(_to_xyz(L))
            if all_pts:
                P = np.vstack(all_pts)
                minx, miny = float(P[:,0].min()-200), float(P[:,1].min()-200)
                maxx, maxy = float(P[:,0].max()+200), float(P[:,1].max()+200)
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
            route_ls = None
            if ego_route_xyz is not None:
                route_ls = LineString(_resample_polyline(ego_route_xyz, ds=1.0)[:, :2])
            minx, miny, maxx, maxy = outside_region.bounds
            tries = 0
            while len(hydrants) < N_h and tries < N_h * 60:
                tries += 1
                x = float(rng.uniform(minx, maxx))
                y = float(rng.uniform(miny, maxy))
                p = Point(x, y)
                if not outside_region.contains(p):
                    continue
                if route_ls is not None and route_ls.distance(p) > spec.ego_max_dist_m:
                    continue
                hd = float(rng.uniform(-math.pi, math.pi))
                hydrants.append((np.array([x, y, 0.0], float), hd, {}))

    # ---- 5) Assemble deterministic output ----
    out: List[Dict[str, Any]] = []
    counter = 0

    def _emit(objs, cls):
        nonlocal counter, out
        LWH = spec.sizes_lwh_m.get(cls, (1.0,1.0,1.0))
        for (xyz, hd, meta) in objs:
            out.append(dict(
                id=f"{cls}_{counter:06d}",
                cls=cls,
                size_lwh_m=LWH,
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                heading_rad=float(hd),
                source=meta.get("line_index", None),
                meta=meta
            ))
            counter += 1

    _emit(pts_cone, "cone")
    _emit(pts_wb, "water_barrier")
    _emit(hydrants, "hydrant")

    return out


# ----------------- quick plot helper (optional) -----------------

def plot_static_on_map(ax, static_objs: List[Dict[str,Any]], color_map: Optional[Dict[str,str]] = None):
    import matplotlib.pyplot as plt
    if color_map is None:
        color_map = {"cone":"orange", "water_barrier":"tab:blue", "hydrant":"red"}
    for o in static_objs:
        x,y = o["x"], o["y"]
        c = color_map.get(o["cls"], "k")
        ax.plot(x, y, marker="o", ms=4, color=c, alpha=0.9)
        # small heading arrow
        hd = o["heading_rad"]
        ax.arrow(x, y, 1.0*math.cos(hd), 1.0*math.sin(hd),
                 head_width=0.35, head_length=0.55, color=c, alpha=0.6)
