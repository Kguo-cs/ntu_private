# static_elements_from_raw_v4.py
# Adds per-line spacing occupancy so placements never overlap prior objects' spacings.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import Counter
from .scene_generator import _interp_xyz_at_s,_heading_at_s_dir

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
    density01: float = 0.1
    ratios: Dict[str, float] = None                  # {"cone":1, "water_barrier":2, "hydrant":1}
    sizes_lwh_m: Dict[str, Tuple[float,float,float]] = None
    seed: int = 1234

    # belong-to filtering around ego edges
    belong_tol_m: float = 10.0
    ego_max_dist_m: float = 40.0  #hydra generate within 40 m to ego route
    drive_arae_m : float = 5.0  #hydra generate far from 10 m to all lines


    # resample params
    ds_resample_m: float = 1.0
    smooth: Optional[Tuple[int,int]] = (9, 2)

    # placement attempts
    max_run_shifts: int = 25       # how many s0 shifts we try when conflicts
    max_point_jitter_trials: int = 3  # after s picked, try small s jitter if still conflict


    jitter_xy = 0.1
    jitter_heading = 0.02


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

    total_cap= support_len / spacing*spec.density01/(spec.ratios["cone"]+spec.ratios["water_barrier"])

    alloc={}

    alloc["cone"] = int(total_cap*spec.ratios["cone"])
    alloc["water_barrier"] =int(total_cap*spec.ratios["water_barrier"])
    alloc["hydrant"] = int(total_cap*spec.ratios["hydrant"])

    lane_avail_lengths = {}
    lane_avail_s = {}
    lane_s = {}

    for id,lane in enumerate(support_lines):

        lane_length = _arclen2d(lane[:, :2])
        lane_avail_lengths[id] = lane_length[-1]
        lane_avail_s[id] = [np.array([0, lane_length[-1]])]

        lane_s[id] = lane_length



    def sample_objects(lanes,lane_s,lane_avail_lengths,lane_avail_s,spacing,object_min, object_max):
        object_list=[]
        cone_number=0
        water_number=0

        while True:
            if cone_number<water_number*(spec.ratios["cone"]/spec.ratios["water_barrier"]):
                type="cone"
            else:
                type="water_barrier"

            weights = np.array(list(lane_avail_lengths.values()), dtype=float)-spacing

            weights=np.clip(weights,a_min=0,a_max=1000)

            if np.sum(weights)==0:
                return object_list,cone_number,water_number

            # sample one lane
            sampled_lane = np.random.choice(range(len(lanes)), p=weights / weights.sum() )

            object_number = int(rng.integers(object_min, object_max + 1))

            object_spacing=object_number*spacing

            new_gaps=[]

            for sampled_lane_s in lane_avail_s[sampled_lane]:
                new_gap=sampled_lane_s[1]-sampled_lane_s[0]-object_spacing
                new_gaps.append(new_gap)

            weights=np.array(new_gaps)

            weights=np.clip(weights,a_min=0,a_max=1000)

            if np.sum(weights)==0:
                continue

            sampled_seg=np.random.choice(len(new_gaps),p=weights/np.sum(weights))

            sampled_pos=np.random.rand()*new_gaps[sampled_seg]

            lane_avail_lengths[sampled_lane]=lane_avail_lengths[sampled_lane]-object_spacing

            original_seg=lane_avail_s[sampled_lane][sampled_seg]

            new_seg1=np.array([original_seg[0],original_seg[0]+sampled_pos])

            new_seg2=np.array([original_seg[0]+sampled_pos+object_spacing,original_seg[1]])

            del lane_avail_s[sampled_lane][sampled_seg]

            lane_avail_s[sampled_lane].append(new_seg1)
            lane_avail_s[sampled_lane].append(new_seg2)

            u=original_seg[0]+sampled_pos

            L = lanes[sampled_lane]
            sL = lane_s[sampled_lane]

            for i in range(object_number):
                #u_try=u+i*spacing

                xyz = _interp_xyz_at_s(L, sL, u+i*spacing)
                heading = _heading_at_s_dir(L, sL, u, dir_sign=+1)

                # P=L
                # Ltot=sL[-1]
                #
                # x = np.interp(u_try, sL, P[:, 0])
                # y = np.interp(u_try, sL, P[:, 1])
                # z = np.interp(u_try, sL, P[:, 2])
                #
                #
                # if len(P) > 1:
                #     idxp = int(np.clip(round(u_try / max(1e-6, Ltot) * (len(P) - 1)), 0, len(P) - 1))
                # else:
                #     idxp = 0
                # hd = _tangent_heading_at(P, idxp)
                # position/heading jitter
                jpar = float(rng.normal(0.0, spec.jitter_xy * 0.15))
                jlat = float(rng.normal(0.0, spec.jitter_xy))
                dx = jpar * math.cos(heading) - jlat * math.sin(heading)
                dy = jpar * math.sin(heading) + jlat * math.cos(heading)
                xyz[0]=xyz[0]+dx
                xyz[1]=xyz[1]+dy
                heading = float(heading + rng.normal(0.0, spec.jitter_heading))

                agent=dict(
                        id=f"{type}_{len(object_list):06d}",
                        cls=type,
                        size_lwh_m=spec.sizes_lwh_m[type],
                        x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                        heading_rad=float(heading),
                    )
                object_list.append(agent)

            if type=="cone":
                cone_number=cone_number+object_number
            else:
                water_number=water_number+object_number

            if cone_number>alloc["cone"] or water_number>alloc["water_barrier"]:
                return object_list,cone_number,water_number

        return object_list,cone_number,water_number

    spacing_m=max(5.0,spec.sizes_lwh_m["water_barrier"][0] )#{"cone":5.0, "water_barrier":5.0}

    # cones
    out,cone_number,water_number=sample_objects(support_lines,lane_s, lane_avail_lengths, lane_avail_s, spacing_m, 5, 50)

    alloc["hydrant"] =int((cone_number+water_number)/(spec.ratios["cone"]+spec.ratios["water_barrier"])*spec.ratios["hydrant"])

    # hydrants (kept simple; not tied to support lines; spacing handled implicitly by random sampling region)
    hydrants: List[Tuple[np.ndarray, float]] = []
    N_h = alloc.get("hydrant", 0)
    if N_h > 0:
        corridor = None
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
                geom = unary_union(lines).buffer(spec.drive_arae_m)
                minx, miny, maxx, maxy = geom.bounds
                big = Polygon([(minx-200,miny-200),(minx-200,maxy+200),(maxx+200,maxy+200),(maxx+200,miny-200)])
                outside_region = big.difference(geom)#.buffer(2.0)

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
            ))
            counter += 1

    _emit(hydrants, "hydrant")

    counts = Counter(a["cls"] for a in out)
    print("Agent class counts:", dict(counts))

    # print("Static class counts: {'cones':", len(cones), 'water_barrier:', len(bars),"hydrant:", len(hydrants), "}")

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
