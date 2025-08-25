import math
from typing import List, Tuple
import numpy as np
import networkx as nx

Point = Tuple[float, float]
Polyline = np.ndarray  # shape [N, 2]

# ---------- geometry helpers ----------
def seg_len(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))

def point_seg_proj(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project point p onto segment ab. Returns (proj_point, t_clamped, dist)."""
    ab = b - a
    l2 = float(ab @ ab)
    if l2 == 0.0:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t = float(((p - a) @ ab) / l2)
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    return proj, t_clamped, float(np.linalg.norm(p - proj))

def nearest_segment(p: np.ndarray, polylines: List[Polyline]):
    """Return (poly_idx, seg_idx, proj_point, dist)."""
    best = (None, None, None, float("inf"))
    for i, line in enumerate(polylines):
        for j in range(len(line) - 1):
            proj, t, d = point_seg_proj(p, line[j], line[j+1])
            if d < best[3]:
                best = (i, j, proj, d)
    return best

# ---------- graph construction ----------
def build_lane_graph(
    centerlines: List[Polyline],
    connect_tol: float = 0.5,
) -> nx.Graph:
    """
    Creates an undirected weighted graph.
    Nodes are (x,y) tuples at polyline vertices; edges connect consecutive vertices.
    Also connects endpoints from different polylines if they are within connect_tol.
    """
    G = nx.Graph()

    # add edges for each polyline’s consecutive vertices
    for line in centerlines:
        line = np.asarray(line, dtype=float)
        for k in range(len(line)):
            G.add_node(tuple(line[k]))
        for a, b in zip(line[:-1], line[1:]):
            a_t, b_t = tuple(a), tuple(b)
            w = seg_len(a, b)
            if w > 0:
                G.add_edge(a_t, b_t, weight=w)

    # connect close endpoints across polylines (intersections/joins)
    endpoints = []
    for line in centerlines:
        line = np.asarray(line, dtype=float)
        endpoints.extend([tuple(line[0]), tuple(line[-1])])

    ep = np.array(endpoints)
    # simple O(n^2) join (fast enough for small/mid graphs)
    for i in range(len(ep)):
        for j in range(i + 1, len(ep)):
            if np.linalg.norm(ep[i] - ep[j]) <= connect_tol:
                a, b = tuple(ep[i]), tuple(ep[j])
                if a != b:
                    G.add_edge(a, b, weight=seg_len(np.array(a), np.array(b)))
    return G

def add_snapped_point(G: nx.Graph, p: Point, polylines: List[Polyline]) -> Point:
    """
    Snap an external point to the nearest segment by inserting a vertex into the graph
    (splitting the original edge if projection falls in the middle).
    Returns the snapped coordinate (x,y).
    """
    p = np.array(p, dtype=float)
    poly_i, seg_j, proj, _ = nearest_segment(p, polylines)
    proj_t = tuple(proj)
    a = tuple(polylines[poly_i][seg_j])
    b = tuple(polylines[poly_i][seg_j + 1])

    # If projection equals an existing node (within tiny tol), just connect to it
    if np.linalg.norm(proj - np.array(a)) < 1e-9:
        return a
    if np.linalg.norm(proj - np.array(b)) < 1e-9:
        return b

    # If the edge (a,b) exists, split it into (a,proj) and (proj,b)
    # Remove the old edge and insert the projected node.
    if G.has_edge(a, b) or G.has_edge(b, a):
        w_ab = seg_len(np.array(a), np.array(b))
        # remove original edge
        if G.has_edge(a, b):
            G.remove_edge(a, b)
        elif G.has_edge(b, a):
            G.remove_edge(b, a)

        # add projected node and new split edges
        G.add_node(proj_t)
        G.add_edge(a, proj_t, weight=seg_len(np.array(a), proj))
        G.add_edge(proj_t, b, weight=seg_len(proj, np.array(b)))
    else:
        # If the base edge isn't in G (rare), just connect proj to nearest of a/b
        G.add_node(proj_t)
        da = seg_len(proj, np.array(a))
        db = seg_len(proj, np.array(b))
        G.add_edge(proj_t, a, weight=da)
        G.add_edge(proj_t, b, weight=db)

    return proj_t

# ---------- routing ----------
def astar_route(G: nx.Graph, start_xy: Point, goal_xy: Point) -> List[Point]:
    def h(u, v):
        ax, ay = u
        bx, by = v
        return math.hypot(ax - bx, ay - by)  # straight-line heuristic
    return nx.astar_path(G, start_xy, goal_xy, heuristic=h, weight="weight")

def route_from_centerlines(
    centerlines: List[Polyline],
    start: Point,
    goal: Point,
    connect_tol: float = 0.5,
) -> List[Point]:
    """
    centerlines: list of Nx2 arrays (each a lane center polyline).
    start, goal: (x, y) in the same coordinates as centerlines.
    connect_tol: distance to auto-connect endpoints from different lines.
    """
    # 1) Build graph
    G = build_lane_graph(centerlines, connect_tol=connect_tol)

    # 2) Snap start/goal into the graph
    s_node = add_snapped_point(G, start, centerlines)
    g_node = add_snapped_point(G, goal, centerlines)

    # 3) Run A*
    path_nodes = astar_route(G, s_node, g_node)

    # 4) Return as list of (x,y)
    return path_nodes

# ---------- example ----------
# if __name__ == "__main__":
#     # toy network: a T-junction
#     line1 = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)     # horizontal
#     line2 = np.array([[10, 0], [10, 10]], dtype=float)            # vertical up
#     centerlines = [line1, line2]
#
#     start = (2, -0.2)
#     goal = (10, 9.5)
#
#     path = route_from_centerlines(centerlines, start, goal, connect_tol=0.25)
#     print("Route has", len(path), "points")
#     for pt in path:
#         print(pt)
