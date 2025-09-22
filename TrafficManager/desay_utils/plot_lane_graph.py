# ----------------------------- Simple plot ----------------------------- #
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

def plot_lane_graph(G: nx.DiGraph, show_nodes=True, figsize=(10,10)):
    """Top-down XY plot of lanes and connectors."""
    fig, ax = plt.subplots(figsize=figsize)

    for u, v, data in G.edges(data=True):
        geom = data.get('geom')
        if geom is None:
            continue
        xy = np.asarray(geom)[:, :2]
        kind = data.get('kind', 'lane')
        if kind == 'lane':
            ax.plot(xy[:,0], xy[:,1], color='blue', linewidth=2, alpha=0.75, zorder=1)
        else:
            subtype = data.get('subtype', 'connector')
            if subtype == 'longitudinal':
                ax.plot(xy[:,0], xy[:,1], color='green', linestyle='--', linewidth=1.6, alpha=0.9, zorder=2)
            elif subtype == 'lateral':
                ax.plot(xy[:,0], xy[:,1], color='orange', linestyle='--', linewidth=1.6, alpha=0.9, zorder=2)
            elif subtype in ('turn_left','turn_right'):
                ax.plot(xy[:,0], xy[:,1], color='purple', linestyle='-.', linewidth=1.8, alpha=0.95, zorder=3)
            else:
                ax.plot(xy[:,0], xy[:,1], color='gray', linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)

    if show_nodes:
        for n, data in G.nodes(data=True):
            p = data.get('xyz')
            if p is not None:
                ax.scatter(p[0], p[1], color='black', s=9, zorder=4)

    ax.set_aspect('equal')
    ax.set_title("Lane Graph with Longitudinal, Lateral, and Turning Connectors")
    plt.show()


# # ----------------------------- Example ----------------------------- #
# if __name__ == "__main__":
#     # Usage:
#     # 1) Get your centerlines (dicts or dataclass from your boundary-only builder)
#     # centerlines_xyz = [...]
#     # 2) Convert to inputs:
#     # inputs = centerline_results_to_inputs(centerlines_xyz)
#     # 3) Build graph (now with turning connectors):
#     # G = build_lane_graph_with_connectors(
#     #     inputs,
#     #     forward_only=True, forward_min_m=1.0, forward_lateral_tol_m=3.0,
#     #     enable_lateral=True,
#     #     enable_turning=True, turn_snap_radius_m=18.0,
#     #     turn_min_angle_deg=45.0, turn_max_angle_deg=145.0, allow_uturn=False,
#     # )
#     # 4) Plot:
#     # plot_lane_graph(G)
#     pass
