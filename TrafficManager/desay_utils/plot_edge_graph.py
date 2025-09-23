import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def plot_edge_graph(edge_result, show_nodes=True, show_labels=False, figsize=(10,10)):
    """
    Plot the edge graph built by build_edge_graph_from_lane_graph_topo.
    - edge_result: BuildResult
    """
    G = edge_result.edge_graph
    node_xyz = edge_result.node_xyz

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for u, v, data in G.edges(data=True):
        geom = data.get('geom')
        if geom is not None:
            xy = np.asarray(geom)[:, :2]
            ax.plot(xy[:,0], xy[:,1], color='blue', linewidth=2, alpha=0.7, zorder=1)
        else:
            # fallback: straight line between node positions
            pu = node_xyz.get(u)
            pv = node_xyz.get(v)
            if pu is not None and pv is not None:
                ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color='blue', linestyle='--', alpha=0.5)

        if show_labels:
            mid = len(geom)//2 if geom is not None else None
            if mid:
                ax.text(geom[mid,0], geom[mid,1], data.get('id',''), fontsize=8, color='purple')

    # Draw nodes
    if show_nodes:
        for nid, p in node_xyz.items():
            if p is None:
                continue
            ax.scatter(p[0], p[1], color='black', s=20, zorder=2)
            if show_labels:
                ax.text(p[0], p[1], nid, fontsize=7, color='red')

    ax.set_aspect('equal')
    ax.set_title("Edge Graph (SUMO-like)")
    plt.show()
