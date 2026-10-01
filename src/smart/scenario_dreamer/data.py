"""Adapt existing SMART samples to the bundled Scenario Dreamer AE graph."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from .core.data_container import ScenarioDreamerData
from .core.data_helpers import reorder_indices
from .core.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite


def attach_model_map(data, scene):
    """Preserve exact SD lanes as ordinary, batchable PyG nodes/edges."""
    kind = int(scene["lg_type"])
    graph = scene["graphs"]["regular" if kind == 0 else "partitioned"]
    lanes = torch.as_tensor(graph["road_points"], dtype=torch.float32)
    edges = get_edge_index_complete_graph(len(lanes))
    # Adjacency is [destination, source]; PyG edges are [source, destination].
    # Upstream chooses self first, then predecessor/successor/left/right.
    relations = torch.zeros((len(lanes), len(lanes)), dtype=torch.long)
    for label, key in enumerate(("pre_adj", "suc_adj", "left_adj", "right_adj"), 1):
        mask = torch.as_tensor(graph[key]).bool().T & (relations == 0)
        relations[mask] = label
    relations.fill_diagonal_(5)
    data["sd_lane"] = {"x": lanes, "num_nodes": len(lanes)}
    data[("sd_lane", "to", "sd_lane")] = {
        "edge_index": edges, "type": relations[edges[0], edges[1]],
    }
    data["sd_lg_type"] = kind


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * x[..., 0] - s * x[..., 1],
                        s * x[..., 0] + c * x[..., 1]), -1)


def _token_lanes(tokens, scene_id, center, angle, max_lanes):
    """Fallback for old SMART caches: geometric segments, no inferred topology.

    This is explicitly a token-map adaptation, not the official compact-lane
    representation. Exact reconstructed SD samples use their saved lanes above.
    """
    mask = (tokens["batch"] == scene_id) & torch.isin(
        tokens["type"], tokens["type"].new_tensor([0, 1, 3]))
    if "traj_pos_local" not in tokens:
        raise KeyError("SD decoder needs traj_pos_local in tokenized_map or saved sd_lane geometry")
    local = tokens["traj_pos_local"][mask]
    local = torch.cat((local.new_zeros((len(local), 1, 2)), local[..., :2]), 1)
    world = rotate(local, tokens["orientation"][mask, None]) + tokens["position"][mask, None, :2]
    lanes = rotate(world - center, angle)
    keep = (lanes.abs() <= 32).all(-1).any(-1)
    lanes = lanes[keep]
    if len(lanes) == 0:
        raise ValueError("No map segments intersect the Scenario Dreamer 64m field of view")
    order = lanes.norm(dim=-1).amin(-1).argsort()[:max_lanes]
    lanes = lanes[order]
    # Resample by arc length (not point index).
    resampled = []
    for lane in lanes:
        arc = torch.cat((lane.new_zeros(1), (lane[1:] - lane[:-1]).norm(dim=-1).cumsum(0)))
        query = torch.linspace(0, 1, 20, device=lane.device) * arc[-1]
        right = torch.searchsorted(arc, query, right=True).clamp(1, len(arc) - 1)
        alpha = (query - arc[right - 1]) / (arc[right] - arc[right - 1]).clamp_min(1e-8)
        resampled.append(lane[right - 1] + alpha[:, None] * (lane[right] - lane[right - 1]))
    lanes = torch.stack(resampled)
    relations = torch.zeros((len(lanes), len(lanes)), device=lanes.device, dtype=torch.long)
    relations.fill_diagonal_(5)
    return lanes, relations


def build_graph(agent, tokens, cfg, *, map_source="auto"):
    """Return AE batch, SD-row -> SMART-row mapping, and reference transforms."""
    device = agent["initial_pos"].device
    batch = agent["batch"]
    num_graphs = int(agent["num_graphs"])
    exact = agent.get("sd_map")
    if map_source == "exact" and exact is None:
        raise ValueError("map_source=exact requires official AE samples or rebuilt samples with saved scenario_dreamer graphs")
    if map_source not in ("auto", "exact", "tokens"):
        raise ValueError("map_source must be auto, exact or tokens")
    use_exact = exact is not None and map_source != "tokens"
    cached = agent.get("sd_cached_posterior")
    if cached is not None and not use_exact:
        raise ValueError("Cached AE posteriors require the exact preprocessed map")
    graphs, row_ids, centers, angles = [], [], [], []
    for b in range(num_graphs):
        rows = torch.where(batch == b)[0]
        ego_rows = rows[agent["ego_mask"][rows]]
        if len(ego_rows) != 1 or len(rows) > cfg.max_num_agents:
            raise ValueError(f"Scene {b} needs one ego and at most {cfg.max_num_agents} agents; got {len(rows)}")
        # SD positional embeddings reserve index zero for ego; SMART uses ego-last.
        rows = torch.cat((ego_rows, rows[~agent["ego_mask"][rows]]))
        if use_exact:
            kind = int(exact["lg_type"].reshape(-1)[b])
            if kind not in (0, 1):
                raise ValueError(f"Unsupported Scenario Dreamer lg_type={kind}")
            center = exact["center"].reshape(num_graphs, 2)[b].to(device)
            angle = exact["angle"].reshape(-1)[b].to(device)
            lane_rows = torch.where(exact["batch"] == b)[0]
            lanes = exact["lanes"][lane_rows].to(device)
            if len(lane_rows) == 0:
                raise ValueError(f"Scene {b} has no saved Scenario Dreamer lanes")
            offset = int(lane_rows[0])
            edges = exact["edges"]
            edge_mask = exact["batch"][edges[0]] == b
            local_edges = edges[:, edge_mask] - offset
            relations = torch.zeros((len(lanes), len(lanes)), device=device, dtype=torch.long)
            relations[local_edges[0], local_edges[1]] = exact["types"][edge_mask].long()
        else:
            kind = 0
            center = agent["initial_pos"][ego_rows[0]]
            angle = torch.pi / 2 - agent["initial_heading"][ego_rows[0]]
            lanes, relations = _token_lanes(tokens, b, center, angle, cfg.max_num_lanes)
        if not 1 <= len(lanes) <= cfg.max_num_lanes:
            raise ValueError(f"Scene {b} has invalid lane count {len(lanes)}")
        # Float64 saved transforms are used before casting the normalized AE input.
        xy = rotate(agent["initial_pos"][rows].to(center.dtype) - center, angle)
        heading = agent["initial_heading"][rows] + angle
        speed = agent["local_vel"][rows].norm(dim=-1)
        shape = agent["shape"][rows, :2]
        state = torch.cat((xy, speed[:, None], heading.cos()[:, None], heading.sin()[:, None], shape), -1).float()
        if "sd_states" in agent:
            state = agent["sd_states"][rows].clone()
        state[:, :2] /= cfg.fov / 2
        for index, low, high in ((2, cfg.min_speed, cfg.max_speed), (5, cfg.min_length, cfg.max_length),
                                 (6, cfg.min_width, cfg.max_width)):
            state[:, index] = 2 * (state[:, index] - low) / (high - low) - 1
        normalized_lanes = lanes / (cfg.fov / 2)
        n, l = len(rows), len(lanes)
        # Use exact upstream recursive ordering/tolerance in normalized coordinates.
        a, _, lane, _, _, agent_partition, lane_partition = reorder_indices(
            np.arange(n)[:, None], np.zeros((n, 1)), np.arange(l)[:, None], np.zeros((l, 1)),
            get_edge_index_complete_graph(l).numpy(), state.detach().cpu().numpy(),
            normalized_lanes.detach().cpu().numpy(), kind,
        )
        ai = torch.as_tensor(a[:, 0], device=device, dtype=torch.long)
        li = torch.as_tensor(lane[:, 0], device=device, dtype=torch.long)
        d = ScenarioDreamerData()
        d.num_agents, d.num_lanes, d.lg_type = n, l, kind
        # AE pickles and old SMART caches have no Nocturne label; use the configured category.
        d.map_id = int(agent.get("sd_map_id", 0))
        d["agent"].x = state[ai].float()
        d["agent"].type = torch.nn.functional.one_hot(agent["type"][rows[ai]].long(), 3)
        d["lane"].x = normalized_lanes[li].float()
        if use_exact:
            d["lane"].source_row = lane_rows[li]
        if cached is not None:
            for stat in ("mu", "log_var"):
                d["agent"][f"posterior_{stat}"] = cached[f"agent_{stat}"][rows[ai]]
                d["lane"][f"posterior_{stat}"] = cached[f"lane_{stat}"][lane_rows[li]]
        partitions = {
            "agent": torch.as_tensor(agent_partition, device=device, dtype=torch.bool),
            "lane": torch.as_tensor(lane_partition, device=device, dtype=torch.bool),
        }
        for source, target, edge in (
            ("agent", "agent", get_edge_index_complete_graph(n)),
            ("lane", "lane", get_edge_index_complete_graph(l)),
            ("lane", "agent", get_edge_index_bipartite(l, n)),
        ):
            store = d[source, "to", target]
            store.edge_index = edge.to(device)
            # Official AE attention cannot cross the y=0 partition.
            store.encoder_mask = partitions[source][store.edge_index[0]] == partitions[target][store.edge_index[1]]
        edge = d["lane", "to", "lane"].edge_index
        labels = relations[li][:, li][edge[0], edge[1]]
        d["lane", "to", "lane"].type = torch.nn.functional.one_hot(labels, 6)
        # Conditional lane-count target is defined only for partitioned scenes.
        d.num_lanes_after_origin = int((~partitions["lane"]).sum()) if kind == 1 else 0
        d["lane"].partition_mask = partitions["lane"]
        d["agent"].partition_mask = partitions["agent"]
        graphs.append(d)
        row_ids.append(rows[ai])
        centers.append(center)
        angles.append(angle)
    return Batch.from_data_list(graphs).to(device), torch.cat(row_ids), torch.stack(centers), torch.stack(angles)
