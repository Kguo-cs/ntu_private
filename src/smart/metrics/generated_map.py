"""Split decoder-produced lane graphs without substituting reference maps."""
from collections.abc import Mapping

import numpy as np

if __package__:
    from .metric_core import as_numpy
else:
    from metric_core import as_numpy


def split_generated_maps(payload, num_scenes):
    if not isinstance(payload, Mapping) or payload.get("coordinate_frame") != "sd_local":
        raise ValueError("generated_map must contain physical SD-local lane coordinates")
    required = ("road_points", "road_connection_types", "edge_index_lane_to_lane", "batch", "lg_type")
    missing = set(required) - set(payload)
    if missing:
        raise ValueError(f"Incomplete generated_map: missing {sorted(missing)}")
    points, connections, edges, batch, kinds = (as_numpy(payload[key]) for key in required)
    if points.ndim != 3 or points.shape[1:] != (20, 2) or not np.isfinite(points).all():
        raise ValueError("Generated road_points must be finite [L,20,2] physical coordinates")
    if batch.shape != (len(points),) or not np.isfinite(batch).all() or not np.equal(batch, np.floor(batch)).all():
        raise ValueError("Generated lane batch must be an integer [L] array")
    if not np.array_equal(np.unique(batch), np.arange(num_scenes)):
        raise ValueError("Each generated scene must have lanes and contiguous batch IDs")
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
        raise ValueError("Generated lane edges must be integer [2,E] indices")
    edges, batch = edges.astype(np.int64), batch.astype(np.int64)
    if (edges < 0).any() or (edges >= len(points)).any():
        raise ValueError("Generated lane edge index is out of bounds")
    if not np.array_equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError("Generated lane edges must not cross scene boundaries")
    if (connections.shape != (edges.shape[1], 6) or not np.isin(connections, (0, 1)).all()
            or not (connections.sum(-1) == 1).all()):
        raise ValueError("Generated lane connection types must be one-hot [E,6] aligned with edge columns")
    if kinds.size != num_scenes or not (kinds.reshape(-1) == 0).all():
        raise ValueError("Initial-scene metrics require non-partitioned generated graphs (lg_type=0)")
    records = []
    for scene_id in range(num_scenes):
        rows = np.flatnonzero(batch == scene_id)
        edge_mask = batch[edges[0]] == scene_id
        # Use an explicit mapping, including when lane rows are interleaved.
        inverse = np.full(len(points), -1, dtype=np.int64)
        inverse[rows] = np.arange(len(rows))
        records.append({
            "lg_type": 0, "num_lanes": len(rows), "road_points": points[rows],
            "road_connection_types": connections[edge_mask],
            "edge_index_lane_to_lane": inverse[edges[:, edge_mask]],
        })
    return records
