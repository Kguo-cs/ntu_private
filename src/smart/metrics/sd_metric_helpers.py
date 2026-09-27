"""Self-contained Waymo initial-scene numerical metrics.

Scope is intentionally Waymo / Waymo GT and the seven agent metrics. This is
not a complete replacement for Scenario Dreamer's nuPlan or lane metrics.
See SOURCES.md for copied functions versus compatibility adaptations.
"""
from __future__ import annotations
import numpy as np
import torch
import networkx as nx
from scipy.spatial import distance
if __package__:
    from .sd_lane_helpers import find_lane_groups, find_lane_group_id, resample_polyline, resample_lanes
else:
    from sd_lane_helpers import find_lane_groups, find_lane_group_id, resample_polyline, resample_lanes

NON_PARTITIONED = 0
NUPLAN_VEHICLE = 0
UNIFIED_FORMAT_INDICES = {
    'pos_x': 0, 'pos_y': 1, 'speed': 2, 'cos_heading': 3,
    'sin_heading': 4, 'length': 5, 'width': 6,
}

def jsd(sim, gt, clip_min, clip_max, bin_size):
    """ Computes the Jensen-Shannon divergence (JSD) between generated (sim) and real (gt) distributions."""
    # Clip the simulated and ground truth values
    gt = np.clip(gt, clip_min, clip_max)
    sim = np.clip(sim, clip_min, clip_max)

    # Calculate bin edges based on the specified bin_size
    bin_edges = np.arange(clip_min, clip_max + bin_size, bin_size)

    # Compute the histograms and normalize to get probability distributions
    P = np.histogram(sim, bins=bin_edges)[0] / len(sim)
    Q = np.histogram(gt, bins=bin_edges)[0] / len(gt)

    # Compute Jensen-Shannon divergence and square it
    jsd_value = distance.jensenshannon(P, Q) ** 2  # Square to get the divergence
    return jsd_value

def compute_vehicle_circles(xy_position, heading, length, width):
    """ Computes the centroids and radii of circles around a vehicle based on its position, heading, length, and width."""
    num_circles = 5
    radius = width / 2
    relative_x_positions = np.linspace(-length / 2 + radius, length / 2 - radius, num_circles)

    # Compute the centroids of the circles
    # First, create the (x, y) relative offsets based on heading
    dx = np.cos(heading) * relative_x_positions
    dy = np.sin(heading) * relative_x_positions

    # Add these offsets to the vehicle's position to get the circle centroids
    centroids = np.column_stack((xy_position[0] + dx, xy_position[1] + dy))

    return centroids, np.array([radius]).repeat(num_circles)

def get_onroad_vehicles(vehicles, lanes, tol=1.5):
    """ Filters the vehicles that are on the road based on their distance to the lanes."""
    lanes = lanes.reshape(-1, 2)

    vehicle_road_dist = np.linalg.norm(
        vehicles[:, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1] - lanes[np.newaxis, :, :], axis=-1).min(1)
    offroad_mask = vehicle_road_dist > tol  # following SceneControl
    onroad_vehicles = np.where(~offroad_mask)[0]

    return vehicles[onroad_vehicles]

def get_nearest_dists(vehicles):
    """ Computes the nearest distance between vehicles in the scene."""
    vehicle_vehicle_dist = np.linalg.norm(vehicles[:, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1] - vehicles[
        np.newaxis, :, :UNIFIED_FORMAT_INDICES['pos_y'] + 1], axis=-1)
    # set the distance to self to a large value to avoid self-distance
    for i in range(len(vehicles)):
        vehicle_vehicle_dist[i, i] = 1000

    return vehicle_vehicle_dist.min(1)

def get_lateral_devs(vehicles, lanes):
    """ Computes the lateral deviations of vehicles from the nearest lane."""
    agents_expanded = vehicles[:, np.newaxis, np.newaxis, :UNIFIED_FORMAT_INDICES['pos_y'] + 1]  # Shape (A, 1, 1, 2)
    diffs = agents_expanded - lanes[np.newaxis, :, :, :]
    dists_squared = np.sum(diffs ** 2, axis=-1)  # Shape (A, N, 20)
    min_dists_squared = np.min(dists_squared, axis=(1, 2))

    return np.sqrt(min_dists_squared)

def get_angular_devs(vehicles, lanes):
    """ Computes the angular deviations of vehicles from the nearest lane segment."""
    agent_positions = vehicles[:, :UNIFIED_FORMAT_INDICES['pos_y'] + 1]  # Extract positions (x, y)
    cos_theta = vehicles[:, UNIFIED_FORMAT_INDICES['cos_heading']]
    sin_theta = vehicles[:, UNIFIED_FORMAT_INDICES['sin_heading']]
    agent_headings = np.arctan2(sin_theta, cos_theta)

    agents_expanded = agent_positions[:, np.newaxis, np.newaxis, :]
    direction_vectors = lanes[:, 1:, :] - lanes[:, :-1, :]
    centerline_headings = np.arctan2(direction_vectors[..., 1], direction_vectors[..., 0])  # Shape (N, 19)
    diffs = agents_expanded - lanes[np.newaxis, :, :, :]  # Shape (A, N, 20, 2)
    dists_squared = np.sum(diffs ** 2, axis=-1)  # Shape (A, N, 20)
    # Find the indices of the nearest centerline point for each agent
    nearest_flat_indices = np.argmin(dists_squared.reshape(dists_squared.shape[0], -1), axis=-1)
    nearest_centerline_indices = nearest_flat_indices // dists_squared.shape[2]
    nearest_point_indices = nearest_flat_indices % dists_squared.shape[2]

    # Handle the case where nearest point is the first or last in the centerline
    nearest_point_indices = np.clip(nearest_point_indices, 1, dists_squared.shape[-1] - 1)
    # Get the corresponding headings of the nearest segments
    nearest_centerline_headings = centerline_headings[nearest_centerline_indices, nearest_point_indices - 1]

    # Compute the angular deviation in radians and convert to degrees
    angular_deviation_radians = np.arctan2(np.sin(agent_headings - nearest_centerline_headings),
                                           np.cos(
                                               agent_headings - nearest_centerline_headings))  # Ensure correct angle difference
    angular_deviation_degrees = np.degrees(angular_deviation_radians)

    return angular_deviation_degrees

def get_lengths(vehicles):
    """ Returns the lengths of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['length']]

def get_widths(vehicles):
    """ Returns the widths of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['width']]

def get_speeds(vehicles):
    """ Returns the speeds of the vehicles in the scene."""
    return vehicles[:, UNIFIED_FORMAT_INDICES['speed']]

def compute_collision_rate(samples):
    """Fraction of vehicles involved in at least one five-circle overlap.

    Single snapshots [N,7], vehicle-weighted; not an average of scene rates.
    The geometry/threshold follows the supplied and upstream metric code.
    """
    total = colliding = 0
    for scene in samples:
        vehicles = scene['vehicles']
        centers = []
        for v in vehicles:
            heading = np.arctan2(v[4], v[3])
            centroids, _ = compute_vehicle_circles(v[:2], heading, v[5], v[6])
            centers.append(centroids)
        centers = np.asarray(centers)
        for j in range(len(vehicles)):
            for k in range(len(vehicles)):
                if j == k:
                    continue
                threshold = (vehicles[j, 6] + vehicles[k, 6]) / np.sqrt(3.8)
                distances = np.linalg.norm(centers[j, :, None] - centers[k, None, :], axis=-1)
                if (distances < threshold).sum() >= 1:
                    colliding += 1
                    break
        total += len(vehicles)
    if not total:
        raise ValueError('Collision rate is undefined for zero vehicles')
    return colliding / total


def get_edge_index_complete_graph(num_nodes):
    """Complete directed edges, including self, in source-major order.

    This dependency-free helper is only a fallback for cache dictionaries that
    omit edge_index_lane_to_lane. Official GT caches carry explicit edge indices;
    get_networkx_lane_graph uses those indices without guessing their order.
    """
    n = int(num_nodes)
    if n < 0 or n != num_nodes:
        raise ValueError('num_nodes must be a nonnegative integer')
    return torch.stack((torch.arange(n).repeat_interleave(n), torch.arange(n).repeat(n)))


def get_networkx_lane_graph(data):
    """Read the predecessor-labelled source->destination edges from a cache.

    In the official convention label 1 on (i,j) means i is a predecessor of j:
    the directed successor graph therefore contains i -> j, NOT j -> i.
    """
    points = np.asarray(data['road_points'])
    n = int(data['num_lanes'])
    if points.ndim != 3 or points.shape[0] != n or points.shape[-1] != 2 or n <= 0:
        raise ValueError('road_points must be a nonempty [num_lanes,P,2] array')
    edge_index = data.get('edge_index_lane_to_lane')
    if edge_index is None:
        edge_index = get_edge_index_complete_graph(n)
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index = np.asarray(edge_index)
    conn = np.asarray(data['road_connection_types'])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index_lane_to_lane must be [2,E]')
    if conn.ndim != 2 or conn.shape[0] != edge_index.shape[1] or conn.shape[1] < 2:
        raise ValueError('road_connection_types must align with lane edge columns')
    if (not np.isfinite(edge_index).all() or not np.equal(edge_index, np.floor(edge_index)).all()
            or (edge_index < 0).any() or (edge_index >= n).any()):
        raise ValueError('Lane edge index is noninteger or out of bounds')
    adjacency = np.zeros((n, n))
    for source, destination in edge_index[:, conn[:, 1] == 1].T:
        adjacency[int(source), int(destination)] = 1
    return nx.DiGraph(incoming_graph_data=adjacency), points


def get_compact_lane_graph(G, lanes, num_points_per_lane=20):
    """Metric-level lane compaction (after the dataset-level compaction).

    Preserve lane/edge iteration, group traversal, omission of the next segment's
    first point and conditional 20-point resampling. Do not sort groups or merge
    lateral relations into the successor graph.
    """
    lane_by_id = dict(enumerate(lanes))
    predecessors = {i: [] for i in lane_by_id}
    successors = {i: [] for i in lane_by_id}
    for source, destination in G.edges():
        predecessors[destination].append(source)
        successors[source].append(destination)
    groups = find_lane_groups(predecessors, successors)
    compact_lanes, compact_successors = {}, {}
    for group_id, lane_ids in groups.items():
        parts = []
        for i, lane_id in enumerate(lane_ids):
            parts.append(lane_by_id[lane_id] if i == 0 else lane_by_id[lane_id][1:])
        compact_lanes[group_id] = np.concatenate(parts, axis=0)
        compact_successors[group_id] = [
            find_lane_group_id(next_id, groups) for next_id in successors[lane_ids[-1]]
        ]
    new_index = {lane_id: i for i, lane_id in enumerate(compact_lanes)}
    reindexed_successors = {}
    output = []
    for lane_id, points in compact_lanes.items():
        reindexed_successors[new_index[lane_id]] = [new_index[j] for j in compact_successors[lane_id]]
        if len(points) != num_points_per_lane:
            points = resample_polyline(points, num_points=num_points_per_lane)
        output.append(points[None, :, :])
    if not output:
        raise ValueError('Cannot compact an empty lane graph')
    points = np.concatenate(output, axis=0)
    adjacency = np.zeros((len(points), len(points)))
    for lane_id, neighbors in reindexed_successors.items():
        for neighbor in neighbors:
            adjacency[lane_id, neighbor] = 1
    return nx.DiGraph(incoming_graph_data=adjacency), points


def convert_data_to_unified_format(data, dataset_name='waymo_gt'):
    """Waymo branch only; cached physical units, not normalized model inputs."""
    if dataset_name not in ('waymo', 'waymo_gt'):
        raise NotImplementedError('This portable backend only supports Waymo / Waymo GT')
    if int(data['lg_type']) != NON_PARTITIONED:
        raise ValueError('Initial-scene metrics require regular lg_type=0')
    graph, lanes = get_networkx_lane_graph(data)
    states = np.asarray(data['agent_states'])
    types = np.asarray(data['agent_types'])
    if states.ndim != 2 or states.shape[1] != 7 or types.shape != (len(states), 3):
        raise ValueError('Expected agent_states [N,7] and agent_types [N,3]')
    if not np.isfinite(states).all() or not np.isfinite(types).all():
        raise ValueError('Nonfinite agent data')
    vehicles = states[np.argmax(types, axis=1) == NUPLAN_VEHICLE]
    compact_graph, compact_lanes = get_compact_lane_graph(graph, lanes)
    return {'G': compact_graph, 'lanes': compact_lanes, 'vehicles': vehicles}


def compute_jsd_metrics(samples, gt_samples):
    """Concatenate-first reference path; evaluator uses equivalent integer sums."""
    if len(samples) != len(gt_samples):
        raise ValueError('Scene counts must match')
    def collect(scenes):
        values = [[] for _ in range(6)]
        for scene in scenes:
            v = scene['vehicles']
            lanes = resample_lanes(scene['lanes'], num_points=100)
            onroad = get_onroad_vehicles(v, lanes)
            if len(v) > 1:
                values[0].append(get_nearest_dists(v))
            if len(onroad):
                values[1].append(get_lateral_devs(onroad, lanes))
                values[2].append(get_angular_devs(onroad, lanes))
            values[3].append(get_lengths(v))
            values[4].append(get_widths(v))
            values[5].append(get_speeds(v))
        if any(not x or sum(len(a) for a in x) == 0 for x in values):
            raise ValueError('One or more metric feature distributions are empty')
        return [np.concatenate(x) for x in values]
    gen, real = collect(samples), collect(gt_samples)
    specs = ((0,50,1,10), (0,1.5,0.1,10), (-200,200,5,100),
             (0,25,0.1,100), (0,5,0.1,100), (0,50,1,100))
    return tuple(jsd(g, r, lo, hi, step) * scale
                 for g,r,(lo,hi,step,scale) in zip(gen,real,specs))


def compute_agent_metrics(samples, gt_samples):
    keys = ('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd')
    result = dict(zip(keys, compute_jsd_metrics(samples, gt_samples)))
    result['collision_rate'] = compute_collision_rate(samples) * 100
    return result
