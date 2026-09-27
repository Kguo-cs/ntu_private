import numpy as np
import tensorflow as tf
import torch
from scipy.spatial import distance
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)

from data_preprocess import decode_map_features_from_proto
from .mmd_metric import compute_mmd_metrics
from .trafficgen_metrics1 import (
    _extract_center_lane_vectors_in_ego,
    get_select_index,
)


# State layouts used below.
REAL_POS = slice(0, 2)
REAL_TYPE = 10
REAL_VEL = slice(7, 9)
REAL_HEADING = 9
MODEL_EGO_INDEX = -1  # This codebase stores the ego as the last local agent.


def resample_polyline(points: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Resample a polyline at uniformly spaced arc-length positions."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :2], num_points, axis=0)

    segment_length = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_length)))

    if arc_length[-1] <= 1e-8:
        return np.repeat(points[:1, :2], num_points, axis=0)

    target = np.linspace(0.0, arc_length[-1], num_points)
    return np.stack(
        (
            np.interp(target, arc_length, points[:, 0]),
            np.interp(target, arc_length, points[:, 1]),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _load_scenario(tfrecord_path: str):
    """Read the single Waymo Scenario stored in one TFRecord file."""
    scenario = scenario_pb2.Scenario()
    records = tf.data.TFRecordDataset([tfrecord_path], compression_type="")
    for record in records:
        scenario.ParseFromString(bytes(record.numpy()))
        return scenario
    raise ValueError(f"No Scenario record found in {tfrecord_path}")


def _extract_resampled_centerlines(scenario, num_points: int = 100) -> np.ndarray:
    """Extract non-bike centerlines for the JSD lane metrics."""
    map_info = decode_map_features_from_proto(scenario.map_features)
    all_polylines = map_info["all_polylines"]
    centerlines = []

    for lane in map_info["lane"]:
        if lane["type"] == 3:  # bike lane in the current preprocessing convention
            continue

        start, end = lane["polyline_index"]
        lane_points = all_polylines[start:end]
        if len(lane_points) == 0:
            continue
        centerlines.append(resample_polyline(lane_points, num_points=num_points))

    if not centerlines:
        return np.empty((0, num_points, 2), dtype=np.float32)
    return np.stack(centerlines, axis=0)


def _build_real_state(data, timestep: int, batch: torch.Tensor):
    """Build real-agent states without deleting invalid slots.

    Keeping the complete local agent order is necessary because TrafficGen
    applies a validity mask but does not compress the agent array before
    sorting and filtering.

    Columns:
        x, y, speed, cos(h), sin(h), length, width, vx, vy, heading, type
    """
    valid = data["agent"]["valid_mask"][:, timestep]
    velocity = data["agent"]["velocity"][:, timestep]
    heading = data["agent"]["heading"][:, timestep]

    state = torch.cat(
        (
            data["agent"]["position"][:, timestep, :2],
            velocity.norm(dim=-1, keepdim=True),
            heading.cos().unsqueeze(-1),
            heading.sin().unsqueeze(-1),
            data["agent"]["shape"][:, :2],
            velocity,
            heading.unsqueeze(-1),
            data["agent"]["type"].unsqueeze(-1),
        ),
        dim=-1,
    )

    return state, valid, batch, data["agent"]["type"]


def _build_generated_state(
    pred_traj,
    pred_vel,
    pred_head,
    pred_sizes,
    timestep: int,
):
    """Build generated states with shape ``[agent, sample, 9]``.

    Columns:
        x, y, speed, cos(h), sin(h), length, width, vx, vy
    """
    if isinstance(pred_vel, (list, tuple)):
        pred_vel = torch.stack(pred_vel, dim=1)

    heading = pred_head[:, :, timestep]
    return torch.cat(
        (
            pred_traj[:, :, timestep],
            pred_vel.norm(dim=-1, keepdim=True),
            heading.cos().unsqueeze(-1),
            heading.sin().unsqueeze(-1),
            pred_sizes[:, :, timestep, :2],
            pred_vel,
        ),
        dim=-1,
    )


def compute_gen_samples(
    data,
    tokenized_agent,
    pred_traj,
    pred_vel,
    pred_head,
    pred_sizes,
    samples,
    gt_samples,
    gt_dist,
    compute_mmd=True
):
    """Append generated and, when needed, ground-truth scene data."""
    init_timestep = 5
    batch = tokenized_agent["batch"]
    agent_type = tokenized_agent["type"]

    generated_state = _build_generated_state(
        pred_traj=pred_traj,
        pred_vel=pred_vel,
        pred_head=pred_head,
        pred_sizes=pred_sizes,
        timestep=init_timestep,
    )

    if gt_dist is None:
        real_state, real_valid, real_batch, real_type = _build_real_state(
            data=data,
            timestep=init_timestep,
            batch=batch,
        )

    for graph_index in range(data.num_graphs):
        output_index = len(samples)

        if gt_dist is None:
            scenario = _load_scenario(data["tfrecord_path"][graph_index])
            centerlines = _extract_resampled_centerlines(scenario)

            graph_mask = real_batch == graph_index
            real_agents_full = real_state[graph_mask].detach().cpu().numpy()
            graph_valid = real_valid[graph_mask].detach().cpu().numpy().astype(bool)
            graph_types = real_type[graph_mask].detach().cpu().numpy()


            if compute_mmd:
                ego_index = MODEL_EGO_INDEX % len(real_agents_full)
                lane_vectors = _extract_center_lane_vectors_in_ego(
                    scenario=scenario,
                    ego_xy=real_agents_full[ego_index, REAL_POS],
                    ego_heading=float(real_agents_full[ego_index, REAL_HEADING]),
                )

                # Exact TrafficGen semantics, adapted only for this model's type
                # encoding: model type 0 corresponds to Waymo TYPE_VEHICLE == 1.
                selected_index = get_select_index(
                    lane_vectors=lane_vectors,
                    positions=real_agents_full[:, REAL_POS],
                    velocities=real_agents_full[:, REAL_VEL],
                    headings=real_agents_full[:, REAL_HEADING],
                    valid=graph_valid,
                    object_types=graph_types,
                    ego_index=ego_index,
                    vehicle_type=0,
                    lane_range=50.0,
                    lane_dist_thres=5.0,
                    max_agent_num=32,
                    raster_resolution=0.25,
                    randomize_agents=False,
                )
                select_agents=real_agents_full[selected_index]
            else:
                select_agents=None

            valid_vehicle = graph_valid & (graph_types == 0)
            gt_samples.append(
                {
                    "lanes": centerlines,
                    #"lane_vectors": lane_vectors,
                    "vehicles": real_agents_full[valid_vehicle],
                    # "all_agents": real_agents_full[graph_valid],
                    # "select_index": selected_index,
                    "select_agents": select_agents
                }
            )

        graph_mask = batch == graph_index
        generated_agents_full = generated_state[graph_mask].detach().cpu().numpy()
        graph_types = agent_type[graph_mask].detach().cpu().numpy()
        valid_vehicle =  (graph_types == 0)
        samples.append({
            "vehicles": generated_agents_full[valid_vehicle]
        })

        # if compute_mmd:
        #
        #     selected_index = np.asarray(
        #         gt_samples[output_index]["select_index"], dtype=np.int64
        #     )
        #     sampled_dict
        #
        # samples.append(
        #     {
        #         "all_agents": generated_agents_full[graph_mask],
        #         "select_index": selected_index,
        #         "agents": generated_agents_full[selected_index],
        #     }
        # )


# centerlines = data['road_points']
# lanes = {}

# for lane_id in data['road_info']['lane']:
#     lane_type = data['road_info']['lane'][lane_id]['type']
#     if lane_type == 'TYPE_UNDEFINED' or lane_type == 'TYPE_BIKE_LANE':
#         continue
#
#     my_lane = data['road_info']['lane'][lane_id]['polyline']
#     lanes[int(lane_id)] = my_lane[:, :2]
#
# compact_lane.append(data['lane_graph']['lanes'][lane_id])
#
# compact_lane_graph_scene = self.normalize_compact_lane_graph(copy.deepcopy(compact_lane_graph),
#                                                              normalize_dict)
# compact_lane_graph = self.get_lane_graph_within_fov(compact_lane_graph_scene)

# resampled_lanes = []
# idx_to_id = {}
# id_to_idx = {}
# i = 0
#
# for lane_id in compact_lane_graph['lanes']:
#     lane = compact_lane_graph['lanes'][lane_id]
#     resampled_lane = resample_polyline(lane, num_points=self.cfg.num_points_per_lane)
#     resampled_lanes.append(resampled_lane)
#
# resampled_lanes = np.array(resampled_lanes)
# num_lanes = min(len(resampled_lanes), self.cfg.max_num_lanes)
# dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
# closest_lane_ids = np.argsort(dist_to_origin)[:num_lanes]
# road_points = resampled_lanes[closest_lane_ids]
# remove_offroad vehicle, fov: 64*64 in metres, max 30 agent

# 20 point each

# unified format for computing metrics
# [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
UNIFIED_FORMAT_INDICES = {
    'pos_x': 0,
    'pos_y': 1,
    'speed': 2,
    'cos_heading': 3,
    'sin_heading': 4,
    'length': 5,
    'width': 6
}

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


def compute_collision_rate(samples):
    """ Computes the collision rate for the vehicles in the samples.
    Collision rate is computed by testing for overlapping circles around vehicles (with some threshold)."""
    print("Computing collision rate")
    num_vehicles_all = 0
    num_vehicles_in_collision_all = 0
    collision=[]
    for i in range(len(samples)):
        data = samples[i]

        for v in range(data['vehicles'].shape[1]):
            vehicles = data['vehicles'][:,v]

            centroids_all = []
            #radii_all = []
            for vehicle in vehicles:
                # vehicle: [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
                heading = np.arctan2(vehicle[UNIFIED_FORMAT_INDICES['sin_heading']],
                                     vehicle[UNIFIED_FORMAT_INDICES['cos_heading']])
                centroids, radii = compute_vehicle_circles(vehicle[:UNIFIED_FORMAT_INDICES['pos_y'] + 1],
                                                           heading,
                                                           vehicle[UNIFIED_FORMAT_INDICES['length']],
                                                           vehicle[UNIFIED_FORMAT_INDICES['width']])
                centroids_all.append(centroids)
               # radii_all.append(radii)
            centroids_all = np.array(centroids_all)
            #radii_all = np.array(radii_all)

            num_vehicles_in_collision = 0
            for j in range(len(vehicles)):
                is_in_collision = False
                for k in range(len(vehicles)):
                    if j == k:
                        continue

                    thresh = (vehicles[j, 6] + vehicles[k, 6]) / np.sqrt(3.8)
                    dist = np.linalg.norm(centroids_all[j, :, None] - centroids_all[k, None, :], axis=-1)
                    bad = dist < thresh
                    if bad.sum() >= 1:
                        is_in_collision = True
                        break

                if is_in_collision:
                    num_vehicles_in_collision += 1
                    collision.append(True)
                else:
                    collision.append(False)

            num_vehicles_in_collision_all += num_vehicles_in_collision
            num_vehicles_all += len(vehicles)

    return num_vehicles_in_collision_all / num_vehicles_all,np.stack(collision, axis=0)

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




def resample_lanes(lanes, num_points):
    """Resample a list of lanes (each lane is a polyline) to have `num_points` equally spaced points along each lane's arc-length."""
    lanes_resampled = []
    for lane in lanes:
        lanes_resampled.append(resample_polyline(lane, num_points=num_points))

    return np.array(lanes_resampled)



def plot_gen_real_distribution(
    gen,
    real,
    title,
    clip_min,
    clip_max,
    bin_size
):
    import matplotlib.pyplot as plt

    # 与 JSD 完全一致的 clipping
    gen = np.clip(gen, clip_min, clip_max)
    real = np.clip(real, clip_min, clip_max)

    bins = np.arange(clip_min, clip_max + bin_size, bin_size)

    # 关键：weights 让 histogram 变成 probability mass
    gen_weights = np.ones_like(gen) / len(gen)
    real_weights = np.ones_like(real) / len(real)

    plt.figure()
    plt.hist(gen, bins=bins, weights=gen_weights, alpha=0.5, label="Generated")
    plt.hist(real, bins=bins, weights=real_weights, alpha=0.5, label="Real")
    plt.legend()
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

def compute_jsd_metrics(samples, gt_samples,gt_dist,vis):
    """ Computes the JSD agent metrics for the samples and ground truth samples."""
    print("Computing agent jsd metrics")

    nearest_dist_gen_all = []
    lat_dev_gen_all = []
    ang_dev_gen_all = []
    length_gen_all = []
    width_gen_all = []
    speed_gen_all = []

    if gt_dist is None:
        nearest_dist_real_all = []
        lat_dev_real_all = []
        ang_dev_real_all = []
        length_real_all = []
        width_real_all = []
        speed_real_all = []

    for i in range(len(samples)):
        data_gen = samples[i]
        data_real = gt_samples[i]

        lanes_gen=lanes_real=data_real['lanes']

        if gt_dist is None:
            vehicles_real = data_real['vehicles']

            #lanes_real = resample_lanes(data_real['lanes'], num_points=100)
            onroad_vehicles_real = get_onroad_vehicles(vehicles_real, lanes_real)
            if len(vehicles_real) > 1:
                nearest_dist_real_all.append(get_nearest_dists(vehicles_real))
            if len(onroad_vehicles_real) > 0:
                lat_dev_real_all.append(get_lateral_devs(onroad_vehicles_real, lanes_real))
                ang_dev_real_all.append(get_angular_devs(onroad_vehicles_real, lanes_real))
            length_real_all.append(get_lengths(vehicles_real))
            width_real_all.append(get_widths(vehicles_real))
            speed_real_all.append(get_speeds(vehicles_real))

        for j in range(data_gen['vehicles'].shape[1]):
            vehicles_gen = data_gen['vehicles'][:,j] # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
            # resample lanes to higher resolution
            #lanes_gen = resample_lanes(data_gen['lanes'], num_points=100)
            onroad_vehicles_gen = get_onroad_vehicles(vehicles_gen, lanes_gen)

            if len(vehicles_gen) > 1:
                nearest_dist_gen_all.append(get_nearest_dists(vehicles_gen))
            if len(onroad_vehicles_gen) > 0:
                lat_dev_gen_all.append(get_lateral_devs(onroad_vehicles_gen, lanes_gen))
                ang_dev_gen_all.append(get_angular_devs(onroad_vehicles_gen, lanes_gen))
            length_gen_all.append(get_lengths(vehicles_gen))
            width_gen_all.append(get_widths(vehicles_gen))
            speed_gen_all.append(get_speeds(vehicles_gen))

            if vis:
                plot_scene(
                    lanes_real,
                    vehicles_real,
                    vehicles_gen,
                    title=f"Frame_{i}, Sample_{j}"
                )

    nearest_dist_gen_all = np.concatenate(nearest_dist_gen_all, axis=0)
    lat_dev_gen_all = np.concatenate(lat_dev_gen_all, axis=0)
    ang_dev_gen_all = np.concatenate(ang_dev_gen_all, axis=0)
    length_gen_all = np.concatenate(length_gen_all, axis=0)
    width_gen_all = np.concatenate(width_gen_all, axis=0)
    speed_gen_all = np.concatenate(speed_gen_all, axis=0)

    if gt_dist is None:
        nearest_dist_real_all = np.concatenate(nearest_dist_real_all, axis=0)
        lat_dev_real_all = np.concatenate(lat_dev_real_all, axis=0)
        ang_dev_real_all = np.concatenate(ang_dev_real_all, axis=0)
        length_real_all = np.concatenate(length_real_all, axis=0)
        width_real_all = np.concatenate(width_real_all, axis=0)
        speed_real_all = np.concatenate(speed_real_all, axis=0)

        gt_dist = (nearest_dist_real_all, lat_dev_real_all, ang_dev_real_all, length_real_all, width_real_all,
                   speed_real_all)
    else:
        nearest_dist_real_all, lat_dev_real_all, ang_dev_real_all, length_real_all, width_real_all,speed_real_all=gt_dist

    nearest_dist_jsd = jsd(nearest_dist_gen_all, nearest_dist_real_all, clip_min=0, clip_max=50, bin_size=1) * 10
    lat_dev_jsd = jsd(lat_dev_gen_all, lat_dev_real_all, clip_min=0, clip_max=1.5, bin_size=0.1) * 10
    ang_dev_jsd = jsd(ang_dev_gen_all, ang_dev_real_all, clip_min=-200, clip_max=200, bin_size=5) * 100
    length_jsd = jsd(length_gen_all, length_real_all, clip_min=0, clip_max=25, bin_size=0.1) * 100
    width_jsd = jsd(width_gen_all, width_real_all, clip_min=0, clip_max=5, bin_size=0.1) * 100
    speed_jsd = jsd(speed_gen_all, speed_real_all, clip_min=0, clip_max=50, bin_size=1) * 100

    # lat_dev_jsd1 = jsd(np.random.rand(*lat_dev_gen_all.shape)*1.5, lat_dev_real_all, clip_min=0, clip_max=1.5, bin_size=0.1) * 10
    #
    # plot_gen_real_distribution(
    #     nearest_dist_gen_all,
    #     nearest_dist_real_all,
    #     "Nearest Distance",
    #     clip_min=0,
    #     clip_max=50,
    #     bin_size=1
    # )
    #
    # plot_gen_real_distribution(
    #     lat_dev_gen_all,
    #     lat_dev_real_all,
    #     "Lateral Deviation",
    #     clip_min=0,
    #     clip_max=1.5,
    #     bin_size=0.1
    # )
    #
    # plot_gen_real_distribution(
    #     ang_dev_gen_all,
    #     ang_dev_real_all,
    #     "Angular Deviation",
    #     clip_min=-200,
    #     clip_max=200,
    #     bin_size=5
    # )
    #
    # plot_gen_real_distribution(
    #     length_gen_all,
    #     length_real_all,
    #     "Length",
    #     clip_min=0,
    #     clip_max=25,
    #     bin_size=0.1
    # )
    #
    # plot_gen_real_distribution(
    #     width_gen_all,
    #     width_real_all,
    #     "Width",
    #     clip_min=0,
    #     clip_max=5,
    #     bin_size=0.1
    # )
    #
    # plot_gen_real_distribution(
    #     speed_gen_all,
    #     speed_real_all,
    #     "Speed",
    #     clip_min=0,
    #     clip_max=50,
    #     bin_size=1
    # )

    jsds=(nearest_dist_jsd, lat_dev_jsd, ang_dev_jsd, length_jsd, width_jsd, speed_jsd)


    return jsds,gt_dist

def compute_agent_metrics(samples, gt_samples,gt_dist,vis=True):
    """ Computes the agent metrics for the samples and ground truth samples."""
    #collision_rate1,collision1 = collision_rate_from_state(samples)
    #collision_rate,collision = collision_rate_from_state1(samples)

    #gt_collision_rate,gt_collision= compute_collision_rate(gt_samples)

    #gt_collision_rate,gt_collision = collision_rate_from_state1(gt_samples)


    #collision_jsd = jsd(collision.astype(np.float32), gt_collision.astype(np.float32), clip_min=-0.5, clip_max=2, bin_size=1) * 100

    jsds, gt_dist = compute_jsd_metrics(samples, gt_samples,gt_dist,vis)

    nearest_dist_jsd, lat_dev_jsd, ang_dev_jsd, length_jsd, width_jsd, speed_jsd=jsds

    collision_rate,collision = compute_collision_rate(samples)

    metrics = {
        "nearest_dist_jsd": nearest_dist_jsd,
        "lat_dev_jsd": lat_dev_jsd,
        "ang_dev_jsd": ang_dev_jsd,
        "length_jsd": length_jsd,
        "width_jsd": width_jsd,
        "speed_jsd": speed_jsd,
        "collision_rate": collision_rate * 100,
    }


    # MMD needs paired generated-vs-real scenes, so gt_samples must be available.
    if gt_samples[0]['select_agents'] is not None:
        mmd_metrics = compute_mmd_metrics(samples, gt_samples)
        metrics.update(mmd_metrics)
        # all_mmd_metrics = compute_mmd_metrics(samples, gt_samples,"all_")
        # metrics.update(all_mmd_metrics)

    return metrics,gt_dist