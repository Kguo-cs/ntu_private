import numpy as np
from scipy.spatial import distance
import torch
import numpy as np
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from data_preprocess import  decode_map_features_from_proto
import tensorflow as tf
from .mmd_metric import compute_mmd_metrics
from .trafficgen_metrics import _get_trafficgen_data

def resample_polyline(points, num_points=20):
    """Resample a polyline to `num_points` equally spaced points along its arc-length."""
    # Calculate the cumulative distances along the polyline
    distances = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create an array of 20 evenly spaced distance values along the polyline
    target_distances = np.linspace(0, cumulative_distances[-1], num=num_points)

    # Interpolate to find x and y values at these target distances
    x_new = np.interp(target_distances, cumulative_distances, points[:, 0])
    y_new = np.interp(target_distances, cumulative_distances, points[:, 1])

    # Combine x and y coordinates into a single array
    new_points = np.stack((x_new, y_new), axis=-1)

    return new_points

def compute_gen_samples(data,tokenized_agent,pred_traj,pred_vel,pred_head,pred_sizes,samples,gt_samples,gt_dist):
    pred_vel = torch.stack(pred_vel, dim=1)

    pred_speeds=pred_vel.norm(dim=-1)
    gt_init_timestep=5
    gen_init_timestep=5

    batch = tokenized_agent["batch"]
    cos = torch.cos(pred_head[:, :, gen_init_timestep])
    sin = torch.sin(pred_head[:, :, gen_init_timestep])

    state = torch.cat(
        [pred_traj[:, :, gen_init_timestep], pred_speeds[:, :, None], cos[:, :, None], sin[:, :, None], pred_sizes[:, :, gen_init_timestep, :2],pred_vel],
        dim=-1)  # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
    type = tokenized_agent["type"]

    if gt_dist is None:
        valid=data["agent"]["valid_mask"][:, gt_init_timestep]#9051

        gt_vel =data["agent"]["velocity"][:, gt_init_timestep]
        gt_speed = gt_vel.norm(dim=-1)
        gt_cos = torch.cos(data["agent"]["heading"][:, gt_init_timestep])
        gt_sin = torch.sin(data["agent"]["heading"][:, gt_init_timestep])
        gt_shape = data["agent"]["shape"]
        gt_pos = data["agent"]["position"][:, gt_init_timestep, :2]
        gt_type= data["agent"]["type"][valid]
        gt_id=data["agent"]["id"][valid]
        gt_batch=batch[valid]

        real_state = torch.cat([gt_pos, gt_speed[:, None], gt_cos[:, None], gt_sin[:, None], gt_shape[:, :2],gt_vel],
                               dim=-1)[valid]  # [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]

    for b in range(data.num_graphs):
        vehicles = state[(batch == b) & (type == 0)].cpu().numpy()

        if gt_dist is None:

            scenario_file = data["tfrecord_path"][b]

            scenario = scenario_pb2.Scenario()
            for data_b in tf.data.TFRecordDataset([scenario_file], compression_type=""):
                scenario.ParseFromString(bytes(data_b.numpy()))


            map_infos = decode_map_features_from_proto(scenario.map_features)
            all_polylines = map_infos["all_polylines"]
            compact_centerlines = []

            for lane in map_infos["lane"]:
                lane_type = lane['type']
                polyline_index = lane['polyline_index']
                if lane_type == 3:  # lane_type == 0 or
                    continue

                lane_point = all_polylines[polyline_index[0]:polyline_index[1]]

                resampled_lane = resample_polyline(lane_point, num_points=100)
                compact_centerlines.append(resampled_lane)

            compact_centerlines = np.stack(compact_centerlines, axis=0)

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

            unified_data = {
                "all_agents": state[(batch == b)].cpu().numpy(),
                'vehicles': vehicles
            }
            samples.append(unified_data)

            gt_id_b=gt_id[gt_batch==b]

            real_vehicles = real_state[(gt_batch == b) & (gt_type == 0)].cpu().numpy()

            all_agents=real_state[gt_batch == b]

            PZH_TRACK_NAMES=_get_trafficgen_data(scenario,current_t=gt_init_timestep)

            mask=torch.isin(gt_id_b,torch.Tensor(PZH_TRACK_NAMES.astype(np.long)).to(device=gt_id.device))
            # mask1=torch.isin(torch.Tensor(PZH_TRACK_NAMES.astype(np.long)).to(device=gt_id.device),gt_id_b)
            #
            # print(torch.all(mask1))

            select_agents=all_agents[mask].cpu().numpy()

            unified_data = {
                'lanes': compact_centerlines,  # [num_lanes, 20, 2]
                'vehicles': real_vehicles,
                "agents": select_agents,
               # "all_agents": all_agents.cpu().numpy()
            }

            gt_samples.append(unified_data)
        else:
            unified_data = {
                'vehicles': vehicles,
                "all_agents": state[(batch == b)].cpu().numpy()
            }
            samples.append(unified_data)

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
    if 'all_agents' in samples[0].keys():
        mmd_metrics = compute_mmd_metrics(samples, gt_samples)
        metrics.update(mmd_metrics)
        # all_mmd_metrics = compute_mmd_metrics(samples, gt_samples,"all_")
        # metrics.update(all_mmd_metrics)

    return metrics,gt_dist
