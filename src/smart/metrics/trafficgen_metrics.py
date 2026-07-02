import numpy as np
import copy

import hydra
import omegaconf
import torch
import torchmetrics
import tqdm
import numpy as np
from data_preprocess import decode_tracks_from_proto,get_agent_features
import copy
from .gen_extract import extract_tracks,extract_map,extract_dynamic

def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def process_lane(lane, max_vec, lane_range, offset=-40):
    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    vec_id_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vec_ids = np.ones([b_s, points.shape[1] - 1, 1]) * id
        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)
        vec_id_list.append(vec_ids)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)
    vec_id = np.concatenate(vec_id_list, axis=1) if vec_id_list else np.zeros([b_s, 0, 1])

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    all_id = np.zeros([b_s, max_vec, 1])

    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]
        vec_id_t = vec_id[t][mask_t]

        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])
        vec_id_t = vec_id_t[idx]

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]
        vec_id_t = vec_id_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        vec_id_t = np.pad(vec_id_t, ([0, max_vec - vec_id_t.shape[0]], [0, 0]))

        all_vec[t] = vector_t
        all_mask[t] = mask_t
        all_id[t] = vec_id_t

    return all_vec, all_mask.astype(bool), all_id.astype(int)

def process_map(lane, traf=None, center_num=384, edge_num=128, lane_range=60, offest=-40, rest_num=192):
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    # print(traf)
    if traf is not None:
        for i in range(b_s):
            traf_t = traf[i]
            lane_id_t = lane_id[i]
            # print(traf_t)
            for a_traf in traf_t:
                # print(a_traf)
                control_lane_id = a_traf[0]
                state = a_traf[-2]
                lane_idx = np.where(lane_id_t == control_lane_id)
                lane_with_traf[i, lane_idx, -1] = state
        lane = lane_with_traf

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask, cent_id = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask, _ = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask, _ = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask, _ = process_lane(lane[:, rest], rest_num, lane_range, offest)

    return cent, cent_mask, cent_id, bound, bound_mask, cross, cross_mask, rest, rest_mask


def _transform_coordinate_map( data):
    """
    Every frame is different
    """
    timestep = data['all_agent'].shape[0]

    ego = data['all_agent'][:, 0]
    pos = ego[:, [0, 1]][:, np.newaxis]

    lane = data['lane'][np.newaxis]
    lane = np.repeat(lane, timestep, axis=0)
    lane[..., :2] -= pos

    x = lane[..., 0]
    y = lane[..., 1]
    ego_heading = ego[:, [4]]
    lane[..., :2] = rotate(x, y, -ego_heading)

    unsampled_lane = data['unsampled_lane'][np.newaxis]
    unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
    unsampled_lane[..., :2] -= pos

    x = unsampled_lane[..., 0]
    y = unsampled_lane[..., 1]
    ego_heading = ego[:, [4]]
    unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
    return lane, unsampled_lane[0]


def _get_trafficgen_data( scenario):
    """
    PZH:
    I don't want to waste time to read through the LCTGen code,
    which essentially is from the TrafficGen code base.
    I've read the TrafficGen code base and I really really don't want
    to look into it for the second time.
    Just copy the code here and modify it to fit the current code base.
    """

    scene = dict()
    scene['id'] = scenario.scenario_id
    sdc_index = scenario.sdc_track_index
    scene['all_agent'] ,PZH_TRACK_NAMES= extract_tracks(scenario.tracks, sdc_index)

    # track_infos=decode_tracks_from_proto(scenario)
    # agent = get_agent_features(
    #     track_infos,
    #     split='val',
    #     num_historical_steps=10 + 1,
    #     num_steps=91,
    # )
    #
    # ego = scenario.tracks[sdc_index]
    scene['traffic_light'] = extract_dynamic(scenario.dynamic_map_states)
    global SAMPLE_NUM
    SAMPLE_NUM = 10
    scene['lane'], scene['center_info'] = extract_map(scenario.map_features)

    SAMPLE_NUM = 10e9
    scene['unsampled_lane'], _ = extract_map(scenario.map_features)

    data=scene

    case_info = {}

    max_time_step = 190
    gap = 190
    index = -1
    RANGE = 50
    data['all_agent'] = data['all_agent'][10:11]
    data['traffic_light'] = data['traffic_light'][10:11]

    data['lane'], _ = _transform_coordinate_map(data)

    def _process_agent(agent, sort_agent):

        ego = agent[:, 0]

        # transform every frame into ego coordinate in the first frame
        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]

        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading

        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)

        mask = agent_mask * agent_type_mask
        # use agent range mask only for the first frame
        # allow agent to be out of range in the future frames
        mask[0, :] *= agent_range_mask[0, :]

        return agent, mask.astype(bool)

    case_info["agent"], case_info["agent_mask"] = _process_agent(data['all_agent'], False)
    case_info['center'], case_info['center_mask'], case_info['center_id'], case_info['bound'], case_info[
        'bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
        data['lane'], data['traffic_light'], lane_range=RANGE, offest=0)

    # get vector-based representatiomn
    def _get_vec_based_rep(case_info, PZH_TRACK_NAMES):
        THRES = 5
        thres = THRES
        # max_agent_num = 32
        # _process future agent

        agent = case_info['agent']
        vectors = case_info["center"]

        agent_mask = case_info['agent_mask']

        vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
        vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]

        b, vec_num = vec_y.shape
        _, agent_num = agent_x.shape

        vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
        vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

        agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
        agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

        dist = np.sqrt((vec_x - agent_x) ** 2 + (vec_y - agent_y) ** 2)

        cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
        dist[cent_mask == 0] = 10e5
        vec_index = np.argmin(dist, -1)
        min_dist_to_lane = np.min(dist, -1)
        min_dist_mask = min_dist_to_lane < thres

        selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

        vx, vy = agent[..., 2], agent[..., 3]
        v_value = np.sqrt(vx ** 2 + vy ** 2)
        low_vel = v_value < 0.1

        dir_v = np.arctan2(vy, vx)
        x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
        dir = np.arctan2(y2 - y1, x2 - x1)
        agent_dir = agent[..., 4]

        v_relative_dir = cal_rel_dir(dir_v, agent_dir)
        relative_dir = cal_rel_dir(agent_dir, dir)

        v_relative_dir[low_vel] = 0

        v_dir_mask = abs(v_relative_dir) < np.pi / 6
        dir_mask = abs(relative_dir) < np.pi / 4

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]
        vec_x = (x1 + x2) / 2
        vec_y = (y1 + y2) / 2

        cent_to_agent_x = agent_x - vec_x
        cent_to_agent_y = agent_y - vec_y

        coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

        vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

        lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
        long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len

        # ignore other masks for future agents (to support out-of-range agent prediction)
        total_mask = agent_mask
        # for the first frame, use all masks to filter out off-road agents
        total_mask[0, :] = (min_dist_mask * agent_mask * v_dir_mask * dir_mask)[0, :]

        total_mask[:, 0] = 1
        total_mask = total_mask.astype(bool)

        b_s, agent_num, agent_dim = agent.shape
        agent_ = np.zeros([b_s, agent_num, agent_dim])
        agent_mask_ = np.zeros([b_s, agent_num]).astype(bool)

        the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 velocity and direction
        # 6-9 lane vector
        # 10-11 lane type and traff state
        info = np.concatenate(
            [
                vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis],
                v_value[..., np.newaxis], v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec
            ], -1
        )

        info_ = np.zeros([b_s, agent_num, info.shape[-1]])

        start_mask = total_mask[0]
        for i in range(agent.shape[0]):
            agent_i = agent[i][start_mask]
            info_i = info[i][start_mask]

            step_mask = total_mask[i]
            valid_mask = step_mask[start_mask]

            agent_i = agent_i[:agent_num]
            info_i = info_i[:agent_num]

            valid_num = agent_i.shape[0]
            agent_i = np.pad(agent_i, [[0, agent_num - agent_i.shape[0]], [0, 0]])
            info_i = np.pad(info_i, [[0, agent_num - info_i.shape[0]], [0, 0]])

            agent_[i] = agent_i
            info_[i] = info_i
            agent_mask_[i, :valid_num] = valid_mask[:valid_num]

        PZH_TRACK_NAMES_new = np.array(list(PZH_TRACK_NAMES[start_mask]) + [None] * (agent_num - start_mask.sum()))

        case_info['vec_based_rep'] = info_[..., 1:]
        case_info['agent_vec_index'] = info_[..., 0].astype(int)
        case_info['agent_mask'] = agent_mask_
        case_info["agent"] = agent_

        return case_info, PZH_TRACK_NAMES_new

    case_info, PZH_TRACK_NAMES = _get_vec_based_rep(case_info, PZH_TRACK_NAMES)

    case_num = case_info['agent'].shape[0]
    case_list = []
    for i in range(case_num):
        dic = {}
        for k, v in case_info.items():
            dic[k] = v[i]
        case_list.append(dic)

    # PZH: Obviously, you only pick T=0 from the data.
    ret = case_list[0]
    ret["PZH_TRACK_NAMES"] = PZH_TRACK_NAMES
    return ret["PZH_TRACK_NAMES"][ret['agent_mask']]