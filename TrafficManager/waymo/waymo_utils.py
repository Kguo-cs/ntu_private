import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

from TrafficManager.LimSim.utils.trajectory import State, Trajectory

# Add LimSim to sys.path
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    to_tensor,
    visualize_bev_hdmap,
)








def limsim2diffusion(
    vehicles,
    ego_idx,
    map_info,
    data_template,
    # vectorized_map: VectorizedLocalMap,
    # map_name,
    agent_command=2,
    last_pose=torch.eye(4),
    drivable_mask=np.ones((200, 200), dtype=np.uint8),
    accel=[0, 0, 9.81],
    rotation_rate=[0, 0, 0],
    vel=[5, 0, 0],
    gen_location="singapore-onenorth",
    gen_prompts="daytime, cloudy, downtown, gray buildings, white cars",
):
    VEH_LENGTH = 4.7
    VEH_WIDTH = 1.6
    VEH_HEIGHT = 1.4

    ego_vehicle = vehicles["egoCar"]
    ego_x, ego_y, ego_yaw = (
        ego_vehicle["xQ"][-1],
        ego_vehicle["yQ"][-1],
        ego_vehicle["yawQ"][-1],
    )
    ego_yaw_deg = ego_vehicle["yawQ"][-1] * 180 / np.pi

    bbox_list = []
    label_list = []

    def transform(pos, origin):
        # pos is the coordinate and orientation to be transformed, origin is the coordinate and orientation of the new origin
        # Returns the transformed coordinate and orientation
        x, y, yaw = pos
        x0, y0, yaw0 = origin
        # Calculate the displacement and angle relative to the new origin
        dx = x - x0
        dy = y - y0
        dtheta = yaw - yaw0
        # Calculate the coordinates and orientation in the new coordinate system
        x_new = dx * np.cos(yaw0) + dy * np.sin(yaw0)
        y_new = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
        yaw_new = dtheta
        return x_new, y_new, yaw_new

    for sur_veh in vehicles["carInAoI"]:
        sur_x, sur_y, sur_yaw = (
            sur_veh["xQ"][-1],
            sur_veh["yQ"][-1],
            sur_veh["yawQ"][-1],
        )
        tran_x, tran_y, tran_yaw = transform(
            (sur_x, sur_y, sur_yaw), (ego_x, ego_y, ego_yaw)
        )
        tran_x, tran_y, tran_yaw = transform(
            (tran_x, tran_y, tran_yaw), (0, 0, -np.pi / 2)
        )
        # print(sur_veh['id'], tran_x, tran_y, tran_yaw,  tran_yaw+np.pi/2)
        bbox_list.append(
            [
                tran_x,
                tran_y,
                -0.8,
                VEH_WIDTH,
                VEH_LENGTH,
                VEH_HEIGHT,
                -(tran_yaw + np.pi / 2),
                0,
                0,
            ]
        )

        # plot_vehicle((tran_x, tran_y, tran_yaw), color='blue')
        label_list.append(0)  # 0 for vehicle

    send_data = {}
    # ------------ meta ------------ #
    send_data["metas"] = data_template["metas"]
    send_data["metas"]["location"] = gen_location
    send_data["metas"]["description"] = gen_prompts
    print(
        f"location: {send_data['metas']['location']}\ndescription: {send_data['metas']['description']}")
    send_data["metas"]["ego_pos"] = torch.Tensor(
        [
            [np.cos(ego_yaw), -np.sin(ego_yaw), 0, ego_x],
            [np.sin(ego_yaw), np.cos(ego_yaw), 0, ego_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    send_data["metas"]["accel"] = accel
    send_data["metas"]["rotation_rate"] = rotation_rate
    send_data["metas"]["vel"] = vel

    # ------------ bboxes ------------ #
    if len(bbox_list) != 0:
        gt_bboxes_3d = torch.tensor(bbox_list)
        send_data["gt_bboxes_3d"] = gt_bboxes_3d
        send_data["gt_labels_3d"] = torch.tensor(label_list)
    else:
        gt_bboxes_3d = torch.empty(0, 9)
        send_data["gt_bboxes_3d"] = gt_bboxes_3d
        send_data["gt_labels_3d"] = torch.empty(0)

    # ------------ HDMap ------------ #
    anns_results = vectorized_map.gen_vectorized_samples(
        map_name, [ego_x, ego_y], np.deg2rad(ego_yaw_deg - 90)
    )

    gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
    if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
    else:
        gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        except:
            gt_vecs_pts_loc = gt_vecs_pts_loc
    send_data["gt_vecs_label"] = gt_vecs_label
    gt_lines_instance = gt_vecs_pts_loc.instance_list
    gt_map_pts = []
    for i in range(len(gt_lines_instance)):
        pts = np.array(list(gt_lines_instance[i].coords))
        gt_map_pts.append(pts.tolist())
    send_data["gt_lines_instance"] = gt_map_pts

    # ---------------ref pose------------------#
    send_data["relative_pose"] = torch.matmul(
        torch.inverse(send_data["metas"]["ego_pos"]), last_pose
    )

    # ---------------drivable mask- -----------------#
    send_data["drivable_mask"] = drivable_mask

    # ---------------Agent command-----------------#
    send_data["agent_command"] = agent_command

    return send_data
