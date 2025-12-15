import pickle
import os
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

file_path='/home/ke/code/catk/src/waymo_data/all_agent/training/'

files = os.listdir(file_path)[::10]
valid_list = []


def load_one_partial(file):
    data=pickle.load(open(file_path+file, "rb"))

    valid = data["agent"]["valid_mask"]
    pos = data["agent"]["position"]
    heading = data["agent"]["heading"]

    pos_1=pos[:,10::5,:2]
    valid_1=valid[:,10::5]
    heading_1=heading[:,10::5]

    for t in range(1,pos_1.shape[1]):
        valid_t=valid_1[:,t]

        entry_agent=~valid_1[:,t-1] & valid_t
        if entry_agent.any():
            present_agent=valid_1[:,t-1]

            pos_t=pos_1[:,t]
            heading_t=heading_1[:,t]

            entry_pos=pos_t[entry_agent]

            present_pos=pos_1[:,t-1][present_agent]
            present_heading=heading_1[:,t-1][present_agent]

            diff = present_pos[:, None, :] - entry_pos[None, :, :]  # (Np, Ne, D)
            cost = torch.linalg.norm(diff, axis=-1)

            row_ind, col_ind = linear_sum_assignment(cost.numpy())

            present_agent_pos=present_pos[ row_ind]

            entry_agent_pos=entry_pos[ col_ind]

            present_agent_heading=present_heading[row_ind]
            entry_agent_heading=heading_t[entry_agent][ col_ind]

            local_pos,local_heading=transform_to_local(
                entry_agent_pos[:,None,:2],  # [n_agent, n_step, 2]
                entry_agent_heading[:,None],  # [n_agent, n_step]
                present_agent_pos[:,:2],  # [n_agent, 2]
                present_agent_heading  # [n_agent]
            )
            local_z=entry_agent_pos[:,2:]-present_agent_pos[:,2:]

            first_pose = torch.cat((local_pos[:,0],local_z, local_heading), dim=-1)

            valid_list.append(first_pose)

    # return valid_list

with Pool(cpu_count()) as pool:
    #valid_pos = []

    for file in tqdm(files):
        out=load_one_partial(file)
        #valid_pos.append(out)

    # for out in tqdm(pool.imap_unordered(load_one_partial, files),
    #                 total=len(files)):
    #     valid_pos.append(out)

valid_pos=torch.cat(valid_list)

torch.save(valid_pos,"/home/ke/code/catk/src/waymo_data/token/first_pose.pt")
