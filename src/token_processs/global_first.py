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
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

file_path='/home/ke/code/sim/src/waymo_data/full/training_map2_a_light/'

files = os.listdir(file_path)[::10]
valid_list = []


def load_one_partial(file):
    data=torch.load(file_path+file)

    valid = data["agent"]["valid_mask"]
    pos = data["agent"]["position"]
    heading = data["agent"]["heading"]

    pos_1=pos[:,10,:2]
    heading_1=heading[:,10]

    av_index = torch.where(data["agent"]["role"][:, 0])[0].item()

    ego_pos=pos_1[av_index]
    ego_head=heading_1[av_index]

    entry_pos=transform_to_local(
        pos_1[None],# [n_agent, n_step, 2]
        None, # [n_agent, n_step]
        ego_pos[None],# [n_agent, 2]
        ego_head[None]# [n_agent]
    )[0]

    #print(torch.linalg.norm(relative_pos[valid_1[:, 1:]],dim=-1).max())

    valid_list.append(entry_pos[0])

# with Pool(cpu_count()) as pool:
    #valid_pos = []

for file in tqdm(files):
    out=load_one_partial(file)
        #valid_pos.append(out)

    # for out in tqdm(pool.imap_unordered(load_one_partial, files),
    #                 total=len(files)):
    #     valid_pos.append(out)

valid_pos=torch.cat(valid_list)

torch.save(valid_pos,"frist_pos.pt")
