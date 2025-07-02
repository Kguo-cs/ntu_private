import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

data_directory = "/home/ke/code/catk/src/waymo_data/full/training_a/"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_inter10_a91v/"

raw_data = "/home/ke/code/catk/src/waymo_data/full/training_inter10/"


os.makedirs(output_path, exist_ok=True)
files = os.listdir(data_directory)

data_dict = {}


def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    for i in range(heading.shape[1] - 1):
        heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
        change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
        heading[:, i + 1][change_needed] = heading[:, i][change_needed]  # sequential
    return heading


def _extrapolate_agent_to_prev_token_step(
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # [n_agent], max will give the first True step
    first_valid_step = torch.max(valid, dim=1).indices

    for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
        n_step_to_extrapolate = t % 5
        if (t == 10) and (not valid[i, 10 - 5]):
            # such that at least one token is valid in the history.
            n_step_to_extrapolate = 5

        if n_step_to_extrapolate > 0:
            vel[i, t - n_step_to_extrapolate: t] = vel[i, t]
            valid[i, t - n_step_to_extrapolate: t] = True
            heading[i, t - n_step_to_extrapolate: t] = heading[i, t]

            for j in range(n_step_to_extrapolate):
                pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

    return valid, pos, heading, vel


for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    tokenized_agent = {}
    # ! get raw trajectory data
    valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
    heading = data["agent"]["heading"]  # [n_agent, n_step]
    pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
    vel = data["agent"]["velocity"]  # [n_agent, n_step, 2]

    # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.
    heading = _clean_heading(valid, heading)
    # ! extrapolate to previous 5th step.
    valid, pos, heading, vel = _extrapolate_agent_to_prev_token_step(
        valid, pos, heading, vel
    )
    # "gt_pos_raw": pos[:, 5:: 5],  # [n_agent, n_step=18, 2]
    # "gt_head_raw": heading[:, self.shift:: self.shift],  # [n_agent, n_step=18]
    # "gt_valid_raw": valid[:, self.shift:: self.shift],  # [n_agent, n_step=18]

    data1["tokenized_agent"]["gt_pos_raw"]= pos#[:, 5:: 5]
    data1["tokenized_agent"]["gt_head_raw"]=heading#[:, 5:: 5]
    data1["tokenized_agent"]["gt_speed_raw"]=torch.norm(vel,dim=-1)#[:, 5:: 5]
    data1["tokenized_agent"]["gt_valid_raw"]=valid#[:, 5:: 5]
    # "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
    # "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
    # "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]

    del data1["tokenized_agent"]["sampled_pos"]
    del data1["tokenized_agent"]["valid_mask"]
    del data1["tokenized_agent"]["sampled_idx"]
    del data1["tokenized_agent"]["sampled_heading"]
    del data1["tokenized_map"]["pl_type"]

    output_file = output_path + filename

    with open(output_file, "wb") as f:
        pickle.dump(data1, f)



