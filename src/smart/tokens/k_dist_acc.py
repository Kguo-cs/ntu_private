import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np
import matplotlib.pyplot as plt
from src.smart.utils import  wrap_angle


def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 6, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()
        x0 = X[choice_index]
        # res_mask = torch.sum((X - x0) ** 2, dim=[1, 2]) / 4.0 > (tol**2)
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
    print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0)  # [N, 6, 3]


traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt").cuda()
type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt").cuda()

codebook_list=[]

for type_id in [0,1,2]:
    veh_traj=traj[type==type_id]

    veh_vel=(veh_traj[:,1:,:2]-veh_traj[:,:-1,:2])/0.5

    veh_speed=torch.norm(veh_vel,dim=-1)
    veh_acc=(veh_speed[:,1:]-veh_speed[:,:-1])/0.5

    # yaw
    veh_rate=wrap_angle(veh_traj[:,1:,2]-veh_traj[:,:-1,2])/0.5#

    # Flatten
    acc_flat = veh_acc.flatten()
    rate_flat = veh_rate[:,1:].flatten()

    rate_flat=torch.cat([rate_flat,-rate_flat])
    acc_flat=torch.cat([acc_flat,acc_flat])


    s=torch.stack([acc_flat,rate_flat],dim=-1)[:100000000,None]
    tol_dist = [0.03, 0.03, 0.03]  # veh, ped, cyc

    ret_traj = Kdisk_cluster(X=s, N=2048, tol=tol_dist[type_id], a_pos=s)

    codebook_list.append(ret_traj)

codebook_list=torch.stack(codebook_list)[:,:,0]
print(codebook_list.shape)
torch.save(codebook_list, "codebook.pt")


