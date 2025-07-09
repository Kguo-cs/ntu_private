import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np
import matplotlib.pyplot as plt
# from src.smart.utils import  wrap_angle
import sys
sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')
sys.path.append('/home/users/ntu/shanhelo/scratch/keguo_projects/sim')
sys.path.append('/mnt/d/code/sim')
from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle






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

res = {"token_all": {}}
tol_dist = [0.05, 0.05, 0.05]  # veh, ped, cyc
num_cluster=2048
torch.manual_seed(2)

for type_id in [0,1,2]:#
    #veh_traj=traj[type==type_id]
    #veh_traj = veh_traj.reshape(-1, 5, 3)  # [N, 5, 2]

    v=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")[:100000000]

    # torch.save(veh_traj, "/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")
    total_n=len(v)
    print(v.shape)

    k= ["veh", "ped", "cyc"][type_id]

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])
        
        
    contour = cal_polygon_contour(
        pos=v[:, -1, :2], head=v[:, -1, 2], width_length=width_length.cuda()
    )  # [n_trajs, 4, 2]



    if k == "veh":
        tol = tol_dist[0]
    elif k == "ped":
        tol = tol_dist[1]
    elif k == "cyc":
        tol = tol_dist[2]
    print(k, tol)
    ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol, a_pos=v)
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    ret_traj=torch.cat([torch.zeros_like(ret_traj[:,:1]),ret_traj], dim=1)

    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],  # [N, 6, 2]
        head=ret_traj[:, :, 2],  # [N, 6]
        width_length=width_length.unsqueeze(0).cuda(),
    )
    res["token_all"][k] = contour.cpu().numpy()

with open("my_kdist.pkl", "wb") as f:
    pickle.dump(res, f)


# torch.Size([100000000, 5, 3])
# veh 0.05
# i=2047, remain=2.88%, n_inside=4607
# torch.Size([16348428, 5, 3])
# ped 0.05
# i=2047, remain=2.39%, n_inside=88
# torch.Size([1590937, 5, 3])
# cyc 0.05
# i=2047, remain=2.80%, n_inside=5