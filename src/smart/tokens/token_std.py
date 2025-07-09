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


agent_token_data = pickle.load(open("my_kdist.pkl", "rb"))['token_all']

for type_id in [0,1,2]:#
    v=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")[:10000000]
    total_n=len(v)
    print(v.shape)

    k= ["veh", "ped", "cyc"][type_id]

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])

    gt_contour = cal_polygon_contour(
        pos=v[:, -1:, :2],  # [N, 6, 2]
        head=v[:, -1:, 2],  # [N, 6]
        width_length=width_length.unsqueeze(0).cuda(),
    )# [n_agent, 1, 4, 2]
    token_world_gt=torch.FloatTensor(agent_token_data[k]).to(torch.float16).cuda() #[1, n_token,4, 2]



    min_dist, token_idx_gt = torch.min(
        torch.norm(token_world_gt[None,:,-1] - gt_contour, dim=-1).sum(-1), dim=-1
    )  # [n_agent]

    print(token_idx_gt)




