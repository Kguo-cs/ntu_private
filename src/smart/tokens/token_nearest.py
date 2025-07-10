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


agent_token_data = pickle.load(open("/home/ke/code/catk/src/smart/tokens/my_kdist.pkl", "rb"))['token_all']

diff_list=[]

for type_id in [0,1,2]:#
    
    all_v=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")[:,-1:]
    
    
    k= ["veh", "ped", "cyc"][type_id]

    token_traj_all=torch.FloatTensor(agent_token_data[k]).to(torch.float16)[:,-1].cuda() #[1, n_token,4, 2]

    print(all_v.shape,token_traj_all.shape)

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])

    nearsest_token = []
        
    for i in range(2000):
        v= all_v[i*110000:(i+1)*110000].cuda()  # [1, 1, 4]

        gt_contour = cal_polygon_contour(
            pos=v[:, :, :2],  # [N, 1, 2]
            head=v[:, :, 2],  # [N, 1]
            width_length=width_length.unsqueeze(0).cuda(),
        )# [n_agent, 1, 4, 2]

        token_idx_gt = torch.argmin(
            torch.norm(token_traj_all[None] - gt_contour, dim=-1).sum(-1), dim=-1
        )  # [n_agent]
        
        nearsest_token.append(token_idx_gt)
        
        # print(i)
        
    nearsest_token = torch.cat(nearsest_token).cpu()
    
    print(nearsest_token.shape)
    
    torch.save(nearsest_token, "/home/ke/code/catk/src/waymo_data/nearest_token_"+str(type_id)+".pt")
    
    
    # pred_pos = token_traj_all.mean(2)
    # diff_xy = token_traj_all[:, :, 0] - token_traj_all[:, :, 3]
    # pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[ :, :, 0])

    # token_local_traj = torch.cat([pred_pos, pred_head[:, :,None]], dim=-1)[:,-1]


    # all_diff=[]

    # for i in range(2048):
    #     traj=v[token_idx_gt==i]

    #     diff=traj-token_local_traj[i,None]

    #     diff[:,:,2]=wrap_angle(diff[:,:,2])

    #     max_diff=diff.abs().amax(dim=0)

    #     all_diff.append(max_diff)

    # all_diff=torch.stack(all_diff)

    # diff_list.append(all_diff)

# diff=torch.stack(diff_list)
# torch.save(diff,"diff.pt")





