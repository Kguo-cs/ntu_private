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




traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt")#.cuda()
type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt")#.cuda()

res = {"token_all": {}}

for type_id in [0,1,2]:
    veh_traj=traj[type==type_id]
    veh_traj = veh_traj.reshape(-1, 5, 3)  # [N, 5, 2]

    print(veh_traj.shape)
    veh_traj[...,1]=veh_traj[...,1].abs()
    veh_traj[...,2]=veh_traj[...,2].abs()


    final_pos = veh_traj[..., -1, :2]
    
    if type_id == 0:
        x_min, x_max = -5, 20
        y_max=1.5
        x_interval = 0.1
        y_interval = 0.05
    elif type_id == 1:
        x_min, x_max = -1,8
        y_max=1
        x_interval = 0.05
        y_interval = 0.05
    elif type_id == 2:
        x_min, x_max = -1.5, 4.5
        y_max=2
        x_interval = 0.05
        y_interval = 0.05
    
    x_bin=(x_max - x_min) / x_interval
    y_bin= y_max / y_interval
    
    x_bin = int(x_bin)
    y_bin = int(y_bin)
    
    x_idx= ((final_pos[:, 0] -x_min) /x_interval).long()
    y_idx= (final_pos[:, 1]/ y_interval).long()
    
    x_idx = x_idx.clamp(0, x_bin-1)
    y_idx = y_idx.clamp(0, y_bin-1)

    joint_idx = x_idx * y_bin + y_idx

    joint_hist = torch.bincount(joint_idx, minlength=250 * 30).reshape(250, 30)
        
    # Top-k
    top_k_value, top_k_flat_idx = torch.topk(joint_hist.flatten(), k=1024)
    
    traj_list= []
    for i in range(1024):
        idx = top_k_flat_idx[i]
        traj2= veh_traj[joint_idx == idx]
        
        meaning_traj= traj2.mean(dim=0)
        
        traj_list.append(meaning_traj)
        
        plt.plot(meaning_traj[:,0],meaning_traj[:,1])#, alpha=0.1, color='C0'

        
    traj_list = torch.stack(traj_list, dim=0)
    
    inverse_traj = traj_list.clone()
    inverse_traj[:, :, 1] = -inverse_traj[:, :, 1]
    inverse_traj[:, :, 2] = -inverse_traj[:, :, 2]
    
    codebook = torch.cat([traj_list, inverse_traj], dim=0)
    
    
    k= ["veh", "ped", "cyc"][type_id]
    
    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])

    contour = cal_polygon_contour(
        pos=codebook[:, :, :2],  # [N, 6, 2]
        head=codebook[:, :, 2],  # [N, 6]
        width_length=width_length.unsqueeze(0),
    )
    res["token_all"][k] = contour.numpy()

            
with open("my2048.pkl", "wb") as f:
    pickle.dump(res, f)

        # print(len(final_pos[final_pos[:, 1]== 0]))
    
    # final_pos=final_pos[final_pos[:, 1] > 0]
        #mask= joint_hist.cpu().numpy()
    # joint_hist=np.log(joint_hist + 1)

    # print((joint_hist>60).sum())
    
    #plt.imshow((joint_hist>60).T)
    # sys_pos=final_pos.clone()
    
    # sys_pos[:,0]=final_pos[:,0]
    # sys_pos[:,1]=-final_pos[:,1]
    

#     acc_idx = ((acc_flat - acc_min) / (acc_max - acc_min) * acc_bins).long()
#     rate_idx = ((rate_flat - rate_min) / (rate_max - rate_min) * rate_bins).long()
    
    
    
    # final_pos = torch.cat([sys_pos, final_pos], dim=0)
        
    # plt.scatter(final_pos[:, 0], final_pos[:, 1], s=1, label=f"Type {type_id}")

    # plt.xlim(-5, 20)
    # plt.ylim(-12.5, 12.5)

    # plt.savefig('/home/ke/code/catk/src/waymo_data/final_pos.png')
    
    # break
    
# torch.save(codebook_list, "codebook.pt")
