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



# traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt")
# type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt")

res = {"token_all": {}}

for type_id in [0,1,2]:#
    #veh_traj=traj[type==type_id]
    #veh_traj = veh_traj.reshape(-1, 5, 3)  # [N, 5, 2]

    veh_traj=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")

    # torch.save(veh_traj, "/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")
    total_n=len(veh_traj)
    print(veh_traj.shape)

    mask=veh_traj[...,1]<0

    veh_traj[...,2][mask]=-veh_traj[...,2][mask]
    veh_traj[...,1][mask]=-veh_traj[...,1][mask]

    if type_id == 0:
        x_min, x_max = -5, 20
        y_max = 1.5
        x_interval = 0.1
        y_interval = 0.05
    elif type_id == 1:
        x_min, x_max = -1.5 , 4.5
        y_max=2
        x_interval = 0.05
        y_interval = 0.05
    elif type_id == 2:
        x_min, x_max = -3, 8
        y_max=1
        x_interval = 0.05
        y_interval = 0.05

    final_pos = veh_traj[..., -1, :2]

    mask=(final_pos[:,0]>x_min) & (final_pos[:,0]<x_max) & (final_pos[:,1]<y_max)

    veh_traj=veh_traj[mask]
    #print(len(veh_traj_in)/len(veh_traj))

    final_pos = veh_traj[..., -1, :2]

    x_bin=(x_max - x_min) / x_interval
    y_bin= y_max / y_interval

    x_bin = int(x_bin)
    y_bin = int(y_bin)

    x_idx= ((final_pos[:, 0] -x_min) /x_interval).long()
    y_idx= (final_pos[:, 1]/ y_interval).long()

    x_idx = x_idx.clamp(0, x_bin-1)
    y_idx = y_idx.clamp(0, y_bin-1)

    joint_idx = x_idx * y_bin + y_idx

    joint_hist = torch.bincount(joint_idx, minlength=x_bin * y_bin)#.reshape(250, 30)

    # Top-k
    cluster_n=1024

    top_k_value, top_k_flat_idx = torch.topk(joint_hist, k=cluster_n)#.flatten()

    print(top_k_value.sum()/total_n,len(veh_traj)/total_n,top_k_value.sum()/len(veh_traj),top_k_value.min())#mask.to(torch.float).mean(),
#
    k= ["veh", "ped", "cyc"][type_id]

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])

    traj_list= []
    for i in range(cluster_n):
        idx = top_k_flat_idx[i]
        traj2 = veh_traj[joint_idx == idx]

        #meaning_traj= traj2.mean(dim=0).cpu() #.numpy()
        traj2=torch.cat([torch.zeros_like(traj2[:,:1]),traj2], dim=1)

        contour = cal_polygon_contour(
            pos=traj2[:, :, :2],  # [N, 6, 2]
            head=traj2[:, :, 2],  # [N, 6]
            width_length=width_length.unsqueeze(0),
        )

        mean_contour=contour.mean(dim=0).cpu()

        traj_list.append(mean_contour)

    #     plt.plot(meaning_traj[:,0],meaning_traj[:,1])#, alpha=0.1, color='C0'
    #
    # plt.show()
    contour = torch.stack(contour, dim=0)

    res["token_all"][k] = contour.numpy()


#
#     inverse_traj = traj_list.clone()
#     inverse_traj[:, :, 1] = -inverse_traj[:, :, 1]
#     inverse_traj[:, :, 2] = -inverse_traj[:, :, 2]
#
#     codebook = torch.cat([traj_list, inverse_traj], dim=0)
#
#     codebook=torch.cat([torch.zeros_like(codebook[:,:1]),codebook], dim=1)



with open("my2048.pkl", "wb") as f:
    pickle.dump(res, f)


# torch.Size([154079694, 5, 3])
# 0.9999526608613333 tensor(0.9980, device='cuda:0') tensor(1118, device='cuda:0')
# torch.Size([10534068, 5, 3])
# 0.9997874515334437 tensor(0.9998, device='cuda:0') tensor(9, device='cuda:0')
# torch.Size([1081674, 5, 3])
# 0.993570151450437 tensor(0.9982, device='cuda:0') tensor(6, device='cuda:0')

# torch.Size([154079694, 5, 3])
# 0.9999526608613333 tensor(0.9980, device='cuda:0') tensor(1118, device='cuda:0')
# torch.Size([10534068, 5, 3])
# 0.9997874515334437 tensor(0.9998, device='cuda:0') tensor(9, device='cuda:0')
# torch.Size([1081674, 5, 3])
# 0.9993232711519368 tensor(0.9932, device='cuda:0') tensor(10, device='cuda:0')

# torch.Size([154079694, 5, 3])
# 0.9999526608613333 tensor(0.9980, device='cuda:0') tensor(1118, device='cuda:0')
# torch.Size([10534068, 5, 3])
# 0.9999438963181175 tensor(0.9997, device='cuda:0') tensor(10, device='cuda:0')
# torch.Size([1081674, 5, 3])
# 0.9993232711519368 tensor(0.9932, device='cuda:0') tensor(10, device='cuda:0')


#
# torch.Size([215685024, 5, 3])
# tensor(0.9964, device='cuda:0') 0.9998906507296492 tensor(0.9966, device='cuda:0') tensor(2380, device='cuda:0')
# torch.Size([16594332, 5, 3])
# tensor(0.9989, device='cuda:0') 0.999764979994374 tensor(0.9991, device='cuda:0') tensor(36, device='cuda:0')
# torch.Size([1608651, 5, 3])
# tensor(0.9901, device='cuda:0') 0.9964125220448686 tensor(0.9937, device='cuda:0') tensor(18, device='cuda:0')

#2048
# torch.Size([215685024, 5, 3])
# tensor(0.9995, device='cuda:0') 0.9998906507296492 tensor(0.9996, device='cuda:0') tensor(113, device='cuda:0')
# torch.Size([16594332, 5, 3])
# tensor(0.9996, device='cuda:0') 0.999764979994374 tensor(0.9998, device='cuda:0') tensor(4, device='cuda:0')
# torch.Size([1608651, 5, 3])
# tensor(0.9954, device='cuda:0') 0.9964125220448686 tensor(0.9990, device='cuda:0') tensor(4, device='cuda: