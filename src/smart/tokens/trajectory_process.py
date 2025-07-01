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
#
# data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10_a91/"
#
#
# traj_list=[]
# type_list=[]
# files = os.listdir(data_directory)
#
#
#
# for filename in tqdm(files):
#     input_path = os.path.join(data_directory, filename)
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#
#     agent=data['tokenized_agent']
#
#     valid=agent['gt_valid_raw']
#     all_valid=valid.all(-1)
#
#     traj=torch.cat([agent['gt_pos_raw'],agent['gt_head_raw'][:,:,None] ],dim=2)[all_valid]
#
#     rel_traj=traj[:,1:]-traj[:,:1]
#
#
#     traj_list.append(rel_traj.to(torch.float16))
#
#     type_list.append(agent['type'][all_valid])
#
#
# traj_list=torch.cat(traj_list)
# type_list=torch.cat(type_list)
#
# torch.save(traj_list,"traj.pt")
# torch.save(type_list,"type.pt")



traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt").cuda()
type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt").cuda()

# ped_traj=traj[type==1]
# cyc_traj=traj[type==2]
#
# print(veh_traj.size())#8559983
# print(ped_traj.size())#585226
# print(cyc_traj.size())#60093

for type_id in [0,1,2]:
    veh_traj=traj[type==type_id][:,::5]

    veh_vel=(veh_traj[:,1:,:2]-veh_traj[:,:-1,:2])/0.5

    veh_speed=torch.norm(veh_vel,dim=-1)
    veh_acc=(veh_speed[:,1:]-veh_speed[:,:-1])/0.5

    # print((veh_acc>-5).to(torch.float).mean())
    # print((veh_acc<5).to(torch.float).mean())

    # yaw
    veh_rate=wrap_angle(veh_traj[:,1:,2]-veh_traj[:,:-1,2])/0.5

    # Flatten
    acc_flat = veh_acc.flatten()
    rate_flat = veh_rate[:,1:].flatten()

    print(rate_flat.max())
    print(rate_flat.min())

    # Clip to reasonable range (e.g., acceleration in [-10, 10], yaw rate in [-2π, 2π])
    acc_min, acc_max = -5.0, 5.0
    rate_min, rate_max = -2*np.pi ,2*np.pi # ~ -2π to 2π

    # Number of bins
    acc_bins = 40
    rate_bins = 40

    # Digitize (map values to bin indices)
    acc_idx = ((acc_flat - acc_min) / (acc_max - acc_min) * acc_bins).long()
    rate_idx = ((rate_flat - rate_min) / (rate_max - rate_min) * rate_bins).long()

    # Clamp to stay within bounds
    acc_idx = acc_idx.clamp(0, acc_bins - 1)
    rate_idx = rate_idx.clamp(0, rate_bins - 1)

    # Joint index (flattened 2D bin index)
    joint_idx = acc_idx * rate_bins + rate_idx

    joint_idx=joint_idx.cpu()

    # Compute 2D histogram
    joint_hist = torch.bincount(joint_idx, minlength=acc_bins * rate_bins).reshape(acc_bins, rate_bins)

    log_hist = torch.log(joint_hist+1)

    plt.imshow(log_hist)

    plt.show()
# tensor(0.9996, device='cuda:0')
# tensor(0.9996, device='cuda:0')
# tensor(0.9999, device='cuda:0')
# tensor(0.9999, device='cuda:0')
# tensor(1.0000, device='cuda:0')
# tensor(1.0000, device='cuda:0')
# tensor(0.9909, device='cuda:0')
# tensor(0.9906, device='cuda:0')
# tensor(0.9999, device='cuda:0')
# tensor(0.9999, device='cuda:0')
# tensor(0.9978, device='cuda:0')
# tensor(0.9978, device='cuda:0')

#yaw rate

#-1 tensor(0.9999, device='cuda:0')
#1 tensor(0.9999, device='cuda:0')

# veh_speed_99 = torch.quantile(veh_speed.to(torch.float32).flatten(), 0.99)
# print(f"99th percentile of vehicle speed: {veh_speed_99.item():.2f} m/s")


# # veh_acc=(veh_speed[:,1:]-veh_speed[:,:-1])/0.1
# bincount=torch.bincount(veh_speed.to(torch.int32).flatten())
#
# # Prepare x and y
# x = torch.arange(len(bincount)).numpy()
# y = bincount.numpy()
# # Plot
# plt.figure(figsize=(10, 5))
# plt.bar(x, y)
# plt.yscale('log')  # <-- Log scale here
# plt.xlabel("Speed (m/s, integer)")
# plt.ylabel("Count (log scale)")
# plt.title("Vehicle Speed Distribution (Log Scale)")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()

#print("max_acc",veh_acc.max())
#print("min_acc",veh_acc.min())

#print(veh_vel.())
#print(veh_acc.size())