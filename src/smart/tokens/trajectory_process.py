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
from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle

data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10_a91/"


traj_list=[]
type_list=[]
files = os.listdir(data_directory)


# #
for filename in tqdm(files[:100000]):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    agent=data['tokenized_agent']

    valid=agent['gt_valid_raw']
    all_valid=valid.all(-1)

    pos=agent['gt_pos_raw'][all_valid]
    head=agent['gt_head_raw'][all_valid]

    # traj=torch.cat([agent['gt_pos_raw'],agent['gt_head_raw'][:,:,None]],dim=2)
    rel_pos=pos[:,1:].reshape(-1,5,2)
    rel_head=head[:,1:].reshape(-1,5)

    local_pos, target_head = transform_to_local(
        pos_global=rel_pos,  # [n_agent*18, 1, 2]
        head_global=rel_head,  # [n_agent*18, 1]
        pos_now=pos[:,:-1:5].flatten(0,1),  # [n_agent*18, 2]
        head_now=head[:,:-1:5].flatten(0,1),  # [n_agent*18]
    )

    traj=torch.cat([local_pos, wrap_angle(target_head)[:,:,None]],dim=-1).reshape(-1,18,5,3)

    traj_list.append(traj)

    type_list.append(agent['type'][all_valid])
#
traj_list=torch.cat(traj_list)
type_list=torch.cat(type_list)

torch.save(traj_list,"/home/ke/code/catk/src/waymo_data/traj.pt")
torch.save(type_list,"/home/ke/code/catk/src/waymo_data/type.pt")


#
# traj=torch.load("/home/ke/code/catk/src/waymo_data/traj_rel.pt").cuda()
# type=torch.load("/home/ke/code/catk/src/waymo_data/type_rel.pt").cuda()
#
# print(veh_traj.size())#8559983
# print(ped_traj.size())#585226
# print(cyc_traj.size())#60093
# codebook_list=[]
#
# for type_id in [0,1,2]:
#     veh_traj=traj[type==type_id]
#
#     veh_vel=(veh_traj[:,1:,:2]-veh_traj[:,:-1,:2])/0.5
#
#     veh_speed=torch.norm(veh_vel,dim=-1)
#     veh_acc=(veh_speed[:,1:]-veh_speed[:,:-1])/0.5
#
#     # yaw
#     veh_rate=wrap_angle(veh_traj[:,1:,2]-veh_traj[:,:-1,2])/0.5#
#
#     # Flatten
#     acc_flat = veh_acc.flatten()
#     rate_flat = veh_rate[:,1:].flatten()
#
#     rate_flat=torch.cat([rate_flat,-rate_flat])
#     acc_flat=torch.cat([acc_flat,acc_flat])
#
#
#     #print(acc_flat.max(),acc_flat.min())
#     #print(rate_flat.max(),rate_flat.min())
#
#     # print((acc_flat>-5).to(torch.float).mean())
#     # print((acc_flat<5).to(torch.float).mean())
#     #
#     # print((rate_flat>-1.5).to(torch.float).mean())
#     # print((rate_flat<1.5).to(torch.float).mean())
#
#     # Clip to reasonable range (e.g., acceleration in [-10, 10], yaw rate in [-2π, 2π])
#     if type_id==0:
#         acc_min, acc_max = -10.0, 5.0
#         rate_min, rate_max = -1*np.pi ,1*np.pi # ~ -2π to 2π
#
#         # Number of bins
#         acc_bins = 500
#         rate_bins = 40
#         #Mean quantization error: 0.020819
# #Mean quantization error: 0.677057
#
#     if type_id == 1:
#         acc_min, acc_max = -10.0, 5.0
#         rate_min, rate_max = -1*np.pi ,1*np.pi # ~ -2π to 2π
#         acc_bins = 100
#         rate_bins = 200
#         #Mean quantization error: 0.029719
#
#     if type_id == 2:
#         acc_min, acc_max = -10.0, 5.0
#         rate_min, rate_max = -1*np.pi ,1*np.pi # ~ -2π to 2π
#         acc_bins = 100
#         rate_bins = 200
#         #Mean quantization error: 0.033888
#
#     # Digitize (map values to bin indices)
#     acc_idx = ((acc_flat - acc_min) / (acc_max - acc_min) * acc_bins).long()
#     rate_idx = ((rate_flat - rate_min) / (rate_max - rate_min) * rate_bins).long()
#
#     # Clamp to stay within bounds
#     acc_idx = acc_idx.clamp(0, acc_bins - 1)
#     rate_idx = rate_idx.clamp(0, rate_bins - 1)
#
#     # Joint index (flattened 2D bin index)
#     joint_idx = acc_idx * rate_bins + rate_idx
#
#     #joint_idx=joint_idx.cpu()
#
#     # Compute 2D histogram
#     joint_hist = torch.bincount(joint_idx, minlength=acc_bins * rate_bins)#.numpy()#.reshape(acc_bins, rate_bins)
#
#     # Get top 2048 bin indices (most frequent bins)
#     topk = 2048
#     topk_vals, topk_indices = torch.topk(joint_hist, topk)
#
#     # Recover corresponding (acc_bin, rate_bin) from flat joint index
#     acc_indices = topk_indices // rate_bins
#     rate_indices = topk_indices % rate_bins
#
#     # Convert bin indices back to acceleration and yaw rate values (center of bins)
#     acc_vals = acc_min + (acc_indices.to(torch.float) + 0.5) * (acc_max - acc_min) / acc_bins
#     rate_vals = rate_min + (rate_indices.to(torch.float) + 0.5) * (rate_max - rate_min) / rate_bins
#
#     print(torch.max(acc_vals),torch.min(acc_vals),torch.max(rate_vals),torch.min(rate_vals))
#
#     # Step 1: Build codebook
#     codebook = torch.stack([acc_vals, rate_vals], dim=1)  # shape: (2048, 2)
#
#     codebook_list.append(codebook)
#
#     # Step 2: Stack original values
#     orig_vals = torch.stack([acc_flat, rate_flat], dim=1) [:100000]# shape: (N, 2)
#
#     # Step 3: Compute L2 distance to each codebook entry
#     # Efficient pairwise distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
#     orig_norm = (orig_vals ** 2).sum(dim=1, keepdim=True)  # (N, 1)
#     codebook_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)  # (1, 2048)
#     dist_sq = orig_norm + codebook_norm - 2 * orig_vals @ codebook.T  # (N, 2048)
#
#     # Step 4: Find nearest codebook entry
#     nearest_idx = dist_sq.argmin(dim=1)  # (N,)
#     quantized_vals = codebook[nearest_idx]  # (N, 2)
#
#     # Step 5: Quantization error # (N,)
#     quant_error = torch.abs(orig_vals[:,0] - quantized_vals[:,0]).square()+torch.abs(orig_vals[:,1] - quantized_vals[:,1]).square()*10
#
#     mean_quant_error = quant_error.mean()
#
#     #print(f"Total quantization error to top-2048 centers: {total_quant_error.item():.4f}")
#     print(f"Mean quantization error: {mean_quant_error.item():.6f}")
#     #print(f"Max quantization error: {max_quant_error.item():.6f}")
#
#     #print((joint_hist[top_2048].sum())/(joint_hist.sum()))#(joint_hist>=top_2048).sum(),
#
#     # plt.imshow(joint_hist)
#     #
#     # plt.show()
#
#     #veh
#     # acc_min, acc_max = -10.0, 5.0
#     # rate_min, rate_max = -2*np.pi ,2*np.pi # ~ -2π to 2π
#
#
# codebook_list=torch.stack(codebook_list)
#
# torch.save(codebook_list, "codebook.pt")

# acc_min, acc_max = -5.0, 5.0
# rate_min, rate_max = -2*np.pi ,2*np.pi # ~ -2π to 2π
#100
#Mean quantization error: 0.076006

#200
#Mean quantization error: 0.039890

#500
#Mean quantization error: 0.024863

#1000
#Mean quantization error: 0.026678


# acc_min, acc_max = -10.0, 10.0
# rate_min, rate_max = -2*np.pi ,2*np.pi # ~ -2π to 2π
#Mean quantization error: 0.026824

# acc_min, acc_max = -10.0, 10.0
# rate_min, rate_max = -2*np.pi ,2*np.pi # ~ -2π to 2π
#Mean quantization error: 0.026824

# acc_min, acc_max = -10.0, 10.0
# rate_min, rate_max = -1*np.pi ,1*np.pi # ~ -2π to 2π
#Mean quantization error: 0.030930

#500   0.025279