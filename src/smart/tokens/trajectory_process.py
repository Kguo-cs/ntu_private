import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np

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



traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt")
type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt")

veh_traj=traj[type==0]
ped_traj=traj[type==1]
cyc_traj=traj[type==2]

print(veh_traj.size())#8559983
print(ped_traj.size())#585226
print(cyc_traj.size())#60093
