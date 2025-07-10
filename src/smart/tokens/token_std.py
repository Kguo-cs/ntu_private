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


res = pickle.load(open("/home/ke/code/catk/src/smart/tokens/my_kdist.pkl", "rb"))
agent_token_data=res['token_all']   
diff_list=[]

for type_id in [0,1,2]:#
    
    all_v=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt").cuda() #[:,-1:]
    all_idx= torch.load("/home/ke/code/catk/src/waymo_data/nearest_token_"+str(type_id)+".pt").cuda() #[:,-1:]
    
    k= ["veh", "ped", "cyc"][type_id]

    token_traj_all=torch.FloatTensor(agent_token_data[k]).to(torch.float16)[:,1:].cuda() #[1, n_token,4, 2]

    print(all_v.shape,token_traj_all.shape)

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8])
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0])
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0])

    pred_pos = token_traj_all.mean(2)
    diff_xy = token_traj_all[:, :, 0] - token_traj_all[:, :, 3]
    pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[ :, :, 0])

    token_local_traj = torch.cat([pred_pos, pred_head[:, :,None]], dim=-1)
    
    traj_diff=[]
    
    for i in range(len(token_local_traj)):
        
        traj2 = all_v[all_idx==i]#.cuda()
        
        meaning_traj= token_local_traj[i]#traj2.mean(dim=0) #.numpy()

        diff=traj2-meaning_traj[None]

        diff[:,2]=wrap_angle(diff[:,2])

        max_diff=diff.abs().amax(dim=0)#torch.minimum(diff.amax(dim=0), -diff.amin(dim=0))

        std_diff = diff.std(dim=0)*4
        
        max_diff = torch.minimum(max_diff, std_diff)
        
        traj_diff.append(max_diff.cpu())

    diff_list.append(torch.stack(traj_diff))

res["max_diff"]=torch.stack(diff_list)
with open("my_kdist.pkl", "wb") as f:
    pickle.dump(res, f)

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





