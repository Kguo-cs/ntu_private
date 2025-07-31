import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import math
import sys
from kinematic_compute import  kinematic_likelihood

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')
sys.path.append('/home/users/ntu/shanhelo/scratch/keguo_projects/sim')
sys.path.append('/mnt/d/code/sim')
from src.smart.utils import cal_polygon_contour



# traj=torch.load("/home/ke/code/catk/src/waymo_data/traj.pt")
# type=torch.load("/home/ke/code/catk/src/waymo_data/type.pt")

res = {"token_all": {}}
diff_list=[]
q = torch.tensor([0.001, 0.999]).cuda()
mid=torch.tensor([0.5]).cuda()

for type_id in [0,1,2]:#
    #veh_traj=traj[type==type_id]
    #veh_traj = veh_traj.reshape(-1, 5, 3)  # [N, 5, 2]

    veh_traj=torch.load("/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")

    # torch.save(veh_traj, "/home/ke/code/catk/src/waymo_data/"+str(type_id)+".pt")
    total_n=len(veh_traj)
    print(veh_traj.shape)

    # mask=veh_traj[...,1]<0
    #
    # veh_traj[...,2][mask]=-veh_traj[...,2][mask]
    # veh_traj[...,1][mask]=-veh_traj[...,1][mask]
    # veh_traj[...,1]=veh_traj[...,1].abs()
    # veh_traj[...,2]=veh_traj[...,2].abs()

    cluster_n=2048

    if type_id == 0:
        x_min, x_max = -5, 20
        y_max = 1.5
        x_interval = 0.1 #0.1->2048
        y_interval = 0.05
        heading_bin = 2  # number of heading bins
    elif type_id == 1:
        x_min, x_max = -1.5 , 4.5
        y_max=2
        x_interval = 0.05
        y_interval = 0.05
        heading_bin = 2  # number of heading bins
    elif type_id == 2:
        x_min, x_max = -3, 8
        y_max=1
        x_interval = 0.05
        y_interval = 0.05
        heading_bin = 2  # number of heading bins

    # cluster_n=4096
    #
    # if type_id == 0:
    #     x_min, x_max = -5, 20
    #     y_max = 1.5
    #     x_interval = 0.05 #0.1->2048
    #     y_interval = 0.05
    #     heading_bin = 2  # number of heading bins
    # elif type_id == 1:
    #     x_min, x_max = -1.5 , 4.5
    #     y_max=2
    #     x_interval = 0.05
    #     y_interval = 0.025
    #     heading_bin = 2  # number of heading bins
    # elif type_id == 2:
    #     x_min, x_max = -3, 8
    #     y_max=1
    #     x_interval = 0.05
    #     y_interval = 0.025
    #     heading_bin = 2  # number of heading bins


    y_max = y_max - y_interval/2
     
    y_min= -y_max

    final_pos = veh_traj[..., -1, :2]

    mask=(final_pos[:,0]>x_min) & (final_pos[:,0]<x_max) & (final_pos[:,1]<y_max) & (final_pos[:,1]>y_min)

    veh_traj=veh_traj[mask]
    #print(len(veh_traj_in)/len(veh_traj))

    final_pos = veh_traj[..., -1, :2]
    final_heading=veh_traj[..., -1, 2]

    x_bin= (x_max - x_min) / x_interval
    y_bin= (y_max - y_min) / y_interval

    x_bin = int(x_bin)
    y_bin = int(y_bin)

    x_idx= ((final_pos[:, 0] -x_min) /x_interval).long()
    y_idx= ((final_pos[:, 1]- y_min)/ y_interval).long()

    x_idx = x_idx.clamp(0, x_bin-1)
    y_idx = y_idx.clamp(0, y_bin-1)

    #joint_idx = x_idx * y_bin + y_idx

    heading_bin_size = 2 * math.pi / heading_bin

    heading_idx = ((final_heading + math.pi) / heading_bin_size).long().clamp(0, heading_bin - 1)
    joint_idx = x_idx * (y_bin * heading_bin) + y_idx * heading_bin + heading_idx

    joint_hist = torch.bincount(joint_idx, minlength=x_bin * y_bin * heading_bin)#.reshape(250, 30)

    # Top-k

    top_k_value, top_k_flat_idx = torch.topk(joint_hist, k=cluster_n)#.flatten()

    print(top_k_value.sum()/total_n,len(veh_traj)/total_n,top_k_value.sum()/len(veh_traj),top_k_value.min())#mask.to(torch.float).mean(),
#
    k= ["veh", "ped", "cyc"][type_id]

    if k == "veh":
        width_length = torch.tensor([2.0, 4.8]).cuda()
    elif k == "ped":
        width_length = torch.tensor([1.0, 1.0]).cuda()
    elif k == "cyc":
        width_length = torch.tensor([1.0, 2.0]).cuda()

    traj_list= []

    traj_diff=[]
    for i in range(cluster_n):
        idx = top_k_flat_idx[i]
        traj2 = veh_traj[joint_idx == idx][:10000000]#

        #mean_traj=torch.mean(traj2[:,-1,:2], dim=0)

        # dist=torch.norm(traj2[:,-1,:2]-mean_traj[None],dim=-1)#.mean(-1) #.argmin()

        # choice_index=torch.argmin(dist)

        # #choice_index = torch.randint(0, traj2.shape[0], (1,)).item()

        # meaning_traj=traj2[choice_index]
        #traj_q=torch.quantile(traj2.to(torch.float32), q, dim=0)

        #meaning_traj=traj_q.mean(dim=0)

        meaning_traj=torch.quantile(traj2.to(torch.float32),mid, dim=0)[0]

        # log_values=kinematic_likelihood(meaning_traj[:,:2].transpose(0,1),meaning_traj[:,2])
        #
        # sim_values=kinematic_likelihood(traj2[:,:,:2].permute(2,0,1),traj2[:,:,2])

        # log_likelihood = histogram_estimate(
        #     feature_config.histogram, log_values, sim_values)

        #meaning_traj= traj2.mean(dim=0) #.numpy()
        #max_diff=traj_q[1]-meaning_traj
        diff=traj2[:10000000]-meaning_traj[None]

        diff_q=torch.quantile(diff.to(torch.float32), q, dim=0)

        max_diff=diff_q.abs().amax(dim=0)

        #max_diff=diff.abs().amax(dim=0)#torch.minimum(diff.amax(dim=0), -diff.amin(dim=0))
        # diff[:,2]=wrap_angle(diff[:,2])
        #
        # max_diff = diff.abs().amax(dim=0)  # torch.minimum(diff.amax(dim=0), -diff.amin(dim=0))
        #
        # std_diff = diff.std(dim=0) * 5
        #
        # max_diff = torch.minimum(max_diff, std_diff)

        traj_diff.append(max_diff.cpu())

        traj_list.append(meaning_traj.cpu().to(torch.float32))

        # traj2=torch.cat([torch.zeros_like(traj2[:,:1]),traj2], dim=1)
        #
        # contour = cal_polygon_contour(
        #     pos=traj2[:, :, :2],  # [N, 6, 2]
        #     head=traj2[:, :, 2],  # [N, 6]
        #     width_length=width_length.unsqueeze(0),
        # )
        #
        # meaning_contour=contour.mean(dim=0)

        # contour = cal_polygon_contour(
        #     pos=meaning_traj[ :, :2],  # [N, 6, 2]
        #     head=meaning_traj[ :, 2],  # [N, 6]
        #     width_length=width_length.unsqueeze(0),
        # )
        #
        # print((meaning_contour-contour).max())

        #traj_list.append(meaning_contour.cpu())

    #     plt.plot(meaning_traj[:,0],meaning_traj[:,1])#, alpha=0.1, color='C0'
    #
    # plt.show()
    diff_list.append(torch.stack(traj_diff))
    codebook = torch.stack(traj_list, dim=0)

    # inverse_contour = traj_list.clone()#0.05 0.2 0.25
    # inverse_contour[:, :, 1] = -inverse_contour[:, :, 1]
    #
    # contour = torch.cat([traj_list, inverse_contour], dim=0)

    # inverse_traj = traj_list.clone()
    # inverse_traj[:, :, 1] = -inverse_traj[:, :, 1]
    # inverse_traj[:, :, 2] = -inverse_traj[:, :, 2]

    # codebook = torch.cat([traj_list, inverse_traj], dim=0)

    codebook=torch.cat([torch.zeros_like(codebook[:,:1]),codebook], dim=1)

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

res["max_diff"]=torch.stack(diff_list)

with open("mid2048_head2k.pkl", "wb") as f:
    pickle.dump(res, f)

