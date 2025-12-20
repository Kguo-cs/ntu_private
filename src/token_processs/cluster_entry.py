import torch
import pickle
# from src.smart.utils import (
#     cal_polygon_contour,
#     transform_to_global,
#     transform_to_local,
#     wrap_angle,
# )



def Kdisk_cluster(
    X,  # [n_trajs, 5, 3], bbox of the last point of the segment
    N,  # int
    tol,  # float
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        choice_index = torch.randint(0, X.shape[0], (1,)).item()
        x0 = X[choice_index]
        # res_mask = torch.sum((X - x0) ** 2, dim=[1, 2]) / 4.0 > (tol**2)
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        ret_traj = X[~res_mask].mean(0, keepdim=False)
        X = X[res_mask]

        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()

        print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    return torch.stack(ret_traj_list, dim=0)  # [N, 6, 3]

# results=torch.load("/home/ke/code/catk/src/waymo_data/token/first_pose.pt")[:,None,:2]
#
#
# trajs=torch.zeros([1,1, 2], dtype=torch.float32)
# for i in range(len(results)):
#     to_add=results[i][None]
#     if not ( (  (trajs[:,:,:2] - to_add[:,:,:2]).abs().sum(-1).sum(-1) < 0.07 ).any()):
#         trajs = torch.cat( [trajs, to_add], dim=0  )
#
#         print(i,len(trajs),len(trajs)/(i+1),i/len(results))
#         if len(trajs) > 2048 * 100:
#             break
#
# torch.save(trajs,"/home/ke/code/catk/src/waymo_data/token/first_pose1.pt")
# 678649 204801 0.3017770573933545 0.5107997058560094

trajs=torch.load("/home/ke/code/catk/src/waymo_data/token/first_pose1.pt")[1:]

# tokenize_center = Kdisk_cluster(X=trajs, N=1024, tol=2.5)
tokenize_center = Kdisk_cluster(X=trajs, N=256, tol=4.5)

tokenize_center=tokenize_center[:,0]

with open("entry256.pkl", "wb") as f:
    pickle.dump(tokenize_center, f)

# i=511, remain=5.46%, n_inside=48
