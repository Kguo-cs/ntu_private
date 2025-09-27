import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from multiprocessing import Pool, cpu_count


data_directory = "./waymo_data/full/" #training_map2_03_pred/"
output_path = "./waymo_data/full/training_map2_03_route40/"



files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)


import numpy as np

def interpolate_traj_lookahead(points, step=2.0, lookahead=40.0, heading_last=None, eps=1e-9):
    """
    Resample a 2D polyline at fixed spacing, then extend forward by `lookahead` meters
    along the last heading.

    Args:
        points: [N,2] original xy waypoints (numpy or torch -> numpy)
        step:   spacing in meters between samples (e.g., 2.0)
        lookahead: extra distance to extend beyond the last point
        heading_last: optional yaw (radians) for the final heading; if None, infer from last segment

    Returns:
        Q: [M,2] resampled + extended points
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be [N,2]")

    # handle degenerate cases
    if len(P) == 0:
        return P.copy()
    if len(P) == 1:
        # only one point: use heading_last for extension
        if heading_last is None:
            heading_last = 0.0
        dir_unit = np.array([np.cos(heading_last), np.sin(heading_last)], dtype=float)
        t = np.arange(0.0, lookahead + 1e-9, step)
        return P[0] + t[:, None] * dir_unit

    # cumulative arc length along the polyline
    seg = P[1:] - P[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    nonzero = seglen > eps
    if not np.any(nonzero):
        # all points identical → same as single-point case
        if heading_last is None:
            heading_last = 0.0
        dir_unit = np.array([np.cos(heading_last), np.sin(heading_last)], dtype=float)
        t = np.arange(0.0, lookahead + 1e-9, step)
        return P[:1] + t[:, None] * dir_unit

    # keep only nonzero-length segments to avoid numerical issues
    P = P[np.r_[True, nonzero]]
    seg = P[1:] - P[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]

    # resample along the original polyline
    s_new = np.arange(0.0, total + 1e-9, step)
    x = np.interp(s_new, s, P[:, 0])
    y = np.interp(s_new, s, P[:, 1])
    Q = np.stack([x, y], axis=1)

    # determine last heading
    if heading_last is None:
        # use last nonzero segment direction
        last_vec = seg[-1]
        n = np.linalg.norm(last_vec)
        if n < eps:
            heading_last = 0.0
        else:
            heading_last = float(np.arctan2(last_vec[1], last_vec[0]))

    dir_unit = np.array([np.cos(heading_last), np.sin(heading_last)], dtype=float)

    # extend forward for `lookahead` meters (avoid duplicating the last point)
    t_ext = np.arange(step, lookahead + 1e-9, step)
    tail = Q[-1] + t_ext[:, None] * dir_unit

    return np.vstack([Q, tail]) if len(t_ext) > 0 else Q

from scipy.spatial import cKDTree

def compute_yaw_from_traj(traj_xy: np.ndarray, heading_hint: float | None = None, eps=1e-9) -> np.ndarray:
    """
    Returns yaw per point. If only one point, uses heading_hint (radians) if provided,
    else 0.0.
    """
    M = len(traj_xy)
    yaw = np.zeros(M, dtype=np.float32)
    if M == 0:
        return yaw
    if M == 1:
        if heading_hint is not None and np.isfinite(heading_hint):
            yaw[0] = float(heading_hint)
        else:
            yaw[0] = 0.0
        return yaw

    d = np.gradient(traj_xy, axis=0)
    m = np.hypot(d[:,0], d[:,1]) > eps
    yaw[m] = np.arctan2(d[m,1], d[m,0])
    if not np.all(m):
        idx = np.where(m)[0]
        if len(idx) > 0:
            fill_from = np.clip(np.searchsorted(idx, np.where(~m)[0]), 0, len(idx)-1)
            yaw[~m] = yaw[idx[fill_from]]
    return yaw

def nearest_edges_biside(traj_xy: np.ndarray, yaw: np.ndarray, edge_xy: np.ndarray,
                         k: int = 16, radius: float = 15.0):
    """
    For each traj point, find nearest edge point on LEFT/RIGHT side given yaw.
    Works even if traj_xy has length 1.
    """
    tree = cKDTree(edge_xy)
    dists, idxs = tree.query(traj_xy, k=k, distance_upper_bound=radius)
    if k == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    E = len(edge_xy)
    cosh, sinh = np.cos(yaw), np.sin(yaw)
    n_left  = np.stack([-sinh,  cosh], axis=1)
    n_right = np.stack([ sinh, -cosh], axis=1)

    cand = np.zeros((len(traj_xy), k, 2), dtype=np.float32)
    for j in range(k):
        valid = idxs[:, j] < E
        cand[valid, j, :] = edge_xy[idxs[valid, j]] - traj_xy[valid]

    dot_left  = (cand * n_left[:, None, :]).sum(axis=2)
    dot_right = (cand * n_right[:, None, :]).sum(axis=2)

    big = np.inf
    left_cost  = np.where((dot_left  > 0) & (idxs < E), dists, big)
    right_cost = np.where((dot_right > 0) & (idxs < E), dists, big)

    li = np.argmin(left_cost,  axis=1)
    ri = np.argmin(right_cost, axis=1)
    Ld = left_cost[np.arange(len(traj_xy)), li]
    Rd = right_cost[np.arange(len(traj_xy)), ri]
    Li = idxs[np.arange(len(traj_xy)), li]
    Ri = idxs[np.arange(len(traj_xy)), ri]

    Li[Ld==big] = -1; Ri[Rd==big] = -1
    Ld[Ld==big] = np.inf; Rd[Rd==big] = np.inf
    return Li, Ri, Ld, Rd



max_len=0

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)

    if "training_map2_03_route40" in filename:

        # with open(input_path, "rb") as f:
        #     data = pickle.load(f)

        new_name=output_path+filename[24:]

        os.rename(input_path, new_name)

    # tokenized_map=data["tokenized_map"]
    #
    # map_type=tokenized_map['type']
    # #mask4 = (map_type == 4)
    # mask45 = (map_type == 4) | (map_type == 5)
    # #
    # # idx4 = mask4.nonzero(as_tuple=True)[0]
    # # idx45 = mask45.nonzero(as_tuple=True)[0]
    #
    # # map idx4 into local indices inside idx45
    # # torch.searchsorted requires sorted input (idx45 is sorted by construction)
    # #idx4_in_45 = torch.searchsorted(idx45, idx4)
    # position=tokenized_map["position"][mask45]
    # x, y = position[:, 0], position[:, 1]
    #
    # edge_xy = np.column_stack([x, y])  # road-edge points
    #
    # tokenized_agent=data["tokenized_agent"]
    #
    # sampled_pos=tokenized_agent["sampled_pos"][:,1:]
    # sampled_heading=tokenized_agent["sampled_heading"][:,1:]
    #
    # valid_mask=tokenized_agent['valid_mask'][:,1:]
    # #route_map_index = build_route_map_index(sampled_pos, sampled_heading, valid_mask, edge_xy)
    #
    # route_map_index=torch.zeros([len(valid_mask),100]).to(torch.int16)-1
    #
    #
    # for i in range(len(sampled_pos)):
    #     agent = sampled_pos[i]
    #     heading = sampled_heading[i]  # radians, same length as agent
    #     valid = valid_mask[i]
    #
    #     valid_traj = agent[valid].numpy()
    #
    #
    #     # heading hint: last valid heading (or first)
    #     heading_hint = float(heading[valid][-1].item())
    #
    #     interpolated_traj = interpolate_traj_lookahead(
    #         valid_traj, step=2.0, lookahead=40.0, heading_last=heading_hint
    #     )
    #
    #     yaw_interp = compute_yaw_from_traj(interpolated_traj, heading_hint=heading_hint)
    #
    #     L_idx, R_idx, L_d, R_d = nearest_edges_biside(
    #         interpolated_traj, yaw_interp, edge_xy, k=16, radius=40.0
    #     )
    #
    #     all_idx=torch.tensor(np.unique(np.concatenate([L_idx,R_idx])))#idx4_in_45[np.unique(np.concatenate([L_idx,R_idx]))]
    #     n = min(len(all_idx), 100)
    #
    #     route_map_index[i][:n] =all_idx[:n]
    #
    # #     max_len=max(max_len, len(all_idx))
    # #
    # # if max_len>120:
    # #     print(max_len)
    #
    # data["tokenized_agent"]["route_map_index"]=route_map_index


        # print(len(np.unique(L_idx)), len(np.unique(R_idx))  )

        # plt.plot(valid_traj[:,0], valid_traj[:,1], 'g-')
        # plt.plot(interpolated_traj[:,0], interpolated_traj[:,1], 'r-')
        # for p, li, ri in zip(interpolated_traj, L_idx, R_idx):
        #     if li >= 0: plt.plot([p[0], edge_xy[li,0]], [p[1], edge_xy[li,1]], 'b-', alpha=0.4)
        #     if ri >= 0: plt.plot([p[0], edge_xy[ri,0]], [p[1], edge_xy[ri,1]], 'm-', alpha=0.4)
        #
        # plt.scatter(x, y)
        #
        # plt.show()

        # output_file = output_path + filename
        #
        # with open(output_file, "wb") as f:
        #     pickle.dump(data, f)
    #
    #

