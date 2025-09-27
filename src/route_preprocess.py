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

data_directory = "./waymo_data/full/training_map2_03_pred/"
output_path = "./waymo_data/full/training_map2_03_route/"



files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)


def interpolate_traj(points, step=0.5):
    """
    Interpolate a 2D trajectory at fixed arc-length intervals.

    Args:
        points: array-like [N,2] of x,y waypoints
        step: spacing in meters (default 0.5)

    Returns:
        np.ndarray [M,2] interpolated points
    """
    line = LineString(points)
    length = line.length
    num = int(np.floor(length / step)) + 1
    dists = np.linspace(0, length, num=num)
    return np.array([line.interpolate(d).coords[0] for d in dists])

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



# ---- Per-agent worker ----
def process_agent(agent, heading, valid, edge_xy, step=0.5, k=16, radius=40.0, max_cap=100):
    if not bool(valid.any()):
        return np.full(max_cap, -1, dtype=np.int16)

    valid_traj = agent[valid].cpu().numpy()
    inter = interpolate_traj(valid_traj, step=step)
    hint = float(heading[valid][-1].item()) if valid.sum() > 0 else None
    yaw = compute_yaw_from_traj(inter, heading_hint=hint)

    L_idx, R_idx, L_d, R_d = nearest_edges_biside(inter, yaw, edge_xy, k=k, radius=radius)
    all_idx = np.unique(np.concatenate([L_idx, R_idx]))

    out = np.full(max_cap, -1, dtype=np.int16)
    n = min(len(all_idx), max_cap)
    out[:n] = all_idx[:n]
    return out

# ---- Parallel driver ----
def build_route_map_index(sampled_pos, sampled_heading, valid_mask, edge_xy,
                          step=0.5, k=16, radius=40.0, max_cap=120):
    args = [
        (sampled_pos[i], sampled_heading[i], valid_mask[i], edge_xy, step, k, radius, max_cap)
        for i in range(len(sampled_pos))
    ]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_agent, args)

    return torch.tensor(np.stack(results), dtype=torch.int16)



max_len=0

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    tokenized_map=data["tokenized_map"]

    map_type=tokenized_map['type']
    mask = (map_type == 4) | (map_type == 5)
    position=tokenized_map["position"][mask]
    x, y = position[:, 0], position[:, 1]

    edge_xy = np.column_stack([x, y])  # road-edge points

    tokenized_agent=data["tokenized_agent"]

    sampled_pos=tokenized_agent["sampled_pos"]
    sampled_heading=tokenized_agent["sampled_heading"]

    valid_mask=tokenized_agent['valid_mask']
    #route_map_index = build_route_map_index(sampled_pos, sampled_heading, valid_mask, edge_xy)

    route_map_index=torch.zeros([len(valid_mask),100]).to(torch.int16)-1


    for i in range(len(sampled_pos)):
        agent = sampled_pos[i]
        heading = sampled_heading[i]  # radians, same length as agent
        valid = valid_mask[i]

        valid_traj = agent[valid].numpy()

        interpolated_traj = interpolate_traj(valid_traj, step=2)  # your function

        # heading hint: last valid heading (or first)
        heading_hint = float(heading[valid][-1].item())

        yaw_interp = compute_yaw_from_traj(interpolated_traj, heading_hint=heading_hint)

        L_idx, R_idx, L_d, R_d = nearest_edges_biside(
            interpolated_traj, yaw_interp, edge_xy, k=16, radius=40.0
        )

        all_idx=np.unique(np.concatenate([L_idx,R_idx]))
        n = min(len(all_idx), 100)

        route_map_index[i][:n] =torch.tensor(all_idx)[:n]

    #     max_len=max(max_len, len(all_idx))
    #
    # if max_len>120:
    #     print(max_len)

    data["tokenized_agent"]["route_map_index"]=route_map_index


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

    output_file = output_path + filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    #
    #

