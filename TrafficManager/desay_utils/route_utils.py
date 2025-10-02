from scipy.spatial import cKDTree
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



import numpy as np

def _arclen2d(xy: np.ndarray) -> np.ndarray:
    """累积弧长"""
    if len(xy) < 2:
        return np.array([0.0])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def _interp_at_s(xy: np.ndarray, s_arr: np.ndarray, s: float) -> np.ndarray:
    """在弧长坐标 s 上插值点坐标"""
    x = np.interp(s, s_arr, xy[:,0])
    y = np.interp(s, s_arr, xy[:,1])
    return np.array([x, y], float)

def resample_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    """对一条 polyline 按 step(m) 均匀重采样"""
    s = _arclen2d(xy)
    L = s[-1]
    if L < 1e-6:
        return xy[:1]
    s_new = np.arange(0.0, L, step)
    if s_new[-1] < L:
        s_new = np.append(s_new, L)
    return np.stack([_interp_at_s(xy, s, u) for u in s_new])

def append_segment_with_step(start: np.ndarray, route: np.ndarray, goal: np.ndarray, step: float=2.0) -> np.ndarray:
    """
    将 start → route → goal 连接在一起并按固定步长重采样.
    start: (2,) numpy
    route: (M,2) polyline
    goal: (2,) numpy
    step: float, 插值间隔
    """
    segs = []

    # start → route[0]
    if np.linalg.norm(route[0] - start) > 1e-3:
        seg = np.vstack([start, route[0]])
        segs.append(resample_polyline(seg, step))

    # route polyline
    segs.append(resample_polyline(route, step))

    # route[-1] → goal
    if np.linalg.norm(route[-1] - goal) > 1e-3:
        seg = np.vstack([route[-1], goal])
        segs.append(resample_polyline(seg, step))

    # 拼接并去重重复点
    out = np.vstack(segs)
    _, idx = np.unique(out.round(6), axis=0, return_index=True)
    out = out[np.sort(idx)]

    return out
