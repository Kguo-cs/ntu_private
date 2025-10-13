import os
import pickle
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
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
import torch
from scipy.spatial import cKDTree

def process_route_noloop(tokenized_map, tokenized_agent,
                         k: int = 16,
                         radius: float = 40.0,
                         max_idx_per_agent: int = 100,
                         eps: float = 1e-9) -> torch.Tensor:
    """
    Fully vectorized (no Python loop over agents):
      1) Flatten all valid agent points.
      2) Compute per-point yaw without looping (forward diff within agent groups).
      3) Single KDTree query for all points.
      4) Per-agent unique (left ∪ right) edge indices in appearance order, capped at K.
      5) Scatter into (N, K) tensor, -1 padded.

    Uses the ORIGINAL valid points (no resampling).
    """

    # --- 1) Road-edge points (types 4 or 5) ---
    map_type = tokenized_map['type']
    mask45 = (map_type == 4) | (map_type == 5)
    pos = tokenized_map['position'][mask45]            # (M,2) torch
    edge_xy = pos.detach().cpu().numpy()
    num_edges = edge_xy.shape[0]

    N = tokenized_agent["valid_mask"].shape[0]
    Tm1 = tokenized_agent["valid_mask"].shape[1] - 0  # already sliced in caller if you did [:,1:]

    if num_edges == 0 or N == 0:
        return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)

    # --- 2) Slice agent tensors (skip t0 like your original code) ---
    sampled_pos = tokenized_agent["sampled_pos"][:, 1:]         # (N, T-1, 2)
    sampled_heading = tokenized_agent["sampled_heading"][:, 1:] # (N, T-1)
    valid_mask = tokenized_agent["valid_mask"][:, 1:]           # (N, T-1)

    pos_np = sampled_pos.detach().cpu().numpy()       # float
    head_np = sampled_heading.detach().cpu().numpy()  # float (radians)
    mask_np = valid_mask.detach().cpu().numpy().astype(bool)

    # --- 3) Flatten valid points across all agents (no loops) ---
    # Indices of valid (agent, time)
    ag_idx, t_idx = np.where(mask_np)                # both shape (M_all,)
    # Points
    pts_flat = pos_np[ag_idx, t_idx, :]              # (M_all, 2)
    # Agent ids per point
    agents_flat = ag_idx                             # (M_all,)

    # Last valid heading per agent (for last point yaw fallback)
    # Find last valid time per agent: argmax on reversed mask trick
    # reverse along time axis
    rev_mask = mask_np[:, ::-1]
    # distance to last True from the end
    last_from_end = rev_mask.argmax(axis=1)          # if no True, returns 0
    has_any = mask_np.any(axis=1)
    # compute last index safely
    last_t = (mask_np.shape[1] - 1) - last_from_end
    # for agents with no valid, last_t meaningless; but they won't appear in agents_flat anyway
    last_heading = head_np[np.arange(N), np.clip(last_t, 0, mask_np.shape[1]-1)]
    last_heading[~has_any] = 0.0

    # --- 4) Per-point yaw without loops (forward diff within agent blocks) ---
    # Data are grouped by agent already because mask_np is row-major in np.where output
    # Build a "next point" difference but zero it across agent boundaries.
    M_all = pts_flat.shape[0]
    if M_all == 0:
        return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)

    # Forward differences
    dxy = np.zeros_like(pts_flat, dtype=np.float64)
    dxy[:-1] = pts_flat[1:] - pts_flat[:-1]

    # Zero out at boundaries where agent changes
    boundary = (agents_flat[1:] != agents_flat[:-1])
    dxy[:-1][boundary] = 0.0

    # Yaw from forward diffs
    norms = np.hypot(dxy[:, 0], dxy[:, 1])
    yaw_flat = np.zeros(M_all, dtype=np.float64)
    nz = norms > eps
    yaw_flat[nz] = np.arctan2(dxy[nz, 1], dxy[nz, 0])

    # For points with zero diff (including the last point of each agent block),
    # fill with that agent's last_heading
    yaw_flat[~nz] = last_heading[agents_flat[~nz]]

    # --- 5) Single KD-Tree query for all points; then side selection ---
    tree = cKDTree(edge_xy)
    dists, idxs = tree.query(pts_flat, k=k, distance_upper_bound=radius)
    if k == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    E = num_edges
    cosh, sinh = np.cos(yaw_flat), np.sin(yaw_flat)
    n_left  = np.stack([-sinh,  cosh], axis=1)      # (M_all, 2)
    n_right = np.stack([ sinh, -cosh], axis=1)

    # Candidate vectors to neighbors
    # Initialize zeros; invalid neighbors (idx>=E) will be ignored via cost=inf
    cand = np.zeros((M_all, k, 2), dtype=np.float64)
    valid_n = (idxs < E)
    # gather edge coords
    safe_idxs = np.where(valid_n, idxs, 0)
    cand = edge_xy[safe_idxs] - pts_flat[:, None, :]

    dot_left  = (cand * n_left[:, None, :]).sum(axis=2)
    dot_right = (cand * n_right[:, None, :]).sum(axis=2)

    big = np.inf
    left_cost  = np.where(valid_n & (dot_left  > 0), dists, big)
    right_cost = np.where(valid_n & (dot_right > 0), dists, big)

    li = np.argmin(left_cost,  axis=1)
    ri = np.argmin(right_cost, axis=1)
    Ld = left_cost[np.arange(M_all), li]
    Rd = right_cost[np.arange(M_all), ri]
    Li = idxs[np.arange(M_all), li]
    Ri = idxs[np.arange(M_all), ri]

    # mark invalids as -1
    Li[np.isinf(Ld)] = -1
    Ri[np.isinf(Rd)] = -1

    # --- 6) Per-agent unique {Li ∪ Ri} in order of first appearance, no loops ---
    cand_edges = np.concatenate([Li, Ri], axis=0)                           # (2*M_all,)
    cand_agents = np.concatenate([agents_flat, agents_flat], axis=0)        # (2*M_all,)
    valid_cand = (cand_edges >= 0)
    cand_edges = cand_edges[valid_cand]
    cand_agents = cand_agents[valid_cand]

    if cand_edges.size == 0:
        return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)

    # Build pair keys (agent, edge) as a structured array to uniquify
    pairs = np.empty(cand_edges.size, dtype=[('a', cand_agents.dtype), ('e', cand_edges.dtype)])
    pairs['a'] = cand_agents
    pairs['e'] = cand_edges

    # indices of first occurrence in ORIGINAL order
    _, first_idx = np.unique(pairs, return_index=True)
    # restore original order of first occurrences
    first_idx_sorted = np.sort(first_idx)

    a_first = cand_agents[first_idx_sorted]  # agent ids (in order encountered)
    e_first = cand_edges[first_idx_sorted]   # edge ids (first time seen for that agent)

    # For each agent, take first K = max_idx_per_agent occurrences in appearance order.
    # Compute index-in-group without loops:
    # mark group starts
    start_flags = np.empty_like(a_first, dtype=bool)
    start_flags[0] = True
    start_flags[1:] = a_first[1:] != a_first[:-1]
    # running start index of the current group
    starts_positions = np.where(start_flags, np.arange(a_first.size), 0)
    last_start_pos = np.maximum.accumulate(starts_positions)
    idx_in_group = np.arange(a_first.size) - last_start_pos  # 0,1,2,... within each agent block

    keep = idx_in_group < max_idx_per_agent
    a_keep = a_first[keep]
    e_keep = e_first[keep]
    c_keep = idx_in_group[keep]  # column for scatter

    # --- 7) Scatter into (N, max_idx_per_agent), -1 padded ---
    out = np.full((N, max_idx_per_agent), -1, dtype=np.int16)
    # scatter (note: if duplicates somehow survive, the first occurrence order already enforced)
    out[a_keep, c_keep] = e_keep.astype(np.int16, copy=False)

    return torch.from_numpy(out)

def process_route(tokenized_map,tokenized_agent):

    map_type = tokenized_map['type']
    mask4 = (map_type == 4)   | (map_type == 5)

    # idx467 = mask467.nonzero(as_tuple=True)[0]
    # idx4 = mask4.nonzero(as_tuple=True)[0]

    # map idx4 into local indices inside idx45
    # torch.searchsorted requires sorted input (idx45 is sorted by construction)
    # idx4_in_467 = torch.searchsorted(idx467, idx4)
    # mask467 = (map_type == 4) |  (map_type == 6) |   (map_type == 7)

    position = tokenized_map["position"][mask4]
    x, y = position[:, 0], position[:, 1]

    edge_xy = np.column_stack([x, y])  # road-edge points


    sampled_pos = tokenized_agent["sampled_pos"][:, 1:]
    sampled_heading = tokenized_agent["sampled_heading"][:, 1:]

    valid_mask = tokenized_agent['valid_mask'][:, 1:]

    route_map_index = torch.zeros([len(valid_mask), 100]).to(torch.int16) - 1

    for i in range(len(sampled_pos)):
        agent = sampled_pos[i]
        heading = sampled_heading[i]  # radians, same length as agent
        valid = valid_mask[i]

        valid_traj = agent[valid].numpy()

        # heading hint: last valid heading (or first)
        heading_hint = float(heading[valid][-1].item())

        interpolated_traj = interpolate_traj_lookahead(
            valid_traj, step=2.0, lookahead=0.0, heading_last=heading_hint
        )

        yaw_interp = compute_yaw_from_traj(interpolated_traj, heading_hint=heading_hint)

        L_idx, R_idx, L_d, R_d = nearest_edges_biside(
            interpolated_traj, yaw_interp, edge_xy, k=16, radius=40.0
        )

        all_idx = torch.tensor(
            np.unique(np.concatenate([L_idx, R_idx])))  # idx4_in_45[np.unique(np.concatenate([L_idx,R_idx]))]

        # all_idx=idx4_in_467[all_idx]
        n = min(len(all_idx), 100)

        route_map_index[i][:n] = all_idx[:n]

    return route_map_index


def process_scenario( filename):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    tokenized_map = data["tokenized_map"]
    tokenized_agent = data["tokenized_agent"]

    route_map_index=process_route(tokenized_map, tokenized_agent)
    route_map_index1=process_route_noloop(tokenized_map, tokenized_agent)

    print(torch.all(route_map_index == route_map_index1))

    # data["tokenized_agent"]["route_map_index"]=route_map_index
    #
    # output_file = output_path + filename
    #
    # with open(output_file, "wb") as f:
    #     pickle.dump(data, f)

if __name__ == "__main__":
    data_directory = "./waymo_data/full/nuplan_cross2_clean"  # training_map2_03_pred/"
    output_path = "./waymo_data/full/nuplan_cross2_clean_route/"

    files = os.listdir(data_directory)

    data_dict = {}

    os.makedirs(output_path, exist_ok=True)

    # with Pool(16) as pool:
    #     results = list(tqdm(pool.imap_unordered(process_scenario, files), total=len(files)))
    for scenario in tqdm(files):
        process_scenario(scenario)
#
    # print(1)
        # print(len(np.unique(L_idx)), len(np.unique(R_idx))  )
        #
        # plt.plot(valid_traj[:,0], valid_traj[:,1], 'g-')
        # plt.plot(interpolated_traj[:,0], interpolated_traj[:,1], 'r-')
        # for p, li, ri in zip(interpolated_traj, L_idx, R_idx):
        #     if li >= 0: plt.plot([p[0], edge_xy[li,0]], [p[1], edge_xy[li,1]], 'b-', alpha=0.4)
        #     if ri >= 0: plt.plot([p[0], edge_xy[ri,0]], [p[1], edge_xy[ri,1]], 'm-', alpha=0.4)
        #
        # plt.scatter(x, y)
        #
        # plt.show()
        #
        # output_file = output_path + filename
        #
        # with open(output_file, "wb") as f:
        #     pickle.dump(data, f)
        #
    #