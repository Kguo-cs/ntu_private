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

#
#
# import numpy as np
# import torch
# from scipy.spatial import cKDTree
#
# def process_route_resample_noloop(tokenized_map, tokenized_agent,
#                                   step: float = 2.0,
#                                   lookahead: float = 0.0,  # kept for API symmetry; here we use 0.0
#                                   k: int = 16,
#                                   radius: float = 40.0,
#                                   max_idx_per_agent: int = 100,
#                                   eps: float = 1e-9) -> torch.Tensor:
#     """
#     No Python loop over agents, but resamples each agent's valid trajectory at fixed `step`.
#     Then finds left/right nearest road-edge points with one KD-tree query and returns
#     per-agent sorted-unique indices (padded with -1 to `max_idx_per_agent`).
#
#     Notes:
#     - Matches your 'sorted unique' behavior per agent (like np.unique's sorted result).
#     - lookahead is kept for signature; here we assume 0.0 to match your current code path.
#     """
#
#     # ---- Road-edge points (type 4 or 5) ----
#     map_type = tokenized_map['type']
#     mask45 = (map_type == 4) | (map_type == 5)
#     pos = tokenized_map['position'][mask45]              # (M,2) torch
#     edge_xy = pos.detach().cpu().numpy()
#     num_edges = edge_xy.shape[0]
#
#     N = tokenized_agent["valid_mask"].shape[0]
#     if num_edges == 0 or N == 0:
#         return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)
#
#     # ---- Agent tensors, drop t0 to match your code ----
#     sampled_pos = tokenized_agent["sampled_pos"][:, 1:]          # (N, T-1, 2)
#     sampled_heading = tokenized_agent["sampled_heading"][:, 1:]  # (N, T-1)
#     valid_mask = tokenized_agent["valid_mask"][:, 1:]            # (N, T-1)
#
#     P = sampled_pos.detach().cpu().numpy()
#     H = sampled_heading.detach().cpu().numpy()
#     Msk = valid_mask.detach().cpu().numpy().astype(bool)
#
#     # ---- Flatten valid points grouped by agent (no loops) ----
#     ag_idx, t_idx = np.where(Msk)           # row-major → grouped by agent
#     if ag_idx.size == 0:
#         return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)
#
#     pts = P[ag_idx, t_idx, :]               # (M_all, 2), grouped by agent
#     agents_flat = ag_idx                    # (M_all,)
#
#     # ---- Per-agent last heading (for yaw fill on degenerate steps) ----
#     rev = Msk[:, ::-1]
#     last_from_end = rev.argmax(axis=1)
#     has_any = Msk.any(axis=1)
#     last_t = (Msk.shape[1] - 1) - last_from_end
#     last_heading = H[np.arange(N), np.clip(last_t, 0, Msk.shape[1]-1)]
#     last_heading[~has_any] = 0.0  # unused for empty agents
#
#     # ---- Compute groupwise cumulative arclength s for original valid points ----
#     # forward diffs within agent; zero across boundaries
#     dxy = np.zeros_like(pts, dtype=np.float64)
#     M_all = pts.shape[0]
#     if M_all > 1:
#         boundary = (agents_flat[1:] != agents_flat[:-1])
#         dxy[:-1] = pts[1:] - pts[:-1]
#         dxy[:-1][boundary] = 0.0
#     seglen = np.hypot(dxy[:, 0], dxy[:, 1])
#
#     # cumsum that resets per agent:
#     # counts of valid points per agent (including agents with zero valid)
#     counts = np.bincount(agents_flat, minlength=N)
#     starts = np.cumsum(np.r_[0, counts[:-1]])               # start index in flat arrays
#     csum = np.cumsum(seglen)                                 # global cumsum
#     # offsets per point = csum at group start repeated by count
#     off = np.repeat(csum[starts], counts, axis=0)
#     s_flat = csum - off                                      # (M_all,), per-agent cumulative s
#     # ensure s[0]==0 for each agent: seglen[start]==0 → ok
#
#     # ---- Drop zero-length segments like your helper (keep first point and points with seg>eps) ----
#     # Equivalent to keeping points where previous segment length > eps or group start
#     keep = np.ones(M_all, dtype=bool)
#     if M_all > 1:
#         keep[1:] = (seglen[1:] > eps)
#         keep[starts] = True  # always keep the first point of each agent
#     pts = pts[keep]
#     s_flat = s_flat[keep]
#     agents_flat = agents_flat[keep]
#
#     if pts.shape[0] == 0:
#         return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)
#
#     # recompute counts/starts after filtering
#     counts = np.bincount(agents_flat, minlength=N)
#     starts = np.cumsum(np.r_[0, counts[:-1]])
#
#     # per-agent total lengths (last s in each group or 0 if empty)
#     totals = np.zeros(N, dtype=np.float64)
#     nonempty = counts > 0
#     last_indices = starts[nonempty] + counts[nonempty] - 1
#     totals[nonempty] = s_flat[last_indices]
#
#     # ---- Build resampling grid for each agent without loops ----
#     # number of resampled points per agent: floor(total/step) + 1 (includes 0)
#     k_per_agent = (np.floor(totals / max(step, eps)).astype(np.int64) + 1) * (counts > 0)
#     K = int(k_per_agent.sum())  # total resampled points across all agents
#
#     if K == 0:
#         return (torch.zeros((N, max_idx_per_agent), dtype=torch.int16) - 1)
#
#     starts_new = np.cumsum(np.r_[0, k_per_agent[:-1]])  # starts in resampled flat arrays, len N
#     j = np.arange(K, dtype=np.int64)
#     # agent id per resampled sample via searchsorted on group starts
#     ag_new = np.searchsorted(starts_new, j, side="right") - 1
#     # position within agent block
#     pos_in_group = j - starts_new[ag_new]
#     s_new = pos_in_group.astype(np.float64) * step      # (K,)
#
#     # ---- Interpolate x/y for all agents at s_new (one shot) ----
#     # Give each agent a large s-offset so groups don't mix in np.interp
#     Lmax = float(totals.max()) if nonempty.any() else 0.0
#     sep = Lmax + 1.0
#     s_off = np.arange(N, dtype=np.float64) * sep
#
#     # Build original (s,x,y) with offsets
#     s_orig_off = s_flat + s_off[agents_flat]
#     # np.interp requires sorted x; groups are already grouped by agent and s increasing
#     x_orig = pts[:, 0]
#     y_orig = pts[:, 1]
#
#     # Build query s with offsets
#     s_query_off = s_new + s_off[ag_new]
#
#     # Perform interpolation
#     x_new = np.interp(s_query_off, s_orig_off, x_orig)
#     y_new = np.interp(s_query_off, s_orig_off, y_orig)
#     Q = np.stack([x_new, y_new], axis=1)                 # (K, 2)
#
#     # ---- Compute yaw for resampled points (vectorized, per-agent) ----
#     # dQ = np.zeros_like(Q, dtype=np.float64)
#     # if K > 1:
#     #     bnd2 = (ag_new[1:] != ag_new[:-1])
#     #     dQ[:-1] = Q[1:] - Q[:-1]
#     #     dQ[:-1][bnd2] = 0.0
#     # norms2 = np.hypot(dQ[:, 0], dQ[:, 1])
#     # yaw = np.zeros(K, dtype=np.float64)
#     # nz2 = norms2 > eps
#     # yaw[nz2] = np.arctan2(dQ[nz2, 1], dQ[nz2, 0])
#     # # fill degenerate with each agent's last heading hint
#     # yaw[~nz2] = last_heading[ag_new[~nz2]]
#     K = Q.shape[0]
#     # prev and next (zeroed across agent boundaries)
#     prevQ = np.zeros_like(Q)
#     nextQ = np.zeros_like(Q)
#
#     if K > 1:
#         # detect group boundaries
#         bnd = (ag_new[1:] != ag_new[:-1])
#         prevQ[1:] = Q[:-1]
#         nextQ[:-1] = Q[1:]
#         # clamp across boundaries (like gradient's one-sided edges)
#         prevQ[1:][bnd] = Q[1:][bnd]  # first in group: prev = self
#         nextQ[:-1][bnd] = Q[:-1][bnd]  # last in group: next = self
#
#     # interior: (next - prev)/2 ; edges fall back to forward/backward diff
#     dQ = (nextQ - prevQ) * 0.5
#     # for first in each group, use forward diff: Q[i+1] - Q[i]
#     first_flags = np.zeros(K, dtype=bool)
#     first_flags[0] = True
#     first_flags[1:] = bnd
#     dQ[first_flags] = (nextQ - Q)[first_flags]
#     # for last in each group, use backward diff: Q[i] - Q[i-1]
#     last_flags = np.zeros(K, dtype=bool)
#     last_flags[-1] = True
#     last_flags[:-1] |= bnd
#     dQ[last_flags] = (Q - prevQ)[last_flags]
#
#     norms = np.hypot(dQ[:, 0], dQ[:, 1])
#     yaw = np.zeros(K, dtype=np.float64)
#     nz = norms > eps
#     yaw[nz] = np.arctan2(dQ[nz, 1], dQ[nz, 0])
#     if not np.all(nz):
#         # positions 0..K-1 and per-sample "rank within agent"
#         j = np.arange(K, dtype=np.int64)
#         # group starts in resampled array
#         starts_new = np.zeros(N + 1, dtype=np.int64)
#         # rebuild starts_new if you don't have it in scope:
#         #   starts_new = np.cumsum(np.r_[0, k_per_agent[:-1]])
#         # ensure we have it from earlier resampling section
#
#         gpos = j - starts_new[ag_new]  # 0..len(agent)-1 within each agent
#         # forward fill: distance to last seen nonzero in same group
#         dist_fwd = np.full(K, np.inf)
#         # encode "seen" positions of nonzero per agent using big-offset trick
#         BIG = (Q.shape[0] + 5)
#         code = np.where(nz, gpos + ag_new * BIG, -np.inf)
#         last_seen = np.maximum.accumulate(code)
#         same_group = (last_seen > -np.inf) & (np.floor_divide(last_seen, BIG) == ag_new)
#         last_pos = np.where(same_group, last_seen - np.floor_divide(last_seen, BIG) * BIG, -np.inf)
#         dist_fwd[~nz & same_group] = gpos[~nz & same_group] - last_pos[~nz & same_group]
#
#         # backward fill: distance to next nonzero in same group
#         dist_bwd = np.full(K, np.inf)
#         # reverse trick
#         ag_rev = ag_new[::-1]
#         nz_rev = nz[::-1]
#         gpos_rev = (k_per_agent[ag_new][::-1] - 1) - gpos[::-1]  # distance from end of group
#         code_rev = np.where(nz_rev, gpos_rev + ag_rev * BIG, -np.inf)
#         next_seen = np.maximum.accumulate(code_rev)
#         same_group_rev = (next_seen > -np.inf) & (np.floor_divide(next_seen, BIG) == ag_rev)
#         next_pos = np.where(same_group_rev, next_seen - np.floor_divide(next_seen, BIG) * BIG, -np.inf)
#         dist_bwd_rev = np.full(K, np.inf)
#         dist_bwd_rev[~nz[::-1] & same_group_rev] = gpos_rev[~nz[::-1] & same_group_rev] - next_pos[
#             ~nz[::-1] & same_group_rev]
#         dist_bwd = dist_bwd_rev[::-1]
#
#         # choose nearer; tie-breaker: prefer forward (arbitrary but stable)
#         pick_fwd = dist_fwd <= dist_bwd
#         # indices of the chosen donor yaw (compute again via indices)
#         donor = np.empty(K, dtype=np.int64)
#         donor.fill(-1)
#         # forward donor index
#         donor_fwd = (starts_new[ag_new] + (gpos - dist_fwd.astype(np.int64)))
#         # backward donor index
#         donor_bwd = (starts_new[ag_new] + (gpos + dist_bwd.astype(np.int64)))
#
#         use_fwd = (~nz) & np.isfinite(dist_fwd) & (pick_fwd)
#         use_bwd = (~nz) & np.isfinite(dist_bwd) & (~pick_fwd)
#
#         donor[use_fwd] = donor_fwd[use_fwd]
#         donor[use_bwd] = donor_bwd[use_bwd]
#
#         has_donor = donor >= 0
#         yaw[~nz & has_donor] = yaw[donor[~nz & has_donor]]
#         # if a whole group is zero-motion, yaw stays 0.0 (same as your fallback behavior)
#
#     # ---- Single KD-tree query + side selection ----
#     tree = cKDTree(edge_xy)
#     dists, idxs = tree.query(Q, k=k, distance_upper_bound=radius)
#     if k == 1:
#         dists = dists[:, None]; idxs = idxs[:, None]
#     E = edge_xy.shape[0]
#
#     cosh, sinh = np.cos(yaw), np.sin(yaw)
#     n_left  = np.stack([-sinh,  cosh], axis=1)
#     n_right = np.stack([ sinh, -cosh], axis=1)
#
#     safe_idxs = np.where(idxs < E, idxs, 0)
#     cand = edge_xy[safe_idxs] - Q[:, None, :]
#
#     dot_left  = (cand * n_left[:, None, :]).sum(axis=2)
#     dot_right = (cand * n_right[:, None, :]).sum(axis=2)
#     big = np.inf
#     valid_n = (idxs < E)
#     left_cost  = np.where(valid_n & (dot_left  > 0), dists, big)
#     right_cost = np.where(valid_n & (dot_right > 0), dists, big)
#
#     li = np.argmin(left_cost,  axis=1)
#     ri = np.argmin(right_cost, axis=1)
#     Ld = left_cost[np.arange(K), li]
#     Rd = right_cost[np.arange(K), ri]
#     Li = idxs[np.arange(K), li]
#     Ri = idxs[np.arange(K), ri]
#     Li[np.isinf(Ld)] = -1
#     Ri[np.isinf(Rd)] = -1
#
#     # ---- Per-agent **sorted unique** {Li ∪ Ri} and scatter (no loops) ----
#     cand_edges = np.concatenate([Li, Ri])
#     cand_agents = np.concatenate([ag_new, ag_new])
#     valid_edges = cand_edges >= 0
#     cand_edges = cand_edges[valid_edges]
#     cand_agents = cand_agents[valid_edges]
#
#     out = np.full((N, max_idx_per_agent), -1, dtype=np.int16)
#     if cand_edges.size > 0:
#         # sort by (agent, edge) then drop duplicates → sorted-unique per agent
#         order = np.lexsort((cand_edges, cand_agents))
#         a_sorted = cand_agents[order]
#         e_sorted = cand_edges[order]
#         dup = np.zeros_like(a_sorted, dtype=bool)
#         dup[1:] = (a_sorted[1:] == a_sorted[:-1]) & (e_sorted[1:] == e_sorted[:-1])
#         a_unique = a_sorted[~dup]
#         e_unique = e_sorted[~dup]
#
#         # within each agent group, take first K indices
#         start_flags = np.ones_like(a_unique, dtype=bool)
#         start_flags[1:] = a_unique[1:] != a_unique[:-1]
#         starts_pos = np.where(start_flags, np.arange(a_unique.size), 0)
#         group_start = np.maximum.accumulate(starts_pos)
#         idx_in_group = np.arange(a_unique.size) - group_start
#         keep = idx_in_group < max_idx_per_agent
#
#         out[a_unique[keep], idx_in_group[keep]] = e_unique[keep].astype(np.int16, copy=False)
#
#     return torch.from_numpy(out)


def process_route(map_save,pt_token,agent):

    map_type = pt_token['type']
    mask4 = (map_type == 4)   | (map_type == 5)

    # idx467 = mask467.nonzero(as_tuple=True)[0]
    # idx4 = mask4.nonzero(as_tuple=True)[0]

    # map idx4 into local indices inside idx45
    # torch.searchsorted requires sorted input (idx45 is sorted by construction)
    # idx4_in_467 = torch.searchsorted(idx467, idx4)
    # mask467 = (map_type == 4) |  (map_type == 6) |   (map_type == 7)

    position = map_save["traj_pos"][mask4][:,0]
    x, y = position[:, 0], position[:, 1]

    edge_xy = np.column_stack([x, y])  # road-edge points


    sampled_pos = agent["position"][:, 10:,:2]
    sampled_heading = agent["heading"][:, 10:]

    valid_mask = agent['valid_mask'][:, 10:]

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

    # tokenized_map = data["tokenized_map"]
    # tokenized_agent = data["tokenized_agent"]

    route_map_index=process_route(data['map_save'], data['pt_token'], data['agent'])
    #route_map_index1=process_route_resample_noloop(tokenized_map, tokenized_agent)

   # print(torch.all(route_map_index == route_map_index1))

    data["agent"]["route_map_index"]=route_map_index

    output_file = output_path + filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    data_directory = "./waymo_data/full/nuplan_cross2"  # training_map2_03_pred/"_clean
    output_path = "./waymo_data/full/nuplan_cross2_route/"

    files = os.listdir(data_directory)[300000:]

    data_dict = {}

    os.makedirs(output_path, exist_ok=True)

    with Pool(32) as pool:
        results = list(tqdm(pool.imap_unordered(process_scenario, files), total=len(files)))
#     for scenario in tqdm(files):
#         process_scenario(scenario)
# #
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