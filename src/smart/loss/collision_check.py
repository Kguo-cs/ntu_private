import torch
from typing import Tuple

import torch
from typing import Tuple, Optional

def _ensure_time(pos: torch.Tensor, heading: torch.Tensor):
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    return pos, heading, has_time

def _knn_candidates(pos: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each agent, pick up to k nearest neighbors (by center distance) within the same batch.
    Returns indices tensor of shape [N, k] with -1 for missing.
    """
    device = pos.device
    N, T, _ = pos.shape
    # use t=0 (or any time) for neighbor pruning
    p = pos[:, 0, :]  # [N,2]
    idxs = torch.full((N, k), -1, device=device, dtype=torch.long)
    # group by batch id to avoid N^2 across scenes
    uniq = torch.unique(batch)
    for b in uniq.tolist():
        mask = (batch == b)
        inds = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if inds.numel() <= 1:
            continue
        P = p[inds]  # [M,2]
        # pairwise dists (M x M) – OK because per-scene M is usually modest
        D = torch.cdist(P, P)
        D.fill_diagonal_(float('inf'))
        m = min(k, max(0, P.shape[0]-1))
        if m > 0:
            nn = D.topk(m, largest=False).indices  # [M,m]
            fill = torch.full((P.shape[0], k), -1, device=device, dtype=torch.long)
            fill[:, :m] = inds[nn]
            idxs[inds] = fill
    return idxs  # [N,k] (-1 padded)

# ---------- Option A: SAT-gap based signed distance (very fast, conservative) ----------

def signed_distance_boxes_sat_fast(
    pos: torch.Tensor,          # [N,2] or [N,T,2]
    heading: torch.Tensor,      # [N] or [N,T] (rad)
    shape_lw: torch.Tensor,     # [N,2] (L,W)
    batch: torch.Tensor,        # [N]
    margin: float = 0.0,
    knn_k: int = 16,
) -> torch.Tensor:
    """
    Signed distance per agent to its nearest other agent in the same scene.
    >0: separated (conservative gap along best separating axis)
    <0: penetration depth (SAT overlap depth)
    Memory/time efficient: SAT only + k-NN pruning.
    """
    pos, heading, has_time = _ensure_time(pos, heading)
    N, T, _ = pos.shape
    device = pos.device

    # half-extents
    hl = shape_lw[:, 0] * 0.5 + margin  # [N]
    hw = shape_lw[:, 1] * 0.5 + margin  # [N]

    # local axes
    c = torch.cos(heading); s = torch.sin(heading)     # [N,T]
    u = torch.stack([c, s], dim=-1)                    # [N,T,2]
    v = torch.stack([-s, c], dim=-1)                   # [N,T,2]

    # k-NN pruning (per scene), indices [N,knn_k]
    nbr_idx = _knn_candidates(pos, batch, knn_k)       # [-1 padded]

    # prepare outputs
    sd = torch.full((N, T), float('inf'), device=device)

    # loop over neighbor slots (vectorized over all agents/time for that slot)
    for slot in range(knn_k):
        j_idx = nbr_idx[:, slot]                       # [N]
        valid = j_idx >= 0
        if not valid.any():
            continue

        i_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        jj = j_idx[valid]                              # [M]

        # pairwise tensors aligned by M
        ci = pos[i_idx]           # [M,T,2]
        cj = pos[jj]              # [M,T,2]
        t_vec = cj - ci           # [M,T,2]

        ui = u[i_idx]; vi = v[i_idx]
        uj = u[jj];    vj = v[jj]

        t_ui = (t_vec * ui).sum(-1)    # [M,T]
        t_vi = (t_vec * vi).sum(-1)

        R00 = (ui * uj).sum(-1)        # [M,T]
        R01 = (ui * vj).sum(-1)
        R10 = (vi * uj).sum(-1)
        R11 = (vi * vj).sum(-1)

        a_u = hl[i_idx][:, None]       # [M,1]
        a_v = hw[i_idx][:, None]
        b_u = hl[jj][:, None]
        b_v = hw[jj][:, None]

        eps = 1e-8
        absR00 = R00.abs() + eps
        absR01 = R01.abs() + eps
        absR10 = R10.abs() + eps
        absR11 = R11.abs() + eps

        # gap along each SAT axis (positive => separated along that axis)
        gap_ui = t_ui.abs() - (a_u + b_u * absR00 + b_v * absR01)[:, :]      # [M,T]
        gap_vi = t_vi.abs() - (a_v + b_u * absR10 + b_v * absR11)[:, :]

        t_uj = (t_ui * R00 + t_vi * R10).abs()
        t_vj = (t_ui * R01 + t_vi * R11).abs()
        gap_uj = t_uj - (b_u + a_u * absR00 + a_v * absR10)[:, :]
        gap_vj = t_vj - (b_v + a_u * absR01 + a_v * absR11)[:, :]

        # max gap over axes: if >0 => separated.
        max_gap = torch.maximum(torch.maximum(gap_ui, gap_vi),
                                torch.maximum(gap_uj, gap_vj))               # [M,T]

        # penetration depth proxy (min overlap over axes) = -max_gap when overlapping
        # sd_pair: separated => +max_gap ; overlapping => -min_overlap = max_gap (negative)
        # So we can just take sd_pair = max_gap, but clamp positive/negative appropriately:
        sep = max_gap > 0
        # when overlapping, penetration = minimum of overlaps across axes:
        min_overlap = torch.minimum(torch.minimum(-gap_ui, -gap_vi),
                                    torch.minimum(-gap_uj, -gap_vj)).clamp_min(0.0)
        sd_pair = torch.where(sep, max_gap, -min_overlap)                     # [M,T]

        # keep nearest (smallest absolute)
        old = sd[i_idx]                     # [M,T]
        take = (sd_pair.abs() < old.abs())
        sd[i_idx] = torch.where(take, sd_pair, old)

    if not has_time:
        sd = sd[:, 0]
    return sd  # [N] or [N,T]

# ---------- Option B: Capsule approximation (segment + radius) ----------

def signed_distance_capsules(
    pos: torch.Tensor,          # [N,2] or [N,T,2]
    heading: torch.Tensor,      # [N] or [N,T]
    shape_lw: torch.Tensor,     # [N,2] (L,W)
    batch: torch.Tensor,        # [N]
    margin: float = 0.0,
    knn_k: int = 16,
) -> torch.Tensor:
    """
    Approximate each box as a capsule:
      centerline segment length ~ L, radius ~ W/2 (+margin).
    Signed distance = dist(segment_i, segment_j) - (r_i + r_j).
    Much lighter than 4x4 edge checks, often closer to Euclidean than SAT gaps.
    """
    pos, heading, has_time = _ensure_time(pos, heading)
    N, T, _ = pos.shape
    device = pos.device

    L = shape_lw[:, 0]
    R = shape_lw[:, 1] * 0.5 + margin

    # segment endpoints in world: c ± 0.5*L * u
    c = torch.cos(heading); s = torch.sin(heading)
    u = torch.stack([c, s], dim=-1)                 # [N,T,2]
    seg_half = (L[:, None] * 0.5)                   # [N,1]
    A = pos - seg_half[..., None] * u               # [N,T,2] (start)
    B = pos + seg_half[..., None] * u               # [N,T,2] (end)

    # neighbors
    nbr_idx = _knn_candidates(pos, batch, knn_k)

    sd = torch.full((N, T), float('inf'), device=device)

    for slot in range(knn_k):
        j_idx = nbr_idx[:, slot]
        valid = j_idx >= 0
        if not valid.any(): continue

        i_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        jj = j_idx[valid]

        A0 = A[i_idx]; B0 = B[i_idx]    # [M,T,2]
        A1 = A[jj];    B1 = B[jj]       # [M,T,2]

        u0 = B0 - A0; v1 = B1 - A1
        w0 = A0 - A1

        a = (u0*u0).sum(-1)             # [M,T]
        b = (u0*v1).sum(-1)
        c = (v1*v1).sum(-1)
        d = (u0*w0).sum(-1)
        e = (v1*w0).sum(-1)

        denom = a*c - b*b
        eps = 1e-9
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)

        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom

        s = s.clamp(0.0, 1.0)
        t = t.clamp(0.0, 1.0)

        P = A0 + s[..., None]*u0
        Q = A1 + t[..., None]*v1

        dist = (P - Q).norm(dim=-1)     # [M,T]
        # signed by radii overlap
        rsum = (R[i_idx][:, None] + R[jj][:, None])  # [M,1] -> [M,T] by broadcast
        sd_pair = dist - rsum

        old = sd[i_idx]
        take = (sd_pair.abs() < old.abs())
        sd[i_idx] = torch.where(take, sd_pair, old)

    if not has_time:
        sd = sd[:, 0]
    return sd

import torch

def value_to_hist_class(x: torch.Tensor, min_val: float, max_val: float, num_bins: int) -> torch.Tensor:
    """
    Quantize continuous values x into histogram bins -> class indices [0..num_bins-1].

    Args:
      x:       tensor of values
      min_val: lower bound of histogram
      max_val: upper bound of histogram
      num_bins: number of bins

    Returns:
      LongTensor of same shape as x with bin indices
    """
    # scale to [0, num_bins)
    bin_width = (max_val - min_val) / num_bins
    idx = ((x - min_val) / bin_width).floor().long()

    # clip to valid range
    idx = idx.clamp(0, num_bins - 1)
    return idx

def oriented_box_collision(
        pos: torch.Tensor,  # [N,2] or [N,T,2]  (meters)
        heading: torch.Tensor,  # [N] or [N,T]      (radians)
        shape_lw: torch.Tensor,  # [N,2]  (length, width in meters)
        batch: torch.Tensor,  # [N] (e.g., [0,0,1,1,1,2])
        eps: float = 1e-6,  # numerical epsilon
        margin: float = 0.0  # inflate half-extents by this (meters)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      any_collision_per_agent: [N] or [N,T] bool — whether each agent collides with ANY other agent in the same batch.
      pairwise_collision:      [N,N] or [N,T,N,N] bool — pairwise collision mask (same-batch only, diag=False).

    Model: Each agent is an oriented rectangle (OBB) with:
      - center at pos
      - heading (yaw) (x-axis forward)
      - half-extents: (length/2, width/2) + margin
    Collision test uses the Separating Axis Theorem (SAT) with 4 axes (two per box).
    """
    # Normalize shapes to half-extents
    half_len = shape_lw[:, 0] * 0.5 + margin  # [N]
    half_wid = shape_lw[:, 1] * 0.5 + margin  # [N]

    # Handle time dimension uniformly: add T dim if absent
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]  # [N,1,2]
        heading = heading[:, None]  # [N,1]
    N, T, _ = pos.shape

    # Local axes for each agent/time
    c = torch.cos(heading)  # [N,T]
    s = torch.sin(heading)  # [N,T]
    # Agent i's local axes in world frame:
    # u = forward (length axis), v = left (width axis)
    u = torch.stack([c, s], dim=-1)  # [N,T,2]
    v = torch.stack([-s, c], dim=-1)  # [N,T,2]

    # Prepare pairwise broadcasting across agents
    # Centers
    ci = pos[:, None, :, :]  # [N,1,T,2]
    cj = pos[None, :, :, :]  # [1,N,T,2]
    t_vec = cj - ci  # [N,N,T,2]   (vector from i to j)

    # Axes per agent
    ui = u[:, None, :, :]  # [N,1,T,2]
    vi = v[:, None, :, :]  # [N,1,T,2]
    uj = u[None, :, :, :]  # [1,N,T,2]
    vj = v[None, :, :, :]  # [1,N,T,2]

    # Project t_vec into i's frame (t in i-frame)
    t_u_i = (t_vec * ui).sum(dim=-1)  # [N,N,T]
    t_v_i = (t_vec * vi).sum(dim=-1)  # [N,N,T]

    # Rotation between frames: R_ij = [[ui·uj, ui·vj],[vi·uj, vi·vj]]
    R00 = (ui * uj).sum(dim=-1)  # [N,N,T]
    R01 = (ui * vj).sum(dim=-1)
    R10 = (vi * uj).sum(dim=-1)
    R11 = (vi * vj).sum(dim=-1)

    # Half extents broadcast
    a_u = half_len[:, None, None]  # [N,1,1]
    a_v = half_wid[:, None, None]  # [N,1,1]
    b_u = half_len[None, :, None]  # [1,N,1]
    b_v = half_wid[None, :, None]  # [1,N,1]

    # Numerical guard for near-parallel axes
    absR00 = R00.abs() + eps
    absR01 = R01.abs() + eps
    absR10 = R10.abs() + eps
    absR11 = R11.abs() + eps

    # SAT tests (4 axes):
    # 1) axis = u_i
    cond1 = t_u_i.abs() <= (a_u + b_u * absR00 + b_v * absR01).squeeze(-1)
    # 2) axis = v_i
    cond2 = t_v_i.abs() <= (a_v + b_u * absR10 + b_v * absR11).squeeze(-1)

    # For j's axes, project t into j's frame: equivalently use t' = -R_ij^T * t_i
    # We can compute projections using symmetry formulas:
    t_u_j = (t_u_i * R00 + t_v_i * R10).abs()  # |t·u_j| = |t_u_i*(ui·uj) + t_v_i*(vi·uj)|
    t_v_j = (t_u_i * R01 + t_v_i * R11).abs()  # |t·v_j| = |t_u_i*(ui·vj) + t_v_i*(vi·vj)|

    # 3) axis = u_j
    cond3 = t_u_j <= (b_u + a_u * absR00 + a_v * absR10).squeeze(-1)
    # 4) axis = v_j
    cond4 = t_v_j <= (b_v + a_u * absR01 + a_v * absR11).squeeze(-1)

    overlap = cond1 & cond2 & cond3 & cond4  # [N,N,T]

    # Same-batch mask and exclude self
    same_batch = (batch[:, None] == batch[None, :])  # [N,N]
    not_self = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    pair_mask = same_batch & not_self  # [N,N]

    # Apply pair mask
    overlap = overlap & pair_mask[..., None]  # [N,N,T]

    # Reduce to any-collision per agent (with any other)
    any_collision_per_agent = overlap.any(dim=1)  # [N,T] (True if collides with any j)

    # Squeeze time dim if input had no time
    if not has_time:
        any_collision_per_agent = any_collision_per_agent[:, 0]  # [N]
        pairwise_collision = overlap[:, :, 0]  # [N,N]
    else:
        pairwise_collision = overlap  # [N,N,T]

    return any_collision_per_agent, pairwise_collision
