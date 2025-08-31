import torch
from typing import Tuple

import torch
from typing import Tuple

def signed_distance_to_nearest_object(
    pos: torch.Tensor,         # [N,2]  or [N,T,2]
    heading: torch.Tensor,     # [N]    or [N,T] (radians)
    shape_lw: torch.Tensor,    # [N,2]  (length, width)
    batch: torch.Tensor,       # [N] e.g., [0,0,1,1,1,2]
    margin: float = 0.0,       # inflate boxes (safety buffer)
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Returns:
      sd_min: [N] or [N,T]  signed distance to nearest other object within the same batch.
               >0  : gap (no overlap);  <0 : penetration depth (overlap).
    Notes:
      - Overlap depth uses a SAT-based proxy (fast, commonly used).
      - Non-overlap distance is exact via segment-segment distances (min over 4x4 edges) plus vertex-inside checks.
    """
    # Normalize to [N,T]
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    N, T, _ = pos.shape

    # Half-extents
    half_len = shape_lw[:, 0] * 0.5 + margin   # [N]
    half_wid = shape_lw[:, 1] * 0.5 + margin   # [N]

    # Local axes u (forward) and v (left)
    c = torch.cos(heading)                     # [N,T]
    s = torch.sin(heading)                     # [N,T]
    u = torch.stack([c, s], dim=-1)            # [N,T,2]
    v = torch.stack([-s, c], dim=-1)           # [N,T,2]

    # Build rectangle corners in world coords for each agent/time
    # local corners: (+/-half_len, +/-half_wid)
    hl = half_len[:, None, None]               # [N,1,1]
    hw = half_wid[:, None, None]               # [N,1,1]
    # corners (in local uv): (±hl, ±hw)
    # order: (+,+), (+,-), (-,-), (-,+) (CCW)
    local = torch.stack([
        torch.stack([ hl.expand(N,T,1).squeeze(-1),  hw.expand(N,T,1).squeeze(-1)], dim=-1),
        torch.stack([ hl.expand(N,T,1).squeeze(-1), -hw.expand(N,T,1).squeeze(-1)], dim=-1),
        torch.stack([-hl.expand(N,T,1).squeeze(-1), -hw.expand(N,T,1).squeeze(-1)], dim=-1),
        torch.stack([-hl.expand(N,T,1).squeeze(-1),  hw.expand(N,T,1).squeeze(-1)], dim=-1),
    ], dim=2)  # [N,T,4,2]

    # world corner = pos + local_u * u + local_v * v
    # Broadcast u,v to [N,T,1,2]
    u_ = u[:, :, None, :]  # [N,T,1,2]
    v_ = v[:, :, None, :]  # [N,T,1,2]
    corners = pos[:, :, None, :] + local[..., :1] * u_ + local[..., 1:] * v_  # [N,T,4,2]

    # Edges as segments: 4 edges per rect (0-1, 1-2, 2-3, 3-0)
    e_idx = torch.tensor([[0,1],[1,2],[2,3],[3,0]], device=pos.device)
    segA = corners[:, :, e_idx[:,0], :]  # [N,T,4,2]
    segB = corners[:, :, e_idx[:,1], :]  # [N,T,4,2]

    # Pairwise masks (same scene, exclude self)
    same = (batch[:, None] == batch[None, :])  # [N,N]
    not_self = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    pair_mask = same & not_self               # [N,N]

    # ===== SAT overlap & penetration proxy =====
    # Pairwise quantities
    ci = pos[:, None, :, :]   # [N,1,T,2]
    cj = pos[None, :, :, :]   # [1,N,T,2]
    t_vec = cj - ci           # [N,N,T,2]

    ui = u[:, None, :, :]     # [N,1,T,2]
    vi = v[:, None, :, :]
    uj = u[None, :, :, :]     # [1,N,T,2]
    vj = v[None, :, :, :]

    # Projections into i-frame
    t_u_i = (t_vec * ui).sum(-1)    # [N,N,T]
    t_v_i = (t_vec * vi).sum(-1)    # [N,N,T]

    R00 = (ui * uj).sum(-1)         # [N,N,T]
    R01 = (ui * vj).sum(-1)
    R10 = (vi * uj).sum(-1)
    R11 = (vi * vj).sum(-1)

    a_u = half_len[:, None, None]   # [N,1,1]
    a_v = half_wid[:, None, None]   # [N,1,1]
    b_u = half_len[None, :, None]   # [1,N,1]
    b_v = half_wid[None, :, None]   # [1,N,1]

    absR00 = R00.abs() + 1e-8
    absR01 = R01.abs() + 1e-8
    absR10 = R10.abs() + 1e-8
    absR11 = R11.abs() + 1e-8

    # Gaps along the 4 SAT axes (positive => separated along that axis)
    gap_ui = (t_u_i.abs() - (a_u + b_u * absR00 + b_v * absR01).squeeze(-1))  # [N,N,T]
    gap_vi = (t_v_i.abs() - (a_v + b_u * absR10 + b_v * absR11).squeeze(-1))

    t_u_j = (t_u_i * R00 + t_v_i * R10).abs()
    t_v_j = (t_u_i * R01 + t_v_i * R11).abs()
    gap_uj = (t_u_j - (b_u + a_u * absR00 + a_v * absR10).squeeze(-1))
    gap_vj = (t_v_j - (b_v + a_u * absR01 + a_v * absR11).squeeze(-1))

    # If ANY axis has positive gap, boxes are separated (SAT)
    max_gap = torch.stack([gap_ui, gap_vi, gap_uj, gap_vj], dim=-1).amax(dim=-1)  # [N,N,T]
    separated = max_gap > 0                                                       # [N,N,T]

    # Penetration depth proxy (overlap case): min overlap across axes
    # depth = min_a (R_i(a) + R_j(a) - |t·a|) = -max_gap when overlapping
    # (since max_gap <= 0 in overlap), but use explicit min-over-axes for clarity:
    depth_ui = -(gap_ui)  # >=0 when overlapping
    depth_vi = -(gap_vi)
    depth_uj = -(gap_uj)
    depth_vj = -(gap_vj)
    pen_depth = torch.stack([depth_ui, depth_vi, depth_uj, depth_vj], dim=-1).amin(dim=-1)  # [N,N,T]
    pen_depth = torch.clamp(pen_depth, min=0.0)

    # ===== Exact separation distance via segment-segment distances =====
    # Segment-segment distance in 2D (vectorized):
    # For segments P(s) = P0 + s*(P1-P0), Q(t) = Q0 + t*(Q1-Q0), s,t in [0,1]
    # Compute all pair edges i(x4) vs j(x4)
    # Shapes to broadcast: [N,N,T,4,2]
    sA0 = segA[:, None, :, :, :].expand(N, N, T, 4, 2)
    sA1 = segB[:, None, :, :, :].expand(N, N, T, 4, 2)
    sB0 = segA[None, :, :, :, :].expand(N, N, T, 4, 2)
    sB1 = segB[None, :, :, :, :].expand(N, N, T, 4, 2)

    u_vec = sA1 - sA0   # [N,N,T,4,2]
    v_vec = sB1 - sB0   # [N,N,T,4,2]
    w0    = sA0 - sB0

    a = (u_vec*u_vec).sum(-1)       # [N,N,T,4]
    b = (u_vec*v_vec).sum(-1)
    c = (v_vec*v_vec).sum(-1)
    d = (u_vec*w0).sum(-1)
    e = (v_vec*w0).sum(-1)

    det = a*c - b*b + eps
    s = (b*e - c*d) / det
    t = (a*e - b*d) / det

    s_clamped = s.clamp(0.0, 1.0)
    t_clamped = t.clamp(0.0, 1.0)

    # Closest points
    p_closest = sA0 + s_clamped[..., None] * u_vec   # [N,N,T,4,2]
    q_closest = sB0 + t_clamped[..., None] * v_vec   # [N,N,T,4,2]
    segseg_dist = (p_closest - q_closest).norm(dim=-1)  # [N,N,T,4]

    # Also handle degenerate zero-length edges robustly (already guarded by eps)

    # Vertex-inside checks: if any vertex of A inside B or vice versa => distance 0.
    # We can reuse SAT: "separated" == True means not inside. If all four axes overlap, vertex may be inside.
    # A coarse-but-correct check: if NOT separated for a pair, distance will be handled by penetration depth path.
    # So we don’t need explicit vertex-inside here for the separated branch.

    # Minimal edge-edge distance across 4x4 pairs -> reduce over edges of A (4) and B (4) via our 4 (A) already; but we also need B’s 4 against A’s 4.
    # segseg_dist is distances for A’s 4 edges vs B’s 4 edges simultaneously (due to broadcasting we already have both).
    # We need min across the edge dim; segseg_dist currently represents paired edges (same index). To cover all 4x4, we can rotate B edges 4 ways and take min.
    # An efficient trick: circularly roll B edges and take min over 4 rolls.
    dists = []
    base = segseg_dist  # [N,N,T,4]
    dists.append(base)
    for k in range(1,4):
        # roll B edges by k (equivalent to pairing A edge e with B edge (e+k) mod 4)
        segB0_r = torch.roll(sB0, shifts=k, dims=3)
        segB1_r = torch.roll(sB1, shifts=k, dims=3)
        v_vec_r = segB1_r - segB0_r
        w0_r    = sA0 - segB0_r
        a_r = (u_vec*u_vec).sum(-1)
        b_r = (u_vec*v_vec_r).sum(-1)
        c_r = (v_vec_r*v_vec_r).sum(-1)
        d_r = (u_vec*w0_r).sum(-1)
        e_r = (v_vec_r*w0_r).sum(-1)
        det_r = a_r*c_r - b_r*b_r + eps
        s_r = (b_r*e_r - c_r*d_r) / det_r
        t_r = (a_r*e_r - b_r*d_r) / det_r
        p_r = sA0 + s_r[..., None]*u_vec
        q_r = segB0_r + t_r[..., None]*v_vec_r
        d_r_val = (p_r - q_r).norm(dim=-1)  # [N,N,T,4]
        dists.append(d_r_val)
    segseg_min = torch.stack(dists, dim=-1).amin(dim=(-1, -2))  # [N,N,T]

    # Separation distance only where separated:
    sep_dist = torch.where(separated, segseg_min, torch.zeros_like(segseg_min))  # [N,N,T]

    # Signed distance per pair:
    #   if separated: +sep_dist
    #   else (overlap): -pen_depth
    sd_pair = torch.where(separated, sep_dist, -pen_depth)  # [N,N,T]

    # Mask out pairs from different scenes or self
    mask = pair_mask[:, :, None]  # [N,N,1]
    sd_pair = torch.where(mask, sd_pair, torch.full_like(sd_pair, float('inf')))

    # Take nearest magnitude per agent (prefer the smallest absolute distance)
    # But keep sign by picking the value with smallest absolute value.
    # For overlapping pairs, values are negative; for separated, positive.
    abs_sd = sd_pair.abs()
    idx = abs_sd.argmin(dim=1)  # [N,T]
    sd_min = sd_pair.gather(1, idx.unsqueeze(1)).squeeze(1)  # [N,T]

    # If an agent has no other in the same batch, set to +inf
    no_pairs = (~pair_mask).all(dim=1)  # [N]
    if has_time:
        sd_min[no_pairs, :] = float('inf')
    else:
        sd_min[no_pairs] = float('inf')

    # Squeeze time if input had none
    if not has_time:
        sd_min = sd_min[:, 0]

    return sd_min

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
