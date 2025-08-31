import torch
from typing import Tuple


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
