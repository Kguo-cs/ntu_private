import torch
from typing import Tuple

EXTREMELY_LARGE_DISTANCE = 1e9

def _dot2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # standard 2D cross z-component: a_x*b_y - a_y*b_x
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@torch.no_grad()
def _signed_distance_points_to_polylines_knn_2d_batch(
    xys: torch.Tensor,          # [M, 2] corner points (ONE batch)
    polylines_xy: torch.Tensor, # [P, S+1, 2] CCW vertices (ONE batch)
    knn_k: int = 16,
) -> torch.Tensor:
    """
    Fully vectorized (no chunking) signed 2D distance per point to the nearest
    boundary among its K nearest polylines within this batch.

    Sign convention (TF-aligned):
      port/left of segment = INSIDE = NEGATIVE distance
      => n_base = -sign(cross(start->point, start->end))

    Returns: sd [M] (float32)
    """
    device = xys.device
    xys = xys.to(torch.float32)
    polylines_xy = polylines_xy.to(torch.float32)

    M = xys.shape[0]
    P, Sp1, _ = polylines_xy.shape
    S1 = Sp1 - 1
    if M == 0 or P == 0 or S1 <= 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    # Segments & per-polyline convexity (non-cyclic padding, TF-style)
    starts = polylines_xy[:, :-1, :]         # [P, S1, 2]
    ends   = polylines_xy[:,  1:, :]         # [P, S1, 2]
    V      = ends - starts                   # [P, S1, 2]

    V_pad = torch.cat([V[:, -1:, :], V, V[:, :1, :]], dim=1)   # [P, S1+2, 2]
    is_locally_convex = (_cross2d(V_pad[:, :-1, :], V_pad[:, 1:, :]) > 0.0)  # [P, S1+1]
    ilc_before_all = is_locally_convex[:, :-1]   # [P, S1]
    ilc_after_all  = is_locally_convex[:,  1:]   # [P, S1]

    # ---- kNN over polylines (centroids) ----
    poly_centroid = polylines_xy.mean(dim=1)          # [P, 2]
    D = torch.cdist(xys, poly_centroid)               # [M, P]
    k = int(min(knn_k, P))
    knn_idx = D.topk(k, largest=False).indices        # [M, k]

    # Build (point, polyline) pairs
    row = torch.arange(M, device=device).unsqueeze(1).expand(M, k)  # [M,k]
    pi = row.reshape(-1)                                            # [Q]
    pj = knn_idx.reshape(-1)                                        # [Q]
    Q  = pi.numel()

    # Gather geometry per pair: [Q, S1, *]
    B0 = starts[pj]                              # [Q, S1, 2]
    VV = V[pj]                                   # [Q, S1, 2]
    ilc_before = ilc_before_all[pj]              # [Q, S1]
    ilc_after  = ilc_after_all[pj]               # [Q, S1]

    X = xys[pi]                                  # [Q, 2]
    stp = X[:, None, :] - B0                     # [Q, S1, 2]  start->point
    ste = VV                                     # [Q, S1, 2]  start->end

    # Projection param (divide-no-nan via clamp)
    VL2 = _dot2d(ste, ste).clamp_min(1e-12)      # [Q, S1]
    t   = _dot2d(stp, ste) / VL2                 # [Q, S1]
    t_c = t.clamp(0.0, 1.0)[..., None]           # [Q, S1, 1]

    # Closest vector & distances
    diff   = stp - t_c * ste                     # [Q, S1, 2]
    dist2  = _dot2d(diff, diff)                  # [Q, S1] (squared)
    dist   = dist2.sqrt()                        # [Q, S1]

    # ----- TF sign rule (with correct convention) -----
    # Base sign for projection within segment:
    # TF doc: "Negative if point is on port side (inside)". With standard cross,
    #   left/port => cross(ste, stp) > 0. TF computes cross(stp, ste),
    #   so we must negate to match "port => negative".
    # Our _cross2d(stp, ste) > 0 means left => POS; negate to make it NEG.
    n_base = -torch.sign(_cross2d(stp, ste))     # [Q, S1]  (inside => -1)

    # For outside projection regions (before/after), mix signs using local convexity
    # Non-cyclic padding per TF: use edge neighbors (we already built ilc_* to align)
    # Shifted signs:
    n_prior = torch.cat([n_base[:, :1],  n_base[:, :-1]], dim=1)  # [Q,S1]
    n_next  = torch.cat([n_base[:, 1:],  n_base[:, -1:]], dim=1)  # [Q,S1]

    sign_if_before = torch.where(ilc_before, torch.maximum(n_base, n_prior),
                                              torch.minimum(n_base, n_prior))
    sign_if_after  = torch.where(ilc_after,  torch.maximum(n_base, n_next),
                                              torch.minimum(n_base, n_next))

    sign_to_segment = torch.where(
        (t < 0.0),  sign_if_before,
        torch.where((t > 1.0), sign_if_after, n_base)
    )  # [Q, S1]

    # Per (point, polyline) — pick nearest segment & its sign
    seg_min_idx = dist.argmin(dim=1)                              # [Q]
    pair_dist   = dist.gather(1, seg_min_idx[:, None]).squeeze(1) # [Q].mean(-1)#
    pair_sign   = sign_to_segment.gather(1, seg_min_idx[:, None]).squeeze(1).to(torch.float32) # [Q]

    # Reduce over the k polylines to 1 winner per point
    pair_dist_mk = pair_dist.view(M, k)         # [M,k]
    pair_sign_mk = pair_sign.view(M, k)         # [M,k]
    kmin = pair_dist_mk.argmin(dim=1)           # [M]
    dmin = pair_dist_mk.gather(1, kmin[:, None]).squeeze(1)   # [M]
    smin = pair_sign_mk.gather(1, kmin[:, None]).squeeze(1)   # [M]

    return smin * dmin                           # [M]  (negative inside)


@torch.no_grad()
def corners_offroad_signed_distance_per_batch(
    pos: torch.Tensor,            # [N,2] or [N,T,2]
    heading: torch.Tensor,        # [N] or [N,T] (radians)
    shape_lw: torch.Tensor,       # [N,2]  (L, W)
    agent_batch: torch.Tensor,    # [N] scene id per agent
    polylines_xy: torch.Tensor,   # [P, S+1, 2] CCW vertices
    poly_batch: torch.Tensor,     # [P] scene id per polyline
    knn_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-scene (batch) vectorized pipeline (no intra-batch chunking):
      - build 4 OBB corners,
      - kNN polylines (same scene),
      - signed 2D distance per corner using TF rule with correct sign,
      - reshape to [N,T,4], reduce max over corners -> [N,T].
    """
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    N, T, _ = pos.shape
    device = pos.device

    corner_d2edge = torch.full((N, T, 4), float("inf"), device=device)
    signed_dists  = torch.full((N, T),   float("inf"), device=device)

    for b in torch.unique(agent_batch):
        idx_a = torch.nonzero(agent_batch == b, as_tuple=True)[0]
        if idx_a.numel() == 0:
            continue
        idx_p = torch.nonzero(poly_batch == b, as_tuple=True)[0]
        if idx_p.numel() == 0:
            continue

        # Slice scene tensors
        pos_b     = pos[idx_a]              # [Nb,T,2]
        heading_b = heading[idx_a]          # [Nb,T]
        shape_b   = shape_lw[idx_a]         # [Nb,2]
        polys_b   = polylines_xy[idx_p]     # [Pb,S+1,2]

        # OBB corners
        hl = shape_b[:, 0] * 0.5
        hw = shape_b[:, 1] * 0.5
        c = torch.cos(heading_b); s = torch.sin(heading_b)
        u = torch.stack([c, s], dim=-1)     # [Nb,T,2] forward
        v = torch.stack([-s, c], dim=-1)    # [Nb,T,2] left

        offs = torch.stack([
            torch.stack([ hl,  hw], dim=-1),
            torch.stack([ hl, -hw], dim=-1),
            torch.stack([-hl, -hw], dim=-1),
            torch.stack([-hl,  hw], dim=-1),
        ], dim=1)  # [Nb,4,2]

        corners_xy = pos_b[:, :, None, :] + offs[:, None, :, 0:1] * u[:, :, None, :] \
                                       + offs[:, None, :, 1:2] * v[:, :, None, :]     # [Nb,T,4,2]

        # Flatten corners for this scene
        Nb = idx_a.numel()
        M  = Nb * T * 4
        flat_corners = corners_xy.reshape(M, 2)#[-1:]

        # Signed distance per corner (vectorized kNN within the scene)
        flat_sd = _signed_distance_points_to_polylines_knn_2d_batch(
            xys=flat_corners,
            polylines_xy=polys_b,
            knn_k=knn_k,
        )  # [M]

        scene_corner = flat_sd.view(Nb, T, 4)               # [Nb,T,4]
        scene_signed = scene_corner.min(dim=-1).values      # [Nb,T]

        corner_d2edge[idx_a] = scene_corner
        signed_dists[idx_a]  = scene_signed

    if not has_time:
        corner_d2edge = corner_d2edge[:, 0, :]
        signed_dists  = signed_dists[:, 0]

    return corner_d2edge, signed_dists
