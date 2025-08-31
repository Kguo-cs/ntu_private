import torch
from typing import Tuple

EXTREMELY_LARGE_DISTANCE = 1e9

def _dot2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

@torch.no_grad()
def _signed_distance_points_to_polylines_knn_2d_fast(
    xys: torch.Tensor,          # [M, 2] points (corners)
    point_batch: torch.Tensor,  # [M] scene id per point
    polylines_xy: torch.Tensor, # [P, S+1, 2] CCW boundary vertices
    poly_batch: torch.Tensor,   # [P] scene id per polyline
    knn_k: int = 16,
) -> torch.Tensor:
    """
    Memory/time-efficient signed 2D distance per point:
      1) Select top-K polylines in the *same* batch (by centroid distance).
      2) For each (point, polyline), compute the minimum squared distance to its segments,
         and remember the winning segment index.
      3) Reduce over K to pick the best polyline per point.
      4) Compute the sign **only for the winning (polyline, segment)** using the TF rule
         (with local convexity around vertices, non-cyclic padding).
    Returns:
      sd: [M] signed distances.
    """
    device = xys.device
    xys = xys.to(torch.float32)
    polylines_xy = polylines_xy.to(torch.float32)
    point_batch  = point_batch.to(polylines_xy.device)
    poly_batch   = poly_batch.to(polylines_xy.device)

    M = xys.shape[0]
    P, Sp1, _ = polylines_xy.shape
    S1 = Sp1 - 1
    if P == 0 or S1 <= 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    # Precompute per-polyline segments and convexity (point-independent)
    starts = polylines_xy[:, :-1, :]             # [P, S1, 2]
    ends   = polylines_xy[:,  1:, :]
    V      = ends - starts                       # [P, S1, 2]
    V_len2 = _dot2d(V, V).clamp_min(1e-12)       # [P, S1]  (avoid div-by-0)

    seg_pad = torch.cat([V[:, -1:, :], V, V[:, :1, :]], dim=1)      # [P, S1+2, 2]
    is_locally_convex = (_cross2d(seg_pad[:, :-1, :], seg_pad[:, 1:, :]) > 0.0)  # [P, S1+1]

    # kNN polylines per point (mask cross-batch with +inf)
    poly_centroid = polylines_xy.mean(dim=1)                          # [P, 2]
    D = torch.cdist(xys, poly_centroid)                               # [M, P]
    same = (point_batch[:, None] == poly_batch[None, :])              # [M, P]
    D = torch.where(same, D, torch.full_like(D, float("inf")))
    m = int(min(knn_k, P))
    if m == 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    knn_idx = D.topk(m, largest=False).indices                        # [M, m]
    # Build (point, polyline) pairs and drop invalid (inf) ones
    row = torch.arange(M, device=device).repeat_interleave(m)         # [M*m]
    col = knn_idx.reshape(-1)                                         # [M*m]
    valid = torch.isfinite(D[row, col])
    if not valid.any():
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    pi = row[valid]        # [Q] point indices
    pj = col[valid]        # [Q] polyline indices
    Q  = pi.numel()

    # For each (point, polyline), compute min squared distance to segments
    X   = xys[pi]                          # [Q, 2]
    B0  = starts[pj]                       # [Q, S1, 2]
    VV  = V[pj]                            # [Q, S1, 2]
    VL2 = V_len2[pj]                       # [Q, S1]

    start_to_point = X[:, None, :] - B0    # [Q, S1, 2]
    t = _dot2d(start_to_point, VV) / VL2   # [Q, S1]
    t_clamped = t.clamp(0.0, 1.0)[..., None]
    diff = start_to_point - t_clamped * VV             # [Q, S1, 2]
    dist2 = _dot2d(diff, diff)                         # [Q, S1]  (squared)

    seg_min_idx = dist2.argmin(dim=-1)                 # [Q]
    pair_min2   = dist2.gather(1, seg_min_idx[:, None]).squeeze(1)  # [Q]

    # Reduce over polylines -> min per point (distance only)
    out_min2 = torch.full((M,), float("inf"), device=device)
    out_min2 = out_min2.scatter_reduce(0, pi, pair_min2, reduce="amin", include_self=True)

    # Identify winning (point, polyline) pair per point (first winner)
    order = torch.arange(Q, device=device)
    is_winner = pair_min2 <= (out_min2[pi] + 1e-9)
    win_order = torch.where(is_winner, order, torch.full_like(order, Q))
    best_pair = torch.full((M,), Q, device=device).scatter_reduce(
        0, pi, win_order, reduce="amin", include_self=True
    )  # [M], each in [0..Q] (Q means none)

    has = best_pair < Q
    # Prepare outputs
    sd = torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    if has.any():
        bp = best_pair[has]          # [M_sel]
        p_sel = pi[bp]               # [M_sel] point idx
        j_sel = pj[bp]               # [M_sel] polyline idx
        k_sel = seg_min_idx[bp]      # [M_sel] winning segment idx (per selected pair)

        # Gather geometry for the winning segment (and neighbors for sign)
        B0_sel   = starts[j_sel, k_sel]                          # [M_sel, 2]
        V_sel    = V[j_sel, k_sel]                               # [M_sel, 2]
        V_prev   = V[j_sel, torch.clamp(k_sel - 1, min=0)]       # [M_sel, 2]
        V_next   = V[j_sel, torch.clamp(k_sel + 1, max=S1-1)]    # [M_sel, 2]
        # Local convexity flags
        ilc_before = is_locally_convex[j_sel, torch.clamp(k_sel,     min=0, max=S1)]     # [M_sel]
        ilc_after  = is_locally_convex[j_sel, torch.clamp(k_sel + 1, min=0, max=S1)]     # [M_sel]

        X_sel = xys[p_sel]                                       # [M_sel, 2]

        # Compute rel_t (unclamped) and n, n_prior, n_next for sign rule
        stp0   = X_sel - B0_sel                                  # [M_sel, 2]
        VL2_0  = _dot2d(V_sel, V_sel).clamp_min(1e-12)           # [M_sel]
        rel_t0 = _dot2d(stp0, V_sel) / VL2_0                     # [M_sel]

        n0     = torch.sign(_cross2d(stp0, V_sel))               # [M_sel]

        # For neighbor segments, use their own starts:
        B0_prev = starts[j_sel, torch.clamp(k_sel - 1, min=0)]   # [M_sel, 2]
        B0_next = starts[j_sel, torch.clamp(k_sel + 1, max=S1-1)]
        stp_prev = X_sel - B0_prev
        stp_next = X_sel - B0_next
        n_prev = torch.sign(_cross2d(stp_prev, V_prev))          # [M_sel]
        n_next = torch.sign(_cross2d(stp_next, V_next))          # [M_sel]

        sign_before = torch.where(ilc_before, torch.maximum(n0, n_prev), torch.minimum(n0, n_prev))
        sign_after  = torch.where(ilc_after,  torch.maximum(n0, n_next), torch.minimum(n0, n_next))
        sign = torch.where(rel_t0 < 0.0, sign_before,
                           torch.where(rel_t0 > 1.0, sign_after, n0)).to(torch.float32)  # [M_sel]

        sd_val = torch.sqrt(out_min2[p_sel]) * sign              # [M_sel]
        sd[p_sel] = sd_val

    return sd  # [M]


@torch.no_grad()
def corners_offroad_signed_distance_batched_2d_knn_fast(
    pos: torch.Tensor,            # [N,2] or [N,T,2]
    heading: torch.Tensor,        # [N] or [N,T] (radians)
    shape_lw: torch.Tensor,       # [N,2]  (L, W)
    agent_batch: torch.Tensor,    # [N] scene ids for agents
    polylines_xy: torch.Tensor,   # [P, S+1, 2] CCW boundary vertices
    poly_batch: torch.Tensor,     # [P] scene ids for polylines
    knn_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient pipeline using the 'fast' k-NN distance core:
      - builds 4 OBB corners,
      - computes signed distance per corner with k-NN + winner-only sign,
      - reshapes to [N,T,4], takes max over corners -> [N,T].
    """
    # unify time
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    N, T, _ = pos.shape

    # half-extents
    hl = shape_lw[:, 0] * 0.5
    hw = shape_lw[:, 1] * 0.5

    # local axes
    c = torch.cos(heading); s = torch.sin(heading)
    u = torch.stack([c, s], dim=-1)          # [N,T,2] forward
    v = torch.stack([-s, c], dim=-1)         # [N,T,2] left

    # four corners in local (±hl, ±hw)
    offs = torch.stack([
        torch.stack([ hl,  hw], dim=-1),
        torch.stack([ hl, -hw], dim=-1),
        torch.stack([-hl, -hw], dim=-1),
        torch.stack([-hl,  hw], dim=-1),
    ], dim=1)  # [N,4,2]

    offs_t = offs[:, None, :, :]             # [N,1,4,2]
    u4 = u[:, :, None, :]                    # [N,T,1,2]
    v4 = v[:, :, None, :]                    # [N,T,1,2]
    corners_xy = pos[:, :, None, :] + offs_t[..., :1] * u4 + offs_t[..., 1:] * v4  # [N,T,4,2]

    # Flatten corners to [M,2] and batches to [M]
    M = N * T * 4
    flat_eval_corners = corners_xy.reshape(M, 2)
    flat_point_batch  = agent_batch[:, None, None].expand(N, T, 4).reshape(M)

    # Compute signed distances with the fast k-NN core
    flat_sd = _signed_distance_points_to_polylines_knn_2d_fast(
        xys=flat_eval_corners,
        point_batch=flat_point_batch,
        polylines_xy=polylines_xy,
        poly_batch=poly_batch,
        knn_k=knn_k,
    )  # [M]

    # Reshape and reduce max over corners
    corner_distance_to_road_edge = flat_sd.view(N, T, 4)         # [N,T,4]
    signed_distances = corner_distance_to_road_edge.max(dim=-1).values  # [N,T]

    if not has_time:
        corner_distance_to_road_edge = corner_distance_to_road_edge[:, 0, :]
        signed_distances = signed_distances[:, 0]

    return corner_distance_to_road_edge, signed_distances
