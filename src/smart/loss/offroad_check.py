import torch

EXTREMELY_LARGE_DISTANCE = 1e9

def _dot2d(a, b):  # [...,2]
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

def _cross2d(a, b):  # scalar z of 2D cross
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

@torch.no_grad()
def _signed_distance_points_to_polylines_knn_2d(
    xys: torch.Tensor,          # [M, 2] points (corners)
    point_batch: torch.Tensor,  # [M] scene id per point
    polylines_xy: torch.Tensor, # [P, S+1, 2] CCW boundary vertices
    poly_batch: torch.Tensor,   # [P] scene id per polyline
    knn_k: int = 16,
) -> torch.Tensor:
    """
    Signed 2D distance from each point to the nearest segment among its top-K
    nearest polylines in the *same* batch. CCW: left/port => negative.
    Returns: [M] float32.
    """
    device = xys.device
    xys = xys.to(torch.float32)
    polylines_xy = polylines_xy.to(torch.float32)
    point_batch = point_batch.to(polylines_xy.device)
    poly_batch  = poly_batch.to(polylines_xy.device)

    M = xys.shape[0]
    P, Sp1, _ = polylines_xy.shape
    S1 = Sp1 - 1
    if P == 0 or S1 <= 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    # Precompute segments & convexity (shared)
    starts = polylines_xy[:, :-1, :]         # [P, S1, 2]
    ends   = polylines_xy[:,  1:, :]
    seg    = ends - starts                   # [P, S1, 2]
    seg_pad = torch.cat([seg[:, -1:, :], seg, seg[:, :1, :]], dim=1)          # [P,S1+2,2]
    is_locally_convex = (_cross2d(seg_pad[:, :-1, :], seg_pad[:, 1:, :]) > 0) # [P,S1+1]

    # kNN polylines per point (mask cross-batch with +inf)
    poly_centroid = polylines_xy.mean(dim=1)             # [P,2]
    D = torch.cdist(xys, poly_centroid)                  # [M,P]
    same = (point_batch[:, None] == poly_batch[None, :]) # [M,P]
    D = torch.where(same, D, torch.full_like(D, float("inf")))
    m = min(knn_k, P)
    if m == 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    knn_idx = D.topk(m, largest=False).indices           # [M,m]
    row = torch.arange(M, device=device).repeat_interleave(m)  # [M*m]
    col = knn_idx.reshape(-1)                                  # [M*m]
    valid = torch.isfinite(D[row, col])
    if not valid.any():
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    # Selected pairs
    pi = row[valid]      # [Q] point indices (0..M-1)
    pj = col[valid]      # [Q] polyline indices (0..P-1)
    Q  = pi.numel()

    # Compute point-to-segments for each (point, polyline) pair
    B0 = starts[pj]                            # [Q, S1, 2]
    V  = seg[pj]                               # [Q, S1, 2]
    ilc = is_locally_convex[pj]                # [Q, S1+1]

    X  = xys[pi]                                # [Q, 2]
    start_to_point = X[:, None, :] - B0         # [Q, S1, 2]
    start_to_end   = V                          # [Q, S1, 2]

    denom = _dot2d(start_to_end, start_to_end)  # [Q, S1]
    num   = _dot2d(start_to_point, start_to_end)
    rel_t = torch.where(denom > 0, num / denom, torch.zeros_like(num))  # [Q, S1]

    n = torch.sign(_cross2d(start_to_point, start_to_end))              # [Q, S1]

    rel_t_c = rel_t.clamp(0.0, 1.0)[..., None]
    seg_to_point = start_to_point - start_to_end * rel_t_c              # [Q, S1, 2]
    dist2d = torch.linalg.norm(seg_to_point, dim=-1)                    # [Q, S1]

    # Non-cyclic neighbors and convexity
    n_prior = torch.cat([n[:, :1],  n[:, :-1]],  dim=-1)  # [Q, S1]
    n_next  = torch.cat([n[:, 1:],  n[:, -1:]],  dim=-1)  # [Q, S1]
    ilc_before = ilc[:, :-1]                              # [Q, S1]
    ilc_after  = ilc[:,  1:]                              # [Q, S1]

    sign_if_before = torch.where(ilc_before, torch.maximum(n, n_prior), torch.minimum(n, n_prior))
    sign_if_after  = torch.where(ilc_after,  torch.maximum(n, n_next),  torch.minimum(n, n_next))

    sign_to_segment = torch.where(
        (rel_t < 0.0), sign_if_before,
        torch.where((rel_t > 1.0), sign_if_after, n)
    )  # [Q, S1]

    # For each (point, polyline) pair, choose nearest segment & its sign
    seg_min_idx = dist2d.argmin(dim=-1)                                  # [Q]
    dist_pair   = dist2d.gather(1, seg_min_idx.unsqueeze(1)).squeeze(1)  # [Q]
    sign_pair   = sign_to_segment.gather(1, seg_min_idx.unsqueeze(1)).squeeze(1)  # [Q]

    # ---- Reduce across polylines to 1 result per point ----
    # 1) per-point min distance via scatter_reduce(amin)
    out_dist = torch.full((M,), float("inf"), device=device)
    if hasattr(out_dist, "scatter_reduce"):
        out_dist = out_dist.scatter_reduce(0, pi, dist_pair, reduce="amin", include_self=True)
        # 2) identify which pair(s) achieved that min (with tolerance), pick first
        order = torch.arange(Q, device=device)
        is_winner = dist_pair <= (out_dist[pi] + 1e-6)      # [Q]
        win_order = torch.where(is_winner, order, torch.full_like(order, Q))
        best_order = torch.full((M,), Q, device=device).scatter_reduce(
            0, pi, win_order, reduce="amin", include_self=True
        )  # [M], each in [0..Q] (Q means none)
        # map to sign; default +1 if no winner (shouldn’t happen if valid.any())
        out_sign = torch.ones((M,), device=device, dtype=sign_pair.dtype)
        has = best_order < Q
        out_sign[has] = sign_pair[best_order[has]]
    else:
        # Fallback for older PyTorch: tiny per-point loop (K is small)
        out_sign = torch.ones((M,), device=device, dtype=sign_pair.dtype)
        for p in torch.unique(pi):
            mask = (pi == p)
            dmin, k = dist_pair[mask].min(dim=0)
            out_dist[p] = dmin
            out_sign[p] = sign_pair[mask][k]

    return out_sign * out_dist  # [M]

@torch.no_grad()
def corners_offroad_signed_distance_batched_2d_knn(
    pos: torch.Tensor,            # [N,2] or [N,T,2]
    heading: torch.Tensor,        # [N] or [N,T] (radians)
    shape_lw: torch.Tensor,       # [N,2]  (L, W)
    agent_batch: torch.Tensor,    # [N] scene ids for agents
    polylines_xy: torch.Tensor,   # [P, S+1, 2] CCW boundary vertices
    poly_batch: torch.Tensor,     # [P] scene ids for polylines
    knn_k: int = 16,
) :
    """
    Efficient pipeline:
      1) build 4 OBB corners per agent/time,
      2) signed distance to nearest boundary using k-NN polylines within batch,
      3) reshape to [N,T,4], take max over corners.
    Returns:
      corner_distance_to_road_edge: [N,T,4]
      signed_distances:             [N,T]
    """
    # unify time
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    N, T, _ = pos.shape
    device = pos.device

    # half-extents
    hl = shape_lw[:, 0] * 0.5
    hw = shape_lw[:, 1] * 0.5

    # local axes
    c = torch.cos(heading); s = torch.sin(heading)
    u = torch.stack([c, s], dim=-1)          # [N,T,2] forward
    v = torch.stack([-s, c], dim=-1)         # [N,T,2] left

    # four corners in local (±hl, ±hw): (+,+), (+,-), (-,-), (-,+)
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

    # Compute signed distances with k-NN preselection
    flat_sd = _signed_distance_points_to_polylines_knn_2d(
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
