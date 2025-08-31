import torch
from typing import Tuple

# ---------- small helpers ----------
def _dot2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@torch.no_grad()
def _sd_points_to_polylines_knn_2d_streaming(
    xys: torch.Tensor,          # [M, 2] points
    polylines_xy: torch.Tensor, # [P, S+1, 2] (only this scene)
    knn_k: int = 16,
    points_chunk: int = 8192,
    polys_chunk: int = 4096,
    seg_chunk: int = 512,
) -> torch.Tensor:
    """
    Streaming, low-memory signed 2D distance for ONE scene (no batch masking).
    CCW boundary => port/left is INSIDE => negative distance.
    Returns: [M]
    """
    device = xys.device
    xys = xys.to(torch.float32)
    polylines_xy = polylines_xy.to(torch.float32)

    M = xys.shape[0]
    P, Sp1, _ = polylines_xy.shape
    S1 = Sp1 - 1
    if P == 0 or S1 <= 0 or M == 0:
        return torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    # Precompute per-polyline segments & convexity once
    starts = polylines_xy[:, :-1, :]       # [P, S1, 2]
    ends   = polylines_xy[:,  1:, :]
    V      = ends - starts                 # [P, S1, 2]
    V_len2 = _dot2d(V, V).clamp_min(1e-12) # [P, S1]
    V_pad  = torch.cat([V[:, -1:, :], V, V[:, :1, :]], dim=1)         # [P, S1+2, 2]
    is_locally_convex = (_cross2d(V_pad[:, :-1, :], V_pad[:, 1:, :]) > 0.0)  # [P, S1+1]

    # Centroids for KNN
    centroids = polylines_xy.mean(dim=1)   # [P, 2]

    # Outputs per point (winner info)
    best_min2 = torch.full((M,), float("inf"), device=device)  # squared distance
    best_poly = torch.full((M,), -1, device=device, dtype=torch.long)
    best_seg  = torch.full((M,), -1, device=device, dtype=torch.long)

    # Process points in chunks to cap RAM
    for p0 in range(0, M, points_chunk):
        p1 = min(p0 + points_chunk, M)
        X = xys[p0:p1]     # [m,2]
        m = X.shape[0]

        # Streaming KNN over polylines (within the scene)
        best_d = torch.full((m, knn_k), float("inf"), device=device)
        best_j = torch.full((m, knn_k), -1, device=device, dtype=torch.long)

        for q0 in range(0, P, polys_chunk):
            q1 = min(q0 + polys_chunk, P)
            C  = centroids[q0:q1]          # [pc,2]
            D  = torch.cdist(X, C)         # [m,pc]
            # Merge into running top-k
            cat_d = torch.cat([best_d, D], dim=1)                          # [m, k+pc]
            cur_j = best_j
            new_j = torch.arange(q0, q1, device=device).repeat(m, 1)       # [m,pc]
            cat_j = torch.cat([cur_j, new_j], dim=1)                       # [m, k+pc]
            vals, inds = torch.topk(cat_d, k=knn_k, dim=1, largest=False)  # [m,k]
            best_d = vals
            best_j = cat_j.gather(1, inds)

        # Skip if no neighbors found
        has_neighbors = (best_j[:, 0] >= 0)
        if not has_neighbors.any():
            continue

        # For each point, scan selected polylines and find nearest segment
        ksel = best_j  # [m,k]
        for k0 in range(0, knn_k, min(knn_k, 32)):   # small neighbor blocks
            k1 = min(k0 + 32, knn_k)
            j_block = ksel[:, k0:k1]                 # [m,kb]
            mask = (j_block >= 0)
            if not mask.any():
                continue
            pi, pj = torch.nonzero(mask, as_tuple=True)     # pi in [0..m-1], pj in [0..kb-1]
            j_idx  = j_block[pi, pj]                        # [Q]
            if j_idx.numel() == 0:
                continue
            Xq = X[pi]                                      # [Q,2]

            pair_min2 = torch.full((Xq.shape[0],), float("inf"), device=device)
            pair_seg  = torch.full((Xq.shape[0],), -1, device=device, dtype=torch.long)

            # Stream segments in blocks
            for s0 in range(0, S1, seg_chunk):
                s1 = min(s0 + seg_chunk, S1)
                B0 = starts[j_idx, s0:s1, :]                # [Q,sc,2]
                VV = V[j_idx,     s0:s1, :]                 # [Q,sc,2]
                VL2= V_len2[j_idx, s0:s1]                   # [Q,sc]

                stp = Xq[:, None, :] - B0                   # [Q,sc,2]
                t   = _dot2d(stp, VV) / VL2                 # [Q,sc]
                t   = t.clamp(0.0, 1.0)[..., None]          # [Q,sc,1]
                diff= stp - t * VV                          # [Q,sc,2]
                d2  = _dot2d(diff, diff)                    # [Q,sc]

                d2_min, seg_local = d2.min(dim=1)           # [Q], [Q]
                better = d2_min < pair_min2
                pair_min2[better] = d2_min[better]
                pair_seg [better] = (s0 + seg_local)[better]

            # Update per-point global best with these pairs
            gpi = p0 + pi
            better = pair_min2 < best_min2[gpi]
            best_min2[gpi[better]] = pair_min2[better]
            best_poly[gpi[better]] = j_idx[better]
            best_seg [gpi[better]] = pair_seg[better]

    # Winner-only sign for points that found a neighbor
    sd = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
    winners = best_poly >= 0
    if winners.any():
        gi = torch.nonzero(winners, as_tuple=True)[0]
        j  = best_poly[gi]
        k  = best_seg [gi]
        Xw = xys[gi]

        B0 = starts[j, k]                                   # [Mw,2]
        V0 = V[j, k]                                        # [Mw,2]
        k_prev = torch.clamp(k - 1, min=0)
        k_next = torch.clamp(k + 1, max=(S1 - 1))
        V_prev = V[j, k_prev]
        V_next = V[j, k_next]
        ilc_before = is_locally_convex[j, torch.clamp(k,     min=0, max=S1)]
        ilc_after  = is_locally_convex[j, torch.clamp(k + 1, min=0, max=S1)]

        stp0   = Xw - B0
        len2_0 = _dot2d(V0, V0).clamp_min(1e-12)
        rel_t0 = _dot2d(stp0, V0) / len2_0
        n0     = torch.sign(_cross2d(stp0, V0))

        B0_prev = starts[j, k_prev]
        B0_next = starts[j, k_next]
        n_prev  = torch.sign(_cross2d(Xw - B0_prev, V_prev))
        n_next  = torch.sign(_cross2d(Xw - B0_next, V_next))

        sign_before = torch.where(ilc_before, torch.maximum(n0, n_prev), torch.minimum(n0, n_prev))
        sign_after  = torch.where(ilc_after,  torch.maximum(n0, n_next), torch.minimum(n0, n_next))
        sign = torch.where(rel_t0 < 0.0, sign_before,
                           torch.where(rel_t0 > 1.0, sign_after, n0)).to(torch.float32)

        sd_val = torch.sqrt(best_min2[gi]) * sign
        sd[gi] = sd_val

    return sd  # [M]


@torch.no_grad()
def corners_offroad_signed_distance_chunked_pre_batch(
    pos: torch.Tensor,            # [N,2] or [N,T,2]
    heading: torch.Tensor,        # [N] or [N,T] (radians)
    shape_lw: torch.Tensor,       # [N,2]  (L, W)
    agent_batch: torch.Tensor,    # [N] scene id per agent
    polylines_xy: torch.Tensor,   # [P, S+1, 2] CCW vertices
    poly_batch: torch.Tensor,     # [P] scene id per polyline
    knn_k: int = 16,
    points_chunk: int = 8192,
    polys_chunk: int = 4096,
    seg_chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CHUNKED-PER-BATCH version (low RAM):
      - loop over scene ids,
      - compute corners only for that scene,
      - run streaming kNN+scan within the scene,
      - scatter to global outputs.

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

    # preallocate outputs
    corner_d2edge = torch.full((N, T, 4), float("inf"), device=device)
    signed_dists  = torch.full((N, T),   float("inf"), device=device)

    # unique scene ids present in agents
    scene_ids = torch.unique(agent_batch)

    for b in scene_ids:
        idx_a = torch.nonzero(agent_batch == b, as_tuple=True)[0]  # agents in scene
        if idx_a.numel() == 0:
            continue
        idx_p = torch.nonzero(poly_batch == b, as_tuple=True)[0]   # polylines in scene
        if idx_p.numel() == 0:
            # leave +inf for this scene
            continue

        # slice scene tensors
        pos_b     = pos[idx_a]             # [Nb,T,2]
        heading_b = heading[idx_a]         # [Nb,T]
        shape_b   = shape_lw[idx_a]        # [Nb,2]
        polys_b   = polylines_xy[idx_p]    # [Pb,S+1,2]

        # corners for this scene
        hl = shape_b[:, 0] * 0.5
        hw = shape_b[:, 1] * 0.5
        c = torch.cos(heading_b); s = torch.sin(heading_b)
        u = torch.stack([c, s], dim=-1)        # [Nb,T,2]
        v = torch.stack([-s, c], dim=-1)       # [Nb,T,2]

        offs = torch.stack([
            torch.stack([ hl,  hw], dim=-1),
            torch.stack([ hl, -hw], dim=-1),
            torch.stack([-hl, -hw], dim=-1),
            torch.stack([-hl,  hw], dim=-1),
        ], dim=1)  # [Nb,4,2]

        offs_t = offs[:, None, :, :]           # [Nb,1,4,2]
        u4 = u[:, :, None, :]                  # [Nb,T,1,2]
        v4 = v[:, :, None, :]                  # [Nb,T,1,2]
        corners_xy = pos_b[:, :, None, :] + offs_t[..., :1] * u4 + offs_t[..., 1:] * v4  # [Nb,T,4,2]

        # flatten to [Mb,2]
        Nb = idx_a.numel()
        Mb = Nb * T * 4
        flat_corners = corners_xy.reshape(Mb, 2)

        # per-scene signed distances (streaming, no batch masks needed)
        flat_sd = _sd_points_to_polylines_knn_2d_streaming(
            xys=flat_corners,
            polylines_xy=polys_b,
            knn_k=knn_k,
            points_chunk=points_chunk,
            polys_chunk=polys_chunk,
            seg_chunk=seg_chunk,
        )  # [Mb]

        scene_corner = flat_sd.view(Nb, T, 4)        # [Nb,T,4]
        scene_signed = scene_corner.max(dim=-1).values  # [Nb,T]

        # scatter back
        corner_d2edge[idx_a] = scene_corner
        signed_dists[idx_a]  = scene_signed

    if not has_time:
        corner_d2edge = corner_d2edge[:, 0, :]
        signed_dists  = signed_dists[:, 0]

    return corner_d2edge, signed_dists
