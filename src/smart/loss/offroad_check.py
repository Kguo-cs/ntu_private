import torch

def signed_distance_capsule_to_oriented_boundary_knn(
    pos: torch.Tensor,           # [N,2] or [N,T,2]
    heading: torch.Tensor,       # [N] or [N,T] (radians)
    shape_lw: torch.Tensor,      # [N,2] (L, W)
    agent_batch: torch.Tensor,   # [N]
    polylines: torch.Tensor,     # [P,S,2] (CCW boundary; S>=2)
    poly_batch: torch.Tensor,    # [P]
    knn_k: int = 32,
    margin: float = 0.0,
    use_mean_pos_for_knn: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Signed distance from each agent capsule (L along heading, radius W/2+margin)
    to the nearest oriented boundary segment. Negative = inside, Positive = outside."""
    device = pos.device
    has_time = (pos.ndim == 3)
    if not has_time:
        pos = pos[:, None, :]
        heading = heading[:, None]
    N, T, _ = pos.shape

    P, S, _ = polylines.shape
    S1 = S - 1
    if P == 0 or S1 <= 0:
        out = torch.full((N, T), float("inf"), device=device)
        return out[:, 0] if not has_time else out

    # ---- agent capsule (segment + radius) ----
    L   = shape_lw[:, 0]                    # [N]
    rad = shape_lw[:, 1] * 0.5 + margin     # [N]  <-- keep this name!
    c = torch.cos(heading); s = torch.sin(heading)
    u = torch.stack([c, s], dim=-1)         # [N,T,2] forward axis
    half = (L[:, None] * 0.5)               # [N,1]
    A0 = pos - half[..., None] * u          # [N,T,2]
    A1 = pos + half[..., None] * u          # [N,T,2]

    # ---- boundary segments ----
    B0_all = polylines[:, :-1, :]           # [P,S1,2]
    B1_all = polylines[:,  1:, :]
    poly_centroid = polylines.mean(dim=1)   # [P,2]

    # ---- kNN over polylines per agent (no batch loops) ----
    Aref = pos.mean(dim=1) if use_mean_pos_for_knn else pos[:, 0, :]  # [N,2]
    D = torch.cdist(Aref, poly_centroid)                               # [N,P]
    same = (agent_batch[:, None] == poly_batch[None, :])
    D = torch.where(same, D, torch.full_like(D, float("inf")))
    m = min(knn_k, P)
    if m == 0:
        out = torch.full((N, T), float("inf"), device=device)
        return out[:, 0] if not has_time else out
    knn_idx = D.topk(m, largest=False).indices                         # [N,m]

    # flattened (agent, slot, poly) mapping
    i_flat = torch.arange(N, device=device).unsqueeze(1).expand(N, m).reshape(-1)
    s_flat = torch.arange(m, device=device).unsqueeze(0).expand(N, m).reshape(-1)
    j_flat = knn_idx.reshape(-1)
    valid  = torch.isfinite(D[i_flat, j_flat])

    # per-(N,m,T) containers for reduction
    mag_ntm  = torch.full((N, m, T), float("inf"), device=device)  # magnitude = |dist - rad|
    sign_ntm = torch.ones((N, m, T), device=device)                # boundary sign (+1/-1)

    if valid.any():
        i_v, s_v, j_v = i_flat[valid], s_flat[valid], j_flat[valid]

        # gather agent segments
        A0p = A0[i_v]                 # [Psel,T,2]
        A1p = A1[i_v]
        u0  = A1p - A0p               # [Psel,T,2]

        # gather boundary segments
        B0p = B0_all[j_v]             # [Psel,S1,2]
        B1p = B1_all[j_v]
        V   = (B1p - B0p)             # [Psel,S1,2]
        Vn  = V / (V.norm(dim=-1, keepdim=True).clamp_min(eps))

        # broadcast for segment–segment closest points
        A0b = A0p[:, :, None, :]      # [Psel,T,1,2]
        u0b = u0[:,  :, None, :]
        B0b = B0p[:, None, :, :]      # [Psel,1,S1,2]
        Vb  = V[:,  None, :, :]
        Vnb = Vn[:, None, :, :]

        w0 = A0b - B0b                # [Psel,T,S1,2]
        a  = (u0b * u0b).sum(-1)      # [Psel,T,1]
        b  = (u0b * Vb ).sum(-1)      # [Psel,T,S1]
        c2 = (Vb  * Vb ).sum(-1)      # [Psel,1,S1]
        d  = (u0b * w0 ).sum(-1)      # [Psel,T,S1]
        e  = (Vb  * w0 ).sum(-1)      # [Psel,T,S1]

        denom = a * c2 - b * b
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)

        s = ((b * e - c2 * d) / denom).clamp(0.0, 1.0)   # [Psel,T,S1]
        t = ((a * e - b  * d) / denom).clamp(0.0, 1.0)

        P_cl = A0b + s[..., None] * u0b     # [Psel,T,S1,2] agent closest point
        Q_cl = B0b + t[..., None] * Vb      # [Psel,T,S1,2] boundary closest point
        dvec = P_cl - Q_cl                  # [Psel,T,S1,2]
        dist = dvec.norm(dim=-1)            # [Psel,T,S1]

        # nearest segment per (pair,time)
        dist_min, seg_idx = dist.min(dim=2)  # [Psel,T]
        # gather Vn and closest points at that segment to compute sign
        sel = seg_idx[..., None, None].expand(-1, -1, 1, 2)
        Vn_min = Vnb.expand(-1, T, -1, -1).gather(2, sel).squeeze(2)  # [Psel,T,2]
        Q_min  = Q_cl.gather(2, sel).squeeze(2)                       # [Psel,T,2]
        P_min  = P_cl.gather(2, sel).squeeze(2)                       # [Psel,T,2]
        r_min  = P_min - Q_min                                        # [Psel,T,2]

        # boundary orientation sign: cross(Vn_min, r_min)
        cross_z = Vn_min[..., 0]*r_min[..., 1] - Vn_min[..., 1]*r_min[..., 0]  # [Psel,T]
        sign_boundary = torch.where(cross_z > 0, -1.0, 1.0)  # CCW: left/port => inside (-)

        # magnitude: distance minus capsule radius
        sd_pair = dist_min - rad[i_v][:, None]    # [Psel,T]

        # stash for reduction over m
        mag_ntm[i_v, s_v]  = sd_pair.abs()
        sign_ntm[i_v, s_v] = sign_boundary

    # choose nearest polyline per agent/time
    idx_m = mag_ntm.argmin(dim=1)                   # [N,T] over m
    mag   = mag_ntm.gather(1, idx_m.unsqueeze(1)).squeeze(1)  # [N,T]
    sgn   = sign_ntm.gather(1, idx_m.unsqueeze(1)).squeeze(1) # [N,T]
    sd = sgn * mag

    if not has_time:
        sd = sd[:, 0]
    return sd
