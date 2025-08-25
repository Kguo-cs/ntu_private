import torch
from typing import Tuple

def idm_planner(
    route: torch.Tensor,                 # [L,2] dense polyline in world coords
    idx: torch.Tensor,                   # scalar LongTensor of ego index
    all_pos: torch.Tensor,               # [N,2] current positions
    all_heading: torch.Tensor,           # [N]   current headings (rad) – unused here except for API parity
    all_velocity: torch.Tensor,          # [N,2] current velocities (m/s)
    all_shape: torch.Tensor,             # [N,2] (length, width) in meters
    desired_speed: float = 10.0,         # m/s
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      future_pos     : [5,2] positions at 0.1s, 0.2s, ..., 0.5s
      future_heading : [5]   headings (rad) at those times
    """
    device = route.device
    ego_idx = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)

    # ---- hyperparams as tensors (avoid float vs tensor issues) ----
    steps = 5
    dt = torch.tensor(0.1, device=device)                 # 10 Hz
    a_max = torch.tensor(3.0, device=device)              # m/s^2
    b_comf = torch.tensor(5.0, device=device)             # m/s^2
    T_headway = torch.tensor(1.0, device=device)          # s
    s0 = torch.tensor(2.0, device=device)                 # m
    delta = 4                                             # exponent (int okay)
    lane_width_default = torch.tensor(3.5, device=device) # m
    v0 = torch.tensor(float(desired_speed), device=device)

    # ---------- geometry helpers ----------
    def poly_s(route_xy: torch.Tensor):
        diffs = route_xy[1:] - route_xy[:-1]
        seglen = torch.linalg.norm(diffs, dim=-1)
        return torch.cat([torch.zeros(1, device=device), torch.cumsum(seglen, dim=0)], dim=0)

    def project_pts_to_poly(pts: torch.Tensor, poly: torch.Tensor):
        P = poly[:-1]            # [M,2]
        Q = poly[1:]             # [M,2]
        V = Q - P                # [M,2]
        VV = (V*V).sum(-1).clamp_min(1e-9)

        pts_exp = pts[:, None, :]                # [N,1,2]
        t = ((pts_exp - P) * V).sum(-1) / VV     # [N,M]
        t_clamped = t.clamp(0.0, 1.0)
        proj = P[None,:,:] + t_clamped[...,None]*V[None,:,:]  # [N,M,2]
        dists = torch.linalg.norm(pts_exp - proj, dim=-1)     # [N,M]

        seg_idx = torch.argmin(dists, dim=-1)                 # [N]
        t_star = t_clamped[torch.arange(pts.size(0), device=device), seg_idx]
        proj_star = proj[torch.arange(pts.size(0), device=device), seg_idx]

        s_poly = poly_s(poly)                                 # [M+1]
        s_at_P = s_poly[seg_idx]                              # [N]
        seg_len = torch.linalg.norm(V[seg_idx], dim=-1).clamp_min(1e-9)
        s = s_at_P + t_star * seg_len                         # [N]

        v_seg = V[seg_idx] / seg_len.unsqueeze(-1)            # [N,2] unit tangent
        rel = (pts - proj_star)
        cross_z = v_seg[:,0]*rel[:,1] - v_seg[:,1]*rel[:,0]   # signed lateral
        lat = cross_z
        return s, lat, proj_star, v_seg

    def pose_at_s(poly: torch.Tensor, s_query: torch.Tensor):
        s_poly = poly_s(poly)
        s_q = s_query.clamp(s_poly[0], s_poly[-1])
        idx_r = torch.bucketize(s_q, s_poly, right=False).clamp(1, s_poly.numel()-1)
        i0 = idx_r - 1
        i1 = idx_r
        s0v = s_poly[i0]
        s1v = s_poly[i1].clamp_min(s0v + 1e-9)
        t = ((s_q - s0v) / (s1v - s0v)).clamp(0, 1)
        P = poly[i0]
        Q = poly[i1]
        pos = P + (Q - P) * t.unsqueeze(-1)
        tan = (Q - P)
        heading = torch.atan2(tan[:,1], tan[:,0])
        return pos, heading

    # ---------- project agents & get longitudinal speeds ----------
    s_all, lat_all, _, tan_all = project_pts_to_poly(all_pos, route)  # [N], [N], [N,2], [N,2]
    v_lon_all = (all_velocity * tan_all).sum(-1)                      # [N] m/s

    s = s_all[ego_idx].clone()
    v = v_lon_all[ego_idx].clone()
    ego_len = all_shape[ego_idx, 0].clamp_min(0.0)
    ego_w = all_shape[ego_idx, 1].clamp_min(0.0)
    lat_tol = torch.maximum(1.2 * ego_w, lane_width_default * 0.5)

    # ---------- pick leader (same lane band & ahead) ----------
    N = all_pos.size(0)
    mask = torch.ones(N, dtype=torch.bool, device=device); mask[ego_idx] = False
    same_lane = lat_all.abs() <= lat_tol
    ahead = s_all > (s + 0.5)  # at least 0.5 m ahead
    cand = same_lane & ahead & mask

    has_leader = cand.any()
    if has_leader:
        L_all = all_shape[:, 0].clamp_min(0.0)
        raw_gap = s_all[cand] - s
        eff_gap = raw_gap - (0.5 * ego_len + 0.5 * L_all[cand])  # front-to-back gap
        min_gap, rel_idx = torch.min(eff_gap, dim=0)
        s_lead = s_all[cand][rel_idx].clone()
        v_lead = v_lon_all[cand][rel_idx].clone()
        L_lead = L_all[cand][rel_idx].clone()
    else:
        # no leader: emulate a distant leader moving at v0
        s_lead = s + 1e6
        v_lead = v0.clone()
        L_lead = torch.tensor(4.5, device=device)  # nominal car length

    # ---------- rollout 5 substeps ----------
    future_pos = []
    future_heading = []
    sqrt_ab = torch.sqrt(a_max * b_comf)  # tensor-safe

    for _ in range(steps):
        # spacing to leader (front-to-back)
        gap = (s_lead - s) - (0.5 * ego_len + 0.5 * L_lead)
        gap = gap.clamp_min(0.1)

        dv = v - v_lead  # >0 means closing in

        # desired spacing s*
        s_star = s0 + v.clamp_min(0.0) * T_headway + (v * dv).clamp_min(0.0) / (2.0 * sqrt_ab)

        # IDM accel
        a = a_max * (1.0 - (v / v0).clamp_min(0.0).pow(delta) - (s_star / gap).pow(2))
        a = torch.clamp(a, min=-2.0 * b_comf, max=a_max)

        # --- Euler–Cromer / semi-implicit Euler ---
        v = (v + a * dt).clamp_min(0.0)  # 1) update to new_v
        s = s + v * dt  # 2) move using new_v over the 0.1 s interval

        # advance leader at constant speed (simplest)
        s_lead = s_lead + v_lead * dt

        pos_i, hdg_i = pose_at_s(route, s.unsqueeze(0))
        future_pos.append(pos_i[0])
        future_heading.append(hdg_i[0])

    future_pos = torch.stack(future_pos, dim=0)  # [5,2]
    future_heading = torch.stack(future_heading, dim=0)  # [5]

    return future_pos, future_heading
