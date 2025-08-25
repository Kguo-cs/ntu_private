import torch
from typing import Tuple

def idm_planner(
    route: torch.Tensor,                 # [L,2] dense polyline (world coords)
    idx: torch.Tensor,                   # scalar LongTensor index for ego
    all_pos: torch.Tensor,               # [N,2] current positions
    all_heading: torch.Tensor,           # [N]   current headings (rad) — for output only
    all_velocity: torch.Tensor,          # [N,2] current velocities (m/s) in world frame
    all_shape: torch.Tensor,             # [N,2] (length, width) in meters
    desired_speed: float = 10.0,         # m/s
    *,
    tau: float = 0.5,                    # horizon (s)
    lane_width_default: float = 3.5,     # m, for lateral gating
    a_max: float = 2.0,                  # IDM accel (m/s^2)
    b_comf: float = 3.0,                 # IDM comfortable decel (m/s^2)
    T_headway: float = 1.0,              # s
    s0: float = 2.0,                     # m
    delta: int = 4,                      # IDM exponent
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-shot 0.5s IDM step along `route`. Uses per-agent (length,width) for gap,
    and longitudinal speeds from projected velocities for dv and spacing dynamics.
    """
    device = route.device
    ego_idx = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)

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

        v_seg = V[seg_idx] / seg_len.unsqueeze(-1)            # [N,2] unit tangent along route at proj
        rel = (pts - proj_star)
        cross_z = v_seg[:,0]*rel[:,1] - v_seg[:,1]*rel[:,0]   # signed lateral (left positive)
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

    # ---------- project to route & get longitudinal speeds ----------
    s_all, lat_all, proj_all, tan_all = project_pts_to_poly(all_pos, route)   # [N], [N], [N,2], [N,2]
    # project agent velocities onto the tangent to get longitudinal speeds (signed)
    v_lon_all = (all_velocity * tan_all).sum(-1)                               # [N] m/s

    s_ego = s_all[ego_idx]
    v_ego = v_lon_all[ego_idx]
    ego_len = all_shape[ego_idx, 0].clamp_min(0.0)
    ego_w = all_shape[ego_idx, 1].clamp_min(0.0)

    # Lateral gating band width set by ego width (or half-lane fallback)
    lat_tol = torch.maximum(1.2 * ego_w, torch.tensor(lane_width_default*0.5, device=device))

    # ---------- leader selection (same lane band, ahead) ----------
    N = all_pos.size(0)
    not_ego = torch.ones(N, dtype=torch.bool, device=device)
    not_ego[ego_idx] = False

    same_lane = lat_all.abs() <= lat_tol
    ahead = s_all > (s_ego + 0.5)  # 0.5 m ahead
    cand = same_lane & ahead & not_ego

    has_leader = cand.any()
    if has_leader:
        L_lead = all_shape[:, 0].clamp_min(0.0)
        raw_gap = s_all[cand] - s_ego
        eff_gap = raw_gap - (0.5 * ego_len + 0.5 * L_lead[cand])     # front-to-back
        min_gap, rel_idx = torch.min(eff_gap, dim=0)
        s_lead = s_all[cand][rel_idx]
        v_lead = v_lon_all[cand][rel_idx]
        s_gap = torch.maximum(min_gap, torch.tensor(0.0, device=device))
    else:
        v_lead = torch.tensor(float(desired_speed), device=device)    # fallback
        s_gap = None

    # ---------- IDM longitudinal update over tau ----------
    v0 = torch.as_tensor(float(desired_speed), device=device)
    v = v_ego.clone() if isinstance(v_ego, torch.Tensor) else torch.as_tensor(v_ego, device=device)

    if has_leader and s_gap is not None:
        dv = v - v_lead                 # approaching rate (>0 means closing in)
        # desired dynamical spacing
        s_star = s0 + v.clamp_min(0) * T_headway + (v * dv).clamp_min(0) / (2.0 * torch.sqrt(a_max * b_comf))
        # guard rails
        s_gap_safe = torch.clamp(s_gap, min=0.1)
        a = a_max * (1.0 - (v / v0).pow(delta) - (s_star / s_gap_safe).pow(2))
        a = torch.clamp(a, min=-2.0 * b_comf, max=a_max)
    else:
        # free road
        a = a_max * (1.0 - (v / v0).pow(delta))

    # advance arclength using 1D kinematics (along centerline)
    v_new = v + a * tau
    v_new = torch.clamp(v_new, min=0.0)                        # no reversing
    ds = torch.clamp(v * tau + 0.5 * a * (tau ** 2), min=0.0)  # forward-only
    s_new = s_ego + ds

    # ---------- pose on route ----------
    new_pos, new_heading = pose_at_s(route, s_new.unsqueeze(0))
    return new_pos[0], new_heading[0]
