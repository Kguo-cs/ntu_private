"""
PyTorch reimplementation of Waymax waypoint-following + IDM policy.

This is a best-effort port from the original JAX/Waymax code you pasted.
It keeps the same high-level API and math, but uses PyTorch tensors and a
minimal Trajectory container so you can integrate outside of Waymax/JAX.

Notes:
- I included a simple placeholder for geometry.has_overlap (OBB overlap).
  Replace `has_overlap_torch` with your own exact test if you already have one.
- Broadcasting/shape semantics follow the comments in the original file:
  prefix dims (...), num_objects (N), timesteps (T).
- The API is self-contained; you can drop this file into a project and import
  `IDMRoutePolicy` and `WaypointFollowingPolicy`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace as dc_replace
from typing import Optional, Tuple, Callable

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- constants ------------------------------------
_DEFAULT_LEAD_DISTANCE = -1.0  # m
_DEFAULT_LEAD_VELOCITY = -1.0  # m/s
_MINIMUM_LEAD_DISTANCE = 0.1   # m
_DEFAULT_TIME_DELTA = 0.1      # s
_REACHED_END_OF_TRAJECTORY_THRESHOLD = 5e-2  # m
_DISTANCE_TO_REF_THRESHOLD = 5.0  # m
_STATIC_SPEED_THRESHOLD = 1.0  # m/s

# --------------------------- helpers --------------------------------------

def _ensure_bool(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bool:
        return x.to(torch.bool)
    return x


def _zeros_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def _ones_like(x: torch.Tensor, dtype=None) -> torch.Tensor:
    return torch.ones_like(x, dtype=dtype or x.dtype)


@dataclass
class Trajectory:
    # shapes: (..., N, T)
    x: torch.Tensor
    y: torch.Tensor
    yaw: torch.Tensor  # radians
    vel_x: torch.Tensor
    vel_y: torch.Tensor
    valid: torch.Tensor  # bool
    length: torch.Tensor  # (..., N, T) vehicle length (m)
    width: torch.Tensor   # (..., N, T) vehicle width (m)

    # --- derived convenience ---
    @property
    def shape(self):
        return self.x.shape

    @property
    def num_objects(self) -> int:
        return self.x.shape[-2]

    @property
    def xy(self) -> torch.Tensor:
        return torch.stack([self.x, self.y], dim=-1)

    @property
    def xyz(self) -> torch.Tensor:
        z = torch.zeros_like(self.x)
        return torch.stack([self.x, self.y, z], dim=-1)

    @property
    def speed(self) -> torch.Tensor:
        return torch.sqrt(self.vel_x**2 + self.vel_y**2 + 1e-12)

    def replace(self, **kwargs) -> "Trajectory":
        return dc_replace(self, **kwargs)

    def validate(self) -> None:
        # Light checks. You can expand if you want strict validation.
        assert self.x.shape == self.y.shape == self.yaw.shape
        assert self.vel_x.shape == self.x.shape
        assert self.vel_y.shape == self.x.shape
        assert self.valid.shape == self.x.shape

    def stack_fields(self, names: list[str]) -> torch.Tensor:
        mapping = {
            'x': self.x,
            'y': self.y,
            'length': self.length,
            'width': self.width,
            'yaw': self.yaw,
        }
        parts = [mapping[n] for n in names]
        return torch.stack(parts, dim=-1)


@dataclass
class SimulatorState:
    log_trajectory: Trajectory           # (..., N, T)
    current_sim_trajectory: Trajectory   # (..., N, 1)


# ---------------------- geometry: OBB overlap (placeholder) ----------------

def _rot2d(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)  # (..., 2, 2)
    return R


def corners_from_obb(xy: torch.Tensor, length: torch.Tensor, width: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Return 4 corners for oriented rectangles.
    xy: (..., 2), length/width/yaw: (...,)
    -> (..., 4, 2)
    """
    # local corners in vehicle frame (front is +x)
    dx = length / 2.0
    dy = width / 2.0
    local = torch.stack([
        torch.stack([ dx,  dy], dim=-1),
        torch.stack([ dx, -dy], dim=-1),
        torch.stack([-dx, -dy], dim=-1),
        torch.stack([-dx,  dy], dim=-1),
    ], dim=-2)  # (..., 4, 2)
    R = _rot2d(yaw)  # (..., 2, 2)
    world = torch.matmul(local, R.transpose(-1, -2)) + xy.unsqueeze(-2)
    return world


def sat_overlap_obb(a_xy, a_l, a_w, a_yaw, b_xy, b_l, b_w, b_yaw) -> torch.Tensor:
    """Separating Axis Theorem for OBBs. Returns boolean overlap with broadcast.
    All inputs broadcast to common shape (...).
    """
    # Corners
    Ac = corners_from_obb(a_xy, a_l, a_w, a_yaw)  # (..., 4, 2)
    Bc = corners_from_obb(b_xy, b_l, b_w, b_yaw)  # (..., 4, 2)

    def _axes(corners: torch.Tensor) -> torch.Tensor:
        # edges: v[i] -> v[i+1]
        e = torch.roll(corners, shifts=-1, dims=-2) - corners  # (..., 4, 2)
        # normals (perp axes)
        axes = torch.stack([ e[..., 1], -e[..., 0] ], dim=-1)  # (..., 4, 2)
        # normalize, guard zero
        n = torch.linalg.norm(axes, dim=-1, keepdim=True).clamp(min=1e-9)
        return axes / n

    axes = torch.cat([_axes(Ac), _axes(Bc)], dim=-2)  # (..., 8, 2)

    def _proj(corners: torch.Tensor, axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # corners (..., 4, 2), axis (..., 8, 2) -> (..., 8, 4)
        val = (corners.unsqueeze(-3) * axis.unsqueeze(-2)).sum(dim=-1)
        return val.min(dim=-1).values, val.max(dim=-1).values  # (..., 8)

    a_min, a_max = _proj(Ac, axes)
    b_min, b_max = _proj(Bc, axes)
    overlap = (a_min <= b_max) & (b_min <= a_max)  # (..., 8)
    return overlap.all(dim=-1)


def has_overlap_torch(traj_bbox: torch.Tensor, obj_bbox: torch.Tensor) -> torch.Tensor:
    """Vectorized OBB overlap.
    traj_bbox, obj_bbox: (..., 5) fields [x, y, length, width, yaw]
    Returns: boolean tensor (...,) whether they overlap.
    """
    x1, y1, l1, w1, yaw1 = [traj_bbox[..., i] for i in range(5)]
    x2, y2, l2, w2, yaw2 = [obj_bbox[..., i] for i in range(5)]
    a_xy = torch.stack([x1, y1], dim=-1)
    b_xy = torch.stack([x2, y2], dim=-1)
    return sat_overlap_obb(a_xy, l1, w1, yaw1, b_xy, l2, w2, yaw2)


# ---------------------- core algorithms (PyTorch) -------------------------

class WaypointFollowingPolicy:
    def __init__(self, is_controlled_func: Optional[Callable[[SimulatorState], torch.Tensor]] = None,
                 invalidate_on_end: bool = False):
        self.is_controlled_func = is_controlled_func
        self.invalidate_on_end = invalidate_on_end

    def update_trajectory(self, state: SimulatorState) -> Trajectory:
        new_speed, new_valid = self.update_speed(state)
        next_traj = self._get_next_trajectory_by_projection(
            state.log_trajectory,
            state.current_sim_trajectory,
            new_speed,
            new_valid,
            dt=_DEFAULT_TIME_DELTA,
        )
        return Trajectory(
            x=next_traj.x, y=next_traj.y, yaw=next_traj.yaw,
            vel_x=next_traj.vel_x, vel_y=next_traj.vel_y,
            valid=next_traj.valid,
            length=next_traj.length, width=next_traj.width,
        )

    def _get_next_trajectory_by_projection(
        self,
        log_traj: Trajectory,
        cur_sim_traj: Trajectory,
        new_speed: torch.Tensor,          # (..., N)
        new_speed_valid: torch.Tensor,    # (..., N) bool
        dt: float = _DEFAULT_TIME_DELTA,
    ) -> Trajectory:
        cur_speed = cur_sim_traj.speed  # (..., N, 1) -> via property returns (..., N, 1)? ours returns (..., N, 1) if input has T=1
        if cur_speed.shape[-1] != 1:
            # ensure last dim timesteps=1 when reading speed
            pass
        valid = _ensure_bool(new_speed_valid).unsqueeze(-1)  # (..., N, 1)
        # choose speed
        new_speed_ = torch.where(valid, new_speed.unsqueeze(-1), cur_speed)

        dist_travel = (new_speed_ + cur_speed) * 0.5 * dt  # (..., N, 1)

        next_x = cur_sim_traj.x + dist_travel * torch.cos(cur_sim_traj.yaw)
        next_y = cur_sim_traj.y + dist_travel * torch.sin(cur_sim_traj.yaw)

        next_xy, next_yaw, reached_last = _project_to_a_trajectory(
            torch.stack([next_x, next_y], dim=-1),
            log_traj,
            extrapolate_traj=(not self.invalidate_on_end),
        )

        if self.invalidate_on_end:
            default_x_vel = torch.zeros_like(cur_sim_traj.vel_x)
            default_y_vel = torch.zeros_like(cur_sim_traj.vel_y)
        else:
            default_x_vel = cur_sim_traj.vel_x
            default_y_vel = cur_sim_traj.vel_y

        new_vel_x = torch.where(reached_last, default_x_vel, new_speed_ * torch.cos(cur_sim_traj.yaw))
        new_vel_y = torch.where(reached_last, default_y_vel, new_speed_ * torch.sin(cur_sim_traj.yaw))

        if self.invalidate_on_end:
            moving_after_last = reached_last & (new_speed_ > _STATIC_SPEED_THRESHOLD)
            valid = valid & (~moving_after_last)

        next_traj = cur_sim_traj.replace(
            x=next_xy[..., 0],
            y=next_xy[..., 1],
            yaw=next_yaw,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            valid=valid & cur_sim_traj.valid,
        )
        next_traj.validate()

        # Fill invalid with sentinel
        def _make_invalid_data(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return torch.zeros_like(t, dtype=torch.bool)
            return -torch.ones_like(t)

        def _where_valid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.where(next_traj.valid, x, y)

        invalid_traj = Trajectory(
            x=_make_invalid_data(next_traj.x),
            y=_make_invalid_data(next_traj.y),
            yaw=_make_invalid_data(next_traj.yaw),
            vel_x=_make_invalid_data(next_traj.vel_x),
            vel_y=_make_invalid_data(next_traj.vel_y),
            valid=_make_invalid_data(next_traj.valid),
            length=_make_invalid_data(next_traj.length),
            width=_make_invalid_data(next_traj.width),
        )
        next_traj = next_traj.replace(
            x=_where_valid(next_traj.x, invalid_traj.x),
            y=_where_valid(next_traj.y, invalid_traj.y),
            yaw=_where_valid(next_traj.yaw, invalid_traj.yaw),
            vel_x=_where_valid(next_traj.vel_x, invalid_traj.vel_x),
            vel_y=_where_valid(next_traj.vel_y, invalid_traj.vel_y),
            valid=_where_valid(next_traj.valid, invalid_traj.valid),
            length=_where_valid(next_traj.length, invalid_traj.length),
            width=_where_valid(next_traj.width, invalid_traj.width),
        )
        return next_traj

    def update_speed(self, state: SimulatorState, dt: float = _DEFAULT_TIME_DELTA) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class IDMRoutePolicy(WaypointFollowingPolicy):
    def __init__(self,
                 is_controlled_func: Optional[Callable[[SimulatorState], torch.Tensor]] = None,
                 desired_vel: float = 30.0,
                 min_spacing: float = 2.0,
                 safe_time_headway: float = 2.0,
                 max_accel: float = 2.0,
                 max_decel: float = 4.0,
                 delta: float = 4.0,
                 max_lookahead: int = 10,
                 lookahead_from_current_position: bool = True,
                 additional_lookahead_points: int = 10,
                 additional_lookahead_distance: float = 10.0,
                 invalidate_on_end: bool = False):
        super().__init__(is_controlled_func=is_controlled_func,
                         invalidate_on_end=invalidate_on_end)
        self.desired_vel = desired_vel
        self.min_spacing_s0 = min_spacing
        self.safe_time_headway = safe_time_headway
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.delta = delta
        self.max_lookahead = max_lookahead
        self.lookahead_from_current_position = lookahead_from_current_position
        self.additional_lookahead_distance = additional_lookahead_distance
        self.additional_headway_points = additional_lookahead_points

    def update_speed(self, state: SimulatorState, dt: float = _DEFAULT_TIME_DELTA) -> Tuple[torch.Tensor, torch.Tensor]:
        log_waypoints = state.log_trajectory
        cur_position = state.current_sim_trajectory.xyz[..., 0, :]  # (..., N, 3)
        cur_speed = state.current_sim_trajectory.speed[..., 0]      # (..., N)

        accel = self._get_accel(log_waypoints, cur_position, cur_speed, state.current_sim_trajectory)
        valid = torch.ones_like(cur_speed, dtype=torch.bool)
        speed = torch.clamp(cur_speed + dt * accel, min=0.0)
        return speed, valid

    def _get_accel(self,
                   log_waypoints: Trajectory,
                   cur_position: torch.Tensor,
                   cur_speed: torch.Tensor,
                   obj_curr_traj: Trajectory) -> torch.Tensor:
        N = obj_curr_traj.num_objects
        # 1) reference traj
        if self.lookahead_from_current_position:
            traj = _find_reference_traj_from_log_traj(cur_position, obj_curr_traj, 1)
            total_lookahead = 1 + self.additional_headway_points
        else:
            traj = _find_reference_traj_from_log_traj(cur_position, log_waypoints, self.max_lookahead)
            total_lookahead = self.max_lookahead + self.additional_headway_points

        if self.additional_headway_points > 0:
            traj = _add_headway_waypoints(traj,
                                          distance=self.additional_lookahead_distance,
                                          num_points=self.additional_headway_points)

        # 2) pairwise collision indicators between traj (..., N, L) and obj_curr_traj (..., N, 1)
        L = traj.x.shape[-1]
        # build broadcast shape (..., N, N, L, 5)
        traj_5 = traj.stack_fields(['x','y','length','width','yaw']).unsqueeze(-3).expand(*traj.x.shape[:-2], N, N, L, 5)
        obj_5  = obj_curr_traj.stack_fields(['x','y','length','width','yaw']).unsqueeze(-4).expand(*traj.x.shape[:-2], N, N, L, 5)

        collision_pairwise = has_overlap_torch(traj_5, obj_5)  # (..., N, N, L)

        eye = torch.eye(N, dtype=torch.bool, device=traj.x.device)
        # expand eye to prefix dims and time L
        self_mask = eye.view(*([1]* (collision_pairwise.ndim-3)), N, N).unsqueeze(-1)

        collision_valid = traj.valid.unsqueeze(-3) & obj_curr_traj.valid.unsqueeze(-4) & (~self_mask)
        collision_pairwise = torch.where(collision_valid, collision_pairwise, torch.zeros_like(collision_pairwise, dtype=torch.bool))

        # 3) lead velocity & distance
        obj_speed_tiled = obj_curr_traj.speed.unsqueeze(-3).expand_as(collision_pairwise)
        lead_vel = self._compute_lead_velocity(obj_speed_tiled, collision_pairwise, obj_curr_traj.valid.unsqueeze(-4))

        lead_dist = self._compute_lead_distance(
            agent_future=traj.xyz,
            collision_indicator=collision_pairwise.any(dim=-2),
            current_position=obj_curr_traj.xyz,
            agent_future_valid=traj.valid,
        )

        # 4) IDM
        s_star = self.min_spacing_s0 + torch.clamp_min(
            0.0,
            cur_speed * self.safe_time_headway +
            cur_speed * (cur_speed - lead_vel) / (2.0 * math.sqrt(self.max_accel * self.max_decel))
        )

        mask_free = (lead_dist == _DEFAULT_LEAD_DISTANCE) | (lead_vel == _DEFAULT_LEAD_VELOCITY)
        s_star = torch.where(mask_free, torch.zeros_like(s_star), s_star)

        lead_dist = torch.where(lead_dist == 0.0, torch.full_like(lead_dist, _MINIMUM_LEAD_DISTANCE), lead_dist)

        accel = self.max_accel * (1.0 - (cur_speed / self.desired_vel).pow(self.delta) - (s_star / lead_dist).pow(2))
        return accel

    def _compute_lead_velocity(self,
                               future_speeds: torch.Tensor,           # (..., N, N, L)
                               collisions_per_agent: torch.Tensor,     # (..., N, N, L) bool
                               future_speeds_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        collision_vels_at = torch.where(collisions_per_agent, future_speeds, torch.full_like(future_speeds, float('inf')))
        if future_speeds_valid is not None:
            collision_vels_at = torch.where(future_speeds_valid, collision_vels_at, torch.full_like(collision_vels_at, float('inf')))
        # min over colliding objects -> (..., N, L)
        collision_vels_t = collision_vels_at.min(dim=-2).values
        mask_t = torch.isfinite(collision_vels_t)
        cumsum_t = torch.cumsum(torch.where(mask_t, collision_vels_t, torch.zeros_like(collision_vels_t)), dim=-1)
        collision_vels_cum = torch.where(mask_t, cumsum_t, torch.full_like(cumsum_t, float('inf')))
        lead_vel = collision_vels_cum.min(dim=-1).values  # (..., N)
        lead_vel = torch.where(torch.isfinite(lead_vel), lead_vel, torch.full_like(lead_vel, _DEFAULT_LEAD_VELOCITY))
        return lead_vel

    def _compute_lead_distance(self,
                               agent_future: torch.Tensor,            # (..., N, L, 3)
                               collision_indicator: torch.Tensor,     # (..., N, L) bool
                               agent_future_valid: Optional[torch.Tensor] = None,
                               current_position: Optional[torch.Tensor] = None,
                               use_arclength: bool = False) -> torch.Tensor:
        if use_arclength:
            arc_lengths = _compute_arclengths(agent_future, agent_future_valid)
            cum = torch.cumsum(arc_lengths, dim=-1)
        else:
            if current_position is None:
                current_position = agent_future[..., 0:1, :]
            dists = torch.linalg.norm(current_position - agent_future, dim=-1)
            cum = dists
        dists_to_collision = torch.where(collision_indicator, cum, torch.full_like(cum, float('inf')))
        lead_dist = dists_to_collision.min(dim=-1).values  # (..., N)
        lead_dist = torch.where(torch.isfinite(lead_dist), lead_dist, torch.full_like(lead_dist, _DEFAULT_LEAD_DISTANCE))
        return lead_dist


# ----------------------------- standalone fns -----------------------------

def dynamic_slice(traj: Trajectory, start_idx: torch.Tensor, count: int) -> Trajectory:
    # start_idx: (...,) int64
    T = traj.x.shape[-1]
    idx = torch.arange(count, device=traj.x.device).view(*([1]*start_idx.ndim), count)
    idx = (start_idx.unsqueeze(-1) + idx).clamp(max=T-1)
    sl = (..., slice(None), idx)
    # we cannot slice tensors with broadcasted index directly in dataclass; use gather
    def g(t: torch.Tensor) -> torch.Tensor:
        # t: (..., N, T)
        gather_dim = t.dim() - 1
        exp_idx = idx.unsqueeze(-2).expand(*t.shape[:-1], count)
        return torch.gather(t, gather_dim, exp_idx)
    return traj.replace(x=g(traj.x), y=g(traj.y), yaw=g(traj.yaw), vel_x=g(traj.vel_x), vel_y=g(traj.vel_y),
                        valid=g(traj.valid), length=g(traj.length), width=g(traj.width))


def _find_reference_traj_from_log_traj(xyz: torch.Tensor, traj: Trajectory, num_pts: int) -> Trajectory:
    # xyz: (..., N, 3) current position; traj: (..., N, T)
    # compute nearest waypoint index per object
    diffs = xyz.unsqueeze(-2) - traj.xyz  # (..., N, T, 3)
    dists = torch.linalg.norm(diffs, dim=-1)  # (..., N, T)
    dists = torch.where(traj.valid, dists, torch.full_like(dists, float('inf')))
    top_idx = dists.argmin(dim=-1)  # (..., N)
    return dynamic_slice(traj, top_idx, num_pts)


def _project_to_a_trajectory(xy: torch.Tensor, traj: Trajectory, extrapolate_traj: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # xy: (..., N, 1, 2), traj: (..., N, T)
    # find closest traj point
    diff = traj.xy - xy  # (..., N, T, 2)
    dist = torch.where(traj.valid, torch.linalg.norm(diff, dim=-1), torch.full_like(traj.x, float('inf')))
    idx = dist.argmin(dim=-1)  # (..., N)

    # gather closest xy, yaw
    def gather_lastdim(t: torch.Tensor) -> torch.Tensor:
        # t: (..., N, T, C?)
        gather_dim = t.dim() - 2
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*t.shape[:-2], t.shape[-2], 1)
        out = torch.gather(t, gather_dim, idx_exp).squeeze(-2)  # (..., N, C?)
        return out

    src_xy = gather_lastdim(traj.xy)  # (..., N, 2)
    src_yaw = gather_lastdim(traj.yaw.unsqueeze(-1)).squeeze(-1)  # (..., N)
    src_dir = torch.stack([torch.cos(src_yaw), torch.sin(src_yaw)], dim=-1)  # (..., N, 2)

    # last valid index per track
    indices = torch.arange(traj.x.shape[-1], device=traj.x.device)
    valid_idx = torch.where(traj.valid, indices.view(*([1]* (traj.valid.dim()-1)), -1), torch.zeros_like(traj.x))
    last_valid_idx = valid_idx.max(dim=-1).values  # (..., N)

    def gather_t(t: torch.Tensor) -> torch.Tensor:
        # t: (..., N, T, C?) -> gather at last_valid_idx
        gather_dim = t.dim() - 2
        idx_exp = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(*t.shape[:-2], t.shape[-2], 1)
        return torch.gather(t, gather_dim, idx_exp).squeeze(-2)

    last_point = gather_t(traj.xy)  # (..., N, 2)
    reached_last = (torch.linalg.norm(last_point - src_xy, dim=-1) < _REACHED_END_OF_TRAJECTORY_THRESHOLD)
    reached_last = reached_last | (dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1) > _DISTANCE_TO_REF_THRESHOLD)

    if not extrapolate_traj:
        src_dir = torch.where(reached_last.unsqueeze(-1), torch.zeros_like(src_dir), src_dir)

    rel = xy.squeeze(-2) - src_xy.unsqueeze(-2)  # (..., N, 1, 2)
    proj_len = (rel * src_dir.unsqueeze(-2)).sum(dim=-1)  # (..., N, 1)
    proj_xy = src_dir.unsqueeze(-2) * proj_len.unsqueeze(-1) + src_xy.unsqueeze(-2)  # (..., N, 1, 2)

    return proj_xy, src_yaw.unsqueeze(-1), reached_last.unsqueeze(-1)


def _compute_arclengths(waypoints: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    # waypoints: (..., N, T, 3)
    seg = torch.linalg.norm(waypoints[..., :-1, :] - waypoints[..., 1:, :], dim=-1)  # (..., N, T-1)
    first = torch.zeros_like(waypoints[..., :1, 0])  # (..., N, 1)
    if valid is not None:
        arc_valid = valid[..., :-1] & valid[..., 1:]
        seg = torch.where(arc_valid, seg, torch.full_like(seg, float('inf')))
        first = torch.where(valid[..., :1], first, torch.full_like(first, float('inf')))
    return torch.cat([first, seg], dim=-1)  # (..., N, T)


def _add_headway_waypoints(traj: Trajectory, distance: float = 2.0, num_points: int = 10) -> Trajectory:
    final_xy = traj.xy[..., -1:, :]                 # (..., N, 1, 2)
    final_yaw = traj.yaw[..., -1:]                  # (..., N, 1)
    final_dir = torch.stack([torch.cos(final_yaw), torch.sin(final_yaw)], dim=-1)  # (..., N, 1, 2)

    spacings = torch.linspace(0.0, distance, steps=num_points+1, device=traj.x.device).view(*([1]* (final_dir.dim()-2)), num_points+1, 1)
    new_pts = (final_dir * spacings + final_xy)[..., 1:, :]  # (..., N, K, 2)

    new_xy = torch.cat([traj.xy, new_pts], dim=-2)

    def rep_last(t: torch.Tensor) -> torch.Tensor:
        last = t[..., -1:]
        tile_shape = [1]* (t.dim()-1) + [num_points]
        ext = last.expand(*t.shape[:-1], num_points)
        return torch.cat([t, ext], dim=-1)

    new_traj = traj.replace(
        x=rep_last(traj.x), y=rep_last(traj.y), yaw=rep_last(traj.yaw),
        vel_x=rep_last(traj.vel_x), vel_y=rep_last(traj.vel_y),
        valid=rep_last(traj.valid), length=rep_last(traj.length), width=rep_last(traj.width),
    )
    new_traj = new_traj.replace(x=new_xy[..., 0], y=new_xy[..., 1])
    new_traj.validate()
    return new_traj
