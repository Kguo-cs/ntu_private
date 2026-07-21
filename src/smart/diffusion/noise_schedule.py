from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from src.smart.layers import MLPLayer



from typing import Sequence

import torch
from torch import Tensor, nn
from torch_scatter import scatter_mean

class LearnableGroupedPowerSchedule(nn.Module):
    """
    State layout:
        [pos_x, pos_y,
         heading_cos, heading_sin,
         length, width,
         vel_x, vel_y]
    """

    def __init__(
        self,
        group_dims: Sequence[int] = (2, 2, 2, 2),
        init_gamma: Sequence[float] = (0.75, 0.5, 1.0, 4.0),
        gamma_min: float = 0.25,
        gamma_max: float = 5.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()

        init_gamma_tensor = torch.tensor(
            init_gamma,
            dtype=torch.float32,
        )

        scaled = (
            (init_gamma_tensor - gamma_min)
            / (gamma_max - gamma_min)
        ).clamp(min=eps, max=1.0 - eps)

        self.learn_schedule=False

        self.piecewise=False


        if self.learn_schedule:
            if self.piecewise:
                num_intervals = 16
                min_interval_mass=0.02

                self.num_groups=len(group_dims)

                self.num_intervals = num_intervals
                self.min_interval_mass = min_interval_mass
                self.interval_logits = nn.Parameter(
                    torch.zeros(len(group_dims), num_intervals)
                )
            else:

                hidden_dim=128
                output_dim=4

                self.lane_embed = MLPLayer(128 + 4, hidden_dim, hidden_dim)

                self.type_embed=nn.Embedding(3,hidden_dim)

                self.schedule_net = nn.Sequential(
                    nn.Linear(
                        hidden_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        hidden_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        hidden_dim,
                        output_dim,
                    ),
                )

        else:
            # self.raw_gamma = nn.Parameter(
            #     torch.logit(scaled)
            # )

            raw_gamma = torch.logit(scaled)
            self.register_buffer("raw_gamma", raw_gamma)

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = eps

        self.sequential=False

        group_index = torch.repeat_interleave(
            torch.arange(len(group_dims)),
            torch.tensor(group_dims),
        )

        self.register_buffer(
            "group_index",
            group_index,
        )

    def resolution_aware_gamma(
            self,
            size,
            gamma0,
            ref_size=128,
            eta=0.15,
    ):
        gamma = gamma0 * (size / ref_size) ** (-eta)
        return gamma.clamp_min( 0.3)

    def flow_time_shift(self,s, shift):
        return s / (s+(1-s)*shift) #shift * s / (1.0 + (shift - 1.0) * s)

    def scene_size_shift(
            self,
            scene_size,
            ref_size=16,
            a=0.1,
            min_shift=0.7,
            max_shift=2.0,
    ):
        shift = 1.0 + a * torch.log(
            torch.as_tensor(scene_size / ref_size)
        )
        return shift#.clamp(min_shift, max_shift) large shift means more noisy / stronger corruption

    def interval_mass(self) -> Tensor:
        learned = torch.softmax(self.interval_logits, dim=-1)
        uniform = torch.full_like(learned, 1.0 / self.num_intervals)
        return (
            (1.0 - self.min_interval_mass) * learned
            + self.min_interval_mass * uniform
        )

    def knot_values(self) -> Tensor:
        mass = self.interval_mass()
        zero = torch.zeros(
            mass.shape[0], 1, device=mass.device, dtype=mass.dtype
        )
        return torch.cat([zero, torch.cumsum(mass, dim=-1)], dim=-1)

    def sequential_smoothstep_schedule(
            self,
            base_t,
            x_ref,
            eps=1e-4,
    ):
        """
        base_t is noise coefficient b.
        b = 1: pure noise
        b = 0: clean data

        Returns:
            grouped_r: noise coefficient, broadcastable to x_ref
            dgrouped_r_dt: dr/db, broadcastable to x_ref
        """
        if base_t.ndim == 1:
            base_t = base_t.unsqueeze(-1)

        while base_t.ndim < x_ref.ndim:
            base_t = base_t.unsqueeze(-1)

        p = base_t.to(device=x_ref.device, dtype=x_ref.dtype)

        # group order: pos, heading, shape, velocity
        # state layout: [px, py, cos, sin, length, width, vx, vy]
        start = torch.tensor(
            [0.00, 0.05, 0.45, 0.35],
            device=x_ref.device,
            dtype=x_ref.dtype,
        )

        end = torch.tensor(
            [0.55, 0.60, 1.00, 0.95],
            device=x_ref.device,
            dtype=x_ref.dtype,
        )

        # expand to dimensions
        start_dim = start[self.group_index]
        end_dim = end[self.group_index]

        start_dim = start_dim.view(
            *([1] * (x_ref.ndim - 1)),
            -1,
        )
        end_dim = end_dim.view(
            *([1] * (x_ref.ndim - 1)),
            -1,
        )

        width = (end_dim - start_dim).clamp_min(eps)

        q = ((p - start_dim) / width).clamp(0.0, 1.0)

        # # q is data progress for each group.
        # q = u * u * (3.0 - 2.0 * u)
        #
        # # dq / dp
        # dq_dp = 6.0 * u * (1.0 - u) / width
        #
        # # because p = 1 - b and r = 1 - q(p):
        # # dr/db = dq/dp
        # dr_db = dq_dp

        return q, q

    def forward(
        self,
        base_t: Tensor,
        tokenized_agent=None,
    ) -> tuple[Tensor]:

       # return base_t
        if base_t.ndim == 1:
            base_t = base_t.unsqueeze(-1)

        safe_t = torch.clamp(
            base_t,
            min=0,
            max=1.0,
        )
        if self.sequential:
            grouped_t, dgrouped_t_dt = self.sequential_smoothstep_schedule(
                base_t=safe_t,
                x_ref=x_ref,
            )
            return grouped_t, dgrouped_t_dt

        if self.piecewise:
            original_shape = safe_t.shape[:-1]
            t_flat = safe_t.reshape(-1)

            scaled = t_flat * self.num_intervals
            index = torch.floor(scaled).long().clamp(
                min=0, max=self.num_intervals - 1
            )
            fraction = (scaled - index.to(scaled.dtype)).clamp(0.0, 1.0)

            values = self.knot_values()  # [G, K + 1]
            left = values[:, index].transpose(0, 1)
            right = values[:, index + 1].transpose(0, 1)

            r_flat = left + fraction[:, None] * (right - left)
            dr_flat = self.num_intervals * (right - left)

            r_group = r_flat.reshape(*original_shape, self.num_groups)
            dr_group = dr_flat.reshape(*original_shape, self.num_groups)
            grouped_t = r_group[..., self.group_index]

            dgrouped_t_dt=dr_group[..., self.group_index]

            self.gamma_groups=r_flat.mean(0).detach()

        else:

            if self.learn_schedule:
                map_feature = tokenized_agent["initial_map_feature"]
                nonego_type = tokenized_agent["nonego_type"]
                agent_batch = tokenized_agent["nonego_batch"]

                batch_pl = map_feature["batch"]
                pos_pl = map_feature["position"]
                orient_pl = map_feature["orientation"]
                feat_map = map_feature["pt_token"]

                feat_map = self.lane_embed(
                    torch.cat([feat_map, pos_pl, orient_pl.cos()[:, None], orient_pl.sin()[:, None]], dim=-1))

                map_context = scatter_mean(feat_map, batch_pl, dim=0)

                type_embed = self.type_embed(nonego_type)

                context = map_context[agent_batch] + type_embed

                raw_gamma = self.schedule_net(
                    context
                )

                gamma_groups = (
                        self.gamma_min
                        + (self.gamma_max - self.gamma_min)
                        * torch.sigmoid(raw_gamma)
                )

                gamma = gamma_groups[:, None, self.group_index]

                self.gamma_groups = gamma_groups.mean(0).detach()

            else:
                self.gamma_groups = (
                        self.gamma_min
                        + (self.gamma_max - self.gamma_min)
                        * torch.sigmoid(self.raw_gamma)
                )

                gamma_dims = self.gamma_groups[self.group_index]

                gamma = gamma_dims.to(
                    device=base_t.device,
                    dtype=base_t.dtype,
                )[None]

               # num_agents=torch.bincount(tokenized_agent["batch"])[tokenized_agent["batch"]]


                # gamma = gamma.view(
                #     *([1] * (base_t.ndim - 1)),
                #     -1,
                # )

                #gamma = self.resolution_aware_gamma(num_agents[:,None,None],gamma  )

            # shift=self.scene_size_shift(num_agents[:,None,None])
            #
            # safe_t=self.flow_time_shift(safe_t, shift)

            grouped_t = torch.pow(
                safe_t,
                gamma,
            )
            #
           # if self.learn_schedule:
           #  dgrouped_t_dt = (
           #          gamma
           #          * torch.pow(
           #      safe_t.clamp(min=0.05, max=0.95),
           #      gamma - 1.0,
           #  )
           #  )#gamma is 1 , the dt=1, if gamma>1, then d_t get very samll t close to 0

            # shift=self.scene_size_shift(num_agents[:,None,None])
            #
            # grouped_t=self.flow_time_shift(grouped_t, shift)
        # else:
        #     dgrouped_t_dt=torch.ones_like(base_t)
        return grouped_t

    def regularization(
        self,
        d_t,
        smoothness_weight: float = 1e-3,
        identity_weight: float = 1e-4,
    ) -> Tensor:
        """Weak regularization against sharp or collapsed path warping."""

        if self.piecewise:
            values = self.knot_values()
            second_difference = (
                values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
            )
            reference = torch.linspace(
                0.0,
                1.0,
                self.num_intervals + 1,
                device=values.device,
                dtype=values.dtype,
            )
            return (
                    smoothness_weight * second_difference.square().mean()
                    + identity_weight * (values - reference).square().mean()
            )

        else:

            return  ((d_t - 1.0) ** 2).mean()
@dataclass
class GroupedFlowMatchingBatch:
    x_t: Tensor
    target_velocity: Tensor
    grouped_t: Tensor
    dgrouped_t_dt: Tensor
    noise: Tensor


def make_grouped_flow_matching_batch(
    x_0: Tensor,
    base_t: Tensor,
    schedule: LearnableGroupedPowerSchedule,
    noise: Optional[Tensor] = None,
    detach_velocity_target: bool = True,
) -> GroupedFlowMatchingBatch:
    """
    Construct the flow-matching training pair.

    Path:
        x_t = (1 - r(t)) * x_0 + r(t) * noise

    Velocity:
        dx_t / dt
        = dr(t) / dt * (noise - x_0)

    Args:
        x_0:
            Clean state.
            Shape: [B, N_agent, 8] or [N_agent, 8].

        base_t:
            Global flow time.
            Shape: [B] or [N_agent].

        schedule:
            Learnable grouped schedule.

        noise:
            Optional Gaussian noise with the same shape as x_0.

        detach_velocity_target:
            Recommended True initially.
            It prevents the schedule from reducing the regression loss
            merely by manipulating the target magnitude.

    Returns:
        GroupedFlowMatchingBatch
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    if noise.shape != x_0.shape:
        raise ValueError(
            "`noise` and `x_0` must have the same shape."
        )

    grouped_t, dgrouped_t_dt = schedule(
        base_t=base_t,
        x_ref=x_0,
    )

    x_t = (
        (1.0 - grouped_t) * x_0
        + grouped_t * noise
    )

    target_velocity = (
        dgrouped_t_dt
        * (noise - x_0)
    )

    if detach_velocity_target:
        target_velocity = target_velocity.detach()

    return GroupedFlowMatchingBatch(
        x_t=x_t,
        target_velocity=target_velocity,
        grouped_t=grouped_t,
        dgrouped_t_dt=dgrouped_t_dt,
        noise=noise,
    )


def expand_base_t_by_gamma(
    base_t: torch.Tensor,
    m_delta_dim,
    gammas=(0.75, 0.5, 1.0, 4.0),
):# smaller gamma -> more dense in large t -> more sparse
    """
    Args:
        base_t: Tensor, shape [..., 1] or [...]
                e.g. [num_graphs, 1]
        gammas: (gamma_pos, gamma_head, gamma_shape, gamma_vel)

    Returns:
        base_t_dim: Tensor, shape [..., 8]
                    dims = [x, y, cos_h, sin_h, length, width, vx, vy]
    """
    if base_t.dim() == 0:
        base_t = base_t.view(1, 1)
    elif base_t.shape[-1] != 1:
        base_t = base_t.unsqueeze(-1)

    gamma_pos, gamma_head, gamma_shape, gamma_vel = gammas

    t_pos = base_t ** gamma_pos
    t_head = base_t ** gamma_head
    t_shape = base_t ** gamma_shape
    t_vel = base_t ** gamma_vel

    if m_delta_dim==8:
        base_t_dim = torch.cat(
            [
                t_pos, t_pos,
                t_head, t_head,
                t_shape, t_shape,
                t_vel, t_vel,
               # t_vel, t_vel,
            ],
            dim=-1,
        )
    else:
        base_t_dim = torch.cat(
            [
                t_pos, t_pos,
                t_head, t_head,
                t_shape, t_shape,
                t_vel, t_vel,
               t_vel, t_vel,
            ],
            dim=-1,
        )


    return base_t_dim