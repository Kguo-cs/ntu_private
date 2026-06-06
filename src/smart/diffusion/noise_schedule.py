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
        init_gamma: Sequence[float] = (0.75, 0.5, 1.0, 4),
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
        )

        self.learn_schedule=False

        self.piecewise=False


        if self.learn_schedule:
            # self.raw_gamma = nn.Parameter(
            #     torch.logit(scaled)
            # )
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
            raw_gamma = torch.logit(scaled)
            self.register_buffer("raw_gamma", raw_gamma)




        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = eps

        group_index = torch.repeat_interleave(
            torch.arange(len(group_dims)),
            torch.tensor(group_dims),
        )

        self.register_buffer(
            "group_index",
            group_index,
        )

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

    def forward(
        self,
        base_t: Tensor,
        x_ref: Tensor,
        tokenized_agent=None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            base_t:
                [N_agent] or [N_agent, 1]

            x_ref:
                [N_agent, 8] or [N_agent, 1, 8]

        Returns:
            grouped_t:
                Broadcastable to x_ref.

            dgrouped_t_dt:
                dr(t) / dt.
        """
        if base_t.ndim == 1:
            base_t = base_t.unsqueeze(-1)

        while base_t.ndim < x_ref.ndim:
            base_t = base_t.unsqueeze(-1)

        base_t = base_t.to(
            device=x_ref.device,
            dtype=x_ref.dtype,
        )

        safe_t = torch.clamp(
            base_t,
            min=0,
            max=1.0,
        )


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
                    device=x_ref.device,
                    dtype=x_ref.dtype,
                )

                gamma = gamma.view(
                    *([1] * (x_ref.ndim - 1)),
                    -1,
                )
            grouped_t = torch.pow(
                safe_t,
                gamma,
            )

           # if self.learn_schedule:
            dgrouped_t_dt = (
                    gamma
                    * torch.pow(
                safe_t,
                gamma - 1.0,
            )
            )

        # else:
        #     dgrouped_t_dt=torch.ones_like(base_t)
        return grouped_t, dgrouped_t_dt

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
        target=x_0,
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