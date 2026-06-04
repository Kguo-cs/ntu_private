from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ============================================================
#  Group definition
# ============================================================

class SceneStateGroups(nn.Module):
    """
    State layout:
        [pos_x, pos_y,
         heading_cos, heading_sin,
         length, width,
         vel_x, vel_y]

    Group layout:
        position: [0, 1]
        heading:  [2, 3]
        shape:    [4, 5]
        velocity: [6, 7]
    """

    def __init__(
        self,
        group_dims: Sequence[int] = (2, 2, 2, 2),
        group_names: Sequence[str] = (
            "position",
            "heading",
            "shape",
            "velocity",
        ),
    ) -> None:
        super().__init__()

        if len(group_dims) != len(group_names):
            raise ValueError(
                "`group_dims` and `group_names` must have the same length."
            )

        self.group_dims = tuple(int(v) for v in group_dims)
        self.group_names = tuple(group_names)

        group_index = torch.repeat_interleave(
            torch.arange(
                len(group_dims),
                dtype=torch.long,
            ),
            torch.as_tensor(
                group_dims,
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "group_index",
            group_index,
            persistent=True,
        )

    @property
    def num_groups(self) -> int:
        return len(self.group_dims)

    @property
    def state_dim(self) -> int:
        return int(self.group_index.numel())

    def expand_group_values(
        self,
        group_values: Tensor,
    ) -> Tensor:
        """
        Args:
            group_values:
                [..., num_groups]

        Returns:
            [..., state_dim]
        """
        if group_values.shape[-1] != self.num_groups:
            raise ValueError(
                f"Expected last dimension {self.num_groups}, "
                f"but got {group_values.shape[-1]}."
            )

        return group_values[..., self.group_index]

    def per_item_group_mse(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Args:
            pred:
                [..., state_dim]

            target:
                Same shape as pred.

        Returns:
            [..., num_groups]
        """
        if pred.shape != target.shape:
            raise ValueError("`pred` and `target` must have the same shape.")

        if pred.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"but got {pred.shape[-1]}."
            )

        squared_error = (pred - target) ** 2

        group_errors = []
        start = 0

        for width in self.group_dims:
            end = start + width

            group_errors.append(
                squared_error[..., start:end].mean(dim=-1)
            )

            start = end

        return torch.stack(
            group_errors,
            dim=-1,
        )


# ============================================================
#  CDTD-style monotonic loss-aware warp
# ============================================================

class CDTDGroupedWarp(nn.Module):
    """
    CDTD-inspired loss-aware group-wise schedule.

    The scheduler learns a monotonic curve:

        predicted_loss_g(c)

    where:
        c = 0 means clean
        c = 1 means pure noise

    A normalized monotonic CDF is used as the curve shape.
    Its inverse maps uniform difficulty coordinates to
    corruption levels.

    Generation uses:
        progress_g(tau) = 1 - corruption_g(1 - tau)

    so:
        tau = 0 -> progress = 0 -> pure noise
        tau = 1 -> progress = 1 -> clean sample
    """

    def __init__(
        self,
        groups: SceneStateGroups,
        eps: float = 1e-4,
        min_slope: float = 0.1,
    ) -> None:
        super().__init__()

        self.groups = groups
        self.eps = float(eps)
        self.min_slope = float(min_slope)

        num_groups = groups.num_groups

        # Curve location in logit-corruption space.
        self.location = nn.Parameter(
            torch.zeros(num_groups)
        )

        # Positive curve steepness.
        self.raw_slope = nn.Parameter(
            torch.zeros(num_groups)
        )

        # Positive loss floor and scale.
        self.raw_floor = nn.Parameter(
            torch.full((num_groups,), -4.0)
        )

        self.raw_scale = nn.Parameter(
            torch.zeros(num_groups)
        )

    @property
    def slope(self) -> Tensor:
        return (
            F.softplus(self.raw_slope)
            + self.min_slope
        )

    @property
    def loss_floor(self) -> Tensor:
        return F.softplus(self.raw_floor)

    @property
    def loss_scale(self) -> Tensor:
        return F.softplus(self.raw_scale)

    def _safe_logit(
        self,
        x: Tensor,
    ) -> Tensor:
        x = x.clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        return torch.logit(x)

    def _curve_endpoints(self) -> tuple[Tensor, Tensor]:
        slope = self.slope
        location = self.location

        low_input = torch.full_like(
            location,
            self.eps,
        )

        high_input = torch.full_like(
            location,
            1.0 - self.eps,
        )

        low = torch.sigmoid(
            slope
            * (
                self._safe_logit(low_input)
                - location
            )
        )

        high = torch.sigmoid(
            slope
            * (
                self._safe_logit(high_input)
                - location
            )
        )

        return low, high

    def normalized_cdf(
        self,
        corruption: Tensor,
    ) -> Tensor:
        """
        Args:
            corruption:
                [..., num_groups]
                0 = clean
                1 = pure noise

        Returns:
            Normalized monotonic curve values in [0, 1].
        """
        if corruption.shape[-1] != self.groups.num_groups:
            raise ValueError(
                "The last dimension of `corruption` must match "
                "the number of groups."
            )

        low, high = self._curve_endpoints()

        raw_value = torch.sigmoid(
            self.slope
            * (
                self._safe_logit(corruption)
                - self.location
            )
        )

        normalized = (
            raw_value - low
        ) / (
            high - low
        ).clamp_min(1e-8)

        return normalized.clamp(
            min=0.0,
            max=1.0,
        )

    def quantile(
        self,
        uniform_level: Tensor,
    ) -> Tensor:
        """
        Inverse normalized CDF.

        Args:
            uniform_level:
                [..., num_groups] or [..., 1]
                Values in [0, 1].

        Returns:
            corruption:
                [..., num_groups]
        """
        if uniform_level.shape[-1] == 1:
            uniform_level = uniform_level.expand(
                *uniform_level.shape[:-1],
                self.groups.num_groups,
            )

        if uniform_level.shape[-1] != self.groups.num_groups:
            raise ValueError(
                "The last dimension of `uniform_level` must be "
                "one or match the number of groups."
            )

        low, high = self._curve_endpoints()

        uniform_clamped = uniform_level.clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        raw_cdf_value = (
            low
            + uniform_clamped
            * (
                high - low
            )
        )

        corruption = torch.sigmoid(
            self.location
            + self._safe_logit(raw_cdf_value)
            / self.slope
        )

        # Preserve exact endpoints for sampling.
        corruption = torch.where(
            uniform_level <= 0.0,
            torch.zeros_like(corruption),
            corruption,
        )

        corruption = torch.where(
            uniform_level >= 1.0,
            torch.ones_like(corruption),
            corruption,
        )

        return corruption

    def predicted_loss(
        self,
        corruption: Tensor,
    ) -> Tensor:
        """
        Monotonic approximation of group-wise reconstruction loss.

        Args:
            corruption:
                [..., num_groups]

        Returns:
            [..., num_groups]
        """
        return (
            self.loss_floor
            + self.loss_scale
            * self.normalized_cdf(corruption)
        )

    def corruption_from_uniform(
        self,
        uniform_level: Tensor,
    ) -> Tensor:
        """
        Uniform difficulty coordinate -> group corruption levels.
        """
        return self.quantile(uniform_level)

    def progress_from_tau(
        self,
        tau: Tensor,
    ) -> Tensor:
        """
        Noise-to-data generation schedule.

        Args:
            tau:
                [...], [..., 1], or [..., num_groups]
                Global generation time:
                    tau = 0 -> pure noise
                    tau = 1 -> clean data

        Returns:
            [..., num_groups]
                Group-specific clean-data progress.
        """
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)

        corruption = self.corruption_from_uniform(
            1.0 - tau
        )

        return 1.0 - corruption


    def regularization(self) -> Tensor:
        """
        Mild regularization against overly sharp curves.
        """
        return (
            1e-4
            * (
                self.slope ** 2
            ).mean()
        )

    @property
    def gamma_groups(self) -> Tensor:
        return self.location

    def forward(
        self,
        base_t: Tensor,
        x_ref: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            progress_group = self.progress_from_tau(
                base_t
            )

        progress_dim = self.groups.expand_group_values(
            progress_group
        )[:,None]

        return progress_dim,torch.ones_like(progress_dim)

    def loss(self,
             corruption,
             observed_group_loss,
             ):
        predicted = self.predicted_loss(
            corruption
        )

        loss=F.smooth_l1_loss(
            predicted,
            observed_group_loss.detach(),
        )+self.regularization()

        return loss


# ============================================================
#  Flow-matching batch construction
# ============================================================

@dataclass
class GroupedX0Batch:
    x_t: Tensor
    x_0: Tensor
    noise: Tensor
    progress_group: Tensor
    progress_dim: Tensor
    corruption_group: Tensor


def make_cdtd_grouped_x0_batch(
    x_0: Tensor,
    tau: Tensor,
    warp: CDTDGroupedWarp,
    noise: Optional[Tensor] = None,
    detach_schedule: bool = True,
) -> GroupedX0Batch:
    """
    Construct a grouped noise-to-data interpolation batch.

    Path:
        x_t = (1 - r(tau)) * noise + r(tau) * x_0

    Args:
        x_0:
            [..., state_dim]

        tau:
            [...], usually [num_agents] or [batch_size]

        warp:
            CDTDGroupedWarp module.

        noise:
            Optional noise tensor with same shape as x_0.

        detach_schedule:
            Recommended True for CDTD-style decoupled training.
            The main model does not directly optimize the scheduler.

    Returns:
        GroupedX0Batch
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    if noise.shape != x_0.shape:
        raise ValueError(
            "`noise` and `x_0` must have the same shape."
        )

    progress_group = warp.progress_from_tau(
        tau
    )

    if detach_schedule:
        progress_group = progress_group.detach()

    progress_dim = warp.groups.expand_group_values(
        progress_group
    )

    x_t = (
        (1.0 - progress_dim) * noise
        + progress_dim * x_0
    )

    return GroupedX0Batch(
        x_t=x_t,
        x_0=x_0,
        noise=noise,
        progress_group=progress_group,
        progress_dim=progress_dim,
        corruption_group=1.0 - progress_group,
    )

def weighted_group_x0_loss(
    pred_x0: Tensor,
    x_0: Tensor,
    groups: SceneStateGroups,
    group_weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Args:
        pred_x0:
            [..., state_dim]

        x_0:
            Same shape as pred_x0.

        groups:
            SceneStateGroups.

        group_weights:
            [num_groups]

    Returns:
        total_loss:
            Scalar.

        mean_group_loss:
            [num_groups]
    """
    per_item_group_loss = groups.per_item_group_mse(
        pred=pred_x0,
        target=x_0,
    )

    reduce_dims = tuple(
        range(
            per_item_group_loss.ndim - 1
        )
    )

    mean_group_loss = per_item_group_loss.mean(
        dim=reduce_dims
    )

    if group_weights is None:
        group_weights = torch.ones_like(
            mean_group_loss
        )

    total_loss = (
        mean_group_loss
        * group_weights
    ).sum() / group_weights.sum()

    return total_loss, mean_group_loss

def train_model_step(
    model: nn.Module,
    warp: CDTDGroupedWarp,
    optimizer: torch.optim.Optimizer,
    x_0: Tensor,
    agent_batch: Tensor,
    model_kwargs: Optional[dict] = None,
) -> dict[str, Tensor]:
    """
    Train the x0-prediction scene generator.

    Args:
        model:
            Your traffic scene generator.

        warp:
            CDTDGroupedWarp.

        optimizer:
            Model optimizer only.

        x_0:
            [N_agent, 8]
            Normalized clean states.

        agent_batch:
            [N_agent]
            Graph index of each agent.

        model_kwargs:
            Additional inputs:
                tokenized_map,
                agent_type,
                shape context,
                edge_index,
                etc.

    Returns:
        Logging dictionary.
    """
    if model_kwargs is None:
        model_kwargs = {}

    num_graphs = int(
        agent_batch.max().item()
    ) + 1

    eps = 1e-4

    # One global generation time per scene.
    scene_tau = (
        torch.rand(
            num_graphs,
            device=x_0.device,
            dtype=x_0.dtype,
        )
        * (
            1.0 - 2.0 * eps
        )
        + eps
    )

    # Map scene-level time to agents.
    agent_tau = scene_tau[
        agent_batch
    ]

    fm_batch = make_cdtd_grouped_x0_batch(
        x_0=x_0,
        tau=agent_tau,
        warp=warp,
        detach_schedule=True,
    )

    pred_x0 = model(
        x_t=fm_batch.x_t,
        base_t=agent_tau,
        grouped_t=fm_batch.progress_dim,
        batch=agent_batch,
        **model_kwargs,
    )

    group_weights = torch.tensor(
        [
            1.0,  # position
            1.0,  # heading
            1.0,  # shape
            1.0,  # velocity
        ],
        device=x_0.device,
        dtype=x_0.dtype,
    )

    loss_model, mean_group_loss = weighted_group_x0_loss(
        pred_x0=pred_x0,
        x_0=x_0,
        groups=warp.groups,
        group_weights=group_weights,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_model.backward()

    optimizer.step()

    return {
        "loss/model": loss_model.detach(),
        "loss/position": mean_group_loss[0].detach(),
        "loss/heading": mean_group_loss[1].detach(),
        "loss/shape": mean_group_loss[2].detach(),
        "loss/velocity": mean_group_loss[3].detach(),
    }

def train_schedule_step(
    model: nn.Module,
    warp: CDTDGroupedWarp,
    schedule_optimizer: torch.optim.Optimizer,
    x_0: Tensor,
    agent_batch: Tensor,
    model_kwargs: Optional[dict] = None,
) -> dict[str, Tensor]:
    """
    Fit CDTD-style group loss curves using uniformly sampled corruption.

    This update is decoupled from the main model optimizer.

    Args:
        model:
            Frozen during this step.

        warp:
            Scheduler to update.

        schedule_optimizer:
            Optimizer for warp parameters only.

        x_0:
            [N_agent, 8]

        agent_batch:
            [N_agent]

        model_kwargs:
            Additional model conditions.

    Returns:
        Logging dictionary.
    """
    if model_kwargs is None:
        model_kwargs = {}

    num_graphs = int(
        agent_batch.max().item()
    ) + 1

    num_groups = warp.groups.num_groups

    eps = 1e-4

    # Probe corruption is sampled uniformly.
    #
    # Shared corruption across groups at each scene is enough
    # to estimate four separate loss curves.
    scene_corruption_scalar = (
        torch.rand(
            num_graphs,
            1,
            device=x_0.device,
            dtype=x_0.dtype,
        )
        * (
            1.0 - 2.0 * eps
        )
        + eps
    )

    scene_corruption_group = (
        scene_corruption_scalar.expand(
            num_graphs,
            num_groups,
        )
    )

    agent_corruption_group = (
        scene_corruption_group[
            agent_batch
        ]
    )

    agent_progress_group = (
        1.0
        - agent_corruption_group
    )

    agent_progress_dim = (
        warp.groups.expand_group_values(
            agent_progress_group
        )
    )

    noise = torch.randn_like(
        x_0
    )

    x_probe = (
        (1.0 - agent_progress_dim) * noise
        + agent_progress_dim * x_0
    )

    # Global tau is used only as an additional time condition.
    # Since c = 1 - tau in this uniform probing step:
    scene_tau = (
        1.0
        - scene_corruption_scalar.squeeze(-1)
    )

    agent_tau = scene_tau[
        agent_batch
    ]

    # Freeze generator graph.
    with torch.no_grad():
        pred_x0 = model(
            x_t=x_probe,
            base_t=agent_tau,
            grouped_t=agent_progress_dim,
            batch=agent_batch,
            **model_kwargs,
        )

        observed_group_loss = (
            warp.groups.per_item_group_mse(
                pred=pred_x0,
                target=x_0,
            )
        )

    schedule_loss = (
        warp.fit_loss(
            corruption=agent_corruption_group,
            observed_group_loss=observed_group_loss,
        )
        + warp.regularization()
    )

    schedule_optimizer.zero_grad(
        set_to_none=True
    )

    schedule_loss.backward()

    schedule_optimizer.step()

    return {
        "loss/schedule": schedule_loss.detach(),
        "schedule/slope_position": warp.slope[0].detach(),
        "schedule/slope_heading": warp.slope[1].detach(),
        "schedule/slope_shape": warp.slope[2].detach(),
        "schedule/slope_velocity": warp.slope[3].detach(),
        "schedule/location_position": warp.location[0].detach(),
        "schedule/location_heading": warp.location[1].detach(),
        "schedule/location_shape": warp.location[2].detach(),
        "schedule/location_velocity": warp.location[3].detach(),
    }
