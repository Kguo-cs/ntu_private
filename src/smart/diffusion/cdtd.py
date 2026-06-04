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
        while base_t.ndim < x_ref.ndim:
            base_t = base_t.unsqueeze(-1)

        with torch.no_grad():
            progress_group = self.progress_from_tau(
                base_t
            )

        progress_dim = self.groups.expand_group_values(
            progress_group
        )

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