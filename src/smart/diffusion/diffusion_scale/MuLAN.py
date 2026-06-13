from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.smart.layers import MLPLayer


def scatter_mean(
    values: Tensor,
    index: Tensor,
    dim_size: Optional[int] = None,
) -> Tensor:
    """
    Pure-PyTorch scatter mean.

    Args:
        values:
            [N, D]

        index:
            [N], integer graph IDs.

        dim_size:
            Number of graphs.

    Returns:
        [num_graphs, D]
    """
    if values.ndim != 2:
        raise ValueError("`values` must have shape [N, D].")

    if index.ndim != 1:
        raise ValueError("`index` must have shape [N].")

    if len(values) != len(index):
        raise ValueError("`values` and `index` must share dimension 0.")

    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    output = torch.zeros(
        dim_size,
        values.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )

    count = torch.zeros(
        dim_size,
        1,
        device=values.device,
        dtype=values.dtype,
    )

    output.index_add_(0, index, values)

    count.index_add_(
        0,
        index,
        torch.ones(
            len(index),
            1,
            device=values.device,
            dtype=values.dtype,
        ),
    )

    return output / count.clamp_min(1.0)

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

    def expand_groups(
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

    def group_mse(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Args:
            pred:
                [..., state_dim]

            target:
                Same shape.

        Returns:
            [..., num_groups]
        """
        if pred.shape != target.shape:
            raise ValueError("`pred` and `target` must have the same shape.")

        squared_error = (pred - target) ** 2

        losses = []
        start = 0

        for width in self.group_dims:
            end = start + width

            losses.append(
                squared_error[..., start:end].mean(dim=-1)
            )

            start = end

        return torch.stack(
            losses,
            dim=-1,
        )

class AuxiliarySceneEncoder(nn.Module):
    """
    Encode clean agent states into a scene-level auxiliary latent.

    During training:
        z ~ q(z | x_0, map_context)

    During generation:
        z ~ N(0, I)

    map_context is optional but recommended.
    """

    def __init__(
        self,
        state_dim: int = 8,
        map_context_dim: int = 0,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.map_context_dim = map_context_dim

        self.agent_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.scene_encoder = nn.Sequential(
            nn.Linear(
                hidden_dim + map_context_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                2 * latent_dim,
            ),
        )

    def forward(
        self,
        clean_state: Tensor,
        agent_batch: Tensor,
        map_context: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            clean_state:
                [N_agent, state_dim]

            agent_batch:
                [N_agent]

            map_context:
                [num_graphs, map_context_dim]

        Returns:
            mean:
                [num_graphs, latent_dim]

            logvar:
                [num_graphs, latent_dim]
        """
        num_graphs = int(
            agent_batch.max().item()
        ) + 1

        agent_feature = self.agent_encoder(
            clean_state
        )

        scene_feature = scatter_mean(
            values=agent_feature,
            index=agent_batch,
            dim_size=num_graphs,
        )

        if self.map_context_dim > 0:
            if map_context is None:
                raise ValueError(
                    "`map_context` is required when map_context_dim > 0."
                )

            if map_context.shape != (
                num_graphs,
                self.map_context_dim,
            ):
                raise ValueError(
                    "`map_context` has an unexpected shape."
                )

            scene_feature = torch.cat(
                [
                    scene_feature,
                    map_context,
                ],
                dim=-1,
            )

        statistics = self.scene_encoder(
            scene_feature
        )

        mean, logvar = statistics.chunk(
            2,
            dim=-1,
        )

        logvar = logvar.clamp(
            min=-10.0,
            max=10.0,
        )

        return mean, logvar

    @staticmethod
    def sample(
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        noise = torch.randn_like(
            mean
        )

        return (
            mean
            + torch.exp(
                0.5 * logvar
            )
            * noise
        )

    @staticmethod
    def kl_loss(
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        KL[q(z | x) || N(0, I)]
        """
        return (
            -0.5
            * (
                1.0
                + logvar
                - mean.pow(2)
                - logvar.exp()
            )
        ).mean()

class AdaptiveGroupedPolynomialSchedule(nn.Module):
    """
    MuLAN-inspired scene-adaptive multivariate schedule.

    For each scene and group:
        p_g(t, z)
            = sum_k a_{g,k}(z) * t^k
              --------------------------------
              sum_k a_{g,k}(z)

    with:
        a_{g,k}(z) > 0

    Then blend with linear path:
        r_g(t, z)
            = (1 - eta_g(z)) * t
              + eta_g(z) * p_g(t, z)

    Properties:
        r_g(0, z) = 0
        r_g(1, z) = 1
        dr_g/dt >= 0
    """

    def __init__(
        self,
        groups= SceneStateGroups(),
        latent_dim: int = 32,
        map_context_dim: int = 0,
        hidden_dim: int = 128,
        polynomial_degree: int = 5,
        max_warp: float = 0.5,
        coefficient_floor: float = 1e-4,
    ) -> None:
        super().__init__()

        if polynomial_degree < 1:
            raise ValueError(
                "`polynomial_degree` must be positive."
            )

        if not 0.0 <= max_warp <= 1.0:
            raise ValueError(
                "`max_warp` must lie in [0, 1]."
            )

        self.groups = groups
        self.polynomial_degree = polynomial_degree
        self.max_warp = float(max_warp)
        self.coefficient_floor = float(
            coefficient_floor
        )

        input_dim = (
            latent_dim
            + map_context_dim
        )

        output_dim = (
            groups.num_groups
            * polynomial_degree
            + groups.num_groups
        )

        self.lane_embed=MLPLayer(128+4,hidden_dim,hidden_dim)

       # self.type_embed=nn.Embedding(3,hidden_dim)

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

        # Make the initial schedule close to linear.
        nn.init.zeros_(
            self.schedule_net[-1].weight
        )

        nn.init.zeros_(
            self.schedule_net[-1].bias
        )

    def _context(
        self,
        latent: Tensor,
        map_context: Optional[Tensor],
    ) -> Tensor:
        if map_context is None:
            return latent

        return torch.cat(
            [
                latent,
                map_context,
            ],
            dim=-1,
        )

    def pred_parameters(
        self,
        latent: Tensor,
        tokenized_agent
    ) -> tuple[Tensor, Tensor]:

        map_feature=tokenized_agent["initial_map_feature"]
        nonego_type=tokenized_agent["nonego_type"]
        agent_batch = tokenized_agent["nonego_batch"]

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        feat_map = self.lane_embed(torch.cat([feat_map,pos_pl,orient_pl.cos()[:,None],orient_pl.sin()[:,None]], dim=-1))

        map_context= scatter_mean(feat_map,batch_pl)

        #type_embed = self.type_embed(nonego_type)

        context=map_context[agent_batch]#+type_embed

        output = self.schedule_net(
            context
        )

        coefficient_size = (
            self.groups.num_groups
            * self.polynomial_degree
        )

        raw_coefficients = output[
            :,
            :coefficient_size,
        ].reshape(
            -1,
            self.groups.num_groups,
            self.polynomial_degree,
        )

        raw_warp_strength = output[
            :,
            coefficient_size:,
        ]

        coefficients = (
            F.softplus(
                raw_coefficients
            )
            + self.coefficient_floor
        )

        warp_strength = (
            self.max_warp
            * torch.sigmoid(
                raw_warp_strength
            )
        )

        return (
            coefficients,
            warp_strength,
        )

    def forward(
        self,
        scene_t: Tensor,
        latent: Tensor,
        map_context: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            scene_t:
                [num_graphs]
                Global noise-to-data time in [0, 1].

            latent:
                [num_graphs, latent_dim]

            map_context:
                [num_graphs, map_context_dim]

        Returns:
            progress_group:
                [num_graphs, num_groups]

            dprogress_dt_group:
                [num_graphs, num_groups]
        """
        # if scene_t.ndim != 1:
        #     raise ValueError(
        #         "`scene_t` must have shape [num_graphs]."
        #     )

        coefficients, warp_strength = self.pred_parameters(
                latent=latent,
                tokenized_agent=map_context,
            )

        t = scene_t.clamp(
            min=0.0,
            max=1.0,
        ).view(
            -1,
            1,
            1,
        )

        powers = torch.arange(
            1,
            self.polynomial_degree + 1,
            device=scene_t.device,
            dtype=scene_t.dtype,
        ).view(
            1,
            1,
            -1,
        )

        t_power = t.pow(
            powers
        )

        denominator = coefficients.sum(
            dim=-1
        ).clamp_min(
            1e-8
        )

        polynomial_progress = (
            coefficients
            * t_power
        ).sum(
            dim=-1
        ) / denominator

        derivative_terms = (
            powers
            * t.clamp_min(
                1e-6
            ).pow(
                powers - 1.0
            )
        )

        polynomial_derivative = (
            coefficients
            * derivative_terms
        ).sum(
            dim=-1
        ) / denominator

        linear_progress = scene_t.view(
            -1,
            1,
        )

        progress_group = (
            (
                1.0
                - warp_strength
            )
            * linear_progress
            + warp_strength
            * polynomial_progress
        )

        self.gamma_groups=warp_strength.mean(0).detach()

        dprogress_dt_group = (
            (
                1.0
                - warp_strength
            )
            + warp_strength
            * polynomial_derivative
        )

        progress_group = (
            self.groups.expand_groups(
                progress_group
            )
        )

        dprogress_dt_group = (
            self.groups.expand_groups(
                dprogress_dt_group
            )
        )
        while progress_group.ndim < latent.ndim:
            progress_group = progress_group.unsqueeze(1)
            dprogress_dt_group = dprogress_dt_group.unsqueeze(1)

        return (
            progress_group,
            dprogress_dt_group,
        )

    def regularization(
        self,
        scene_t: Tensor,
        latent: Tensor,
        map_context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Penalize overly stiff schedules.

        The default linear path has:
            dr/dt = 1
        """
        _, derivative = self.forward(
            scene_t=scene_t,
            latent=latent,
            map_context=map_context,
        )

        return (
            (
                derivative - 1.0
            ) ** 2
        ).mean()

@dataclass
class AdaptiveFlowBatch:
    x_t: Tensor
    clean_state: Tensor
    noise: Tensor
    scene_t: Tensor
    agent_t: Tensor
    latent_scene: Tensor
    latent_agent: Tensor
    progress_group_scene: Tensor
    progress_group_agent: Tensor
    progress_dim_agent: Tensor
    derivative_group_agent: Tensor
    derivative_dim_agent: Tensor
    target_velocity: Tensor
    latent_kl: Tensor
    schedule_regularization: Tensor