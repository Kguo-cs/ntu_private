"""Clean agent token encoder with explicit time alignment."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    project_to_local_frame,
    weight_init,
)


def _time_major(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Flatten [agent, time, ...] in time-major order."""
    x = x.transpose(0, 1)
    return x.reshape(-1, *x.shape[2:]) if mask is None else x[mask.T]


class AgentTokenEncoder(nn.Module):
    """Fuse trajectory-token embeddings with motion/context embeddings.

    ``pos_a`` may contain either ``T`` positions or ``T + 1`` positions.
    In the latter case, consecutive differences provide exactly ``T`` motions.
    Valid nodes are returned in time-major order.
    """

    NUM_TYPES = 3

    def __init__(
        self,
        hidden_dim: int,
        num_freq_bands: int,
        token_processor,
        discriminator: bool,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.token_processor = token_processor
        self.discriminator = bool(discriminator)

        self.use_goal = token_processor.use_goal

        self.differentiable_edge = bool(
            token_processor.use_gradient_penalty or not self.discriminator
        )
        # self.differentiable_edge=True

        # Compatibility flags used by existing configs.
        self.shape_dim = 2

        motion_dim = 2
        spatial_dim = motion_dim * (2 if self.use_goal else 1)
        self.x_a_emb = FourierEmbedding(
            input_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        self.type_a_emb = nn.Embedding(self.NUM_TYPES, hidden_dim)
        self.shape_emb = MLPLayer(
            self.shape_dim,
            hidden_dim,
            hidden_dim,
        )
        self.ego_embed = nn.Embedding(2, hidden_dim)

        input_dim_token = 8
        if not self.discriminator:
            self.token_emb_veh = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_ped = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_cyc = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )

            self.token_embedders=[self.token_emb_veh, self.token_emb_ped, self.token_emb_cyc]

            self.fusion_emb = MLPEmbedding(
                hidden_dim * 2,
                hidden_dim,
        )

        self.apply(weight_init)

    # ------------------------------------------------------------------
    # Token embeddings
    # ------------------------------------------------------------------
    def _token_table(self, type_id: int, device: torch.device) -> Tensor:
        names = (
            "trajectory_token_veh",
            "trajectory_token_ped",
            "trajectory_token_cyc",
        )
        table = getattr(self.token_processor, names[type_id]).to(device)
        return table.reshape(table.shape[0], -1)


    def _typed_token_embedding(
        self,
        token: Tensor,
        agent_type: Tensor,
        token_mask: Optional[Tensor],
    ) -> Tensor:
        n_agent, n_step = token.shape[:2]
        reference = next(self.token_embedders[0].parameters())
        output = reference.new_zeros(
            n_agent,
            n_step,
            self.hidden_dim,
            device=token.device,
        )

        for type_id, embedder in enumerate(self.token_embedders):
            selected = (agent_type[:, None] == type_id).expand(
                -1, n_step
            )
            if token_mask is not None:
                selected = selected & token_mask
            if not selected.any():
                continue

            table = self._token_table(type_id, token.device)
            table_embedding = embedder(table.to(reference.dtype))
            output[selected] = table_embedding[token[selected]]

        return output

    def get_embedding(
        self,
        agent_token_index: Tensor,
        agent_type: Tensor,
        token_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Return dense [agent,time,hidden] token embeddings."""
        if self.discriminator:
            return None

        return self._typed_token_embedding(
                agent_token_index,
                agent_type,
                token_mask,
            )

    # ------------------------------------------------------------------
    # Motion/context features
    # ------------------------------------------------------------------
    @staticmethod
    def _motion(
        pos: Tensor,
        num_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Return positions aligned to T states and T displacement vectors."""
        if pos.shape[1] == num_steps:
            #raise ValueError( f"pos shape error."   )
            motion = torch.cat(
                [
                    pos.new_zeros(pos.shape[0], 1, pos.shape[-1]),
                    pos[:, 1:] - pos[:, :-1],
                ],
                dim=1,
            )
            return pos, motion

        if pos.shape[1] == num_steps + 1:
            return pos[:, 1:], pos[:, 1:] - pos[:, :-1]

    def _goal_feature(
        self,
        pos: Tensor,
        heading_vector: Tensor,
        goal_pos: Optional[Tensor],
        goal_mask: Optional[Tensor],
    ) -> Tensor:
        n_agent, n_step, pos_dim = pos.shape
        if goal_pos is None:
            return pos.new_zeros(n_agent, n_step, pos_dim)

        if goal_pos.shape != (n_agent, pos_dim):
            raise ValueError(
                f"goal_pos must have shape [{n_agent}, {pos_dim}]."
            )

        vector = goal_pos[:, None] - pos
        feature = torch.stack(
            [
                torch.linalg.vector_norm(vector, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=heading_vector,
                    nbr_vector=vector[..., :2],
                ),
            ],
            dim=-1,
        )

        if goal_mask is None:
            return feature
        if goal_mask.ndim == 1:
            goal_mask = goal_mask[:, None].expand(-1, n_step)
        if goal_mask.shape != (n_agent, n_step):
            raise ValueError(
                f"goal_mask must have shape [{n_agent}] or "
                f"[{n_agent}, {n_step}]."
            )
        return torch.where(
            goal_mask.bool()[..., None],
            feature,
            torch.zeros_like(feature),
        )

    def _categorical_feature(
        self,
        agent_type: Tensor,
        agent_shape: Optional[Tensor],
        ego_mask: Optional[Tensor],
        num_steps: int,
    ) -> Optional[Tensor]:
        feature = self.type_a_emb(agent_type) + self.shape_emb(
            agent_shape[..., : self.shape_dim]
        )+self.ego_embed(ego_mask.long())
        return feature[:, None].expand(-1, num_steps, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        agent_token_index: Tensor,
        pos_a: Tensor,
        head_vector_a: Tensor,
        valid_mask: Optional[Tensor],
        agent_type: Tensor,
        agent_shape: Optional[Tensor],
        token_mask: Optional[Tensor] = None,
        goal_pos: Optional[Tensor] = None,
        goal_mask: Optional[Tensor] = None,
        ego_mask: Optional[Tensor] = None,
    ):
        n_agent, n_step = head_vector_a.shape[:2]

        token_embedding = self.get_embedding(
            agent_token_index,
            agent_type,
            token_mask,
        )

        aligned_pos, motion = self._motion(pos_a, n_step)
        motion_feature = project_to_local_frame(
            motion,
            head_vector_a,
            self.differentiable_edge,
        )

        # The old -10 sentinel generated arbitrary Fourier features and could
        # dominate valid context. Invalid token motion is now neutral.
        if token_mask is not None:
            motion_feature = torch.where(
                token_mask[..., None],
                motion_feature,
                torch.zeros_like(motion_feature)-10,
            )

        continuous = motion_feature
        if self.use_goal:
            continuous = torch.cat(
                [
                    continuous,
                    self._goal_feature(
                        aligned_pos,
                        head_vector_a,
                        goal_pos,
                        goal_mask,
                    ),
                ],
                dim=-1,
            )


        categorical = self._categorical_feature(
            agent_type,
            agent_shape,
            ego_mask,
            n_step,
        )

        state_embedding = self.x_a_emb(
            continuous_inputs=_time_major(continuous, valid_mask),
            categorical_embs=(
                None
                if categorical is None
                else _time_major(categorical, valid_mask)
            ),
        )

        compressed_token = None
        if token_embedding is not None:
            token_steps = token_embedding.shape[1]
            if token_steps == n_step:
                token_valid = valid_mask
            elif token_steps == n_step - 1:
                token_valid = (
                    None if valid_mask is None else valid_mask[:, 1:]
                )
            else:
                raise ValueError(
                    f"Unexpected token time dimension: {token_steps}."
                )
            compressed_token = _time_major(
                token_embedding,
                token_valid,
            )

        if not self.discriminator:
            if compressed_token is None:
                raise RuntimeError("Policy encoding requires token embeddings.")
            if len(compressed_token) != len(state_embedding):
                raise ValueError(
                    "Token/state valid-node counts differ: "
                    f"{len(compressed_token)} != {len(state_embedding)}."
                )
            state_embedding = self.fusion_emb(
                torch.cat(
                    [compressed_token, state_embedding],
                    dim=-1,
                )
            )

        return state_embedding