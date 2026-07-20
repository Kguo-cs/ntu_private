# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""SMART map-token encoder with explicit scene and index handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import MLPEmbedding
from src.smart.utils import transform_to_local, weight_init

from .edge_encoder import EdgeEncoder


class SMARTMapDecoder(nn.Module):
    """Encode map points and return road-edge tokens in time-independent form."""

    NUM_MAP_TYPES = 10
    NUM_LIGHT_TYPES = 5
    ROAD_EDGE_TYPES = (4, 5)

    def __init__(
        self,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        pt2pt_neighbor: int,
        token_processor,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.pl2pl_radius = float(pl2pl_radius)
        self.num_layers = int(num_layers)
        self.pt2pt_neighbor = int(pt2pt_neighbor)
        self.token_processor = token_processor

        # Names are preserved for checkpoint compatibility.
        self.type_pt_emb = nn.Embedding(10, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(5, hidden_dim)
        self.token_emb = MLPEmbedding(22, hidden_dim)

        self.register_buffer(
            "_polygon_type",
            torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 3]),
            persistent=False,
        )

        self.edge_encoder = (
            EdgeEncoder(
                hidden_dim,
                num_freq_bands,
                use_pl2a=True,
            )
            if self.num_layers > 0
            else None
        )
        self.pt2pt_layers = nn.ModuleList(
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                bipartite=False,
                has_pos_emb=True,
            )
            for _ in range(self.num_layers)
        )

        # The old code skipped initialization when num_layers == 0.
        self.apply(weight_init)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _required(mapping: Mapping[str, Any], key: str):
        if key not in mapping:
            raise KeyError(f"Missing required key {key!r}.")
        return mapping[key]

    @staticmethod
    def _vector(
        value: Tensor,
        name: str,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() != length:
            raise ValueError(
                f"{name} must contain {length} values, got {value.numel()}."
            )
        return value

    def _read_map(self, data: Mapping[str, Tensor]):
        position = self._required(data, "position")
        if position.ndim != 2 or position.shape[-1] < 2:
            raise ValueError("position must have shape [M,D] with D >= 2.")

        size = len(position)
        device = position.device
        map_type = self._vector(
            self._required(data, "type"),
            "type",
            size,
            device=device,
            dtype=torch.long,
        )
        batch = self._vector(
            self._required(data, "batch"),
            "batch",
            size,
            device=device,
            dtype=torch.long,
        )
        orientation = self._vector(
            self._required(data, "orientation"),
            "orientation",
            size,
            device=device,
            dtype=position.dtype,
        )
        token_index = self._vector(
            self._required(data, "token_idx"),
            "token_idx",
            size,
            device=device,
            dtype=torch.long,
        )
        light_type = self._vector(
            self._required(data, "light_type"),
            "light_type",
            size,
            device=device,
            dtype=torch.long,
        )
        return map_type, batch, position, orientation, token_index, light_type

    @classmethod
    def _road_edge_mask(cls, map_type: Tensor) -> Tensor:
        return (map_type == cls.ROAD_EDGE_TYPES[0]) | (
            map_type == cls.ROAD_EDGE_TYPES[1]
        )

    # ------------------------------------------------------------------
    # Scene ego pose and map cropping
    # ------------------------------------------------------------------
    @staticmethod
    def _scene_ego_pose(
        agent: Mapping[str, Any],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        for key in ("batch", "ego_mask", "initial_pos", "initial_heading"):
            if key not in agent:
                raise KeyError(f"tokenized_agent is missing {key!r}.")

        batch = agent["batch"].to(device=device, dtype=torch.long).reshape(-1)
        ego_mask = agent["ego_mask"].to(
            device=device,
            dtype=torch.bool,
        ).reshape(-1)
        position = agent["initial_pos"].to(device=device)
        heading = agent["initial_heading"].to(device=device).reshape(-1)

        num_agents = len(batch)
        if len(ego_mask) != num_agents:
            raise ValueError("ego_mask and batch have different lengths.")
        if position.ndim != 2 or position.shape[0] != num_agents:
            raise ValueError("initial_pos must have shape [N,D].")
        if position.shape[-1] < 2 or len(heading) != num_agents:
            raise ValueError("Invalid initial ego pose tensors.")

        num_graphs = (
            int(agent["num_graphs"])
            if "num_graphs" in agent
            else int(batch.max()) + 1
        )
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")
        if batch.numel() and (batch.min() < 0 or batch.max() >= num_graphs):
            raise ValueError("Agent batch contains an invalid scene index.")

        ego_batch = batch[ego_mask]
        counts = torch.bincount(ego_batch, minlength=num_graphs)
        if not torch.all(counts == 1):
            raise ValueError(
                "Exactly one ego is required per scene; "
                f"counts={counts.tolist()}."
            )

        scene_position = position.new_empty(num_graphs, 2)
        scene_heading = heading.new_empty(num_graphs)
        scene_position[ego_batch] = position[ego_mask, :2]
        scene_heading[ego_batch] = heading[ego_mask]
        return scene_position, scene_heading

    def _crop(
        self,
        map_type: Tensor,
        batch: Tensor,
        position: Tensor,
        orientation: Tensor,
        token_index: Tensor,
        light_type: Tensor,
        agent: Optional[Mapping[str, Any]],
    ):
        if agent is None:
            return (
                map_type,
                batch,
                position,
                orientation,
                token_index,
                light_type,
                self._road_edge_mask(map_type),
                None,
                None,
            )

        scene_position, scene_heading = self._scene_ego_pose(
            agent,
            position.device,
        )
        if batch.numel() and (batch.min() < 0 or batch.max() >= len(scene_position)):
            raise ValueError("Map batch contains an invalid scene index.")

        distance = torch.linalg.vector_norm(
            position[..., :2] - scene_position[batch],
            dim=-1,
        )
        init_range = float(self.token_processor.init_map_range)
        context = distance < init_range + self.pl2pl_radius

        map_type = map_type[context]
        batch = batch[context]
        position = position[context]
        orientation = orientation[context]
        token_index = token_index[context]
        light_type = light_type[context]
        distance = distance[context]

        output = (
            (distance < init_range)
            & self._road_edge_mask(map_type)
        )
        return (
            map_type,
            batch,
            position,
            orientation,
            token_index,
            light_type,
            output,
            scene_position,
            scene_heading,
        )

    # ------------------------------------------------------------------
    # Embedding and graph attention
    # ------------------------------------------------------------------
    def _embed(
        self,
        map_type: Tensor,
        token_index: Tensor,
        light_type: Tensor,
    ) -> Tensor:
        token_source = self.token_processor.map_token_traj_src
        if not torch.is_tensor(token_source):
            raise TypeError("map_token_traj_src must be a tensor.")

        token_source = token_source.to(token_index.device)
        token_source = token_source.reshape(len(token_source), -1)
        if token_source.shape[-1] != 22:
            raise ValueError(
                f"Map trajectory tokens must flatten to 22 values, "
                f"got {token_source.shape[-1]}."
            )

        self._check_index(token_index, len(token_source), "map token index")
        parameter = next(self.token_emb.parameters())
        table = self.token_emb(token_source.to(parameter.dtype))

        polygon_type = self._polygon_type[map_type]
        return (
            table[token_index]
            + self.type_pt_emb(map_type)
            + self.polygon_type_emb(polygon_type)
            + self.light_pl_emb(light_type)
        )

    @staticmethod
    def _attention_tensor(result):
        """Support AttentionLayer returning either Tensor or (Tensor, aux)."""
        if isinstance(result, tuple):
            if not result:
                raise RuntimeError("AttentionLayer returned an empty tuple.")
            return result[0]
        return result

    def _edges(
        self,
        source_position: Tensor,
        source_orientation: Tensor,
        source_batch: Tensor,
        target_position: Tensor,
        target_orientation: Tensor,
        target_batch: Tensor,
    ):
        if self.edge_encoder is None:
            raise RuntimeError("edge_encoder is unavailable.")

        heading_vector = torch.stack(
            [target_orientation.cos(), target_orientation.sin()],
            dim=-1,
        )
        return self.edge_encoder.build_map2map_edge(
            source_position,
            source_orientation,
            target_position,
            target_orientation,
            heading_vector,
            target_batch,
            source_batch,
            self.pl2pl_radius,
            self.pt2pt_neighbor,
        )

    def _encode_context(
        self,
        feature: Tensor,
        position: Tensor,
        orientation: Tensor,
        batch: Tensor,
        output_mask: Tensor,
    ):
        if not output_mask.any():
            return (
                feature.new_empty((0, self.hidden_dim)),
                position[output_mask],
                orientation[output_mask],
                batch[output_mask],
            )

        # No graph layer: still return only the requested road-edge nodes.
        if self.num_layers == 0:
            return (
                feature[output_mask],
                position[output_mask],
                orientation[output_mask],
                batch[output_mask],
            )

        # One layer updates selected targets from all nearby context nodes.
        if self.num_layers == 1:
            target_feature = feature[output_mask]
            target_position = position[output_mask]
            target_orientation = orientation[output_mask]
            target_batch = batch[output_mask]

            edge_index, relation = self._edges(
                position,
                orientation,
                batch,
                target_position,
                target_orientation,
                target_batch,
            )
            target_feature = self._attention_tensor(
                self.pt2pt_layers[0](
                    (feature, target_feature),
                    relation,
                    edge_index,
                )
            )
            return (
                target_feature,
                target_position,
                target_orientation,
                target_batch,
            )

        # Multiple layers update all context nodes before selecting outputs.
        edge_index, relation = self._edges(
            position,
            orientation,
            batch,
            position,
            orientation,
            batch,
        )
        for layer in self.pt2pt_layers:
            feature = self._attention_tensor(
                layer(
                    (feature, feature),
                    relation,
                    edge_index,
                )
            )

        return (
            feature[output_mask],
            position[output_mask],
            orientation[output_mask],
            batch[output_mask],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        tokenized_map: Mapping[str, Tensor],
        tokenized_agent: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Tensor]:
        values = self._read_map(tokenized_map)
        (
            map_type,
            batch,
            position,
            orientation,
            token_index,
            light_type,
            output_mask,
            scene_position,
            scene_heading,
        ) = self._crop(*values, tokenized_agent)

        if len(position) == 0 or not output_mask.any():
            return {
                "pt_token": position.new_empty((0, self.hidden_dim)),
                "position": position[output_mask],
                "orientation": orientation[output_mask],
                "batch": batch[output_mask],
            }

        feature = self._embed(
            map_type,
            token_index,
            light_type,
        )
        feature, position, orientation, batch = self._encode_context(
            feature,
            position,
            orientation,
            batch,
            output_mask,
        )

        if tokenized_agent is not None:
            if scene_position is None or scene_heading is None:
                raise RuntimeError("Scene ego pose was not prepared.")
            position, orientation = transform_to_local(
                position,
                orientation,
                scene_position[batch],
                scene_heading[batch],
            )

        return {
            "pt_token": feature,
            "position": position,
            "orientation": orientation,
            "batch": batch,
        }
