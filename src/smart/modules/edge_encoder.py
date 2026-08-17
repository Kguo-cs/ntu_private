import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.pool import knn, knn_graph
from torch_scatter import scatter_mean

from src.smart.layers.fourier_embedding import FourierEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    project_to_local_frame,
    wrap_angle,
)
from src.smart.utils.edge_utils import radiusGraphNearest,radiusGraphNearest2

def _empty_edges(device) -> Tensor:
    return torch.empty(2, 0, dtype=torch.long, device=device)


def _validate_edge_index(
    edge_index: Tensor,
    num_source: int,
    num_target: Optional[int] = None,
) -> Tensor:
    """Validate a source-to-target edge index."""
    num_target = num_source if num_target is None else num_target
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index [2,E], got {edge_index.shape}.")
    edge_index = edge_index.long()
    if edge_index.numel():
        if edge_index[0].min() < 0 or edge_index[0].max() >= num_source:
            raise IndexError("Invalid source index.")
        if edge_index[1].min() < 0 or edge_index[1].max() >= num_target:
            raise IndexError("Invalid target index.")
    return edge_index

def _compressed_mask(
    value: Optional[Tensor],
    valid: Optional[Tensor],
    num_nodes: int,
    name: str,
) -> Optional[Tensor]:
    """Accept either a dense mask or a mask over already-valid nodes."""
    if value is None:
        return None
    value = value.reshape(-1)
    if value.numel() == num_nodes:
        return value
    if valid is not None and value.numel() == valid.numel():
        value = value[valid]
        if value.numel() == num_nodes:
            return value
    raise ValueError(f"{name} has {value.numel()} values; expected {num_nodes}.")


def _remap_agent_to_time_major(edge_index: Tensor, mask: Tensor) -> Tensor:
    """Remap compact agent-major destinations to compact time-major."""
    num_agent, num_step = mask.shape
    if num_step == 1 or edge_index.numel() == 0:
        return edge_index

    agent_valid = mask.reshape(-1)
    time_valid = mask.T.reshape(-1)
    kept_agent = agent_valid.nonzero(as_tuple=True)[0]

    global_agent = kept_agent[edge_index[1]]
    agent = torch.div(global_agent, num_step, rounding_mode="floor")
    step = global_agent.remainder(num_step)
    global_time = step * num_agent + agent

    dense_to_compact = torch.full(
        (num_agent * num_step,),
        -1,
        dtype=torch.long,
        device=mask.device,
    )
    kept_time = time_valid.nonzero(as_tuple=True)[0]
    dense_to_compact[kept_time] = torch.arange(
        len(kept_time), device=mask.device
    )

    target = dense_to_compact[global_time]
    if (target < 0).any():
        raise RuntimeError("Failed to remap map-to-agent edges.")
    return torch.stack([edge_index[0], target])


class EdgeEncoder(nn.Module):
    """Build temporal, agent-agent, and map-agent relation embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_freq_bands: int,
        hist_drop_prob: float = 0.0,
        time_span: Optional[int] = 30,
        shift: int = 0,
        discriminator: bool = False,
        use_bird: bool = False,
        use_pl2a: bool = False,
        use_a2a: bool = False,
        use_t2t: bool = False,
        differentiable_edge: bool = True,
    ) -> None:
        super().__init__()

        if not 0 <= hist_drop_prob <= 1:
            raise ValueError("hist_drop_prob must be in [0,1].")
        if use_t2t and shift <= 0:
            raise ValueError("shift must be positive when use_t2t=True.")

        self.hist_drop_prob = hist_drop_prob
        self.time_span = time_span
        self.shift = shift
        self.discriminator = discriminator
        self.differentiable_edge = False#differentiable_edge
        self.rollout_traj = False
        self.use_t2t = use_t2t
        self.use_a2a = use_a2a
        self.use_pl2a = use_pl2a
        self.tokenized_pos = False

        spatial_dim = 4 if use_bird else 3
        if use_pl2a:
            self.r_pt2a_emb = FourierEmbedding(
                input_dim=spatial_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
        if use_a2a:
            self.r_a2a_emb = FourierEmbedding(
                input_dim=spatial_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
        if use_t2t:
            self.r_t_emb = FourierEmbedding(
                input_dim=spatial_dim + 1,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

    @staticmethod
    def _select_agents(
        value: Optional[Tensor],
        train_mask: Optional[Tensor],
        full_agents: int,
        selected_agents: int,
        name: str,
    ) -> Optional[Tensor]:
        if value is None:
            return None
        if value.shape[0] == selected_agents:
            return value
        if train_mask is not None and value.shape[0] == full_agents:
            return value[train_mask]
        raise ValueError(f"{name} has an incompatible agent dimension.")

    def build_temporal_edge(
        self,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Tensor,
        inference_mask: Optional[Tensor] = None,
        agent_train_mask: Optional[Tensor] = None,
    ):
        if not self.use_t2t:
            raise RuntimeError("Temporal embedding is disabled.")

        full_agents = len(pos_a)
        train_mask = (
            None if agent_train_mask is None else agent_train_mask
        )
        if train_mask is not None:
            if train_mask.shape != (full_agents,):
                raise ValueError("agent_train_mask has the wrong shape.")
            pos_a = pos_a[train_mask]
            head_a = head_a[train_mask]
            head_vector_a = head_vector_a[train_mask]
            mask = mask[train_mask]

        inference_mask = self._select_agents(
            inference_mask,
            train_mask,
            full_agents,
            len(pos_a),
            "inference_mask",
        )

        mask = mask.clone()

        source_valid = mask.clone()
        if self.training and self.hist_drop_prob:
            source_valid &= (
                torch.rand(mask.shape, device=mask.device)
                >= self.hist_drop_prob
            )

        target_valid = mask.clone()
        if inference_mask is not None:
            target_valid &= inference_mask

        num_agent, num_step = mask.shape
        max_lag = num_step - 1
        if self.time_span is not None:
            max_lag = min(
                max_lag,
                max(0, math.floor(self.time_span / self.shift)),
            )

        agents, source_steps, target_steps = [], [], []
        for lag in range(1, max_lag + 1):
            agent, source = (
                source_valid[:, :-lag] & target_valid[:, lag:]
            ).nonzero(as_tuple=True)
            agents.append(agent)
            source_steps.append(source)
            target_steps.append(source + lag)

        if agents:
            agent = torch.cat(agents)
            source_step = torch.cat(source_steps)
            target_step = torch.cat(target_steps)
        else:
            agent = torch.empty(0, dtype=torch.long, device=mask.device)
            source_step = target_step = agent

        relative_pos = (
            pos_a[agent, source_step] - pos_a[agent, target_step]
        )
        relative_head = wrap_angle(
            head_a[agent, source_step] - head_a[agent, target_step]
        )
        local_pos = project_to_local_frame(
            relative_pos,
            head_vector_a[agent, target_step],
            self.differentiable_edge,
        )
        relation = self.r_t_emb(
            continuous_inputs=torch.cat(
                [
                    local_pos,
                    relative_head[:, None],
                    (source_step - target_step).to(pos_a.dtype)[:, None],
                ],
                dim=-1,
            ),
            categorical_embs=None,
        )

        time_valid = mask.T.reshape(-1)
        dense_to_compact = torch.full(
            (num_agent * num_step,),
            -1,
            dtype=torch.long,
            device=mask.device,
        )
        kept = time_valid.nonzero(as_tuple=True)[0]
        dense_to_compact[kept] = torch.arange(
            len(kept), device=mask.device
        )

        source = dense_to_compact[source_step * num_agent + agent]
        target = dense_to_compact[target_step * num_agent + agent]
        edge_index = torch.stack([source, target])

        return edge_index, relation

    def build_interaction_edge(
        self,
        pos_s: Tensor,
        head_s: Tensor,
        head_vector_s: Tensor,
        batch_s: Tensor,
        mask: Optional[Tensor],
        max_num_neighbors: int,
        max_radius: float,
        agent_train_mask: Optional[Tensor] = None,
        layer_num: int = 1,
        dis_edge_mask: Optional[Tensor] = None,
        a2a_edge_index: Optional[Tensor] = None,
    ):
        valid = None
        if mask is not None:
            valid = mask.reshape(-1)
            pos_s = pos_s[valid]
            head_s = head_s[valid]
            head_vector_s = head_vector_s[valid]
            batch_s = batch_s[valid]

        batch_s = batch_s.reshape(-1)
        if len(batch_s) != len(pos_s):
            raise ValueError("batch_s must match pos_s.")

        edge_index = (
            radiusGraphNearest(
                pos_s, batch_s, max_radius, False, max_num_neighbors
            )
            if a2a_edge_index is None
            else _validate_edge_index(a2a_edge_index, len(pos_s))
        )

        if agent_train_mask is not None and layer_num == 1:
            train = _compressed_mask(
                agent_train_mask, valid, len(pos_s), "agent_train_mask"
            )
            edge_index = edge_index[:, train[edge_index[1]]]

        if dis_edge_mask is not None:
            dis_mask = _compressed_mask(
                dis_edge_mask, valid, len(pos_s), "dis_edge_mask"
            )
            edge_index = edge_index[:, dis_mask[edge_index[1]]]

        source, target = edge_index
        relative_pos = pos_s[source] - pos_s[target]
        relative_head = wrap_angle(head_s[source] - head_s[target])
        distance = torch.linalg.vector_norm(relative_pos, dim=-1)

        local_pos = project_to_local_frame(
            relative_pos,
            head_vector_s[target],
            self.differentiable_edge,
        )
        relation = self.r_a2a_emb(
            continuous_inputs=torch.cat(
                [local_pos, relative_head[:, None]], dim=-1
            ),
            categorical_embs=None,
        )

        neighbor_relation = center_pos = center_heading = None

        return (
            edge_index,
            relation,
            distance,
            relative_pos,
            neighbor_relation,
            center_pos,
            center_heading,
        )

    def _map_relation(
        self,
        pos_pl: Tensor,
        orient_pl: Tensor,
        pos_s: Tensor,
        head_s: Tensor,
        head_vector_s: Tensor,
        edge_index: Tensor,
        categorical: Optional[Tensor] = None,
    ) -> Tensor:
        source, target = edge_index
        relative_pos = pos_pl[source] - pos_s[target]
        relative_head = wrap_angle(orient_pl[source] - head_s[target])
        local_pos = project_to_local_frame(
            relative_pos,
            head_vector_s[target],
            self.differentiable_edge,
        )
        return self.r_pt2a_emb(
            continuous_inputs=torch.cat(
                [local_pos, relative_head[:, None]], dim=-1
            ),
            categorical_embs=categorical,
        )

    def build_map2map_edge(
        self,
        pos_pl: Tensor,
        orient_pl: Tensor,
        pos_s: Tensor,
        head_s: Tensor,
        head_vector_s: Tensor,
        batch_s: Tensor,
        batch_pl: Tensor,
        pl2a_radius: float,
        max_num_neighbors: int,
        l2l_edge_index: Optional[Tensor] = None,
        l2l_feature: Optional[Tensor] = None,
    ):
        if not self.use_pl2a:
            raise RuntimeError("Map embedding is disabled.")

        edge_index = (
            radiusGraphNearest2(
                pos_s,
                pos_pl,
                pl2a_radius,
                batch_s,
                batch_pl,
                max_num_neighbors,
            )
            if l2l_edge_index is None
            else _validate_edge_index(
                l2l_edge_index, len(pos_pl), len(pos_s)
            )
        )
        relation = self._map_relation(
            pos_pl,
            orient_pl,
            pos_s,
            head_s,
            head_vector_s,
            edge_index,
            l2l_feature,
        )
        return edge_index, relation

    def build_map2agent_edge(
        self,
        pos_pl: Tensor,
        orient_pl: Tensor,
        pos_a: Tensor,
        head_a: Tensor,
        head_vector_a: Tensor,
        mask: Optional[Tensor],
        batch_s: Tensor,
        batch_pl: Tensor,
        pl2a_radius: float,
        max_num_neighbors: int,
        agent_train_mask: Optional[Tensor] = None,
        layer_num: int = 1,
        l2a_edge_index: Optional[Tensor] = None,
    ):

        if not self.use_pl2a:
            raise RuntimeError("Map embedding is disabled.")

        if pos_a.ndim == 2:
            pos_a = pos_a[:, None]
            head_a = head_a[:, None]
            head_vector_a = head_vector_a[:, None]
        if pos_a.ndim != 3:
            raise ValueError("pos_a must be [A,T,D] or [A,D].")

        if mask is None:
            mask = torch.ones(
                pos_a.shape[:2], dtype=torch.bool, device=pos_a.device
            )
        else:
            mask = mask.clone()

        num_agent, num_step = mask.shape
        if agent_train_mask is not None and layer_num == 1:
            if agent_train_mask.shape != (num_agent,):
                raise ValueError("agent_train_mask has the wrong shape.")
            mask &= agent_train_mask[:, None]

        if batch_s.shape != mask.shape:
            if batch_s.numel() != mask.numel():
                raise ValueError("batch_s must have A*T values.")
            batch_s = batch_s.reshape_as(mask)

        pos_s = pos_a[mask]
        head_s = head_a[mask]
        head_vector_s = head_vector_a[mask]
        batch_agent_major = batch_s[mask]

        edge_index = (
            radiusGraphNearest2(
                pos_s,
                pos_pl,
                pl2a_radius,
                batch_agent_major,
                batch_pl,
                max_num_neighbors,
            )
            if l2a_edge_index is None
            else _validate_edge_index(
                l2a_edge_index, len(pos_pl), len(pos_s)
            )
        )
        relation = self._map_relation(
            pos_pl,
            orient_pl,
            pos_s,
            head_s,
            head_vector_s,
            edge_index,
        )

        edge_index = _remap_agent_to_time_major(edge_index, mask)
        return edge_index, relation
