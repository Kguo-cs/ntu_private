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
from typing import Optional

import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder
from torch_scatter import scatter_max,scatter_mean,scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding

from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle



class InterativeDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            time_span: Optional[int],
            pl2a_radius: float,
            a2a_radius: float,
            num_freq_bands: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
            hist_drop_prob: float,
            n_token_agent: int,
            pt2a_neighbor: int,
            a2a_neighbor: int,
            token_processor,
            dis_weight=0,
            dist_decay=0,
            reward_weight=0,
            reward_decay=0,
            discriminator=False,
    ) -> None:
        super(InterativeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_span = time_span
        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob
        self.dis_weight=dis_weight
        self.dis_decay=dist_decay
        self.reward_weight=reward_weight
        self.reward_decay=reward_decay

        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor
        self.token_processor=token_processor

        self.head_dim = hidden_dim // num_heads

        self.agent_hist = self.time_span // self.shift
        self.n_token_agent=n_token_agent
        self.discriminator = discriminator
        self.use_decompose=True
        self.use_full_feature=False
        self.use_airl=False

        if self.discriminator and self.use_decompose and self.dis_decay <= 0:
            raise ValueError(
                f"dist_decay must be positive for the decomposed discriminator, "
                f"got {self.dis_decay}."
            )

        if self.token_processor.pred_init:
            self.gail_start_step=0 #first action not used
            if self.token_processor.learn_init:
                self.dis_start_step=0 # first state not used
            else:
                self.dis_start_step=1 # first state not used
        else:
            self.dis_start_step = 2  # first state not used
            self.gail_start_step=1 #first action not used

        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        hist_drop_prob=hist_drop_prob,
                                        time_span=time_span,
                                        shift=token_processor.shift,
                                        discriminator=discriminator,
                                        use_bird=token_processor.use_bird,
                                        use_pl2a=True,
                                        use_a2a=True,
                                        use_t2t=True,
                                        differentiable_edge=(token_processor.use_gradient_penalty or not discriminator)
                                        )

        if discriminator:
            self.t_num_layers = 1
        else:
            self.t_num_layers = num_layers

        if self.edge_encoder.use_t2t:
            self.t_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=hist_drop_prob,
                        bipartite=False,
                        has_pos_emb=True,
                    )
                    for _ in range(self.t_num_layers)
                ]
            )
        else:
            self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=hist_drop_prob,
                                              hist_len=self.agent_hist)

        if not token_processor.use_bird:
            self.pt2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=True,
                        has_pos_emb=True,
                    )
                    for _ in range(num_layers)
                ]
            )

        if not (discriminator and self.use_decompose and not self.use_full_feature):
            self.a2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=False,
                        has_pos_emb=True,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.pred_map_logit=False

        if self.discriminator:
            if self.use_decompose:
                self.interact_head = MLPLayer(
                    input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

                if self.pred_map_logit:
                    self.map_head = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

                if self.use_full_feature:
                    self.all_head = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

            if self.use_airl:
                self.token_predict_head1 = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )

        self.feat_a_cache=[[] for _ in range(num_layers)]

        self.apply(weight_init)

    def predict_agent(self, feat_a, feat_map,
                      r_t, edge_index_t,
                      r_pl2a, edge_index_pl2a,
                      r_a2a, edge_index_a2a,
                      agent_train_mask, dist,
                      train_repeat_mask, mask_a,
                      n_current, inference_mask,
                      token_embedding_later, pred_mask, n_agent,
                      feat_a_later_mask=None):
        """
        Run the decoder using two explicitly separated compressed-node spaces.

        feat_a:
            Features for every valid node in time-major order.

        feat_a_later_mask:
            Boolean mask over feat_a selecting valid nodes whose timestep is
            >= self.dis_start_step.  The a2a graph is built in exactly this
            post-start-step node space.

        train_repeat_mask:
            Optional Boolean mask over feat_a selecting train agents.  It is
            used for the ego branch.  The interaction branch retains all
            agents as potential neighbors but restricts destination agents.
        """
        mask_ta = mask_a.transpose(0, 1)  # [T, A_selected]
        mask_ta_flatten = mask_ta.flatten(0, 1)
        n_pred_agent = inference_mask.shape[0]
        n_step = mask_a.shape[1]
        current_len = int(inference_mask.sum().item())

        if self.discriminator:
            if feat_a_later_mask is None:
                raise ValueError(
                    "feat_a_later_mask is required for discriminator decoding."
                )

            # Boolean mask over the compressed ego-node space after any
            # optional train-agent selection.  This replaces the invalid
            # slicing pattern feat_a[n_agent * self.dis_start_step:].
            ego_later_mask_ta = mask_ta.clone()
            ego_later_mask_ta[:self.dis_start_step] = False
            ego_later_within_valid = ego_later_mask_ta.flatten(0, 1)[
                mask_ta_flatten
            ]

            # Boolean mask over post-start valid interaction nodes.  This is
            # aligned with edge_index_a2a and valid_interact_reward.
            if train_repeat_mask is not None:
                train_repeat_mask_later = train_repeat_mask[feat_a_later_mask]
            else:
                train_repeat_mask_later = None
        else:
            ego_later_within_valid = None
            train_repeat_mask_later = None

        for layer_i in range(self.num_layers):
            if self.use_decompose and self.discriminator:
                # edge_index_a2a indexes nodes after removing early states.
                start_index = edge_index_a2a[0]
                end_index = edge_index_a2a[1]

                feat_a_later = feat_a[feat_a_later_mask]

                start_edge_feature = feat_a_later[start_index]

                if token_embedding_later is not None:
                    if token_embedding_later.shape[0] != feat_a_later.shape[0]:
                        raise ValueError(
                            "token_embedding_later and feat_a_later must use "
                            "the same compressed node space: "
                            f"{token_embedding_later.shape[0]} != "
                            f"{feat_a_later.shape[0]}."
                        )
                    end_edge_feature = (
                        feat_a_later + token_embedding_later
                    )[end_index]
                else:
                    end_edge_feature = feat_a_later[end_index]

                # For the ego branch, retain only train agents.  The a2a
                # interaction branch above still sees non-train agents as
                # possible neighbors.
                if agent_train_mask is not None and self.num_layers == 1:
                    feat_a = feat_a[train_repeat_mask]

                if not self.token_processor.use_bird:
                    feat_a_pt = self.pt2a_attn_layers[layer_i](
                        (feat_map, feat_a), r_pl2a, edge_index_pl2a
                    )
                    if self.pred_map_logit:
                        map_logit= self.map_head(feat_a_pt[-current_len:][ego_later_within_valid])[:, 0]
                    else:
                        feat_a=feat_a_pt

                if self.use_full_feature:
                    feat_a_all = self.a2a_attn_layers[layer_i](
                        feat_a, r_a2a, edge_index_a2a
                    )
                    all_logits = self.all_head(feat_a_all)

                feat_interact = torch.cat(
                    [start_edge_feature, r_a2a, end_edge_feature], dim=-1
                )
                interact_logits = self.interact_head(feat_interact)
            else:
                feat_a = self.a2a_attn_layers[layer_i](
                    feat_a, r_a2a, edge_index_a2a
                )
                feat_a = self.pt2a_attn_layers[layer_i](
                    (feat_map, feat_a), r_pl2a, edge_index_pl2a
                )

            if not self.edge_encoder.rollout_traj and not self.discriminator:
                feat_a_t = torch.zeros(
                    [n_step, n_pred_agent, self.hidden_dim],
                    device=feat_a.device,
                    dtype=feat_a.dtype,
                )

                feat_a_t[mask_ta] = feat_a

                if n_current == 0:
                    self.feat_a_cache[layer_i] = feat_a_t
                else:
                    self.feat_a_cache[layer_i] = torch.cat(
                        (self.feat_a_cache[layer_i], feat_a_t), dim=0
                    )[-self.agent_hist:]

                    feat_a = self.feat_a_cache[layer_i][
                        self.mask_cache.transpose(0, 1)
                    ]

            feat_a = self.t_attn_layers[layer_i](feat_a, r_t, edge_index_t)

            feat_a = feat_a[-current_len:]

        if self.edge_encoder.rollout_traj:
            if pred_mask is not None:
                pred_repeat_mask = pred_mask[:, None].repeat(
                    1, n_step
                ).transpose(0, 1)
            else:
                pred_repeat_mask = torch.ones(
                    [n_step, n_agent], dtype=torch.bool, device=feat_a.device
                )

            pred_repeat_mask[:self.gail_start_step] = False
            pred_repeat_mask = pred_repeat_mask.flatten(0, 1)
            feat_a = feat_a[pred_repeat_mask]

        if self.discriminator:
           # feat_a=feat_a+feat_a_pt
            # Select post-start ego nodes using the actual validity mask.
            # Do not assume that all agents are valid at early timesteps.
            feat_a = feat_a[ego_later_within_valid]

        next_token_logits = self.token_predict_head(feat_a)

        weight = rewards = None

        if self.discriminator:
            next_token_logits=next_token_logits

            scene_reward = next_token_logits[:, 0].detach()

            if self.pred_map_logit:
                scene_reward=scene_reward+map_logit.detach()*2
            else:
                map_logit=None

            if self.use_decompose:
                valid_number = int(feat_a_later_mask.sum().item())
                #
                weight = torch.exp(-dist / self.dis_decay) * self.dis_weight

                #weight=torch.ones_like(weight)

                # weight_logit = interact_logits[:, 0].detach() * weight
                #
                # valid_interact_reward = scatter_sum(
                #     weight_logit,
                #     end_index,
                #     dim=0,
                #     dim_size=valid_number,
                # )

                valid_interact_reward, edge_weights, effective_mass = (
                    aggregate_interaction_reward(
                        interaction_logits=interact_logits[:, 0].detach(),
                        distances=dist,
                        destination_index=end_index,
                        num_nodes=valid_number,
                        distance_decay=self.dis_decay,
                        interaction_reward_weight=self.dis_weight,
                    )
                )
                #
                if train_repeat_mask_later is not None:
                    interaction_reward = valid_interact_reward[
                        train_repeat_mask_later
                    ]
                else:
                    interaction_reward = valid_interact_reward

                # scene_reward = scene_reward.clamp(-5.0, 5.0)
                # interaction_reward = interaction_reward.clamp(-5.0, 5.0)
                #
                total_reward = scene_reward + interaction_reward#

                #total_reward = total_reward.clamp(-8.0, 8.0)

                next_token_logits = (
                    next_token_logits[:, 0],
                    interact_logits[:, 0],
                    map_logit
                )
                nei_rewards = torch.zeros_like(total_reward)
            else:
                next_token_logits = (
                    next_token_logits[:, 0],
                    next_token_logits[:0, 0],
                )
                total_reward = scene_reward
                valid_interact_reward = scene_reward[:0]
                nei_rewards = total_reward[:0]

            rewards = (
                total_reward,
                nei_rewards,
                scene_reward,
                valid_interact_reward,
            )

        return next_token_logits, feat_a, rewards, weight

    def forward(self, all_features, feat_a, token_embedding, map_feature,
                agent_train_mask, n_current, tokenized_agent,
                counter_feat_a=None):
        pred_mask = tokenized_agent.get("pred_mask")
        train_mask = tokenized_agent.get("train_mask")

        if self.discriminator and token_embedding is not None and not self.use_airl:
            all_features = [feat[:, :-1] for feat in all_features]

        pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s = all_features

        n_agent, n_step = mask_a.shape
        mask_a_full = mask_a

        if n_current == 0:
            self.pos_cache = pos_a
            self.head_cache = head_a
            self.mask_cache = mask_a
            self.head_vector_cache = head_vector_a

            inference_mask = mask_a.clone()
        else:
            self.pos_cache = torch.cat(
                (self.pos_cache, pos_a), dim=1
            )[:, -self.agent_hist:]
            self.head_cache = torch.cat(
                (self.head_cache, head_a), dim=1
            )[:, -self.agent_hist:]
            self.mask_cache = torch.cat(
                (self.mask_cache, mask_a), dim=1
            )[:, -self.agent_hist:]
            self.head_vector_cache = torch.cat(
                (self.head_vector_cache, head_vector_a), dim=1
            )[:, -self.agent_hist:]

            inference_mask = self.mask_cache.clone()
            inference_mask[:, :-1] = False

        if agent_train_mask is not None:
            inference_mask = inference_mask[agent_train_mask]

        if self.edge_encoder.use_t2t:
            edge_index_t, r_t = self.edge_encoder.build_temporal_edge(
                pos_a=self.pos_cache,
                head_a=self.head_cache,
                head_vector_a=self.head_vector_cache,
                mask=self.mask_cache,
                inference_mask=inference_mask,
                agent_train_mask=agent_train_mask,
            )
        else:
            edge_index_t, r_t = None, None

        if not self.token_processor.use_bird:
            batch_pl = map_feature["batch"]
            pos_pl = map_feature["position"]
            orient_pl = map_feature["orientation"]
            feat_map = map_feature["pt_token"]

            if self.discriminator:
                feat_map = feat_map.detach()

            edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                pos_pl=pos_pl,
                orient_pl=orient_pl,
                pos_a=pos_a,
                head_a=head_a,
                head_vector_a=head_vector_a,
                mask=mask_a,
                batch_s=batch_s_repeat,
                batch_pl=batch_pl,
                pl2a_radius=self.pl2a_radius,
                max_num_neighbors=self.pt2a_neighbor,
                agent_train_mask=agent_train_mask,
                layer_num=self.num_layers,
            )
        else:
            edge_index_pl2a = r_pl2a = feat_map = None

        (
            pos_s,
            head_s,
            head_vector_s,
            mask_s_full,
            _,
            batch_s,
        ) = [feat.transpose(0, 1).flatten(0, 1) for feat in all_features]

        mask_s_full = mask_s_full.bool()

        if agent_train_mask is not None:
            dense_train_repeat_mask = agent_train_mask[None, :].expand(
                n_step, -1
            ).reshape(-1)

            # Boolean mask over compressed valid feat_a nodes.
            train_repeat_mask = dense_train_repeat_mask[mask_s_full]
            mask_a_for_predict = mask_a_full[agent_train_mask]

            if pred_mask is not None:
                pred_mask = pred_mask[agent_train_mask]
        else:
            dense_train_repeat_mask = None
            train_repeat_mask = None
            mask_a_for_predict = mask_a_full

        if self.discriminator:
            # Build one dense time-major mask for valid nodes at or after the
            # discriminator start step.  Every a2a tensor is derived from this
            # mask, ensuring a single shared compressed index space.
            mask_s_a2a = mask_s_full.clone()
            mask_s_a2a[:n_agent * self.dis_start_step] = False

            # Boolean mask over feat_a, which is already compressed by
            # mask_s_full in AgentTokenEncoder.
            feat_a_later_mask = mask_s_a2a[mask_s_full]

            # Boolean mask over interaction nodes after start-step filtering.
            if dense_train_repeat_mask is not None:
                train_repeat_mask_a2a = dense_train_repeat_mask[mask_s_a2a]
            else:
                train_repeat_mask_a2a = None
        else:
            mask_s_a2a = mask_s_full
            feat_a_later_mask = None
            train_repeat_mask_a2a = train_repeat_mask

        if self.discriminator and token_embedding is not None:
            if token_embedding.shape[0] != mask_s_full.numel():
                raise ValueError(
                    "token_embedding must be dense time-major [T*A, H] "
                    "before compression: "
                    f"{token_embedding.shape[0]} != {mask_s_full.numel()}."
                )

            token_embedding_valid = token_embedding[mask_s_full]
            token_embedding_later = token_embedding_valid[feat_a_later_mask]
        else:
            token_embedding_later = None

        if self.discriminator and "dis_mask" in tokenized_agent:
            dis_mask = tokenized_agent["dis_mask"].bool()
            later_mask_ta = mask_a_full.transpose(0, 1)[self.dis_start_step:]

            if train_mask is not None:
                all_dis_mask = torch.zeros_like(later_mask_ta)
                selected_shape = all_dis_mask[:, train_mask].shape
                all_dis_mask[:, train_mask] = dis_mask.reshape(selected_shape)
            else:
                all_dis_mask = dis_mask.reshape(later_mask_ta.shape)

            # Boolean mask over the same compressed post-start full-agent
            # interaction-node space used by edge_index_a2a.
            dis_edge_mask = all_dis_mask[later_mask_ta]
        else:
            dis_edge_mask = None

        (
            edge_index_a2a,
            r_a2a,
            dist,
            relative_pos,
            r_a2a_nei,
            center_nei_pos,
            center_nei_heading,
        ) = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,
            head_s=head_s,
            head_vector_s=head_vector_s,
            batch_s=batch_s,
            mask=mask_s_a2a,
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            agent_train_mask=train_repeat_mask_a2a,
            layer_num=self.num_layers,
            counter_feat_a=counter_feat_a,
            dis_edge_mask=dis_edge_mask,
        )

        next_token_logits, feat_a_value, rewards, weight = self.predict_agent(
            feat_a,
            feat_map,
            r_t,
            edge_index_t,
            r_pl2a,
            edge_index_pl2a,
            r_a2a,
            edge_index_a2a,
            agent_train_mask,
            dist,
            train_repeat_mask,
            mask_a_for_predict,
            n_current,
            inference_mask,
            token_embedding_later,
            pred_mask,
            n_agent,
            feat_a_later_mask=feat_a_later_mask,
        )

        return (
            next_token_logits,
            feat_a_value,
            rewards,
            weight,
            (edge_index_a2a, r_a2a, relative_pos),
        )

from torch import Tensor
from torch_scatter import scatter_sum

def aggregate_interaction_reward(
        interaction_logits: Tensor,
        distances: Tensor,
        destination_index: Tensor,
        num_nodes: int,
        distance_decay: float,
        interaction_reward_weight: float,
        eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate edge-level interaction logits into node-level rewards.

    The normalization prevents interaction reward magnitude from growing
    linearly with the number of neighbors, while preserving attenuation when
    all neighbors are far away.

    Args:
        interaction_logits:
            Edge-level discriminator logits with shape [E].
        distances:
            Edge distances in meters with shape [E].
        destination_index:
            Destination node index for every edge, shape [E].
        num_nodes:
            Number of destination nodes.
        distance_decay:
            Positive exponential-decay scale in meters.
        interaction_reward_weight:
            Relative contribution of the interaction branch.
        eps:
            Numerical stability constant.

    Returns:
        weighted_reward:
            Node-level interaction reward with shape [num_nodes].
        edge_weights:
            Unnormalized distance weights with shape [E].
        effective_mass:
            Sum of incoming edge weights per node.
    """

    edge_weights = torch.exp(-distances / distance_decay)

    weighted_sum = scatter_sum(
        edge_weights * interaction_logits,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )
    effective_mass = scatter_sum(
        edge_weights,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )

    # If mass < 1, distance attenuation remains active.
    # If mass > 1, aggregation becomes a weighted average.
    normalizer = effective_mass.clamp_min(1e-5)

    interaction_reward = weighted_sum / (normalizer + eps)
    interaction_reward = interaction_reward_weight * interaction_reward

    return interaction_reward, edge_weights, effective_mass