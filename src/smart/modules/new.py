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
import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_scatter import scatter_mean

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from .kl_loss import DiagGaussian
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from .build_edge import radiusGraphNearest, radiusGraphNearest2, positionalencoding1d, generate_causal_mask, \
    generate_limited_causal_mask
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock, general_rope, \
    general_rope1


class SMARTAgentDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_historical_steps: int,
            num_future_steps: int,
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
            use_latent: bool = False
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.alpha = 0.1

        self.head_dim = hidden_dim // num_heads

        self.pred_agent = False

        if self.pred_agent:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

            self.x_a_emb = FourierEmbedding(
                input_dim=input_dim_x_a,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
            self.token_emb_veh = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_ped = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_cyc = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.fusion_emb = MLPEmbedding(
                input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
            )

            self.agent_hist = self.time_span // self.shift

            self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
            self.pt2a_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
            self.a2a_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

            self.token_predict_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
            )

        self.pred_light = True

        if self.pred_light:
            self.lg_time_span = time_span

            self.light_hist = time_span // self.shift

            self.light_type = 5

            self.light_dropout = 0

            self.light_embedding = nn.Embedding(self.light_type, hidden_dim)

            self.lg_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
            self.lg2lg_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

            self.light_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)

        self.pred_route = False

        self.apply(weight_init)

    def agent_token_embedding(
            self,
            agent_token_index,  # [n_agent, n_step]
            trajectory_token_veh,  # [n_token, 8]
            trajectory_token_ped,  # [n_token, 8]
            trajectory_token_cyc,  # [n_token, 8]
            pos_a,  # [n_agent, n_step, 2]
            head_vector_a,  # [n_agent, n_step, 2]
            agent_type,  # [n_agent]
            agent_shape,  # [n_agent, 3]
            inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    def temporal_embed(self, feature, network, n_step, n_current, hist_len, mask=None):
        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        positions = torch.arange(n_current, n_step + n_current, device=feature.device)[:, None]

        sinusoidal_pos = general_rope(positions, self.head_dim)

        if mask is not None:
            causal_mask = causal_mask & mask

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature

    def spatial_embed(self, feature, network, lengths, sinusoidal_poshead, spatial_mask=None, mask=None):

        padded_feature = pad_sequence(torch.split(feature, lengths), batch_first=True, padding_value=0)

        padded_feature = padded_feature.permute(2, 0, 1, 3).flatten(0, 1)

        src_key_padding_mask = (padded_feature.abs().sum(dim=-1) == 0)

        if mask is not None:
            attn_mask = src_key_padding_mask & mask

        if spatial_mask is not None:
            attn_mask = src_key_padding_mask[:, :, None] & src_key_padding_mask[:, None] & spatial_mask
        else:
            attn_mask = src_key_padding_mask[:, :, None]

        padded_feature = network(padded_feature, attn_mask[:, None], sinusoidal_pos=sinusoidal_poshead)

        feature = padded_feature[~src_key_padding_mask].reshape(feature.shape[1], feature.shape[0], -1).swapaxes(0, 1)

        return feature

    def predict_light_roformer(self, light_idx, sinusoidal_poshead, lengths, n_current=0):
        n_step = light_idx.shape[1]

        feat_lg = self.light_embedding(light_idx)

        feat_lg = self.temporal_embed(feat_lg, self.lg_t_roformer, n_step, n_current, self.light_hist)

        (padded_sin, padded_cos, spatial_mask) = sinusoidal_poshead

        # sin=padded_sin.repeat_interleave(n_step,dim=0)
        #
        # cos=padded_cos.repeat_interleave(n_step,dim=0)

        sin = padded_sin[None].repeat(n_step, 1, 1, 1, 1).flatten(0, 1)  # [:,None]

        cos = padded_cos[None].repeat(n_step, 1, 1, 1, 1).flatten(0, 1)  # [:,None]

        # spatial_mask=spatial_mask.repeat_interleave(n_step,dim=0)

        feat_lg = self.spatial_embed(feat_lg, self.lg2lg_roformer, lengths, (sin, cos))

        next_light_logits = self.light_token_predict_head(feat_lg)

        return feat_lg, next_light_logits

    def predict_agent_roformer(self, tokenized_agent, map_feature, n_current=0):
        n_step = tokenized_agent["sampled_idx"].shape[1]
        mask = tokenized_agent["valid_mask"][:, :n_step]
        pos_a = tokenized_agent["sampled_pos"][:, :n_step]
        head_a = tokenized_agent["sampled_heading"][:, :n_step]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"][:, :n_step],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        feat_a = self.temporal_embed(feat_a, self.a_t_roformer, n_step, n_current, self.agent_hist, mask)

        feat_a = self.spatial_embed(feat_a, self.a2a_roformer, n_agent, n_current)

        return feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"]
            lengths = tokenized_agent["lengths"]
            sinusoidal_poshead = tokenized_agent["sinusoidal_poshead"]

            noised_light_idx = light_idx.clone()

            random_light = torch.randint(low=0, high=3, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            noised_light_idx[random_mask] = random_light[random_mask]

            feat_lg, next_light_logits = self.predict_light_roformer(noised_light_idx, sinusoidal_poshead, lengths)

        if not self.pred_agent:
            tokenized_agent["next_light_logits"] = next_light_logits
            tokenized_agent["feat_lg"] = feat_lg

            return {
                "q_value": next_light_logits[:, 1:],
            }
        result = self.predict_agent_roformer(tokenized_agent, map_feature)

    def autoregressive_light_prediction(self, predicted_tokens, tokenized_agent, max_len):
        current_len = predicted_tokens.shape[1]
        lengths = tokenized_agent["lengths"]
        sinusoidal_poshead = tokenized_agent["sinusoidal_poshead"]

        self.lg_t_roformer.attn.kv_caching(self.light_hist)

        for t in range(current_len, max_len + current_len):
            if t == current_len:
                if "feat_lg" in tokenized_agent.keys():
                    lg_features = tokenized_agent["feat_lg"][:, :current_len]
                    next_light_logits = tokenized_agent["next_light_logits"][:, :current_len]

                    self.lg_t_roformer.attn.cached_k = self.lg_t_roformer.attn.cached_k[:, :, :current_len]
                    self.lg_t_roformer.attn.cached_v = self.lg_t_roformer.attn.cached_v[:, :, :current_len]
                else:
                    lg_features, next_light_logits = self.predict_light_roformer(predicted_tokens, sinusoidal_poshead,
                                                                                 lengths)
            else:
                feat_lg, next_light_logits = self.predict_light_roformer(predicted_tokens[:, -1:], sinusoidal_poshead,
                                                                         lengths, t - 1)

                lg_features = torch.cat([lg_features, feat_lg[:, -1:]], dim=1)

            cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

            samples = cat_dist.sample()

            predicted_tokens = torch.cat([predicted_tokens, samples[:, None]], dim=1)

        self.lg_t_roformer.attn.kv_caching(0)

        #
        # print(torch.max(diff.abs()).item(),torch.max(diff[:,0].abs()).item())
        # feat_lg, next_light_logits = self.predict_light_roformer(predicted_tokens, sinusoidal_poshead,lengths)
        #
        # diff=lg_features-feat_lg[:,:-1]
        #
        # print(torch.mean(diff.abs()).item())

        return predicted_tokens, lg_features

    def inference(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            sampling_scheme: DictConfig
    ) -> Dict[str, torch.Tensor]:
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"][:, :step_current_2hz].clone()

            pred_light_idx, lg_features = self.autoregressive_light_prediction(light_idx, tokenized_agent,
                                                                               n_step_future_2hz)

        if not self.pred_agent:
            return {"light_idx": pred_light_idx}

        n_agent = tokenized_agent["valid_mask"].shape[0]

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()

        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )

        pred_valid = tokenized_agent["valid_mask"].clone()
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}

        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                batch_s=batch_s,  # [n_agent*hist_step]
                batch_pl=batch_pl,  # [n_pl*hist_step]
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            if self.pred_light:
                batch_lg = map_feature["batch_lg"]

                if t == 0:  # init
                    batch_lg = torch.cat(
                        [
                            batch_lg + tokenized_agent["num_graphs"] * t
                            for t in range(hist_step)
                        ],
                        dim=0,
                    )
                edge_index_lg2a, r_lg2a = self.build_map2agent_edge(
                    pos_pl=map_feature["pos_lg"],  # [n_pl, 2]
                    orient_pl=map_feature["orient_lg"],  # [n_pl]
                    pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                    head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                    head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                    mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                    batch_s=batch_s,  # [n_agent*hist_step]
                    batch_pl=batch_lg,  # [n_pl*hist_step]
                    pl2a_radius=100
                )

            pt_token = map_feature["pt_token"]

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    # [hist_step*n_pl, hidden_dim]
                    _feat_map = (
                        pt_token
                        .unsqueeze(0)
                        .expand(hist_step, -1, -1)
                        .flatten(0, 1)
                    )

                    _feat_temporal = self.pt2a_attn_layers[i](
                        (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                    )

                    if self.pred_light:
                        _feat_temporal = self.lg2a_attn_layers[i](
                            (lg_features[:, :hist_step].flatten(0, 1), _feat_temporal), r_lg2a, edge_index_lg2a
                        )
                    if i == 0 and self.pred_route:
                        route_idx = torch.zeros_like(head_a).long() + self.route_type
                        route_embedding = self.route_embedding(route_idx)
                        _feat_temporal = _feat_temporal + route_embedding.view(-1, self.hidden_dim)

                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )  # 3 s history
                    # feat_a_now=_feat_temporal[:, -1]
                    # * give same results as below, but more efficient
                    # feat_a_now = self.t_attn_layers[i](
                    #     _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    # ).view(n_agent, n_step, -1)[:, -1]

                    feat_a_now = self.pt2a_attn_layers[i](
                        (pt_token, feat_a_now), r_pl2a, edge_index_pl2a
                    )
                    if self.pred_light:
                        feat_a_now = self.lg2a_attn_layers[i](
                            (lg_features[:, t_now], feat_a_now), r_lg2a, edge_index_lg2a
                        )
                    if i == 0 and self.pred_route:
                        next_route_logits = self.route_token_predict_head(feat_a_now)
                        cat_dist = Categorical(logits=next_route_logits / self.alpha)
                        route_idx = cat_dist.sample()  # [n_agent] in K

                        route_embedding = self.route_embedding(route_idx)
                        feat_a_now = feat_a_now + route_embedding

                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            next_token_idx, next_token_traj_all, prev_log_prob = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits / self.alpha,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=None,  # tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=None,  # tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=None,  # tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5: (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5: (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
            # "sample_list":sample_list,
            # "feat_a": torch.stack(feat_a_list,dim=1),
            # "action_log_probs": torch.stack(action_log_probs_list, dim=1),
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": pred_valid,  # [n_agent, 18]
            "sampled_idx": pred_idx,  # [n_agent, 18]
            # "rollout_entropy":torch.stack(entropy_list)
        }

        if self.pred_light:
            out_dict["light_idx"] = pred_light_idx
            # out_dict["lg_features"]=lg_features

        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
            out_dict["gt_pos_raw"] = tokenized_agent["gt_pos_raw"]  # [n_agent, 18, 2]
            out_dict["gt_head_raw"] = tokenized_agent["gt_head_raw"]  # [n_agent, 18]
            out_dict["gt_valid_raw"] = tokenized_agent["gt_valid_raw"]  # [n_agent, 18]

        return out_dict
