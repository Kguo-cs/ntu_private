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
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from torch.distributions import Categorical
from .build_edge import radiusGraphNearest2, nearest_mask, generate_limited_causal_mask, nearest_mask2, \
    radiusGraphNearest_head, radiusGraphNearest_inv
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.loss.iq_loss import padding



class LightEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            time_span: Optional[int],
            num_heads: int,
            light_type,
            shift,
            predict_step,
        ) -> None:
        super(LightEncoder, self).__init__()

        self.head_dim = hidden_dim // num_heads

        self.lg_time_span = time_span

        self.light_hist = time_span // shift
        self.light_type = light_type
        self.shift = shift
        self.light_dropout = 0
        self.predict_step=predict_step

        self.light_embedding = nn.Embedding(5, hidden_dim)

        self.lg_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
        self.lg2lg_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

        self.lg2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    dropout=self.light_dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(1)
            ]
        )

        # self.predict_feature = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                             output_dim=hidden_dim * self.predict_step)


        self.light_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=self.light_type* self.predict_step)

    def temporal_embed(self, feature, pos, heading, network, n_step, n_current, hist_len, mask):

        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        time = torch.arange(n_current, n_step + n_current, device=feature.device)[None, :, None]

        # pos_time =torch.concat([pos,time.repeat_interleave(len(pos),dim=0)],dim=-1)#time.repeat_interleave(len(pos),dim=0)#
        #
        # sinusoidal_pos = general_rope(pos_time, self.head_dim,heading)
        sinusoidal_pos = self.network.rotary_embedding(pos, heading, time)

        if mask is not None:
            causal_mask = causal_mask[None, None] | mask[:, None, None, :]

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature
    #
    # def light2agent(self, feat_a, feat_lg, tokenized_agent, sinusoidal_lg, pos_a, head_a, n_step):
    #
    #     sinusoidal_a = self.rotary_embedding(pos_a, head_a)
    #     lengths_a = torch.bincount(tokenized_agent["batch"]).tolist()
    #     padded_a_feature = padding(feat_a, lengths_a)
    #     feature_mask = (padded_a_feature[:, :, 0] != 0).any(-1)
    #
    #     sinusoidal_lg = sinusoidal_lg.repeat_interleave(n_step, dim=0)
    #     feat_lg = feat_lg.flatten(1, 2)
    #     padded_a_feature = self.lg2a_roformer(padded_a_feature, None, agent_sinusoidal, feat_lg, sinusoidal_lg)
    #
    #     feat_a = padded_a_feature[feature_mask]
    #
    #     return feat_a

    def get_lg_sinusoidal(self,tokenized_agent):
        if "lg_sinusoidal" in tokenized_agent.keys():
            lg_sinusoidal=tokenized_agent["lg_sinusoidal"]
        else:
            pos_lg, orient_lg, lengths_lg= tokenized_agent["pos_lg"], tokenized_agent["orient_lg"], tokenized_agent["lengths_lg"]
            lg_sinusoidal = self.lg2lg_roformer.rotary_embedding(pos_lg, orient_lg)
            lg_sinusoidal = padding(lg_sinusoidal, lengths_lg)
            tokenized_agent["lg_sinusoidal"]=lg_sinusoidal

        return lg_sinusoidal


    def light2agent(self,tokenized_agent,feat_a,feat_lg, n_step,pos_a,head_a,head_vector_a,mask,batch_s):

        pos_lg = tokenized_agent["pos_lg"]
        head_lg = tokenized_agent["orient_lg"]
        batch_lg = tokenized_agent["batch_lg"]

        batch_lg = torch.cat(
            [
                batch_lg + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )

        edge_index_lg2a, r_lg2a = self.build_map2agent_edge(
            pos_pl=pos_lg,  # [n_pl, 2]
            orient_pl=head_lg,  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s,  # [n_agent*n_step]
            batch_pl=batch_lg,  # [n_pl*n_step]
            pl2a_radius=100
        )

        feat_a = self.lg2a_attn_layers[0](
            (feat_lg.swapaxes(0,1).flatten(0, 1), feat_a), r_lg2a, edge_index_lg2a
        )

        return feat_a

    def predict_lightfeature(  self, feat_lg, lg_sinusoidal, lengths ,mask_lg ,n_step ) :

        padded_lg_feature = padding(feat_lg, lengths)

        feature_mask = (padded_lg_feature[:, :, 0] != 0).any(-1)

        padded_lg_feature = padded_lg_feature.swapaxes(1, 2).flatten(0, 1)

        padding_light_mask = padding(mask_lg[:, -n_step:], lengths, padding_value=True).swapaxes(1, 2).flatten(0, 1)

        lg_sinusoidal = lg_sinusoidal.repeat_interleave(n_step, dim=0)

        lg2lg_mask = padding_light_mask[:, None, None]

        padded_lg_feature = self.lg2lg_roformer(padded_lg_feature, lg2lg_mask, lg_sinusoidal)
        
        padded_lg_feature = padded_lg_feature.reshape(len(lengths), n_step, -1, padded_lg_feature.shape[-1])

        feat_lg = padded_lg_feature.swapaxes(1, 2)[feature_mask]

        next_light_logits = self.light_token_predict_head(feat_lg)

        return feat_lg,next_light_logits

    def forward(self, light_idx, mask_lg, lg_sinusoidal, lengths, n_current=0):
        n_light, n_step = light_idx.shape[0], light_idx.shape[1]

        feat_lg = self.light_embedding(light_idx)

        feat_lg = self.temporal_embed(feat_lg, None, None, self.lg_t_roformer, n_step, n_current, self.light_hist,
                                      mask_lg)

        padded_lg_feature = padding(feat_lg, lengths)

        feature_mask = (padded_lg_feature[:, :, 0] != 0).any(-1)

        padded_lg_feature = padded_lg_feature.swapaxes(1, 2).flatten(0, 1)

        padding_light_mask = padding(mask_lg[:, -n_step:], lengths, padding_value=True).swapaxes(1, 2).flatten(0,
                                                                                                               1)

        lg_sinusoidal = lg_sinusoidal.repeat_interleave(n_step, dim=0)

        lg2lg_mask = padding_light_mask[:, None, None]

        padded_lg_feature = self.lg2lg_roformer(padded_lg_feature, lg2lg_mask, lg_sinusoidal)

        padded_lg_feature = padded_lg_feature.reshape(len(lengths), n_step, -1, padded_lg_feature.shape[-1])

        feat_lg = padded_lg_feature.swapaxes(1, 2)[feature_mask]

        #        feat_lg=self.predict_feature(feat_lg)

        next_light_logits = self.light_token_predict_head(feat_lg).reshape(n_light, n_step, self.predict_step, -1)

        return feat_lg, next_light_logits

    def autoregressive_light_predict(self, tokenized_agent, current_step, max_step):
        predicted_tokens = tokenized_agent["light_idx"][:, :current_step].clone()
        lengths_lg = tokenized_agent["lengths_lg"]
        pos_lg = tokenized_agent["pos_lg"]
        orient_lg = tokenized_agent["orient_lg"]

        lg_sinusoidal = self.rotary_embedding(pos_lg, orient_lg)
        lg_sinusoidal = padding(lg_sinusoidal, lengths_lg)

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "feat_lg" in tokenized_agent.keys():
                    lg_features = tokenized_agent["feat_lg"][:, :current_step]
                    next_light_logits = tokenized_agent["next_light_logits"][:, :current_step]

                    self.lg_t_roformer.attn.cached_k = self.lg_t_roformer.attn.cached_k[:, :, :current_step]
                    self.lg_t_roformer.attn.cached_v = self.lg_t_roformer.attn.cached_v[:, :, :current_step]
                else:
                    lg_features, next_light_logits = self.predict_light(predicted_tokens, lg_sinusoidal,
                                                                        lengths_lg)
                self.lg_t_roformer.attn.kv_caching(self.light_hist)
            else:
                feat_lg, next_light_logits = self.predict_light(predicted_tokens[:, -1:],
                                                                lg_sinusoidal, lengths_lg, t - 1)

                lg_features = torch.cat([lg_features, feat_lg[:, -1:]], dim=1)

            cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

            samples = cat_dist.sample()

            predicted_tokens = torch.cat([predicted_tokens, samples[:, None]], dim=1)

        self.lg_t_roformer.attn.kv_caching(0)

        out_dict = {"light_idx": predicted_tokens,
                    }

        return out_dict, lg_features
