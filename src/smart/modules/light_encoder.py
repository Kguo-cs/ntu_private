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
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.loss.iq_loss import padding



class LightEncoder(nn.Module):
    def __init__(
            self,
            edge_encoder,
            hidden_dim: int,
            light_hist: Optional[int],
            num_heads: int,
            light_type,
            shift,
            pred_light,
            alpha
        ) -> None:
        super(LightEncoder, self).__init__()

        self.head_dim = hidden_dim // num_heads

        self.light_hist = light_hist
        self.light_type = light_type
        self.shift = shift
        self.light_dropout = 0

        self.light_embedding = nn.Embedding(5, hidden_dim)

        self.share=False
        self.alpha=alpha

        self.pred_light=pred_light

        if pred_light:
            self.autoRegressive_light=True

            if not self.autoRegressive_light:
                self.use_gnn=True
                self.share=True

                if self.use_gnn:
                    self.edge_encoder=edge_encoder
                    self.lg2lg_layers = AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=self.head_dim,
                        dropout=self.light_dropout,
                        bipartite=False,
                        has_pos_emb=True,
                    )
                else:
                    self.lg2lg_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

            if not self.share:
                self.lg_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout,hist_len=self.light_hist)

        self.light_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=self.light_type)


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

    def forward(self, tokenized_agent,light_idx, mask_lg, batch_lg,n_step, n_current, feat_lg=None,feat_lg_t=None):
        n_light, light_step = light_idx.shape[0], light_idx.shape[1]

        if self.autoRegressive_light:
            lengths_lg=tokenized_agent["lengths_lg"]
            pad_pos=tokenized_agent["pad_pos_lg"]
            pad_orient=tokenized_agent["pad_orient_lg"]

            padded_lg_feature = padding(feat_lg, lengths_lg)#b,agent,t,

            feature_mask = (padded_lg_feature[:, :, 0] != 0).any(-1)

            pad_pos_lg=pad_pos[:,:,None].repeat(1,1,light_step,1).flatten(1,2)
            pad_head_lg=pad_orient[:,:,None].repeat(1,1,light_step).flatten(1,2)

            feat_lg=padded_lg_feature.flatten(1,2)

            pad_mask=padding(mask_lg, lengths_lg)

            padding_light_mask = pad_mask.flatten(1,2)
            n_agent = pad_pos.shape[1]

            #if self.training:
            feat_lg = self.lg_t_roformer.temporal_embed(feat_lg, pad_pos_lg, pad_head_lg, light_step, n_current,  padding_light_mask,n_agent).reshape(padded_lg_feature.shape)

            feat_lg=feat_lg[feature_mask]
            feat_lg = feat_lg[:, -n_step:]

            next_light_logits = self.light_token_predict_head(feat_lg).reshape(n_light, n_step,  self.light_type)#$self.predict_step,
            # else:
            #            batch_size=tokenized_agent["num_graphs"]
            #     light_logit=torch.zeros((batch_size,n_agent,n_step,self.light_type),device=light_idx.device)-1e10
            #
            #     for i in range(n_agent):
            #         if self.lg_t_roformer.attn.caching==False:
            #             feat_lg=feat_lg[:,-1:]
            #             pad_pos_lg=pad_pos_lg[:,-1:]
            #             pad_head_lg=pad_head_lg[:,-1:]
            #
            #         feat_lg1 = self.lg_t_roformer.temporal_embed(feat_lg, pad_pos_lg, pad_head_lg, n_step+1, n_current,
            #                                                     padding_light_mask,n_agent)
            #
            #         last_feature=feat_lg1[:,-1]
            #
            #         next_light_logits = self.light_token_predict_head(last_feature).reshape(batch_size, self.light_type)
            #
            #         cat_dist = Categorical(logits=next_light_logits / self.alpha)
            #
            #         next_light_idx = cat_dist.sample()
            #
            #         light_logit[torch.arange(batch_size),i,-1, next_light_idx] = 0
            #
            #         next_light_feature=self.light_embedding(next_light_idx)
            #
            #         feat_lg=next_light_feature[:, None]
            #
            #         pad_pos_lg=pad_pos[:, i:i+1]
            #         pad_head_lg=pad_head[:, i:i+1]
            #
            #         padding_light_mask=torch.cat((padding_light_mask, pad_mask[:, i:i+1,0]), dim=1)[:, -self.light_hist*n_agent:]
            #         self.lg_t_roformer.attn.kv_caching(self.light_hist)
            #
            #     next_light_logits=light_logit[feature_mask]

        else:
            if not self.share:
                feat_lg = self.lg_t_roformer.temporal_embed(feat_lg, None, None, light_step, n_current,  mask_lg)

                feat_lg_t = feat_lg[:, -n_step:]

            mask_lg=mask_lg[:, -n_step:]

            if self.use_gnn:
                pos_lg=tokenized_agent["pos_lg"]
                head_lg=tokenized_agent["orient_lg"]
                head_vector_lg = torch.stack([head_lg.cos(), head_lg.sin()], dim=-1)

                edge_index_lg2lg, r_lg2lg=self.edge_encoder.build_interaction_edge(
                    pos_a=pos_lg[:,None].repeat(1,n_step,1),  # [n_agent, n_step, 2]
                    head_a=head_lg[:,None].repeat(1,n_step),  # [n_agent, n_step]
                    head_vector_a=head_vector_lg[:,None].repeat(1,n_step,1),  # [n_agent, n_step, 2]
                    batch_s=batch_lg,  # [n_agent*n_step]
                    mask=mask_lg,  # [n_agent, n_step]
                    max_radius=100,
                    max_num_neighbors=10
                )

                feat_lg_t = self.lg2lg_layers(feat_lg_t.transpose(0, 1).flatten(0, 1), r_lg2lg, edge_index_lg2lg)

                feat_lg=feat_lg_t.reshape( n_step, n_light, -1).swapaxes(0, 1)

            else:
                lengths=tokenized_agent["lengths_lg"]

                padded_lg_feature = padding(feat_lg, lengths)

                feature_mask = (padded_lg_feature[:, :, 0] != 0).any(-1)

                padded_lg_feature = padded_lg_feature.swapaxes(1, 2).flatten(0, 1)

                #padding_light_mask = padding(mask_lg, lengths).swapaxes(1, 2).flatten(0,1)

                lg_sinusoidal=self.get_lg_sinusoidal(tokenized_agent)

                lg_sinusoidal = lg_sinusoidal.repeat_interleave(n_step, dim=0)

                #lg2lg_mask = padding_light_mask[:, None, None]

                padded_lg_feature = self.lg2lg_roformer(padded_lg_feature, None, lg_sinusoidal)

                padded_lg_feature = padded_lg_feature.reshape(len(lengths), n_step, -1, padded_lg_feature.shape[-1])

                feat_lg = padded_lg_feature.swapaxes(1, 2)[feature_mask]

            next_light_logits = self.light_token_predict_head(feat_lg).reshape(n_light, n_step,  self.light_type)

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
