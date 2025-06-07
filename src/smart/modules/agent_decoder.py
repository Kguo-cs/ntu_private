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
from .build_edge import radiusGraphNearest, radiusGraphNearest2,nearest_mask,generate_limited_causal_mask,nearest_mask2,radiusGraphNearest_inv,radiusGraphNearest_head
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock, general_rope
from src.multi_agent.Qatten import QattenMixer
from torch_scatter import scatter_mean,scatter_max


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

        self.alpha = 0.1

        self.head_dim = hidden_dim // num_heads

        self.pred_agent = True

        if self.pred_agent:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

            input_dim_x_a = 2
            input_dim_token = 8

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

            self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=hist_drop_prob)


            self.use_gnn=True
            input_dim_x_a = 2
            input_dim_r_t = 4
            input_dim_r_pt2a = 3
            input_dim_r_a2a = 3
            input_dim_token = 8

            if self.use_gnn:
                self.r_pt2a_emb = FourierEmbedding(
                    input_dim=input_dim_r_pt2a,
                    hidden_dim=hidden_dim,
                    num_freq_bands=num_freq_bands,
                    )

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

                self.r_a2a_emb = FourierEmbedding(
                    input_dim=input_dim_r_a2a,
                    hidden_dim=hidden_dim,
                    num_freq_bands=num_freq_bands,
                )
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
            else:
                self.pt2a_roformer =nn.ModuleList(
                    [
                        RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,pos_emb=False)
                        for _ in range(self.num_layers)
                    ]
                )

                self.a2a_roformer =nn.ModuleList(
                    [
                        RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,pos_emb=False)
                        for _ in range(self.num_layers)
                    ]
                )

            self.token_predict_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
            )

        self.pred_light = False

        if self.pred_light:
            self.lg_time_span = time_span

            self.light_hist = time_span // self.shift

            self.light_type = 4

            self.light_dropout = 0

            self.light_embedding = nn.Embedding(5, hidden_dim)

            self.lg_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
            self.lg2lg_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

            self.light_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.light_type)

            if self.pred_agent:
                
                self.lg2a_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.pred_route = False

        self.mixing=True

        # if self.mixing:
        #     self.Q_mixer = QattenMixer(hidden_dim, 4)
        #     self.V_mixer = QattenMixer(hidden_dim, 4)

        self.apply(weight_init)

    def padding(self,tensor,lengths,padding_value=0.0 ):
        padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=padding_value)

        return padded_tensor

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
        n_agent, n_step = agent_token_index.shape
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
                pos_a.new_zeros(agent_token_index.shape[0], 1, 2),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        ) [:,-n_step:] # [n_agent, n_step, 2]
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

    def build_interaction_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            batch_s,  # [n_agent*n_step]
            mask  # [n_agent, n_step]
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        edge_index_a2a = radiusGraphNearest_head(x=pos_s[:, :2],
                                            x_heading=head_s,
                                            r=self.a2a_radius,
                                            batch=batch_s,
                                            loop=False,
                                            max_num_neighbors=self.a2a_neighbor)

        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        return edge_index_a2a, r_a2a

    def build_full_interaction_r_a2a(
            self,
            pos_s,  # [B, N, 2]
            head_s,  # [B, N,]
            head_vector_s,  # [B, N,2]
            pos_s1,
            head_s1,
            mask
    ):
        B, N, _ = pos_s.shape
        B, N1, _ = pos_s1.shape

        mask=~mask

        # Compute pairwise relative positions: [B, N, N, 2]
        rel_pos = pos_s[:, :, None, :] - pos_s1[:, None, :, :]  # [B, N, N, 2]

        rel_pos=rel_pos[mask]

        # Pairwise distance
        dist = torch.norm(rel_pos, dim=-1)  # [B, N, N]

        # Relative heading difference
        rel_head = wrap_angle(head_s[:, :, None] - head_s1[:, None, :])[mask]  # [B, N, N]

        head_vector_s=head_vector_s[:,  :, None,:].expand(-1, -1, N1, -1)[mask]

        # Angle between head_vector of neighbor and vector to target
        ang = angle_between_2d_vectors(
            ctr_vector=head_vector_s,  # [B, N, N, 2]
            nbr_vector=rel_pos  # [B, N, N, 2]
        )  # [B, N, N]

        # Stack into r_a2a feature: [B, N, N, 3]
        r_a2a = torch.stack([dist, ang, rel_head], dim=-1)

        # Apply embedding
        r_a2a = self.r_a2a_emb(r_a2a)  # [B, N, N, d_emb]

        return r_a2a

    def build_map2agent_edge(
            self,
            pos_pl,  # [n_pl, 2]
            orient_pl,  # [n_pl]
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            mask,  # [n_agent, n_step]
            batch_s,  # [n_agent*n_step]
            batch_pl,  # [n_pl*n_step]
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        # edge_index_pl2a = radiusGraphNearest2(x=pos_s[:, :2],
        #                                       y=pos_pl[:, :2],
        #                                       x_heading=head_s,
        #                                       r=self.pl2a_radius,
        #                                       batch_x=batch_s,
        #                                       batch_y=batch_pl,
        #                                       max_num_neighbors=30)

        edge_index_pl2a = radiusGraphNearest_inv(x=pos_s[:, :2],
                                              y=pos_pl[:, :2],
                                              r=self.pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=self.pt2a_neighbor)

        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )

        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        return edge_index_pl2a, r_pl2a

    def temporal_embed(self, feature,rotary_embedding,pos,heading, network, n_step, n_current, hist_len, mask):

        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        time = torch.arange(n_current, n_step + n_current, device=feature.device)[None,:, None]

        #positions=torch.concat([pos,time.repeat_interleave(len(pos),dim=0)],dim=-1)#time.repeat_interleave(len(pos),dim=0)#

        #sinusoidal_pos = general_rope(positions, self.head_dim,heading)
        sinusoidal_pos=rotary_embedding(pos,heading,time)

        if mask is not None:
            causal_mask = causal_mask[None,None] | mask[:,None,None,:]

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature

    def predict_light(self, light_idx, lg_sinusoidal, lengths, n_current=0):
        
        n_step = light_idx.shape[1]
        
        mask_lg=light_idx>2

        # print(torch.any(mask_lg))

        feat_lg = self.light_embedding(light_idx)

        feat_lg = self.temporal_embed(feat_lg, self.lg_t_roformer, n_step, n_current, self.light_hist,mask_lg)

        padded_lg_feature = self.padding(feat_lg, lengths)

        feature_mask = (padded_lg_feature[:,:,0]!=0).any(-1)

        padded_lg_feature = padded_lg_feature.swapaxes(1,2).flatten(0, 1)
        
        padding_light_mask= self.padding(mask_lg[:,-n_step:], lengths,padding_value=True).swapaxes(1,2).flatten(0, 1)

        lg_sinusoidal = lg_sinusoidal.repeat_interleave(n_step,dim=0)

        lg2lg_mask = padding_light_mask[:,None,None] 

        padded_lg_feature = self.lg2lg_roformer(padded_lg_feature, lg2lg_mask, lg_sinusoidal)

        padded_lg_feature=padded_lg_feature.reshape(len(lengths),n_step,-1,padded_lg_feature.shape[-1])

        feat_lg = padded_lg_feature.swapaxes(1,2)[feature_mask]

        next_light_logits = self.light_token_predict_head(feat_lg)

        return padded_lg_feature, next_light_logits

    def predict_agent(self, sampled_idx, mask ,pos_a,head_a,tokenized_agent, map_feature,feat_lg, n_current=0):
        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        # ! get agent token embeddings
        feat_a_token = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        pos_a=pos_a[:,-n_step:]

        mask_a=~mask

        rotary_embedding=map_feature["rotary_embedding"]

        feat_a = self.temporal_embed(feat_a_token,rotary_embedding,pos_a,head_a, self.a_t_roformer, n_step, n_current, self.agent_hist, mask_a)

        if self.use_gnn:
            batch_s = torch.cat(
                [
                    tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                    for t in range(n_step)
                ],
                dim=0,
            )  # [n_agent*n_step]

            batch_pl = torch.cat(
                [
                    map_feature["batch"] + tokenized_agent["num_graphs"] * t
                    for t in range(n_step)
                ],
                dim=0,
            )  # [n_pl*n_step]

            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask,  # [n_agent, n_step]
                batch_s=batch_s,  # [n_agent*n_step]
                batch_pl=batch_pl,  # [n_pl*n_step]
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                batch_s=batch_s,  # [n_agent*n_step]
                mask=mask,  # [n_agent, n_step]
            )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

            # r_pl2a_a2a = self.r_pt2a_emb(continuous_inputs=torch.cat([r_pl2a,r_a2a]), categorical_embs=None)
            #
            # r_pl2a=r_pl2a_a2a[:len(r_pl2a)]
            # r_a2a=r_pl2a_a2a[len(r_pl2a):]

            feat_a = feat_a.transpose(0, 1).flatten(0, 1)
            feat_map = (
                map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
            )

            feat_a = self.pt2a_attn_layers[0](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )


            feat_a = self.a2a_attn_layers[0](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)


        else:
            sinusoidal_a = rotary_embedding(pos_a, head_a)
            lengths_a = torch.bincount(tokenized_agent["batch"]).tolist()
            padded_a_feature = self.padding(feat_a, lengths_a)
            feature_mask = (padded_a_feature[:, :, 0] != 0).any(-1)

            pt_feature = map_feature["padded_pt"]
            map_mask = map_feature["map_mask"]
            map_sinusoidal = map_feature["map_sinusoidal"]
            pt_pos = map_feature["padd_pos"]
            pt_heading = map_feature["padd_heading"]

            agent_sinusoidal = self.padding(sinusoidal_a, lengths_a)
            padd_pos = self.padding(pos_a, lengths_a)
            padding_mask = self.padding(mask_a[:, -n_step:], lengths_a, padding_value=True)
            padd_head = self.padding(head_a, lengths_a)
            padd_head_vector = self.padding(head_vector_a, lengths_a)

            pt2a_mask= map_mask | padding_mask.flatten(1, 2)[:,:,None]

            pt2a_mask = nearest_mask2(padd_pos.flatten(1, 2),pt_pos, self.pt2a_neighbor, self.pl2a_radius, pt2a_mask)

            if feat_lg is not None:
                sinusoidal_lg = tokenized_agent["sinusoidal_lg"]
                sinusoidal_lg=sinusoidal_lg.repeat_interleave(n_step,dim=0)
                feat_lg=feat_lg.flatten(1, 2)
                padded_a_feature = self.lg2a_roformer(padded_a_feature, None, agent_sinusoidal,    feat_lg, sinusoidal_lg )

            # r_a2pt = self.build_full_interaction_r_a2a(padd_pos.flatten(1, 2), padd_head.flatten(1, 2), padd_head_vector.flatten(1, 2),pt_pos, pt_heading,
            #                                           pt2a_mask)
            r_a2pt=None

            padding_agent_mask = padding_mask.swapaxes(1, 2).flatten(0, 1)

            padd_pos = padd_pos.swapaxes(1, 2).flatten(0, 1)

            a2a_mask = padding_agent_mask[:, None] | padding_agent_mask[:, :, None]

            a2a_mask = nearest_mask(padd_pos, self.a2a_neighbor, self.a2a_radius, a2a_mask)

            #padd_head = padd_head.swapaxes(1, 2).flatten(0, 1)
            #padd_head_vector = padd_head_vector.swapaxes(1, 2).flatten(0, 1)

            # r_a2a = self.build_full_interaction_r_a2a(padd_pos, padd_head, padd_head_vector, padd_pos, padd_head,
            #                                           a2a_mask)
            r_a2a=None

            for i in range(len(self.pt2a_roformer)):
                padded_a_feature=padded_a_feature.flatten(1, 2)

                sinusoidal_a=agent_sinusoidal.flatten(1, 2)

                padded_a_feature = self.pt2a_roformer[i](padded_a_feature, pt2a_mask[:,None], sinusoidal_a,    pt_feature, map_sinusoidal,r_a2pt )

                padded_a_feature=padded_a_feature.reshape(len(lengths_a),-1,n_step,self.hidden_dim).swapaxes(1,2).flatten(0,1)

                sinusoidal_a=agent_sinusoidal.swapaxes(1,2).flatten(0, 1)

                padded_a_feature = self.a2a_roformer[i](padded_a_feature, a2a_mask[:,None], sinusoidal_a,pos_embeding=r_a2a)

                padded_a_feature=padded_a_feature.reshape(len(lengths_a),n_step,-1,padded_a_feature.shape[-1]).swapaxes(1,2)

            feat_a = padded_a_feature[feature_mask]

        next_token_logits = self.token_predict_head(feat_a).reshape( n_agent, n_step,-1)

        return next_token_logits,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"]
            lengths_lg = tokenized_agent["lengths_lg"]
            sinusoidal_lg = tokenized_agent["sinusoidal_lg"]

            noised_light_idx = light_idx.clone()

            random_light = torch.randint(low=0, high=self.light_type, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            noised_light_idx[random_mask] = random_light[random_mask]

            feat_lg, next_light_logits = self.predict_light(noised_light_idx,sinusoidal_lg, lengths_lg)
        else:
            feat_lg=None

        if not self.pred_agent:
            tokenized_agent["next_light_logits"] = next_light_logits
            tokenized_agent["feat_lg"] = feat_lg

            return {
                "q_value": next_light_logits[:, 1:]
            }

        sampled_idx=tokenized_agent["sampled_idx"]
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]

        next_token_logits,feat_a= self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature,feat_lg)

        tokenized_agent["next_token_logits"] = next_token_logits

        if self.pred_light:
            next_light_logits = torch.cat(
                [next_light_logits,
                    -torch.inf + torch.zeros([next_light_logits.shape[0], next_light_logits.shape[1],
                                            next_token_logits.shape[2] - next_light_logits.shape[2]],
                                            device=next_light_logits.device)], dim=-1)
    
            next_token_logits = torch.cat([next_token_logits, next_light_logits], dim=0)

        if self.mixing:
            batch=tokenized_agent["batch"]

            q_value=next_token_logits[:, 1:]

            q = q_value[:, :-1]

            action = tokenized_agent["sampled_idx"][:, 2:].reshape(-1)

            Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

            lengths_a = torch.bincount(batch).tolist()

            agent_qs=self.padding(Q,lengths_a).swapaxes(1,2)

            agent_states=self.padding(feat_a[:,1:],lengths_a,padding_value=-1e10).swapaxes(1,2)

            state_mask = self.padding(~mask[:,1:],lengths_a,padding_value=True).swapaxes(1,2)

            states=agent_states.amax(dim=2)

            agent_mask=(~state_mask).all(1)[:,None]

            total_q=(agent_qs*agent_mask).sum(dim=2)

            #total_q=self.Q_mixer(agent_qs.flatten(0,1), states[:,:-1].flatten(0,1),agent_states[:,:-1].flatten(0,1),state_mask[:,:-1].flatten(0,1)).reshape(-1,Q.shape[1])

            V = self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

            agent_value=self.padding(V,lengths_a).swapaxes(1,2)

            total_v=(agent_value*agent_mask).sum(dim=2)

            #total_v=self.V_mixer(agent_value.flatten(0,1),states.flatten(0,1),agent_states.flatten(0,1),state_mask.flatten(0,1)).reshape(-1,V.shape[1])
        else:
            total_q=0
            total_v=0

        return {
            "total_q": total_q,
            "total_v":total_v,
             "q_value": next_token_logits[:, 1:],            # action that goes from [(10->15), ..., (85->90)]
         }

    def autoregressive_light_predict(self,  tokenized_agent, current_len,max_len):
        predicted_tokens = tokenized_agent["light_idx"][:, :current_len].clone()
        lengths_lg = tokenized_agent["lengths_lg"]
        sinusoidal_lg = tokenized_agent["sinusoidal_lg"]

        for t in range(current_len, max_len + current_len):
            if t == current_len:
                if "feat_lg" in tokenized_agent.keys():
                    lg_features = tokenized_agent["feat_lg"][:, :current_len]
                    next_light_logits = tokenized_agent["next_light_logits"][:, :current_len]

                    self.lg_t_roformer.attn.cached_k = self.lg_t_roformer.attn.cached_k[:, :, :current_len]
                    self.lg_t_roformer.attn.cached_v = self.lg_t_roformer.attn.cached_v[:, :, :current_len]
                else:
                    lg_features, next_light_logits = self.predict_light(predicted_tokens, sinusoidal_lg,
                                                                                 lengths_lg)
                self.lg_t_roformer.attn.kv_caching(self.light_hist)
            else:
                feat_lg, next_light_logits = self.predict_light(predicted_tokens[:, -1:],
                                                                 sinusoidal_lg, lengths_lg, t - 1)

                lg_features = torch.cat([lg_features, feat_lg[:, -1:]], dim=1)

            cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

            samples = cat_dist.sample()

            predicted_tokens = torch.cat([predicted_tokens, samples[:, None]], dim=1)

        self.lg_t_roformer.attn.kv_caching(0)

        out_dict={"light_idx": predicted_tokens,
                  }

        return out_dict,lg_features

    def autoregressive_agent(self, tokenized_agent, map_feature,lg_features,current_len,max_len):

        sampled_idx=tokenized_agent["sampled_idx"][:, :current_len].clone()
        mask = tokenized_agent["valid_mask"][:, :current_len].clone()
        pos_a = tokenized_agent["sampled_pos"][:, :current_len].clone()
        head_a = tokenized_agent["sampled_heading"][:, :current_len].clone()
        token_traj_all = tokenized_agent["token_traj_all"]

        if "gt_z_raw" in tokenized_agent.keys():
            n_agent=sampled_idx.shape[0]
            pred_traj_10hz = torch.zeros(
                [n_agent, 0, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, 0], dtype=pos_a.dtype, device=pos_a.device
            )

        logit_list=[]
        for t in range(current_len, max_len + current_len):
            if t == current_len:
                if "next_token_logits" in tokenized_agent.keys():
                    next_token_logits = tokenized_agent["next_token_logits"][:, :current_len]

                    self.a_t_roformer.attn.cached_k = self.a_t_roformer.attn.cached_k[:, :, :current_len]
                    self.a_t_roformer.attn.cached_v = self.a_t_roformer.attn.cached_v[:, :, :current_len]
                else:
                    if lg_features is not None:
                        lg_feat=lg_features[:,:t]
                    else:
                        lg_feat=None

                    next_token_logits,feat_a = self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature,lg_feat)
                logit_list.append(next_token_logits)

                self.a_t_roformer.attn.kv_caching(self.agent_hist)
   
            else:
                if lg_features is not None:
                    lg_feat=lg_features[:,-1:]
                else:
                    lg_feat=None

                next_token_logits,feat_a = self.predict_agent(sampled_idx[:, -1:], mask[:, -self.agent_hist:], pos_a[:, -2:], head_a[:, -1:],tokenized_agent, map_feature,lg_feat,t - 1)
                logit_list.append(next_token_logits[:, -1:])

            cat_dist = Categorical(logits=next_token_logits[:, -1] / self.alpha)

            next_token_idx = cat_dist.sample()

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            range_a = torch.arange(next_token_idx.shape[0])

            next_token_traj_all = token_traj_all[range_a, next_token_idx]

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, -1],  # [n_agent, 2]
                head_now=head_a[:, -1],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if "gt_z_raw" in tokenized_agent.keys():

                pred_traj=token_traj_global[:, 1:].mean( 2 )
                pred_traj_10hz=torch.cat([pred_traj_10hz, pred_traj],dim=1)
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head= torch.arctan2( diff_xy[:, :, 1], diff_xy[:, :, 0]   )
                pred_head_10hz=torch.cat([pred_head_10hz,pred_head],dim=1)

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

            if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
                mask =torch.cat([mask,torch.ones_like(head_a_next).to(torch.bool).unsqueeze(1)], dim=1)
            else:
                mask=torch.cat([mask,tokenized_agent["valid_mask"][:,t:t+1]], dim=1)

        self.a_t_roformer.attn.kv_caching(0)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "sampled_idx": sampled_idx,  # [n_agent, 18]
        }


        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission

            if 'centering_pos' in tokenized_agent.keys():
                batch=tokenized_agent["batch"]
                centering_heading=tokenized_agent['centering_heading'][batch]
                centering_pos=tokenized_agent["centering_pos"][batch]


                cos_a = torch.cos(-centering_heading)[:,None]
                sin_a = torch.sin(-centering_heading)[:,None]

                x, y = pred_traj_10hz[..., 0], pred_traj_10hz[..., 1]
                x_rot = cos_a * x + sin_a * y
                y_rot = -sin_a * x + cos_a * y

                pred_traj_10hz=torch.stack([x_rot, y_rot], dim=-1)+centering_pos[:,None]

                pred_head_10hz=pred_head_10hz+ centering_heading[:,None]


            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
            out_dict["gt_pos_raw"] = tokenized_agent["gt_pos_raw"]  # [n_agent, 18, 2]
            out_dict["gt_head_raw"] = tokenized_agent["gt_head_raw"]  # [n_agent, 18]
            out_dict["gt_valid_raw"] = tokenized_agent["gt_valid_raw"]  # [n_agent, 18]

        # next_token_logits=torch.cat(logit_list, dim=1)
        #
        # next_token_logits1 = self.predict_agent(sampled_idx[:,:-1], mask[:,:-1], pos_a[:,:-1], head_a[:,:-1], tokenized_agent, map_feature, None)
        #
        # print((next_token_logits1-next_token_logits).max())

        return out_dict

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
            out_dict,lg_features = self.autoregressive_light_predict(tokenized_agent,step_current_2hz,
                                                                               n_step_future_2hz)
        else:
            lg_features=None

        if not self.pred_agent:
            return out_dict
        
        out_dict =self.autoregressive_agent(tokenized_agent, map_feature,lg_features,step_current_2hz, n_step_future_2hz)

        return out_dict
