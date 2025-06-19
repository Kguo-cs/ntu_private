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
from .build_edge import radiusGraphNearest2,nearest_mask,generate_limited_causal_mask,nearest_mask2, \
    radiusGraphNearest_head,radiusGraphNearest_inv
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.utils.rollout import cal_polygon_contour
from src.smart.loss.gmm_dist import  GMM_Dist
from src.smart.loss.iq_loss import padding
from src.smart.modules.light_encoder import LightEncoder


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
            token_processor,
            alpha,
            output_gmm
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

        self.alpha = alpha

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
            input_dim_r_pt2a = 3
            input_dim_r_a2a = 3

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

            self.output_gmm=output_gmm
            self.n_token_agent = n_token_agent

            if self.output_gmm:
                k_ego_gmm=1
                self.cov_gmm=0.1 #[1.0, 0.1]
                self.cov_learnable=True
                self.use_GT=True

                self.gmm_logits_head = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=k_ego_gmm
                )
                self.gmm_pose_head = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=k_ego_gmm * 3
                )
                self.output_dim=3

                if self.cov_learnable:
                    self.gmm_cov_head = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=k_ego_gmm * self.output_dim
                    )
                self.pred_res = False

                # self.cholesky_head = nn.Linear(
                #     hidden_dim, k_ego_gmm * (self.output_dim * (self.output_dim + 1) // 2)
                # )
            else:

                if n_token_agent>1:
                    self.token_predict_head = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )
                    self.pred_res = False

                    if self.pred_res:
                        self.res_head = MLPLayer(hidden_dim*2,hidden_dim, output_dim=3)

                else:
                    self.token_predict_head = MLPLayer(
                        input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

        self.pred_light = True

        if self.pred_light:
            self.light_type = 4

            self.light_encoder = LightEncoder(hidden_dim,time_span,num_heads,self.light_type,self.shift)

        self.mixing=False
        self.rotary_embedding = RoFormerSinusoidalPositionalEmbedding(hidden_dim=hidden_dim, num_heads=num_heads)
        self.token_processor= token_processor

        self.apply(weight_init)
        # if self.mixing:
        #     self.Q_mixer = QattenMixer(hidden_dim, 4)
            #self.V_mixer = QattenMixer(hidden_dim, 4)

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
            return feat_a # [n_agent, n_step, hidden_dim]

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
            pl2a_radius=40
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radiusGraphNearest2(x=pos_s[:, :2],
                                              y=pos_pl[:, :2],
                                              x_heading=head_s,
                                              r=pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=20)

        # edge_index_pl2a = radiusGraphNearest_inv(x=pos_s[:, :2],
        #                                       y=pos_pl[:, :2],
        #                                       r=self.pl2a_radius,
        #                                       batch_x=batch_s,
        #                                       batch_y=batch_pl,
        #                                       max_num_neighbors=self.pt2a_neighbor)

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

    def temporal_embed(self, feature,pos,heading, network, n_step, n_current, hist_len, mask):

        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        time = torch.arange(n_current, n_step + n_current, device=feature.device)[None,:, None]

        # pos_time =torch.concat([pos,time.repeat_interleave(len(pos),dim=0)],dim=-1)#time.repeat_interleave(len(pos),dim=0)#
        #
        # sinusoidal_pos = general_rope(pos_time, self.head_dim,heading)
        sinusoidal_pos=self.rotary_embedding(pos,heading,time)

        if mask is not None:
            causal_mask = causal_mask[None,None] | mask[:,None,None,:]

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature

    def predict_agent(self, sampled_idx, mask ,pos_a,head_a,tokenized_agent, map_feature,light_idx,lg_sinusoidal=None, n_current=0):
        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        # ! get agent token embeddings
        feat_a_token = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            trajectory_token_veh=self.token_processor.trajectory_token_veh,
            trajectory_token_ped=self.token_processor.trajectory_token_ped,
            trajectory_token_cyc=self.token_processor.trajectory_token_cyc,
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        pos_a=pos_a[:,-n_step:]

        if self.pred_light:
            feat_lg = self.light_encoder.light_embedding(light_idx)
            feat_a_token=torch.cat([feat_a_token,feat_lg],dim=0)

            pos_lg = tokenized_agent["pos_lg"]
            orient_lg = tokenized_agent["orient_lg"]

            pos_a=torch.cat([pos_a,pos_lg[:,:n_step]],dim=0)

            head_a=torch.cat([head_a,orient_lg[:,:n_step]],dim=0)
            #head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            #n_agent, n_step = head_a.shape

        mask_a=~mask

        feat_a = self.temporal_embed(feat_a_token,pos_a,head_a, self.a_t_roformer, n_step, n_current, self.agent_hist, mask_a)

        if self.pred_light:
            lengths=tokenized_agent["lengths_lg"]
            mask_lg=mask_a[len(sampled_idx):]
            feat_lg= feat_a[len(sampled_idx):]


            pos_a=pos_a[:len(sampled_idx)]
            head_a=head_a[:len(sampled_idx)]
            feat_a=feat_a[:len(sampled_idx)]

            feat_lg,next_light_logits=self.light_encoder(feat_lg, lg_sinusoidal, lengths ,mask_lg ,n_step)

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


            feat_a = feat_a.transpose(0, 1).flatten(0, 1)
            feat_map = (
                map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
            )


            feat_a = self.pt2a_attn_layers[0](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )

            if feat_lg is not None:#  # [B, L, D]
                pos_lg = tokenized_agent["pos_lg"][:,0]
                head_lg = tokenized_agent["orient_lg"][:,0]
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
                feat_a = self.light_encoder.lg2a_attn_layers[0](
                    (feat_lg.swapaxes(0, 1).flatten(0, 1), feat_a), r_lg2a, edge_index_lg2a
                )

                # feat_a=self.light_encoder.light2agent(tokenized_agent,feat_a,feat_lg, n_step,pos_a,head_a,head_vector_a,mask,batch_s)

            feat_a = self.a2a_attn_layers[0](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
        else:
            sinusoidal_a = self.rotary_embedding(pos_a, head_a)
            lengths_a = torch.bincount(tokenized_agent["batch"]).tolist()
            padded_a_feature = padding(feat_a, lengths_a)
            feature_mask = (padded_a_feature[:, :, 0] != 0).any(-1)

            pt_feature = map_feature["padded_pt"]
            map_mask = map_feature["map_mask"]
            map_sinusoidal = map_feature["map_sinusoidal"]
            pt_pos = map_feature["padd_pos"]
            pt_heading = map_feature["padd_heading"]

            agent_sinusoidal = padding(sinusoidal_a, lengths_a)
            padd_pos = padding(pos_a, lengths_a)
            padding_mask = padding(mask_a[:, -n_step:], lengths_a, padding_value=True)
            padd_head = padding(head_a, lengths_a)
            padd_head_vector = padding(head_vector_a, lengths_a)

            pt2a_mask= map_mask | padding_mask.flatten(1, 2)[:,:,None]

            pt2a_mask = nearest_mask2(padd_pos.flatten(1, 2),pt_pos, self.pt2a_neighbor, self.pl2a_radius, pt2a_mask)

            if feat_lg is not None:
                sinusoidal_lg = tokenized_agent["sinusoidal_lg"]
                sinusoidal_lg = sinusoidal_lg.repeat_interleave(n_step,dim=0)
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

        if self.output_gmm:
            next_logits = self.gmm_logits_head(feat_a)
            next_poses = self.gmm_pose_head(feat_a).view(*next_logits.shape, 3)
            if self.cov_learnable:
                next_cov =self.gmm_cov_head(feat_a).view(*next_logits.shape, -1).exp()
            else:
                next_cov = torch.zeros_like(next_poses)+0.1
            #raw_L = self.cholesky_head(feat_a).view( *next_logits.shape, -1 )  # [B, M, 6] for 3x3 lower triangle

            #
            # tril_indices = torch.tril_indices(self.output_dim, self.output_dim, device=raw_L.device)
            # L = torch.zeros(*raw_L.shape[:-1], self.output_dim, self.output_dim, device=raw_L.device)
            # L[..., tril_indices[0], tril_indices[1]] = raw_L
            # diag_idx = torch.arange(self.output_dim, device=raw_L.device)
            # L[..., diag_idx, diag_idx] = torch.exp(L[..., diag_idx, diag_idx])
            #
            # next_cov = L @ L.transpose(-1, -2)  # [B, T, M, 3, 3]
            # eye = torch.eye(self.output_dim, device=next_cov.device).expand(next_cov.shape[:-2] + (self.output_dim, self.output_dim))
            # next_cov = next_cov + eye * 1e-3
            next_token_logits=torch.cat([next_logits[...,None],next_poses,next_cov],dim=-1)
            # next_token_logits={
            # "next_logits": next_logits,  # [n_batch, 16, n_k_ego_gmm]
            # "next_poses": next_poses,  # [n_batch, 16, n_k_ego_gmm, 3]
            # "next_cov": next_cov,  # [2], one for pos, one for heading.
            # }
        else:
            if self.n_token_agent>1:
                next_token_logits = self.token_predict_head(feat_a[:sampled_idx.shape[0]])#.reshape( -1, n_step,self.n_token_agent)

                # next_light_logits = self.light_token_predict_head(feat_a[sampled_idx.shape[0]:])#.reshape( -1, n_step,self.light_type)

                if self.pred_res and self.training:

                    res_traj=self.res_head(torch.cat([feat_a[:,:-1],feat_a_token[:,1:]],dim=-1))

                    res_traj=torch.cat([res_traj,res_traj[:,:1]],dim=1)

                    next_token_logits=torch.cat([next_token_logits,res_traj],dim=-1)
            else:
                next_token_logits=feat_a

        return next_token_logits,next_light_logits,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"]
            lengths_lg = tokenized_agent["lengths_lg"]
            pos_lg = tokenized_agent["pos_lg"]
            orient_lg = tokenized_agent["orient_lg"]

            lg_sinusoidal = self.rotary_embedding(pos_lg[:,0], orient_lg[:,0])
            lg_sinusoidal = padding(lg_sinusoidal, lengths_lg)

            noised_light_idx = light_idx.clone()

            random_light = torch.randint(low=0, high=self.light_type, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            noised_light_idx[random_mask] = random_light[random_mask]
        #
        #     # feat_lg, next_light_logits = self.predict_light(noised_light_idx,lg_sinusoidal, lengths_lg)
        # else:
        #     feat_lg=None

        # if not self.pred_agent:
        #     tokenized_agent["next_light_logits"] = next_light_logits
        #     tokenized_agent["feat_lg"] = feat_lg
        #     return {
        #         "q_value": next_light_logits[:, 1:]
        #     }

        sampled_idx=tokenized_agent["sampled_idx"].long()
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]

        #if self.pred_light:
        #     sampled_idx=torch.cat([sampled_idx,noised_light_idx],dim=0)
        #
        #     pos_a=torch.cat([pos_a,pos_lg],dim=0)
        #
        #     head_a=torch.cat([head_a,orient_lg],dim=0)
        #
            # light_mask=light_idx<self.light_type

            # mask=torch.cat([mask,light_mask],dim=0)

        next_token_logits,next_light_logits,feat_a= self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature,noised_light_idx,lg_sinusoidal)#,feat_lg,lg_sinusoidal
#
        if self.n_token_agent>1:
            tokenized_agent["next_token_logits"] = next_token_logits
            tokenized_agent["next_light_logits"] = next_light_logits

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

            agent_qs=padding(Q,lengths_a).swapaxes(1,2)

            agent_states=padding(feat_a[:,1:],lengths_a,padding_value=-1e10).swapaxes(1,2)

            state_mask = padding(~mask[:,1:],lengths_a,padding_value=True).swapaxes(1,2)

            states=agent_states.amax(dim=2)

            agent_mask=(~state_mask).all(1)[:,None]

            state_mask=~agent_mask.repeat(1,state_mask.shape[1],1)

            total_q=(agent_qs*agent_mask).sum(dim=2)

            #total_q=self.Q_mixer(agent_qs.flatten(0,1), states[:,:-1].flatten(0,1),agent_states[:,:-1].flatten(0,1),state_mask[:,:-1].flatten(0,1)).reshape(-1,Q.shape[1])

            V = self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

            agent_value=padding(V,lengths_a).swapaxes(1,2)

            total_v=(agent_value*agent_mask).sum(dim=2)

            #total_v=self.Q_mixer(agent_value.flatten(0,1),states.flatten(0,1),agent_states.flatten(0,1),state_mask.flatten(0,1)).reshape(-1,V.shape[1])
        else:
            total_q=0
            total_v=0

        return {
            "total_q": total_q,
            "total_v":total_v,
            "feat_a": feat_a[:,1:],
             "q_value": next_token_logits[:, 1:],            # action that goes from [(10->15), ..., (85->90)]
         }

    def autoregressive_agent(self, tokenized_agent, map_feature,current_step,max_step):

        sampled_idx=tokenized_agent["sampled_idx"][:, :current_step].clone().long()
        mask = tokenized_agent["valid_mask"][:, :current_step].clone()
        pos_a = tokenized_agent["sampled_pos"][:, :current_step].clone()
        head_a = tokenized_agent["sampled_heading"][:, :current_step].clone()
        token_traj_all = tokenized_agent["token_traj_all"]
        n_agent = sampled_idx.shape[0]

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"][:, :current_step].clone()
            lengths_lg = tokenized_agent["lengths_lg"]
            pos_lg = tokenized_agent["pos_lg"]
            orient_lg = tokenized_agent["orient_lg"]

            lg_sinusoidal = self.rotary_embedding(pos_lg[:,0], orient_lg[:,0])
            lg_sinusoidal = padding(lg_sinusoidal, lengths_lg)

            # sampled_idx=torch.cat([sampled_idx,light_idx],dim=1)

        else:
            lg_sinusoidal = None
            light_idx = None


        if "gt_z_raw" in tokenized_agent.keys():
            pred_traj_10hz = torch.zeros(
                [n_agent, 0, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, 0], dtype=pos_a.dtype, device=pos_a.device
            )

        # logit_list=[]
        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys():
                    next_token_logits = tokenized_agent["next_token_logits"][:, :current_step]
                    next_light_logits = tokenized_agent["next_light_logits"][:, :current_step]

                    self.a_t_roformer.attn.cached_k = self.a_t_roformer.attn.cached_k[:, :, :current_step]
                    self.a_t_roformer.attn.cached_v = self.a_t_roformer.attn.cached_v[:, :, :current_step]

                else:
                    # if lg_features is not None:
                    #     lg_feat=lg_features[:,:t]
                    # else:
                    #     lg_feat=None

                    next_token_logits,next_light_logits,feat_a = self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature,light_idx,lg_sinusoidal)#,lg_feat

                #logit_list.append(next_token_logits)

                self.a_t_roformer.attn.kv_caching(self.agent_hist)
   
            else:
                # if lg_features is not None:
                #     lg_feat=lg_features[:,-1:]
                # else:
                #     lg_feat=None

                next_token_logits,next_light_logits,feat_a = self.predict_agent(sampled_idx[:, -1:], mask[:, -self.agent_hist:], pos_a[:, -2:], head_a[:, -1:],tokenized_agent, map_feature,light_idx[:,-1:],lg_sinusoidal,t - 1)
                #logit_list.append(next_token_logits[:, -1:])

            if self.output_gmm:
                #next_token_traj_all = token_traj_all[torch.arange(n_agent), sampled_idx[:,-1]]
                token_agent_shape = tokenized_agent["token_agent_shape"]  # [n_token, 2]

                gmm= GMM_Dist(next_token_logits)

                sample = gmm.sample()[:,-1]  # [n_batch, 4]

                if self.output_dim==4:
                    head=torch.arctan2(sample[..., -1], sample[..., -2])
                else:
                    head=sample[..., 2]

                contour_local = cal_polygon_contour(
                    sample[..., :2],  # [n_batch, 2]
                    head,# [n_batch]
                    token_agent_shape,  # [n_batch, 2]
                )  # [n_batch, 4, 2] in local coord
                token_traj=token_traj_all[:,:,-1]
                dist = torch.norm(contour_local.unsqueeze(1) - token_traj, dim=-1).mean(  -1  )  # [n_batch, n_token]

                next_token_idx = dist.argmin(-1)

                next_token_traj_all = token_traj_all[torch.arange(n_agent), next_token_idx]

                countour_start = next_token_traj_all[:, 0]  # [n_batch, 4, 2]
                n_step = next_token_traj_all.shape[1]
                diff = (contour_local - countour_start) / (n_step - 1)
                ego_token_interp = [countour_start + diff * i for i in range(n_step)]
                # [n_batch, 6, 4, 2]
                next_token_traj_all  = torch.stack(ego_token_interp, dim=1)

            else:
                cat_dist = Categorical(logits=next_token_logits[:, -1,:self.n_token_agent] / self.alpha)

                next_token_idx = cat_dist.sample()

                range_a = torch.arange(next_token_idx.shape[0])

                next_token_traj_all = token_traj_all[range_a, next_token_idx]

                cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

                next_light_idx=   cat_dist.sample()

            light_idx = torch.cat([light_idx, next_light_idx[:, None]], dim=1)

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

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

            if self.pred_res:
                head_vector_a = torch.stack([head_a[:, -1:].cos(), head_a[:, -1:].sin()], dim=-1)

                feat_a_token=self.agent_token_embedding(
                    agent_token_index=sampled_idx[:, -1:],  # [n_ag, n_step]
                    trajectory_token_veh=self.token_processor.trajectory_token_veh,
                    trajectory_token_ped=self.token_processor.trajectory_token_ped,
                    trajectory_token_cyc=self.token_processor.trajectory_token_cyc,
                    pos_a=pos_a[:, -2:],  # [n_agent, n_step, 2]
                    head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                    agent_type=tokenized_agent["type"],  # [n_agent]
                    agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
                )

                res_traj = self.res_head(torch.cat([feat_a[:,-1], feat_a_token[:,-1]], dim=-1)).reshape(n_agent,-1,3)#.reshape(-1,4,2)

                pos_global, head_global = transform_to_global(
                        pos_local=res_traj[:,:,:2],
                        head_local=res_traj[:,:,2],
                        pos_now=pos_a[:, -1],  # [n_agent, 2]
                        head_now=head_a[:, -1],  # [n_agent]
                    )
                # res_traj_global = transform_to_global(
                #     pos_local=res_traj,  # [n_agent, 6*4, 2]
                #     head_local=None,
                #     pos_now=pos_a[:, -1],  # [n_agent, 2]
                #     head_now=head_a[:, -1],  # [n_agent]
                # )[0]
                #
                # pos_a_next = res_traj_global.mean(dim=1)
                # diff_xy_next = res_traj_global[:, 0] - res_traj_global[:, 3]
                # head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
                # pos_a[:,-1]=pos_a_next#pos_global[:,-1]
                # head_a[:,-1]=head_a_next#head_global[:,-1]

                # head_global=wrap_angle(head_global)
                #
                pos_a[:,-1]=pos_global[:,-1]
                head_a[:,-1]=head_global[:,-1]

                #print(1)

                # pred_traj_10hz[:,-5:]=pos_global
                # pred_head_10hz[:,-5:]=head_global

           # if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            mask =torch.cat([mask,torch.ones_like(mask[:,-1:]).to(torch.bool)], dim=1)
            # else:
            #     mask=torch.cat([mask,tokenized_agent["valid_mask"][:,t:t+1]], dim=1)

        self.a_t_roformer.attn.kv_caching(0)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "batch":tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "sampled_idx": sampled_idx,  # [n_agent, 18]
            "light_idx": light_idx
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
            sampling_scheme=None,
            step_current_10hz=None,
            n_step_future_10hz=None,
    ) -> Dict[str, torch.Tensor]:
        if n_step_future_10hz is None:
            n_step_future_10hz = self.num_future_steps  # 80
        if step_current_10hz is None:
            step_current_10hz = self.num_historical_steps - 1  # 10

        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_2hz = step_current_10hz // self.shift  # 2

        # if self.pred_light:
        #     out_dict,lg_features = self.autoregressive_light_predict(tokenized_agent,step_current_2hz,
        #                                                                        n_step_future_2hz)
        # else:
        #     lg_features=None
        #     out_dict={}
        #
        # if not self.pred_agent:
        #     return out_dict
        
        out_dict=self.autoregressive_agent(tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz)

        return out_dict
