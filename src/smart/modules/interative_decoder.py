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

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.modules.edge_encoder import EdgeEncoder
import time
from torch_scatter import scatter_max,scatter_mean

class InterativeDecoder(nn.Module):
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
            output_gmm,
            pred_last_res,
            pred_all_res,
            discriminator=False,
            value_network=False
    ) -> None:
        super(InterativeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob

        self.head_dim = hidden_dim // num_heads

        self.agent_hist = self.time_span // self.shift

        self.edge_encoder = EdgeEncoder(hidden_dim, num_freq_bands,share=discriminator)

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
        self.output_gmm = output_gmm

        self.pred_last_res = pred_last_res
        self.pred_all_res = pred_all_res
        self.n_token_agent=n_token_agent

        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )

        if self.pred_last_res or self.pred_all_res:
            self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3 * 5)

        self.start_step=10//self.shift-1

        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.token_processor=token_processor

        self.state_action = False
        self.reward_shaping = False
        self.use_bottleneck = False

        self.discriminator=discriminator
        self.value_network=value_network

        self.filter_ratio=0

        if self.discriminator:
            self.centric=False

            if  self.reward_shaping:
                self.reward_net = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

            self.use_bottleneck=False

            if self.use_bottleneck:
                z_dim=self.hidden_dim//2
                # self.a2pl_linear=nn.Sequential(nn.ReLU(),nn.Linear(z_dim, self.hidden_dim))
                self.a2a_linear=nn.Sequential(nn.ReLU(),nn.Linear(z_dim, self.hidden_dim))

    def forward(self,all_features,map_feature,train_mask ):
        feat_a_t,pos_a, head_a, head_vector_a,mask_a, batch_s_repeat,batch_s,agent_token_emb=all_features#,vis_mask,agent_token_emb, sampled_idx,batch_pl

        n_agent = mask_a.shape[0]
        n_step=mask_a.shape[1]

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
            pos_pl=pos_pl,  # [n_pl, 2]
            orient_pl=orient_pl,  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask_a,  # [n_agent, n_step]
            batch_s=batch_s_repeat,  # [n_agent,n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
            pl2a_radius=self.pl2a_radius,
            max_num_neighbors=self.pt2a_neighbor,
            train_mask=train_mask,
            num_layers=self.num_layers
        )

        feat_a,pos_s, head_s, head_vector_s,mask_s, _,batch_s=[feat.transpose(0, 1).flatten(0, 1) for feat in all_features[:-1] ]

        #batch_s_repeat=batch_s_repeat.reshape(n_step,n_agent).transpose(0, 1).flatten(0, 1)

        if train_mask is not None:
            train_repeat_mask=train_mask[:,None].repeat(1,n_step).transpose(0, 1).flatten(0, 1)
        else:
            train_repeat_mask=None

        edge_index_a2a, r_a2a = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,  # [n_agent, n_step, 2]
            head_s=head_s,  # [n_agent, n_step]
            head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask_s,  # [n_agent, n_step]
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            proposal=None,
            vis_mask=None,
            value=False,
            train_mask = train_repeat_mask,
            num_layers = self.num_layers

        #shape=tokenized_agent["shape"]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        if self.use_bottleneck:
            mu,sigma=r_a2a.chunk(2,dim=-1)

            if self.training:
                std = torch.exp(sigma / 2)
                eps = torch.randn_like(std)
                z=mu+eps*std
            else:
                z=mu

            r_a2a=self.a2a_linear(z)

        #n_a=len(r_a2a)
        #n_pt=len(r_pl2a)

       # a2a_list=[]

        for layer_i in range(self.num_layers):
            if self.num_layers>1 and layer_i == self.num_layers - 1 and train_mask is not None:
                end_mask=train_repeat_mask[edge_index_a2a[1]]
                edge_index_a2a = edge_index_a2a[:, end_mask]
                r_a2a=r_a2a[end_mask]

                end_pt_mask=train_repeat_mask[edge_index_pl2a[1]]
                edge_index_pl2a = edge_index_pl2a[:, end_pt_mask]
                r_pl2a=r_pl2a[end_pt_mask]

            feat_a,a2a_attn = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

            if  train_mask is not None and self.num_layers==1:
                feat_a = feat_a.view(-1,n_agent,self.hidden_dim)[:16,train_mask]
                n_agent = feat_a.shape[1]
                feat_a=feat_a.flatten(0,1)

            feat_a,pt_attn  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

        #     if layer_i<self.num_layers-1 and self.filter_ratio>0:
        #         a2a_mask=a2a_attn>self.filter_ratio
        #         r_a2a=r_a2a[a2a_mask]
        #         edge_index_a2a=edge_index_a2a[:,a2a_mask]
        #
        #         pt_mask=pt_attn>self.filter_ratio
        #
        #         r_pl2a=r_pl2a[pt_mask]
        #         edge_index_pl2a=edge_index_pl2a[:,pt_mask]
        #
        #     a2a_list.append(a2a_attn)
        #
        # a2a_feature=torch.cat(a2a_list,dim=-1)

        # if self.num_layers>1:
        #     self.a_ratio=len(r_a2a)/n_a
        #     self.pt_ratio=len(r_pl2a)/n_pt

        feat_a_all = feat_a.view( -1,  n_agent,self.hidden_dim).transpose(0, 1)
        proposal=None

        if self.num_layers>1 and train_mask is not None:
            feat_a=feat_a_all[train_mask]
        else:
            feat_a=feat_a_all

        if self.discriminator and self.centric:
            index=batch_s_repeat[train_mask]
            feat_a, argmax = scatter_max(feat_a, index, dim=0)  # out: [B,T,C]

        if self.n_token_agent>1 and self.n_token_agent<2048:
            # feat_a=torch.mean(feat_a, dim=1,keepdim=True)
            index=batch_s_repeat[train_mask]
            #feat_a, argmax = scatter_mean(feat_a, index, dim=0)  # out: [B,T,C]
            feat_a = scatter_mean(feat_a, index, dim=0)  # out: [B,T,C]

        if self.pred_last_res:
            if self.training:
                proposal_feature = feat_a.detach()#[:, :-1]
            else:
                proposal_feature = feat_a#[:, -1:]

            proposal = self.traj_head(proposal_feature)  #
            proposal = proposal.reshape(proposal.shape[0], proposal.shape[1], 1, -1, 3)

        if self.pred_all_res and self.training:
            next_token_idx = sampled_idx[:, 1 + self.start_step:]

            token_local_traj = self.token_processor.token_local_traj

            if train_mask is not None:
                token_local_traj=token_local_traj[train_mask]
                next_token_idx=next_token_idx[train_mask]
                agent_token_emb=agent_token_emb[train_mask]

            proposal_feature = feat_a[:, :-1] + agent_token_emb[:, 1:]
            proposal = self.traj_head(proposal_feature)  #
            proposal = proposal.reshape(proposal.shape[0], proposal.shape[1], 1, -1, 3)

            next_token_traj_all = token_local_traj[torch.arange(n_agent)[:, None], next_token_idx]

            if self.token_processor.max_diff is not None:

                proposal_max_diff = self.token_processor.token_diff[torch.arange(n_agent)[:, None], next_token_idx]

                proposal = torch.tanh(proposal) * proposal_max_diff[:, :, None]

            proposal = proposal + next_token_traj_all[:, :, None]

        if self.discriminator:
            if self.state_action:
                feat_a = feat_a + agent_token_emb[train_mask]
            # else:
            #     feat_a = feat_a[:, 1:]

        next_token_logits = self.token_predict_head(feat_a)

        if self.discriminator and self.reward_shaping:
            r=self.reward_net(feat_a[:, 1:] )#+ agent_token_emb
            v_s=next_token_logits[:, :-1]
            v_next=next_token_logits[:,1 :]
            done=torch.ones_like(v_s)
            done[:,-1]=0
            next_token_logits = r + (0.99*v_next - v_s)*done

        # next_token_logits=torch.zeros([n_agent,n_step,token_logits.shape[-1]],device=feat_a.device)
        #
        # next_token_logits[mask_a]=token_logits

        if self.use_bottleneck:
            next_token_logits=(next_token_logits,mu,sigma)

        return next_token_logits,feat_a_all,proposal,r_a2a,r_pl2a

        # if self.output_gmm:
        #     next_logits = self.gmm_logits_head(feat_a)
        #     next_poses = self.gmm_pose_head(feat_a).view(*next_logits.shape, 3)
        #     if self.cov_learnable:
        #         next_cov =self.gmm_cov_head(feat_a).view(*next_logits.shape, -1).exp()
        #     else:
        #         next_cov = torch.zeros_like(next_poses)+0.1
        #     next_token_logits=torch.cat([next_logits[...,None],next_poses,next_cov],dim=-1)
        # else:
