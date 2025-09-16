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

import numpy as np
import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention,feat_list_mask_each_agent_cached
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.modules.edge_encoder import EdgeEncoder
import time
from torch_scatter import scatter_max,scatter_mean,scatter_sum,scatter_min
from src.smart.modules.diffusion_discriminator import Discriminator



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
            value_network=False,
            use_roformer=True
    ) -> None:
        super(InterativeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob
        self.output_gmm = output_gmm

        self.head_dim = hidden_dim // num_heads

        self.agent_hist = self.time_span // self.shift

        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        share=discriminator,
                                        hist_drop_prob=hist_drop_prob,
                                        time_span=time_span,
                                        use_roformer=use_roformer,
                                        discriminator=discriminator)

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
        self.discriminator = discriminator

        self.state_action = False
        self.reward_shaping = False
        self.diff_dicriminator = False

        self.use_ego_loop=False
        self.use_counterfactual=False
        self.use_edge_feature=True

        if discriminator and self.use_counterfactual:
            self.a2a_attn_layers = nn.ModuleList(
                [
                    CacheAttention(
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
        elif (discriminator and self.use_edge_feature):
            self.a2a_attn_layers = None
        else:
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

        self.pred_last_res = pred_last_res
        self.pred_all_res = pred_all_res
        self.n_token_agent=n_token_agent

        if self.pred_last_res or self.pred_all_res:
            if self.output_gmm:
                self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3*2*1) #mean and std
            else:
               self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3 * 5)

        self.start_step=10//self.shift-1

        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.token_processor=token_processor


        self.filter_ratio=0
        if self.discriminator and self.diff_dicriminator:
            self.token_predict_head = Discriminator(hidden_dim, hidden_dim, False, num_units=128)
        else:
            if self.use_edge_feature and discriminator:

                self.use_ego_loop=False

                if not self.use_ego_loop:
                    self.ego_head= MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

                self.learn_weight=True

                if self.learn_weight:
                    self.token_predict_head= MLPLayer(
                        input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=2
                    )
                else:
                    self.token_predict_head = MLPLayer(
                        input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

                # self.token_predict_head = nn.Sequential(
                #     nn.Linear(hidden_dim*3, hidden_dim*2),
                #     nn.LayerNorm(hidden_dim*2),
                #     nn.ReLU(inplace=True),
                #     MLPLayer(
                #     input_dim=hidden_dim*2, hidden_dim=hidden_dim, output_dim=n_token_agent
                # ))
            else:
                self.token_predict_head = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

        if self.discriminator:
            self.centric=False
            if  self.reward_shaping:
                self.reward_net = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

    def forward(self,all_features,map_feature,train_mask ):
        feat_a_t,feat_a_token,pos_a, head_a, head_vector_a,mask_a, batch_s_repeat,batch_s,agent_token_emb,sampled_idx=all_features

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
            use_counterfactual=self.use_counterfactual
        )

        feat_a,feat_a_token,pos_s, head_s, head_vector_s,mask_s, _,batch_s=[feat.transpose(0, 1).flatten(0, 1) for feat in all_features[:-2] ]

        if train_mask is not None:
            train_repeat_mask=train_mask[:,None].repeat(1,n_step).transpose(0, 1).flatten(0, 1)
        else:
            train_repeat_mask=None

        edge_index_a2a, r_a2a,dist = self.edge_encoder.build_interaction_edge(
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
            loop= self.use_ego_loop
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        for layer_i in range(self.num_layers):

            if (self.use_edge_feature and self.discriminator):
                start_index=edge_index_a2a[0]
                end_index=edge_index_a2a[1]

                start_edge_feature=feat_a[start_index]
                end_edge_feature=feat_a[end_index]

                if  train_mask is not None and self.num_layers==1:
                    feat_a = feat_a.view(-1,n_agent,self.hidden_dim)[:,train_mask]
                    n_agent = feat_a.shape[1]
                    feat_a=feat_a.flatten(0,1)

                feat_a_pt, pt_attn = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

                if not self.use_ego_loop:
                    ego_feat = feat_a_pt.view(n_step,-1,self.hidden_dim).flatten(0,1)

                    ego_logits=self.ego_head(ego_feat)[:,None]

                feat_a=torch.cat([start_edge_feature,r_a2a,end_edge_feature],dim=-1)[:,None]
            elif (self.discriminator and self.use_counterfactual):
                if  train_mask is not None:
                    connected_agent=torch.unique(edge_index_a2a[0])
                    in_mask=torch.isin(edge_index_pl2a[1], connected_agent)
                    r_pl2a=r_pl2a[in_mask]
                    edge_index_pl2a = edge_index_pl2a[:, in_mask]

                feat_a_pt, pt_attn = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

                feat_a, a2a_attn = self.a2a_attn_layers[layer_i](feat_a_pt, r_a2a, edge_index_a2a)

                feat_a = feat_a.view(-1, n_agent, self.hidden_dim)[:, train_mask]

                n_agent = feat_a.shape[1]

                # valid_agent=torch.where(train_mask)[0]
                # Efficient ablation for all agents in valid_agent (1 pass, O(E))
                # masked_outputs = ablation_outputs_all_valid(self.a2a_attn_layers[layer_i],
                #                                             x_input=feat_a_pt,  # the x fed to this layer
                #                                             valid_agent=valid_agent)  # LongTensor of node ids
                # feat_list1 = get_feat_list_from_masked(masked_outputs, valid_agent, n_step, n_agent)
                # 1) Vectorized ablation + packing (no loops over agents, no extra forwards)
                # feat_list1 = feat_list_mask_each_agent_cached(
                #     self.a2a_attn_layers[layer_i], feat_a_pt, r_a2a, edge_index_a2a, train_mask, batch_s_repeat, n_step
                # )
                # feat_list1 = self.a2a_attn_layers[layer_i].refer1(feat_a_pt, r_a2a, edge_index_a2a, train_mask,
                #                                                 batch_s_repeat, n_step)

                feat_list = self.a2a_attn_layers[layer_i].refer(feat_a_pt, r_a2a, edge_index_a2a, train_mask,
                                                                batch_s_repeat, n_step)

                feat_list_mean=[]
                valid_mask=[]
                for feat in feat_list:
                    if feat.shape[1]>0:
                        feat_list_mean.append(feat.mean(dim=1))
                        valid_mask.append(torch.ones(1))
                    else:
                        valid_mask.append(torch.zeros(1))

                feat_ablated = torch.stack(feat_list_mean, dim=1)

                valid_batch=batch_s_repeat[train_mask,0]

                feat_mean=scatter_mean(feat_a, valid_batch,dim=1)

                feat_a=feat_mean[:,valid_batch]

                feat_a = torch.cat([feat_a, feat_ablated], dim=1)

                feat_a = feat_a.flatten(0, 1)
            else:
                if self.num_layers > 1 and layer_i == self.num_layers - 1 and train_mask is not None:
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

        if  self.use_edge_feature and self.discriminator:
            feat_a_all = None
        else:
            feat_a_all = feat_a.view( n_step,  -1,self.hidden_dim).transpose(0, 1)

            if self.num_layers>1 and train_mask is not None:
                feat_a=feat_a_all[train_mask]
            else:
                feat_a=feat_a_all

        proposal=None

        if self.discriminator and self.centric:
            index=batch_s_repeat[train_mask]
            feat_a, argmax = scatter_max(feat_a, index, dim=0)  # out: [B,T,C]

        if self.n_token_agent>1 and self.n_token_agent<2048:
            feat_a=torch.amax(feat_a, dim=1,keepdim=True)
            # index=batch_s_repeat[train_mask]
            # #feat_a, argmax = scatter_mean(feat_a, index, dim=0)  # out: [B,T,C]
            # feat_a = scatter_mean(feat_a, index, dim=0)  # out: [B,T,C]

        if self.pred_last_res:
            if self.training:
                proposal_feature = feat_a.detach()#[:, :-1]
            else:
                proposal_feature = feat_a#[:, -1:]

            proposal = self.traj_head(proposal_feature)  #
            proposal = proposal.reshape(proposal.shape[0], proposal.shape[1], 1, -1, 3)

        if self.pred_all_res and self.training:
            next_token_idx = sampled_idx

            token_local_traj = self.token_processor.token_local_traj

            if train_mask is not None:
                token_local_traj=token_local_traj[train_mask]
                next_token_idx=next_token_idx[train_mask]
                agent_token_emb=agent_token_emb[train_mask]
                n_agent=next_token_idx.shape[0]

            proposal_feature = feat_a + agent_token_emb
            proposal = self.traj_head(proposal_feature)  #
            proposal = proposal.reshape(proposal.shape[0], proposal.shape[1], 1, -1, 3)

            next_token_traj_all = token_local_traj[torch.arange(n_agent)[:, None], next_token_idx]

            if self.token_processor.max_diff is not None:

                proposal_max_diff = self.token_processor.token_diff[torch.arange(n_agent)[:, None], next_token_idx]

                proposal = torch.tanh(proposal) * proposal_max_diff[:, :, None]

            if self.output_gmm:
                proposal=    proposal.reshape(proposal.shape[0], proposal.shape[1], 2,-1, 3)

                proposal=torch.arange(0.2,1.2,0.2,device=proposal.device)[None,None,None,:,None]*proposal

                proposal[:,:,0]+=next_token_traj_all
                proposal[:,:,1]=0.001#torch.exp(proposal[:,:,1])+0.01

            else:
                proposal = proposal + next_token_traj_all[:, :, None]

        if self.discriminator:
            if self.state_action:
                feat_a = feat_a + agent_token_emb[train_mask]

        if self.discriminator and self.diff_dicriminator:
            state = feat_a.reshape(-1, 128)

            state = (state - state.mean(0, keepdim=True)) / (state.std(0, keepdim=True) + 1e-5)

            next_token_logits=self.token_predict_head._compute_disc_val(state, None).reshape(-1,16)

        else:
            next_token_logits = self.token_predict_head(feat_a)

        if self.discriminator and self.reward_shaping:
            r=self.reward_net(feat_a[:, 1:] )#+ agent_token_emb
            v_s=next_token_logits[:, :-1]
            v_next=next_token_logits[:,1 :]
            done=torch.ones_like(v_s)
            done[:,-1]=0
            next_token_logits = r + (0.99*v_next - v_s)*done
        weight = None

        if self.discriminator:
            if self.use_edge_feature:
                end_idx = edge_index_a2a[1]  # shape: [E]

                if self.learn_weight:
                    weight =next_token_logits[:,:,-1:]
                else:
                    weight=torch.exp(-dist[:,None,None]/3)*1

                interact_logits=next_token_logits[:,:,:1]*weight

                interact_logits_sum = scatter_sum(interact_logits, end_idx, dim=0, dim_size=len(train_repeat_mask))#[0]

                rewards=interact_logits_sum[train_repeat_mask].view( n_step,  -1).transpose(0, 1)

                if not self.use_ego_loop:
                    #rewards[rewards==0]=1000

                    # next_token_logits = torch.cat([ego_logits,next_token_logits], dim=0)#ego_logits#

                    ego_rewards=ego_logits.view(n_step,  -1).transpose(0, 1)

                    rewards=ego_rewards+rewards#torch.minimum(rewards,ego_rewards)#rewards+ego_rewards#+torch.zeros_like(torch.minimum(ego_rewards,rewards)#)#rewards+ego_rewards#

                    if self.learn_weight:
                        next_token_logits=rewards.flatten(0,1)[:,None,None]

                all_rewards=torch.zeros_like(head_a)

                all_rewards[train_mask]=rewards

                weight2=torch.exp(-dist/3)*2

                flatten_reward=all_rewards.transpose(0, 1).flatten(0,1)

                weighted_nei_reward=flatten_reward[edge_index_a2a[0]]*weight2

                nei_sum = scatter_sum(weighted_nei_reward, end_idx, dim=0, dim_size=len(train_repeat_mask))

                nei_sum_rewards=nei_sum[train_repeat_mask].view( n_step,  -1).transpose(0, 1)

                rewards=rewards+nei_sum_rewards

                rewards=(rewards.detach(),nei_sum_rewards.detach())

            elif self.use_counterfactual:

                logit_original= next_token_logits[:n_agent,:,0]
                ablated_logit = torch.zeros_like(logit_original)
                valid_mask=torch.stack(valid_mask,dim=0).to(bool)[:,0]
                ablated_logit[valid_mask]=next_token_logits[n_agent:,:,0]

                rewards=(logit_original - ablated_logit).detach()
                # reward_list=[]
                #
                # batch_id=batch_s_repeat[train_mask,0]
                #
                # a_i=0
                #
                # for i,b in enumerate(batch_id):
                #     logit_a=logit_original[batch_id==b]
                #
                #     ablated_logit_a=ablated_logit[a_i:a_i+feat_list[i].shape[1]]
                #
                #     a_i+=feat_list[i].shape[1]
                #
                #     if len(ablated_logit_a)>0:
                #         reward = logit_a.mean(dim=0)-ablated_logit_a.mean(dim=0)
                #     else:
                #         reward=logit_a.mean(dim=0)
                #
                #     reward_list.append(reward)
                #
                # rewards=torch.stack(reward_list)
            else:
                rewards=next_token_logits[:,:,0].detach()
        else:
            rewards=None

        return next_token_logits,feat_a_all,proposal,rewards,weight