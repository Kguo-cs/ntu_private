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
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges
from torch_scatter import scatter_max,scatter_mean,scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding

from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle



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
            dis_weight,
            dist_decay,
            reward_weight,
            reward_decay,
            discriminator=False,
    ) -> None:
        super(InterativeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob
        self.dis_weight=dis_weight
        self.dis_decay=dist_decay
        self.reward_weight=reward_weight
        self.reward_decay=reward_decay

        self.head_dim = hidden_dim // num_heads

        self.agent_hist = self.time_span // self.shift

        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        share=discriminator,
                                        hist_drop_prob=hist_drop_prob,
                                        time_span=time_span,
                                        shift=token_processor.shift,
                                        discriminator=discriminator,
                                        use_bird=token_processor.use_bird,
                                        use_cross=not token_processor.use_bird,
                                        pred_exit=token_processor.pred_exit,
                                        )


        self.pred_exit=token_processor.pred_exit

        self.t_num_layers = 1

        self.agent_hist = self.time_span // self.shift*self.t_num_layers

        if self.edge_encoder.use_roformer:
            self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=hist_drop_prob,
                                              hist_len=self.agent_hist)
        else:
            self.t_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=hist_drop_prob,
                        bipartite=False,
                        has_pos_emb=True,
                       # gated_attention=discriminator,
                    )
                    for _ in range(self.t_num_layers)
                ]
            )

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
                      #  gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )
            # self.a2pt_attn_layers = nn.ModuleList(
            #     [
            #         AttentionLayer(
            #             hidden_dim=hidden_dim,
            #             num_heads=num_heads,
            #             head_dim=head_dim,
            #             dropout=dropout,
            #             bipartite=True,
            #             has_pos_emb=True,
            #           #  gated_attention=discriminator,
            #         )
            #         for _ in range(num_layers)
            #     ]
            # )
            #

        self.discriminator = discriminator
        self.use_decompose=True
        self.use_full_feature=False
        self.use_airl=False

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
                    #    gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.n_token_agent=n_token_agent

        self.start_eval_step=self.num_historical_steps//self.shift-1

        self.mask_pred=False

        if token_processor.pred_init :
            self.start_step=0
        else:
            self.start_step=self.num_historical_steps//self.shift-1


        self.add_a2a=False
        
        if self.add_a2a and not discriminator:
            self.a2a_inter =AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=False,
                        has_pos_emb=True,
                    )
            self.start_eval_step = 0
            self.start_step=0



        if self.mask_pred:
            self.action_embed=nn.Embedding(n_token_agent+1,hidden_dim)

            self.a2a_inter =AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=False,
                        has_pos_emb=True,
                    )
            
            self.action_predict_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
            )
            self.start_eval_step = 0
            self.start_step=0


        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor
        self.token_processor=token_processor

        if self.discriminator:
            if self.use_decompose:
                self.interact_head = MLPLayer(
                    input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=n_token_agent
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

        self.apply(weight_init)

    def pred_mask_logit(self, action, pred_action_mask, a2a_feature, target_valid, feat_a):

        action[pred_action_mask] = self.n_token_agent

        edge_index_a2a, r_a2a, relative_pos = a2a_feature

        action_feature = self.action_embed(action)

        feat_a = feat_a + action_feature

        pred_valid =  pred_action_mask

        end_mask = pred_valid[edge_index_a2a[1]]
        edge_index_a2a = edge_index_a2a[:, end_mask]
        r_a2a = r_a2a[end_mask]

        feat_a_all = self.a2a_inter(feat_a, r_a2a, edge_index_a2a)

        mask_token_logit = self.action_predict_head(feat_a_all[pred_valid])

        return mask_token_logit

    def predict_agent(self,feat_a,feat_map,
                      r_t,edge_index_t,
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      agent_train_mask,dist,
                      train_repeat_mask,mask_a,
                      n_current,inference_mask,
                      token_embeding,pos_a, head_a
                      ):
        valid_number=len(feat_a)
        mask_ta=mask_a.transpose(0, 1)
        mask_ta_flatten=mask_ta.flatten(0,1)
        n_agent = inference_mask.shape[0]
        n_step = mask_a.shape[1]

        #rank=topo_rank_among_edges(edge_index_a2a[1],dist)

        for layer_i in range(self.num_layers):
            if (self.use_decompose and self.discriminator):
                start_index = edge_index_a2a[0]       #edge_index[1] = src indices = its k nearest neighbors
                end_index = edge_index_a2a[1]        #edge_index[0] = dst indices = query point

                start_edge_feature = feat_a[start_index]

                if token_embeding is not None:
                    end_edge_feature   = (feat_a+token_embeding[mask_ta_flatten])[end_index]
                else:
                    end_edge_feature   = feat_a[end_index]

                if  agent_train_mask is not None and self.num_layers==1:
                    feat_a = feat_a[train_repeat_mask]

                if not self.token_processor.use_bird:
                    feat_a = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

                if self.use_full_feature:
                    feat_a_all = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)
                    all_logits= self.all_head(feat_a_all)

                feat_interact = torch.cat([start_edge_feature, r_a2a, end_edge_feature], dim=-1)
                interact_logits = self.interact_head(feat_interact)
            else:
                if self.num_layers > 1 and layer_i == self.num_layers - 1 and agent_train_mask is not None:
                    end_mask=train_repeat_mask[edge_index_a2a[1]]
                    edge_index_a2a = edge_index_a2a[:, end_mask]
                    r_a2a=r_a2a[end_mask]

                    end_pt_mask=train_repeat_mask[edge_index_pl2a[1]]
                    edge_index_pl2a = edge_index_pl2a[:, end_pt_mask]
                    r_pl2a=r_pl2a[end_pt_mask]

                feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                if  agent_train_mask is not None and self.num_layers==1:
                    feat_a=feat_a[train_repeat_mask]

                if not self.token_processor.use_bird:#[:len(r_pl2a)//2]
                    #feat_map  = self.a2pt_attn_layers[layer_i]((feat_a,feat_map), r_pl2a[len(r_pl2a)//2:], edge_index_pl2a.flip(0))  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst
                    feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

                if self.num_layers > 1 and layer_i == self.num_layers - 1 and agent_train_mask is not None :
                    feat_a = feat_a[train_repeat_mask]

        feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

        feat_a_t[mask_ta] = feat_a

        if self.edge_encoder.use_roformer:
            # if agent_train_mask is not None:
            #     self.pos_cache = self.pos_cache[agent_train_mask]
            #     self.head_cache = self.head_cache[agent_train_mask]
            #     self.mask_cache = inference_mask[agent_train_mask]
            feat_a_t = self.a_t_roformer.temporal_embed(feat_a_t.transpose(0,1), None, None, n_step, n_current, inference_mask)

            feat_a=feat_a_t.transpose(0,1).flatten(0,1)
        else:
            if self.discriminator or self.edge_encoder.rollout_traj:
                feat_a=feat_a_t.flatten(0,1)
            else:
                if n_current == 0:
                    self.feat_a_cache = feat_a_t
                else:
                    self.feat_a_cache = torch.cat((self.feat_a_cache, feat_a_t), dim=0)[-self.agent_hist:]  # t,a

                    feat_a = self.feat_a_cache[self.mask_cache.transpose(0, 1)]

            for i in range(self.t_num_layers):
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)

            #each step entry agent allow attend to current agent feature

        if self.discriminator:
            if token_embeding is not None :
                if self.use_airl:
                    feat_sa=feat_a[:-n_agent]+token_embeding
                else:
                    feat_a=feat_a+token_embeding
        else:
            current_len = inference_mask.sum()
            feat_a = feat_a[-current_len:]
            
        if self.add_a2a and not self.discriminator:
            feat_a=self.a2a_inter(feat_a, r_a2a, edge_index_a2a)

        next_token_logits = self.token_predict_head(feat_a)

        if self.discriminator and self.use_airl:
            next_token_logits1 = self.token_predict_head1(feat_sa)
            next_token_logits=next_token_logits1+0.99*next_token_logits[n_agent:]-next_token_logits[:-n_agent]

        weight = None

        if self.discriminator:
            valid_ego_reward = next_token_logits[:, 0].detach()

            if self.use_decompose:
                weight=torch.exp(-dist / self.dis_decay)* self.dis_weight#torch.ones_like(dist) #=

                interact_reward=torch.zeros_like(next_token_logits[:,0])

                weight_logit= interact_logits[:,0].detach() * weight

                valid_interact_reward=scatter_sum(weight_logit, end_index, dim=0,  dim_size=valid_number)

                interact_reward[mask_ta_flatten] = valid_interact_reward[train_repeat_mask]

                ego_rewards = valid_ego_reward + interact_reward

                next_token_logits = (next_token_logits[:, 0], interact_logits[:, 0])

                weight2=torch.exp(-dist / self.reward_decay) * self.reward_weight  #torch.exp(-dist/self.dis_decay)*self.dis_weight

                all_rewards = torch.zeros_like(valid_interact_reward)
                all_rewards[train_repeat_mask] = ego_rewards[mask_ta_flatten]

                weighted_nei_reward=all_rewards[start_index]*weight2

                nei_rewards=torch.zeros_like(ego_rewards)

                nei_rewards_sum= scatter_sum(weighted_nei_reward, end_index, dim=0, dim_size=valid_number)

                nei_rewards[mask_ta_flatten] =nei_rewards_sum[train_repeat_mask]  #the source
            else:
                next_token_logits = (next_token_logits[:, 0], next_token_logits[:0,0])

                ego_rewards = valid_ego_reward
                valid_interact_reward=valid_ego_reward[:0]
                nei_rewards=ego_rewards[:0]

            rewards = (ego_rewards, nei_rewards, valid_ego_reward,valid_interact_reward)
        else:
            rewards=None

        return next_token_logits,feat_a,rewards,weight

    def forward(self,all_features,feat_a,token_embedding,map_feature,agent_train_mask,n_current,pred_mask,counter_feat_a=None ):

        if self.discriminator and token_embedding is not None and not self.use_airl:
            all_features=[feat[:,:-1] for feat in all_features]

        pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s=all_features

        n_agent,n_step = mask_a.shape

        if self.discriminator and self.edge_encoder.use_roformer:
            inference_mask = torch.ones_like(mask_a)
        else:
            if n_current == 0:
                self.pos_cache = pos_a
                self.head_cache = head_a
                self.mask_cache = mask_a
                self.head_vector_cache = head_vector_a

                if self.discriminator or self.edge_encoder.rollout_traj:
                    inference_mask = torch.ones_like(mask_a)
                else:
                    inference_mask = mask_a.clone()

                if not self.discriminator:
                    if self.training:
                        inference_mask[:, :self.start_step] = False
                    else:
                        inference_mask[:, :self.start_eval_step] = False# 5,10
            else:
                self.pos_cache = torch.cat((self.pos_cache, pos_a), dim=1)[:, -self.agent_hist:]
                self.head_cache = torch.cat((self.head_cache, head_a), dim=1)[:, -self.agent_hist:]
                self.mask_cache = torch.cat((self.mask_cache, mask_a), dim=1)[:, -self.agent_hist:]
                self.head_vector_cache = torch.cat((self.head_vector_cache, head_vector_a), dim=1)[:, -self.agent_hist:]

                inference_mask = self.mask_cache.clone()

                inference_mask[:, :-1] = False  # a,t

        if agent_train_mask is not None:
            inference_mask = inference_mask[agent_train_mask]

        if not self.edge_encoder.use_roformer:
            edge_index_t, r_t = self.edge_encoder.build_temporal_edge(
                pos_a=self.pos_cache,  # [n_agent, n_step, 2]
                head_a=self.head_cache,  # [n_agent, n_step]
                head_vector_a=self.head_vector_cache,  # [n_agent, n_step, 2]
                mask=self.mask_cache,  # [n_agent, n_step]
                inference_mask=inference_mask,
                agent_train_mask=agent_train_mask
            )
        else:
            edge_index_t, r_t = None,None

        if not self.token_processor.use_bird:
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
                agent_train_mask=agent_train_mask,
                layer_num=self.num_layers
            )
        else:
            edge_index_pl2a=r_pl2a=feat_map=None

        pos_s, head_s, head_vector_s,mask_s, _,batch_s=[feat.transpose(0, 1).flatten(0, 1) for feat in all_features ]

        if agent_train_mask is not None:
            train_repeat_mask=agent_train_mask[:,None].repeat(1,n_step).transpose(0, 1).flatten(0, 1)[mask_s]
            mask_a=mask_a[agent_train_mask]
            pred_mask=pred_mask[agent_train_mask]
        else:
            train_repeat_mask=None

        edge_index_a2a, r_a2a, dist,relative_pos,r_a2a_nei,center_nei_pos,center_nei_heading = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,  # [n_agent, n_step, 2]
            head_s=head_s,  # [n_agent, n_step]
            head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask_s,  # [n_agent, n_step]
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            agent_train_mask=train_repeat_mask,
            layer_num=self.num_layers,
            counter_feat_a=counter_feat_a
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        next_token_logits, feat_a_value, rewards, weight=self.predict_agent(feat_a,feat_map,
                                                                      r_t,edge_index_t,
                                                                      r_pl2a, edge_index_pl2a,
                                                                      r_a2a,edge_index_a2a,
                                                                      agent_train_mask,dist,
                                                                      train_repeat_mask,mask_a,
                                                                      n_current,inference_mask,
                                                                      token_embedding,pos_a, head_a
                                                                      )

        if r_a2a_nei is not None:
            counter_feat_a=counter_feat_a.transpose(0, 1).flatten(0, 1)[mask_s]

            next_token_logits_counter, _, rewards_counter, weight= self.predict_agent(counter_feat_a, feat_map,
                                                                            r_t, edge_index_t,
                                                                            r_pl2a, edge_index_pl2a,
                                                                            r_a2a_nei, edge_index_a2a,
                                                                            agent_train_mask, dist,
                                                                            train_repeat_mask, mask_a,
                                                                            n_current, inference_mask,
                                                                            token_embedding,pos_a, head_a
                                                                            )

            next_token_logits=(next_token_logits[0],next_token_logits_counter[0])

            rewards=(rewards[0]-rewards_counter[0],rewards[1],rewards[2],rewards[3])

        if not self.discriminator and self.pred_exit and pred_mask is not None:
            next_token_logits[pred_mask[None].repeat(inference_mask.shape[1],1)[inference_mask.transpose(0, 1)], -1] = -torch.inf #t,a

        return next_token_logits,feat_a_value,rewards,weight,(edge_index_a2a, r_a2a,relative_pos)

    def get_reward(self,weight,edge_value,end_index,valid_number,mask_ta_flatten,train_repeat_mask,n_step,n_agent):

        batch_reward = torch.zeros([n_step,n_agent],device=edge_value.device)

        edge_sum_weight = scatter_sum(edge_value * weight, end_index, dim=0,dim_size=valid_number)

        batch_reward[mask_ta_flatten] = edge_sum_weight[train_repeat_mask]

        return batch_reward
