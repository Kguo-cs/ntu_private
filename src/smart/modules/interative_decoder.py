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
                                        hist_drop_prob=hist_drop_prob,
                                        time_span=time_span,
                                        shift=token_processor.shift,
                                        discriminator=discriminator,
                                        use_bird=token_processor.use_bird,
                                        use_pl2a=True,
                                        use_a2a=True,
                                        use_t2t=True,
                                        )

        if discriminator:
            self.t_num_layers = 1
        else:
            self.t_num_layers = num_layers

        self.agent_hist = self.time_span // self.shift

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
                       # gated_attention=discriminator,
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
                      #  gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )

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

        self.mask_pred=False
        self.gail_start_step=2
        self.dis_start_step=2

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

        self.feat_a_cache=[[] for _ in range(num_layers)]

        self.apply(weight_init)

    def predict_agent(self,feat_a,feat_map,
                      r_t,edge_index_t,
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      agent_train_mask,dist,
                      train_repeat_mask,mask_a,
                      n_current,inference_mask,
                      token_embeding,pred_mask,n_agent
                      ):
        valid_number=len(feat_a)
        mask_ta=mask_a.transpose(0, 1)
        mask_ta_flatten=mask_ta.flatten(0,1)
        n_pred_agent = inference_mask.shape[0]
        n_step = mask_a.shape[1]

        for layer_i in range(self.num_layers):
            if (self.use_decompose and self.discriminator):
                start_index = edge_index_a2a[0]       #edge_index[1] = src indices = its k nearest neighbors
                end_index = edge_index_a2a[1]        #edge_index[0] = dst indices = query point

                feat_a_later=feat_a[n_agent*self.dis_start_step:]

                start_edge_feature = feat_a_later[start_index]

                if token_embeding is not None:
                    end_edge_feature   = (feat_a_later+token_embeding[mask_ta_flatten])[end_index]
                else:
                    end_edge_feature   = feat_a_later[end_index]

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
                feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)
                feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

            if  not self.edge_encoder.rollout_traj and not self.discriminator:#rollout or expert
                feat_a_t = torch.zeros([n_step, n_pred_agent, self.hidden_dim], device=feat_a.device)

                feat_a_t[mask_ta] = feat_a

                if n_current == 0:
                    self.feat_a_cache[layer_i] = feat_a_t
                else:
                    self.feat_a_cache[layer_i] = torch.cat((self.feat_a_cache[layer_i], feat_a_t),
                                                           dim=0)[-self.agent_hist:]  # t,a

                    feat_a = self.feat_a_cache[layer_i][self.mask_cache.transpose(0, 1)]

            feat_a = self.t_attn_layers[layer_i](feat_a, r_t, edge_index_t)

            if self.discriminator:
                if token_embeding is not None:
                    if self.use_airl:
                        feat_sa = feat_a[:-n_pred_agent] + token_embeding
                    else:
                        feat_a = feat_a + token_embeding
            else:
                current_len = inference_mask.sum()
                feat_a = feat_a[-current_len:]

        if   self.edge_encoder.rollout_traj:
            if pred_mask is not None:
                pred_repeat_mask = pred_mask[:, None].repeat(1, n_step).transpose(0, 1)
            else:
                pred_repeat_mask=torch.ones([n_step, n_agent], dtype=torch.bool,device=feat_a.device)

            pred_repeat_mask[:max(0,self.gail_start_step-1)]=False
            pred_repeat_mask=pred_repeat_mask.flatten(0, 1)
            feat_a=feat_a[pred_repeat_mask]

        if self.discriminator:
            feat_a=feat_a[n_pred_agent*self.dis_start_step:]

        next_token_logits = self.token_predict_head(feat_a)

        weight =rewards= None

        if self.discriminator:
            valid_ego_reward = next_token_logits[:, 0].detach()

            if self.use_decompose:

                valid_number = valid_number - n_agent * self.dis_start_step

                weight=torch.exp(-dist / self.dis_decay)* self.dis_weight#torch.ones_like(dist) #=

                weight_logit= interact_logits[:,0].detach() * weight

                valid_interact_reward=scatter_sum(weight_logit, end_index, dim=0,  dim_size=valid_number)#

                if train_repeat_mask is not None:
                    train_repeat_mask = train_repeat_mask[n_agent * self.dis_start_step:]

                    interact_reward = valid_interact_reward[train_repeat_mask]
                else:
                    interact_reward = valid_interact_reward

                ego_rewards = valid_ego_reward + interact_reward

                next_token_logits = (next_token_logits[:, 0], interact_logits[:, 0])

                # weight2=torch.exp(-dist / self.reward_decay) * self.reward_weight  #torch.exp(-dist/self.dis_decay)*self.dis_weight

                # all_rewards = torch.zeros_like(valid_interact_reward)
                # all_rewards[train_repeat_mask] = ego_rewards[mask_ta_flatten]

                # weighted_nei_reward=all_rewards[start_index]*weight2

                nei_rewards=torch.zeros_like(ego_rewards)

                # nei_rewards_sum= scatter_sum(weighted_nei_reward, end_index, dim=0, dim_size=valid_number)

                # nei_rewards[mask_ta_flatten] =nei_rewards_sum[train_repeat_mask]  #the source
            else:
                next_token_logits = (next_token_logits[:, 0], next_token_logits[:0,0])

                ego_rewards = valid_ego_reward
                valid_interact_reward=valid_ego_reward[:0]
                nei_rewards=ego_rewards[:0]

            rewards = (ego_rewards, nei_rewards, valid_ego_reward,valid_interact_reward)

        return next_token_logits,feat_a,rewards,weight

    def forward(self,all_features,feat_a,token_embedding,map_feature,agent_train_mask,n_current,tokenized_agent,counter_feat_a=None ):
        
        pred_mask=tokenized_agent["pred_mask"] if "pred_mask" in tokenized_agent else None
        train_mask=tokenized_agent["train_mask"] if "train_mask" in tokenized_agent else None

        if self.discriminator and token_embedding is not None and not self.use_airl:
            all_features=[feat[:,:-1] for feat in all_features]

        pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s=all_features

        n_agent,n_step = mask_a.shape

        if n_current == 0:
            self.pos_cache = pos_a
            self.head_cache = head_a
            self.mask_cache = mask_a
            self.head_vector_cache = head_vector_a

            inference_mask = mask_a.clone()
        else:
            self.pos_cache = torch.cat((self.pos_cache, pos_a), dim=1)[:, -self.agent_hist:]
            self.head_cache = torch.cat((self.head_cache, head_a), dim=1)[:, -self.agent_hist:]
            self.mask_cache = torch.cat((self.mask_cache, mask_a), dim=1)[:, -self.agent_hist:]
            self.head_vector_cache = torch.cat((self.head_vector_cache, head_vector_a), dim=1)[:, -self.agent_hist:]

            inference_mask = self.mask_cache.clone()

            inference_mask[:, :-1] = False  # a,t

        if agent_train_mask is not None:
            inference_mask = inference_mask[agent_train_mask]

        if self.edge_encoder.use_t2t:
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
            if pred_mask is not None:
                pred_mask=pred_mask[agent_train_mask]
        else:
            train_repeat_mask=None

        if self.discriminator:
            mask_s[:n_agent*self.dis_start_step]=False
            if  train_repeat_mask is not None:
                train_repeat_mask_a2a=train_repeat_mask[n_agent*self.dis_start_step:]
            else:
                train_repeat_mask_a2a=None
        else:
            train_repeat_mask_a2a=train_repeat_mask

        if self.discriminator:
            dis_mask = tokenized_agent["dis_mask"] if "dis_mask" in tokenized_agent else None
            mask_transpose = tokenized_agent["valid_mask"].transpose(0, 1)[self.gail_start_step:]

            if train_mask is not None:
                all_dis_mask = torch.zeros_like(mask_transpose)

                all_dis_mask[:, train_mask] = dis_mask.reshape(all_dis_mask.shape[0], -1)
            else:
                all_dis_mask = dis_mask.reshape(mask_transpose.shape)

            dis_edge_mask = all_dis_mask[mask_transpose]
        else:
            dis_edge_mask = None

        edge_index_a2a, r_a2a, dist,relative_pos,r_a2a_nei,center_nei_pos,center_nei_heading = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,  # [n_agent, n_step, 2]
            head_s=head_s,  # [n_agent, n_step]
            head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask_s,  # [n_agent, n_step]
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            agent_train_mask=train_repeat_mask_a2a,
            layer_num=self.num_layers,
            counter_feat_a=counter_feat_a,
            dis_edge_mask=dis_edge_mask
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        next_token_logits, feat_a_value, rewards, weight=self.predict_agent(feat_a,feat_map,
                                                                      r_t,edge_index_t,
                                                                      r_pl2a, edge_index_pl2a,
                                                                      r_a2a,edge_index_a2a,
                                                                      agent_train_mask,dist,
                                                                      train_repeat_mask,mask_a,
                                                                      n_current,inference_mask,
                                                                      token_embedding,pred_mask,n_agent
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
                                                                            token_embedding,pred_mask,n_agent
                                                                            )

            next_token_logits=(next_token_logits[0],next_token_logits_counter[0])

            rewards=(rewards[0]-rewards_counter[0],rewards[1],rewards[2],rewards[3])

        return next_token_logits,feat_a_value,rewards,weight,(edge_index_a2a, r_a2a,relative_pos)


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
