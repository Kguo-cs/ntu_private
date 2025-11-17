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
from src.smart.my_model.diffusion_discriminator import Discriminator
from src.smart.my_model.NoiseSchedule import NoiseSchedule,SinusoidalTimestep,cosine_beta_schedule
from src.smart.layers.relative_transformer import RoFormerBlock



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
            dis_weight,
            dist_decay,
            discriminator=False,
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
        self.dis_weight=dis_weight
        self.dis_decay=dist_decay

        self.head_dim = hidden_dim // num_heads

        self.agent_hist = self.time_span // self.shift

        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        share=discriminator,
                                        hist_drop_prob=hist_drop_prob,
                                        time_span=time_span,
                                        use_roformer=use_roformer,
                                        use_route=token_processor.use_route,
                                        discriminator=discriminator,
                                        use_bird=token_processor.use_bird
                                        )

        self.use_roformer=use_roformer

        self.pred_exit=token_processor.pred_exit

        self.t_num_layers = 1

        self.agent_hist = self.time_span // self.shift*self.t_num_layers

        if self.use_roformer:
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
                    )
                    for _ in range(self.t_num_layers)
                ]
            )
            self.token_cache=None

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
        self.discriminator = discriminator

        self.state_action = False
        self.reward_shaping = False
        self.diff_dicriminator = False

        self.use_counterfactual=False
        self.use_edge_feature=True

        self.use_full_feature=False

        if not (discriminator and self.use_edge_feature and not self.use_full_feature):
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

        self.use_diffusion=False

        if self.use_diffusion:
            n_token_agent=5*4*2
            
            self.n_steps = n_steps = 1000

            betas = cosine_beta_schedule(self.n_steps)
            self.betas = betas#.to(self.args.device)
            alphas = 1 - betas
            alphas_prod = torch.cumprod(alphas, 0)
            alphas_bar_sqrt = torch.sqrt(alphas_prod)#.to(self.args.device)
            one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)#.to(self.args.device)
            
            self.register_buffer("alphas_bar_sqrt",alphas_bar_sqrt)
            self.register_buffer("one_minus_alphas_bar_sqrt",one_minus_alphas_bar_sqrt)     


            self.schedule = NoiseSchedule.cosine(timesteps=1000)
            self.t_embed = nn.Embedding(n_steps, hidden_dim)
            self.fut_embed = nn.Linear(n_token_agent, hidden_dim)

        self.pred_last_res = pred_last_res
        self.pred_all_res = pred_all_res
        self.n_token_agent=n_token_agent

        if self.pred_last_res or self.pred_all_res:
            if self.output_gmm:
                self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3*2*1) #mean and std
            else:
               self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3 * 5)

        self.start_step=self.num_historical_steps//self.shift-1

        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.token_processor=token_processor

        self.filter_ratio=0
        if self.discriminator and self.diff_dicriminator:
            self.token_predict_head = Discriminator(hidden_dim, hidden_dim, False, num_units=128)
        else:
            if self.discriminator and self.use_edge_feature:
                self.interact_head = MLPLayer(
                    input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

                if self.use_full_feature:
                    self.all_head = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                    )

            self.token_predict_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
            )

        if self.discriminator:
            self.centric=False
            if  self.reward_shaping:
                self.reward_net = MLPLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
                )

    def predict_agent(self,feat_a,feat_map,n_step,n_agent,
                      r_t,edge_index_t,
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      agent_train_mask,dist,
                      train_repeat_mask,mask_a,n_current,inference_mask
                      ):
        start_index = edge_index_a2a[0]
        end_index = edge_index_a2a[1]
        valid_number=len(feat_a)
        mask_ta=mask_a.transpose(0, 1)
        mask_ta_flatten=mask_ta.flatten(0,1)

        # if not self.discriminator and not self.token_processor.use_bird:
        #     feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)
        #
        #     feat_a_t[mask_ta] = feat_a
        #
        #     if n_current == 0:
        #         self.feat_a_cache = feat_a_t
        #     else:
        #         self.feat_a_cache = torch.cat((self.feat_a_cache, feat_a_t), dim=0)[-self.agent_hist:]  # t,a
        #
        #         feat_a = self.feat_a_cache[self.mask_cache.transpose(0, 1)]
        #
        #     for i in range(self.t_num_layers):
        #         feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
        #
        #     if n_current != 0:
        #         current_len = mask_a.sum()
        #         feat_a = feat_a[-current_len:]

        for layer_i in range(self.num_layers):
            if (self.use_edge_feature and self.discriminator):
                start_edge_feature=feat_a[start_index]
                end_edge_feature=feat_a[end_index]

                if  agent_train_mask is not None and self.num_layers==1:
                    feat_a = feat_a[train_repeat_mask]
                    n_agent=agent_train_mask.sum()

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
                    n_agent=agent_train_mask.sum()

                feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                if  agent_train_mask is not None and self.num_layers==1:
                    feat_a = feat_a.view(-1,n_agent,self.hidden_dim)[:,agent_train_mask]
                    n_agent = feat_a.shape[1]
                    feat_a=feat_a.flatten(0,1)

                if not self.token_processor.use_bird:
                    feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

        if  agent_train_mask is not None and self.num_layers>1:
            feat_a = feat_a[train_repeat_mask]

        #if self.discriminator or self.token_processor.use_bird:
        feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

        feat_a_t[mask_ta] = feat_a

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

        current_len = inference_mask.sum()
        feat_a = feat_a[-current_len:]

        next_token_logits = self.token_predict_head(feat_a)

        weight = None

        if self.discriminator:
            if self.use_edge_feature:
                weight=torch.exp(-dist / self.dis_decay) * self.dis_weight#torch.ones_like(dist) #=

                interact_reward=torch.zeros_like(next_token_logits[:,0])

                valid_interact_reward=scatter_sum(interact_logits[:,0].detach() * weight, end_index, dim=0,  dim_size=valid_number)

                interact_reward[mask_ta_flatten] = valid_interact_reward[train_repeat_mask]

                valid_ego_reward=next_token_logits[:,0].detach()

                ego_rewards = valid_ego_reward + interact_reward

                if self.use_full_feature:
                    next_token_logits=torch.cat([next_token_logits, all_logits, interact_logits], dim=0)

                    all_weight  = torch.ones_like(ego_rewards)*0.1

                    ego_rewards = all_weight*all_logits[:,0] +  ego_rewards

                    weight=torch.cat([all_weight,weight], dim=0)
                else:
                    next_token_logits = (next_token_logits[:,0], interact_logits[:,0])

                weight2=weight  #torch.exp(-dist/self.dis_decay)*self.dis_weight

                all_rewards = torch.zeros_like(valid_interact_reward)
                all_rewards[train_repeat_mask] = ego_rewards[mask_ta_flatten]

                weighted_nei_reward=all_rewards[start_index]*weight2

                nei_rewards=torch.zeros_like(ego_rewards)

                nei_rewards_sum= scatter_sum(weighted_nei_reward, end_index, dim=0, dim_size=valid_number)

                nei_rewards[mask_ta_flatten] =nei_rewards_sum[train_repeat_mask]  #the source

                rewards=(ego_rewards,nei_rewards,valid_ego_reward,valid_interact_reward)
            else:
                rewards=next_token_logits[...,0].detach(),torch.tensor(0.0)
        else:
            rewards=torch.tensor(0.0),torch.tensor(0.0)

        return next_token_logits,feat_a,rewards,weight

    def forward(self,all_features,map_feature,agent_train_mask,n_current,pred_mask ):

        feat_a, pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s=all_features

        n_agent,n_step = mask_a.shape

        if n_current==0:
            self.pos_cache = pos_a
            self.head_cache = head_a
            self.mask_cache = mask_a
            self.head_vector_cache = head_vector_a

            if self.discriminator or self.edge_encoder.rollout_traj:
                inference_mask = torch.ones_like(self.mask_cache)
            else:
                inference_mask = self.mask_cache.clone()

            if not self.discriminator:
                inference_mask[:, :self.start_step] = False
        else:
            self.pos_cache = torch.cat((self.pos_cache, pos_a), dim=1)[:, -self.agent_hist:]
            self.head_cache = torch.cat((self.head_cache, head_a), dim=1)[:, -self.agent_hist:]
            self.mask_cache = torch.cat((self.mask_cache, mask_a), dim=1)[:, -self.agent_hist:]
            self.head_vector_cache = torch.cat((self.head_vector_cache, head_vector_a), dim=1)[:, -self.agent_hist:]

            inference_mask = self.mask_cache.clone()

            inference_mask[:, :-1] = False #a,t

        if agent_train_mask is not None:
            inference_mask=inference_mask[agent_train_mask]

        edge_index_t, r_t = self.edge_encoder.build_temporal_edge(
            pos_a=self.pos_cache,  # [n_agent, n_step, 2]
            head_a=self.head_cache,  # [n_agent, n_step]
            head_vector_a=self.head_vector_cache,  # [n_agent, n_step, 2]
            mask=self.mask_cache,  # [n_agent, n_step]
            inference_mask=inference_mask,
            agent_train_mask=agent_train_mask
        )

        # if not self.discriminator and not self.token_processor.use_bird:
        #     if n_current == 0:
        #         self.feat_a_cache = feat_a
        #         feat_a=feat_a.flatten(0,1)
        #     else:
        #         self.feat_a_cache = torch.cat((self.feat_a_cache, feat_a), dim=1)[:,-self.agent_hist:]  # t,a
        #
        #         feat_a = self.feat_a_cache[self.mask_cache]
        #
        #     for i in range(self.t_num_layers):
        #         feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
        #
        #     # feat_a_t = torch.zeros_like(self.feat_a_cache)
        #     #
        #     # feat_a_t[self.mask_cache] = feat_a
        #     feat_a_t=feat_a.reshape(n_agent,-1,self.hidden_dim)
        #
        #     all_features[0] = feat_a_t[:,-n_step:]
        #
        #     if n_step>1:
        #         all_features=[feat[:,self.start_step:] for  feat in all_features]

        # feat_a, pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s=all_features

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
                use_counterfactual=self.use_counterfactual,
                route_map_index=None,
                layer_num=self.num_layers
            )
        else:
            edge_index_pl2a=r_pl2a=feat_map=None

        feat_a,pos_s, head_s, head_vector_s,mask_s, _,batch_s=[feat.transpose(0, 1).flatten(0, 1) for feat in all_features ]

        if agent_train_mask is not None:
            train_repeat_mask=agent_train_mask[:,None].repeat(1,n_step).transpose(0, 1).flatten(0, 1)[mask_s]
            mask_a=mask_a[agent_train_mask]
            pred_mask=pred_mask[agent_train_mask]
        else:
            train_repeat_mask=None

        feat_a=feat_a[mask_s]

        edge_index_a2a, r_a2a, dist,relative_pos = self.edge_encoder.build_interaction_edge(
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
            agent_train_mask=train_repeat_mask,
            loop=False
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        next_token_logits, feat_a, rewards, weight=self.predict_agent(feat_a,feat_map,n_step,n_agent,
                                                                         r_t,edge_index_t,
                                                                          r_pl2a, edge_index_pl2a,
                                                                          r_a2a,edge_index_a2a,
                                                                          agent_train_mask,dist,
                                                                          train_repeat_mask,mask_a,n_current,inference_mask)


        if not self.discriminator and self.pred_exit and pred_mask is not None:
            next_token_logits[pred_mask[None].repeat(inference_mask.shape[1],1)[inference_mask.transpose(0, 1)], -1] = -10000 #t,a

        return next_token_logits,feat_a,rewards,weight,(edge_index_a2a,relative_pos)
