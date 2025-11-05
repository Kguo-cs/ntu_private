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
                self.interact_head = MLPLayer(
                    input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=n_token_agent
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
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      batch_s_repeat,train_mask,dist,
                      train_repeat_mask,mask_a,head_a
                      ):
        start_index = edge_index_a2a[0]
        end_index = edge_index_a2a[1]

        for layer_i in range(self.num_layers):

            if (self.use_edge_feature and self.discriminator):

                start_edge_feature=feat_a[start_index]
                end_edge_feature=feat_a[end_index]

                if  train_mask is not None and self.num_layers==1:
                    feat_a = feat_a.view(-1,n_agent,self.hidden_dim)[:,train_mask]
                    n_agent = feat_a.shape[1]
                    feat_a=feat_a.flatten(0,1)

                if not self.token_processor.use_bird:
                    feat_a = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

                feat_interact = torch.cat([start_edge_feature, r_a2a, end_edge_feature], dim=-1)
                interact_logits = self.interact_head(feat_interact)

            elif (self.discriminator and self.use_counterfactual):
                if  train_mask is not None:
                    connected_agent=torch.unique(edge_index_a2a[0])
                    in_mask=torch.isin(edge_index_pl2a[1], connected_agent)
                    r_pl2a=r_pl2a[in_mask]
                    edge_index_pl2a = edge_index_pl2a[:, in_mask]

                feat_a_pt = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

                feat_a = self.a2a_attn_layers[layer_i](feat_a_pt, r_a2a, edge_index_a2a)

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

                feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                if  train_mask is not None and self.num_layers==1:
                    feat_a = feat_a.view(-1,n_agent,self.hidden_dim)[:,train_mask]
                    n_agent = feat_a.shape[1]
                    feat_a=feat_a.flatten(0,1)

                if not self.token_processor.use_bird:
                    feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)

        if not ( self.use_edge_feature and self.discriminator) and (self.num_layers>1 and train_mask is not None):
            feat_a_all = feat_a.view( n_step,  -1,self.hidden_dim).transpose(0, 1)

            feat_a = feat_a_all[train_mask]

        if self.discriminator and self.centric:
            index=batch_s_repeat[train_mask]
            feat_a, argmax = scatter_max(feat_a, index, dim=0)  # out: [B,T,C]

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
                weight = torch.exp(-dist / self.dis_decay) * self.dis_weight#1/(1+dist.exp())

                interact_logits_sum = scatter_sum(interact_logits[:,0] * weight, end_index, dim=0,  dim_size=len(feat_a)) #a_number

                if train_repeat_mask is not None:
                    interact_logits_sum = interact_logits_sum[train_repeat_mask]

                ego_rewards = next_token_logits[:,0] + interact_logits_sum

                weight2=weight#torch.exp(-dist/self.dis_decay)*self.dis_weight

                weighted_nei_reward=ego_rewards[start_index]*weight2

                nei_sum_rewards = scatter_sum(weighted_nei_reward, end_index, dim=0, dim_size=len(feat_a))

                rewards=(ego_rewards.detach(),nei_sum_rewards.detach())

                next_token_logits = torch.cat([next_token_logits,interact_logits], dim=0)

            elif self.use_counterfactual:

                logit_original= next_token_logits[:n_agent,:,0]
                ablated_logit = torch.zeros_like(logit_original)
                valid_mask=torch.stack(valid_mask,dim=0).to(bool)[:,0]
                ablated_logit[valid_mask]=next_token_logits[n_agent:,:,0]

                rewards=(logit_original - ablated_logit).detach()
            else:
                rewards=next_token_logits[...,0].detach(),torch.tensor(0.0)
        else:
            rewards=torch.tensor(0.0),torch.tensor(0.0)

        return next_token_logits,feat_a,rewards,weight

    def forward(self,all_features,map_feature,train_mask,route_map_index ):
        feat_a_t,feat_a_token,pos_a, head_a, head_vector_a,mask_a, batch_s_repeat,batch_s,agent_token_emb,sampled_idx=all_features

        n_agent = mask_a.shape[0]
        n_step=mask_a.shape[1]

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
                train_mask=train_mask,
                use_counterfactual=self.use_counterfactual,
                route_map_index=route_map_index,
                layer_num=self.num_layers
            )
        else:
            edge_index_pl2a=r_pl2a=feat_map=None

        feat_a,feat_a_token,pos_s, head_s, head_vector_s,mask_s, _,batch_s=[feat.transpose(0, 1).flatten(0, 1) for feat in all_features[:-2] ]

        if train_mask is not None:
            train_repeat_mask=train_mask[:,None].repeat(1,n_step).transpose(0, 1).flatten(0, 1)[mask_s]
        else:
            train_repeat_mask=None

        feat_a=feat_a[mask_s]

        edge_index_a2a, r_a2a, dist = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s[mask_s],  # [n_agent, n_step, 2]
            head_s=head_s[mask_s],  # [n_agent, n_step]
            head_vector_s=head_vector_s[mask_s],  # [n_agent, n_step, 2]
            batch_s=batch_s[mask_s],  # [n_agent*n_step]
            mask=mask_s[mask_s],  # [n_agent, n_step]
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            proposal=None,
            vis_mask=None,
            value=False,
            train_mask=train_repeat_mask,
            loop=False
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]


        if self.use_diffusion:
            if self.training:

                token_traj_all =self.token_processor.token_traj_all

                future = token_traj_all[torch.arange(len(sampled_idx))[:, None], sampled_idx]
                batch_idx = batch_s_repeat[:,0]
                T = self.schedule.timesteps

                t_idx_batch = torch.randint(low=0, high=T, size=(max(batch_idx) + 1, future.shape[1]),
                                            device=batch_idx.device)
                t_idx = t_idx_batch[batch_idx]

                noise = torch.randn_like(future)
                
                a = self.alphas_bar_sqrt[t_idx][:, :, None]

                # coefficient of eps
                aml = self.one_minus_alphas_bar_sqrt[t_idx][:, :, None]  

                fut_noisy = a * future + aml * noise
                fut_embed=self.fut_embed(fut_noisy)+self.t_embed(t_idx)
                feat_a=feat_a+fut_embed.transpose(0, 1).flatten(0, 1)
            else:
                device = feat_a.device
                steps=50

                x = torch.randn(n_agent, 1, self.n_token_agent, device=device)
                a = self.n_steps// steps

                time_steps = torch.arange(0, self.n_steps, a,device=device)[:,None,None]
                time_steps = time_steps + 1
                # previous sequence
                time_steps_prev = torch.cat([torch.zeros_like(time_steps[:1]), time_steps[:-1]])

                for i in reversed(range(0, steps)):
                    t_idx = time_steps[i]
                    t_prev= time_steps_prev[i]
                    a = self.alphas_bar_sqrt[t_idx]
                    aml = self.one_minus_alphas_bar_sqrt[t_idx]
                    
                    a_prev= self.alphas_bar_sqrt[t_prev]
                    aml_prev= self.one_minus_alphas_bar_sqrt[t_prev]

                    fut_embed = self.fut_embed(x) + self.t_embed(t_idx)

                    feat_a_f_t = feat_a + fut_embed.transpose(0, 1).flatten(0, 1)

                    eps = self.predict_agent(feat_a_f_t,feat_map,n_step,n_agent,
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      batch_s_repeat,train_mask,dist,
                      train_repeat_mask)[0]

                    x0 = (x - aml * eps) / a
                    if i == 0:
                        x = x0
                        return x, feat_a, None, None, None
                    
                    x = a_prev * x0 +  aml_prev * eps

        next_token_logits, feat_a, rewards, weight=self.predict_agent(feat_a,feat_map,n_step,n_agent,
                      r_pl2a, edge_index_pl2a,
                      r_a2a,edge_index_a2a,
                      batch_s_repeat,train_mask,dist,
                      train_repeat_mask,mask_a,head_a)

        return next_token_logits,feat_a,rewards,weight,edge_index_a2a
