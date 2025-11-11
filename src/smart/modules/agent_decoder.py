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
from src.smart.utils import (
    transform_to_global,
    weight_init,
)
from torch.distributions import Categorical
from .build_edge import build_batch
from ..layers.relative_transformer import RoFormerBlock
from src.smart.utils.rollout import cal_polygon_contour
from src.smart.modules.light_encoder import LightEncoder
from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder
import numpy as np

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
            output_gmm,
            pred_last_res,
            pred_all_res,
            dis_weight,
            dist_decay,
            discriminator=False
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob

        self.alpha = alpha

        self.head_dim = hidden_dim // num_heads

        #if not discriminator:
        self.agent_token_embedding=AgentTokenEncoder(hidden_dim,num_freq_bands,token_processor,discriminator)
        self.t_num_layers = 1

        self.agent_hist = self.time_span // self.shift*self.t_num_layers

        self.use_roformer=False

        self.n_token_agent = n_token_agent
        self.output_gmm = output_gmm

        self.pred_last_res = pred_last_res
        self.pred_all_res = pred_all_res

        self.interative_decoder = InterativeDecoder(hidden_dim,num_historical_steps,num_future_steps,time_span,
                                                    pl2a_radius,a2a_radius,num_freq_bands,
                                                    num_layers,num_heads,head_dim,
                                                    dropout,hist_drop_prob,n_token_agent,
                                                    pt2a_neighbor,a2a_neighbor,
                                                    token_processor,output_gmm,pred_last_res,pred_all_res,
                                                    dis_weight,
                                                    dist_decay,
                                                    discriminator=discriminator,
                                                    use_roformer=self.use_roformer
                                                    )

        self.use_diffusion=self.interative_decoder.use_diffusion

        self.use_light = False
        self.pred_light = True
        self.light_type = 5
        self.light_hist = self.agent_hist

        if self.use_light:
            self.light_encoder = LightEncoder(self.interative_decoder.edge_encoder,hidden_dim,self.light_hist,num_heads,self.light_type,self.shift,self.pred_light,alpha)

            self.lg2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=True,
                        has_pos_emb=True,
                    )
                    for _ in range(1)
                ]
            )

        else:
            self.pred_light=False

        self.start_step=self.num_historical_steps//self.shift-1
        self.pred_vis = False

        self.target_net=False

        if self.pred_vis:
            self.vis_head=MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1 )

        #if not discriminator:
        self.use_infogail=False
        self.use_vae=False

        if self.use_infogail and not discriminator:
            self.k1_dim=1
            self.k2_dim=2

            self.k_dim=self.k1_dim*self.k2_dim
            self.latent_embed=nn.Embedding(self.k_dim, hidden_dim)
           # self.latent_embed=RoleHead(self.hidden_dim, self.k_dim)


        if self.use_vae and not discriminator:
            self.k_dim=32
            self.use_dicrete=True

            if self.use_dicrete:
                self.latent_embed=nn.Embedding(self.k_dim, hidden_dim) #MLPLayer(self.k_dim,hidden_dim,hidden_dim)#nn.Embedding(self.k_dim, hidden_dim)
            else:
                self.latent_embed=MLPLayer(self.k_dim,hidden_dim,hidden_dim)

        self.pred_entry=token_processor.pred_entry & (not discriminator)

        if self.pred_entry:
            self.entry_decoder = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=token_processor.n_token_entry
                    )
            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=32
                    )

        self.pred_exit=token_processor.pred_exit & (not discriminator)

        # if self.pred_exit:
        #     self.exit_decoder = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=2)

        self.pred_col=False
        self.use_sign_dist=False

        self.token_processor= token_processor
        self.discriminator=discriminator
        self.apply(weight_init)

    def predict_agent(self, sampled_idx,token_mask, mask ,pos_a,head_a,tokenized_agent, map_feature,light_idx,mask_lg, n_current=0,latent_z=None,abs_time=None):

        #pos_a=torch.round(pos_a*10)/10
        #head_a=torch.round(head_a*10)/10

        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        # if self.discriminator:
        #     feat_a_token=tokenized_agent["feat_a_token"]
        #     agent_token_emb=tokenized_agent["agent_token_emb"]
        # else:
        # ! get agent token embeddings
        if "mean_speed" in tokenized_agent.keys():
            mean_speed = tokenized_agent["mean_speed"]
        else:
            mean_speed = None

        feat_a_token,agent_token_emb = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            mean_speed=mean_speed,
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
            token_mask=token_mask,
            batch_idx=tokenized_agent['batch'],
            goal_pos=tokenized_agent["goal_pos"],
            goal_mask=tokenized_agent["goal_mask"],
            abs_time=abs_time,
        )  # feat_a: [n_agent, n_step, hidden_dim]

        pos_a = pos_a[:, -n_step:]

        if latent_z is not None:
            latent_embedding=self.latent_embed(latent_z)#[:,n_current:n_current+n_step]
            feat_a_token=feat_a_token+latent_embedding

        # feat_a_t, head_vector_a=self.temporal_embed(feat_a_token, pos_a, head_a,head_vector_a,n_agent, n_step, n_current, mask)
        feat_a_t=feat_a_token

        if self.training or self.discriminator or self.target_net:
            n_step=n_step-self.start_step
            pos_a=pos_a[:,-n_step:]
            head_a=head_a[:,-n_step:]
            head_vector_a=head_vector_a[:,-n_step:]
            #agent_token_emb=agent_token_emb[:,-n_step:]
            feat_a_t=feat_a_t[:,-n_step:]
            feat_a_token=feat_a_token[:,-n_step:]

        mask_a=mask[:,-n_step:]
        batch_a=tokenized_agent["batch"]
        batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

        if ("train_mask" in tokenized_agent.keys() and self.training) or self.target_net:
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        if n_step>1:
            batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step - 1).reshape(-1,n_agent).transpose(
                0, 1)

            all_features=[]
            next_all_features=[]
            for feature in [feat_a_t,feat_a_token,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat]:
                all_features.append(feature[:, :-1])
                next_all_features.append(feature[:, 1:])  # .clone()[:,1:]

            next_all_features.append(batch_s)
            all_features.append(batch_s)

            if self.discriminator:
                if self.interative_decoder.reward_shaping:
                    batch_s = build_batch(batch_a, tokenized_agent["num_graphs"],n_step).reshape(-1, n_agent).transpose(
                        0, 1)

                    all_features=[feat_a_t,feat_a_token,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s,agent_token_emb[:,2:],sampled_idx[:,2:]]
                elif self.interative_decoder.state_action:
                    all_features.extend([agent_token_emb[:, 2:], sampled_idx[:, 2:]])
                else:
                    all_features=next_all_features
                    all_features.extend([None,None])
            else:
                if not (self.training or self.target_net):
                    all_features=next_all_features

                all_features.extend([agent_token_emb[:,2:],sampled_idx[:,2:]])
        else:
            batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(
                0, 1)

            all_features=[feat_a_t,feat_a_token,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s,None,None]

        next_token_logits,feat_a,rewards,weight,edge_index_a2a=self.interative_decoder(all_features,map_feature,train_mask,n_current)

        if self.pred_entry:

            entry_logit=self.entry_decoder(feat_a)

            if self.training:
                entry_idx=tokenized_agent["entry_idx"][:,self.start_step+1:].transpose(0, 1).flatten(0, 1)[mask_a[:,:-1].transpose(0, 1).flatten(0, 1)]
                entry_mask=(entry_idx<self.token_processor.n_token_entry - 1 )
                entry_local=self.token_processor.entry_pos_token[entry_idx[entry_mask]]

                feat_new=torch.cat([entry_local,feat_a[entry_mask]],dim=-1)
                head_logit=self.entry_head_decoder(feat_new)

                entry_logit=(entry_logit,head_logit)
        else:
            entry_logit=None


        # if self.pred_exit:
        #     exit_logit=self.exit_decoder(feat_a)
        # else:
        exit_logit=None

        return next_token_logits,edge_index_a2a,rewards,weight,entry_logit,exit_logit,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            post_sampling=False
    ) :
        light_idx = tokenized_agent["light_idx"].clone()

        if "next_token_logits" not in tokenized_agent.keys() and len(light_idx):
            random_light = torch.randint(low=0, high=self.light_type, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            light_idx[random_mask] = random_light[random_mask]

        mask_lg=light_idx<self.light_type

        if self.use_infogail or self.use_vae:
            if "latent_z" not in tokenized_agent.keys():
                batch_idx = tokenized_agent['batch']
                latent_z1 = torch.randint(low=0, high=self.k1_dim, size=(max(batch_idx) + 1, 1), device=batch_idx.device)
                latent_z1 = latent_z1[batch_idx] * self.k2_dim
                latent_z = torch.randint(low=0, high=self.k2_dim, size=(len(batch_idx), 1), device=batch_idx.device)

                latent_z = latent_z1 + latent_z

                tokenized_agent["latent_z"] = latent_z
        else:
            tokenized_agent["latent_z"]=None

        next_token_logits,edge_index_a2a,rewards,agent_token_emb,entry_logit,exit_logit,feat_a= self.predict_agent(tokenized_agent["sampled_idx"],
                                                                                tokenized_agent["token_mask"],
                                                                                tokenized_agent["valid_mask"],
                                                                                tokenized_agent["sampled_pos"],
                                                                                tokenized_agent["sampled_heading"] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                light_idx,
                                                                                mask_lg,
                                                                                latent_z=tokenized_agent["latent_z"],
                                                                                abs_time=tokenized_agent["abs_time"]
                                                                                                     )

        tokenized_agent["next_token_logits"] = next_token_logits
        tokenized_agent["edge_index_a2a"] = edge_index_a2a
        tokenized_agent["feat_a"] = feat_a
        tokenized_agent["entry_logit"] = entry_logit
        tokenized_agent["exit_logit"] = exit_logit
        next_map_token_logits=None

        return {
            "edge_index_a2a":edge_index_a2a,
            "exit_logit":exit_logit,
            "entry_logit":entry_logit,
            "goal_q":None,
            "agent_q": next_token_logits,            # action that goes from [(10->15), ..., (85->90)]
            'next_map_token_logits':next_map_token_logits
         }

    def autoregressive_agent(self, tokenized_agent, map_feature,current_step,max_step,post_sampling):

        gt_valid=tokenized_agent["valid_mask"].clone()
        gt_sampled_idx=tokenized_agent["sampled_idx"].clone()
        gt_pos=tokenized_agent["sampled_pos"].clone()
        gt_head=tokenized_agent["sampled_heading"].clone()

        sampled_idx=gt_sampled_idx[:, :current_step]
        mask = gt_valid[:, :current_step]
        pos_a = gt_pos[:, :current_step]
        head_a = gt_head[:, :current_step]

        #token_traj=tokenized_agent["token_traj"]
        token_traj_all = tokenized_agent["token_traj_all"]
        light_idx = tokenized_agent["light_idx"][:, :current_step].clone()
        batch = tokenized_agent['batch']

        if self.token_processor.use_time:
            abs_time=tokenized_agent["abs_time"][:, :current_step].clone()
        else:
            abs_time=gt_valid[:, :current_step]

        mask_lg=light_idx<self.light_type

        n_agent = sampled_idx.shape[0]

        present_mask=mask.any(-1)

        pred_traj_10hz = []
        pred_head_10hz = []

        if self.use_infogail or self.use_vae:
            if "latent_z" in tokenized_agent.keys():
                latent_z = tokenized_agent["latent_z"]
            else:
                latent_z1 = torch.randint(low=0, high=self.k1_dim, size=(max(batch) + 1, 1)).to(batch.device)
                latent_z1 = latent_z1[batch] * self.k2_dim
                latent_z = torch.randint(low=0, high=self.k2_dim, size=(len(batch), 1), device=batch.device)

                latent_z=latent_z1+latent_z
        else:
            latent_z=None

        token_mask=tokenized_agent["token_mask"][:, :current_step].clone()

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None and not self.use_diffusion:

                    # if tokenized_agent["proposal"] is not None:
                    #     proposal=tokenized_agent["proposal"][:, :1]#[current_mask][keep_mask]
                    # next_token_logits=torch.zeros([20,len(type),self.token_processor.n_token_agent],device=sampled_idx.device)
                    #
                    # next_token_logits[gt_valid[:,:-1].transpose(0, 1)]= tokenized_agent["next_token_logits"]
                    #
                    # next_token_logits1=next_token_logits[0][gt_valid[:,0]]
                    a_num=torch.sum(mask[:,t-1])

                    next_token_logits=tokenized_agent["next_token_logits"][:a_num]

                    if tokenized_agent["entry_logit"] is not None:
                        exit_logit=tokenized_agent["exit_logit"][:a_num]
                    else:
                        exit_logit=None


                    if tokenized_agent["entry_logit"] is not None:

                        entry_logit=tokenized_agent["entry_logit"][:,:1]
                    else:
                        entry_logit=None


                    feat_a = tokenized_agent["feat_a"][:a_num]

                    if self.use_roformer:
                        self.a_t_roformer.attn.kv_caching(self.agent_hist, current_step)
                        if self.pred_light and not self.light_encoder.share:
                            lg_num = tokenized_agent["pad_pos_lg"].shape[1]
                            self.light_encoder.lg_t_roformer.attn.kv_caching(self.light_hist, current_step * lg_num)
                    else:
                        self.feat_a_token_cache = self.feat_a_token_cache[:, :current_step]
                        self.pos_cache = self.pos_cache[:, :current_step]
                        self.head_cache = self.head_cache[:, :current_step]

                    # self.a_t_roformer.attn.cached_k=self.a_t_roformer.attn.cached_k[current_mask][keep_mask]
                    # self.a_t_roformer.attn.cached_v=self.a_t_roformer.attn.cached_v[current_mask][keep_mask]
                else:
                    if self.use_roformer:
                        self.a_t_roformer.attn.caching=True
                        if self.pred_light and not self.light_encoder.share:
                            self.light_encoder.lg_t_roformer.attn.caching=True

                    next_token_logits,next_light_logits,_,_,entry_logit,exit_logit,feat_a = self.predict_agent(sampled_idx,token_mask, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,light_idx,mask_lg,0,latent_z,abs_time)


            else:
                next_token_logits, next_light_logits, _, _, entry_logit,exit_logit, feat_a = self.predict_agent(
                    sampled_idx[:, -1:], token_mask[:, -1:], mask[:, - self.agent_hist:],
                    pos_a[:, -2:], head_a[:, -1:], tokenized_agent, map_feature, light_idx[:, -1:],
                    mask_lg[:, -self.light_hist:], t - 1, latent_z,abs_time[:, -1:])

            if post_sampling:
                next_token_idx=gt_sampled_idx[:,t]
            else:
                if self.use_diffusion:

                    dist=torch.linalg.norm(next_token_logits.reshape(-1,1,5,4,2)[:,:,-1] - token_traj,dim=-1).mean(-1)

                    next_token_idx=torch.argmin(dist,dim=1)
                else:
                    next_token_idx = torch.zeros_like(sampled_idx[:, -1])
                    if len(next_token_logits):
                        next_token_idx[mask[:, -1]] = Categorical(logits=next_token_logits / self.alpha).sample()

                    #use_gt_exit
                    # gt_exit_mask= gt_valid[:, t-1] & ~gt_valid[:, t]
                    #
                    # next_token_idx[mask[:, -1]] = Categorical(logits=next_token_logits[:,:-1] / self.alpha).sample()
                    #
                    # next_token_idx[gt_exit_mask] =token_traj_all.shape[1]

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            if self.token_processor.pred_exit:
                if exit_logit is None:
                    exit_mask=next_token_idx==token_traj_all.shape[1]
                    next_token_idx=torch.clip(next_token_idx,0,token_traj_all.shape[1]-1)
                else:
                    exit_mask = torch.zeros_like(mask[:, -1])
                    exit_mask[mask[:, -1]]= Categorical(logits=exit_logit ).sample().to(torch.bool)

            if not self.pred_all_res:
                next_token_traj_all = token_traj_all[torch.arange(n_agent), next_token_idx]

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2)[...,:2],  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, -1,:2],  # [n_agent, 2]
                head_now=head_a[:, -1],  # [n_agent]
            )[0].view(*next_token_traj_all.shape[:-1],2)

            if self.token_processor.use_bird:
                token_traj_global_z=next_token_traj_all[:,:,0,2:]+pos_a[:, -1:,2:]

                pred_traj=torch.cat([token_traj_global[:,:,0], token_traj_global_z], dim=-1)

                _invalid_mask=~mask[:,-1] | exit_mask

                pred_traj[_invalid_mask]=10000

                pos_a_next   = pred_traj[:,-1]
                diff_xy_next = pred_traj[:, -1, :2] - pred_traj[:, -2, :2]
            else:
                if "gt_z_raw" in tokenized_agent.keys():
                    pred_traj = token_traj_global[:, :].mean(2)
                    pred_traj_10hz.append(pred_traj)
                    diff_xy = token_traj_global[:, :, 0] - token_traj_global[:, :, 3]
                    pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
                    pred_head_10hz.append(pred_head)

                pos_a_next = token_traj_global[:, -1].mean(dim=1)
                diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]

            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            if self.token_processor.use_bird:

                if entry_logit is not None:
                    entry_token_idx = Categorical(logits=entry_logit).sample()

                    entry_mask =entry_token_idx < self.token_processor.n_token_agent-1

                    entry_agent_mask = torch.zeros_like(present_mask)

                    if entry_mask.any():

                        new_agent_mask= torch.zeros_like(mask[:, -1])

                        new_agent_mask[torch.nonzero(mask[:,-1])[entry_mask]]=True

                        entry_local_traj = self.token_processor.entry_pos_token[entry_token_idx[entry_mask]]

                        new_xy = transform_to_global(
                            entry_local_traj[:, None, :2],
                            None,
                            pos_a[:, -1, :2][new_agent_mask],
                            head_a[:, -1][new_agent_mask],
                        )[0][:,0]

                        new_z = pos_a[:, -1, 2][new_agent_mask] + entry_local_traj[:, 2]

                        new_pos = torch.cat([new_xy, new_z[:, None]], dim=1)

                        feat_new = torch.cat([entry_local_traj, feat_a[entry_mask]], dim=-1)
                        head_logit = self.entry_head_decoder(feat_new)

                        entry_head_idx = Categorical(logits=head_logit).sample()

                        new_head=(entry_head_idx-16)/16*np.pi

                        new_agent_batch=batch[new_agent_mask]

                        non_present_mask = ~present_mask

                        unique_batches = batch.unique()
                        for b in unique_batches:
                            batch_mask = new_agent_batch==b
                            n_new = int(batch_mask.sum())
                            if n_new == 0:
                                continue

                            non_present_idx = torch.nonzero((batch == b) & non_present_mask, as_tuple=False).squeeze(1)
                            if len(non_present_idx) == 0:
                                continue

                            chosen = non_present_idx[:n_new]

                            pos_a_next[chosen] = new_pos[batch_mask][:len(chosen)]
                            head_a_next[chosen] = new_head[batch_mask][:len(chosen)]

                            entry_agent_mask[chosen]=True
                else:
                    entry_agent_mask = ~present_mask & gt_valid[:, t]

                    pos_a_next[entry_agent_mask] = gt_pos[entry_agent_mask, t]

                    head_a_next[entry_agent_mask] = gt_head[entry_agent_mask, t]

                pred_traj[entry_agent_mask,-1]=pos_a_next[entry_agent_mask]

                pred_traj_10hz.append(pred_traj)

                if self.token_processor.pred_exit:
                    next_mask = (mask[:, -1] & ~exit_mask) | entry_agent_mask
                else:
                    next_mask = gt_valid[:, t]

                present_mask = present_mask | next_mask
            else:
                next_mask = mask[:, -1]

            next_token_mask = mask[:, -1] & next_mask

            mask = torch.cat([mask, next_mask[:, None]], dim=1)

            token_mask = torch.cat([token_mask, next_token_mask[:, None]], dim=1)

            abs_time = torch.cat([abs_time, abs_time[:, -1:] + self.shift], dim=1)


            pos_a_next=pos_a_next.masked_fill(~next_mask.unsqueeze(1), 0)
            head_a_next=head_a_next.masked_fill(~next_mask, 0)

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

        if self.use_roformer:

            self.a_t_roformer.attn.kv_caching(0)
            if self.pred_light and not self.light_encoder.share:
                self.light_encoder.lg_t_roformer.attn.kv_caching(0)


        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "token_mask":token_mask,
            "sampled_idx": sampled_idx,  # [n_agent, 18]
            "gt_idx": sampled_idx,
            "light_idx": light_idx,
            "abs_time" :abs_time
        }

        if len(pred_traj_10hz):
            out_dict["pred_traj_10hz"] = torch.cat(pred_traj_10hz, dim=1)


        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_head_10hz"] =torch.cat(pred_head_10hz, dim=1)
            out_dict["pred_z_10hz"] = tokenized_agent["gt_z_raw"].unsqueeze(1) .expand(-1, out_dict["pred_traj_10hz"].shape[1])

        # pred_mask = (out_dict["pred_traj_10hz"]  != 10000).any(-1)
        #
        # pred_mask=pred_mask[:,4::5]
        #
        # gt_mask=tokenized_agent["valid_mask"][:, 1:]
        #
        # print(torch.all(pred_mask==gt_mask))
        #
        return out_dict

    def inference(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            sampling_scheme=None,
            post_sampling=False,
            step_current_10hz=None,
            n_step_future_10hz=None,
    ) -> Dict[str, torch.Tensor]:
        if n_step_future_10hz is None:
            n_step_future_10hz = self.num_future_steps  # 80
        if step_current_10hz is None:
            step_current_10hz = self.num_historical_steps - 1  # 10

        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_2hz = step_current_10hz // self.shift  # 2

        out_dict=self.autoregressive_agent(tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz,post_sampling)

        return out_dict

# if len(next_light_logits):
#
#     next_light_idx = Categorical(logits=next_light_logits[:, -1] / self.alpha).sample()
#
#     light_idx = torch.cat([light_idx, next_light_idx[:, None]], dim=1)
#
# elif self.use_light:
#     light_idx = tokenized_agent["light_idx"][:,t:t+1]


# if self.pred_vis:
#     vis=torch.rand_like(visibility[:,-1:,0])<torch.sigmoid(visibility[:,-1:,0])
#     vis=vis_mask[:,-1:] & vis
#     vis_mask=torch.cat([vis_mask,vis],dim=1)

# if self.pred_goal:
#     next_goal_idx=Categorical(logits=next_goal_logits[:, -1] / self.alpha).sample()
#
#     goal_idx = torch.cat([goal_idx, next_goal_idx[:, None]], dim=1)
