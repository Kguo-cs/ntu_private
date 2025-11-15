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

    def predict_agent(self, sampled_idx,token_mask, mask_a ,pos_a,head_a,tokenized_agent, map_feature, n_current=0,abs_time=None):

        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        if not self.token_processor.use_token:
            token_mask=mask_a

        feat_a_token,agent_token_emb = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
            token_mask=token_mask,
            batch_idx=tokenized_agent['batch'],
            goal_pos=tokenized_agent["goal_pos"],
            goal_mask=tokenized_agent["goal_mask"],
            abs_time=abs_time,
        )

        pos_a = pos_a[:, -n_step:]
        batch_a=tokenized_agent["batch"]
        batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

        if ("train_mask" in tokenized_agent.keys() and self.training) or self.target_net:
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(0, 1)

        all_features=[feat_a_token,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s]

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

        next_token_logits,edge_index_a2a,rewards,agent_token_emb,entry_logit,exit_logit,feat_a= self.predict_agent(tokenized_agent["sampled_idx"][:,:-1],
                                                                                tokenized_agent["token_mask"][:,:-1],
                                                                                tokenized_agent["valid_mask"][:,:-1],
                                                                                tokenized_agent["sampled_pos"][:,:-1],
                                                                                tokenized_agent["sampled_heading"][:,:-1] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                abs_time=tokenized_agent["abs_time"][:,:-1]
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
        abs_time = tokenized_agent["abs_time"][:, :current_step].clone()
        token_mask=tokenized_agent["token_mask"][:, :current_step].clone()
        token_traj_all = tokenized_agent["token_traj_all"]
        batch = tokenized_agent['batch']
        pred_mask =tokenized_agent["pred_mask"]

        sampled_idx=gt_sampled_idx[:, :current_step]
        mask = gt_valid[:, :current_step]
        pos_a = gt_pos[:, :current_step]
        head_a = gt_head[:, :current_step]

        n_agent = sampled_idx.shape[0]

        present_mask=mask.any(-1)
        next_mask=mask[:, -1]

        pred_traj_10hz = []
        pred_head_10hz = []

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None and not self.use_diffusion:
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
                        self.interative_decoder.pos_cache = self.interative_decoder.pos_cache[:, :current_step]
                        self.interative_decoder.head_cache = self.interative_decoder.head_cache[:, :current_step]
                        self.interative_decoder.mask_cache = self.interative_decoder.mask_cache[:, :current_step]
                        self.interative_decoder.head_vector_cache = self.interative_decoder.head_vector_cache[:, :current_step]

                       # if self.token_processor.use_bird:
                        self.interative_decoder.feat_a_cache = self.interative_decoder.feat_a_cache[:current_step]
                        # else:
                        #     self.interative_decoder.feat_a_cache = self.interative_decoder.feat_a_cache[:, :current_step]
                else:
                    next_token_logits,_,_,_,entry_logit,exit_logit,feat_a = self.predict_agent(sampled_idx,token_mask, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,0,abs_time)
            else:
                next_token_logits, _, _, _, entry_logit,exit_logit, feat_a = self.predict_agent(
                    sampled_idx[:, -1:], token_mask[:, -1:], mask[:, -1:],
                    pos_a[:, -2:], head_a[:, -1:], tokenized_agent, map_feature, t - 1,abs_time[:, -1:])

            if post_sampling:
                next_token_idx=gt_sampled_idx[:,t]
            else:
                if self.use_diffusion:

                    dist=torch.linalg.norm(next_token_logits.reshape(-1,1,5,4,2)[:,:,-1] - token_traj,dim=-1).mean(-1)

                    next_token_idx=torch.argmin(dist,dim=1)
                else:
                    next_token_idx = torch.zeros_like(sampled_idx[:, -1])
                    if len(next_token_logits):
                        if self.pred_exit and pred_mask is not None:
                            next_token_logits[pred_mask[next_mask],-1] = -10000
                        next_token_idx[next_mask] = Categorical(logits=next_token_logits / self.alpha).sample()

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
                    exit_mask = torch.zeros_like(next_mask)
                    exit_mask[next_mask]= Categorical(logits=exit_logit ).sample().to(torch.bool)

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

                _invalid_mask = ~next_mask | exit_mask

                pred_traj[_invalid_mask] = 10000

                pos_a_next   = pred_traj[:,-1]
                diff_xy_next = pred_traj[:, -1, :2] - pred_traj[:, -2, :2]
            else:
                if "gt_z_raw" in tokenized_agent.keys():
                    pred_traj = token_traj_global[:, :].mean(2)

                    if self.pred_exit:
                        _invalid_mask = ~next_mask | exit_mask
                        pred_traj[_invalid_mask] = 10000

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
                    next_mask = (next_mask & ~exit_mask) | entry_agent_mask
                else:
                    next_mask = gt_valid[:, t]

                present_mask = present_mask | next_mask
            else:
                if self.token_processor.pred_exit:
                    next_mask = next_mask & ~exit_mask
                else:
                    next_mask = next_mask

            next_token_mask = mask[:, -1] & next_mask

            mask = torch.cat([mask, next_mask[:, None]], dim=1)

            token_mask = torch.cat([token_mask, next_token_mask[:, None]], dim=1)

            abs_time = torch.cat([abs_time, abs_time[:, -1:] + self.shift], dim=1)

            pos_a_next=pos_a_next.masked_fill(~next_mask.unsqueeze(1), 0)
            head_a_next=head_a_next.masked_fill(~next_mask, 0)

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "token_mask":token_mask,
            "sampled_idx": sampled_idx,  # [n_agent, 18]
        }

        if len(pred_traj_10hz):
            out_dict["pred_traj_10hz"] = torch.cat(pred_traj_10hz, dim=1)

        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_head_10hz"] =torch.cat(pred_head_10hz, dim=1)
            out_dict["pred_z_10hz"] = tokenized_agent["gt_z_raw"].unsqueeze(1) .expand(-1, out_dict["pred_traj_10hz"].shape[1])

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
