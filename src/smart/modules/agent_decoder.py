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

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.utils import (
    transform_to_global,
    weight_init,
)
from torch.distributions import Categorical
from .build_edge import build_batch
from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder
import numpy as np
from src.smart.modules.entry_encoder import EntryDecoder


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
            dis_weight,
            dist_decay,
            reward_weight,
            reward_decay,
            discriminator=False
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps

        self.shift = token_processor.shift

        self.agent_token_embedding=AgentTokenEncoder(hidden_dim,num_freq_bands,token_processor,discriminator)

        self.interative_decoder = InterativeDecoder(hidden_dim,num_historical_steps,num_future_steps,time_span,
                                                    pl2a_radius,a2a_radius,num_freq_bands,
                                                    num_layers,num_heads,head_dim,
                                                    dropout,hist_drop_prob,n_token_agent,
                                                    pt2a_neighbor,a2a_neighbor,
                                                    token_processor,
                                                    dis_weight,
                                                    dist_decay,
                                                    reward_weight,
                                                    reward_decay,
                                                    discriminator=discriminator
                                                    )

        self.start_step=self.num_historical_steps//self.shift-1
        self.t_num_layers = 1
        self.agent_hist = self.time_span // self.shift*self.t_num_layers
        self.alpha = alpha


        self.pred_entry=token_processor.pred_entry & (not discriminator)
        self.pred_exit=token_processor.pred_exit & (not discriminator)

        self.new_agent=True

        if self.pred_entry:
            self.entry_decoder=EntryDecoder(hidden_dim,num_heads,num_freq_bands,token_processor,self.start_step)

        self.token_processor= token_processor
        self.discriminator=discriminator
        self.apply(weight_init)


    def predict_agent(self, sampled_idx,token_mask, mask_a ,pos_a,head_a,tokenized_agent, map_feature, n_current=0,abs_time=None):

        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        # if self.discriminator:#not self.token_processor.use_token:
        #     token_mask=None#mask_a

        feat_a_token,agent_token_emb,counter_feat_a = self.agent_token_embedding(
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
        if self.training:
            feat_a_token=feat_a_token[:,:-1]
            mask_a=mask_a[:,:-1]
            pos_a=pos_a[:,:-1]
            head_a=head_a[:,:-1]
            head_vector_a=head_vector_a[:,:-1]
            n_step=n_step-1

        batch_a=tokenized_agent["batch"]
        batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

        if ("train_mask" in tokenized_agent.keys() and self.training):
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(0, 1)

        all_features=[feat_a_token,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s]

        next_token_logits,feat_a,rewards,weight,edge_index_a2a=self.interative_decoder(all_features,counter_feat_a,agent_token_emb,map_feature,train_mask,n_current,tokenized_agent["pred_mask"])

        entry_logit = exit_logit=None

        if self.training:
            # feat_a=feat_a+agent_token_emb[:,1+self.start_step:].transpose(0, 1).flatten(0, 1)[mask_a[:,self.start_step:].transpose(0, 1).flatten(0, 1)]

            if self.pred_entry:
                entry_logit= self.entry_decoder(feat_a,mask_a,pos_a,head_a,tokenized_agent ,edge_index_a2a)

        return next_token_logits,edge_index_a2a,rewards,weight,entry_logit,exit_logit,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            post_sampling=False
    ) :

        next_token_logits,edge_index_a2a,rewards,agent_token_emb,entry_logit,exit_logit,feat_a= self.predict_agent(tokenized_agent["sampled_idx"],#[:,:-1],
                                                                                tokenized_agent["token_mask"],#[:,:-1],
                                                                                tokenized_agent["valid_mask"],#[:,:-1],
                                                                                tokenized_agent["sampled_pos"],#[:,:-1],
                                                                                tokenized_agent["sampled_heading"],#[:,:-1] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                abs_time=tokenized_agent["abs_time"],#[:,:-1]
                                                                                                     )

        tokenized_agent["next_token_logits"] = next_token_logits
        tokenized_agent["entry_logit"] = entry_logit
        tokenized_agent["exit_logit"] = exit_logit
        tokenized_agent["feat_a"] = feat_a

        return {
            "exit_logit":exit_logit,
            "entry_logit":entry_logit,
            "agent_q": next_token_logits
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
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None:
                    a_num=next_mask.sum()

                    next_token_logits=tokenized_agent["next_token_logits"][:a_num]

                    if tokenized_agent["entry_logit"] is not None:
                        exit_logit=tokenized_agent["exit_logit"][:a_num]
                    else:
                        exit_logit=None

                    if tokenized_agent["entry_logit"] is not None:
                        entry_logit=tokenized_agent["entry_logit"][:,:1]
                    else:
                        entry_logit=None

                    #feat_a = tokenized_agent["feat_a"][:a_num]

                    self.interative_decoder.pos_cache = self.interative_decoder.pos_cache[:, :current_step]
                    self.interative_decoder.head_cache = self.interative_decoder.head_cache[:, :current_step]
                    self.interative_decoder.mask_cache = self.interative_decoder.mask_cache[:, :current_step]
                    self.interative_decoder.head_vector_cache = self.interative_decoder.head_vector_cache[:,
                                                                :current_step]

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

            next_token_idx = torch.zeros_like(sampled_idx[:, -1])
            if len(next_token_logits):
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
                    exit_mask=next_token_idx==self.token_processor.n_token_agent-1
                    next_token_idx=torch.clip(next_token_idx,0,self.token_processor.n_token_agent-2)
                else:
                    exit_mask = torch.zeros_like(next_mask)
                    exit_mask[next_mask]= Categorical(logits=exit_logit ).sample().to(torch.bool)

            if len(token_traj_all.shape)==4:
                next_token_traj_all = token_traj_all[next_token_idx]
            else:
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

                    diff_xy = token_traj_global[:, :, 0] - token_traj_global[:, :, 3]
                    pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
                    pred_head_10hz.append(pred_head)

                pos_a_next = token_traj_global[:, -1].mean(dim=1)
                diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]

            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            if self.new_agent:

                # agent_token_emb=self.agent_token_embedding.get_embedding(next_token_idx[next_mask][:,None],tokenized_agent["type"][next_mask],~exit_mask[next_mask][:,None])
                #
                # feat_a = feat_a + agent_token_emb[:,0]

                entry_logit= self.entry_decoder(feat_a,next_mask[:,None],pos_a[:, -1:], head_a[:, -1:],tokenized_agent)

                if entry_logit is not None:

                    if len(entry_logit)!=2:
                        non_present_mask = ~present_mask
                        entry_agent_mask = torch.zeros_like(present_mask)

                        unique_batches = batch.unique()
                        for b in unique_batches:
                            entry_agent =entry_logit[b]

                            entry_agent=entry_agent[entry_agent[:,0]!=0]
                            n_new = len(entry_agent)
                            if n_new == 0:
                                continue

                            non_present_idx = torch.nonzero((batch == b) & non_present_mask, as_tuple=False).squeeze(1)
                            if len(non_present_idx) == 0:
                                print('full entry',t,b)
                                continue

                            chosen = non_present_idx[:n_new]

                            pos_a_next[chosen] = entry_agent[:len(chosen),:3]
                            head_a_next[chosen] = entry_agent[:len(chosen),-1]

                            entry_agent_mask[chosen] = True

                    else:
                        entry_mask, entry_local_traj=entry_logit

                        entry_agent_mask = torch.zeros_like(present_mask)

                        if entry_mask.any():

                            new_agent_mask= torch.zeros_like(next_mask)

                            new_agent_mask[torch.nonzero(next_mask)[entry_mask]]=True

                            present_head=head_a[:, -1][new_agent_mask]

                            present_pos=pos_a[:, -1][new_agent_mask]

                            global_xy,global_head = transform_to_global(
                                entry_local_traj[:, None, :2],
                                entry_local_traj[:, None, -1],
                                present_pos[:, :2],
                                present_head,
                            )

                            new_z = present_pos[:, 2:self.entry_decoder.pos_dim] + entry_local_traj[:, 2:self.entry_decoder.pos_dim]

                            new_pos = torch.cat([global_xy[:,0], new_z], dim=1)

                            new_head=wrap_angle(global_head[:,0])

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

                if self.token_processor.pred_exit:
                    next_mask = (next_mask & ~exit_mask) | entry_agent_mask
                else:
                    next_mask = gt_valid[:, t]

                present_mask = present_mask | next_mask
            else:
                if self.token_processor.pred_exit:
                    next_mask = next_mask & ~exit_mask

            pred_traj_10hz.append(pred_traj)

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

            if self.token_processor.pred_entry :
                current_valid=gt_valid[:, current_step-1]
                out_dict["pred_traj_10hz"]=out_dict["pred_traj_10hz"][current_valid]
                out_dict["pred_head_10hz"]=out_dict["pred_head_10hz"][current_valid]
                out_dict["pred_z_10hz"]=out_dict["pred_z_10hz"][current_valid]

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
