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
    weight_init
)
from torch.distributions import Categorical
from src.smart.utils.edge_utils import build_batch
from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder
import numpy as np
from src.smart.modules.entry_encoder import EntryDecoder
from src.smart.modules.inf_encoder import InfGenAgentDecoder
from src.smart.modules.initial_decoder import InitDecoder
import math
import random
import torch.nn.functional as F

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
        self.token_processor= token_processor
        self.discriminator=discriminator

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

        self.alpha = alpha

        self.pred_entry=token_processor.pred_entry & (not discriminator)
        self.pred_exit=token_processor.pred_exit & (not discriminator)
        self.pred_init=token_processor.pred_init & (not discriminator)

        if token_processor.use_infgen:
            self.inf_decoder = InfGenAgentDecoder(attr_tokenizer=token_processor.attr_tokenizer )

        if self.pred_entry:
            self.entry_decoder=EntryDecoder(hidden_dim,num_heads,num_freq_bands,token_processor,self.start_step)

        self.learn_init=token_processor.learn_init

        self.learn_init_only=True

        self.token_initial=token_processor.token_initial

        self.learn_autoencoder=token_processor.learn_autoencoder

        self.use_gan=False

        if self.pred_init and self.learn_init:

            if self.token_initial:
                self.init_decoder=InitDecoder(hidden_dim,num_heads,num_freq_bands,token_processor)
            else:
                if self.use_gan:
                    from src.smart.gan.initial_gan import InitGAN
                    self.init_decoder=InitGAN(hidden_dim,num_heads,num_freq_bands,token_processor)
                else:
                    from src.smart.diffusion.initial_diffusion import InitDiffusion

                    self.init_decoder=InitDiffusion(hidden_dim,num_heads,num_freq_bands,token_processor)

    def predict_agent(self, sampled_idx,token_mask, mask_a ,pos_a,head_a,tokenized_agent, map_feature, n_current=0):

        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        feat_a_token,agent_token_emb,counter_feat_a = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask_a=mask_a,
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
            token_mask=token_mask,
            batch_idx=tokenized_agent['batch'],
            goal_pos=tokenized_agent["goal_pos"],
            goal_mask=tokenized_agent["goal_mask"],
            ego_mask=tokenized_agent["ego_mask"],
        )

        pos_a = pos_a[:, -n_step:]
        batch_a=tokenized_agent["batch"]
        batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

        if ("train_mask" in tokenized_agent.keys() and self.training):
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(0, 1)

        all_features=[pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s]

        next_token_logits,feat_a,rewards,weight,a2a_feature=self.interative_decoder(all_features,feat_a_token,agent_token_emb,map_feature,train_mask,n_current,tokenized_agent)

        entry_logit =None

        if self.pred_entry:
            entry_logit= self.entry_decoder(feat_a_token[-len(feat_a):],mask_a,pos_a,head_a,tokenized_agent)
        initial_logit=None

        return next_token_logits,a2a_feature,rewards,weight,entry_logit,initial_logit,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) :
        entry_logit = next_token_logits = mask_token_logit=pred_action_mask = None

        if self.learn_init:
            initial_logit = self.init_decoder(tokenized_agent)

            if self.init_decoder.use_perceptual_loss:
                gt_initial_pos,gt_initial_heading, gt_initial_idx, gt_initial_type,gt_initial_shape,sort_idx,non_ego=initial_logit[1]

                local_vel=tokenized_agent["local_vel"].clone()

                center_token_traj = tokenized_agent["token_traj"].mean(-2)

                sampled_idx = torch.linalg.norm(center_token_traj - local_vel[:, None]*0.5, dim=-1).argmin(-1)

                sampled_idx2=torch.cat([gt_initial_idx,sampled_idx],dim=0)  [:,None]

                batch2=torch.cat([tokenized_agent["batch"],tokenized_agent["batch"]+tokenized_agent["num_graphs"]],dim=0)

                initial_pos=tokenized_agent["initial_pos"].clone()

                initial_pos[non_ego]=initial_pos[non_ego][sort_idx]

                initial_heading=tokenized_agent["initial_heading"].clone()

                initial_heading[non_ego]=initial_heading[non_ego][sort_idx]

                sampled_pos2=torch.cat([initial_pos,gt_initial_pos],dim=0)  [:,None]

                sampled_heading2=torch.cat([initial_heading,gt_initial_heading],dim=0)  [:,None]

                initial_shape= tokenized_agent["initial_shape"]

                initial_shape[non_ego]=initial_shape[non_ego][sort_idx]

                tokenized_agent["shape"]=torch.cat([initial_shape,gt_initial_shape])

                tokenized_agent["batch"]=batch2

                tokenized_agent["type"]=torch.cat([gt_initial_type,gt_initial_type],dim=0)

                tokenized_agent["ego_mask"]=torch.cat([tokenized_agent["ego_mask"],tokenized_agent["ego_mask"]],dim=0)

                token_mask2=valid_mask2=torch.ones_like(tokenized_agent["ego_mask"]) [:,None]

                tokenized_agent["pred_mask"]=torch.ones_like(tokenized_agent["ego_mask"])

                token_logits,a2a_feature,rewards,agent_token_emb,entry_logit,_,feat_a_list= self.predict_agent(sampled_idx2,
                                                                                        token_mask2,
                                                                                        valid_mask2,
                                                                                        sampled_pos2,
                                                                                        sampled_heading2,
                                                                                        tokenized_agent,
                                                                                        map_feature,
                                                                                        abs_time=tokenized_agent["abs_time"]
                                                                                        )

                perception_Loss=0

                for feat_a in feat_a_list:
                    perception_Loss+=F.mse_loss(feat_a[:len(sampled_idx)], feat_a[len(sampled_idx):],reduction="mean")

                #perception_Loss=F.mse_loss(token_logits[:len(sampled_idx)], token_logits[len(sampled_idx):],reduction="mean")

                initial_logit=(perception_Loss+initial_logit[0],perception_Loss,initial_logit[2],initial_logit[3],initial_logit[4],initial_logit[5],initial_logit[6])


        else:
            next_token_logits,a2a_feature,rewards,agent_token_emb,entry_logit,initial_logit,feat_a= self.predict_agent(tokenized_agent["sampled_idx"][:,:-1],
                                                                                    tokenized_agent["token_mask"][:,:-1],
                                                                                    tokenized_agent["valid_mask"][:,:-1],
                                                                                    tokenized_agent["sampled_pos"][:,:-1],
                                                                                    tokenized_agent["sampled_heading"][:,:-1] ,
                                                                                    tokenized_agent,
                                                                                    map_feature,
                                                                                    abs_time=tokenized_agent["abs_time"][:,:-1]
                                                                                                         )

            tokenized_agent["next_token_logits"] = next_token_logits
            tokenized_agent["entry_logit"] = entry_logit
            tokenized_agent["initial_logit"] = initial_logit
            tokenized_agent["feat_a"] = feat_a

            if self.interative_decoder.mask_pred:
                action=tokenized_agent["sampled_idx"][:,1:].clone()
                action_mask=tokenized_agent["token_mask"][:,1:].clone() #n_agent,n_step
                state_mask=tokenized_agent["valid_mask"][:,:-1]

                # batch=tokenized_agent["batch"]
                #
                # n_batch=tokenized_agent["num_graphs"]
                #
                # mask_rate= torch.rand((n_batch,state_mask.shape[1]),device=action.device)
                #
                # rand=torch.rand_like(action.to(torch.float32))
                #
                # idx_to_mask_si= rand<mask_rate[batch]
                #
                # action_mask[idx_to_mask_si] =False
                #
                action=action[state_mask]
                # action_mask=action_mask[state_mask]
                #
                # pred_action_mask=~action_mask

                probs = torch.softmax(next_token_logits / self.alpha, dim=-1)

                max_probs=torch.amax(probs, dim=-1)

                # prob_sampled= torch.gather(probs,1,next_sampled.unsqueeze(-1)).squeeze(-1)

                pred_action_mask=max_probs<0.001#p<0.001,mask ratio 0.01

                target_valid = action_mask[state_mask]

                mask_token_logit=self.interative_decoder.pred_mask_logit( action, pred_action_mask, a2a_feature, target_valid, feat_a)

        return {
            "pred_action_mask":pred_action_mask,
            "mask_token_logit":mask_token_logit,
            "initial_logit":initial_logit,
            "entry_logit":entry_logit,
            "next_token_logits": next_token_logits
         }

    def autoregressive_agent(self, tokenized_agent, map_feature,current_step,max_step):
        gt_pos=tokenized_agent["sampled_pos"].clone()
        gt_head=tokenized_agent["sampled_heading"].clone()
        gt_valid=tokenized_agent["valid_mask"].clone()
        gt_sampled_idx=tokenized_agent["sampled_idx"].clone()
        sampled_idx=gt_sampled_idx[:, :current_step]

        token_traj_all = tokenized_agent["token_traj_all"]
        batch = tokenized_agent['batch']

        if gt_pos.shape[1]==gt_head.shape[1]:
            pos_a = gt_pos[:, :current_step]
        else:
            pos_a = gt_pos[:, :current_step+1]

        head_a = gt_head[:, :current_step]
        mask = gt_valid[:, :current_step]
        token_mask=tokenized_agent["token_mask"][:, :current_step].clone()

        if self.pred_init:
            current_step=1

            if self.learn_init:
                pos_a, head_a,sampled_idx,initial_speed=self.init_decoder( tokenized_agent)
                max_step=18
            else:
                pos_a = tokenized_agent["gt_initial_pos"]
                head_a= tokenized_agent["gt_initial_heading"]
                sampled_idx=tokenized_agent["gt_initial_idx"]
                initial_speed=tokenized_agent["gt_initial_idx"]

                max_step=16

            mask=torch.ones_like(mask[:, :current_step])
            if self.token_processor.pred_vel:
                token_mask=torch.ones_like(token_mask[:, :current_step])
            else:
                token_mask=torch.zeros_like(token_mask[:, :current_step])

        n_agent = sampled_idx.shape[0]

        present_mask=mask.any(-1)
        next_mask=mask[:, -1]

        pred_traj_10hz = []
        pred_head_10hz = []
        a_num = next_mask.sum()

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None:

                    next_token_logits=tokenized_agent["next_token_logits"][a_num*(current_step-1):a_num*current_step]

                    if tokenized_agent["entry_logit"] is not None:
                        entry_logit=tokenized_agent["entry_logit"][:,:1]
                    else:
                        entry_logit=None

                    self.interative_decoder.pos_cache = self.interative_decoder.pos_cache[:, :current_step]
                    self.interative_decoder.head_cache = self.interative_decoder.head_cache[:, :current_step]
                    self.interative_decoder.mask_cache = self.interative_decoder.mask_cache[:, :current_step]
                    self.interative_decoder.head_vector_cache = self.interative_decoder.head_vector_cache[:,
                                                                :current_step]
                    self.interative_decoder.feat_a_cache =[feat[:current_step] for feat in self.interative_decoder.feat_a_cache ]
                else:
                    next_token_logits,a2a_feature,_,_,entry_logit,init_logit,feat_a = self.predict_agent(sampled_idx,token_mask, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,0)
            else:
                next_token_logits, a2a_feature, _, _, entry_logit,init_logit, feat_a = self.predict_agent(
                    sampled_idx[:, -1:], token_mask[:, -1:], mask[:, -1:],
                    pos_a[:, -2:], head_a[:, -1:], tokenized_agent, map_feature, t - 1)

            next_sampled=Categorical(logits=next_token_logits / self.alpha).sample()

            if self.interative_decoder.mask_pred:
                probs = torch.softmax(next_token_logits / self.alpha, dim=-1)

                max_probs=torch.amax(probs, dim=-1)

                # prob_sampled= torch.gather(probs,1,next_sampled.unsqueeze(-1)).squeeze(-1)

                pred_action_mask=max_probs<1#p<0.001,mask ratio 0.01

               # print(pred_action_mask.float().mean())

                mask_logit=self.interative_decoder.pred_mask_logit(next_sampled.clone(),pred_action_mask,a2a_feature,torch.ones_like(pred_action_mask),feat_a)

                next_sampled[pred_action_mask]= Categorical(logits=mask_logit / self.alpha).sample()

            next_token_idx = torch.zeros_like(sampled_idx[:, -1])
            if len(next_token_logits):
                next_token_idx[next_mask] = next_sampled[-next_mask.sum():]

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            if self.token_processor.pred_exit:
                exit_mask = next_token_idx == self.token_processor.n_token_agent - 1
                next_token_idx = torch.clip(next_token_idx, 0, self.token_processor.n_token_agent - 2)

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

            if self.pred_entry or self.token_processor.use_bird:
               self.enter(entry_logit,present_mask,batch,next_mask,pos_a_next,head_a_next,pos_a,head_a,tokenized_agent,
              gt_valid,gt_pos,gt_head,t,pred_traj,exit_mask)
            else:
                if self.token_processor.pred_exit:
                    next_mask = next_mask & ~exit_mask

            if "gt_z_raw" in tokenized_agent.keys():
                pred_traj_10hz.append(pred_traj)

            next_token_mask = mask[:, -1] & next_mask

            mask = torch.cat([mask, next_mask[:, None]], dim=1)

            token_mask = torch.cat([token_mask, next_token_mask[:, None]], dim=1)

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

            if self.pred_init:
                out_dict["pred_traj_10hz"] = torch.cat([pos_a[:,:1],out_dict["pred_traj_10hz"]], dim=1)
                out_dict["pred_head_10hz"] = torch.cat([head_a[:,:1],out_dict["pred_head_10hz"]], dim=1)
                out_dict["initial_speed"]=initial_speed

            out_dict["pred_z_10hz"] = tokenized_agent["gt_z_raw"].unsqueeze(1) .expand(-1, out_dict["pred_traj_10hz"].shape[1])

            if self.token_processor.pred_entry :
                current_valid=gt_valid[:, current_step-1]

                new_xy=out_dict["pred_traj_10hz"][~current_valid]
                new_head=out_dict["pred_head_10hz"][~current_valid][:,:,None]
                new_shape=tokenized_agent["shape"][~current_valid][:,None].repeat(1,new_xy.shape[1],1)

                out_dict["new_agent"]=torch.cat([new_xy, new_head, new_shape], dim=-1)
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

        out_dict=self.autoregressive_agent(tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz)

        return out_dict

    def enter(self,entry_logit,present_mask,batch,next_mask,pos_a_next,head_a_next,pos_a,head_a,tokenized_agent,
              gt_valid,gt_pos,gt_head,t,pred_traj,exit_mask  ):
        # agent_token_emb=self.agent_token_embedding.get_embedding(next_token_idx[next_mask][:,None],tokenized_agent["type"][next_mask],~exit_mask[next_mask][:,None])
        #
        # feat_a = feat_a + agent_token_emb[:,0]
        #
        # entry_logit= self.entry_decoder(feat_a,next_mask[:,None],pos_a[:, -1:], head_a[:, -1:],tokenized_agent)

        if entry_logit is not None:

            if self.token_processor.autoregressive_entry:
                entry_agent, entry_type, entry_shape = entry_logit
                non_present_mask = ~present_mask
                entry_agent_mask = torch.zeros_like(present_mask)

                unique_batches = batch.unique()
                for b in unique_batches:
                    entry_agent_b = entry_agent[b]
                    entry_type_b = entry_type[b][entry_agent_b[:, 0] != 0]
                    entry_shape_b = entry_shape[b][entry_agent_b[:, 0] != 0]
                    entry_agent_b = entry_agent_b[entry_agent_b[:, 0] != 0]

                    n_new = len(entry_agent_b)
                    if n_new == 0:
                        continue

                    non_present_idx = torch.nonzero((batch == b) & non_present_mask, as_tuple=False).squeeze(1)
                    if len(non_present_idx) == 0:
                        print('full entry',  b)
                        continue

                    chosen = non_present_idx[:n_new]
                    entry_pos_b = entry_agent_b[:len(chosen), :pos_a_next.shape[-1]]
                    entry_heading_b = entry_agent_b[:len(chosen), -1]

                    if not self.token_processor.use_bird:
                        ego_mask = tokenized_agent["ego_mask"]

                        ego_pos = pos_a[:, -1][ego_mask][b]
                        ego_heading = head_a[:, -1][ego_mask][b]

                        entry_pos_b, entry_heading_b = transform_to_global(
                            entry_pos_b[None],
                            entry_heading_b[None],
                            ego_pos[None],
                            ego_heading[None]
                        )
                        entry_pos_b = entry_pos_b[0]
                        entry_heading_b = entry_heading_b[0]

                    pos_a_next[chosen] = entry_pos_b
                    head_a_next[chosen] = entry_heading_b
                    tokenized_agent["type"][chosen] = entry_type_b[:len(chosen)]
                    tokenized_agent["shape"][chosen] = entry_shape_b[:len(chosen)]

                    entry_agent_mask[chosen] = True

            else:
                entry_mask, entry_local_traj, entry_type, pred_shape = entry_logit

                entry_agent_mask = torch.zeros_like(present_mask)

                if entry_mask.any():

                    new_agent_mask = torch.zeros_like(next_mask)

                    new_agent_mask[torch.nonzero(next_mask)[entry_mask]] = True

                    present_head = head_a[:, -1][new_agent_mask]

                    present_pos = pos_a[:, -1][new_agent_mask]

                    global_xy, global_head = transform_to_global(
                        entry_local_traj[:, None, :2],
                        entry_local_traj[:, None, -1],
                        present_pos[:, :2],
                        present_head,
                    )

                    new_z = present_pos[:, 2:self.entry_decoder.pos_dim] + entry_local_traj[:,
                                                                           2:self.entry_decoder.pos_dim]

                    new_pos = torch.cat([global_xy[:, 0], new_z], dim=1)

                    new_head = wrap_angle(global_head[:, 0])

                    new_agent_batch = batch[new_agent_mask]

                    non_present_mask = ~present_mask

                    unique_batches = batch.unique()
                    for b in unique_batches:
                        batch_mask = new_agent_batch == b
                        n_new = int(batch_mask.sum())
                        if n_new == 0:
                            continue

                        non_present_idx = torch.nonzero((batch == b) & non_present_mask, as_tuple=False).squeeze(1)
                        if len(non_present_idx) == 0:
                            continue

                        chosen = non_present_idx[:n_new]

                        pos_a_next[chosen] = new_pos[batch_mask][:len(chosen)]
                        head_a_next[chosen] = new_head[batch_mask][:len(chosen)]
                        tokenized_agent["type"][chosen] = entry_type[batch_mask][:len(chosen)]
                        tokenized_agent["shape"][chosen] = pred_shape[batch_mask][:len(chosen)]

                        entry_agent_mask[chosen] = True
        elif self.token_processor.use_bird:
            entry_agent_mask = ~present_mask & gt_valid[:, t]

            pos_a_next[entry_agent_mask] = gt_pos[entry_agent_mask, t]

            head_a_next[entry_agent_mask] = gt_head[entry_agent_mask, t]
        else:
            entry_agent_mask = torch.zeros_like(present_mask)

        pred_traj[entry_agent_mask, -1] = pos_a_next[entry_agent_mask]

        if self.token_processor.pred_exit:
            next_mask = (next_mask & ~exit_mask) | entry_agent_mask
        else:
            next_mask = gt_valid[:, t]

        present_mask = present_mask | next_mask