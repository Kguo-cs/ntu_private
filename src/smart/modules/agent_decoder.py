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
from src.smart.diffusion.initial_diffusion import InitDiffusion


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

        self.pred_init=token_processor.pred_init & (not discriminator)

        self.learn_init=token_processor.learn_init

        if self.pred_init:
            self.init_decoder = InitDiffusion(hidden_dim, num_heads, num_freq_bands, token_processor)

    def predict_agent(self, sampled_idx,token_mask, mask_a ,pos_a,head_a,tokenized_agent, map_feature, n_current=0):

        # if self.discriminator:
        #     pos_a=pos_a+torch.randn_like(pos_a)*1e-3
        #     head_a=head_a+torch.randn_like(head_a)*1e-3

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

        return next_token_logits,a2a_feature,rewards,weight,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) :
        entry_logit = next_token_logits = mask_token_logit=pred_action_mask =initial_logit=None

        if self.learn_init:
            initial_logit = self.init_decoder(tokenized_agent)
        else:
            next_token_logits,a2a_feature,rewards,agent_token_emb,feat_a= self.predict_agent(tokenized_agent["sampled_idx"][:,:-1],
                                                                                    tokenized_agent["token_mask"][:,:-1],
                                                                                    tokenized_agent["valid_mask"][:,:-1],
                                                                                    tokenized_agent["sampled_pos"][:,:-1],
                                                                                    tokenized_agent["sampled_heading"][:,:-1] ,
                                                                                    tokenized_agent,
                                                                                    map_feature
                                                                                                         )

            tokenized_agent["next_token_logits"] = next_token_logits
            tokenized_agent["entry_logit"] = entry_logit
            tokenized_agent["initial_logit"] = initial_logit
            tokenized_agent["feat_a"] = feat_a

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
        token_traj_all = tokenized_agent["token_traj_all"]

        if "gt_z_raw" not in tokenized_agent.keys():
            max_step = 17
            current_step=1

        head_a = gt_head[:, :current_step]
        mask = gt_valid[:, :current_step]
        pos_a = gt_pos[:, :current_step]
        sampled_idx=gt_sampled_idx[:, :current_step]
        token_mask=tokenized_agent["token_mask"][:, :current_step].clone()

        if self.pred_init:

            # if "gt_z_raw" not in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            #     batch=tokenized_agent["batch"]
            #     exist_n=5
            #     batch_mask=batch<exist_n
            #
            #     tokenized_agent_new={}
            #
            #     for key in {"initial_pos","initial_heading","batch","ego_mask","initial_type","initial_shape","initial_vel","token_traj"}:
            #         tokenized_agent_new[key]=tokenized_agent[key][batch_mask]
            #
            #     tokenized_agent_new["num_graphs"]=exist_n
            #     tokenized_agent_new["ego_traj"]=tokenized_agent["ego_traj"][:exist_n]
            #
            #     batch_pl_mask=map_feature["batch"]<exist_n
            #
            #     new_map_features={}
            #
            #     for key in map_feature.keys():
            #         new_map_features[key]=map_feature[key][batch_pl_mask]
            #
            #     tokenized_agent_new["map_feature"]=new_map_features
            # else:
            tokenized_agent_new=tokenized_agent

            pos_a1, head_a1, sampled_idx1, initial_speed = self.init_decoder(tokenized_agent_new)

            if self.token_processor.use_all_pos:
                out_dict = {
                    "shape": tokenized_agent["shape"],
                    "pred_traj_10hz": pos_a,
                    "pred_head_10hz": head_a,
                    "pred_z_10hz": torch.zeros_like(pos_a[:, :, 0]),
                    "initial_speed": initial_speed,
                }

                return out_dict

            if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
                pos_a=pos_a1
                head_a=head_a1
                sampled_idx=sampled_idx1
                max_step = 18
                current_step = 1

            else:
                # head_a = head_a[:, 1:]
                # pos_a = pos_a[:, 1:]
                # sampled_idx = sampled_idx[:, 1:]

                # pos_a[batch_mask]=pos_a1
                # head_a[batch_mask]=head_a1
                # sampled_idx[batch_mask]=sampled_idx1
                pos_a=pos_a1
                head_a=head_a1
                sampled_idx=sampled_idx1

                current_step = 1
                max_step = 17

            mask=torch.ones_like(mask[:, :current_step])
            token_mask = torch.ones_like(token_mask[:, :current_step])

        n_agent = sampled_idx.shape[0]
        next_mask=mask[:, -1]
        pred_traj_10hz = []
        pred_head_10hz = []
        a_num = next_mask.sum()

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None:

                    next_token_logits=tokenized_agent["next_token_logits"][a_num*(current_step-1):a_num*current_step]

                    self.interative_decoder.pos_cache = self.interative_decoder.pos_cache[:, :current_step]
                    self.interative_decoder.head_cache = self.interative_decoder.head_cache[:, :current_step]
                    self.interative_decoder.mask_cache = self.interative_decoder.mask_cache[:, :current_step]
                    self.interative_decoder.head_vector_cache = self.interative_decoder.head_vector_cache[:,
                                                                :current_step]
                    self.interative_decoder.feat_a_cache =[feat[:current_step] for feat in self.interative_decoder.feat_a_cache ]
                else:
                    next_token_logits,a2a_feature,_,_,feat_a = self.predict_agent(sampled_idx,token_mask, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,0)
            else:
                next_token_logits, a2a_feature, _, _, feat_a = self.predict_agent(
                    sampled_idx[:, -1:], token_mask[:, -1:], mask[:, -1:],
                    pos_a[:, -2:], head_a[:, -1:], tokenized_agent, map_feature, t - 1)

            next_sampled=Categorical(logits=next_token_logits / self.alpha).sample()

            next_token_idx = torch.zeros_like(sampled_idx[:, -1])
            if len(next_token_logits):
                next_token_idx[next_mask] = next_sampled[-next_mask.sum():]

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

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

            if "gt_z_raw" in tokenized_agent.keys():
                pred_traj = token_traj_global[:, :].mean(2)
                diff_xy = token_traj_global[:, :, 0] - token_traj_global[:, :, 3]
                pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
                pred_head_10hz.append(pred_head)
                pred_traj_10hz.append(pred_traj)

            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]

            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

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