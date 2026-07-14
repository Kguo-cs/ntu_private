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
    weight_init,
    infer_prev_pose
)
from torch.distributions import Categorical
from src.smart.utils.edge_utils import build_batch
from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder

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
            use_gail=False,
            discriminator=False,
            traj_diffusion=False,
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.token_processor= token_processor
        self.discriminator=discriminator
        self.alpha = alpha

        self.shift = token_processor.shift

        self.agent_token_embedding=AgentTokenEncoder(hidden_dim,num_freq_bands,token_processor,discriminator,traj_diffusion)

        self.interative_decoder = InterativeDecoder(hidden_dim,time_span,
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

        self.pred_init=token_processor.pred_init & (not discriminator)

        self.learn_init=token_processor.learn_init

        # self.use_gail=use_gail
        #
        # if self.use_gail:
        #     self.init_decoder = InitDiffusion(hidden_dim, num_heads, num_freq_bands, token_processor)

    def predict_agent(self, sampled_idx,token_mask, mask_a ,pos_a,head_a,tokenized_agent, map_feature,shape, n_current=0):

        # if self.discriminator and self.training and not self.token_processor.use_gradient_penalty:
        #     pos_a=pos_a+torch.randn_like(pos_a)*1e-1
        #     head_a=head_a+torch.randn_like(head_a)*1e-2
        #     shape=shape+torch.randn_like(shape)*1e-2

        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        feat_a_token,agent_token_emb,counter_feat_a = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask_a=mask_a,
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=shape,  # [n_agent, 3]
            token_mask=token_mask,
            goal_pos=tokenized_agent.get("goal_pos"),
            goal_mask=tokenized_agent.get("goal_mask"),
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
        if tokenized_agent["sampled_idx"].shape[1]==1:
            return {}
        next_token_logits,a2a_feature,rewards,agent_token_emb,feat_a= self.predict_agent(tokenized_agent["sampled_idx"][:,:-1],
                                                                                tokenized_agent["token_mask"][:,:-1],
                                                                                tokenized_agent["valid_mask"][:,:-1],
                                                                                tokenized_agent["sampled_pos"][:,:-1],
                                                                                tokenized_agent["sampled_heading"][:,:-1] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                tokenized_agent["shape"]
                                                                                )

        tokenized_agent["next_token_logits"] = next_token_logits
        tokenized_agent["feat_a"] = feat_a

        return {
            "next_token_logits": next_token_logits
         }

    def autoregressive_agent(self,init_decoder, tokenized_agent, map_feature,current_step,max_step):
        gt_pos=tokenized_agent["sampled_pos"].clone()
        gt_head=tokenized_agent["sampled_heading"].clone()
        gt_valid=tokenized_agent["valid_mask"].clone()
        gt_sampled_idx=tokenized_agent["sampled_idx"].clone()

        pred_traj_10hz = []
        pred_head_10hz = []

        if self.pred_init:
            # pos_a=gt_pos[:,:2]
            # head_a=gt_head[:,:2]
            # sampled_idx=gt_sampled_idx[:,:2]
            # shape=tokenized_agent["initial_shape"]
            # initial_local_vel =tokenized_agent["local_vel"]

            pos_a,head_a, sampled_idx,shape,initial_local_vel = init_decoder(tokenized_agent)

            if "gt_z_raw" in tokenized_agent.keys():

                token_traj_all = tokenized_agent["token_traj_all"]

                pos5, head5=infer_prev_pose(pos_a[:,:1],head_a[:,:1],sampled_idx[:,-1:],token_traj_all)

                if sampled_idx.shape[1]>1:
                    pos_a=torch.cat([pos5,pos_a],dim=1)
                    head_a=torch.cat([head5,head_a],dim=1)

                    pos0, head0=infer_prev_pose(pos5, head5,sampled_idx[:,:1],token_traj_all)

                    pred_traj_10hz.append(pos0)
                    pred_head_10hz.append(head0)

                    new_pos,new_head=self.get_next(sampled_idx[:,:1],pos0, head0,pred_traj_10hz,pred_head_10hz,tokenized_agent)
                    new_pos1,new_head1=self.get_next(sampled_idx[:,1:2],pos5, head5,pred_traj_10hz,pred_head_10hz,tokenized_agent)

                else:
                    pred_traj_10hz.append(pos5)
                    pred_head_10hz.append(head5)
                    new_pos,new_head=self.get_next(sampled_idx[:,:1],pos5, head5,pred_traj_10hz,pred_head_10hz,tokenized_agent)

            current_step=pos_a.shape[1]
            if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
                max_step = 18-current_step
            else:
                max_step=gt_valid.shape[1]-current_step
            mask = torch.ones_like(gt_valid[:, :current_step])
            token_mask = torch.ones_like(tokenized_agent["token_mask"][:, :current_step])
        else:
            head_a = gt_head[:, :current_step]
            pos_a = gt_pos[:, :current_step]
            sampled_idx = gt_sampled_idx[:, :current_step]
            shape=tokenized_agent["shape"]
            mask = gt_valid[:, :current_step]
            token_mask = tokenized_agent["token_mask"][:, :current_step]

        next_mask=mask[:, -1]
        a_num = next_mask.sum()

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None and not self.pred_init:

                    next_token_logits=tokenized_agent["next_token_logits"][a_num*(current_step-1):a_num*current_step]

                    self.interative_decoder.pos_cache = self.interative_decoder.pos_cache[:, :current_step]
                    self.interative_decoder.head_cache = self.interative_decoder.head_cache[:, :current_step]
                    self.interative_decoder.mask_cache = self.interative_decoder.mask_cache[:, :current_step]
                    self.interative_decoder.head_vector_cache = self.interative_decoder.head_vector_cache[:,
                                                                :current_step]
                    self.interative_decoder.feat_a_cache =[feat[:current_step] for feat in self.interative_decoder.feat_a_cache ]
                else:
                    next_token_logits,a2a_feature,_,_,feat_a = self.predict_agent(sampled_idx,token_mask, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,shape,t-1)
            else:
                next_token_logits, a2a_feature, _, _, feat_a = self.predict_agent(
                    sampled_idx[:, -1:], token_mask[:, -1:], mask[:, -1:],
                    pos_a[:, -2:], head_a[:, -1:], tokenized_agent, map_feature, shape,t - 1)

            num_active = int(next_mask.sum().item())
            next_token_idx = sampled_idx[:, -1].clone()
            if num_active > 0:
                active_logits = next_token_logits[-num_active:] / self.alpha
                probs = torch.softmax(active_logits, dim=-1)
                next_token_idx[next_mask] = torch.multinomial(probs, 1).squeeze(-1)

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            pos_a,head_a=self.get_next(sampled_idx,pos_a,head_a,pred_traj_10hz,pred_head_10hz,tokenized_agent)

            next_token_mask = mask[:, -1] & next_mask

            mask = torch.cat([mask, next_mask[:, None]], dim=1)

            token_mask = torch.cat([token_mask, next_token_mask[:, None]], dim=1)

        out_dict = {
            "num_graphs":tokenized_agent["num_graphs"],
            "type": tokenized_agent["type"],
            "shape": shape,
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "token_mask":token_mask,
            "sampled_idx": sampled_idx,  # [n_agent, 18]
        }

        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            if self.pred_init:
                out_dict["initial_local_vel"] = initial_local_vel

            out_dict["pred_head_10hz"] = torch.cat(pred_head_10hz, dim=1)  # tokenized_agent['gt_head_10hz']#
            out_dict["pred_traj_10hz"] = torch.cat(pred_traj_10hz, dim=1)  # tokenized_agent['gt_traj_10hz'] #

            out_dict["pred_z_10hz"] = tokenized_agent["gt_z_raw"].unsqueeze(1) .expand(-1, out_dict["pred_traj_10hz"].shape[1])

        return out_dict

    def get_next(self,sampled_idx,pos_a,head_a,pred_traj_10hz,pred_head_10hz,tokenized_agent):
        token_traj_all = tokenized_agent["token_traj_all"]

        next_token_traj_all = token_traj_all[torch.arange(len(sampled_idx),device=sampled_idx.device), sampled_idx[:, -1]]

        token_traj_global = transform_to_global(
            pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
            head_local=None,
            pos_now=pos_a[:, -1],  # [n_agent, 2]
            head_now=head_a[:, -1],  # [n_agent]
        )[0].view(*next_token_traj_all.shape)

        if "gt_z_raw" in tokenized_agent.keys():
            pred_traj = token_traj_global.mean(2)
            diff_xy = token_traj_global[:, :, 0] - token_traj_global[:, :, 3]
            pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

            pred_head_10hz.append(pred_head)
            pred_traj_10hz.append(pred_traj)

        pos_a_next = token_traj_global[:, -1].mean(dim=1)[:, None]
        diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
        head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])[:, None]

        pos_a = torch.cat([pos_a, pos_a_next], dim=1)
        head_a = torch.cat([head_a, head_a_next], dim=1)

        return pos_a, head_a

    def inference(
            self,
            init_decoder,
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

        out_dict=self.autoregressive_agent(init_decoder,tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz)

        return out_dict