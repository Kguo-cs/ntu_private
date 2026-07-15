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
from torch import Tensor


from src.smart.layers import MLPLayer
import torch.nn.functional as F
from src.smart.modules.map_decoder import SMARTMapDecoder
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.diffusion.initial_diffusion import InitDiffusion

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
rotate_to_local,
infer_prev_pose
)

class SMARTDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        pt2pt_neighbor:int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        n_token_agent: int,
        dis_a2a_radius: int,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
        token_processor=None,
        finetune=False,
    ) -> None:
        super(SMARTDecoder, self).__init__()

        self.pl2a_radius = pl2a_radius
        self.pt2a_neighbor = pt2a_neighbor

        self.pred_init=token_processor.pred_init
        self.learn_init=token_processor.learn_init
        self.finetune=finetune
        self.token_processor=token_processor

        self.use_lcf=reward_weight!=0
        self.use_kl_penalty=False
        self.gail=dis_a2a_radius>0

        self.alpha = 0.1

        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=token_processor
        )

        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=token_processor,
            alpha=self.alpha,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            use_gail=self.gail
        )

        if self.pred_init:
            self.init_decoder = InitDiffusion(hidden_dim, num_heads, num_freq_bands, token_processor,self.gail)#
            self.sep_map= self.init_decoder.sep_map

            if self.init_decoder.sep_map:
                self.init_map_encoder = SMARTMapDecoder(
                    hidden_dim=hidden_dim,
                    pl2pl_radius=pl2pl_radius,
                    num_freq_bands=num_freq_bands,
                    num_layers=1,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    pt2pt_neighbor=pt2pt_neighbor,
                    token_processor=token_processor
                )

        if self.gail:
            self.discriminator = SMARTAgentDecoder(
                hidden_dim=hidden_dim,#//2
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=10,
                pl2a_radius=pl2a_radius,
                a2a_radius=dis_a2a_radius,#20 bad
                num_freq_bands=num_freq_bands,
                num_layers=1,
                num_heads=num_heads,
                head_dim=head_dim,#//2
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                n_token_agent=1,
                pt2a_neighbor=pt2a_neighbor,
                a2a_neighbor=a2a_neighbor,
                token_processor=token_processor,
                alpha=self.alpha,
                dis_weight=dis_weight,
                dist_decay=dist_decay,
                reward_weight=reward_weight,
                reward_decay=reward_decay,
                discriminator=True
            )

            self.value_network =MLPLayer(hidden_dim,hidden_dim*2,1)
            if self.use_lcf:
                self.nei_value_network =MLPLayer(hidden_dim,hidden_dim*2,1)

            if token_processor.learn_init:
                self.init_value_network =MLPLayer(self.init_decoder.G1.hidden_dim,hidden_dim*2,1)

            self.agent_encoder.interative_decoder.gail=self.gail

    def forward( self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]  ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_agent:
            map_feature = tokenized_agent["map_feature"]
        else:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature

        # self.init_decoder.eval()
        # gt_initial_pos, gt_initial_heading, gt_initial_idx, shape, gt_initial_vel = self.init_decoder(tokenized_agent,resampling=True)
        # self.init_decoder.train()
        #
        # pos=tokenized_agent["gt_traj_10hz"]
        # heading=tokenized_agent["gt_head_10hz"]
        # valid= tokenized_agent["gt_valid_10hz"]
        # token_traj=tokenized_agent["token_traj"]
        # agent_shape=tokenized_agent["token_agent_shape"]
        # token_traj_all = tokenized_agent["token_traj_all"]
        #
        # pos0, head0 = infer_prev_pose(gt_initial_pos[:, :1], gt_initial_heading[:, :1], gt_initial_idx[:, -1:], token_traj_all)
        #
        # pos[:,5]=gt_initial_pos[:,0]
        # heading[:,5]=gt_initial_heading[:,0]
        # pos[:,0]=pos0[:,0]
        # heading[:,0]=head0[:,0]
        #
        # token_dict = self.token_processor._match_agent_token(
        #     valid=valid,
        #     pos=pos,
        #     heading=heading,
        #     agent_shape=agent_shape,
        #     token_traj=token_traj,
        # )
        # tokenized_agent.update(token_dict)

        if self.learn_init and self.finetune and not self.gail:
            pred_dict={}
        else:
            pred_dict = self.agent_encoder(tokenized_agent, map_feature) #not use when only learning init


        if self.learn_init and not self.gail:
            if self.sep_map:
                initial_map_feature = self.init_map_encoder(tokenized_map,tokenized_agent=tokenized_agent)
                tokenized_agent["initial_map_feature"] = initial_map_feature

            initial_logit = self.init_decoder(tokenized_agent)

            pred_dict["initial_logit"]=initial_logit

        return pred_dict

    def inference(
            self,
            tokenized_map: Dict[str, Tensor],
            tokenized_agent: Dict[str, Tensor],
            post_sampling=False,
            n_step_future_10hz=None,
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_agent:
            map_feature = tokenized_agent["map_feature"]
        else:
            if post_sampling:
                map_feature = None
            else:
                map_feature = self.map_encoder(tokenized_map)

        pred_dict = self.agent_encoder.inference(self.init_decoder,
            tokenized_agent, map_feature,n_step_future_10hz=n_step_future_10hz
        )
        return pred_dict
