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
        dis_a2a_neighbor: int,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
        token_processor=None,
    ) -> None:
        super(SMARTDecoder, self).__init__()

        self.pl2a_radius = pl2a_radius
        self.pt2a_neighbor = pt2a_neighbor

        self.use_smart=token_processor.use_smart
        self.pred_init=token_processor.pred_init

        self.use_lcf=reward_weight!=0
        self.use_value=True
        self.use_kl_penalty=False
        self.gail=dis_a2a_radius>0
        self.learn_dis=True

        if self.use_smart:
            self.alpha = 1
            from .old_agent_decoder import SMARTAgentDecoder
            from .old_map_encoder import SMARTMapDecoder

            self.map_encoder = SMARTMapDecoder(
                hidden_dim=hidden_dim,
                pl2pl_radius=10,
                num_freq_bands=num_freq_bands,
                num_layers=3,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )

            self.agent_encoder = SMARTAgentDecoder(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=time_span,
                pl2a_radius=30,
                a2a_radius=a2a_radius,
                num_freq_bands=num_freq_bands,
                num_layers=6,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                n_token_agent=n_token_agent,
            )
        else:
            from .agent_decoder import SMARTAgentDecoder
            from .map_decoder import SMARTMapDecoder
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
            )
            if token_processor.learn_init :
                self.sep_map= self.agent_encoder.init_decoder.sep_map
            else:
                self.sep_map=False

            if self.sep_map:

                self.map_encoder1 = SMARTMapDecoder(
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

        from .agent_decoder import SMARTAgentDecoder

        if self.gail:
            self.discriminator = SMARTAgentDecoder(
                hidden_dim=hidden_dim,#//2
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=time_span,
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
                a2a_neighbor=dis_a2a_neighbor,
                token_processor=token_processor,
                alpha=self.alpha,
                dis_weight=dis_weight,
                dist_decay=dist_decay,
                reward_weight=reward_weight,
                reward_decay=reward_decay,
                discriminator=True
            )

            if self.use_value:
                self.value_network =MLPLayer(hidden_dim,hidden_dim*2,1)
                if self.use_lcf:
                    self.nei_value_network =MLPLayer(hidden_dim,hidden_dim*2,1)

            self.agent_encoder.interative_decoder.gail=self.gail

    def forward( self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]  ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_agent:
            map_feature = tokenized_agent["map_feature"]
        else:
            if self.agent_encoder.learn_init and not self.agent_encoder.init_decoder.use_all_pos:
                map_feature = self.map_encoder(tokenized_map,tokenized_agent=tokenized_agent)
                tokenized_agent["initial_map_feature"] = map_feature
            else:
                map_feature = self.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature

        pred_dict = self.agent_encoder(tokenized_agent, map_feature)

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

        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, post_sampling,n_step_future_10hz=n_step_future_10hz
        )
        return pred_dict
