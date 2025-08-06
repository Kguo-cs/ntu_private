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
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder
from src.smart.layers import MLPLayer
from .interative_decoder import InterativeDecoder

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
        token_processor=None,
    ) -> None:
        super(SMARTDecoder, self).__init__()

        self.tokenizer_training=False
        self.pl2a_radius = pl2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.iq_learn=False
        self.output_gmm=False
        self.use_gail=False

        self.use_value=True

        if self.tokenizer_training:
            from src.smart.loss.vq_vae import VQVAE

            self.vq_vae=VQVAE(token_processor)

        else:
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

            self.alpha=0.1

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
                output_gmm=self.output_gmm,
                pred_last_res=token_processor.pred_last_res,
                pred_all_res=token_processor.pred_all_res,
            )
            if self.iq_learn and self.use_gail:
                # self.discriminator=InterativeDecoder(hidden_dim,num_historical_steps,num_future_steps,time_span,
                #                                     10,10,num_freq_bands,
                #                                     1,num_heads,head_dim,
                #                                     0,hist_drop_prob,1,
                #                                     pt2a_neighbor//2,a2a_neighbor//2,
                #                                     token_processor,False,False,False,discriminator=True
                #                                         )
                self.discriminator = SMARTAgentDecoder(
                    hidden_dim=hidden_dim,
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                    time_span=10,
                    pl2a_radius=10,
                    a2a_radius=10,
                    num_freq_bands=num_freq_bands,
                    num_layers=1,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    hist_drop_prob=hist_drop_prob,
                    n_token_agent=1,
                    pt2a_neighbor=pt2a_neighbor//2,
                    a2a_neighbor=a2a_neighbor//2,
                    token_processor=token_processor,
                    alpha=self.alpha,
                    output_gmm=self.output_gmm,
                    pred_last_res=False,
                    pred_all_res=False,
                    discriminator=True
                )
                if self.use_value:
                    # self.value_network=InterativeDecoder(hidden_dim,num_historical_steps,num_future_steps,time_span,
                    #                                 pl2a_radius,a2a_radius,num_freq_bands,
                    #                                 1,num_heads,head_dim,
                    #                                 dropout,hist_drop_prob,1,
                    #                                 pt2a_neighbor,a2a_neighbor,
                    #                                 token_processor,False,False,False  )

                    self.value_network = SMARTAgentDecoder(
                        hidden_dim=hidden_dim,
                        num_historical_steps=num_historical_steps,
                        num_future_steps=num_future_steps,
                        time_span=time_span,
                        pl2a_radius=pl2a_radius,
                        a2a_radius=a2a_radius,
                        num_freq_bands=num_freq_bands,
                        num_layers=1,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        hist_drop_prob=hist_drop_prob,
                        n_token_agent=1,
                        pt2a_neighbor=pt2a_neighbor,
                        a2a_neighbor=a2a_neighbor,
                        token_processor=token_processor,
                        alpha=self.alpha,
                        output_gmm=False,
                        pred_last_res=False,
                        pred_all_res=False,
                    )

    def run_async_rollout(self,agent_encoder, tokenized_agent, detach_map_feature, post_sampling):
        encoder_was_training = agent_encoder.training

        with torch.no_grad(), torch.cuda.stream(self.rollout_stream):
            agent_encoder.eval()
            rollout_result = agent_encoder.inference(
                tokenized_agent, detach_map_feature, post_sampling
            )
            if encoder_was_training:
                agent_encoder.train()

        return rollout_result

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor],post_sampling=False,
            use_critic=False
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_agent:
            map_feature = tokenized_agent["map_feature"]
        else:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
            tokenized_agent["map_feature"] = map_feature
            # self.rollout_result = self.run_async_rollout(tokenized_agent, tokenized_map["detach_map_feature"] , post_sampling)

        pred_dict = self.agent_encoder(tokenized_agent, map_feature, post_sampling)

        return pred_dict

    def get_Q(self,feat_a,action):

        state_action=torch.cat([feat_a,action],dim=-1)

        current_Q = self.critic.token_predict_head(state_action)[...,0]

        return current_Q

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        post_sampling=False,
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_agent:
            map_feature = tokenized_agent["detach_map_feature"]
        else:
            if post_sampling:
                map_feature = None
            else:
                map_feature = self.map_encoder(tokenized_map)

        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, post_sampling
        )
        return pred_dict
