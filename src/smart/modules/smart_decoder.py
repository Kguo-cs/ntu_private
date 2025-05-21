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
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder
from .kl_loss import  BalancedKL

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
        n_token_agent: int,
        pt2pt_neighbor:int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        state_action=False,
        use_latent=False
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
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
            use_latent=use_latent
        )

        self.use_latent=use_latent

        if self.use_latent:
            self.post_encoder = SMARTAgentDecoder(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=100,
                pl2a_radius=pl2a_radius,
                a2a_radius=a2a_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_agent_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                n_token_agent=16,
                state_action=state_action,
            )

            self.prior_encoder = SMARTAgentDecoder(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=100,
                pl2a_radius=pl2a_radius,
                a2a_radius=a2a_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_agent_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                n_token_agent=16,
                state_action=state_action,
            )

            self.l_vae_kl = BalancedKL(kl_balance_scale=0.2, kl_free_nats=1.0)

    def compute_disc_val(self,state,action):
        if 'token_idx' in state[0].keys():
            tokenized_map,tokenized_agent=state
            map_feature = self.map_encoder(tokenized_map)
        else:
            map_feature,tokenized_agent=state
        pred_dict = self.agent_encoder(tokenized_agent, map_feature)
        action_embed=self.action_encoder(action)
        state_embed=pred_dict["cur_pred"]
        state_action=torch.cat([state_embed,action_embed],dim=-1)
        score=self.pred_score(state_action)[:,0]

        return score

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor],kl_loss=True
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_map:
            map_feature = tokenized_map["map_feature"]
            #map_feature["pt_token"]=map_feature['pt_token'].detach()
        else:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_map["map_feature"] = map_feature

        if self.use_latent:
            post_dist = self.post_encoder(tokenized_agent, map_feature,get_latent_dist=True)

            latent_feature = post_dist.sample(deterministic=False)
        else:
            latent_feature=None

        pred_dict = self.agent_encoder(tokenized_agent, map_feature,latent_feature)

        if kl_loss and self.use_latent:
            prior_dist = self.prior_encoder(tokenized_agent, map_feature,n_step=2,get_latent_dist=True)

            error_vae = self.l_vae_kl.compute(post_dist.distribution, prior_dist.distribution)

            pred_dict["kl_loss"]=error_vae.mean()
        else:
            pred_dict["kl_loss"] =torch.tensor(0)

        return pred_dict

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_map:
            map_feature = tokenized_map["map_feature"]
        else:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_map["map_feature"] = map_feature

        if self.use_latent:
            prior_dist = self.prior_encoder(tokenized_agent, map_feature,n_step=2,get_latent_dist=True)

            latent_feature = prior_dist.sample(deterministic=False)
        else:
            latent_feature =None

        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme,latent_feature
        )
        return pred_dict
