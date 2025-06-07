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
from torch_scatter import scatter_mean,scatter_max
from .build_edge import  radiusGraphNearest2

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
        token_processor=None,
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
            token_processor=token_processor
        )

        self.pl2a_radius=pl2a_radius
        self.pt2a_neighbor=pt2a_neighbor


    def scene_centric(self,pos,heading,centering_pos,centering_heading,batch):


        heading = heading - centering_heading[batch]

        pos = pos - centering_pos[batch]

        cos_a = torch.cos(centering_heading)[batch]
        sin_a = torch.sin(centering_heading)[batch]

        x, y = pos[..., 0], pos[..., 1]
        x_rot = cos_a * x + sin_a * y
        y_rot = -sin_a * x + cos_a * y

        pos = torch.stack([x_rot, y_rot], dim=-1)

        return  pos,heading

    def preprocess(self,tokenized_map,tokenized_agent):
        batch = tokenized_map["batch"]

        pos = tokenized_map["position"]

        heading = tokenized_map["orientation"]

        centering_pos = scatter_mean(pos, batch, dim=0)

        centering_heading = scatter_mean(heading, batch, dim=0)

        pos, heading = self.scene_centric(pos, heading, centering_pos, centering_heading, batch)

        tokenized_map["position"] = pos
        tokenized_map["heading"] = heading

        pos = tokenized_agent["sampled_pos"]
        heading = tokenized_agent["sampled_heading"]
        batch = tokenized_agent["batch"]

        pos, heading = self.scene_centric(pos, heading, centering_pos[:,None], centering_heading[:,None], batch)

        tokenized_agent["sampled_pos"] = pos

        tokenized_agent["sampled_heading"] = heading

        tokenized_agent["centering_pos"]=centering_pos
        tokenized_agent["centering_heading"]=centering_heading

        return tokenized_map,tokenized_agent

    def filter_map(self,tokenized_map,tokenized_agent):

        pos_a=tokenized_agent["sampled_pos"]
        n_step = pos_a.shape[1]

        pos_pl=tokenized_map["position"]
        mask=tokenized_agent["valid_mask"]
        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]

        batch_pl = torch.cat(
            [
                tokenized_map["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_pl*n_step]

        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        map_point_num=len(pos_pl)
        pos_pt = pos_pl.repeat(n_step, 1)
        edge_index_pl2a = radiusGraphNearest2(x=pos_s[:, :2],
                                              y=pos_pt[:, :2],
                                              r=self.pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=self.pt2a_neighbor)
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        used_point=torch.unique(edge_index_pl2a[0]%map_point_num)


        # edge_index_pl2pl = radiusGraphNearest2(x=pos_pl[used_point],
        #                                       y=pos_pl,
        #                                       r=20,
        #                                       batch_x=tokenized_map["batch"][used_point],
        #                                       batch_y=tokenized_map["batch"],
        #                                       max_num_neighbors=10)
        #
        # used_point=torch.unique(edge_index_pl2pl[0])

        used_mask=torch.isin(torch.arange(map_point_num,device=pos_s.device),used_point)


        for key in tokenized_map.keys():
            if key != 'token_traj_src':
                tokenized_map[key] = tokenized_map[key][used_mask]
        return  tokenized_map

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor],kl_loss=True
    ) -> Dict[str, Tensor]:
        if "map_feature" in tokenized_map:
            map_feature = tokenized_map["map_feature"]
        else:
            #tokenized_map,tokenized_agent = self.preprocess(tokenized_map, tokenized_agent)
            #tokenized_map=self.filter_map(tokenized_map, tokenized_agent)

            map_feature = self.map_encoder(tokenized_map)
            map_feature_dict={}
            for key in map_feature.keys():
                map_feature[key] = map_feature[key].detach()
            tokenized_map["map_feature"] = map_feature

        pred_dict = self.agent_encoder(tokenized_agent, map_feature)

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

        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme
        )
        return pred_dict
