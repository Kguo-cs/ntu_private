import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from sd.utils.layers import ResidualMLP, AttentionLayer, AutoEncoderFactorizedAttentionBlock
from sd.utils.train_helpers import weight_init
from sd.utils.losses import GeometricLosses
from sd.utils.data_container import get_batches, get_features, get_edge_indices, get_encoder_edge_indices
from sd.utils.data_helpers import reparameterize
from sd.cfgs.config import NON_PARTITIONED

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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.edge_encoder import EdgeEncoder
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
    weight_init
)

from src.smart.layers import MLPLayer


class MapDecoder(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
    ) -> None:
        super(MapDecoder, self).__init__()
        self.num_layers = num_layers

        self.lane_emb = MLPLayer(44,hidden_dim, hidden_dim)


        self.edge_encoder = EdgeEncoder(hidden_dim, num_freq_bands, use_pl2a=True)

        self.pt2pt_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer=MLPLayer(hidden_dim,hidden_dim, 44)

        self.apply(weight_init)

    def forward(self, z_lane,t_batch,batch):
        pos_pt=z_lane[:,:2]
        orient_pt=torch.atan2(z_lane[:,3],z_lane[:,2])

        x_pt=self.lane_emb(z_lane)+t_batch[batch]

        head_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        edge_index_pt2pt, r_pt2pt = self.edge_encoder.build_map2map_edge(
            pos_pt,  # [n_pl, 2]
            orient_pt,  # [n_pl]
            pos_pt,  # [n_agent, n_step, 2]
            orient_pt,  # [n_agent, n_step]
            head_vector,  # [n_agent, n_step, 2]
            batch,  # [n_agent*n_step]
            batch,  # [n_pl*n_step]
            100,
            100,
        )

        for i in range(self.num_layers):
            x_pt= self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)

        lane_pred=self.output_layer(x_pt)

        output={
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": batch,
        }

        return output,lane_pred

class AgentDecoder(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
    ) -> None:
        super(AgentDecoder, self).__init__()
        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        use_a2a=True,
                                        use_pl2a=True
                                        )

        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.agent_emb = MLPLayer(10,hidden_dim, hidden_dim)

        self.output_layer=MLPLayer(hidden_dim,hidden_dim, 10)

        self.num_layers=num_layers

        self.apply(weight_init)

    def forward(self, map_feature,z_agent,t_batch,batch):

        theta = torch.atan2(z_agent[:, 3], z_agent[:, 2])

        pos_s = z_agent[:, :2]

        feat_a = self.agent_emb(z_agent)

        feat_a = feat_a + t_batch[batch]


        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        head_vector_s = torch.stack([theta.cos(), theta.sin()], dim=-1)

        edge_index_a2a, r_a2a, dist, relative_pos, r_a2a_nei, center_nei_pos, center_nei_heading = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,  # [n_agent, n_step, 2]
            head_s=theta,  # [n_agent, n_step]
            head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
            batch_s=batch,  # [n_agent*n_step]
            mask=None,  # [n_agent, n_step]
            max_radius=60,
            max_num_neighbors=20,
            agent_train_mask=None,
            layer_num=self.num_layers,
            counter_feat_a=None,
            dis_edge_mask=None
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        mask = None

        edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
            pos_pl=pos_pl,  # [n_pl, 2]
            orient_pl=orient_pl,  # [n_pl]
            pos_a=pos_s,  # [n_agent, n_step, 2]
            head_a=theta,  # [n_agent, n_step]
            head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch,  # [n_agent,n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
            pl2a_radius=100,
            max_num_neighbors=100,
            agent_train_mask=None,
            layer_num=self.num_layers
        )

        for layer_i in range(self.num_layers):
            feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

            feat_a = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a,
                                                    edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

        res = self.output_layer(feat_a)

        res_theta = torch.atan2(res[:, 3], res[:, 2])

        local_pos, local_theta = transform_to_global(
            res[:, :2],
            res_theta,
            pos_s,
            theta,
        )

        a_pred = torch.cat(
            [local_pos, torch.cos(local_theta)[:, None], torch.sin(local_theta)[:, None], res[:, 4:]], dim=-1)

        return a_pred



class Agent_Diffuser(nn.Module):
    """Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(Agent_Diffuser, self).__init__()
        hidden_dim=128

        num_freq_bands=64

        num_heads=8

        head_dim=16

        self.map_encoder=MapDecoder(
            hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0,
            )

        self.agent_encoder=AgentDecoder(
            hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=3,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0,
            )

        self.t_embed=MLPLayer(1,hidden_dim,hidden_dim)

        self.apply(weight_init)


    def forward(self, z_agent,z_lane,t_batch,agent_batch,lane_batch):

        t_batch=self.t_embed(t_batch)

        map_feature,lane_pred=self.map_encoder(z_lane,t_batch,lane_batch)

        agent_pred=self.agent_encoder(map_feature,z_agent,t_batch,agent_batch)


        return lane_pred,agent_pred




