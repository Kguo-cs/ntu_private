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
from .edge_encoder import EdgeEncoder
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
    weight_init
)

from src.smart.layers import MLPLayer


class SMARTMapDecoder(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            pl2pl_radius: float,
            num_freq_bands: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
            pt2pt_neighbor: int,
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers
        self.pt2pt_neighbor = pt2pt_neighbor

        self.token_emb = MLPLayer(44,hidden_dim, hidden_dim)


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

        self.apply(weight_init)

    def forward(self, z_lane,t_batch,batch):
        pos_pt=z_lane[:,:2]
        orient_pt=torch.atan2(z_lane[:,3],z_lane[:,2])

        x_pt=self.token_emb(z_lane)+t_batch[batch]

        head_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        edge_index_pt2pt, r_pt2pt = self.edge_encoder.build_map2map_edge(
            pos_pt,  # [n_pl, 2]
            orient_pt,  # [n_pl]
            pos_pt,  # [n_agent, n_step, 2]
            orient_pt,  # [n_agent, n_step]
            head_vector,  # [n_agent, n_step, 2]
            batch,  # [n_agent*n_step]
            batch,  # [n_pl*n_step]
            self.pl2pl_radius,
            self.pt2pt_neighbor,
        )

        for i in range(self.num_layers):
            x_pt= self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)

        output={
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": batch,
        }

        return output

class Agent_Diffuser(nn.Module):
    """Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(Agent_Diffuser, self).__init__()
        hidden_dim=128

        num_freq_bands=64

        num_heads=8

        head_dim=16

        self.map_encoder=SMARTMapDecoder(
            hidden_dim,
            pl2pl_radius=20,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0,
            pt2pt_neighbor=20,
            )


        self.cfg = cfg

        self.edge_encoder = EdgeEncoder(hidden_dim,
                                        num_freq_bands,
                                        use_a2a=True,
                                        use_pl2a=True
                                        )
        self.to_out_m_delta = MLPLayer(hidden_dim, hidden_dim, m_delta_dim)

        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=0,
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
                    dropout=0,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(weight_init)


    def forward(self, z_agent,z_lane,t_batch,agent_batch,lane_batch):

        lane_pred=self.map_encoder(z_lane,t_batch,lane_batch)

        agent_pred=self.agent_encoder(lane_pred,z_agent,t_batch,agent_batch)


        return lane_pred,agent_pred




