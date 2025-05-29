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

import torch
import torch.nn as nn
from torch_cluster import radius_graph

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from torch_scatter import scatter_mean,scatter_max
from .agent_decoder import  radiusGraphNearest
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock, general_rope
from torch.nn.utils.rnn import pad_sequence

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
        pt2pt_neighbor:int
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers
        self.use_map=True

        if self.use_map:
            self.type_pt_emb = nn.Embedding(10, hidden_dim)
            self.polygon_type_emb = nn.Embedding(4, hidden_dim)
            self.light_pl_emb = nn.Embedding(5, hidden_dim)

            self.head_dim=head_dim

            # map_token_traj_src: [n_token, 11, 2].flatten(0,1)
            self.token_emb = MLPEmbedding(input_dim=22, hidden_dim=hidden_dim)

            self.pt2pt_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

            self.apply(weight_init)

    def padding(self,tensor,lengths ):
        padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=0)

        return padded_tensor

    def forward(self, tokenized_map: Dict):
        if not self.use_map:
            return {}

        pos_pt = tokenized_map["position"]
        orient_pt = tokenized_map["orientation"]
        pt_token_emb_src = self.token_emb(tokenized_map["token_traj_src"])
        x_pt = pt_token_emb_src[tokenized_map["token_idx"]]

        x_pt_categorical_embs = [
            self.type_pt_emb(tokenized_map["type"]),
            self.polygon_type_emb(tokenized_map["pl_type"]),
            self.light_pl_emb(tokenized_map["light_type"]),
        ]

        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)

        batch = tokenized_map["batch"]

        unique_ids, counts = batch.unique_consecutive(return_counts=True)

        lengths = counts.tolist()

        padded_pt_feature = self.padding(x_pt, lengths)

        map_mask = (padded_pt_feature == 0).all(-1)[:,None, None]

        sinusoidal_pos = general_rope(pos_pt, self.head_dim, orient_pt)

        map_sinusoidal = self.padding(sinusoidal_pos, lengths)

        x_pt = self.pt2pt_roformer(padded_pt_feature, map_mask, map_sinusoidal)

        return {
            "pt_token": x_pt,
            # "position": pos_pt,
            # "orientation": orient_pt,
             "batch": batch,
             "map_mask": map_mask ,
             "map_sinusoidal": map_sinusoidal
        }

