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

        batch = tokenized_map["batch"]
        pos_pt = tokenized_map["position"]

        # N = pos_pt.size(0)
        # max_k = 500
        #
        # # Step 1: Compute per-batch centroids
        # centroid = scatter_mean(pos_pt, batch, dim=0)  # shape: (B, D)
        #
        # # Step 2: Compute distance of each point to its batch centroid
        # centroid_per_point = centroid[batch]  # shape: (N, D)
        # distances = torch.norm(pos_pt - centroid_per_point, dim=1)  # shape: (N,)
        #
        # # Step 3: For each batch index, select the nearest 500 points
        # # Group indices per batch
        # # unique_batches = torch.unique(batch)
        # # mask = torch.zeros_like(batch, dtype=torch.bool)
        #
        # # for b in unique_batches:
        # #     idx = (batch == b).nonzero(as_tuple=False).squeeze()  # indices for batch b
        # #     dist_b = distances[idx]
        # #     if dist_b.numel() <= 500:
        # #         mask[idx] = True
        # #     else:
        # #         topk = torch.topk(dist_b, k=500, largest=False).indices
        # #         selected_idx = idx[topk]
        # #         mask[selected_idx] = True
        #
        # # Step 3: Sort distances within each batch (vectorized using a trick)
        # # Offset batch index to separate groups in global sort
        # max_dist = distances.max() + 1  # ensure batch shift won't change intra-batch order
        # sort_key = distances + batch.to(distances.dtype) * max_dist
        # sorted_indices = torch.argsort(sort_key)  # global sort, but grouped by batch
        #
        # # Step 4: Count how many elements per batch
        # batch_sizes = torch.bincount(batch)  # shape: (B,)
        # cumsum = torch.cumsum(batch_sizes, dim=0)
        # start = torch.zeros_like(cumsum)
        # start[1:] = cumsum[:-1]
        #
        # # Step 5: Create rank per item in sorted list
        # rank = torch.empty_like(batch, dtype=torch.long)
        # rank[sorted_indices] = torch.arange(N, device=batch.device) - start[batch[sorted_indices]]
        #
        # # Step 6: Keep only top-k (rank < 500)
        # mask = rank < max_k

        # batch=batch[mask]
        # pos_pt=pos_pt[mask]


        orient_pt = tokenized_map["orientation"]#[mask]
        pt_token_emb_src = self.token_emb(tokenized_map["token_traj_src"])
        x_pt = pt_token_emb_src[tokenized_map["token_idx"]]#[mask]

        x_pt_categorical_embs = [
            self.type_pt_emb(tokenized_map["type"]),#[mask]
            self.polygon_type_emb(tokenized_map["pl_type"]),#[mask]
            self.light_pl_emb(tokenized_map["light_type"]),#[mask]
        ]

        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)

        lengths = torch.bincount(batch).tolist()

        padded_pt_feature = self.padding(x_pt, lengths)

        map_mask = (padded_pt_feature == 0).all(-1)[:, None,None]

        sinusoidal_pos = general_rope(pos_pt, self.head_dim, orient_pt)

        map_sinusoidal = self.padding(sinusoidal_pos, lengths)

        padd_pos=self.padding(pos_pt, lengths)

        pt2pt_dist=torch.linalg.norm(padd_pos[:,None]-padd_pos[:,:,None],dim=-1)

        pt2pt_mask = map_mask | (pt2pt_dist[:,None]>self.pl2pl_radius)

        x_pt = self.pt2pt_roformer(padded_pt_feature, pt2pt_mask, map_sinusoidal)

        return {
            "pt_token": x_pt,
            "position": padd_pos,
            # "orientation": orient_pt,
             "batch": batch,
             "map_mask": map_mask ,
             "map_sinusoidal": map_sinusoidal
        }

