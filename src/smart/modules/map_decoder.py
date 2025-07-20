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

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, padding
from .build_edge import radiusGraphNearest


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
        pt2pt_neighbor:int,
        token_processor
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers
        self.use_map=True
        self.pt2pt_neighbor=pt2pt_neighbor

        self.gnn= True
        self.token_processor=token_processor

        if self.use_map:
            self.type_pt_emb = nn.Embedding(11, hidden_dim)
            self.polygon_type_emb = nn.Embedding(4, hidden_dim)

            # if not self.token_processor.pred_light:
            #     self.light_pl_emb = nn.Embedding(5, hidden_dim)

            self.head_dim=head_dim

            # map_token_traj_src: [n_token, 11, 2].flatten(0,1)
            self.my_map=False

            if self.my_map:
                self.token_emb = MLPEmbedding(input_dim=27, hidden_dim=hidden_dim)
            else:
                self.token_emb = MLPEmbedding(input_dim=22, hidden_dim=hidden_dim)
            #self.token_emb = nn.Embedding(token_processor.n_token_map, hidden_dim)

            if self.gnn:
                input_dim_r_pt2pt = 3
                self.r_pt2pt_emb = FourierEmbedding(
                    input_dim=input_dim_r_pt2pt,
                    hidden_dim=hidden_dim,
                    num_freq_bands=num_freq_bands,
                )
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
            # else:
            #     self.pt2pt_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)


            self.apply(weight_init)

    def forward(self, tokenized_map: Dict):
        if not self.use_map:
            return {}

        map_type=tokenized_map["type"].long()
       # map_type[map_type>9] = 9
        
        mask = torch.ones_like(map_type, dtype=bool)

        type4_indices=torch.where((map_type==4) |(map_type==5))[0]

        sampled_indices = type4_indices[1::5]

        mask[sampled_indices] = False
        sampled_indices = type4_indices[2::5]

        mask[sampled_indices] = False
        sampled_indices = type4_indices[3::5]
        mask[sampled_indices] = False
        sampled_indices = type4_indices[4::5]

        mask[sampled_indices] = False

        #mask[map_type>10] = False


        batch = tokenized_map["batch"][mask]

        map_type=map_type[mask]

        if "orientation" in tokenized_map.keys():
            pos_pt = tokenized_map["position"][mask]
            orient_pt = tokenized_map["orientation"][mask]
            pt_token_emb_src = self.token_emb(self.token_processor.map_token_traj_src)
            x_pt = pt_token_emb_src[tokenized_map["token_idx"].long()[mask]]#
            #x_pt = self.token_emb(tokenized_map["token_idx"].long())
        else:
            traj_pos= tokenized_map["traj_pos"]
            pos_pt=traj_pos[:,0,:2]
            orient_pt=traj_pos[:,0,2]
            relative_pos=(traj_pos[:,1:]-traj_pos[:,:1]).flatten(1,2)
            x_pt=self.token_emb(relative_pos)
            
        pl_type_mapping= torch.tensor([0,0,0,0,1,1,2,2,2,3,3,3]).to(device=pos_pt.device, dtype=torch.long)
        pl_type=pl_type_mapping[map_type]

        x_pt_categorical_embs = [
            self.type_pt_emb(map_type),  #
            self.polygon_type_emb(pl_type),  #
            # self.light_pl_emb(tokenized_map["light_type"].long()),#
        ]

        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)


        if self.gnn:
            orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

            edge_index_pt2pt = radiusGraphNearest(
                x=pos_pt,
                r=self.pl2pl_radius,
                batch=batch,
                loop=False,
                max_num_neighbors=self.pt2pt_neighbor,
            )
            rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
            rel_orient_pt2pt = wrap_angle(
                orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
            )
            r_pt2pt = torch.stack(
                [
                    torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                        nbr_vector=rel_pos_pt2pt[:, :2],
                    ),
                    rel_orient_pt2pt,
                ],
                dim=-1,
            )
            r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
            for i in range(self.num_layers):
                x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        # mask=torch.isin(tokenized_map["type"],torch.tensor([0,1,2,3,4,5]).to(batch.device))#9,,10
        #mask=(tokenized_map["type"]!=90) #& (tokenized_map["type"]!=4)#tensor([  589, 29076,  1180,  2036,  8661,  1502,  3782,  4237,  1011,  7563],
        #tensor([0.010, 0.488, 0.020, 0.034, 0.145, 0.025, 0.063, 0.071, 0.017, 0.127],
        # polyline_type = {
        #     # for lane
        #     "TYPE_FREEWAY": 0,
        #     "TYPE_SURFACE_STREET": 1,
        #     "TYPE_STOP_SIGN": 2,
        #     "TYPE_BIKE_LANE": 3,
        #     # for roadedge
        #     "TYPE_ROAD_EDGE_BOUNDARY": 4,
        #     "TYPE_ROAD_EDGE_MEDIAN": 5,
        #     # for roadline
        #     "BROKEN": 6,
        #     "SOLID_SINGLE": 7,
        #     "DOUBLE": 8,
        #     # for crosswalk, speed bump and drive way
        #     "TYPE_CROSSWALK": 9,
        # }

        return {
            "pt_token": x_pt,#[mask],
            "position": pos_pt,#[mask],
            "orientation": orient_pt,#[mask],
            "batch": batch,#[mask],
        }
        #
        # lengths = torch.bincount(batch).tolist()
        #
        # padded_pt_feature = padding(x_pt, lengths)
        #
        # feature_mask=(padded_pt_feature == 0).all(-1)
        #
        # map_mask = feature_mask[:, None]
        #
        # sinusoidal_pos=self.rotary_embedding(pos_pt, orient_pt)
        #
        # map_sinusoidal = padding(sinusoidal_pos, lengths)
        #
        # padd_pos=padding(pos_pt, lengths)
        # padd_heading=padding(orient_pt, lengths)
        #
        # # if not self.gnn:
        # #
        # #     pt2pt_mask = feature_mask[:, :, None] | feature_mask[:, None]
        # #
        # #     pt2pt_mask=nearest_mask(padd_pos,10,self.pl2pl_radius,pt2pt_mask)
        # #
        # #     padded_pt_feature = self.pt2pt_roformer(padded_pt_feature, pt2pt_mask[:,None], map_sinusoidal)
        # #
        # #     x_pt = padded_pt_feature[~feature_mask]
        #
        # return {
        #     "pt_token": x_pt,
        #     "padded_pt": padded_pt_feature,
        #     "position": pos_pt,
        #     "orientation": orient_pt,
        #     "padd_pos": padd_pos,
        #     "padd_heading": padd_heading,
        #
        #     # "centering_pos":centering_pos,
        #   #  "centering_heading":centering_heading,
        #     "rotary_embedding":self.rotary_embedding,
        #     "batch": batch,
        #     "map_mask": map_mask,
        #     "map_sinusoidal": map_sinusoidal
        # }

        #pos_pt1=pos_pt+torch.tensor(np.array([[10,100]])).to(device=x_pt.device)

        #orient_pt1=orient_pt+torch.pi*2


        # sinusoidal_pos = general_rope(pos_pt1, self.head_dim, orient_pt1)

        # map_sinusoidal1 = self.padding(sinusoidal_pos, lengths)

        # padd_pos=self.padding(pos_pt1, lengths)

        # pt2pt_dist=torch.linalg.norm(padd_pos[:,None]-padd_pos[:,:,None],dim=-1)

        # pt2pt_mask = map_mask | (pt2pt_dist>self.pl2pl_radius) | (pt2pt_dist==0)

        # x_pt1 = self.pt2pt_roformer(padded_pt_feature, pt2pt_mask[:,None], map_sinusoidal1)


