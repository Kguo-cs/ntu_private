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
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, padding
from .build_edge import radiusGraphNearest
from .edge_encoder import EdgeEncoder
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors
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
        pt2pt_neighbor:int,
        token_processor
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers
        self.pt2pt_neighbor=pt2pt_neighbor

        self.token_processor=token_processor

        self.type_pt_emb = nn.Embedding(10, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        # if not self.token_processor.pred_light:
        #     self.light_pl_emb = nn.Embedding(5, hidden_dim)


        self.head_dim=head_dim

        # map_token_traj_src: [n_token, 11, 2].flatten(0,1)
        self.my_map=False

        if self.my_map:
            self.token_emb = MLPEmbedding(input_dim=4, hidden_dim=hidden_dim)
        else:
            self.token_emb = MLPEmbedding(input_dim=22, hidden_dim=hidden_dim)
        #self.token_emb = nn.Embedding(token_processor.n_token_map, hidden_dim)

        self.edge_encoder = EdgeEncoder(hidden_dim,num_freq_bands,share=False,a2a=False)

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

        self.pred_offroad=False

        self.pred_map=True

        if self.pred_map:
            self.token_size = 1024
            self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                               output_dim=self.token_size)

        self.apply(weight_init)

    def forward(self, tokenized_map: Dict):

        map_type=tokenized_map["type"].long()
        map_type[map_type>9] = 9
        
        # mask = torch.zeros_like(map_type, dtype=bool)
        # #mask = torch.ones_like(map_type, dtype=bool)
        #
        # type4_indices=torch.where((map_type==4) |(map_type==5))[0]
        #
        # sampled_indices = type4_indices[::2]
        #
        # mask[sampled_indices] = True
        #
        # mask[(map_type!=4)&(map_type!=5)] = True
        #
        map_type[map_type==5]=4
        map_type[map_type==8]=7
        map_type[map_type==0]=1
        map_type[map_type==2]=1
        map_type[map_type==3]=1
        #
        mask=(map_type==4) | (map_type==6) | (map_type==7)  | (map_type==9) | (map_type==1)

        batch = tokenized_map["batch"][mask]
        pos_pt = tokenized_map["position"][mask]
        orient_pt = tokenized_map["orientation"][mask]
        token_idx=tokenized_map["token_idx"].long()[mask]
        map_type=map_type[mask]

        mask =(map_type==4) | (map_type==6) | (map_type==7)  | (map_type==1)  #| (map_type==9)# |#|  (map_type==4) | # | (map_type==7) #(map_type == 4) | (map_type == 5)

        if self.pred_offroad:

            edge_mask=map_type == 4#mask #
            token_edge_idx=token_idx[edge_mask]
            pos_edge=pos_pt[edge_mask]
            orient_edge=orient_pt[edge_mask]
            batch_edge=batch[edge_mask]

            local_traj=self.token_processor.map_token_traj_src[token_edge_idx]

            global_edge,_=transform_to_global(pos_local=local_traj.reshape(-1,11,2), head_local=None, pos_now=pos_edge, head_now=orient_edge)


            # edge_type=map_type[edge_mask]
            #
            # mid_road_edge=global_edge[edge_type==5]
            #
            # if len(mid_road_edge):
            #
            #     mid_road_edge_last=mid_road_edge[:,-1]
            #     mid_road_edge_start=mid_road_edge[:,0]
            #
            #     dist_to_other_seg=torch.linalg.norm(mid_road_edge_last[:,None]-mid_road_edge_start[None],dim=-1)
            #
            #     mask=(dist_to_other_seg>0.5) & (dist_to_other_seg<5)#there is a gap
            #
            #     new_start=mid_road_edge_last[mask]
            #     new_end=mid_road_edge_start[mask]
            #
            #     inter_seg=torch.stack((new_start,new_end),dim=1)
            #
            #     import matplotlib as mpl
            #
            #     mpl.rcParams['toolbar'] = 'None'
            #     import matplotlib.pyplot as plt
            #
            #     global_edge=global_edge[edge_type==4].cpu().detach().numpy()
            #
            #     mid_road_edge=mid_road_edge.cpu().detach().numpy()
            #     inter_seg=inter_seg.cpu().detach().numpy()
            #     for i in range(len(global_edge)):
            #        plt.plot(global_edge[i,:,0],global_edge[i,:,1],'r')
            #     for j in range(len(mid_road_edge)):
            #        plt.plot(mid_road_edge[j,:,0],mid_road_edge[j,:,1],'g')
            #        plt.plot(mid_road_edge[j,:2,0],mid_road_edge[j,:2,1],'r')
            #
            #     for j in range(len(inter_seg)):
            #        plt.plot(inter_seg[j,:,0],inter_seg[j,:,1],'b')
            #
            #     plt.show()

            #global_edge[edge_type==5]=global_edge[edge_type==5].flip(dims=[1])

            tokenized_map["global_edge"]=global_edge
            tokenized_map["batch_edge"]=batch_edge



        if self.my_map:
            traj_pos_local=tokenized_map["traj_pos_local"].flatten(1,2)
            x_pt = self.token_emb(traj_pos_local)
        else:
            pt_token_emb_src = self.token_emb(self.token_processor.map_token_traj_src)
            x_pt = pt_token_emb_src[token_idx]

        pl_type_mapping= torch.tensor([0,0,0,0,1,1,2,2,2,3,3,3]).to(device=pos_pt.device, dtype=torch.long)
        pl_type=pl_type_mapping[map_type]

        x_pt_categorical_embs = [
            self.type_pt_emb(map_type),
            self.polygon_type_emb(pl_type),
            # self.light_pl_emb(tokenized_map["light_type"].long()),#
        ]

        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)


        # x_pt=x_pt[~mask]
        # pos_pt=pos_pt[~mask]
        # orient_pt=orient_pt[~mask]
        # batch=batch[~mask]

        if self.num_layers>1:
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
                x_pt ,_= self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)


            x_pt=x_pt[mask]
            pos_pt=pos_pt[mask]
            orient_pt=orient_pt[mask]
            batch=batch[mask]

        else:

            edge_pt=x_pt[mask]#[::2]
            pos_edge=pos_pt[mask]#[::2]
            orient_edge=orient_pt[mask]#[::2]
            batch_edge=batch[mask]#[::2].contiguous()

            head_vector_edge = torch.stack([orient_edge.cos(), orient_edge.sin()], dim=-1)

            edge_index_pt2pt, r_pt2pt = self.edge_encoder.build_map2map_edge(
                                                            pos_pt,  # [n_pl, 2]
                                                            orient_pt,  # [n_pl]
                                                            pos_edge,  # [n_agent, n_step, 2]
                                                            orient_edge,  # [n_agent, n_step]
                                                            head_vector_edge,  # [n_agent, n_step, 2]
                                                            batch_edge,  # [n_agent*n_step]
                                                            batch,  # [n_pl*n_step]
                                                            self.pl2pl_radius,
                                                            self.pt2pt_neighbor,
                                                        )

            edge_pt = self.pt2pt_layers[0]((x_pt, edge_pt), r_pt2pt, edge_index_pt2pt)

            x_pt=edge_pt
            pos_pt=pos_edge
            orient_pt=orient_edge
            batch=batch_edge



        output={
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": batch,
        }

        if self.pred_map:
            pt_pred_mask = tokenized_map['pt_pred_mask']

            next_token_prob = self.token_predict_head(x_pt[pt_pred_mask])

            output['map_next_token_prob']=next_token_prob


        return output

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


