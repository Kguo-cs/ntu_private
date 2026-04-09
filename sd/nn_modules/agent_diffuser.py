import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from tensorflow.python.layers.core import dropout

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
            pred_lane=False,
            pred_lane_conn=False,
    ) -> None:
        super(MapDecoder, self).__init__()
        self.num_layers = num_layers

        self.lane_emb = MLPLayer(42,hidden_dim, hidden_dim)


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
        self.pred_lane_conn=pred_lane_conn
        self.pred_lane=pred_lane

        if self.pred_lane_conn:
            self.pred_lane_conn=MLPLayer(hidden_dim*3,hidden_dim, 5)
        elif self.pred_lane:
            self.output_layer=MLPLayer(hidden_dim,hidden_dim, 42)

            # self.con_emb = MLPLayer(5,hidden_dim, hidden_dim)
            #
            # self.pred_lane_conn=MLPLayer(hidden_dim*2,hidden_dim, 5)

        self.apply(weight_init)

    def forward(self, z_lane,batch,t_batch=None,l2l_edge_index=None):
        pos_pt=z_lane[:,:2]
        orient_pt=torch.atan2(z_lane[:,3],z_lane[:,2])

        x_pt=self.lane_emb(z_lane)

        if t_batch is not None:
            x_pt=x_pt+t_batch[batch]

        head_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        #l2l_feature=self.con_emb(z_lane_conn)

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
            l2l_edge_index=l2l_edge_index,
            l2l_feature=None
        )

        for i in range(self.num_layers):
            x_pt= self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)

        output={
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": batch,
        }

        if self.pred_lane_conn:
            lane_conn_logits = self.pred_lane_conn(
                torch.cat([x_pt[l2l_edge_index[0]], x_pt[l2l_edge_index[1]],r_pt2pt], dim=-1))

            return output,lane_conn_logits

        elif self.pred_lane:
            res = self.output_layer(x_pt)

            res_theta = torch.atan2(res[:, 3], res[:, 2])

            local_pos, local_theta = transform_to_global(
                res[:, :2],
                res_theta,
                pos_pt,
                orient_pt,
            )

            res = torch.cat(
                [local_pos, torch.cos(local_theta)[:, None], torch.sin(local_theta)[:, None], res[:, 4:]], dim=-1)


            con_pred=None
            # con_pred=self.pred_lane_conn( torch.cat([x_pt[l2l_edge_index[0]], x_pt[l2l_edge_index[1]]], dim=-1))
            return output, res, con_pred
        else:
            return output


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

        #self.attr_emb = MLPLayer(6,hidden_dim, hidden_dim)

        self.num_layers=num_layers
        ego_shape= torch.tensor( [[0,     0,     0,     1,     5.2860,     2.3320,    1.0000,     0.0000,     0.0000]])

        self.register_buffer("ego_shape", ego_shape)

        self.apply(weight_init)

    def forward(self, map_feature,z_agent,t_batch,batch):

        ego_mask = batch[1:] != batch[:-1]

        ego_mask = torch.cat([torch.ones_like(ego_mask[:1]), ego_mask])

        z_agent[ego_mask, :6] = self.ego_shape[:, :6]
        z_agent[ego_mask, -3:] = self.ego_shape[:, -3:]

        theta = torch.atan2(z_agent[:, 3], z_agent[:, 2])

        pos_s = z_agent[:, :2]

        feat_a = self.agent_emb(z_agent)#[:,:4]

        feat_a = feat_a + t_batch[batch]

        # feat_attr=self.attr_emb(z_agent[:,4:])

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
            max_radius=100,
            max_num_neighbors=60,
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
            #feat_a=feat_a+feat_attr

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

        res = torch.cat(
            [local_pos, torch.cos(local_theta)[:, None], torch.sin(local_theta)[:, None], res[:, 4:]], dim=-1)

        return res

from sd.utils.dit_layers import FactorizedDiTBlock, FinalLayer, LabelEmbedder, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, TwoLayerResMLP
from sd.utils.pyg_helpers import get_indices_within_scene

class Agent_Diffuser(nn.Module):
    """Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(Agent_Diffuser, self).__init__()
        hidden_dim=128

        num_freq_bands=hidden_dim//2

        num_heads=8

        head_dim=hidden_dim//num_heads

        dropout=0.1

        num_layers=3

        self.map_encoder=MapDecoder(
            hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pred_lane=True
        )

        self.connect_encoder=MapDecoder(
            hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pred_lane_conn=True
            )
        self.agent_encoder=AgentDecoder(
            hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            )

        #self.t_embed=MLPLayer(1,hidden_dim,hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

       # self.t_embedder = TimestepEmbedder(self.cfg_model.hidden_dim)
        self.num_agents_embedder = LabelEmbedder(30 + 1, hidden_dim, 0)
        self.num_lanes_embedder = LabelEmbedder(100 + 1, hidden_dim, 0)
        self.scene_type_embedder = LabelEmbedder(2 * 2, hidden_dim, 0)

        self.apply(weight_init)

    def predict_con(self,x_lane,l2l_edge_index,lane_batch,embed):

        map_feature,lane_conn_logits=self.connect_encoder(x_lane,lane_batch,t_batch=embed,l2l_edge_index=l2l_edge_index)

        return map_feature,lane_conn_logits

    def pred_agent(self,z_agent,t_batch,agent_batch,c):

        map_feature,embed=c

        t_batch = self.t_embedder(t_batch.reshape(-1))

        agent_pred = self.agent_encoder(map_feature, z_agent, t_batch + embed, agent_batch)

        return agent_pred

    def pred_lane(self, z_lane, t_batch, lane_batch, c):

        l2l_edge_index, embed = c

        t_batch = self.t_embedder(t_batch.reshape(-1))

        _, lane_pred, con_pred = self.map_encoder(z_lane, lane_batch, t_batch + embed ,    l2l_edge_index=l2l_edge_index)

        return lane_pred

    def forward(self, z_agent,z_lane,x_lane,l2l_edge_index,t_batch,agent_batch,lane_batch,scene_idx):

        t_batch = self.t_embedder(t_batch.reshape(-1))

        num_agents = torch.bincount(agent_batch)
        num_lanes = torch.bincount(lane_batch)

        num_agents_emb = self.num_agents_embedder(num_agents, train=self.training)
        num_lanes_emb = self.num_lanes_embedder(num_lanes, train=self.training)

        scene_type = self.scene_type_embedder(scene_idx.long(), train=self.training)#, force_drop_ids=torch.ones_like(scene_idx))

        _,lane_pred,_=self.map_encoder(z_lane,lane_batch,t_batch=t_batch+num_lanes_emb+scene_type,l2l_edge_index=l2l_edge_index)

        map_feature,con_pred=self.connect_encoder(x_lane,lane_batch,t_batch=num_lanes_emb+scene_type,l2l_edge_index=l2l_edge_index)

        agent_pred=self.agent_encoder(map_feature,z_agent,t_batch+num_agents_emb+scene_type,agent_batch)


        return agent_pred,lane_pred,con_pred




