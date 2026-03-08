import math
import copy
import numpy as np

from typing import Dict, Mapping, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli

from .transformer_decoder import TransformerDecoderLayerDiff,sinusoidal_embedding

from src.smart.layers.fourier_embedding import FourierEmbedding

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init
)
import warnings
from torch.nn.modules.container import ModuleList
import copy
from src.smart.layers.relative_transformer import RoFormerBlock,RoFormerDecoder
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.func import functional_call, jvp
from src.smart.utils.cluster import batch_increasing_schedule,allocate_k_per_type
from src.smart.utils.earth_match import get_closest_sum_idx
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges,project_to_local_frame
from src.smart.layers.relative_transformer import padding
from torch_geometric.nn.pool import knn_graph,knn
# from lpips import LPIPS
from src.smart.utils.cluster import cluster_point_per_type
from torch_scatter import scatter_sum

class InitDenoiser(nn.Module):

    def __init__(self,
                 token_processor,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 init_timestep: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 diff_type: str,
                 m_dim: int,
                 mean_flow
                 ) -> None:
        super(InitDenoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim

        self.use_roformer=True
        self.use_padding=True
        self.use_all_type=False

        if self.use_all_type:
            # self.type_a_emb = MLPLayer(3, hidden_dim, hidden_dim)
            m_delta_dim = 11
        else:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            m_delta_dim = 5+3


        self.use_graph=True
        self.ego_rel = True
        self.use_scale=True
        noise_dim = 1
        if mean_flow:
            noise_dim = 2
            self.use_padding = True

        self.use_all_pos=token_processor.use_all_pos

        if self.use_all_pos:
            m_delta_dim=m_delta_dim+90*4-2

        self.output_dim=m_delta_dim

        self.normal_scale=None
        self.normal_mean=None

        if self.use_roformer:
            self.noise_embedding = MLPLayer(1, hidden_dim, hidden_dim)
            if self.ego_rel:
                self.proj_in_m_delta = nn.Linear(m_delta_dim-4, self.hidden_dim)#MLPLayer(m_delta_dim-4, hidden_dim, hidden_dim)#
            else:
                self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)#MLPLayer(m_delta_dim, hidden_dim, hidden_dim)#

            self.to_out_m_delta= MLPLayer(hidden_dim, hidden_dim, m_delta_dim)

            if self.use_graph:
                self.use_padding = False

                self.edge_encoder = EdgeEncoder(hidden_dim,
                                                num_freq_bands,
                                                use_a2a=True,
                                                use_pl2a=True
                                                )


                self.ego_encoder = EdgeEncoder(hidden_dim,
                                                num_freq_bands,
                                                use_pl2a=True,
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

                self.a2ego_attn_layers = nn.ModuleList(
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


            else:
                module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
                self.entry_formers = ModuleList([copy.deepcopy(module) for i in range(num_layers)])

        else:

            self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)

            self.proj_in_m_delta_2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.proj_out_m_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

            self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)

            # self.interact_pt2m = nn.ModuleList(
            #     [TransformerDecoderLayerDiff(
            #         n_embd=hidden_dim,
            #         n_head=num_heads,
            #         ff_dim=4 * hidden_dim,
            #         dropout=0,
            #         layer_id=i,
            #     ) for i in range(num_layers)])
            module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
            self.interact_pt2m = ModuleList([copy.deepcopy(module) for i in range(num_layers)])
            # self.interact_pt2m = nn.ModuleList([
            #     nn.TransformerDecoderLayer(
            #         d_model=hidden_dim,
            #         nhead=num_heads,
            #         dim_feedforward=4 * hidden_dim,
            #         dropout=0.0,
            #         batch_first=True,
            #         activation="gelu",
            #         norm_first=True
            #     )
            #     for _ in range(num_layers)
            # ])
            self.to_out_m_delta = SkipMLP(d_model=hidden_dim)

        self.apply(weight_init)

    def normalize(self,input):
        return (input - self.normal_mean) / self.normal_scale

    def denormalize(self,input):
        return input* self.normal_scale+self.normal_mean

    def normalize_z(self,z):
        m_delta = z[:, 0]

        m_delta = self.denormalize(m_delta)

        m_delta[:, 2:4] = m_delta[:, 2:4] / torch.linalg.norm(m_delta[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)

        z = self.normalize(m_delta)

        return z[:,None]


    def padding(self, pos, heading, feature, batch, batch_num):
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        mask = torch.any(padding_features_a != 0, dim=-1)

        return padding_pos_a, padding_heading_a, padding_features_a,mask

    def get_output(self, pred_init, tokenized_agent, non_ego, batch, ego_position, ego_heading, gt_initial_pos, gt_initial_heading):
        pred_init = self.denormalize(pred_init)

        pred_trans, pred_head, pred_shape, pred_vel = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4:6], \
        pred_init[..., 6:]
        pred_head = torch.atan2(pred_head[..., 1], pred_head[..., 0])
        shape = tokenized_agent["initial_shape"].clone()

        shape[non_ego, :2] = pred_shape[:, :2]

        tokenized_agent["shape"] = shape

        global_pos,global_heading=transform_to_global(
            pred_trans,
            pred_head,
            ego_position[batch],
            ego_heading[batch],
        )

        if self.use_all_pos:

            all_posHeading=pred_vel.reshape(-1,90,4)

            all_pos=all_posHeading[:,:,:2]

            all_heading=torch.atan2(all_posHeading[:,:,3],all_posHeading[:,:,2])

            all_pos,all_heading=transform_to_global(
                all_pos,
                all_heading,
                ego_position[batch],
                ego_heading[batch],
            )

            gt_initial_pos=torch.cat([all_pos[:,:10],global_pos[:,None],all_pos[:,10:]],dim=1)

            gt_initial_heading=torch.cat([all_heading[:,:10],global_heading[:,None],all_heading[:,10:]],dim=1)

            gt_initial_vel= (gt_initial_pos[:,11]-gt_initial_pos[:,10])/0.1
            gt_initial_idx=None
        else:

            global_pred_vel=rotate_to_global(pred_vel,ego_heading[batch])

            gt_initial_pos[non_ego]=global_pos
            gt_initial_heading[non_ego]=global_heading

            gt_initial_vel=tokenized_agent["initial_vel"].clone()

            gt_initial_vel[non_ego] =global_pred_vel

            rel_vel=rotate_to_local(gt_initial_vel,gt_initial_heading)

            center_token_traj = tokenized_agent["token_traj"].mean(-2)

            if pred_init.shape[-1]==10:
                pred_vel_heading=torch.atan2(pred_head[..., 9], pred_head[..., 8])

                vel_heading = torch.atan2(rel_vel[:, 1], rel_vel[:, 0])

                vel_heading[non_ego]=wrap_angle(pred_vel_heading+ego_heading[batch]-gt_initial_heading[non_ego])

                pred_pos=transform_to_global(
                    center_token_traj,#.flatten(1, 2)
                    None,
                    - rel_vel*0.5,
                    vel_heading,
                )[0].reshape(center_token_traj.shape)


                #
                # # static_token=center_token_traj[:,0]
                # #
                # # gt_initial_idx=torch.linalg.norm(static_token[:,None]-pred_pos,dim=-1).sum(-1).argmin(-1)
                #
                #
                gt_initial_idx = torch.linalg.norm(pred_pos, dim=-1).argmin(-1)
            else:
                gt_initial_idx = torch.linalg.norm(center_token_traj - rel_vel[:, None] * 0.5, dim=-1).argmin(-1)

            gt_initial_pos,gt_initial_heading,gt_initial_idx=gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None]


        return gt_initial_pos,gt_initial_heading,shape,gt_initial_vel,gt_initial_idx

    def get_input(self,tokenized_agent,non_ego,nonego_batch,nonego_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading):

        batch_ego_pos=ego_position[nonego_batch]
        batch_ego_heading=ego_heading[nonego_batch]

        initial_shape = tokenized_agent["initial_shape"][non_ego]

        non_ego_pos=gt_initial_pos[non_ego]
        non_ego_head=gt_initial_heading[non_ego]

        init_trans, real_heading = transform_to_local(non_ego_pos,
                                                    non_ego_head,
                                                    batch_ego_pos,
                                                    batch_ego_heading,
                                                    )

        delta_rot = real_heading.unsqueeze(-1)

        init_angle = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)  # [0,2]

        if self.use_all_pos:
            local_pos,local_heading = transform_to_local(tokenized_agent["all_pos"][non_ego],
                                           tokenized_agent["all_heading"][non_ego],
                                           batch_ego_pos,
                                           batch_ego_heading)

            local_vel=torch.cat([local_pos,local_heading.cos()[:,:,None],local_heading.sin()[:,:,None]],dim=-1)

            tokenized_agent["nonego_valid_mask"]=tokenized_agent["valid_mask"][non_ego]

            local_vel[~tokenized_agent["nonego_valid_mask"]]=0

            local_vel=local_vel.flatten(1,2)

            valid=tokenized_agent["nonego_valid_mask"][:,:,None].repeat(1,1,4).flatten(1,2)

            tokenized_agent["nonego_valid"]=torch.cat([torch.ones_like(valid[:,:6]),valid],dim=-1).to(torch.float32)
        else:
           local_vel = rotate_to_local(tokenized_agent["initial_vel"][non_ego],  ego_heading[nonego_batch])

        m_init = torch.cat([init_trans, init_angle, initial_shape[:, :2], local_vel], dim=-1)

        if 'prev_heading' in tokenized_agent.keys():
            prev_heading = wrap_angle(tokenized_agent["prev_heading"][non_ego],ego_heading[nonego_batch])

            m_init = torch.cat([m_init, prev_heading.cos()[:,None],prev_heading.sin()[:,None]], dim=-1)

        tokenized_agent['nonego_type_sorted'] = nonego_type

        if self.use_scale:
            old_nonego_type_sorted = tokenized_agent["nonego_type_sorted"].clone()

            if self.use_all_type:
                one_hot = F.one_hot(old_nonego_type_sorted, num_classes=tokenized_agent["type_counts"].shape[-1])

                m_init = torch.cat([m_init, one_hot], dim=-1)

            diff_input, m_init , nonego_batch= cluster_point_per_type(m_init, nonego_batch, tokenized_agent)

            if self.normal_mean is None:
                valid = ~torch.isnan(m_init)
                count = valid.sum(0, keepdim=True).clamp_min(1)

                mean = torch.where(valid, m_init, 0).sum(0, keepdim=True) / count
                std = torch.sqrt(torch.where(valid, (m_init - mean) ** 2, 0).sum(0, keepdim=True) / count).clamp_min(
                    1e-8)

                self.normal_mean = mean
                self.normal_scale = std
                # self.normal_mean = torch.nanmean(m_init, dim=0,keepdim=True)
                # self.normal_scale = torch.nanstd(m_init, dim=0,keepdim=True)

            m_init = self.normalize(m_init)
            diff_input = self.normalize(diff_input)
        else:
            diff_input = m_init

        return diff_input,m_init,nonego_batch


    def forward(self,
                m_delta,
                beta,
                tokenized_agent: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                eval_mask,
                num_samples=1,
                mode=0
                ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        batch = tokenized_agent["nonego_batch"]
        type = tokenized_agent["nonego_type_sorted"]
        num_graphs = tokenized_agent["num_graphs"]
        ego_embedding = tokenized_agent["ego_embedding"]

        if self.use_roformer:
            m_delta=m_delta[:,0]

            m_delta=self.denormalize(m_delta)

            m_delta[:, 2:4] =m_delta[:, 2:4]/ torch.linalg.norm(m_delta[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)

            if self.ego_rel:
                feat_a=self.proj_in_m_delta(m_delta[:,4:])
            else:
                feat_a=self.proj_in_m_delta(m_delta)

            #beta_emb_m = self.noise_embedding(beta,categorical_embs=self.type_a_emb(type))
            beta_emb_m = self.noise_embedding(beta) +self.type_a_emb(type)#

            feat_a = feat_a + beta_emb_m

            theta=torch.atan2(m_delta[:,3],m_delta[:,2])

            pos_s=m_delta[:, :2]

            if self.use_graph:

                # # number of agents per batch
                # counts = torch.bincount(batch)
                #
                # # first index of each batch
                # first_idx = torch.cumsum(counts, dim=0) - counts
                #
                # ego_tokens = ego_embedding[first_idx]  # (B, A)
                # B = ego_tokens.shape[0]
                # N = feat_a.shape[0]
                #
                # new_feature = torch.cat([feat_a, ego_tokens], dim=0)
                # new_batch = torch.cat([batch, batch[first_idx]], dim=0)
                #
                # ego_theta=torch.atan2(self.normal_mean[:,3],self.normal_mean[:,2])
                #
                # pos_s=torch.cat([pos_s, torch.zeros_like(ego_tokens[:,:2])+self.normal_mean[:,:2]], dim=0)
                # theta=torch.cat([theta, torch.zeros_like(ego_tokens[:,0])+ego_theta], dim=0)
                #
                # order = torch.argsort(new_batch)
                # feat_a = new_feature[order]
                # batch = new_batch[order]
                # theta=theta[order]
                # pos_s=pos_s[order]
                #
                # # mask BEFORE sorting
                # non_ego_mask = torch.cat([
                #     torch.ones(N, dtype=torch.bool, device=batch.device),
                #     torch.zeros(B, dtype=torch.bool, device=batch.device)
                # ], dim=0)
                # non_ego_mask = non_ego_mask[order]

                pos_pl, orient_pl, batch_pl, feat_map=scene_enc

                head_vector_s = torch.stack([theta.cos(), theta.sin()], dim=-1)

                edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                    pos_pl=pos_pl,  # [n_pl, 2]
                    orient_pl=orient_pl,  # [n_pl]
                    pos_a=pos_s,  # [n_agent, n_step, 2]
                    head_a=theta,  # [n_agent, n_step]
                    head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
                    mask=None,  # [n_agent, n_step]
                    batch_s=batch,  # [n_agent,n_step]
                    batch_pl=batch_pl,  # [n_pl*n_step]
                    pl2a_radius=40,
                    max_num_neighbors=20,
                    agent_train_mask=None,
                    layer_num=self.num_layers
                )

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

                ego_theta = torch.atan2(self.normal_mean[:, 3], self.normal_mean[:, 2]).repeat(num_graphs)
                ego_pos=self.normal_mean[:,:2].repeat(num_graphs,1)

                ego_batch=torch.arange(num_graphs).to(device)

                edge_index_ego2a, r_ego2a = self.ego_encoder.build_map2agent_edge(
                    pos_pl=ego_pos,  # [n_pl, 2]
                    orient_pl=ego_theta,  # [n_pl]
                    pos_a=pos_s,  # [n_agent, n_step, 2]
                    head_a=theta,  # [n_agent, n_step]
                    head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
                    mask=None,  # [n_agent, n_step]
                    batch_s=batch,  # [n_agent,n_step]
                    batch_pl=ego_batch,  # [n_pl*n_step]
                    pl2a_radius=1000,
                    max_num_neighbors=1,
                    agent_train_mask=None,
                    layer_num=self.num_layers
                )

                #
                # rel_pos_a2ego = pos_s-ego_pos
                # rel_head_a2ego = wrap_angle(theta-ego_theta)
                #
                # r_a2ego = torch.cat(
                #     [
                #         project_to_local_frame(rel_pos_a2ego, head_vector_s, False),
                #         rel_head_a2ego[:, None],
                #     ],
                #     dim=-1,
                # )
                #
                # r_a2ego = self.ego_encoder.r_a2a_emb(continuous_inputs=r_a2ego, categorical_embs=None)#+ego_embedding

                # edge_index_a2ego=torch.arange(len(feat_a))

                # feat_a = feat_a + r_a2ego

                for layer_i in range(self.num_layers):

                    feat_a = self.a2ego_attn_layers[layer_i]((ego_embedding, feat_a), r_ego2a, edge_index_ego2a)

                    feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                    feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

            else:
                pos_pl, orient_pl,map_mask, map_emb=scene_enc

                # pos_pl,orient_pl,feat_map,map_mask = self.padding(pos_pl, orient_pl, feat_map, batch_pl, batch_size)  # b, n, d

                pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.padding(pos_s, theta, feat_a, batch,
                                                                        batch_size)  # b, n, d

                pos_emb = sinusoidal_embedding(feat_a_b.shape[1], self.hidden_dim).to(device).unsqueeze(0)

                feat_a_b=feat_a_b+pos_emb

                # pos_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], 2, device=type.device)
                # heading_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], device=type.device)

                for mod in self.entry_formers:
                    feat_a_b = feat_a_b + beta_emb_m

                    feat_a_b = mod(feat_a_b, pos_a_b,
                                   heading_a_b, mask_a_b,
                                   map_emb,
                                   pos_pl,
                                   orient_pl, map_mask
                                   )

                feat_a = feat_a_b[mask_a_b]

            res=self.to_out_m_delta(feat_a)

            res=self.denormalize(res)

            res_theta=torch.atan2(res[:,3],res[:,2])

            global_pos,global_theta = transform_to_global(
                res[:,:2],
                res_theta,
                pos_s,
                theta,
            )

            local_vel=res[:,6:]

            if self.use_all_pos:
                loca_posHead=local_vel.reshape(-1, 90, 4)

                loca_heading=torch.atan2(loca_posHead[:,:,3],loca_posHead[:,:,2])

                global_vpos,global_vheading=transform_to_global(
                    loca_posHead[:,:,:2],
                    loca_heading,
                    pos_s,
                    theta,
                )

                global_vel=torch.cat([global_vpos,torch.cos(global_vheading)[:,:,None],torch.sin(global_vheading)[:,:,None]],dim=-1).flatten(1,2)
            else:
                global_vel=rotate_to_global(local_vel,theta)

            if res.shape[-1]==10:
                res_heading = torch.atan2(res[:, 9], res[:, 8])

                new_heading=wrap_angle(res_heading+theta)

                res = torch.cat(
                    [global_pos, torch.cos(global_theta)[:, None], torch.sin(global_theta)[:, None], res[:, 4:6],
                     global_vel,torch.cos(new_heading)[:, None], torch.sin(new_heading)[:, None],], dim=-1)[:, None]

            else:
                res = torch.cat(
                    [global_pos, torch.cos(global_theta)[:, None], torch.sin(global_theta)[:, None], res[:, 4:6],
                     global_vel], dim=-1)[:, None]

            # cos_d = res[:, 2]
            # sin_d = res[:, 3]
            #
            # cos_t = torch.cos(theta)
            # sin_t = torch.sin(theta)
            #
            # cos_global = cos_t * cos_d - sin_t * sin_d
            # sin_global = sin_t * cos_d + cos_t * sin_d
            #
            # global_pos, _ = transform_to_global(
            #     res[:, :2],
            #     None,
            #     pos_s,
            #     theta,
            # )
            #
            # res = torch.cat([
            #     global_pos,
            #     cos_global[:, None],
            #     sin_global[:, None],
            #     res[:, 4:]
            # ], dim=-1)[:, None]

            res=self.normalize(res)
        else:
            beta_emb = self.noise_emb(beta)
            # num_agents x 128

            m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)

            if self.use_all_type:
                m_delta = m_delta +ego_embedding

            else:
                m_delta = m_delta + self.type_a_emb(type)+ego_embedding
            m_delta = self.proj_in_m_delta_2(m_delta)

            self.num_samples = num_samples

            if self.use_padding:
                pos_pl, orient_pl,map_mask, map_emb=scene_enc

                lengths=tokenized_agent["lengths"]

                #lengths = torch.bincount(batch, minlength=batch_size).tolist()

                m_delta = padding(m_delta, lengths, padding_value=0)  # b, n, d

                mask_agent = torch.any(m_delta != 0, dim=-1)

                beta_emb_m= padding(beta_emb, lengths, padding_value=0)

            else:
                pos_pl, orient_pl, batch_pl, feat_map=scene_enc

                x_pt = feat_map#.repeat(self.num_samples, 1)
                map_batch_list = batch_pl

                poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
                map_emb_batch = torch.split(x_pt, poly_cnt_per_batch.tolist())

                map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0)

                agent_cnt_per_batch = batch.bincount(minlength=batch_size)
                agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())

                m_delta = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)

                beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())

                beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0)

            pos_emb = sinusoidal_embedding(m_delta.shape[1], self.hidden_dim).to(device).unsqueeze(0)
            m_delta += pos_emb

            B, N, D = m_delta.shape
            B, N_map, _ = map_emb.shape

            if self.use_padding:
                attn_mask_agent_layers = ~mask_agent
                attn_mask_map_layers = ~map_mask
            else:
                #mask_map_layers = []
                #mask_agent_layers = []

                attn_mask_map_layers = []
                attn_mask_agent_layers = []

                for i in range(batch_size):
                   # mask_attn_map_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    #mask_attn_map_agent_i = mask_attn_map_agent_i.unsqueeze(-1).expand(-1, N_map)
                    mask_attn_map_pt_i = torch.arange(N_map).to(m_delta.device) < poly_cnt_per_batch[i]
                    attn_mask_map_layers.append(mask_attn_map_pt_i)
                    #mask_attn_map_pt_i = mask_attn_map_pt_i.unsqueeze(0).expand(N, -1)

                    #mask_attn_i = mask_attn_map_agent_i & mask_attn_map_pt_i
                    # mask_map_layers.append(mask_attn_i)

                    mask_attn_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    attn_mask_agent_layers.append(mask_attn_agent_i)
                    # mask_attn_agent_i = mask_attn_agent_i.unsqueeze(-1).expand(-1, N)
                    # mask_attn_i = mask_attn_agent_i & mask_attn_agent_i.t()
                    # mask_agent_layers.append(mask_attn_i)

                attn_mask_agent_layers = ~torch.stack(attn_mask_agent_layers)
                attn_mask_map_layers = ~torch.stack(attn_mask_map_layers)
                mask_agent = torch.arange(N).expand(B, N).to(m_delta.device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
                #mask_agent = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]


            # attn_mask_agent_layers1 = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
            # attn_mask_map_layers1 = attn_mask_map_layers.view(B, 1, 1, N_map). \
            #     expand(-1, self.num_heads * 2, N, -1)
            # #
            # # 0: don't attend others
            # if mode == 0:
            #     attn_mask_agent_layers = attn_mask_agent_layers + ~torch.eye(N).to(torch.bool).unsqueeze(0).to(
            #         m_delta.device)

            for i in range(self.num_layers):
                m_delta = m_delta + beta_emb_m

                # m_delta = self.interact_pt2m[i](
                #     tgt=m_delta,
                #     memory=map_emb,
                #     tgt_key_padding_mask=attn_mask_agent_layers,
                #     memory_key_padding_mask=attn_mask_map_layers
                # )

                # m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                #                                 mask=attn_mask_agent_layers1,
                #                                 map_mask=attn_mask_map_layers1)
                m_delta = self.interact_pt2m[i](m_delta, torch.zeros_like(m_delta[:,:,:2]),
                               torch.zeros_like(m_delta[:,:,0]), ~attn_mask_agent_layers,
                               map_emb,
                               torch.zeros_like(map_emb[:,:,:2]),
                               torch.zeros_like(map_emb[:,:,0]), ~attn_mask_map_layers
                               )

            m_out_delta = m_delta[mask_agent]#.view(-1, D)  # [sum(agent_cnt_per_batch), D]

            out_m_delta = self.to_out_m_delta(m_out_delta)
            out_m_delta = out_m_delta.view(-1, self.num_samples, self.hidden_dim)
            res=self.proj_out_m_delta(out_m_delta)
        return res




class SkipMLP(torch.nn.Module):
    def __init__(self, d_model=128, act_layer=nn.GELU):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.ac = act_layer()
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x):
        out = x + self.ac(self.linear(x))
        out = self.norm2(x + self.norm1(out))
        return out
