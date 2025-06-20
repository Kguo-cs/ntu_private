from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from torch.distributions import Categorical
from .build_edge import radiusGraphNearest2,nearest_mask,generate_limited_causal_mask,nearest_mask2, \
    radiusGraphNearest_head,radiusGraphNearest_inv
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.utils.rollout import cal_polygon_contour
from src.smart.loss.gmm_dist import  GMM_Dist
from src.smart.loss.iq_loss import padding
from src.smart.modules.light_encoder import LightEncoder


class EdgeEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int
    ) -> None:
        super(EdgeEncoder, self).__init__()
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3

        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

    def build_interaction_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            batch_s,  # [n_agent*n_step]
            mask,  # [n_agent, n_step]
            max_num_neighbors,
            max_radius,
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        edge_index_a2a = radiusGraphNearest_head(x=pos_s[:, :2],
                                                 x_heading=head_s,
                                                 r=max_radius,
                                                 batch=batch_s,
                                                 loop=False,
                                                 max_num_neighbors=max_num_neighbors)

        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
            self,
            pos_pl,  # [n_pl, 2]
            orient_pl,  # [n_pl]
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            mask,  # [n_agent, n_step]
            batch_s,  # [n_agent*n_step]
            batch_pl,  # [n_pl*n_step]
            pl2a_radius,
            max_num_neighbors
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radiusGraphNearest2(x=pos_s[:, :2],
                                              y=pos_pl[:, :2],
                                              x_heading=head_s,
                                              r=pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=20)

        # edge_index_pl2a = radiusGraphNearest_inv(x=pos_s[:, :2],
        #                                       y=pos_pl[:, :2],
        #                                       r=self.pl2a_radius,
        #                                       batch_x=batch_s,
        #                                       batch_y=batch_pl,
        #                                       max_num_neighbors=self.pt2a_neighbor)

        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )

        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        return edge_index_pl2a, r_pl2a


    #
    # def forward(self, tokenized_agent,map_feature, ):
    #
    #
    #     batch_s = torch.cat(
    #         [
    #             tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
    #             for t in range(n_step)
    #         ],
    #         dim=0,
    #     )  # [n_agent*n_step]
    #
    #     batch_pl = torch.cat(
    #         [
    #             map_feature["batch"] + tokenized_agent["num_graphs"] * t
    #             for t in range(n_step)
    #         ],
    #         dim=0,
    #     )  # [n_pl*n_step]
    #
    #     edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
    #         pos_pl=map_feature["position"],  # [n_pl, 2]
    #         orient_pl=map_feature["orientation"],  # [n_pl]
    #         pos_a=pos_a,  # [n_agent, n_step, 2]
    #         head_a=head_a,  # [n_agent, n_step]
    #         head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
    #         mask=mask,  # [n_agent, n_step]
    #         batch_s=batch_s,  # [n_agent*n_step]
    #         batch_pl=batch_pl,  # [n_pl*n_step]
    #         pl2a_radius=self.pl2a_radius,
    #         max_num_neighbors=self.pt2a_neighbor
    #     )
    #
    #     edge_index_a2a, r_a2a = self.edge_encoder.build_interaction_edge(
    #         pos_a=pos_a,  # [n_agent, n_step, 2]
    #         head_a=head_a,  # [n_agent, n_step]
    #         head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
    #         batch_s=batch_s,  # [n_agent*n_step]
    #         mask=mask,  # [n_agent, n_step]
    #         max_radius=self.a2a_radius,
    #         max_num_neighbors=self.a2a_neighbor
    #     )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]
    #
    #     feat_a = feat_a.transpose(0, 1).flatten(0, 1)
    #     feat_map = (
    #         map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
    #     )
    #
    #     feat_a = self.pt2a_attn_layers[0](
    #         (feat_map, feat_a), r_pl2a, edge_index_pl2a
    #     )
    #
    #     if self.pred_light:
    #         feat_lg, next_light_logits = self.light_encoder.predict_light(light_idx, lg_sinusoidal,
    #                                                                       tokenized_agent["lengths_lg"], n_current)
    #
    #         pos_lg = tokenized_agent["pos_lg"]
    #         head_lg = tokenized_agent["orient_lg"]
    #         batch_lg = tokenized_agent["batch_lg"]
    #
    #         batch_lg = torch.cat(
    #             [
    #                 batch_lg + tokenized_agent["num_graphs"] * t
    #                 for t in range(n_step)
    #             ],
    #             dim=0,
    #         )
    #
    #         edge_index_lg2a, r_lg2a = self.edge_encoder.build_map2agent_edge(
    #             pos_pl=pos_lg,  # [n_pl, 2]
    #             orient_pl=head_lg,  # [n_pl]
    #             pos_a=pos_a,  # [n_agent, n_step, 2]
    #             head_a=head_a,  # [n_agent, n_step]
    #             head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
    #             mask=mask,  # [n_agent, n_step]
    #             batch_s=batch_s,  # [n_agent*n_step]
    #             batch_pl=batch_lg,  # [n_pl*n_step]
    #             pl2a_radius=100,
    #             max_num_neighbors=10
    #         )
    #         feat_a = self.light_encoder.lg2a_attn_layers[0](
    #             (feat_lg.swapaxes(0, 1).flatten(0, 1), feat_a), r_lg2a, edge_index_lg2a
    #         )
    #     else:
    #         next_light_logits = None
    #
    #         # feat_a=self.light_encoder.light2agent(tokenized_agent,feat_a,feat_lg, n_step,pos_a,head_a,head_vector_a,mask,batch_s)
    #
    #     feat_a = self.a2a_attn_layers[0](feat_a, r_a2a, edge_index_a2a)
    #     feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
    #
    #     return feat_a

    # def build_full_interaction_r_a2a(
    #         self,
    #         pos_s,  # [B, N, 2]
    #         head_s,  # [B, N,]
    #         head_vector_s,  # [B, N,2]
    #         pos_s1,
    #         head_s1,
    #         mask
    # ):
    #     B, N, _ = pos_s.shape
    #     B, N1, _ = pos_s1.shape
    #
    #     mask = ~mask
    #
    #     # Compute pairwise relative positions: [B, N, N, 2]
    #     rel_pos = pos_s[:, :, None, :] - pos_s1[:, None, :, :]  # [B, N, N, 2]
    #
    #     rel_pos = rel_pos[mask]
    #
    #     # Pairwise distance
    #     dist = torch.norm(rel_pos, dim=-1)  # [B, N, N]
    #
    #     # Relative heading difference
    #     rel_head = wrap_angle(head_s[:, :, None] - head_s1[:, None, :])[mask]  # [B, N, N]
    #
    #     head_vector_s = head_vector_s[:, :, None, :].expand(-1, -1, N1, -1)[mask]
    #
    #     # Angle between head_vector of neighbor and vector to target
    #     ang = angle_between_2d_vectors(
    #         ctr_vector=head_vector_s,  # [B, N, N, 2]
    #         nbr_vector=rel_pos  # [B, N, N, 2]
    #     )  # [B, N, N]
    #
    #     # Stack into r_a2a feature: [B, N, N, 3]
    #     r_a2a = torch.stack([dist, ang, rel_head], dim=-1)
    #
    #     # Apply embedding
    #     r_a2a = self.r_a2a_emb(r_a2a)  # [B, N, N, d_emb]
    #
    #     return r_a2a