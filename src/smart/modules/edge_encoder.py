from typing import Dict, Optional

import torch
import torch.nn as nn
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from .build_edge import radiusGraphNearest2,nearest_mask,generate_limited_causal_mask,nearest_mask2, \
    radiusGraphNearest,radiusGraphNearest_inv
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_cluster import radius_graph


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

        input_dim_r_t = 4

        # self.r_t_emb = FourierEmbedding(
        #     input_dim=input_dim_r_t,
        #     hidden_dim=hidden_dim,
        #     num_freq_bands=num_freq_bands,
        # )


    def build_temporal_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2],
            mask,  # [n_agent, n_step]
            inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
                       :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
                       ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            batch_s,  # [n_agent*n_step]
            mask,  # [n_agent, n_step]
            max_num_neighbors,
            max_radius,
            proposal=None,
            shape=None
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        if proposal is None:
            full_edge_index = radiusGraphNearest(x=pos_s,
                                                 r=max_radius,
                                                 batch=batch_s,
                                                 loop=False,
                                                 max_num_neighbors=max_num_neighbors)


        else:
            proposal_pos=proposal[...,::5,:2]
            pos_local=proposal_pos.transpose(0, 1).flatten(0,1).flatten(1, 2)

            full_edge_index = radius_graph(x=pos_s, r=60,max_num_neighbors=300, batch=batch_s, loop=False)

            global_pos,_ = transform_to_global(
                                        pos_local=pos_local,  # [n_agent, n_step, 2]
                                        head_local=None,  # [n_agent, n_step]
                                        pos_now=pos_s,  # [n_agent, 2]
                                        head_now=head_s  # [n_agent]
                        )

            global_pos=global_pos.reshape(-1,proposal_pos.shape[-3],proposal_pos.shape[-2], 2)

            src, dst = full_edge_index

            mask1=src<dst

            src=src[mask1]
            dst=dst[mask1]


            src_traj=global_pos[src][:,:,None]
            dst_traj=global_pos[dst][:,None]

            dist=torch.norm(src_traj - dst_traj,dim=-1).reshape(-1,32*32*6).amin(-1)

            # # shape: (E, 32, 6, 2), unsqueezed for broadcasting
            # src_traj = global_pos[src].transpose(1,2).flatten(0,1)  # (E, 32, 6, 2)
            # dst_traj = global_pos[dst].transpose(1,2).flatten(0,1)  # (E, 32, 6, 2)
            #
            # # Compute minimum distance across all points (broadcasted)
            # dist = torch.cdist(
            #     src_traj.flatten(1, 2),  # (E, 192, 2)
            #     dst_traj.flatten(1, 2),  # (E, 192, 2),
            #     p=2
            # ).amin(dim=1)  # (E,)

            # shape: (n_batch, 2)
            radius_single = torch.norm(shape[:, :2] / 2, dim=-1)  # (n_batch,)
            # pos_a: (n_batch, n_agent_per_batch, ?)
            radius = radius_single[batch_s]  # (n_agent,)

            src_radius=radius[src]
            dst_radius=radius[dst]

            radius_sum=src_radius+dst_radius #+5

            intersecting=dist<radius_sum

            src=src[intersecting]
            dst=dst[intersecting]

            src_full = torch.cat([src, dst], dim=0)
            dst_full = torch.cat([dst, src], dim=0)

            full_edge_index=torch.stack([src_full, dst_full], dim=0)

        edge_index_a2a = subgraph(subset=mask, edge_index=full_edge_index)[0]
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
            max_num_neighbors,
            mask_pl=None
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

        if mask_pl is not None:
            mask_a2pl = mask_pl.transpose(0, 1).reshape(-1)
            edge_index_pl2a=edge_index_pl2a[:,mask_a2pl[edge_index_pl2a[0]]]

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