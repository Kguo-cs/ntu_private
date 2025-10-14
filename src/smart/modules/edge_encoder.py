import random
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
    radiusGraphNearest,radiusGraphNearest_inv,visibility_aware_knn_with_radius_batch
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_cluster import radius_graph
import time

class EdgeEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int,
            share,
            a2a=True,
            hist_drop_prob=0.0,
            time_span=30,
            use_roformer=True,
            use_route=False,
            discriminator=False
    ) -> None:
        super(EdgeEncoder, self).__init__()

        self.use_route=use_route & (not discriminator)

        if self.use_route:
            input_dim_r_pt2a = 4
            self.route_drop=nn.Dropout(p=0.5)
        else:
            input_dim_r_pt2a = 3

        input_dim_r_a2a = 3

        share=share

        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            share=share
        )
        self.discriminator=discriminator

        # if self.discriminator:
        #     input_dim_r_a2a = 2

        if a2a:
            self.r_a2a_emb = FourierEmbedding(
                input_dim=input_dim_r_a2a,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
                share=share
            )

        self.use_roformer = use_roformer

        if not self.use_roformer:
            input_dim_r_t = 4

            self.r_t_emb = FourierEmbedding(
                input_dim=input_dim_r_t,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
                share=share
            )

            self.hist_drop_prob=hist_drop_prob
            self.time_span=time_span

            self.shift=5


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
            pos_s,  # [n_agent, n_step, 2]
            head_s,  # [n_agent, n_step]
            head_vector_s,  # [n_agent, n_step, 2]
            batch_s,  # [n_agent*n_step]
            mask,  # [n_agent, n_step]
            max_num_neighbors,
            max_radius,
            proposal=None,
            vis_mask=None,
            value=False,
            train_mask=None,
            loop=False
        ):
        if proposal is None:
            if vis_mask is not None:
                vis_mask=vis_mask.transpose(0, 1).reshape(-1)
                edge_index_a2a =visibility_aware_knn_with_radius_batch(pos_s, vis_mask,batch_s, max_num_neighbors, max_radius)
            else:
                # full_edge_index = radiusGraphNearest2(x=pos_s,
                #                                       y=pos_s,
                #                                       x_heading=head_s,
                #                                       r=max_radius,
                #                                       batch_x=batch_s,
                #                                       batch_y=batch_s,
                #                                       max_num_neighbors=max_num_neighbors)

                if value:
                    pos_a=pos_s.reshape(17,-1,2)
                    batch_s1=batch_s.reshape(17,-1) [:-1].flatten()

                    pos_a1=pos_a[:-1].flatten(0,1)
                    pos_a2=pos_a[1:].flatten(0,1)

                    edge_index_a2a = radiusGraphNearest2(x=pos_a1,
                                                          y=pos_a2,
                                                          x_heading=head_s,
                                                          r=max_radius,
                                                          batch_x=batch_s1,
                                                          batch_y=batch_s1,
                                                          max_num_neighbors=max_num_neighbors)

                    n_agent= pos_a.shape[1]

                    mask_ego=edge_index_a2a[0]!=edge_index_a2a[1]

                    edge_index_a2a = edge_index_a2a[:,mask_ego]

                    edge_index_a2a[0]=edge_index_a2a[0]+n_agent

                else:
                    edge_index_a2a = radiusGraphNearest(x=pos_s,
                                                     r=max_radius,
                                                     batch=batch_s,
                                                     loop=loop,
                                                     max_num_neighbors=max_num_neighbors)
        else:
            proposal=proposal.reshape(proposal.shape[0],proposal.shape[1],6,-1)[:,:,-6:].detach().transpose(0, 1).flatten(0,1)

            pos_local=proposal[...,:2]#.to(torch.float16)
            proposal_sigma = 3*torch.norm(proposal[..., 2:].exp(),dim=-1)#.to(torch.float16)

            full_edge_index = radiusGraphNearest(x=pos_s, r=max_radius,max_num_neighbors=10, batch=batch_s, loop=False)

            #.flatten(1, 2)

            global_pos,_ = transform_to_global(
                                        pos_local=pos_local,  # [n_agent, n_step, 2]
                                        head_local=None,  # [n_agent, n_step]
                                        pos_now=pos_s,  # [n_agent, 2]
                                        head_now=head_s  # [n_agent]
                        )

            #global_pos=global_pos.reshape(-1,proposal_pos.shape[-3],proposal_pos.shape[-2], 2)

            src, dst = full_edge_index

            # mask1=src<dst
            #
            # src=src[mask1]
            # dst=dst[mask1]

            src_traj=global_pos[src]#[:,:,None]
            dst_traj=global_pos[dst]#[:,None]

            dist=torch.norm(src_traj - dst_traj,dim=-1)#.reshape(-1,proposal_pos.shape[-3]*proposal_pos.shape[-3]*6).amin(-1)

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
            radius_single = torch.norm(shape[:, :2] / 2, dim=-1)  # (n_batch,)#.to(torch.float16)
            # pos_a: (n_batch, n_agent_per_batch, ?)
            radius = radius_single[batch_s][:,None]  # (n_agent,)

            src_radius=radius[src]
            dst_radius=radius[dst]

            src_sigma=proposal_sigma[src]
            dst_sigma=proposal_sigma[dst]

            radius_sum=src_radius+dst_radius+src_sigma+dst_sigma+5

            intersecting=(dist<radius_sum).any(dim=-1)

            full_edge_index=full_edge_index[:,intersecting]

        if self.discriminator and train_mask is not None:
            edge_index_a2a = edge_index_a2a[:, train_mask[edge_index_a2a[1]]]

        if mask is not None:
            edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]

        # if self.training:
        #     keep_mask = torch.rand(len(edge_index_a2a[0])) > 0.1
        #     edge_index_a2a = edge_index_a2a[:, keep_mask]

        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])

        # if self.discriminator:
        #
        #     r_a2a = torch.stack(
        #         [
        #             torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
        #             rel_head_a2a,
        #         ],
        #         dim=-1,
        #     )
        # else:
        dist=torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1)

        r_a2a = torch.stack(
            [
                dist,
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )


        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)


        return edge_index_a2a, r_a2a,dist

    def build_map2map_edge(self,
                           pos_pl,  # [n_pl, 2]
                           orient_pl,  # [n_pl]
                           pos_s,  # [n_agent, n_step, 2]
                           head_s,  # [n_agent, n_step]
                           head_vector_s,  # [n_agent, n_step, 2]
                           batch_s,  # [n_agent*n_step]
                           batch_pl,  # [n_pl*n_step]
                           pl2a_radius,
                           max_num_neighbors,
                           ):

        edge_index_pl2a = radiusGraphNearest2(x=pos_s,
                                              y=pos_pl,
                                              x_heading=head_s,
                                              r=pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=max_num_neighbors)


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
            mask_pl=None,
            train_mask=None,
            use_counterfactual=False,
            route_map_index=None,
            layer_num=1
    ):

        if train_mask is not None and layer_num==1 and not use_counterfactual:
            mask=mask[train_mask,:16]
            pos_a=pos_a[train_mask,:16]
            head_a=head_a[train_mask,:16]
            head_vector_a=head_vector_a[train_mask,:16]
            batch_s=batch_s[train_mask,:16]

        n_agent=pos_a.shape[0]
        n_step=pos_a.shape[1]

        pos_s=pos_a.flatten(0,1)
        head_s=head_a.flatten(0,1)
        batch_s=batch_s.flatten(0,1).contiguous()

        edge_index_pl2a = radiusGraphNearest2(x=pos_s,
                                              y=pos_pl,
                                              x_heading=head_s,
                                              r=pl2a_radius,
                                              batch_x=batch_s,
                                              batch_y=batch_pl,
                                              max_num_neighbors=max_num_neighbors)

        # edge_index_pl2a = radiusGraphNearest_inv(x=pos_s[:, :2],
        #                                       y=pos_pl[:, :2],
        #                                       r=pl2a_radius,
        #                                       batch_x=batch_s,
        #                                       batch_y=batch_pl,
        #                                       max_num_neighbors=8)

        edge_index_pl2a[1] = (edge_index_pl2a[1] % n_step) * n_agent + edge_index_pl2a[1] // n_step

        pos_s=pos_a.transpose(0,1).flatten(0,1)
        head_s=head_a.transpose(0,1).flatten(0,1)
        head_vector_s=head_vector_a.transpose(0,1).flatten(0,1)
        mask=mask.transpose(0,1).flatten(0,1)

        if mask is not None:
            edge_index_pl2a = edge_index_pl2a[:, mask[edge_index_pl2a[1]]]

        # if dropout:
        #     keep_mask=torch.rand(len(edge_index_pl2a[0]))>0.1
        #     edge_index_pl2a=edge_index_pl2a[:,keep_mask]

        if mask_pl is not None:
            mask_a2pl = mask_pl.transpose(0, 1).reshape(-1)
            edge_index_pl2a=edge_index_pl2a[:,mask_a2pl[edge_index_pl2a[0]]]

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )

        if self.use_route:
            point_isin = torch.zeros_like(rel_orient_pl2a)-1

            # if route_map_index is not None:
            #
            #     #route_number=torch.sum(route_map_index>0,dim=-1)
            #
            #     #max_num=torch.unique(route_map_index,dim=-1)
            #
            #     keep_mask=self.route_drop(torch.ones(n_agent,device=head_s.device)).to(bool)
            #
            #     # drop_mask = torch.rand(n_agent).to(head_s.device) < 0.5
            #     #
            #     # keep_mask= drop_mask #& (route_number>2)
            #
            #     agent_idx = edge_index_pl2a[1] % n_agent
            #
            #     keep_agent_mask = keep_mask[agent_idx]
            #
            #     route_idx = route_map_index[agent_idx[keep_agent_mask]]
            #
            #     map_idx = edge_index_pl2a[0][keep_agent_mask]
            #
            #     point_num = torch.bincount(batch_pl)
            #
            #     point_num = torch.cat([torch.zeros_like(point_num[:1]), point_num[:-1]])
            #
            #     cum_num = torch.cumsum(point_num, dim=0)
            #
            #     batch_cum_num = cum_num[batch_pl]
            #
            #     map_batch = map_idx - batch_cum_num[map_idx]
            #
            #     mask=(route_idx==map_batch[:,None]).any(dim=1)
            #
            #     point_isin[keep_agent_mask] =mask.to(torch.float32)

            r_pl2a = torch.stack(
                [
                    torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_s[edge_index_pl2a[1]],
                        nbr_vector=rel_pos_pl2a[:, :2],
                    ),
                    rel_orient_pl2a,
                    point_isin
                ],
                dim=-1,
            )

        else:
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