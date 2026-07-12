import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
    project_to_local_frame
)
from src.smart.utils.edge_utils import radiusGraphNearest2, radiusGraphNearest
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_scatter import scatter_mean

class EdgeEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int,
            hist_drop_prob=0.0,
            time_span=30,
            shift=0,
            discriminator=False,
            use_bird=False,
            use_pl2a=False,
            use_a2a=False,
            use_t2t=False,
            differentiable_edge=True
    ) -> None:
        super(EdgeEncoder, self).__init__()

        self.differentiable_edge=differentiable_edge

        self.rollout_traj=False

        self.hist_drop_prob = hist_drop_prob
        self.time_span = time_span
        self.shift = shift
        self.use_t2t=use_t2t

        if not use_bird:
            input_dim = 3
        else:
            input_dim = 4

        if use_pl2a:
            self.r_pt2a_emb = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

        if use_a2a:
            self.tokenized_pos=False

            self.r_a2a_emb = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

        if use_t2t:
            self.r_t_emb = FourierEmbedding(
                input_dim=input_dim+1,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

    def build_temporal_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2],
            mask,  # [n_agent, n_step]
            inference_mask=None,  # [n_agent, n_step]
            agent_train_mask=None
    ):
        if agent_train_mask is not None:
            pos_a=pos_a[agent_train_mask]
            head_a=head_a[agent_train_mask]
            head_vector_a=head_vector_a[agent_train_mask]
            mask=mask[agent_train_mask]

        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        flat_mask = mask.transpose(0, 1).flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        if self.shift <= 0:
            raise ValueError("shift must be positive when temporal edges are enabled.")

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])

        feat_a=project_to_local_frame(rel_pos_t,head_vector_t[edge_index_t[1]],self.differentiable_edge)

        r_t = torch.cat(
            [
                feat_a,
                rel_head_t[:,None],
                (edge_index_t[0] - edge_index_t[1])[:,None],
            ],
            dim=-1,
        )

        n_agent, n_step = mask.shape

        edge_index_t = (edge_index_t % n_step) * n_agent + edge_index_t // n_step

        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        if torch.any(flat_mask==False):

            N_total = n_step * n_agent  # total nodes in transposed ordering

            kept_nodes = torch.nonzero(flat_mask, as_tuple=True)[0]  # shape [M]
            map_to_compact = torch.full((N_total,), -1, dtype=torch.long, device=kept_nodes.device)
            map_to_compact[kept_nodes] = torch.arange(kept_nodes.size(0), device=kept_nodes.device, dtype=torch.long)

            edge_index_t = map_to_compact[edge_index_t]

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
            agent_train_mask=None,
            layer_num=1,
            counter_feat_a=None,
            dis_edge_mask=None,
            a2a_edge_index=None
        ):
        if mask is not None:
            pos_s = pos_s[mask]
            head_s = head_s[mask]
            head_vector_s = head_vector_s[mask]
            batch_s = batch_s[mask]

        if a2a_edge_index is None:
            edge_index_a2a = radiusGraphNearest(x=pos_s.detach(),
                                                r=max_radius,
                                                batch=batch_s,
                                                loop=False,
                                                max_num_neighbors=max_num_neighbors)
        else:
            edge_index_a2a = a2a_edge_index

        if agent_train_mask is not None and layer_num==1:
            edge_index_a2a = edge_index_a2a[:, agent_train_mask[edge_index_a2a[1]]]

        # if mask is not None:
        #     edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]

        # if self.training:
        #     keep_mask = torch.rand(len(edge_index_a2a[0])) > 0.1
        #     edge_index_a2a = edge_index_a2a[:, keep_mask]
        if dis_edge_mask is not None:
            dis_edge_mask=dis_edge_mask[edge_index_a2a[1]]
            edge_index_a2a=edge_index_a2a[:,dis_edge_mask]

        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])

        dist=torch.norm(rel_pos_a2a, p=2, dim=-1)

        feat_a=project_to_local_frame(rel_pos_a2a,head_vector_s[edge_index_a2a[1]],self.differentiable_edge)

        r_a2a = torch.cat(
            [
                feat_a,
                rel_head_a2a[:,None],
            ],
            dim=-1,
        )

        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        if counter_feat_a is not None:
            start_index = edge_index_a2a[0]
            end_index = edge_index_a2a[1]

            start_pos = pos_s[start_index]

            start_heading = head_s[start_index]

            center_nei_pos = scatter_mean(start_pos, end_index, dim=0, dim_size=len(pos_s))

            center_nei_heading= scatter_mean(start_heading, end_index, dim=0, dim_size=len(head_s))

            rel_pos_a2a = start_pos - center_nei_pos[end_index]
            rel_head_a2a = wrap_angle(start_heading - center_nei_heading[end_index])

            r_a2a_nei = torch.stack(
                [
                    torch.norm(rel_pos_a2a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_s[edge_index_a2a[1]],
                        nbr_vector=rel_pos_a2a[:, :2],
                    ),
                    rel_head_a2a
                ],
                dim=-1,
            )

            r_a2a_nei = torch.cat([r_a2a_nei, rel_pos_a2a[:, 2:]], dim=-1)

            r_a2a_nei = self.r_a2a_emb(continuous_inputs=r_a2a_nei, categorical_embs=None)
        else:
            r_a2a_nei=center_nei_pos=center_nei_heading=None

        return edge_index_a2a, r_a2a,dist,None,r_a2a_nei,center_nei_pos,center_nei_heading

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
                           l2l_edge_index=None,
                           l2l_feature=None
                           ):

        if l2l_edge_index is None:
            edge_index_pl2pl = radiusGraphNearest2(x=pos_s,
                                                  y=pos_pl,
                                                  r=pl2a_radius,
                                                  batch_x=batch_s,
                                                  batch_y=batch_pl,
                                                  max_num_neighbors=max_num_neighbors)
        else:
            edge_index_pl2pl=l2l_edge_index

        # #edge_index[0] → indices in y (query points)            edge_index[1] → indices in x (neighbor points)
        rel_pos_pl2a = pos_pl[edge_index_pl2pl[0]] - pos_s[edge_index_pl2pl[1]]   #src, dst
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2pl[0]] - head_s[edge_index_pl2pl[1]]
        )

        feat_a=project_to_local_frame(rel_pos_pl2a,head_vector_s[edge_index_pl2pl[1]],self.differentiable_edge)


        r_pl2a = torch.cat(
            [
                feat_a,
                rel_orient_pl2a[:,None],
            ],
            dim=-1,
        )

        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=l2l_feature)

        return edge_index_pl2pl, r_pl2a


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
            agent_train_mask=None,
            use_counterfactual=False,
            route_map_index=None,
            layer_num=1,
            l2a_edge_index=None
    ):

        if agent_train_mask is not None and layer_num==1:
            mask = mask & agent_train_mask[:,None]

        if mask is not None:
            n_agent, n_step = mask.shape

            pos_s=pos_a[mask]
            head_s=head_a[mask]
            head_vector_s=head_vector_a[mask]
            batch_s=batch_s[mask]
        else:
            pos_s=pos_a
            head_s=head_a
            head_vector_s=head_vector_a
            batch_s=batch_s
            n_step=1


        if l2a_edge_index is None:
            edge_index_pl2a = radiusGraphNearest2(x=pos_s,
                                                  y=pos_pl,
                                                  r=pl2a_radius,
                                                  batch_x=batch_s,
                                                  batch_y=batch_pl,
                                                  max_num_neighbors=max_num_neighbors)

        else:
            edge_index_pl2a=l2a_edge_index

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )

        feat_a=project_to_local_frame(rel_pos_pl2a,head_vector_s[edge_index_pl2a[1]],self.differentiable_edge)

        r_pl2a = torch.cat(
            [
                feat_a,
                rel_orient_pl2a[:,None],
            ],
            dim=-1,
        )

        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        if n_step>1:
            N_total = n_agent * n_step

            # 1) Kept global indices in both orderings
            flat_mask_agent = mask.flatten(0, 1)  # agent-major
            flat_mask_time = mask.transpose(0, 1).flatten(0, 1)  # time-major

            kept_agent = torch.nonzero(flat_mask_agent, as_tuple=False).squeeze(1)  # [M], global idx
            kept_time = torch.nonzero(flat_mask_time, as_tuple=False).squeeze(1)  # [M], global idx

            map_global_to_compact_time = torch.full((N_total,), -1, dtype=torch.long, device=mask.device)
            map_global_to_compact_time[kept_time] = torch.arange(kept_time.numel(), device=mask.device)

            # 3) Convert compact agent-major indices -> global -> compact time-major indices
            dst_compact_agent = edge_index_pl2a[1]  # indices into pos_a[mask]
            dst_global = kept_agent[dst_compact_agent]  # global flattened indices

            dst_global=(dst_global % n_step) * n_agent + dst_global // n_step

            new_dst = map_global_to_compact_time[dst_global]
            edge_index_pl2a = torch.stack([edge_index_pl2a[0], new_dst], dim=0)

        return edge_index_pl2a, r_pl2a

