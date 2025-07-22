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
from .build_edge import radiusGraphNearest2,nearest_mask,generate_limited_causal_mask,nearest_mask2,radiusGraphNearest_inv,build_batch
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from src.smart.utils.rollout import cal_polygon_contour
from src.smart.loss.gmm_dist import  GMM_Dist
from src.smart.loss.iq_loss import padding
from src.smart.modules.light_encoder import LightEncoder
from src.smart.modules.edge_encoder import EdgeEncoder
from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder

class SMARTAgentDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_historical_steps: int,
            num_future_steps: int,
            time_span: Optional[int],
            pl2a_radius: float,
            a2a_radius: float,
            num_freq_bands: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
            hist_drop_prob: float,
            n_token_agent: int,
            pt2a_neighbor: int,
            a2a_neighbor: int,
            token_processor,
            alpha,
            output_gmm,
            pred_light,
            pred_last_res,
            pred_all_res
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.num_layers = num_layers
        self.shift = token_processor.shift
        self.hist_drop_prob = hist_drop_prob

        self.alpha = alpha

        self.head_dim = hidden_dim // num_heads

        self.agent_token_embedding=AgentTokenEncoder(hidden_dim,num_freq_bands,token_processor)

        self.agent_hist = self.time_span // self.shift

        self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=hist_drop_prob,hist_len=self.agent_hist)

        self.n_token_agent = n_token_agent
        self.output_gmm = output_gmm

        self.pred_last_res = pred_last_res
        self.pred_all_res = pred_all_res

        self.interative_decoder = InterativeDecoder(hidden_dim,num_historical_steps,num_future_steps,time_span,
                                                    pl2a_radius,a2a_radius,num_freq_bands,
                                                    num_layers,num_heads,head_dim,
                                                    dropout,hist_drop_prob,n_token_agent,
                                                    pt2a_neighbor,a2a_neighbor,
                                                    token_processor,output_gmm,pred_last_res,pred_all_res
                                                    )

        # self.pt2a_attn_layers = nn.ModuleList(
        #     [
        #         AttentionLayer(
        #             hidden_dim=hidden_dim,
        #             num_heads=num_heads,
        #             head_dim=head_dim,
        #             dropout=dropout,
        #             bipartite=True,
        #             has_pos_emb=True,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        #
        # self.a2a_attn_layers = nn.ModuleList(
        #     [
        #         AttentionLayer(
        #             hidden_dim=hidden_dim,
        #             num_heads=num_heads,
        #             head_dim=head_dim,
        #             dropout=dropout,
        #             bipartite=False,
        #             has_pos_emb=True,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        #
        #
        # if self.output_gmm:
        #     self.k_ego_gmm=1
        #     self.cov_gmm=0.1 #[1.0, 0.1]
        #     self.cov_learnable=True
        #     self.use_GT=True
        #
        #     if self.k_ego_gmm>1:
        #         self.gmm_logits_head = MLPLayer(
        #             input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.k_ego_gmm
        #         )
        #     self.gmm_pose_head = MLPLayer(
        #         input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.k_ego_gmm * 3
        #     )
        #     self.output_dim=3
        #
        #     if self.cov_learnable:
        #         self.gmm_cov_head = MLPLayer(
        #             input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.k_ego_gmm * self.output_dim
        #         )
        #
        #     # self.cholesky_head = nn.Linear(
        #     #     hidden_dim, k_ego_gmm * (self.output_dim * (self.output_dim + 1) // 2)
        #     # )
        # else:
        #     self.token_predict_head = MLPLayer(
        #         input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        #     )
        #
        #     if self.pred_last_res or self.pred_all_res:
        #         self.traj_head = MLPLayer(hidden_dim, hidden_dim, output_dim=3 * 5)

        self.use_light = token_processor.use_light
        self.pred_light=pred_light
        self.light_type = token_processor.light_type
        self.light_hist = self.agent_hist

        if self.use_light:
            self.light_encoder = LightEncoder(self.edge_encoder,hidden_dim,self.light_hist,num_heads,self.light_type,self.shift,self.pred_light,alpha)

            self.lg2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=True,
                        has_pos_emb=True,
                    )
                    for _ in range(1)
                ]
            )

        else:
            self.pred_light=False

        self.use_dynamic=token_processor.use_dynamic
        self.start_step=10//self.shift-1
        self.pred_vis = False

        if self.pred_vis:
            self.vis_head=MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1 )

        self.token_processor= token_processor
        self.apply(weight_init)

    def predict_agent(self, sampled_idx, mask ,pos_a,head_a,tokenized_agent, map_feature,light_idx,mask_lg, n_current=0,vis_mask=None,post_sampling=False):
        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        # ! get agent token embeddings
        feat_a_token,agent_token_emb = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            trajectory_token_veh=self.token_processor.agent_token_all_veh,
            trajectory_token_ped=self.token_processor.agent_token_all_ped,
            trajectory_token_cyc=self.token_processor.agent_token_all_cyc,
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        # if self.pred_proposal:
        #     proposal_feature = feat_a_token[:,: ,None] + self.proposal_embedding.weight[None, None]  # [B,T, N, D]
        #     proposal = self.proposal_head(proposal_feature).reshape(proposal_feature.shape[0],proposal_feature.shape[1],proposal_feature.shape[2],-1,3)
        # elif self.pred_gaussian:
        #     proposal = self.gaussian_head(feat_a_token)
        # else:
        #     proposal=None

        # if post_sampling:
        #     return None,None,None,proposal

        pos_a=pos_a[:,-n_step:]

        feat_a_t = self.a_t_roformer.temporal_embed(feat_a_token, pos_a, head_a, n_step, n_current, mask)

        if self.training:
            n_step=n_step-self.start_step
            pos_a=pos_a[:,-n_step:]
            head_a=head_a[:,-n_step:]
            head_vector_a=head_vector_a[:,-n_step:]
            agent_token_emb=agent_token_emb[:,-n_step:]
            feat_a_t=feat_a_t[:,-n_step:]

        mask_a=mask[:,-n_step:]

        batch_s = build_batch(tokenized_agent["batch"], tokenized_agent["num_graphs"], n_step)

        batch_pl = build_batch(map_feature["batch"], tokenized_agent["num_graphs"], n_step)

        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]

        if len(light_idx):
            feat_lg = self.light_encoder.light_embedding(light_idx)

            batch_lg = build_batch(tokenized_agent["batch_lg"],tokenized_agent["num_graphs"],n_step )

            if self.pred_light:
                _, next_light_logits = self.light_encoder(tokenized_agent,light_idx, mask_lg, batch_lg,  n_step, n_current,feat_lg)
            else:
                next_light_logits = []

            mask_lg = mask_lg[:, -n_step:]

            edge_index_lg2a, r_lg2a = self.edge_encoder.build_map2agent_edge(
                pos_pl= tokenized_agent["pos_lg"],  # [n_pl, 2]
                orient_pl=tokenized_agent["orient_lg"],  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask_a,  # [n_agent, n_step]
                batch_s=batch_s,  # [n_agent*n_step]
                batch_pl=batch_lg,  # [n_pl*n_step]
                pl2a_radius=100,
                max_num_neighbors=8,
                mask_pl=mask_lg[:,-n_step:]
            )

            feat_lg=feat_lg.swapaxes(0, 1).flatten(0, 1)
        else:
            next_light_logits =feat_lg=r_lg2a=edge_index_lg2a= []

        feat_map = map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)

        feat_a = feat_a_t.transpose(0, 1).flatten(0, 1)

        if len(feat_lg):
            feat_a = self.lg2a_attn_layers[0]((feat_lg, feat_a), r_lg2a, edge_index_lg2a)

        if "train_mask" in tokenized_agent.keys() and self.training:
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        if vis_mask is not None:
            vis_mask = vis_mask[:, -n_step:]

        all_features= feat_a, agent_token_emb, sampled_idx, feat_map, pos_pl, orient_pl, \
            pos_a, head_a, head_vector_a, mask_a, batch_s, batch_pl,vis_mask

        if self.training:
            detach_all_features=[]
            for feature in all_features:
                if feature is not None:
                    detach_all_features.append(feature.detach())#.clone()
                else:
                    detach_all_features.append(feature)#.clone()
            tokenized_agent["detach_all_features"]=detach_all_features

        next_token_logits,feat_a,proposal=self.interative_decoder(all_features,train_mask)

        visibility=None

        if self.pred_vis:
            visibility=self.vis_head(feat_a.detach())

        return next_token_logits,next_light_logits,feat_a,proposal,visibility

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            post_sampling=False
    ) -> Dict[str, torch.Tensor]:

        light_idx = tokenized_agent["light_idx"].clone()

        # random_light = torch.randint(low=0, high=self.light_type, size=light_idx.shape, device=light_idx.device).long()

        # random_mask = torch.rand_like(light_idx.float()) > 0.9

        # random_mask[:, :2] = False

        # light_idx[random_mask] = random_light[random_mask]

        mask_lg=light_idx<self.light_type

        next_token_logits,next_light_logits,feat_a,proposal,visibility= self.predict_agent(tokenized_agent["sampled_idx"],
                                                                                tokenized_agent["valid_mask"],
                                                                                tokenized_agent["sampled_pos"],
                                                                                tokenized_agent["sampled_heading"] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                light_idx,
                                                                                mask_lg,
                                                                                vis_mask=tokenized_agent["vis_mask"],
                                                                                post_sampling=post_sampling)

        tokenized_agent["next_token_logits"] = next_token_logits
        tokenized_agent["next_light_logits"] = next_light_logits
        tokenized_agent["visibility"] = visibility
        tokenized_agent["proposal"] = proposal

        return {
            "proposal":proposal,
            "visibility":visibility,
            "light_q": next_light_logits,
            "agent_q": next_token_logits,            # action that goes from [(10->15), ..., (85->90)]
         }

    def autoregressive_agent(self, tokenized_agent, map_feature,current_step,max_step,post_sampling):

        sampled_idx=tokenized_agent["sampled_idx"][:, :current_step].clone()
        mask = tokenized_agent["valid_mask"][:, :current_step].clone()
        pos_a = tokenized_agent["sampled_pos"][:, :current_step].clone()
        head_a = tokenized_agent["sampled_heading"][:, :current_step].clone()
        token_agent_shape=tokenized_agent["token_agent_shape"]
        token_traj=tokenized_agent["token_traj"]
        n_agent = sampled_idx.shape[0]
        light_idx = tokenized_agent["light_idx"][:, :current_step].clone()
        token_traj_all = tokenized_agent["token_traj_all"]

        mask_lg=light_idx<self.light_type

        if post_sampling:
            gt_valid=tokenized_agent["valid_mask"]
            gt_pos=tokenized_agent["sampled_pos"]
            gt_head=tokenized_agent["sampled_heading"]
            gt_sampled_idx = tokenized_agent["sampled_idx"]

        pred_traj_10hz = []
        pred_head_10hz = []
        sampled_log_prob=[]

        vis_mask=mask.clone()

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None:
                    next_token_logits = tokenized_agent["next_token_logits"][:, :1]

                    if tokenized_agent["proposal"] is not None:
                        proposal=tokenized_agent["proposal"][:, :1]

                    if tokenized_agent["visibility"] is not None:
                        visibility=tokenized_agent["visibility"][:, :1]

                    if self.pred_light:
                        next_light_logits = tokenized_agent["next_light_logits"][:, :1]
                    else:
                        next_light_logits = []
                else:
                    self.a_t_roformer.attn.caching=True
                    if self.pred_light:
                        self.light_encoder.lg_t_roformer.attn.caching=True
                    next_token_logits,next_light_logits,feat_a,proposal,visibility = self.predict_agent(sampled_idx, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,light_idx,mask_lg,0,vis_mask,post_sampling)

                self.a_t_roformer.attn.kv_caching(self.agent_hist,current_step)
                if self.pred_light:
                    self.light_encoder.lg_t_roformer.attn.kv_caching(self.light_hist)
            else:
                next_token_logits,next_light_logits,feat_a,proposal,visibility  = self.predict_agent(sampled_idx[:, -1:], mask[:, -self.agent_hist:],
                                                            pos_a[:, -2:], head_a[:, -1:],tokenized_agent, map_feature,light_idx[:, -1:],
                                                                                    mask_lg[:,-self.light_hist:],t - 1,vis_mask[:, -1:],post_sampling)

            if post_sampling:
                next_token_idx=gt_sampled_idx[:,t]
            else:
                dist=Categorical(
                    logits=next_token_logits[:, -1, ] / self.alpha)
                next_token_idx = dist.sample()

                log_prob=dist.log_prob(next_token_idx)

                sampled_log_prob.append(log_prob)

            if self.pred_all_res:
                token_embedding=self.agent_token_embedding.embedding(next_token_idx)

                proposal_feature=feat_a[:,-1]+token_embedding

                proposal=self.interative_decoder.traj_head(proposal_feature).reshape(n_agent,-1,3)

                if self.token_processor.max_diff is not None:

                    proposal_max_diff = self.token_processor.token_diff[torch.arange(n_agent), next_token_idx]

                    proposal = torch.tanh(proposal) * proposal_max_diff

                next_token_traj_all = self.token_processor.token_local_traj[torch.arange(n_agent), next_token_idx]

                proposal=proposal+next_token_traj_all

                next_token_traj_all=cal_polygon_contour(proposal[:,:,:2],proposal[:,:,2],token_agent_shape[:,None])

                next_token_idx = self.token_processor.traj_to_idx(proposal[:, -1:, None], token_agent_shape,
                                                                token_traj)[:, 0]

            elif self.pred_last_res:
                proposal=proposal[:,-1,0]

                proposal_token=cal_polygon_contour(proposal[:,:,:2],proposal[:,:,2],token_agent_shape[:,None])

                token_traj_current=torch.cat([token_traj_all, proposal_token[:, None]], dim=1)
            else:
                token_traj_current=token_traj_all

            if not self.pred_all_res:
                next_token_traj_all = token_traj_current[torch.arange(n_agent), next_token_idx]

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, -1],  # [n_agent, 2]
                head_now=head_a[:, -1],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            if "gt_z_raw" in tokenized_agent.keys():
                pred_traj = token_traj_global[:, :].mean(2)
                pred_traj_10hz.append(pred_traj)
                diff_xy = token_traj_global[:, :, 0] - token_traj_global[:, :, 3]
                pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
                pred_head_10hz.append(pred_head)

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

            if len(next_light_logits):

                cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

                next_light_idx= cat_dist.sample()

                light_idx = torch.cat([light_idx, next_light_idx[:, None]], dim=1)
            elif self.use_light:
                light_idx = tokenized_agent["light_idx"][:,t:t+1]

            if post_sampling:
                prev_valid=gt_valid[:,t-1]
                pos_a[:,-1][~prev_valid] = gt_pos[:, t][~prev_valid]
                head_a[:,-1][~prev_valid] = gt_head[:, t][~prev_valid]

                valid_mask=prev_valid & gt_valid[:,t]
                _invalid_mask = ~valid_mask

                pos_a[:,-1]=pos_a[:,-1].masked_fill(_invalid_mask.unsqueeze(1), 0)
                head_a[:,-1]=head_a[:,-1].masked_fill(_invalid_mask, 0)

                mask =torch.cat([mask,valid_mask[:,None]], dim=1)
            else:
                if "gt_z_raw" in tokenized_agent.keys():
                    mask =torch.cat([mask,torch.ones_like(mask[:,-1:]).to(torch.bool)], dim=1)
                else:
                    mask=torch.cat([mask,tokenized_agent["valid_mask"][:,t:t+1]], dim=1)
                mask_lg =torch.cat([mask_lg,torch.ones_like(mask_lg[:,-1:]).to(torch.bool)], dim=1)

            if self.pred_vis:
                vis=torch.rand_like(visibility[:,-1:,0])<torch.sigmoid(visibility[:,-1:,0])
                vis=vis_mask[:,-1:] & vis
                vis_mask=torch.cat([vis_mask,vis],dim=1)

        self.a_t_roformer.attn.kv_caching(0)
        if self.pred_light:
            self.light_encoder.lg_t_roformer.attn.kv_caching(0)

        sampled_log_prob=torch.stack(sampled_log_prob,dim=1)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "sampled_idx": sampled_idx,  # [n_agent, 18]
            "sampled_log_prob":sampled_log_prob,
            "vis_mask": vis_mask,
            #"light_idx": light_idx,
        }

        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = torch.cat(pred_traj_10hz, dim=1)
            out_dict["pred_head_10hz"] =torch.cat(pred_head_10hz, dim=1)
            out_dict["pred_z_10hz"] = tokenized_agent["gt_z_raw"].unsqueeze(1) .expand(-1, out_dict["pred_traj_10hz"].shape[1])

        return out_dict

    def inference(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            post_sampling=False,
            step_current_10hz=None,
            n_step_future_10hz=None,
    ) -> Dict[str, torch.Tensor]:
        if n_step_future_10hz is None:
            n_step_future_10hz = self.num_future_steps  # 80
        if step_current_10hz is None:
            step_current_10hz = self.num_historical_steps - 1  # 10

        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_2hz = step_current_10hz // self.shift  # 2

        out_dict=self.autoregressive_agent(tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz,post_sampling)

        return out_dict
