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
from tensorflow.python.layers.core import dropout
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
            pred_last_res,
            pred_all_res,
            discriminator=False
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

        self.pred_goal=token_processor.pred_goal

        if self.pred_goal and not discriminator:
            self.goal_embedding=nn.Embedding(11, hidden_dim)
            self.goal_head = MLPLayer(hidden_dim, hidden_dim, 11)

        #if not discriminator:
        self.agent_token_embedding=AgentTokenEncoder(hidden_dim,num_freq_bands,token_processor,discriminator)

        self.agent_hist = self.time_span // self.shift

        #if not discriminator:
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
                                                    token_processor,output_gmm,pred_last_res,pred_all_res,discriminator
                                                    )

        self.use_light = token_processor.use_light
        self.pred_light = True
        self.light_type = token_processor.light_type
        self.light_hist = self.agent_hist

        if self.use_light:
            self.light_encoder = LightEncoder(self.interative_decoder.edge_encoder,hidden_dim,self.light_hist,num_heads,self.light_type,self.shift,self.pred_light,alpha)

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

        self.use_kl_penalty=False

        self.target_net=False

        if self.pred_vis:
            self.vis_head=MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1 )

        if not discriminator:
            self.use_lcf=False

            # if self.use_lcf:
            #     self.lcf_head=MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=10 )
            #
            #     self.svo_embedding=nn.Embedding(10, hidden_dim)

        self.use_latent=False

        if self.use_latent and not discriminator:
            self.k_dim=16
            self.latent_embed=MLPLayer(self.k_dim,hidden_dim,hidden_dim)#nn.Embedding(self.k_dim, hidden_dim)

        self.token_processor= token_processor
        self.discriminator=discriminator
        self.apply(weight_init)

    def predict_agent(self, sampled_idx,goal_idx, mask ,pos_a,head_a,tokenized_agent, map_feature,light_idx,mask_lg, n_current=0,latent_z=None,post_sampling=False):
        n_agent, n_step = head_a.shape

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        # if self.discriminator:
        #     feat_a_token=tokenized_agent["feat_a_token"]
        #     agent_token_emb=tokenized_agent["agent_token_emb"]
        # else:
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

            # if self.use_kl_penalty and "feat_a_token" not in tokenized_agent.keys():
            #     return None,None,(feat_a_token.detach(),agent_token_emb.detach()),None,None

        if latent_z is not None:
            latent_embedding=self.latent_embed(latent_z)
            feat_a_token=feat_a_token+latent_embedding

        if self.pred_goal and not self.discriminator:
            goal_token_emb=self.goal_embedding(goal_idx)
            feat_a_token=feat_a_token+goal_token_emb

        pos_a=pos_a[:,-n_step:]

        if len(light_idx):
            feat_lg = self.light_encoder.light_embedding(light_idx)

        if len(light_idx) and self.light_encoder.share:
            feat_a_lg_token=torch.cat((feat_a_token,feat_lg),dim=0)
            mask_a_lg=torch.cat((mask,mask_lg),dim=0)
            feat_a_lg_t = self.a_t_roformer.temporal_embed(feat_a_lg_token, None, None, n_step, n_current, mask_a_lg)
            feat_a_t=feat_a_lg_t[:len(mask)]
            feat_lg_t=feat_a_lg_t[len(mask):]
        else:
            # if self.discriminator:
            #     feat_a_t=feat_a_token
            # else:
            feat_a_t = self.a_t_roformer.temporal_embed(feat_a_token, pos_a, head_a, n_step, n_current, mask)
            feat_lg_t=None

        if self.training or self.discriminator or self.target_net:
            n_step=n_step-self.start_step
            pos_a=pos_a[:,-n_step:]
            head_a=head_a[:,-n_step:]
            head_vector_a=head_vector_a[:,-n_step:]
            #agent_token_emb=agent_token_emb[:,-n_step:]
            feat_a_t=feat_a_t[:,-n_step:]
            if len(light_idx) and self.light_encoder.share:
                feat_lg_t = feat_lg_t[:, -n_step:]
                feat_lg=feat_lg[:, -n_step:]
                light_idx=light_idx[:, -n_step:]

        mask_a=mask[:,-n_step:]
        batch_a=tokenized_agent["batch"]

        batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], max(1, n_step - 1)).reshape(-1,n_agent).transpose(
            0, 1)
        batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

        #batch_pl=build_batch(map_feature["batch"], tokenized_agent["num_graphs"], n_step).reshape(n_step,-1).transpose(0,1)
        #batch_pl=map_feature["batch"]

        if len(light_idx):
            batch_lg = build_batch(tokenized_agent["batch_lg"],tokenized_agent["num_graphs"],n_step )

            if self.pred_light:
                _, next_light_logits = self.light_encoder(tokenized_agent,light_idx, mask_lg, batch_lg,  n_step, n_current,feat_lg=feat_lg,feat_lg_t=feat_lg_t)
            else:
                next_light_logits = []

            mask_lg = mask_lg[:, -n_step:]

            edge_index_lg2a, r_lg2a = self.interative_decoder.edge_encoder.build_map2agent_edge(
                pos_pl= tokenized_agent["pos_lg"],  # [n_pl, 2]
                orient_pl=tokenized_agent["orient_lg"],  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask_a,  # [n_agent, n_step]
                batch_s=batch_s,  # [n_agent*n_step]
                batch_pl=batch_lg,  # [n_pl*n_step]
                pl2a_radius=100,
                max_num_neighbors=10,
                mask_pl=mask_lg[:,-n_step:]
            )

            feat_lg=feat_lg.swapaxes(0, 1).flatten(0, 1)
        else:
            next_light_logits =feat_lg=r_lg2a=edge_index_lg2a= []

        #feat_a = feat_a_t.flatten(0, 1)#.transpose(0, 1)

        if len(feat_lg):
            feat_a = self.lg2a_attn_layers[0]((feat_lg, feat_a), r_lg2a, edge_index_lg2a)

        if ("train_mask" in tokenized_agent.keys() and self.training) or self.target_net:
            train_mask=tokenized_agent["train_mask"]
        else:
            train_mask=None

        # if vis_mask is not None:
        #     vis_mask = vis_mask[:, -n_step:]

        if n_step>1:
            batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step - 1).reshape(-1,n_agent).transpose(
                0, 1)

            all_features=[]
            next_all_features=[]
            for feature in [feat_a_t,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat]:
                all_features.append(feature[:, :-1])
                next_all_features.append(feature[:, 1:])  # .clone()[:,1:]

            next_all_features.append(batch_s)
            all_features.append(batch_s)

            # tokenized_agent["detach_all_features"]=[feature.detach() for feature in next_all_features]

            if self.discriminator:
                if self.interative_decoder.reward_shaping:
                    batch_s = build_batch(batch_a, tokenized_agent["num_graphs"],n_step).reshape(-1, n_agent).transpose(
                        0, 1)

                    all_features=[feat_a_t,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s,agent_token_emb[:,2:]]
                else:
                    all_features=next_all_features
                    all_features.append(None)
            else:
                if not (self.training or self.target_net):
                    all_features=next_all_features

                all_features.append(None)
        else:
            batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(
                0, 1)

            all_features=[feat_a_t,pos_a, head_a, head_vector_a,mask_a,batch_s_repeat,batch_s,None]

        next_token_logits,feat_a,proposal,r_a2a,r_pl2a=self.interative_decoder(all_features,map_feature,train_mask)

        # proposal=torch.zeros([n_agent,n_step,15],device=feat_a.device)
        # proposal[mask_a]=proposal_
        # proposal=proposal.reshape([n_agent,n_step,1,5,3])
        # if self.pred_goal and not self.discriminator:
        #     next_goal_logits=self.goal_head(feat_a.detach())
        # else:
        #     next_goal_logits=[]
        # visibility=None
        #
        # if self.pred_vis:
        #     visibility=self.vis_head(feat_a.detach())
        # if n_step>1:
        #     batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1, n_agent).transpose(
        #     0, 1)
        #
        #     all_features = [feat_a_t.detach(), pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s, None]

        return next_token_logits,next_light_logits,pos_a.detach(),agent_token_emb,proposal,feat_a

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            post_sampling=False
    ) -> Dict[str, torch.Tensor]:

        light_idx = tokenized_agent["light_idx"].clone()

        if "next_token_logits" not in tokenized_agent.keys() and len(light_idx):
            random_light = torch.randint(low=0, high=self.light_type, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            light_idx[random_mask] = random_light[random_mask]

        mask_lg=light_idx<self.light_type

        if self.use_latent:
            if "latent_z" not in tokenized_agent.keys():
                batch_idx = tokenized_agent['batch']
                #latent_z = torch.randint(low=0, high=self.k_dim, size=(max(batch_idx)+1, 1)).to(batch_idx.device)
                #latent_z = latent_z[batch_idx]
                latent_z=torch.randint(low=0, high=self.k_dim, size=(len(batch_idx), 1),device=batch_idx.device)

                tokenized_agent["latent_z"] = latent_z
        else:
            tokenized_agent["latent_z"]=None

        next_token_logits,next_light_logits,feat_a_token,agent_token_emb,proposal,feat_a= self.predict_agent(tokenized_agent["sampled_idx"],
                                                                                tokenized_agent["goal_idx"],
                                                                                tokenized_agent["valid_mask"],
                                                                                tokenized_agent["sampled_pos"],
                                                                                tokenized_agent["sampled_heading"] ,
                                                                                tokenized_agent,
                                                                                map_feature,
                                                                                light_idx,
                                                                                mask_lg,
                                                                                latent_z=tokenized_agent["latent_z"],
                                                                                post_sampling=post_sampling)

        # if self.use_lcf:
        #     lcf=torch.sigmoid(self.lcf_head(feat_a))*torch.pi/2
        # else:
        #     lcf=None



        tokenized_agent["next_token_logits"] = next_token_logits
        tokenized_agent["next_light_logits"] = next_light_logits
        tokenized_agent["feat_a"] =feat_a.detach()
        tokenized_agent["feat_a_nodetach"] =feat_a
        tokenized_agent["proposal"] = proposal
        tokenized_agent["feat_a_token"]=feat_a_token
        tokenized_agent["agent_token_emb"]=agent_token_emb

        return {
            "proposal":proposal,#[:,:-1],
            "goal_q":None,
            "light_q": next_light_logits,
            "agent_q": next_token_logits,            # action that goes from [(10->15), ..., (85->90)]
         }

    def autoregressive_agent(self, tokenized_agent, map_feature,current_step,max_step,post_sampling):

        # if "gt_z_raw" not in tokenized_agent.keys():
        #
        #     current_mask=tokenized_agent["valid_mask"][:,1]
        #     keep_mask=torch.rand(current_mask.sum())>0.05
        #
        #     for key in ['token_agent_shape', 'token_traj', 'token_traj_all', 'sampled_pos', 'sampled_heading', 'type', 'batch',
        #                 'shape', 'valid_mask', 'sampled_idx', 'next_token_logits'                       ]:
        #         tokenized_agent[key]=tokenized_agent[key][current_mask][keep_mask]

        # if "gt_z_raw" not in tokenized_agent.keys():
        # pos=tokenized_agent["gt_pos_raw"][:,:current_step+1]
        # heading=tokenized_agent["gt_head_raw"] [:,:current_step+1]
        # valid=tokenized_agent["gt_valid_raw"][:,:current_step+1]
        #
        # pos[:,0]=pos[:,0]+torch.randn_like(pos[:,0])*0.01
        # heading[:,0]=heading[:,0]+torch.randn_like(heading[:,0])*0.01
        #
        # agent_dict=self.token_processor._match_agent_token(valid,pos,heading,
        #                                                   tokenized_agent["token_agent_shape"],
        #                                                   tokenized_agent["token_traj"],shift=1)
        # #
        # sampled_idx=agent_dict["sampled_idx"]
        # mask=agent_dict["valid_mask"]
        # pos_a=agent_dict["sampled_pos"]
        # head_a=agent_dict["sampled_heading"]
        # else:
        sampled_idx=tokenized_agent["sampled_idx"][:, :current_step].clone()
        mask = tokenized_agent["valid_mask"][:, :current_step].clone()
        pos_a = tokenized_agent["sampled_pos"][:, :current_step].clone()
        head_a = tokenized_agent["sampled_heading"][:, :current_step].clone()
        #
        # print(torch.all(sampled_idx==agent_dict["sampled_idx"]))
        # print(torch.all(mask==agent_dict["valid_mask"]))
        # print(torch.all(pos_a==agent_dict["sampled_pos"]))
        # print(torch.all(head_a==agent_dict["sampled_heading"]))

        token_agent_shape=tokenized_agent["token_agent_shape"]
        token_traj=tokenized_agent["token_traj"]
        token_traj_all = tokenized_agent["token_traj_all"]
        light_idx = tokenized_agent["light_idx"][:, :current_step].clone()


        mask_lg=light_idx<self.light_type

        n_agent = sampled_idx.shape[0]

        if post_sampling:
            gt_valid=tokenized_agent["valid_mask"]
            gt_pos=tokenized_agent["sampled_pos"]
            gt_head=tokenized_agent["sampled_heading"]
            gt_sampled_idx = tokenized_agent["sampled_idx"]

        pred_traj_10hz = []
        pred_head_10hz = []
        sampled_log_prob=[]

        # if self.pred_vis:
        #     vis_mask=mask.clone()
        # else:
        #     vis_mask=None
        if self.use_latent:
            if "latent_z" in tokenized_agent.keys():
                latent_z = tokenized_agent["latent_z"]
            else:
                batch_idx = tokenized_agent['batch']
                #latent_z = torch.randint(low=0, high=self.k_dim, size=(max(batch_idx)+1, 1)).to(batch_idx.device)
                #latent_z = latent_z[batch_idx]
                latent_z=torch.randint(low=0, high=self.k_dim, size=(len(batch_idx), 1),device=batch_idx.device)
        else:
            latent_z=None

        goal_idx=tokenized_agent["goal_idx"][:,:2]

        for t in range(current_step, max_step + current_step):
            if t == current_step:
                if "next_token_logits" in tokenized_agent.keys() and tokenized_agent["next_token_logits"] is not None:
                    next_token_logits = tokenized_agent["next_token_logits"][:, :1]

                    if tokenized_agent["proposal"] is not None:
                        proposal=tokenized_agent["proposal"][:, :1]#[current_mask][keep_mask]

                    # if tokenized_agent["visibility"] is not None:
                    #     visibility=tokenized_agent["visibility"][:, :1]

                    if self.pred_light:
                        next_light_logits = tokenized_agent["next_light_logits"][:, :1]
                    else:
                        next_light_logits = []

                    if self.pred_goal:
                        next_goal_logits = tokenized_agent["next_goal_logits"][:, :1]
                    else:
                        next_goal_logits = []

                    # self.a_t_roformer.attn.cached_k=self.a_t_roformer.attn.cached_k[current_mask][keep_mask]
                    # self.a_t_roformer.attn.cached_v=self.a_t_roformer.attn.cached_v[current_mask][keep_mask]
                else:
                    self.a_t_roformer.attn.caching=True
                    if self.pred_light and not self.light_encoder.share:
                        self.light_encoder.lg_t_roformer.attn.caching=True
                    next_token_logits,next_light_logits,_,_,proposal,next_goal_logits = self.predict_agent(sampled_idx,goal_idx, mask, pos_a,
                                                                head_a,tokenized_agent, map_feature,light_idx,mask_lg,0,latent_z,post_sampling)

                self.a_t_roformer.attn.kv_caching(self.agent_hist,current_step)
                if self.pred_light and not self.light_encoder.share:
                    lg_num = tokenized_agent["pad_pos_lg"].shape[1]
                    self.light_encoder.lg_t_roformer.attn.kv_caching(self.light_hist,current_step*lg_num)
            else:
                next_token_logits,next_light_logits,_,_,proposal,next_goal_logits  = self.predict_agent(sampled_idx[:, -1:],goal_idx[:,-1:], mask[:, -self.agent_hist:],
                                                            pos_a[:, -2:], head_a[:, -1:],tokenized_agent, map_feature,light_idx[:, -1:],
                                                                                    mask_lg[:,-self.light_hist:],t - 1,latent_z,post_sampling)

            if post_sampling:
                next_token_idx=gt_sampled_idx[:,t]
            else:
                next_token_idx = Categorical(
                    logits=next_token_logits[:, -1, ] / self.alpha).sample()

                # log_prob=dist.log_prob(next_token_idx)
                #
                # sampled_log_prob.append(log_prob)

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

                next_light_idx = Categorical(logits=next_light_logits[:, -1] / self.alpha).sample()

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
                #if "gt_z_raw" in tokenized_agent.keys():
                mask =torch.cat([mask,torch.ones_like(mask[:,-1:]).to(torch.bool)], dim=1)
                # else:
                #     mask=torch.cat([mask,tokenized_agent["valid_mask"][:,t:t+1]], dim=1)
                mask_lg =torch.cat([mask_lg,torch.ones_like(mask_lg[:,-1:]).to(torch.bool)], dim=1)

            # if self.pred_vis:
            #     vis=torch.rand_like(visibility[:,-1:,0])<torch.sigmoid(visibility[:,-1:,0])
            #     vis=vis_mask[:,-1:] & vis
            #     vis_mask=torch.cat([vis_mask,vis],dim=1)

            if self.pred_goal:
                next_goal_idx=Categorical(logits=next_goal_logits[:, -1] / self.alpha).sample()

                goal_idx = torch.cat([goal_idx, next_goal_idx[:, None]], dim=1)

        self.a_t_roformer.attn.kv_caching(0)
        if self.pred_light and not self.light_encoder.share:
            self.light_encoder.lg_t_roformer.attn.kv_caching(0)

        #sampled_log_prob=torch.stack(sampled_log_prob,dim=1)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "sampled_idx": sampled_idx,  # [n_agent, 18]
           # "sampled_log_prob":sampled_log_prob,
           # "vis_mask": vis_mask,
            "light_idx": light_idx,
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
