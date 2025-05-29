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
import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_scatter import scatter_mean

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from .kl_loss import DiagGaussian
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from .build_edge import radiusGraphNearest, radiusGraphNearest2, positionalencoding1d, generate_causal_mask, \
    generate_limited_causal_mask
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock, general_rope


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
            use_latent: bool = False
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor

        self.alpha = 0.1

        self.head_dim = hidden_dim // num_heads

        self.pred_agent = True

        if self.pred_agent:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

            input_dim_x_a = 2
            input_dim_token = 8

            self.x_a_emb = FourierEmbedding(
                input_dim=input_dim_x_a,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
            self.token_emb_veh = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_ped = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.token_emb_cyc = MLPEmbedding(
                input_dim=input_dim_token, hidden_dim=hidden_dim
            )
            self.fusion_emb = MLPEmbedding(
                input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
            )

            self.agent_hist = self.time_span // self.shift

            self.a_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=hist_drop_prob)
            self.pt2a_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            self.a2a_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

            self.token_predict_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
            )

        self.pred_light = False

        if self.pred_light:
            self.lg_time_span = time_span

            self.light_hist = time_span // self.shift

            self.light_type = 5

            self.light_dropout = 0

            self.light_embedding = nn.Embedding(self.light_type, hidden_dim)

            self.lg_t_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)
            self.lg2lg_roformer = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=self.light_dropout)

            self.light_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)

        self.pred_route = False

        self.apply(weight_init)

    def padding(self,tensor,lengths ):
        padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=0)

        return padded_tensor

    def agent_token_embedding(
            self,
            agent_token_index,  # [n_agent, n_step]
            trajectory_token_veh,  # [n_token, 8]
            trajectory_token_ped,  # [n_token, 8]
            trajectory_token_cyc,  # [n_token, 8]
            pos_a,  # [n_agent, n_step, 2]
            head_vector_a,  # [n_agent, n_step, 2]
            agent_type,  # [n_agent]
            agent_shape,  # [n_agent, 3]
            inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    def temporal_embed(self, feature, network, n_step, n_current, hist_len, mask=None):

        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        positions = torch.arange(n_current, n_step + n_current, device=feature.device)[:, None]

        sinusoidal_pos = general_rope(positions, self.head_dim)

        if mask is not None:
            causal_mask = causal_mask[None,None] | mask[:,None,None,:]

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature

    def spatial_embed(self, feature, network, lengths, sinusoidal_poshead, spatial_mask=None):

        padded_feature = self.padding(feature, lengths)

        padded_feature = padded_feature.permute(2, 0, 1, 3).flatten(0, 1)

        src_key_padding_mask = (padded_feature!=0).any(-1)
        # src_key_padding_mask = (padded_feature.abs().sum(dim=-1) != 0)
        # nn.TransformerEncoderLayer()
        attn_mask = src_key_padding_mask & (padded_feature != -1).any(-1)  # batch 1 time step 0

        if spatial_mask is not None:
            attn_mask =  attn_mask[:, None] & spatial_mask
        else:
            attn_mask = attn_mask[:, None]

        padded_feature1 = network(padded_feature, ~attn_mask[:, None], sinusoidal_pos=sinusoidal_poshead)

        feature_out = padded_feature1[src_key_padding_mask].reshape(feature.shape[1], feature.shape[0], -1).swapaxes(0, 1)

        return feature_out
            

    def predict_light(self, light_idx, sinusoidal_poshead, lengths, n_current=0):
        n_step = light_idx.shape[1]

        feat_lg = self.light_embedding(light_idx)

        feat_lg = self.temporal_embed(feat_lg, self.lg_t_roformer, n_step, n_current, self.light_hist)

        (padded_sin, padded_cos, spatial_mask) = sinusoidal_poshead

        sin = padded_sin[None].repeat(n_step, 1, 1, 1, 1).flatten(0, 1)

        cos = padded_cos[None].repeat(n_step, 1, 1, 1, 1).flatten(0, 1)

        spatial_mask=spatial_mask[None].repeat(n_step, 1, 1, 1).flatten(0, 1)

        feat_lg = self.spatial_embed(feat_lg, self.lg2lg_roformer, lengths, (sin, cos),spatial_mask)

        next_light_logits = self.light_token_predict_head(feat_lg)

        return feat_lg, next_light_logits

    def predict_agent(self, sampled_idx, mask ,pos_a,head_a,tokenized_agent, map_feature, n_current=0):
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=sampled_idx,  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        feat_a = self.temporal_embed(feat_a, self.a_t_roformer, n_step, n_current, self.agent_hist, ~mask)

        feat_a[~mask[:,-n_step:]] = -1

        lengths=tokenized_agent["batch_lengths"]

        # padded_mask1 = pad_sequence(torch.split(mask, lengths), batch_first=True, padding_value=False)
        #
        # padded_mask=padded_mask1.permute(2,0,1).flatten(0,1)#batch 1 agent 1 no first time step ?



        # feat_a = self.pt2a_attn_layers(feat_map
        #     (feat_map, feat_a), r_pl2a, edge_index_pl2a
        # )

        sin, cos = general_rope(pos_a.flatten(0,1), self.head_dim, head_a.flatten(0,1))

        padded_sin = self.padding(sin.reshape(n_agent, n_step, -1), lengths)

        padded_cos = self.padding(cos.reshape(n_agent, n_step,-1), lengths)

        sin = padded_sin.permute(2,0,1,3).flatten(0, 1)[:,None]

        cos = padded_cos.permute(2,0,1,3).flatten(0, 1)[:,None]

        # padded_pos = pad_sequence(torch.split(pos_a, lengths), batch_first=True, padding_value=0)
        # spatial_mask=spatial_mask[None].repeat(n_step, 1, 1, 1).flatten(0, 1)
        pt_feature=map_feature["pt_token"]
        map_lengths=map_feature["map_lengths"] 
        map_sinusoidal=map_feature["map_sinusoidal"] 

        padded_pt_feature = self.padding(pt_feature, map_lengths)

        padded_pt_feature=padded_pt_feature[None].repeat(n_step,1,1,1).flatten(0,1)

        src_key_padding_mask = (padded_pt_feature!=0).any(-1)

        attn_mask = src_key_padding_mask[:, None]

        padded_a_feature = self.padding(feat_a, lengths)

        padded_a_feature = padded_a_feature.permute(2, 0, 1, 3).flatten(0, 1)
        
        src_key_padding_mask = (padded_a_feature!=0).any(-1)

        (map_sin,map_cos)=map_sinusoidal

        map_sin=map_sin[None].repeat(n_step,1,1,1,1).flatten(0,1)
        map_cos=map_cos[None].repeat(n_step,1,1,1,1).flatten(0,1)

        feat_pta = self.pt2a_roformer(padded_a_feature, ~attn_mask[:, None], (sin,cos),
                                             padded_pt_feature,(map_sin,map_cos)
                                             )

        feat_a = feat_pta[src_key_padding_mask].reshape(feat_a.shape[1], feat_a.shape[0], -1).swapaxes(0, 1)

        feat_st = self.spatial_embed(feat_a, self.a2a_roformer, lengths, (sin, cos))

        next_token_logits = self.token_predict_head(feat_st)

        #next_token_logits[~mask]=0

        return next_token_logits

    def forward(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.pred_light:
            light_idx = tokenized_agent["light_idx"]
            lengths = tokenized_agent["lengths"]
            sinusoidal_poshead = tokenized_agent["sinusoidal_poshead"]

            noised_light_idx = light_idx.clone()

            random_light = torch.randint(low=0, high=3, size=light_idx.shape, device=light_idx.device).long()

            random_mask = torch.rand_like(light_idx.float()) > 0.9

            random_mask[:, :2] = False

            noised_light_idx[random_mask] = random_light[random_mask]

            feat_lg, next_light_logits = self.predict_light(noised_light_idx, sinusoidal_poshead, lengths)

        if not self.pred_agent:
            tokenized_agent["next_light_logits"] = next_light_logits
            tokenized_agent["feat_lg"] = feat_lg

            return {
                "q_value": next_light_logits[:, 1:]
            }

        sampled_idx=tokenized_agent["sampled_idx"]
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]

        next_token_logits = self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature)

        tokenized_agent["next_token_logits"] = next_token_logits

        return {
             "q_value": next_token_logits[:, 1:],            # action that goes from [(10->15), ..., (85->90)]
         }

    def autoregressive_light_predict(self,  tokenized_agent, current_len,max_len):
        predicted_tokens = tokenized_agent["light_idx"][:, :current_len].clone()
        lengths = tokenized_agent["lengths"]
        sinusoidal_poshead = tokenized_agent["sinusoidal_poshead"]

        self.lg_t_roformer.attn.kv_caching(self.light_hist)

        for t in range(current_len, max_len + current_len):
            if t == current_len:
                if "feat_lg" in tokenized_agent.keys():
                    lg_features = tokenized_agent["feat_lg"][:, :current_len]
                    next_light_logits = tokenized_agent["next_light_logits"][:, :current_len]

                    self.lg_t_roformer.attn.cached_k = self.lg_t_roformer.attn.cached_k[:, :, :current_len]
                    self.lg_t_roformer.attn.cached_v = self.lg_t_roformer.attn.cached_v[:, :, :current_len]
                else:
                    lg_features, next_light_logits = self.predict_light(predicted_tokens, sinusoidal_poshead,
                                                                                 lengths)
            else:
                feat_lg, next_light_logits = self.predict_light(predicted_tokens[:, -1:], sinusoidal_poshead,
                                                                         lengths, t - 1)

                lg_features = torch.cat([lg_features, feat_lg[:, -1:]], dim=1)

            cat_dist = Categorical(logits=next_light_logits[:, -1] / self.alpha)

            samples = cat_dist.sample()

            predicted_tokens = torch.cat([predicted_tokens, samples[:, None]], dim=1)

        self.lg_t_roformer.attn.kv_caching(0)

        return predicted_tokens, lg_features

    def autoregressive_agent(self, tokenized_agent, map_feature,current_len,max_len):

        sampled_idx=tokenized_agent["sampled_idx"][:, :current_len].clone()
        mask = tokenized_agent["valid_mask"][:, :current_len].clone()
        pos_a = tokenized_agent["sampled_pos"][:, :current_len].clone()
        head_a = tokenized_agent["sampled_heading"][:, :current_len].clone()
        token_traj_all = tokenized_agent["token_traj_all"]

        if "gt_z_raw" in tokenized_agent.keys():
            n_agent=sampled_idx.shape[0]
            pred_traj_10hz = torch.zeros(
                [n_agent, 0, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, 0], dtype=pos_a.dtype, device=pos_a.device
            )

        for t in range(current_len, max_len + current_len):
            if t == current_len:
                if "next_token_logits" in tokenized_agent.keys():
                    next_token_logits = tokenized_agent["next_token_logits"][:, :current_len]

                    self.a_t_roformer.attn.cached_k = self.a_t_roformer.attn.cached_k[:, :, :current_len]
                    self.a_t_roformer.attn.cached_v = self.a_t_roformer.attn.cached_v[:, :, :current_len]
                else:
                    next_token_logits = self.predict_agent(sampled_idx, mask, pos_a, head_a,tokenized_agent, map_feature)

                self.a_t_roformer.attn.kv_caching(self.agent_hist)
   
            else:
                next_token_logits = self.predict_agent(sampled_idx[:, -1:], mask[:, -min(t,self.agent_hist):], pos_a[:, -1:], head_a[:, -1:],tokenized_agent, map_feature,t - 1)

            cat_dist = Categorical(logits=next_token_logits[:, -1] / self.alpha)

            next_token_idx = cat_dist.sample()

            sampled_idx = torch.cat([sampled_idx, next_token_idx[:, None]], dim=1)

            range_a = torch.arange(next_token_idx.shape[0])

            next_token_traj_all = token_traj_all[range_a, next_token_idx]

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, -1],  # [n_agent, 2]
                head_now=head_a[:, -1],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if "gt_z_raw" in tokenized_agent.keys():

                pred_traj=token_traj_global[:, 1:].mean( 2 )
                pred_traj_10hz=torch.cat([pred_traj_10hz, pred_traj],dim=1)
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head= torch.arctan2( diff_xy[:, :, 1], diff_xy[:, :, 0]   )
                pred_head_10hz=torch.cat([pred_head_10hz,pred_head],dim=1)


            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            mask =torch.cat([mask,torch.ones_like(head_a_next).to(torch.bool).unsqueeze(1)], dim=1)

        self.a_t_roformer.attn.kv_caching(0)

        out_dict = {
            "type": tokenized_agent["type"],
            "shape": tokenized_agent["shape"],
            "sampled_pos": pos_a,  # [n_agent, 18, 2]
            "sampled_heading": head_a,  # [n_agent, 18]
            "valid_mask": mask,  # [n_agent, 18]
            "sampled_idx": sampled_idx,  # [n_agent, 18]
        }


        if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
            out_dict["gt_pos_raw"] = tokenized_agent["gt_pos_raw"]  # [n_agent, 18, 2]
            out_dict["gt_head_raw"] = tokenized_agent["gt_head_raw"]  # [n_agent, 18]
            out_dict["gt_valid_raw"] = tokenized_agent["gt_valid_raw"]  # [n_agent, 18]

        return out_dict

    def inference(
            self,
            tokenized_agent: Dict[str, torch.Tensor],
            map_feature: Dict[str, torch.Tensor],
            sampling_scheme: DictConfig
    ) -> Dict[str, torch.Tensor]:
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        if self.pred_light:
            pred_light_idx, lg_features = self.autoregressive_light_predict(tokenized_agent,step_current_2hz,
                                                                               n_step_future_2hz)

        if not self.pred_agent:
            return {"light_idx": pred_light_idx}

        out_dict =self.autoregressive_agent(tokenized_agent, map_feature,step_current_2hz, n_step_future_2hz)

        # out_dict = {
        #     # action that goes from [(10->15), ..., (85->90)]
        #     "next_token_logits": torch.stack(next_token_logits_list, dim=1),
        #     "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
        #     # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
        #     "pred_pos": pos_a,  # [n_agent, 18, 2]
        #     "pred_head": head_a,  # [n_agent, 18]
        #     "pred_valid": pred_valid,  # [n_agent, 18]
        #     "pred_idx": pred_idx,  # [n_agent, 18]
        #     # or use the tokenized gt
        #     "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
        #     "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
        #     "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        #     # for shifting proxy targets by lr
        #     "next_token_action": torch.stack(next_token_action_list, dim=1),
        #     # "sample_list":sample_list,
        #     # "feat_a": torch.stack(feat_a_list,dim=1),
        #     # "action_log_probs": torch.stack(action_log_probs_list, dim=1),
        #     "type": tokenized_agent["type"],
        #     "shape": tokenized_agent["shape"],
        #     "sampled_pos": pos_a,  # [n_agent, 18, 2]
        #     "sampled_heading": head_a,  # [n_agent, 18]
        #     "valid_mask": pred_valid,  # [n_agent, 18]
        #     "sampled_idx": pred_idx,  # [n_agent, 18]
        #     # "rollout_entropy":torch.stack(entropy_list)
        # }
        #
        # if self.pred_light:
        #     out_dict["light_idx"] = pred_light_idx
        #     # out_dict["lg_features"]=lg_features

        # if "gt_z_raw" in tokenized_agent.keys():  # 10hz predictions for wosac evaluation and submission
        #     out_dict["pred_traj_10hz"] = pred_traj_10hz
        #     out_dict["pred_head_10hz"] = pred_head_10hz
        #     pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
        #     out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
        #     out_dict["gt_pos_raw"] = tokenized_agent["gt_pos_raw"]  # [n_agent, 18, 2]
        #     out_dict["gt_head_raw"] = tokenized_agent["gt_head_raw"]  # [n_agent, 18]
        #     out_dict["gt_valid_raw"] = tokenized_agent["gt_valid_raw"]  # [n_agent, 18]
        #
        return out_dict
