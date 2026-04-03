# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import copy
import numpy as np

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized.functional import clamp
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli

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
from src.smart.layers.relative_transformer import RoFormerBlock, RoFormerDecoder
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.func import functional_call, jvp
from src.smart.utils.cluster import batch_increasing_schedule, allocate_k_per_type
from .denoiser import InitDenoiser
from src.smart.diffusion.diffusion_planner.sde import SDE, VPSDE_linear
from src.smart.diffusion.diffusion_planner.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from src.smart.layers import MLPLayer

from src.smart.loss.earth_match import get_matching_loss
from src.smart.modules.agent_decoder import SMARTAgentDecoder


class TrajFlow(nn.Module):

    def __init__(self,
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
            dis_weight,
            dist_decay,
            reward_weight,
            reward_decay,
            use_gail=False,
            discriminator=False,
            traj_diffusion=False,
        ):
        super().__init__()
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=5*3,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=token_processor,
            alpha=1,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            use_gail=False,
            traj_diffusion=True
        )

        self.register_buffer("normal_mean", torch.zeros(1, 90,3))
        self.register_buffer("normal_scale", torch.ones(1, 90,3))

        self.apply(weight_init)


    def forward(self, tokenized_agent,map_feature):

        sampled_pos=tokenized_agent["sampled_pos"]

        sampled_heading=tokenized_agent["sampled_heading"]

        valid_mask=tokenized_agent["valid_mask"]

        sampled_pos[~valid_mask]=torch.nan

        sampled_heading[~valid_mask]=torch.nan

        gt_traj_10hz=tokenized_agent['gt_traj_10hz']

        gt_head_10hz=tokenized_agent["gt_head_10hz"]

        gt_valid_10hz=tokenized_agent["gt_valid_10hz"]

        gt_traj_10hz[~gt_valid_10hz]=torch.nan
        gt_head_10hz[~gt_valid_10hz]=torch.nan

        sampled_idx=tokenized_agent["sampled_idx"]

        token_traj_all=tokenized_agent['token_traj_all']

        token_traj_sampled=token_traj_all[torch.arange(len(sampled_idx))[:,None],sampled_idx]

        token_sampled_pos=token_traj_sampled.mean(-2)

        diff_xy = token_traj_sampled[:, :, :,0] - token_traj_sampled[:, :,:, 3]
        token_sampled_head = torch.arctan2(diff_xy[:, :, :,1], diff_xy[:, :,:, 0])

        global_sampled_pos,global_global_head=transform_to_global(
            token_sampled_pos.reshape(-1,5,2),
            token_sampled_head.reshape(-1,5),
            torch.cat([gt_traj_10hz[:,:1],sampled_pos[:,:-1]],dim=1).reshape(-1,2),
            torch.cat([gt_head_10hz[:,:1],sampled_heading[:,:-1]],dim=1).reshape(-1),
        )

        sampled_pos=sampled_pos.reshape(-1,2)
        sampled_heading=sampled_heading.reshape(-1)

        local_sampled_pos,local_sampled_heading=transform_to_local(
            global_sampled_pos,
            global_global_head,
            sampled_pos,
            sampled_heading
        )

        sampled_init=torch.cat([local_sampled_pos,local_sampled_heading[:,:,None]],dim=-1).reshape(-1,18*5,3)

        local_pos,local_heading=transform_to_local(
            gt_traj_10hz[:,1:].reshape(-1,5,2),
            gt_head_10hz[:,1:].reshape(-1,5),
            sampled_pos,
            sampled_heading,
        )


        m_init=torch.cat([local_pos-local_sampled_pos,wrap_angle(local_heading-local_sampled_heading)[:,:,None]],dim=-1).reshape(-1,18*5,3)

        m_init[(m_init.abs()>4).any(-1)]=torch.nan

        if torch.all(self.normal_mean==0):

            valid = ~torch.isnan(m_init)
            count = valid.sum(0, keepdim=True).clamp_min(1)

            mean = torch.where(valid, m_init, 0).sum(0, keepdim=True) / count
            std = torch.sqrt(torch.where(valid, (m_init - mean) ** 2, 0).sum(0, keepdim=True) / count).clamp_min(
                1e-8)
            self.normal_mean.copy_(mean)
            self.normal_scale.copy_(std)

        e=torch.randn_like(m_init)

        e=e* self.normal_scale+self.normal_mean

        device = m_init.device
        num_graphs = tokenized_agent["num_graphs"]
        agent_batch=tokenized_agent["batch"]

        t_batch = torch.rand(num_graphs, device=device)[:, None, None]  # t ~ U[0,1]

        t = t_batch[agent_batch]

        diff = (1 - t) * e + t * m_init  # large t, low noise        target velocity e-x = (z-x)/(1-t)

        new_local_traj=sampled_init+diff

        global_sampled_pos,global_global_head=transform_to_global(
            new_local_traj[:,:,:2].reshape(-1,5,2),
            new_local_traj[:,:,2].reshape(-1,5),
            sampled_pos,
            sampled_heading
        )

        noisy_sampled_pos=global_sampled_pos[:,-1]

        noisy_sampled_heading=global_global_head[:,-1]

        noisy_local_pos,noisy_local_heading=transform_to_local(
            global_sampled_pos[:,:-1],
            global_global_head[:,:-1],
            noisy_sampled_pos,
            noisy_sampled_heading
        )

        noisy_poses=torch.cat([noisy_local_pos,wrap_angle(noisy_local_heading)[:,:,None]],dim=-1).clone()

        noisy_poses[torch.isnan(noisy_poses)] = -10

        noisy_sampled_pos=noisy_sampled_pos.reshape(-1,18,2)
        noisy_sampled_heading=noisy_sampled_heading.reshape(-1,18)

        valid_mask=~torch.isnan(noisy_sampled_heading)

        noisy_sampled_pos[~valid_mask]=0
        noisy_sampled_heading[~valid_mask]=0

        token_mask=torch.cat([valid_mask[:,:1] , valid_mask[:,1:] & valid_mask[:,:-1]],dim=-1)

        noise_pred=self.agent_encoder.predict_agent(noisy_poses.reshape(-1,18,12),
                                        token_mask,
                                        valid_mask,
                                        noisy_sampled_pos,
                                        noisy_sampled_heading ,
                                        tokenized_agent,
                                        map_feature,
                                        tokenized_agent["shape"])[0]

        noise_pred=noise_pred.reshape(-1,5,3)

        pred_global_pos,pred_global_heading=transform_to_global(
            noise_pred[:,:,:2],
            noise_pred[:,:,2],
            noisy_sampled_pos[valid_mask],
            noisy_sampled_heading[valid_mask]
        )

        gt_traj=gt_traj_10hz[:, 1:].reshape(-1,18,5,2)[valid_mask]
        gt_head=gt_head_10hz[:, 1:].reshape(-1,18,5)[valid_mask]

        nan_mask=~(torch.isnan(gt_head) | torch.isnan(pred_global_heading))

        pos_loss=(pred_global_pos-gt_traj)[nan_mask].abs().mean()

        heading_loss=wrap_angle(pred_global_heading-gt_head)[nan_mask].abs().mean()

        loss={
            "next_token_logits":None,
            "initial_logit":None,
            "pos_loss":pos_loss,
            "heading_loss":heading_loss,
        }

        return loss

    def sample(self,tokenized_agent,map_feature,steps=20):

        agent_batch = tokenized_agent["batch"]
        num_agents = len(agent_batch)

        z = torch.randn(num_agents, 90, 3, device=agent_batch.device)

        z=z* self.normal_scale+self.normal_mean

        timesteps = torch.linspace(0, 1, steps + 1, device=agent_batch.device)

        global_sampled_pos=tokenized_agent["pred_traj_10hz"]
        global_global_head=tokenized_agent["pred_head_10hz"]
        sampled_pos=tokenized_agent["sampled_pos"].reshape(-1, 2)
        sampled_heading=tokenized_agent["sampled_heading"].reshape(-1)

        local_sampled_pos, local_sampled_heading = transform_to_local(
            global_sampled_pos[:,1:].reshape(-1,5,2),
            global_global_head[:,1:].reshape(-1,5),
            sampled_pos,
            sampled_heading
        )

        token_mask=tokenized_agent["token_mask"]
        valid_mask=tokenized_agent["valid_mask"]

        sampled_init = torch.cat([local_sampled_pos, local_sampled_heading[:, :, None]], dim=-1).reshape(-1, 18 * 5, 3)

        for i in range(steps):  # - 1
            t = timesteps[i]
            t_next = timesteps[i + 1]

            new_local_traj = sampled_init + z

            global_sampled_pos, global_global_head = transform_to_global(
                new_local_traj[:, :, :2].reshape(-1, 5, 2),
                new_local_traj[:, :, 2].reshape(-1, 5),
                sampled_pos,
                sampled_heading
            )

            noisy_sampled_pos = global_sampled_pos[:, -1]

            noisy_sampled_heading = global_global_head[:, -1]

            noisy_local_pos, noisy_local_heading = transform_to_local(
                global_sampled_pos[:, :-1],
                global_global_head[:, :-1],
                noisy_sampled_pos,
                noisy_sampled_heading
            )

            noisy_poses = torch.cat([noisy_local_pos, wrap_angle(noisy_local_heading)[:, :, None]], dim=-1).clone()

            noisy_sampled_pos = noisy_sampled_pos.reshape(-1, 18, 2)
            noisy_sampled_heading = noisy_sampled_heading.reshape(-1, 18)

            noise_pred = self.agent_encoder.predict_agent(noisy_poses.reshape(-1, 18, 12),
                                                          token_mask,
                                                          valid_mask,
                                                          noisy_sampled_pos,
                                                          noisy_sampled_heading,
                                                          tokenized_agent,
                                                          map_feature,
                                                          tokenized_agent["shape"])[0]

            noise_pred = noise_pred.reshape(-1, 5, 3)

            pred_global_pos, pred_global_heading = transform_to_global(
                noise_pred[:, :, :2],
                noise_pred[:, :, 2],
                noisy_sampled_pos[valid_mask],
                noisy_sampled_heading[valid_mask]
            )

            v_pred = (x_cond - z) / (1.0 - t_n).clamp_min(self.t_eps)



        return 1