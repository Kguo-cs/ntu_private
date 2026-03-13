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
from src.smart.diffusion.diffusion_planner.sde import SDE,VPSDE_linear
from src.smart.diffusion.diffusion_planner.dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver


def power_schedule(steps, device, alpha=2.0):
    return 1 - (1 - torch.linspace(0, 1, steps, device=device)) ** alpha


def cosine_schedule(steps, device):
    i = torch.arange(steps + 1, device=device)
    return 0.5 * (1 - torch.cos(torch.pi * i / steps))


class ScaleFlow(nn.Module):

    def __init__(self, args, token_processor):
        super().__init__()
        self.ego_embedding = MLPLayer(20 + 3, args.hidden_dim, args.hidden_dim)  # +3

        self.net = InitDenoiser(
            token_processor,
            dataset=args.dataset,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            output_head=args.output_head,
            init_timestep=args.init_timestep,
            num_freq_bands=args.num_freq_bands,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
            diff_type=args.diff_type,
            m_dim=args.m_dim,
        )

        self.sde = VPSDE_linear()


    def get_loss(self,
            diff_input,
            tokenized_agent: HeteroData,
            scene_enc: Mapping[str, torch.Tensor],
            eval_mask,
            eps: float = 1e-3,
    ):

        all_gt = diff_input

        B = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["nonego_batch"]

        t = torch.rand(B, device=all_gt.device) * (1 - eps) + eps  # [B,]
        z = torch.randn_like(all_gt, device=all_gt.device)  # [B, T, 4]

        t = t[agent_batch][:,None]


        mean, std = self.sde.marginal_prob(all_gt, t)
        # std = std.view(-1, *([1] * (len(all_gt.shape) - 1)))

        xT = mean + std * z

        score = self.net(xT[:,None], t[:,None], tokenized_agent, scene_enc)[:,0]  # [B, T, 4]

        supervision_type='x_start'

        # model_type='x_start'
        #
        # supervision_type = supervision_type if supervision_type is not None else model_type
        # pred_pattern = f"{model_type}->{supervision_type}"
        # score = self.sde.transform(pred_pattern, score, t, xT)

        if supervision_type == "score":
            dpm_loss = torch.sum((score * std + z) ** 2, dim=-1)  # to avoid exploding variance
        elif supervision_type == "x_start":
            dpm_loss = torch.sum((score - all_gt) ** 2, dim=-1)
        elif supervision_type == "noise":
            dpm_loss = torch.sum((score - z) ** 2, dim=-1)
        elif supervision_type == "v":
            v = self.sde.transform("noise->v", z, t, xT)
            dpm_loss = torch.sum((score - v) ** 2, dim=-1)

        denom = 1#(1 - t).clamp_min(self.t_eps)[:,0]

        return dpm_loss,score[:,0],z,denom

    def sample(self,
               tokenized_agent: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               eval_mask,
               num_samples: int,
               start_data=None,
               reverse_steps=None,
               sampling="ddpm",
               stride=20,
               if_output_diffusion_process=False,
               ) -> Dict[str, torch.Tensor]:

        agent_batch = tokenized_agent["nonego_batch"]
        num_agents = len(agent_batch)

        x_T = torch.randn([num_agents,  self.net.output_dim]).to(agent_batch.device)#* 0.1

        noise_schedule = NoiseScheduleVP(
            schedule='linear'
        )

        other_model_params = {
            "scene_enc": scene_enc,
            "tokenized_agent": tokenized_agent,
        }
        dpm_solver_params = {}
        model_wrapper_params = {}

        model_fn = model_wrapper(
            self.net,  # use your noise prediction model here
            noise_schedule,
            model_type="x_start" ,  # or "x_start" or "v" or "score"
            model_kwargs=other_model_params,
            **model_wrapper_params
        )
        diffusion_steps=10

        dpm_solver = DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params) # w.o. dynamic thresholding

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
        )


        return sample_dpm,[],[],[]

