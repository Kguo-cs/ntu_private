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
from tensorflow_probability.python.bijectors import scale
from torch.ao.nn.quantized.functional import clamp
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli

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
from .denoiser import InitDenoiser
from src.smart.diffusion.diffusion_planner.sde import SDE,VPSDE_linear
from src.smart.diffusion.diffusion_planner.dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver
from src.smart.layers import MLPLayer

from src.smart.loss.earth_match import get_matching_loss,multi_circle_collision_loss_mem_efficient,get_scale
from ..loss.earth_match import gaussian_nll
from src.smart.diffusion.dit.dit import DiT


def calculate_shift(
    image_seq_len,
    base_seq_len: int =32, #256,
    max_seq_len: int = 256,#4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class ScaleFlow(nn.Module):

    def __init__(self, args,token_processor):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

        self.mean_flow=False

        self.hidden_dim=args.hidden_dim

        self.use_dit=False

        if self.use_dit:
            self.x_pred=False
        else:
            self.x_pred=True

        if self.use_dit:
            self.model = DiT(self.hidden_dim)
            self.lane_embed1 = nn.Linear(128 + 4, self.hidden_dim)
        else:
            self.model = InitDenoiser(
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
                mean_flow=self.mean_flow,
                x_pred=self.x_pred
            )

        if not self.model.use_rel_ego:
            self.ego_embedding1 = MLPLayer(16 + 3, args.hidden_dim, args.hidden_dim)

        self.infer_time_per_step = []
        self.GPU_incre_memory = []
        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)

        self.use_scale=self.model.use_scale

        self.use_all_type=self.model.use_all_type

        if self.x_pred:
            self.t_eps=0.05
        else:
            self.t_eps=0

        self.P_std=1

        self.P_mean=2

        self.use_cluster=False

        self.use_vp=False
        if self.use_vp:
            self.sde = VPSDE_linear()

        self.use_dpm_solver=False

        self.use_flow_ode=False

        self.use_flux=False

        self.use_sde=False

        self.noise_level=0.7

        self.rationorm=False

        self.global_step=0

        self.use_nft=False

        if self.use_nft:
            self.old_model = copy.deepcopy(self.model)
            self.use_sde=False

        if self.use_flow_ode:
            from .flow_planner.flow_ode import FlowODE
            from flow_matching.path.affine import  AffineProbPath
            from flow_matching.path.scheduler import  CondOTScheduler
            from .flow_planner.time_sampler import TimeSampler
            path=AffineProbPath(CondOTScheduler())
            time_sampler=TimeSampler(device='cuda',eps=1e-3,alpha=1.0,beta=1.5)
            self.flow_ode=FlowODE(path,time_sampler,cfg_weight=1.8,sample_steps=self.steps+1,sample_method='midpoint',sample_temperature=1)

        self.apply(weight_init)

    def get_loss(self,
                 x,
                 tokenized_agent: HeteroData,
                 initial_map_feature: Mapping[str, torch.Tensor],
                 eval_mask,
                 num_samples=1) :
        device = x.device
        num_graphs = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["nonego_batch"]
        nonego_type=tokenized_agent["nonego_type"]
        
        if self.use_dit:
            lane_batch = initial_map_feature["batch"][::2]
            pos_pl = initial_map_feature["position"][::2]
            orient_pl = initial_map_feature["orientation"][::2]
            feat_map = initial_map_feature["pt_token"][::2]
            initial_map_feature = self.lane_embed1(
                torch.cat([feat_map, pos_pl, orient_pl.cos()[:, None], orient_pl.sin()[:, None]], dim=-1))

            tokenized_agent["lane_batch"]=lane_batch

        x=x.unsqueeze(1).repeat(1, num_samples, 1)

        e = torch.randn_like(x)  # base distribution N(0, I)

        e=self.model.denormalize(e,nonego_type)

        if "step_idx" in tokenized_agent.keys():
            timesteps=torch.linspace(0,1,tokenized_agent["step_number"]+1,device=device)
            t_batch = timesteps[tokenized_agent["step_idx"]]
            t_batch = t_batch[:, None, None]
        elif self.use_flow_ode:
            t_batch = self.flow_ode.time_sampler.sample(num_graphs).to(device)[:, None,None]
        else:
            t_batch = torch.rand(num_graphs, device=device)[:, None,None]  # t ~ U[0,1]

        t=t_batch[agent_batch]

        if self.use_scale:
            nan_mask=torch.isnan(x)

            padding_mask =torch.all(nan_mask,dim=-1)

            t[padding_mask]=0

            x[nan_mask]=0

        if self.use_vp:
            mean, std = self.sde.marginal_prob(x, t)
            z = mean + std * e
        elif self.use_flow_ode:
            path_sample = self.flow_ode.path.sample(x_0=e, x_1=x, t=t[:,0,0])

            z=path_sample.x_t
        else:
            z = (1 - t) * e + t * x #large t, low noise        target velocity e-x = (z-x)/(1-t)

        if "advantages" in tokenized_agent.keys():
            advantages=tokenized_agent["advantages"]

            if self.use_sde:

                z_list=tokenized_agent["z_list"]

                t_list=tokenized_agent["t_list"]

                z_sampled, prev_sample, log = z_list
                t_n_sampled, t_next_sampled = t_list
            else:
                x_sampled=tokenized_agent["z_list"]#.repeat(2,1,1)
                e_sampled=torch.randn_like(e)
                t_n_sampled=torch.rand_like(t_batch)[agent_batch]

                z_sampled = (1 - t_n_sampled) * e_sampled + t_n_sampled * x_sampled

            denom = (1.0 - t_n_sampled).clamp_min(self.t_eps)

            if self.use_nft:
                with torch.no_grad():
                    decay = return_decay(self.global_step, 2)
                    # print(decay,self.global_step)
                    for src_param, tgt_param in zip(  self.model.parameters(),   self.old_model.parameters(), strict=True    ):
                        tgt_param.data.copy_(
                            tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

                    self.global_step+=1

                    self.old_model.eval()
                    old_prediction = self.old_model(z_sampled, t_n_sampled, tokenized_agent, tokenized_agent["initial_map_feature"])
                    if self.x_pred:
                        old_v_pred = (old_prediction - z_sampled) / denom
                    else:
                        old_v_pred = old_prediction

            t_n=torch.cat((t_n_sampled,t),dim=0)

            z_all=torch.cat((z_sampled,z),dim=0)

            tokenized_agent=self.repeat_input(tokenized_agent,2)

            if self.model.use_return_conditioned:
                tokenized_agent["advantages"]=torch.cat((advantages,torch.ones_like(advantages)),dim=0)

            x_pred_all = self.model(z_all, t_n, tokenized_agent, tokenized_agent["initial_map_feature"])

            if self.model.use_return_conditioned:
                x_pred=x_pred_all
                x=torch.cat((x_sampled,x),dim=0)
                z=z_all
                e=torch.cat((e_sampled,e),dim=0)
                t=t_n
                policy_loss = 0
            else:
                if self.x_pred:
                    v_pred = (x_pred_all[:len(z_sampled)] - z_sampled) / denom
                else:
                    v_pred = x_pred_all[:len(z_sampled)]

                adv_clip_max=5

                advantages_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

                if self.use_nft:
                    beta=1
                    forward_prediction=v_pred   # v=(x0-z)/(1-t)     z=(1-t)*e+t*x0
                    x0=x_sampled
                    xt=z_sampled
                    t_expanded=-denom #  x0_pred=    (1-t) *v+ z
                    old_prediction=old_v_pred

                    normalized_advantages_clip = (advantages_clip-advantages_clip.min())/(advantages_clip.max()-advantages_clip.min())#(advantages_clip / adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    positive_prediction = beta * forward_prediction + (1 - beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + beta
                    ) * old_prediction.detach() - beta * forward_prediction
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                   # weight_factor=1
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                  #  negative_weight_factor=1
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
                    policy_loss = (ori_policy_loss * adv_clip_max).mean()*10
                else:
                    if self.use_sde:
                        prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sde_step_with_logprob(
                            1 - t_n_sampled,
                            1 - t_next_sampled,
                            -v_pred,
                            z_sampled,
                            noise_level=self.noise_level,
                            prev_sample=prev_sample
                        )
                    else:
                        x_pred = x_pred_all[:len(z_sampled)]

                        match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_matching_loss(
                            tokenized_agent,
                            x_pred[:, 0],
                            x_sampled[:, 0],
                            z_sampled[:, 0],
                            e_sampled[:, 0],
                            t_n_sampled[:, 0],
                            x_pred=self.x_pred
                        )
                        log_prob=torch.exp(match_loss.detach()-match_loss )#-match_loss#

                    per_sample_policy_loss = - log_prob * advantages_clip

                    if self.rationorm:
                        sigma_t = std_dev_t.mean()

                        per_sample_policy_loss=per_sample_policy_loss * sigma_t

                    policy_loss = per_sample_policy_loss.mean()*10

                x_pred=x_pred_all[len(z_sampled):]
        else:
            policy_loss=0
            x_pred = self.model(z, t, tokenized_agent, initial_map_feature)

        match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_matching_loss(
            tokenized_agent,
            x_pred[:,0],
            x[:,0],
            z[:,0],
            e[:,0],
            t[:,0],
            use_col=False,
            x_pred=self.x_pred
        )


        loss=(match_loss, collision_loss+policy_loss, pos_loss, heading_loss, shape_loss, vel_loss)

        return loss ,x_pred[:,0],z[:,0],t[:,0] #,denom[:,0]

    def sde_step_with_logprob(
            self,
            sigma,
            sigma_prev,
            model_output: torch.FloatTensor,
            sample: torch.FloatTensor,
            noise_level = 0.7,
            prev_sample=None,
            sde_type: Optional[str] = 'sde',
            return_sqrt_dt: Optional[bool] = False,

    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        model_output = model_output/self.model.normal_scale[None]
        sample=self.model.normalize(sample)

        if prev_sample is not None:
            prev_sample=self.model.normalize(prev_sample)

        dt = sigma_prev - sigma

        if sde_type == 'sde':
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_prev, sigma))) * noise_level

            # our sde
            prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + model_output * (
                    1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)

                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                    - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == 'cps':
            std_dev_t = sigma_prev * torch.sin(noise_level * math.pi / 2)  # sigma_t in paper
            pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
                sigma_prev ** 2 - std_dev_t ** 2)

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)

                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)/( 2*std_dev_t.clamp_min(0.05) ** 2)

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        prev_sample=self.model.denormalize(prev_sample)

        if return_sqrt_dt:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt)
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels,noise_level):
        v_pred,t_n,x = self._forward_sample(z, t, labels)
        tokenized_agent, initial_map_feature, eval_mask = labels

        if self.use_cluster:
            increasing=tokenized_agent["increasing"]
            non_increasing=~increasing
            z[non_increasing] = z[non_increasing] + (t_next - t_n)[non_increasing] * v_pred[non_increasing]
            z[increasing] = (1-t_next[increasing])*torch.randn_like(x[increasing])+ t_next[increasing] * x[increasing]
        elif self.use_vp:
            dt = t_next - t_n  # negative

            # β(t)
            beta_t = (self.sde._beta_max - self.sde._beta_min) * t_n + self.sde._beta_min

            # α_t
            alpha_t = self.sde.marginal_alpha(t_n)

            # σ_t
            sigma_t = self.sde.marginal_prob_std(t_n)

            score=(alpha_t * x - z) / (sigma_t ** 2 + 1e-8)

            drift = -0.5 * beta_t * z - beta_t * score

            noise = torch.randn_like(x)

            z = z + drift * dt + torch.sqrt(beta_t * (-dt)) * noise
        elif self.use_sde and torch.any(noise_level>0) and "gt_z_raw" not in tokenized_agent.keys():
            z, log_prob, prev_sample_mean, std_dev_t = self.sde_step_with_logprob(
                1-t_n,
                1-t_next,
                -v_pred,
                z,
                noise_level
            )
        else:
            z = z + (t_next - t_n) * v_pred
            log_prob=None#torch.zeros_like(z)

        return z,x,t_n,log_prob


    @torch.no_grad()
    def _forward_sample(self, z, t_n, labels):


        tokenized_agent, initial_map_feature, eval_mask=labels
        num_agents = len(z)

        if self.model.use_return_conditioned:
            tokenized_agent["advantages"]=torch.ones_like(tokenized_agent["nonego_batch"]).to(torch.float32)

        if self.use_cluster:
            t_n=t_n[:,None,None]
        elif self.use_flux:
            t_n=t_n
        else:
            t_n=torch.full((num_agents,1,1), t_n, device=z.device)

        if self.use_scale:
            padding_mask=tokenized_agent["padding_mask"]

            t_n[padding_mask]=0

        x_cond = self.model(z, t_n, tokenized_agent, initial_map_feature)#[...,:z.shape[-1]]

        # conditional
        if self.x_pred:

            if x_cond.shape[-1]!=z.shape[-1]:
                x_cond=x_cond[...,:z.shape[-1]]+torch.randn_like(z)*(x_cond[...,z.shape[-1]:])
            else:
                x_cond=x_cond[...,:z.shape[-1]]

            v_cond = (x_cond- z) / (1.0 - t_n).clamp_min(self.t_eps)
        else:
            v_cond=x_cond

        # if self.model.label_drop_prob>0:
        #     # unconditional
        #     x_uncond = self.model(z, t_n, tokenized_agent, initial_map_feature, num_samples=1, eval_mask=eval_mask,mode=0)
        #     v_uncond = (x_uncond - z) / (1.0 - t_n).clamp_min(self.t_eps)
        #
        #     self.cfg_interval = (0.1, 1.0)
        #     self.cfg_scale=3
        #
        #     # cfg interval
        #     low, high = self.cfg_interval
        #     interval_mask = (t_n < high) & ((low == 0) | (t_n > low))
        #     cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        #
        #     v_cond=v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        return v_cond,t_n,x_cond

    @torch.no_grad()
    def sample(self,tokenized_agent,initial_map_feature,eval_mask,infer_steps=20,num_samples=1,noise_level=None):

        agent_batch = tokenized_agent["nonego_batch"]
        num_graphs = tokenized_agent["num_graphs"]
        nonego_type = tokenized_agent["nonego_type"]
        num_agents = len(agent_batch)

        if self.use_dit:
            lane_batch = initial_map_feature["batch"][::2]
            pos_pl = initial_map_feature["position"][::2]
            orient_pl = initial_map_feature["orientation"][::2]
            feat_map = initial_map_feature["pt_token"][::2]
            initial_map_feature = self.lane_embed1(
                torch.cat([feat_map, pos_pl, orient_pl.cos()[:, None], orient_pl.sin()[:, None]], dim=-1))

            tokenized_agent["lane_batch"] = lane_batch

        z = torch.randn(num_agents, num_samples, 8, device=agent_batch.device)#*0.9 #.clamp(min=-3,max=3)

        t_list=[]

        z=self.model.denormalize(z,nonego_type)

        z_list=[z]
        x_list=[]
        log_prob_list=[]

        if self.use_scale:
            agent_type = tokenized_agent["nonego_type_sorted"]

            type_counts=tokenized_agent["type_counts"]

            num_types=3

            idx = agent_batch * num_types + agent_type

            mask = agent_type >= 0  # or specific valid condition

            cumsum = torch.cumsum(mask.long(), dim=0)

            per_group = torch.bincount(idx[mask], minlength=num_graphs * num_types)

            offsets = torch.cumsum(per_group, dim=0)
            offsets = torch.cat([torch.zeros(1, device=offsets.device).to(torch.long), offsets[:-1]])

            rank = torch.full_like(agent_batch, -1)
            rank[mask] = cumsum[mask] - offsets[idx[mask]]

            counts=type_counts.sum(-1)

            schedule,noise_scedule=batch_increasing_schedule(counts)#[agent_batch]

            steps=schedule.shape[1]-1

        else:
            steps=infer_steps

        if self.mean_flow:
            t = torch.ones(num_agents, device=agent_batch.device)[:,None]
            r = torch.zeros(num_agents, device=agent_batch.device)[:,None]
            beta = torch.cat([t, r], dim=-1)

            z = self.model(z, beta, tokenized_agent, initial_map_feature,eval_mask)

        elif self.use_flow_ode:
            other_model_params = {
                "initial_map_feature": initial_map_feature,
                "tokenized_agent": tokenized_agent,
            }

            z = self.flow_ode.generate(z, self.model, 'x_start', use_cfg=False, **other_model_params)#cfg_weight=1.8,

        elif self.use_dpm_solver:
            noise_schedule = NoiseScheduleVP(
                schedule='linear'
            )

            other_model_params = {
                "initial_map_feature": initial_map_feature,
                "tokenized_agent": tokenized_agent,
            }
            dpm_solver_params = {}
            model_wrapper_params = {}

            model_fn = model_wrapper(
                self.model,  # use your noise prediction model here
                noise_schedule,
                model_type="x_start",  # or "x_start" or "v" or "score"
                model_kwargs=other_model_params,
                **model_wrapper_params
            )
            diffusion_steps=self.steps

            dpm_solver = DPM_Solver(
                model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params) # w.o. dynamic thresholding

            z = dpm_solver.sample(
                z[:,0],
                steps=diffusion_steps,
                order=3,
                skip_type="logSNR",
                method="singlestep_fixed",
                denoise_to_zero=True,
            )[:,None]
        else:
            if self.use_vp:
                timesteps = torch.linspace(self.sde.T, 1e-3, steps + 1, device=agent_batch.device)
            else:
                timesteps=torch.linspace(0,1,steps+1,device=agent_batch.device)

            if self.use_flux:
                count=tokenized_agent["type_counts"].sum(-1)

                mu = calculate_shift( count)[None]

                t_batch=1-timesteps[:,None]

                sigma = mu * t_batch / (1 + (mu - 1) * t_batch)

                timesteps=1-sigma

                timesteps[0]=0

                timesteps=timesteps[:,agent_batch][:,:,None,None]

            noise_level = torch.zeros(num_agents, steps, 1, device=agent_batch.device)

            if self.use_sde:
                t_rand = torch.randint(0, steps, (num_graphs,), device=agent_batch.device)

                t_rand=t_rand[agent_batch]

                noise_level[torch.arange(num_agents), t_rand, 0] = self.noise_level

                noise_mask=noise_level[:,:,0] > 0

                noise_level=noise_level[:,:,None]

            for i in range(steps):# - 1
                t = timesteps[i]
                t_next = timesteps[i + 1]

                if self.use_scale:
                    schedule_i=schedule[:,i]
                    schedule_i1=schedule[:,i+1]

                    k = allocate_k_per_type(schedule_i, type_counts)[agent_batch, agent_type]
                    k1 = allocate_k_per_type(schedule_i1, type_counts)[agent_batch, agent_type]

                    eval_mask = rank <= k1

                    padding_mask = (eval_mask &  (rank> k))[eval_mask]

                    tokenized_agent['padding_mask'] = padding_mask

                    if self.use_cluster:
                        tokenized_agent["increasing"] = (schedule_i != schedule_i1)[agent_batch][eval_mask]

                        t=noise_scedule[:,i][agent_batch][eval_mask]
                        t_next=noise_scedule[:,i+1][agent_batch][eval_mask][:,None,None]

                    z[eval_mask],x_cond,t_n=  self._euler_step(z[eval_mask], t, t_next, (tokenized_agent, initial_map_feature,eval_mask))

                else:
                    z,x_cond,t_n,log_prob =  self._euler_step(z, t, t_next, (tokenized_agent, initial_map_feature,eval_mask),noise_level[:,i])

                x_list.append(x_cond)
                z_list.append(z)
                t_list.append(t_n)
                log_prob_list.append(log_prob)

                #feat_list.append(tokenized_agent["noise_feat"])

        t_list.append(torch.ones_like(t_n))

        if self.use_sde:
           # log_prob_list=torch.stack(log_prob_list,dim=1)
            #log_prob_list=log_prob_list[noise_mask]

            z_list=torch.stack(z_list,dim=1)
            z_list=(z_list[:,:-1][noise_mask],z_list[:,1:][noise_mask],log_prob_list)
            t_list=torch.stack(t_list,dim=1)
            t_list=(t_list[:,:-1][noise_mask],t_list[:,1:][noise_mask])

            #tokenized_agent["noise_feat"]=torch.stack(feat_list,dim=1)[noise_mask]

            tokenized_agent["z_list"]=z_list
            tokenized_agent["t_list"]=t_list
        else:
            tokenized_agent["z_list"]=z

        return z[:, 0], x_list


    def repeat_input(self,tokenized_agent,n_step):
        num_graphs=tokenized_agent["num_graphs"]

        batch = tokenized_agent["nonego_batch"]

        tokenized_agent["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step)  # n_agent ,n_step

        batch = torch.stack(
            [
                batch + num_graphs * t
                for t in range(n_step)
            ],
            dim=1,
        ).transpose(0, 1).flatten(0, 1)  # [n_agent*n_step]

        tokenized_agent["nonego_batch"] = batch

        tokenized_agent["nonego_type"] = tokenized_agent["nonego_type"][None].repeat(n_step, 1).flatten(0,
                                                                                                        1)

        tokenized_agent["num_graphs"] = num_graphs * n_step

        if self.model.use_rel_ego:
            tokenized_agent["ego_feat"] = tokenized_agent["ego_feat"][None].repeat(n_step, 1, 1).flatten(0, 1)
        else:
            tokenized_agent["ego_embedding"] = tokenized_agent["ego_embedding"][None].repeat(n_step, 1,
                                                                                             1).flatten(0, 1)

        return tokenized_agent



def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.99 #9
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)
