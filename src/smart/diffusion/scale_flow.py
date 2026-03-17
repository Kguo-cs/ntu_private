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
from src.smart.layers.relative_transformer import RoFormerBlock,RoFormerDecoder
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.func import functional_call, jvp
from src.smart.utils.cluster import batch_increasing_schedule,allocate_k_per_type
from .denoiser import InitDenoiser
from src.smart.diffusion.diffusion_planner.sde import SDE,VPSDE_linear
from src.smart.diffusion.diffusion_planner.dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver


def power_schedule(steps, device, alpha=2.0):
    return  1 - (1 - torch.linspace(0, 1, steps, device=device)) **alpha

def cosine_schedule(steps, device):
    i = torch.arange(steps + 1, device=device)
    return 0.5 * (1 - torch.cos(torch.pi * i / steps))

class ScaleFlow(nn.Module):

    def __init__(self, args,token_processor):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

        # self.pos_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        # self.head_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        self.ego_embedding = MLPLayer(20+3, args.hidden_dim, args.hidden_dim)#+3

        self.mean_flow=False

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
            mean_flow=self.mean_flow
        )

        self.var_sched = VarianceSchedule(
            num_steps=args.num_diffusion_steps,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            mode='linear'
        )
        self.infer_time_per_step = []
        self.GPU_incre_memory = []
        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)

        self.flow_matching=True

        self.x_pred=True

        self.use_scale=self.net.use_scale

        self.use_all_type=self.net.use_all_type

        self.t_eps=5e-2

        self.P_std=1

        self.P_mean=2

        self.steps=100

        self.use_cluster=False

        self.use_vp=False

        self.use_dpm_solver=False

        self.use_flow_ode=False

        if self.use_flow_ode:
            from .flow_planner.flow_ode import FlowODE
            from flow_matching.path.affine import  AffineProbPath
            from flow_matching.path.scheduler import  CondOTScheduler

            from .flow_planner.time_sampler import TimeSampler
            path=AffineProbPath(CondOTScheduler())


            time_sampler=TimeSampler(device='cuda',eps=1e-3,alpha=1.0,beta=1.5)


            self.flow_ode=FlowODE(path,time_sampler,cfg_weight=1.8,sample_steps=self.steps+1,sample_method='midpoint',sample_temperature=1)

        if self.use_vp:
            self.sde = VPSDE_linear()

        self.apply(weight_init)

    def get_loss(self,
                 diff_input,
                 tokenized_agent: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 eval_mask,
                 num_samples=1) :
        if self.flow_matching:
            return self.flow_matching_loss(diff_input, tokenized_agent, scene_enc,eval_mask, num_samples )
        else:
            return self.get_loss_vd(diff_input, tokenized_agent, scene_enc,eval_mask, num_samples)


    def sample_t(self, n: int, device=None):
        # z = torch.randn(n, device=device) * self.P_std + self.P_mean
        # z = torch.sigmoid(z)
        #timesteps = power_schedule(self.steps+1, device, alpha=2)
        #dist = torch.distributions.Beta(0.5, 1)
        #z = dist.sample((n,)).to(device)
        z=torch.rand(n, device=device)#timesteps[torch.randint(low=0,high=self.steps,size=(n,),device=device)] #/self.steps#
        return z

    def flow_matching_loss(self,x, tokenized_agent, scene_enc,eval_mask,num_samples):
        """
        x1: target samples, shape [B, 2]
        """
        device = x.device
        num_scenes = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["nonego_batch"]

        x=x.unsqueeze(1).repeat(1, num_samples, 1)

        e = torch.randn_like(x)  # base distribution N(0, I)

        if "step_idx" in tokenized_agent.keys():
            timesteps=torch.linspace(0,1,tokenized_agent["step_number"]+1,device=device)
            t_batch = timesteps[tokenized_agent["step_idx"]]
            # timesteps1=tokenized_agent["noise_schedule"]
            #
            # t_batch=timesteps1[torch.arange(len(timesteps1)), tokenized_agent["step_idx"]]
            #
            # print(torch.all(t_batch1>t_batch))
            # t_batch=torch.zeros_like(t_batch)
            #
            t_batch = t_batch[:, None, None]
            # if self.use_cluster:
            #     t_batch[tokenized_agent["clustering"]]=0.9
            #     t_batch[~tokenized_agent["clustering"]]=0.9+0.1*t_batch[~tokenized_agent["clustering"]]
        elif self.use_flow_ode:
            t_batch = self.flow_ode.time_sampler.sample(num_scenes).to(device)[:, None,None]
        else:
            t_batch = self.sample_t(num_scenes, device=device)[:, None,None].to(device)  # t ~ U[0,1]

        tokenized_agent["lengths"] = torch.bincount(agent_batch, minlength=num_scenes).tolist()

        if self.mean_flow:

            # r ~ U[0, t]
            r_batch = torch.rand(num_scenes, device=device) [:, None]* t_batch

            t=t_batch[agent_batch]
            r=r_batch[agent_batch]

            # Avoid numerical issues at t=0
            t = torch.clamp(t, min=1e-2)

            z = (1 - t[:,:, None]) * x + t[:,:, None] * e #large t, low noise
            v_target = e - x
            
            params = dict()
            buffers = dict()
            params_and_buffers = {**params, **buffers}

            def net_call(z_arg, r_arg, t_arg, tokenized_agent, scene_enc,eval_mask):
                beta=torch.cat([t_arg,r_arg],dim=-1)
                
                return functional_call(self.net,params_and_buffers, (z_arg, beta, tokenized_agent, scene_enc,eval_mask))

            def u_fn(z_arg, r_arg, t_arg, tokenized_agent, scene_enc,eval_mask):
                x_pred_arg = net_call(z_arg, r_arg, t_arg, tokenized_agent, scene_enc,eval_mask)
                return (z_arg - x_pred_arg) / t_arg[:,:,None]

            v = u_fn(z, t, t, tokenized_agent, scene_enc,eval_mask)
            func = lambda z_, r_, t_: u_fn(z_, r_, t_, tokenized_agent, scene_enc,eval_mask)
            primals = (z, r, t)
            tangents = (v, torch.zeros_like(r), torch.ones_like(t))
            u_out, dudt_out = jvp(func, primals, tangents)

            # V = u + (t - r) * stop_grad(dudt)
            v_pred = u_out + (t[:,:,None] - r[:,:,None]) * dudt_out.detach()

            # Perceptual Loss
            x_init_0_reconstructed = z - t[:,:,None] * u_out
        else:
            if self.steps==1:
                t_batch=torch.zeros_like(t_batch)

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
                z = (1 - t) * e + t * x #large t, low noise

            #z=self.net.normalize_z(z)

            if self.x_pred:

                x_pred = self.net(z, t, tokenized_agent, scene_enc,mode=1)

                denom = (1 - t).clamp_min(self.t_eps)

                v_target = (x - z) /denom

                v_pred = (x_pred - z) /denom

            else:
                v_target =x - e

                v_pred = self.net(z, t, tokenized_agent, scene_enc,mode=1)

                x_pred =e+v_pred

        return F.mse_loss(x_pred , x,reduction="none") ,x_pred[:,0],z,denom[:,0]

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred,t_n,x = self._forward_sample(z, t, labels)

        if self.use_cluster:

            #t_next=torch.zeros_like(x)+t_next
            tokenized_agent, scene_enc, eval_mask = labels
            #z_next=(1 - t_next) * torch.randn_like(x)+ t_next * x
            increasing=tokenized_agent["increasing"]
            non_increasing=~increasing


            z[non_increasing] = z[non_increasing] + (t_next - t_n)[non_increasing] * v_pred[non_increasing]
            z[increasing] = (1-t_next[increasing])*torch.randn_like(x[increasing])+ t_next[increasing] * x[increasing]

            # z[~clustering]=z[~clustering]+ (0.9+0.1*t_next - t_n[~clustering]) * v_pred[~clustering]
            #
            # z[clustering]=0.1*torch.randn_like(x[clustering])+ 0.9 * x[clustering]
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
        else:

            z = z + (t_next - t_n) * v_pred
        return z,x

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next


    @torch.no_grad()
    def _forward_sample(self, z, t_n, labels):

        tokenized_agent, scene_enc, eval_mask=labels
        num_agents = len(z)

        if self.use_cluster:
            t_n=t_n[:,None,None]
        else:
            t_n=torch.full((num_agents,1,1), t_n, device=z.device)

        if self.use_scale:
            # if self.use_cluster:
            #     t_n[tokenized_agent["clustering"]]=0.9
            #     t_n[~tokenized_agent["clustering"]]=0.9+0.1*t_n[~tokenized_agent["clustering"]]

            padding_mask=tokenized_agent["padding_mask"]

            t_n[padding_mask]=0

        # conditional
        x_cond = self.net(z, t_n, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)
        v_cond = (x_cond- z) / (1.0 - t_n).clamp_min(self.t_eps)

        if self.net.label_drop_prob>0:


            # tokenized_agent["nonego_type_sorted"]=torch.full_like(tokenized_agent["nonego_type_sorted"], self.net.num_classes)
            #tokenized_agent["ego_embedding"]=torch.full_like(tokenized_agent["ego_embedding"], 0)

            # unconditional
            x_uncond = self.net(z, t_n, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=0)
            v_uncond = (x_uncond - z) / (1.0 - t_n).clamp_min(self.t_eps)

            self.cfg_interval = (0.1, 1.0)
            self.cfg_scale=3

            # cfg interval
            low, high = self.cfg_interval
            interval_mask = (t_n < high) & ((low == 0) | (t_n > low))
            cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

            v_cond=v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        return v_cond,t_n,x_cond

    @torch.no_grad()
    def sample_flow(self,num_samples,tokenized_agent, scene_enc,    eval_mask):
        agent_batch = tokenized_agent["nonego_batch"]
        num_scenes = tokenized_agent["num_graphs"]
        num_agents = len(agent_batch)

        tokenized_agent["lengths"] = torch.bincount(agent_batch, minlength=num_scenes).tolist()

        z = torch.randn(num_agents, num_samples, self.net.output_dim, device=agent_batch.device)

        if self.use_scale:
            agent_type = tokenized_agent["nonego_type_sorted"]

            type_counts=tokenized_agent["type_counts"]

            num_types=3

            idx = agent_batch * num_types + agent_type

            mask = agent_type >= 0  # or specific valid condition

            cumsum = torch.cumsum(mask.long(), dim=0)

            per_group = torch.bincount(idx[mask], minlength=num_scenes * num_types)

            offsets = torch.cumsum(per_group, dim=0)
            offsets = torch.cat([torch.zeros(1, device=offsets.device).to(torch.long), offsets[:-1]])

            rank = torch.full_like(agent_batch, -1)
            rank[mask] = cumsum[mask] - offsets[idx[mask]]

            counts=type_counts.sum(-1)

            schedule,noise_scedule=batch_increasing_schedule(counts)#[agent_batch]

            steps=schedule.shape[1]-1#max(veh_rank)+1#self.steps#512#

        else:
            steps=20

        x_list=[]
        batch_list=[]
        step_list=[]

        if self.mean_flow:
            t = torch.ones(num_agents, device=agent_batch.device)[:,None]
            r = torch.zeros(num_agents, device=agent_batch.device)[:,None]
            beta = torch.cat([t, r], dim=-1)

            z = self.net(z, beta, tokenized_agent, scene_enc,eval_mask)

        elif self.use_flow_ode:
            other_model_params = {
                "scene_enc": scene_enc,
                "tokenized_agent": tokenized_agent,
            }

            z = self.flow_ode.generate(z, self.net, 'x_start', use_cfg=False, **other_model_params)#cfg_weight=1.8,

        elif self.use_dpm_solver:
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
                model_type="x_start",  # or "x_start" or "v" or "score"
                model_kwargs=other_model_params,
                **model_wrapper_params
            )
            diffusion_steps=self.steps

            dpm_solver = DPM_Solver(
                model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params) # w.o. dynamic thresholding

            # Steps in [10, 20] can generate quite good samples.
            # And steps = 20 can almost converge.
            z = dpm_solver.sample(
                z[:,0],
                steps=diffusion_steps,
                order=3,
                skip_type="logSNR",
                method="singlestep_fixed",
                denoise_to_zero=True,
            )[:,None]
        else:
            #dt = 1.0 / steps
            #timesteps = cosine_schedule(steps+1, z.device)
            if self.use_vp:
                timesteps = torch.linspace(self.sde.T, 1e-3, steps + 1, device=agent_batch.device)
            else:
                timesteps=torch.linspace(0,1,steps+1,device=agent_batch.device)

            #timesteps = power_schedule(steps+1, z.device, alpha=2)
           # ts[0] = 1e-4
           # z[..., 0, 2:] = tokenized_agent["m_init"][..., 2:]
            # ode
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

                    z[eval_mask],x_cond=  self._euler_step(z[eval_mask], t, t_next, (tokenized_agent, scene_enc,eval_mask))

                else:
                    z,x_cond =  self._euler_step(z, t, t_next, (tokenized_agent, scene_enc,eval_mask))

                    #print(z[0])

                #x_list.append(x_cond)
                # batch_list.append(tokenized_agent_scale["nonego_batch"])
                # step_list.append(torch.zeros_like(tokenized_agent_scale["nonego_batch"])+i)

            # last step euler
           # z = self._euler_step(z, timesteps[-2], timesteps[-1], (tokenized_agent, scene_enc,eval_mask))

            # for i in range(steps):
            #        # t = ts[i].expand(z.shape[0],z.shape[1])
            #        # dt = ts[i + 1] - ts[i]
            #
            #     t = torch.full((num_agents,num_samples), i / steps, device=eval_mask.device)
            #     if self.x_pred:
            #         x_pred=self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)
            #
            #         v_pred = (x_pred - z) / (1 - t[:,:, None]).clamp_min(self.t_eps)
            #     else:
            #         v_pred=self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)
            #
            #     z = z + v_pred * dt

                #z[...,0, 2:] = tokenized_agent["m_init"][..., 2:]

        # if self.use_all_type:
        #     tokenized_agent['nonego_type_sorted']=torch.argmax(type_count,dim=-1)

        return z[:,0],x_list,batch_list,step_list

    def get_loss_vd(self,
                    m_init,
                    tokenized_agent: HeteroData,
                    scene_enc: Mapping[str, torch.Tensor],
                    eval_mask,
                    num_samples=1, ) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]

        agent_batch = tokenized_agent["nonego_batch"]
        num_scenes = tokenized_agent["num_graphs"]

        x_init_0 = m_init.unsqueeze(1).repeat(1, num_samples, 1)
        device = m_init.device

        t = torch.tensor(self.var_sched.uniform_sample_t(num_scenes)).to(device)

        alpha_bar = self.var_sched.alpha_bars[t][:, None].to(device)
        beta = self.var_sched.betas[t][:, None].to(device)[agent_batch]
        c0 = torch.sqrt(alpha_bar).unsqueeze(-1).repeat(1, num_samples, 1)
        c1 = torch.sqrt(1 - alpha_bar).unsqueeze(-1).repeat(1, num_samples, 1)

        e_init_rand = torch.randn_like(x_init_0).to(device)

        x_init_t = c0[agent_batch] * x_init_0 + c1[agent_batch] * e_init_rand
        mode = self.B_dist.sample()
        # now delta_rot_pred is angle! add the ego initial angle, then we can get the heading relative to its own
        g_init_theta = self.net(copy.deepcopy(x_init_t), beta, tokenized_agent, scene_enc, num_samples=num_samples, eval_mask=eval_mask,mode=mode)

        loss_init = ((e_init_rand - g_init_theta) ** 2)  # .mean()

      #  x_init_0_reconstructed = (x_init_t - c1[agent_batch] * g_init_theta) / c0[agent_batch]
        return loss_init#, x_init_0_reconstructed

    def sample(self,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               eval_mask,
               num_samples: int,
               start_data=None,
               reverse_steps=None,
               sampling="ddpm",
               stride=20,
               if_output_diffusion_process=False,
               ) -> Dict[str, torch.Tensor]:
        if self.flow_matching:
            return self.sample_flow(num_samples, data, scene_enc,    eval_mask)

        else:
            return self.sample_vd(num_samples, data, scene_enc, if_output_diffusion_process, start_data, reverse_steps,
                                  eval_mask, sampling, stride)

    def sample_vd(self,
                  num_samples: int,
                  data: HeteroData,
                  scene_enc: Mapping[str, torch.Tensor],
                  if_output_diffusion_process=False,
                  start_data=None,
                  reverse_steps=None,
                  eval_mask=None,
                  sampling="ddpm",
                  stride=20,
                  ) -> Dict[str, torch.Tensor]:

        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps

        device = eval_mask.device

        num_agents = eval_mask.sum()

        e_init_rand = torch.randn([num_agents, num_samples, self.net.output_dim]).to(device)

        if start_data == None:
            x_init_T = e_init_rand

        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1 - self.var_sched.alpha_bars[reverse_steps]).to(device)
            x_init_T = c0 * start_data.unsqueeze(1) + c1 * e_init_rand

        x_init_t_list = [x_init_T]
        torch.cuda.empty_cache()

        for t in range(reverse_steps, 0, -stride):

            beta = self.var_sched.betas[t]

            alpha_bar = self.var_sched.alpha_bars[t]

            x_init_t = x_init_t_list[-1]

            with torch.no_grad():
                beta = beta.unsqueeze(-1).repeat(num_agents * num_samples, 1).to(device)
                g_init_theta = self.net(copy.deepcopy(x_init_t), beta, data, scene_enc, num_samples=num_samples,
                                        eval_mask=eval_mask, mode=1)

            if sampling == 'ddpm':

                z_init = torch.randn_like(x_init_T) if t > 1 else torch.zeros_like(x_init_T)

                alpha = self.var_sched.alphas[t]

                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                sigma = self.var_sched.get_sigmas(t, 0)

                x_init_next = c0 * (x_init_t - c1 * g_init_theta) + sigma * z_init

            elif sampling == 'ddim':
                alpha_bar_next = self.var_sched.alpha_bars[t - stride]

                x0_init_t = (x_init_t - g_init_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_init_next = alpha_bar_next.sqrt() * x0_init_t + (1 - alpha_bar_next).sqrt() * g_init_theta

            if True in torch.isnan(x_init_next):
                print('nan:', t)
            x_init_t_list.append(x_init_next.detach())
            if not if_output_diffusion_process:
                x_init_t_list.pop(0)

        if if_output_diffusion_process:
            return x_init_t_list
        else:
            return x_init_t_list[-1]



class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
                    torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas

        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

        # kt
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        kt = 1 - sqrt_alpha_bars  # shifted diffusion
        self.register_buffer('kt', kt)

        inv_sqrt_alpha = 1 / torch.sqrt(alphas)
        co_g = betas / torch.sqrt(1 - alpha_bars)
        co_st = torch.sqrt(alphas[1:]) * (1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])
        co_st = torch.cat([torch.tensor([0]), co_st])
        co_z = torch.sqrt((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:]) * betas[1:])
        co_z = torch.cat([torch.tensor([0]), co_z])
        self.register_buffer('inv_sqrt_alpha', inv_sqrt_alpha)
        self.register_buffer('co_g', co_g)
        self.register_buffer('co_st', co_st)
        self.register_buffer('co_z', co_z)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas