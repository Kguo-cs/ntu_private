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

from src.smart.layers.transformer_decoder import TransformerDecoderLayerDiff,sinusoidal_embedding

from src.smart.layers.fourier_embedding import FourierEmbedding

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    weight_init
)
import warnings
from torch.nn.modules.container import ModuleList
import copy
from src.smart.layers.relative_transformer import RoFormerBlock,RoFormerDecoder
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.func import functional_call, jvp
from src.smart.utils.cluster import batch_increasing_schedule


def power_schedule(steps, device, alpha=2.0):
    return  1 - (1 - torch.linspace(0, 1, steps, device=device)) **alpha

def cosine_schedule(steps, device):
    i = torch.arange(steps + 1, device=device)
    return 0.5 * (1 - torch.cos(torch.pi * i / steps))

class InitDiffusion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

        # self.pos_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        # self.head_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        self.ego_embedding = MLPLayer(20+3, args.hidden_dim, args.hidden_dim)#+3

        self.pose_embedding= MLPLayer(128+2+2, args.hidden_dim, args.hidden_dim)
        self.mean_flow=False

        self.net = InitDenoiser(
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

        self.use_scale=True

        self.use_all_type=self.net.use_all_type

        self.t_eps=5e-2

        self.P_std=1

        self.P_mean=2

        self.steps=512

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

        #timesteps = torch.linspace(0, 1, self.steps + 1, device=device)

        # idx = torch.randint(0, timesteps.shape[0], (n,), device=timesteps.device)
        # z = timesteps[idx]#.repeat(n)
        # timesteps = torch.linspace(0, 1, self.steps + 1, device=eval_mask.device)
        #
        # step=tokenized_agent["step"]
        #
        # t =timesteps[step]
        #
        # t_batch=t.repeat(num_scenes)[:, None]

        z=torch.rand(n, device=device)#timesteps[torch.randint(low=0,high=self.steps,size=(n,),device=device)] #/self.steps#
        return z

    def flow_matching_loss(self,x, tokenized_agent, scene_enc,eval_mask,num_samples):
        """
        x1: target samples, shape [B, 2]
        """
        device = x.device
        num_scenes = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["nonego_batch"]
        mode =1#self.B_dist.sample()

        x=x.unsqueeze(1).repeat(1, num_samples, 1)

        e = torch.randn_like(x)  # base distribution N(0, I)

        if tokenized_agent["step_idx"] is not None:
            timesteps=torch.linspace(0,1,tokenized_agent["step_number"]+1,device=eval_mask.device)
            t_batch = timesteps[tokenized_agent["step_idx"]]

            # t_batch=t_batch+torch.randn(num_scenes, device=device) *0.1
            #
            # t_batch=torch.clamp(t_batch, min=0,max=1)

            t_batch=t_batch[:, None]


        else:
            t_batch = self.sample_t(num_scenes, device=device)[:, None].to(device)  # t ~ U[0,1]
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
                padding_mask =torch.all(x==0,dim=-1)

                t[padding_mask]=0

            z = (1 - t[:,:, None]) * e + t[:,:, None] * x #large t, low noise

            if self.x_pred:
                v_target = (x - z) / (1 - t[:,:, None]).clamp_min(self.t_eps)

                x_pred = self.net(z, t, tokenized_agent, scene_enc, eval_mask) #t=0 ,0.1

                v_pred = (x_pred[:,:,:x.shape[-1]] - z) / (1 - t[:, :, None]).clamp_min(self.t_eps)

                x_init_0_reconstructed = x_pred  # x0+v_pred

            else:
                v_target =x - e

                v_pred = self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=mode)

                x_init_0_reconstructed =e+v_pred

        return F.mse_loss(v_pred , v_target,reduction="none") ,x_init_0_reconstructed[:,0],t_batch,t #t>0.5 #F.l1_loss(x_init_0_reconstructed , x1,reduction="none")

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred,t_n,x_cond = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t_n) * v_pred
        return z_next,x_cond

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):

        tokenized_agent, scene_enc, eval_mask=labels
        num_agents = len(z)

        t_n=torch.full((num_agents,1), t, device=eval_mask.device)

        if self.use_scale:

            padding_mask=tokenized_agent["padding_mask"]

            t_n[padding_mask]=0

        # conditional
        x_cond = self.net(z, t_n, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)
        v_cond = (x_cond[:,:,:z.shape[-1]] - z) / (1.0 - t_n[:,:,None]).clamp_min(self.t_eps)

        return v_cond,t_n[:,:,None],x_cond

    @torch.no_grad()
    def sample_flow(self,num_samples,tokenized_agent, scene_enc,    eval_mask):
        agent_batch = tokenized_agent["nonego_batch"]
        num_scenes = tokenized_agent["num_graphs"]

        tokenized_agent["lengths"] = torch.bincount(agent_batch, minlength=num_scenes).tolist()

        num_agents = eval_mask.sum()

        z = torch.randn(num_agents,num_samples, 8, device=eval_mask.device)

        if self.use_scale:

            agent_type = tokenized_agent["nonego_type_sorted"]

            if self.use_all_type:
                veh_mask=torch.ones_like(agent_type).to(torch.bool)
            else:
                veh_mask = agent_type == 0

            # 1. cumulative vehicle count globally
            veh_cumsum = torch.cumsum(veh_mask.long(), dim=0)

            # 2. total vehicles per scene
            veh_per_scene = torch.bincount(
                agent_batch[veh_mask],
                minlength=num_scenes
            )

            # 3. prefix vehicle offsets per scene
            veh_offsets = torch.cumsum(veh_per_scene, dim=0)
            veh_offsets = torch.cat([
                torch.zeros(1, device=veh_offsets.device, dtype=veh_offsets.dtype),
                veh_offsets[:-1]
            ])

            # 4. vehicle rank inside its own scene
            veh_rank = veh_cumsum - veh_offsets[agent_batch] #- 1

            schedule=batch_increasing_schedule(veh_per_scene)[agent_batch]

            steps=schedule.shape[1]-1#max(veh_rank)+1#self.steps#512#

        else:
            steps=self.steps

        x_list=[]
        batch_list=[]
        step_list=[]

        type_count=torch.zeros((num_agents,3), device=eval_mask.device)

        if self.mean_flow:
            t = torch.ones(num_agents, device=eval_mask.device)[:,None]
            r = torch.zeros(num_agents, device=eval_mask.device)[:,None]
            beta = torch.cat([t, r], dim=-1)

            z = self.net(z, beta, tokenized_agent, scene_enc,eval_mask)

        else:
            #dt = 1.0 / steps
            #timesteps = cosine_schedule(steps+1, z.device)
            timesteps=torch.linspace(0,1,steps+1,device=eval_mask.device)

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

                    # if torch.any(schedule_i1>schedule_i+1):
                    #     print(schedule_i1-schedule_i)

                    first_i_veh_mask = (~veh_mask) | (veh_rank <= schedule_i1)

                    tokenized_agent_scale = {}
                    tokenized_agent_scale["nonego_batch"]=tokenized_agent["nonego_batch"][first_i_veh_mask]

                    if self.use_all_type:
                        tokenized_agent_scale["nonego_type_sorted"]=type_count[first_i_veh_mask]
                    else:
                        tokenized_agent_scale["nonego_type_sorted"]=tokenized_agent["nonego_type_sorted"][first_i_veh_mask]
                    tokenized_agent_scale["num_graphs"]=tokenized_agent["num_graphs"]
                    tokenized_agent_scale["ego_embedding"]=tokenized_agent["ego_embedding"][first_i_veh_mask]

                    agent_batch_scale=agent_batch[first_i_veh_mask]

                    tokenized_agent_scale["lengths"] = torch.bincount(agent_batch_scale, minlength=num_scenes).tolist()

                    padding_mask=(((veh_rank<=schedule_i1) & (veh_rank>schedule_i)) & veh_mask)[first_i_veh_mask]

                    tokenized_agent_scale["padding_mask"]=padding_mask

                    z_scale=z[first_i_veh_mask]

                    res,x_cond=  self._euler_step(z_scale, t, t_next, (tokenized_agent_scale, scene_enc,eval_mask))

                    z[first_i_veh_mask]=res[:,:,:8]
                    if self.use_all_type:
                        type_count[first_i_veh_mask]=torch.relu(x_cond[:,0,8:])
                else:
                    z =  self._euler_step(z, t, t_next, (tokenized_agent, scene_enc,eval_mask))

                x_list.append(x_cond)
                batch_list.append(tokenized_agent_scale["nonego_batch"])
                step_list.append(torch.zeros_like(tokenized_agent_scale["nonego_batch"])+i)

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

        if self.use_all_type:
            tokenized_agent['nonego_type_sorted']=torch.argmax(type_count,dim=-1)

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

        e_init_rand = torch.randn([num_agents, num_samples, 5+3]).to(device)

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



class InitDenoiser(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 init_timestep: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 diff_type: str,
                 m_dim: int,
                 mean_flow
                 ) -> None:
        super(InitDenoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim

        self.use_roformer=False
        self.use_padding=True
        self.use_all_type=False

        if self.use_all_type:
            self.type_a_emb = MLPLayer(3, hidden_dim, hidden_dim)
            self.output_dim=11
        else:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.output_dim=8

        if self.use_roformer:

            module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
            self.entry_formers = ModuleList([copy.deepcopy(module) for i in range(num_layers)])

            self.noise_embedding = MLPLayer(1, hidden_dim, hidden_dim)
            m_delta_dim = 5+3

            self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)

            self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, 2)
            self.head_decoder = MLPLayer(hidden_dim, hidden_dim, 2)
            self.shape_head_decoder = MLPLayer(hidden_dim, hidden_dim, 2)
            self.vel_head_decoder = MLPLayer(hidden_dim, hidden_dim, 2)

            self.normal_scale = torch.tensor([[35.013, 30.234, 0.764, 0.638, 1.326, 0.417, 4.860, 0.230]])
            self.normal_mean = torch.tensor([[2.896e+00, 8.604e-01, 9.726e-02, 9.904e-04, 4.409e+00, 1.989e+00,
                                              2.447e+00, 1.321e-03]])

        else:
            m_delta_dim = 5+3

            self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)

            self.proj_in_m_delta_2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.proj_out_m_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

            noise_dim = 1
            if mean_flow:
                noise_dim=2
                self.use_padding=True
            self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)

            self.interact_pt2m = nn.ModuleList(
                [TransformerDecoderLayerDiff(
                    n_embd=hidden_dim,
                    n_head=num_heads,
                    ff_dim=4 * hidden_dim,
                    dropout=0,
                    layer_id=i,
                ) for i in range(num_layers)])
            module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
            self.interact_pt2m = ModuleList([copy.deepcopy(module) for i in range(num_layers)])

            self.to_out_m_delta = SkipMLP(d_model=hidden_dim)

        self.apply(weight_init)

    def padding(self, pos, heading, feature, batch, batch_num):
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        mask = torch.any(padding_features_a != 0, dim=-1)

        return padding_pos_a, padding_heading_a, padding_features_a,mask

    def forward(self,
                m_delta,
                beta,
                tokenized_agent: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                eval_mask,                
                num_samples=1,
                mode=0
                ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        batch = tokenized_agent["nonego_batch"]
        type = tokenized_agent["nonego_type_sorted"]
        batch_size = tokenized_agent["num_graphs"]
        ego_embedding = tokenized_agent["ego_embedding"]

        if self.use_roformer:
            pos_pl, orient_pl, batch_pl, feat_map=scene_enc
            m_delta=m_delta[:,0]

            m_delta=m_delta*self.normal_scale.to(device)+self.normal_mean.to(device)

            feature = self.noise_embedding(beta) + self.type_a_emb(type) +ego_embedding+self.proj_in_m_delta(m_delta)

            pos_pl,orient_pl,feat_map,map_mask = self.padding(pos_pl, orient_pl, feat_map, batch_pl, batch_size)  # b, n, d

            theta=torch.atan2(m_delta[:,3],m_delta[:,2])

            pos_a_b,heading_a_b,feat_a_b,mask_a_b = self.padding(m_delta[:,:2], theta, feature, batch, batch_size)  # b, n, d

            pos_emb = sinusoidal_embedding(feat_a_b.shape[1], self.hidden_dim).to(device).unsqueeze(0)

            feat_a_b=feat_a_b+pos_emb

            # pos_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], 2, device=type.device)
            # heading_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], device=type.device)

            for mod in self.entry_formers:
                feat_a_b = mod(feat_a_b, pos_a_b,
                               heading_a_b, mask_a_b,
                               feat_map,
                               pos_pl,
                               orient_pl, map_mask
                               )

            attr_feature = feat_a_b[mask_a_b]

            pos = self.pos_decoder(attr_feature)  # * 80

            heading = self.head_decoder(attr_feature)

            shape = self.shape_head_decoder(attr_feature)

            vel = self.vel_head_decoder(attr_feature)

            res = torch.cat([pos, heading, shape,vel], dim=1)[:,None]

        else:
            beta_emb = self.noise_emb(beta)
            # num_agents x 128
            categorical_embs_m = [
                self.type_a_emb(type),
            ]

            m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)
            m_delta = m_delta + categorical_embs_m[0]+ego_embedding
            m_delta = self.proj_in_m_delta_2(m_delta)

            self.num_samples = num_samples

            if self.use_padding:
                pos_pl, orient_pl,map_mask, map_emb=scene_enc

                lengths=tokenized_agent["lengths"]

                #lengths = torch.bincount(batch, minlength=batch_size).tolist()

                m_delta = padding(m_delta, lengths, padding_value=0)  # b, n, d

                mask_agent = torch.any(m_delta != 0, dim=-1)

                beta_emb_m= padding(beta_emb, lengths, padding_value=0)

            else:
                pos_pl, orient_pl, batch_pl, feat_map=scene_enc

                x_pt = feat_map#.repeat(self.num_samples, 1)
                map_batch_list = batch_pl

                poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
                map_emb_batch = torch.split(x_pt, poly_cnt_per_batch.tolist())

                map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0)

                agent_cnt_per_batch = batch.bincount(minlength=batch_size)
                agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())

                m_delta = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)

                beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())

                beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0)

            pos_emb = sinusoidal_embedding(m_delta.shape[1], self.hidden_dim).to(device).unsqueeze(0)
            m_delta += pos_emb

            if self.use_padding:
                attn_mask_agent_layers = ~mask_agent
                attn_mask_map_layers = ~map_mask
            else:
                #mask_map_layers = []
                #mask_agent_layers = []

                attn_mask_map_layers = []
                attn_mask_agent_layers = []

                B, N, D = m_delta.shape
                B, N_map, _ = map_emb.shape

                for i in range(batch_size):
                   # mask_attn_map_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    #mask_attn_map_agent_i = mask_attn_map_agent_i.unsqueeze(-1).expand(-1, N_map)
                    mask_attn_map_pt_i = torch.arange(N_map).to(m_delta.device) < poly_cnt_per_batch[i]
                    attn_mask_map_layers.append(mask_attn_map_pt_i)
                    #mask_attn_map_pt_i = mask_attn_map_pt_i.unsqueeze(0).expand(N, -1)

                    #mask_attn_i = mask_attn_map_agent_i & mask_attn_map_pt_i
                    # mask_map_layers.append(mask_attn_i)

                    mask_attn_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    attn_mask_agent_layers.append(mask_attn_agent_i)
                    # mask_attn_agent_i = mask_attn_agent_i.unsqueeze(-1).expand(-1, N)
                    # mask_attn_i = mask_attn_agent_i & mask_attn_agent_i.t()
                    # mask_agent_layers.append(mask_attn_i)

                attn_mask_agent_layers = ~torch.stack(attn_mask_agent_layers)
                attn_mask_map_layers = ~torch.stack(attn_mask_map_layers)
                mask_agent = torch.arange(N).expand(B, N).to(m_delta.device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
                #mask_agent = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]


            # attn_mask_agent_layers = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
            # attn_mask_map_layers = attn_mask_map_layers.view(B, 1, 1, N_map). \
            #     expand(-1, self.num_heads * 2, N, -1)
            #
            # # 0: don't attend others
            # if mode == 0:
            #     attn_mask_agent_layers = attn_mask_agent_layers + ~torch.eye(N).to(torch.bool).unsqueeze(0).to(
            #         m_delta.device)

            for i in range(self.num_layers):
                m_delta = m_delta + beta_emb_m
                # m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                #                                 mask=attn_mask_agent_layers,
                #                                 map_mask=attn_mask_map_layers)
                m_delta = self.interact_pt2m[i](m_delta, torch.zeros_like(m_delta[:,:,:2]),
                               torch.zeros_like(m_delta[:,:,0]), ~attn_mask_agent_layers,
                               map_emb,
                               torch.zeros_like(map_emb[:,:,:2]),
                               torch.zeros_like(map_emb[:,:,0]), ~attn_mask_map_layers
                               )

            m_out_delta = m_delta[mask_agent]#.view(-1, D)  # [sum(agent_cnt_per_batch), D]

            out_m_delta = self.to_out_m_delta(m_out_delta)
            out_m_delta = out_m_delta.view(-1, self.num_samples, self.hidden_dim)
            res=self.proj_out_m_delta(out_m_delta)
        return res


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


class SkipMLP(torch.nn.Module):
    def __init__(self, d_model=128, act_layer=nn.GELU):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.ac = act_layer()
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x):
        out = x + self.ac(self.linear(x))
        out = self.norm2(x + self.norm1(out))
        return out
