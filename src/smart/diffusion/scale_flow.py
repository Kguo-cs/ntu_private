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

from .transformer_decoder import TransformerDecoderLayerDiff,sinusoidal_embedding

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
from src.smart.utils.earth_match import get_closest_sum_idx
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges,project_to_local_frame
from src.smart.layers.relative_transformer import padding
from torch_geometric.nn.pool import knn_graph,knn
# from lpips import LPIPS
from src.smart.utils.cluster import cluster_points
from torch_scatter import scatter_sum


def power_schedule(steps, device, alpha=2.0):
    return  1 - (1 - torch.linspace(0, 1, steps, device=device)) **alpha

def cosine_schedule(steps, device):
    i = torch.arange(steps + 1, device=device)
    return 0.5 * (1 - torch.cos(torch.pi * i / steps))

class ScaleFlow(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

        # self.pos_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        # self.head_embedding = MLPLayer(2, args.hidden_dim, args.hidden_dim)
        self.ego_embedding = MLPLayer(20+3, args.hidden_dim, args.hidden_dim)#+3

       # self.pose_embedding= MLPLayer(128+2+2, args.hidden_dim, args.hidden_dim)
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

        self.use_scale=self.net.use_scale

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

        # if tokenized_agent["step_idx"] is not None:
        #     timesteps=torch.linspace(0,1,tokenized_agent["step_number"]+1,device=eval_mask.device)
        #     t_batch = timesteps[tokenized_agent["step_idx"]]
        #     t_batch=t_batch[:, None]
        # else:
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

            #z=self.net.normalize_z(z)

            if self.x_pred:

                x_pred = self.net(z, t, tokenized_agent, scene_enc, eval_mask)

                denom = (1 - t[:, :, None]).clamp_min(self.t_eps)

                v_target = (x - z) /denom

                v_pred = (x_pred - z) /denom

            else:
                v_target =x - e

                v_pred = self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=mode)

                x_pred =e+v_pred

        return F.mse_loss(v_pred , v_target,reduction="none") ,x_pred[:,0],z,denom[:,0]

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
        num_agents = len(agent_batch)

        tokenized_agent["lengths"] = torch.bincount(agent_batch, minlength=num_scenes).tolist()

        if self.use_all_type:
            z = torch.randn(num_agents,num_samples, 11, device=eval_mask.device)
        else:
            z = torch.randn(num_agents,num_samples, 8, device=eval_mask.device)

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

            veh_per_scene=type_counts.sum(-1)

            schedule=batch_increasing_schedule(veh_per_scene)#[agent_batch]

            steps=schedule.shape[1]-1#max(veh_rank)+1#self.steps#512#

        else:
            steps=self.steps

        x_list=[]
        batch_list=[]
        step_list=[]

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

                    k = allocate_k_per_type(schedule_i, type_counts)[agent_batch, agent_type]
                    k1 = allocate_k_per_type(schedule_i1, type_counts)[agent_batch, agent_type]

                    first_i_veh_mask = rank <= k1#(~veh_mask) | (veh_rank <= schedule_i1)

                    tokenized_agent_scale = {}
                    tokenized_agent_scale["nonego_batch"]=tokenized_agent["nonego_batch"][first_i_veh_mask]
                    tokenized_agent_scale["nonego_type_sorted"]=tokenized_agent["nonego_type_sorted"][first_i_veh_mask]
                    tokenized_agent_scale["num_graphs"]=tokenized_agent["num_graphs"]
                    tokenized_agent_scale["ego_embedding"]=tokenized_agent["ego_embedding"][first_i_veh_mask]

                    agent_batch_scale=agent_batch[first_i_veh_mask]

                    tokenized_agent_scale["lengths"] = torch.bincount(agent_batch_scale, minlength=num_scenes).tolist()

                    padding_mask = (first_i_veh_mask &  (rank> k))[first_i_veh_mask]

                    #padding_mask=(((veh_rank<=schedule_i1) & (veh_rank>schedule_i)) & veh_mask)

                    tokenized_agent_scale["padding_mask"]=padding_mask

                    z_scale=z[first_i_veh_mask]

                    #t_next=torch.clamp_max(t_next+0.1,max=1)
                    #z_scale = self.net.normalize_z(z_scale)

                    z[first_i_veh_mask],x_cond=  self._euler_step(z_scale, t, t_next, (tokenized_agent_scale, scene_enc,eval_mask))

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

        self.use_roformer=True
        self.use_padding=True
        self.use_all_type=False

        if self.use_all_type:
            # self.type_a_emb = MLPLayer(3, hidden_dim, hidden_dim)
            self.output_dim=11
            m_delta_dim = 11

        else:
            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.output_dim=8
            m_delta_dim = 5+3

        self.use_graph=True
        self.ego_rel = True
        self.use_scale=True
        noise_dim = 1
        if mean_flow:
            noise_dim = 2
            self.use_padding = True

        # normal_scale = torch.tensor([[33.699, 28.851,  0.774,  0.622,  1.207,  0.364,  4.970,  0.239]])
        # normal_mean = torch.tensor([[3.609e+00,  1.850e+00,  1.162e-01, -1.249e-05,  4.515e+00,  2.022e+00,
        #  2.568e+00,  9.085e-04]])

        normal_scale = torch.tensor([[33.699, 28.851,  0.760,  0.606,  1.207,  0.364,  4.927,  2.453]])
        normal_mean = torch.tensor([[3.609e+00,  1.850e+00,  1.148e-01, -1.181e-03,  4.515e+00,  2.022e+00,
         6.634e-01, -5.857e-02]]) # ego velocity

        if self.use_all_type:
            normal_scale = torch.tensor([[35.039, 29.354, 0.758, 0.606, 1.317, 0.405, 4.842, 0.281, 0.290,
                                               0.282, 0.071]])

            normal_mean = torch.tensor([[2.345e+00, 3.163e+00, 1.098e-01, 1.229e-02, 4.424e+00, 1.992e+00,
                                              2.421e+00, -3.600e-05, 9.040e-01, 9.043e-02, 5.601e-03]])

        self.register_buffer( "normal_scale",normal_scale )
        self.register_buffer( "normal_mean",normal_mean)

        if self.use_roformer:
            self.noise_embedding = MLPLayer(1, hidden_dim, hidden_dim)
            if self.ego_rel:
                self.proj_in_m_delta = nn.Linear(m_delta_dim-4, self.hidden_dim)#MLPLayer(m_delta_dim-4, hidden_dim, hidden_dim)#
            else:
                self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)#MLPLayer(m_delta_dim, hidden_dim, hidden_dim)#

            self.to_out_m_delta= MLPLayer(hidden_dim, hidden_dim, m_delta_dim)

            if self.use_graph:
                self.use_padding = False

                self.edge_encoder = EdgeEncoder(hidden_dim,
                                                num_freq_bands,
                                                use_a2a=True,
                                                use_pl2a=True
                                                )


                self.ego_encoder = EdgeEncoder(hidden_dim,
                                                num_freq_bands,
                                                use_pl2a=True,
                                                )


                self.pt2a_attn_layers = nn.ModuleList(
                    [
                        AttentionLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=True,
                            has_pos_emb=True,
                        )
                        for _ in range(num_layers)
                    ]
                )

                self.a2a_attn_layers = nn.ModuleList(
                    [
                        AttentionLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=False,
                            has_pos_emb=True,
                        )
                        for _ in range(num_layers)
                    ]
                )

                self.a2ego_attn_layers = nn.ModuleList(
                    [
                        AttentionLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=True,
                            has_pos_emb=True,
                        )
                        for _ in range(num_layers)
                    ]
                )


            else:
                module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
                self.entry_formers = ModuleList([copy.deepcopy(module) for i in range(num_layers)])

        else:

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

            self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)

            # self.interact_pt2m = nn.ModuleList(
            #     [TransformerDecoderLayerDiff(
            #         n_embd=hidden_dim,
            #         n_head=num_heads,
            #         ff_dim=4 * hidden_dim,
            #         dropout=0,
            #         layer_id=i,
            #     ) for i in range(num_layers)])
            module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
            self.interact_pt2m = ModuleList([copy.deepcopy(module) for i in range(num_layers)])
            # self.interact_pt2m = nn.ModuleList([
            #     nn.TransformerDecoderLayer(
            #         d_model=hidden_dim,
            #         nhead=num_heads,
            #         dim_feedforward=4 * hidden_dim,
            #         dropout=0.0,
            #         batch_first=True,
            #         activation="gelu",
            #         norm_first=True
            #     )
            #     for _ in range(num_layers)
            # ])
            self.to_out_m_delta = SkipMLP(d_model=hidden_dim)

        self.apply(weight_init)

    def normalize(self,input):
        return (input - self.normal_mean) / self.normal_scale

    def denormalize(self,input):
        return input* self.normal_scale+self.normal_mean

    def normalize_z(self,z):
        m_delta = z[:, 0]

        m_delta = self.denormalize(m_delta)

        m_delta[:, 2:4] = m_delta[:, 2:4] / torch.linalg.norm(m_delta[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)

        z = self.normalize(m_delta)

        return z[:,None]


    def padding(self, pos, heading, feature, batch, batch_num):
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        mask = torch.any(padding_features_a != 0, dim=-1)

        return padding_pos_a, padding_heading_a, padding_features_a,mask

    def get_original_state(self, pred_init, tokenized_agent, non_ego, batch, ego_position, ego_heading, gt_initial_pos, gt_initial_heading):
        pred_init = self.denormalize(pred_init)

        pred_trans, pred_head, pred_shape, pred_vel = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4:6], \
        pred_init[..., 6:8]
        pred_head = torch.atan2(pred_head[..., 1], pred_head[..., 0])

        global_pos,global_heading=transform_to_global(
            pred_trans,
            pred_head,
            ego_position[batch],
            ego_heading[batch],
        )

        global_pred_vel=rotate_to_global(pred_vel,ego_heading[batch])

        gt_initial_pos[non_ego]=global_pos
        gt_initial_heading[non_ego]=global_heading

        gt_initial_vel=tokenized_agent["initial_vel"].clone()

        gt_initial_vel[non_ego] =global_pred_vel

        shape=tokenized_agent["initial_shape"].clone()

        shape[non_ego,:2]=pred_shape[:,:2]

        tokenized_agent["shape"] = shape

        rel_vel=rotate_to_local(gt_initial_vel,gt_initial_heading)

        center_token_traj = tokenized_agent["token_traj"].mean(-2)

        # gt_initial_idx = torch.linalg.norm(center_token_traj - rel_vel[:, None]*0.5, dim=-1).argmin(-1)

        vel_heading=torch.atan2(rel_vel[:, 1], rel_vel[:, 0])

        pred_pos=transform_to_global(
            center_token_traj,#.flatten(1, 2)
            None,
            - rel_vel*0.5,
            vel_heading,
        )[0].reshape(center_token_traj.shape)

        # static_token=center_token_traj[:,0]
        #
        # gt_initial_idx=torch.linalg.norm(static_token[:,None]-pred_pos,dim=-1).sum(-1).argmin(-1)


        gt_initial_idx = torch.linalg.norm(pred_pos, dim=-1).argmin(-1)


        return gt_initial_pos,gt_initial_heading,shape,gt_initial_vel,gt_initial_idx

    def get_data(self,tokenized_agent,non_ego,nonego_batch,nonego_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading):

        local_vel = rotate_to_local(tokenized_agent["initial_vel"][non_ego],  ego_heading[nonego_batch])

        initial_shape = tokenized_agent["initial_shape"][non_ego]

        non_ego_pos=gt_initial_pos[non_ego]
        non_ego_head=gt_initial_heading[non_ego]

        init_trans, real_heading = transform_to_local(non_ego_pos,
                                                    non_ego_head,
                                                    ego_position[nonego_batch],
                                                    ego_heading[nonego_batch],
                                                    )

        delta_rot = real_heading.unsqueeze(-1)

        init_angle = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)  # [0,2]

        m_init = torch.cat([init_trans, init_angle, initial_shape[:, :2], local_vel], dim=-1)

        tokenized_agent['nonego_type_sorted'] = nonego_type

        if self.use_scale:
            old_nonego_type_sorted = tokenized_agent["nonego_type_sorted"].clone()

            if self.use_all_type:
                one_hot = F.one_hot(old_nonego_type_sorted, num_classes=tokenized_agent["type_counts"].shape[-1])

                m_init = torch.cat([m_init, one_hot], dim=-1)

            diff_input, nonego_batch, m_init, type, step_idx, step_number = cluster_points(m_init, nonego_batch,
                                                                                           old_nonego_type_sorted,
                                                                                           tokenized_agent["type_counts"],
                                                                                           self.use_all_type)

            pad_mask = torch.all(diff_input == 0, dim=-1)

            diff_input[:, 2:4] /= torch.linalg.norm(diff_input[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)
            m_init[:, 2:4] /= torch.linalg.norm(m_init[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)

            m_init = self.normalize(m_init)
            diff_input = self.normalize(diff_input)

            diff_input[pad_mask] = 0

            tokenized_agent['nonego_type_sorted'] = type
            tokenized_agent["step_idx"] = step_idx
            tokenized_agent["step_number"] = step_number
        else:
            diff_input = m_init

        return diff_input,m_init,nonego_batch


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
        num_graphs = tokenized_agent["num_graphs"]
        ego_embedding = tokenized_agent["ego_embedding"]

        if self.use_roformer:
            m_delta=m_delta[:,0]

            m_delta=self.denormalize(m_delta)

            m_delta[:, 2:4] =m_delta[:, 2:4]/ torch.linalg.norm(m_delta[:, 2:4], dim=1, keepdim=True).clamp_min(1e-8)


            if self.ego_rel:
                feat_a=self.proj_in_m_delta(m_delta[:,4:])
            else:
                feat_a=self.proj_in_m_delta(m_delta)

            #beta_emb_m = self.noise_embedding(beta,categorical_embs=self.type_a_emb(type))
            beta_emb_m = self.noise_embedding(beta) +self.type_a_emb(type)#

            feat_a = feat_a + beta_emb_m

            theta=torch.atan2(m_delta[:,3],m_delta[:,2])

            pos_s=m_delta[:, :2]

            if self.use_graph:

                # # number of agents per batch
                # counts = torch.bincount(batch)
                #
                # # first index of each batch
                # first_idx = torch.cumsum(counts, dim=0) - counts
                #
                # ego_tokens = ego_embedding[first_idx]  # (B, A)
                # B = ego_tokens.shape[0]
                # N = feat_a.shape[0]
                #
                # new_feature = torch.cat([feat_a, ego_tokens], dim=0)
                # new_batch = torch.cat([batch, batch[first_idx]], dim=0)
                #
                # ego_theta=torch.atan2(self.normal_mean[:,3],self.normal_mean[:,2])
                #
                # pos_s=torch.cat([pos_s, torch.zeros_like(ego_tokens[:,:2])+self.normal_mean[:,:2]], dim=0)
                # theta=torch.cat([theta, torch.zeros_like(ego_tokens[:,0])+ego_theta], dim=0)
                #
                # order = torch.argsort(new_batch)
                # feat_a = new_feature[order]
                # batch = new_batch[order]
                # theta=theta[order]
                # pos_s=pos_s[order]
                #
                # # mask BEFORE sorting
                # non_ego_mask = torch.cat([
                #     torch.ones(N, dtype=torch.bool, device=batch.device),
                #     torch.zeros(B, dtype=torch.bool, device=batch.device)
                # ], dim=0)
                # non_ego_mask = non_ego_mask[order]

                pos_pl, orient_pl, batch_pl, feat_map=scene_enc

                head_vector_s = torch.stack([theta.cos(), theta.sin()], dim=-1)

                edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                    pos_pl=pos_pl,  # [n_pl, 2]
                    orient_pl=orient_pl,  # [n_pl]
                    pos_a=pos_s,  # [n_agent, n_step, 2]
                    head_a=theta,  # [n_agent, n_step]
                    head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
                    mask=None,  # [n_agent, n_step]
                    batch_s=batch,  # [n_agent,n_step]
                    batch_pl=batch_pl,  # [n_pl*n_step]
                    pl2a_radius=40,
                    max_num_neighbors=20,
                    agent_train_mask=None,
                    layer_num=self.num_layers
                )

                edge_index_a2a, r_a2a, dist, relative_pos, r_a2a_nei, center_nei_pos, center_nei_heading = self.edge_encoder.build_interaction_edge(
                    pos_s=pos_s,  # [n_agent, n_step, 2]
                    head_s=theta,  # [n_agent, n_step]
                    head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
                    batch_s=batch,  # [n_agent*n_step]
                    mask=None,  # [n_agent, n_step]
                    max_radius=60,
                    max_num_neighbors=20,
                    agent_train_mask=None,
                    layer_num=self.num_layers,
                    counter_feat_a=None,
                    dis_edge_mask=None
                )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

                ego_theta = torch.atan2(self.normal_mean[:, 3], self.normal_mean[:, 2]).repeat(num_graphs)
                ego_pos=self.normal_mean[:,:2].repeat(num_graphs,1)

                ego_batch=torch.arange(num_graphs).to(device)

                edge_index_ego2a, r_ego2a = self.ego_encoder.build_map2agent_edge(
                    pos_pl=ego_pos,  # [n_pl, 2]
                    orient_pl=ego_theta,  # [n_pl]
                    pos_a=pos_s,  # [n_agent, n_step, 2]
                    head_a=theta,  # [n_agent, n_step]
                    head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
                    mask=None,  # [n_agent, n_step]
                    batch_s=batch,  # [n_agent,n_step]
                    batch_pl=ego_batch,  # [n_pl*n_step]
                    pl2a_radius=1000,
                    max_num_neighbors=1,
                    agent_train_mask=None,
                    layer_num=self.num_layers
                )

                #
                # rel_pos_a2ego = pos_s-ego_pos
                # rel_head_a2ego = wrap_angle(theta-ego_theta)
                #
                # r_a2ego = torch.cat(
                #     [
                #         project_to_local_frame(rel_pos_a2ego, head_vector_s, False),
                #         rel_head_a2ego[:, None],
                #     ],
                #     dim=-1,
                # )
                #
                # r_a2ego = self.ego_encoder.r_a2a_emb(continuous_inputs=r_a2ego, categorical_embs=None)#+ego_embedding

                # edge_index_a2ego=torch.arange(len(feat_a))

                # feat_a = feat_a + r_a2ego

                for layer_i in range(self.num_layers):

                    feat_a = self.a2ego_attn_layers[layer_i]((ego_embedding, feat_a), r_ego2a, edge_index_ego2a)

                    feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                    feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

            else:
                pos_pl, orient_pl,map_mask, map_emb=scene_enc

                # pos_pl,orient_pl,feat_map,map_mask = self.padding(pos_pl, orient_pl, feat_map, batch_pl, batch_size)  # b, n, d

                pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.padding(pos_s, theta, feat_a, batch,
                                                                        batch_size)  # b, n, d

                pos_emb = sinusoidal_embedding(feat_a_b.shape[1], self.hidden_dim).to(device).unsqueeze(0)

                feat_a_b=feat_a_b+pos_emb

                # pos_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], 2, device=type.device)
                # heading_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], device=type.device)

                for mod in self.entry_formers:
                    feat_a_b = feat_a_b + beta_emb_m

                    feat_a_b = mod(feat_a_b, pos_a_b,
                                   heading_a_b, mask_a_b,
                                   map_emb,
                                   pos_pl,
                                   orient_pl, map_mask
                                   )

                feat_a = feat_a_b[mask_a_b]

            res=self.to_out_m_delta(feat_a)

            res=self.denormalize(res)

            res_theta=torch.atan2(res[:,3],res[:,2])

            global_pos,global_theta = transform_to_global(
                res[:,:2],
                res_theta,
                pos_s,
                theta,
            )

            local_vel=res[:,6:]

            global_vel=rotate_to_global(local_vel,theta)

            res=torch.cat([global_pos,torch.cos(global_theta)[:,None],torch.sin(global_theta)[:,None],res[:,4:6],global_vel], dim=-1)[:,None]
            # cos_d = res[:, 2]
            # sin_d = res[:, 3]
            #
            # cos_t = torch.cos(theta)
            # sin_t = torch.sin(theta)
            #
            # cos_global = cos_t * cos_d - sin_t * sin_d
            # sin_global = sin_t * cos_d + cos_t * sin_d
            #
            # global_pos, _ = transform_to_global(
            #     res[:, :2],
            #     None,
            #     pos_s,
            #     theta,
            # )
            #
            # res = torch.cat([
            #     global_pos,
            #     cos_global[:, None],
            #     sin_global[:, None],
            #     res[:, 4:]
            # ], dim=-1)[:, None]

            res=self.normalize(res)
        else:
            beta_emb = self.noise_emb(beta)
            # num_agents x 128

            m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)

            if self.use_all_type:
                m_delta = m_delta +ego_embedding

            else:
                m_delta = m_delta + self.type_a_emb(type)+ego_embedding
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

            B, N, D = m_delta.shape
            B, N_map, _ = map_emb.shape

            if self.use_padding:
                attn_mask_agent_layers = ~mask_agent
                attn_mask_map_layers = ~map_mask
            else:
                #mask_map_layers = []
                #mask_agent_layers = []

                attn_mask_map_layers = []
                attn_mask_agent_layers = []

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


            # attn_mask_agent_layers1 = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
            # attn_mask_map_layers1 = attn_mask_map_layers.view(B, 1, 1, N_map). \
            #     expand(-1, self.num_heads * 2, N, -1)
            # #
            # # 0: don't attend others
            # if mode == 0:
            #     attn_mask_agent_layers = attn_mask_agent_layers + ~torch.eye(N).to(torch.bool).unsqueeze(0).to(
            #         m_delta.device)

            for i in range(self.num_layers):
                m_delta = m_delta + beta_emb_m

                # m_delta = self.interact_pt2m[i](
                #     tgt=m_delta,
                #     memory=map_emb,
                #     tgt_key_padding_mask=attn_mask_agent_layers,
                #     memory_key_padding_mask=attn_mask_map_layers
                # )

                # m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                #                                 mask=attn_mask_agent_layers1,
                #                                 map_mask=attn_mask_map_layers1)
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
