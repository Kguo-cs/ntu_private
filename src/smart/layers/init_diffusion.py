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

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def power_schedule(steps, device, alpha=2.0):
    t = torch.linspace(0., 1., steps + 1, device=device)
    return t ** alpha

def cosine_schedule(steps, device):
    i = torch.arange(steps + 1, device=device)
    return 0.5 * (1 - torch.cos(torch.pi * i / steps))

class InitDiffusion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

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
            m_dim=args.m_dim
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

        self.t_eps=5e-2

        self.P_std=1

        self.P_mean=2
        self.apply(weight_init)


        #increase mean


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
       # z=torch.sigmoid(z)
        z=torch.rand(n, device=device)
        return z

    def flow_matching_loss(self,x1, tokenized_agent, scene_enc,eval_mask,num_samples):
        """
        x1: target samples, shape [B, 2]
        """
        device = x1.device
        num_scenes = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["batch"][eval_mask]
        mode = self.B_dist.sample()

        x1=x1.unsqueeze(1).repeat(1, num_samples, 1)

        x0 = torch.randn_like(x1)  # base distribution N(0, I)

        t = self.sample_t(num_scenes, device=device)[:, None].to(device)[agent_batch]  # t ~ U[0,1]

        z = (1 - t[:,:, None]) * x0 + t[:,:, None] * x1 #large t, low noise

        if self.x_pred:
            v_target = (x1 - z) / (1 - t[:,:, None]).clamp_min(self.t_eps)

            x_pred = self.net(copy.deepcopy(z), t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,
                              mode=mode)

            v_pred = (x_pred - z) / (1 - t[:, :, None]).clamp_min(self.t_eps)

           # x_init_0_reconstructed = x_pred  # x0+v_pred

        else:
            v_target =x1 - x0

            v_pred = self.net(copy.deepcopy(z), t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=mode)

           # x_init_0_reconstructed =x0+v_pred

        return ((v_pred - v_target) ** 2) #,x_init_0_reconstructed

    @torch.no_grad()
    def sample_flow(self,num_samples,tokenized_agent, scene_enc,    eval_mask, steps=50, device="cuda"):

        num_agents = eval_mask.sum()

        z = torch.randn(num_agents,num_samples, 8, device=device)
        #dt = 1.0 / steps
        #ts = cosine_schedule(steps, z.device)
        ts=torch.linspace(0,1,steps,device=device)

        #ts = power_schedule(steps, z.device, alpha=2)
        ts[0] = 1e-4

        for i in range(steps):
            t = ts[i].expand(z.shape[0],z.shape[1])
            dt = ts[i + 1] - ts[i]

            # t = torch.full((num_agents,num_samples), i / steps, device=device)
            if self.x_pred:
                x_pred=self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)

                v_pred = (x_pred - z) / (1 - t[:,:, None]).clamp_min(self.t_eps)
            else:
                v_pred=self.net(z, t, tokenized_agent, scene_enc, num_samples=1, eval_mask=eval_mask,mode=1)

            z = z + v_pred * dt

        return z

    def get_loss_vd(self,
                    m_init,
                    tokenized_agent: HeteroData,
                    scene_enc: Mapping[str, torch.Tensor],
                    eval_mask,
                    num_samples=1, ) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]

        agent_batch = tokenized_agent["batch"][eval_mask]
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
               grad_guid=None,
               cond_gen=None,
               guid_param=None,
               uc=None,
               clean_data=None,
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
                 m_dim: int) -> None:
        super(InitDenoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim
        self.type_a_emb = nn.Embedding(3, hidden_dim)

        self.use_roformer=False

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
                nn.Linear(self.hidden_dim, m_delta_dim),
            )

            noise_dim = 1
            self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)

            self.interact_pt2m = nn.ModuleList(
                [TransformerDecoderLayerDiff(
                    n_embd=hidden_dim,
                    n_head=num_heads,
                    ff_dim=4 * hidden_dim,
                    dropout=dropout,
                    layer_id=i,
                ) for i in range(num_layers)])

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
                num_samples: int,
                eval_mask=None,
                mode=0
                ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        batch = tokenized_agent["batch"][eval_mask]
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
            #heading_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], device=type.device)

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

            self.num_samples = num_samples

            pos_pl, orient_pl, batch_pl, feat_map=scene_enc

            x_pt = feat_map.repeat(self.num_samples, 1)
            map_batch_list = batch_pl

            poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
            map_emb_batch = torch.split(x_pt, poly_cnt_per_batch.tolist())

            map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0)

            beta_emb = self.noise_emb(beta)

            # num_agents x 128
            categorical_embs_m = [
                self.type_a_emb(type),
            ]

            m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)
            m_delta = m_delta + categorical_embs_m[0]+ego_embedding
            m_delta = self.proj_in_m_delta_2(m_delta)

            agent_cnt_per_batch = batch.bincount(minlength=batch_size)
            agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())
            m_delta = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)
            pos_emb = sinusoidal_embedding(m_delta.shape[1], self.hidden_dim).to(device).unsqueeze(0)
            m_delta += pos_emb

            beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())
            beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0)

            mask_map_layers = []
            mask_agent_layers = []

            attn_mask_map_layers = []
            attn_mask_agent_layers = []

            B, N, D = m_delta.shape
            B, N_map, _ = map_emb.shape

            for i in range(batch_size):
                mask_attn_map_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                mask_attn_map_agent_i = mask_attn_map_agent_i.unsqueeze(-1).expand(-1, N_map)
                mask_attn_map_pt_i = torch.arange(N_map).to(m_delta.device) < poly_cnt_per_batch[i]
                attn_mask_map_layers.append(mask_attn_map_pt_i)
                mask_attn_map_pt_i = mask_attn_map_pt_i.unsqueeze(0).expand(N, -1)

                mask_attn_i = mask_attn_map_agent_i & mask_attn_map_pt_i
                mask_map_layers.append(mask_attn_i)

                mask_attn_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                attn_mask_agent_layers.append(mask_attn_agent_i)
                mask_attn_agent_i = mask_attn_agent_i.unsqueeze(-1).expand(-1, N)
                mask_attn_i = mask_attn_agent_i & mask_attn_agent_i.t()
                mask_agent_layers.append(mask_attn_i)

            attn_mask_agent_layers = ~torch.stack(attn_mask_agent_layers)
            attn_mask_map_layers = ~torch.stack(attn_mask_map_layers)

            attn_mask_agent_layers = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
            attn_mask_map_layers = attn_mask_map_layers.view(B, 1, 1, N_map). \
                expand(-1, self.num_heads * 2, N, -1)

            # 0: don't attend others
            if mode == 0:
                attn_mask_agent_layers = attn_mask_agent_layers + ~torch.eye(N).to(torch.bool).unsqueeze(0).to(
                    m_delta.device)

            for i in range(self.num_layers):
                m_delta = m_delta + beta_emb_m
                m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                                                mask=attn_mask_agent_layers,
                                                map_mask=attn_mask_map_layers)

            mask = torch.arange(N).expand(B, N).to(m_delta.device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
            mask_agent = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
            m_out_delta = m_delta[mask_agent].view(-1, D)  # [sum(agent_cnt_per_batch), D]

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
