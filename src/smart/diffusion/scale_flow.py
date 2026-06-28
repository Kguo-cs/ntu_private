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
from src.smart.diffusion.diffusion_utils import get_diff_loss,get_closest_sum_idx,get_type_position_index,sort_agents_by_xy_keep_last
from src.smart.diffusion.dit.dit import DiT
import torch

from torch_scatter import scatter_mean


class ScaleFlow(nn.Module):

    def __init__(self, args, token_processor):
        super().__init__()
        self.diff_type = args.diff_type
        self.guid_sampling = args.guid_sampling

        self.hidden_dim = args.hidden_dim

        self.use_dit = False

        if self.use_dit:
            self.x_pred = False
        else:
            self.x_pred = True

        if self.use_dit:
            self.model = DiT(self.hidden_dim)
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
                mean_flow=False,
                x_pred=self.x_pred
            )

        if not self.model.use_rel_ego:
            self.ego_embedding1 = MLPLayer(16 + 3, args.hidden_dim, args.hidden_dim)

        self.infer_time_per_step = []
        self.GPU_incre_memory = []
        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)

        self.use_scale = self.model.use_scale

        self.use_all_type = self.model.use_all_type

        if self.x_pred:
            self.t_eps = 0.05
        else:
            self.t_eps = 0

        self.lognorm_t = True

        self.P_std = 2  # 1#

        self.P_mean = 1  # 2#

        self.use_cluster = False

        self.use_sde = False

        self.noise_level = 0.7

        self.rationorm = False

        self.global_step = 0

        self.use_nft = False

        self.use_uniform = False

        self.learn_noise = False

        # self.info_sampler=InfoNoiseSampler()

        self.mc_num = 1

        self.init_adv_clip = 3.0
        self.init_logprob_clip = 50.0
        self.init_ppo_clip = 0.2
        self.use_init_ppo_ratio = True

        if self.use_nft:
            self.old_model = copy.deepcopy(self.model)
            self.use_sde = False

        if self.learn_noise:
            self.noise_model = DiT(256)

        self.pred_all_pos = token_processor.pred_all_pos

        if self.pred_all_pos:
            self.pred_model = InitDenoiser(
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
                mean_flow=False,
                x_pred=self.x_pred,
                learn_noise=self.learn_noise,
                pred_all_pos=True
            )

        self.use_ref = True

        if self.use_ref:
            self.ref_model = copy.deepcopy(self.model)

        num_bins=20

        self.bin_loss_sum = torch.zeros(num_bins)
        self.bin_count = torch.zeros(num_bins)

        self.apply(weight_init)


    def _sanitize_init_advantages(self, advantages, ego_mask, selected_agent_idx=None):
        """Return finite, clipped advantages aligned to sampled SDE transitions."""
        if advantages is None:
            return None
        advantages = advantages.detach().to(dtype=torch.float32)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        if selected_agent_idx is not None:
            if advantages.shape[0] != ego_mask.shape[0]:
                raise RuntimeError(
                    "learn_init advantages must be all-agent length before indexing: "
                    f"advantages={tuple(advantages.shape)}, ego_mask={tuple(ego_mask.shape)}"
                )
            advantages = advantages[selected_agent_idx]

        if advantages.numel() > 1:
            advantages = advantages - advantages.mean()
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

        return advantages.clamp(-self.init_adv_clip, self.init_adv_clip)

    def get_loss(self,
                 x,
                 out,
                 tokenized_agent: HeteroData,
                 initial_map_feature: Mapping[str, torch.Tensor],
                 num_samples=1):
        device = x.device
        num_graphs = tokenized_agent["num_graphs"]
        agent_batch = tokenized_agent["init_agent_batch"]
        init_agent_type = tokenized_agent["init_agent_type"]

        x = x.unsqueeze(1).repeat(1, num_samples, 1)

        if self.use_uniform:
            e = torch.rand_like(x) * 2 - 1  # base distribution N(0, I)
        else:
            e = torch.randn_like(x)

        e = self.model.denormalize(e, init_agent_type)

        if self.learn_noise:
            t = torch.zeros((len(agent_batch), 1, self.model.m_delta_dim), device=x.device, dtype=torch.float32)

            x_pred_noise = self.noise_model(torch.zeros_like(e), t, tokenized_agent, initial_map_feature)

            fake_idx = get_closest_sum_idx(x_pred_noise[:, 0], x[:, 0], tokenized_agent)

            x_pred_noise = x_pred_noise[fake_idx]

            tokenized_agent["x_pred_noise"] = x_pred_noise.detach()

            policy_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
                tokenized_agent,
                x_pred_noise[:, 0],
                x[:, 0],
                e[:, 0],
                e[:, 0],
                t[:, 0],
                use_col=False,
                x_pred=True
            )

            std = torch.clamp(x_pred_noise[:, :, 8:].exp(), max=50, min=1e-5).detach()

            std[:, :, 2:6] = std[:, :, 2:6] * 2
            std[:, :, 4:6] = std[:, :, 4:6] * 2

            tokenized_agent["noise_std"] = std

            e = std * torch.randn_like(x) + x_pred_noise[:, :, :8]  #

            e = e.detach()

            # tokenized_agent["x_pred_noise"]=x_pred_noise.detach()
        else:
            policy_loss = 0

        if "step_idx" in tokenized_agent.keys():
            timesteps = torch.linspace(0, 1, tokenized_agent["step_number"] + 1, device=device)
            t_batch = timesteps[tokenized_agent["step_idx"]]
            t = t_batch[:, None, None]
            t_dt = torch.ones_like(t)
        else:
            if self.lognorm_t:

                # base_t = (torch.randn((num_graphs,1), device=x.device, dtype=torch.float32)*self.P_std+self.P_mean).sigmoid()#.repeat(1,8)
                # t, bin_idx = self.info_sampler.sample(batch_size=num_graphs)

                # base_t = t[:,None]

                base_t = torch.rand((num_graphs, 1, 1), device=x.device, dtype=torch.float32)

                base_t = base_t[agent_batch]

                t, t_dt = self.model.schedule(base_t, x, tokenized_agent)

                t_dt = torch.ones_like(t)

                # policy_loss=self.model.schedule.regularization(t_dt)#
            else:
                base_t = torch.rand((num_graphs, 1, 1), device=x.device, dtype=torch.float32).repeat(1, 1,
                                                                                                     self.model.m_delta_dim)

                t = base_t[agent_batch]

        ego_mask = tokenized_agent["ego_mask"]

        t = torch.where(
            ego_mask[:, None, None],
            torch.ones_like(t),
            t,
        )

        if self.use_scale:
            nan_mask = torch.isnan(x)

            padding_mask = torch.all(nan_mask, dim=-1)

            t[padding_mask] = 0

            x[nan_mask] = 0

        z = (1 - t) * e + t * x  # large t, low noise        target velocity e-x = (z-x)/(1-t)

        if self.model.use_cfg_cond:
            tokenized_agent["cfg"] = torch.ones(num_graphs,
                                                device=agent_batch.device) * 2  # sample_cfg_scale(num_graphs,device=z.device)#t

        if "advantages" in tokenized_agent.keys():
            raw_advantages = tokenized_agent["advantages"]

            if self.use_sde:

                z_sampled, prev_sample, old_log_prob = tokenized_agent["gen_z"]
                t_n_sampled, t_next_sampled = tokenized_agent["gen_t"]
                selected_agent_idx = tokenized_agent.get(
                    "gen_agent_idx",
                    torch.arange(z_sampled.shape[0], device=z_sampled.device),
                )
                advantages = self._sanitize_init_advantages(
                    raw_advantages,
                    ego_mask,
                    selected_agent_idx=selected_agent_idx,
                )
            else:
                advantages = raw_advantages
                gen_z=tokenized_agent["gen_z"]
                gen_z[:, :, 2:4] = gen_z[:, :, 2:4] / gen_z[:, :, 2:4].norm(dim=-1, keepdim=True)

                x_sampled = gen_z[None].repeat(self.mc_num, 1, 1, 1).flatten(0, 1)
                e_sampled = torch.randn_like(e[None].repeat(self.mc_num, 1, 1, 1).flatten(0, 1))

                agent_batch = torch.stack(
                    [
                        agent_batch + num_graphs * t0
                        for t0 in range(self.mc_num)
                    ],
                    dim=1,
                ).transpose(0, 1).flatten(0, 1)  # [n_agent*n_step]

                sampled_base_t = torch.rand_like(base_t)[agent_batch]
                t_n_sampled, _ = self.model.schedule(
                    sampled_base_t, x_sampled, tokenized_agent
                )

                t_n_sampled = torch.where(
                    ego_mask[:, None, None],
                    torch.ones_like(t_n_sampled),
                    t_n_sampled,
                )

                advantages = advantages[None].repeat(self.mc_num, 1).flatten(0, 1)

                z_sampled = (1 - t_n_sampled) * e_sampled + t_n_sampled * x_sampled

            denom = (1.0 - t_n_sampled).clamp_min(self.t_eps)

            with torch.no_grad():
                if self.use_ref:
                    if self.global_step == 0:
                        decay = 0
                        for src_param, tgt_param in zip(self.model.parameters(), self.ref_model.parameters(),
                                                        strict=True):
                            tgt_param.data.copy_(
                                tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

                        self.ref_model.eval()
                    ref_prediction = self.ref_model(z_sampled, t_n_sampled, tokenized_agent, initial_map_feature)

                if self.use_nft:
                    decay = return_decay(self.global_step, 2)
                    for src_param, tgt_param in zip(self.model.parameters(), self.old_model.parameters(), strict=True):
                        tgt_param.data.copy_(
                            tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

                    self.old_model.eval()
                    old_prediction = self.old_model(z_sampled, t_n_sampled, tokenized_agent, initial_map_feature)
                    if self.x_pred:
                        old_v_pred = (old_prediction - z_sampled) / denom
                    else:
                        old_v_pred = old_prediction

                self.global_step += 1

            t_all = t_n_sampled  # torch.cat((t_n_sampled, t), dim=0)
            z_all = z_sampled  # torch.cat((z_sampled, z), dim=0)
            model_tokenized_agent = tokenized_agent

            # model_tokenized_agent = self.repeat_input_copy(
            #     tokenized_agent,
            #     self.mc_num + 1,
            # )

            x_pred_all = self.model(
                z_all,
                t_all,
                model_tokenized_agent,
                initial_map_feature,
            )

            if self.x_pred:
                v_pred = (x_pred_all[:len(z_sampled)] - z_sampled) / denom
            else:
                v_pred = x_pred_all[:len(z_sampled)]

            adv_clip_max = 3

            if self.use_nft:
                beta = 1
                forward_prediction = v_pred  # v=(x0-z)/(1-t)     z=(1-t)*e+t*x0
                x0 = x_sampled
                xt = z_sampled
                t_expanded = -denom  # x0_pred=    (1-t) *v+ z
                old_prediction = old_v_pred

                normalized_advantages_clip = (
                                                         advantages / adv_clip_max) / 2.0 + 0.5  # (advantages_clip-advantages_clip.min())/(advantages_clip.max()-advantages_clip.min())#
                r = torch.clamp(normalized_advantages_clip, 0, 1)
                # positive_prediction = beta * forward_prediction + (1 - beta) * old_prediction.detach()
                implicit_negative_prediction = (
                                                       1.0 + beta
                                               ) * old_prediction.detach() - beta * forward_prediction
                x0_prediction = x_pred_all[:len(z_sampled)]
                positive_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
                    tokenized_agent,
                    x0_prediction[:, 0],
                    x_sampled[:, 0],
                    z_sampled[:, 0],
                    e_sampled[:, 0],
                    t_n_sampled[:, 0],
                    use_col=False,
                    x_pred=self.x_pred
                )
                negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                negative_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
                    tokenized_agent,
                    negative_x0_prediction[:, 0],
                    x_sampled[:, 0],
                    z_sampled[:, 0],
                    e_sampled[:, 0],
                    t_n_sampled[:, 0],
                    use_col=False,
                    x_pred=self.x_pred
                )
                #
                ori_policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
                policy_loss = (ori_policy_loss * adv_clip_max).mean() * 0.2
            elif self.use_sde:
                sampled_non_ego = ~ego_mask[selected_agent_idx]
                if sampled_non_ego.sum() == 0:
                    policy_loss = z_sampled.new_zeros(())
                else:
                    prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sde_step_with_logprob(
                        1 - t_n_sampled[sampled_non_ego],
                        1 - t_next_sampled[sampled_non_ego],
                        -v_pred[sampled_non_ego],
                        z_sampled[sampled_non_ego],
                        noise_level=self.noise_level,
                        prev_sample=prev_sample[sampled_non_ego]
                    )
                    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
                    log_prob = log_prob.clamp(-self.init_logprob_clip, self.init_logprob_clip)
                    advantages_pg = advantages[sampled_non_ego]
                    if self.use_init_ppo_ratio and old_log_prob is not None:
                        old_lp = old_log_prob[sampled_non_ego].detach()
                        old_lp = torch.nan_to_num(old_lp, nan=0.0, posinf=0.0, neginf=0.0)
                        old_lp = old_lp.clamp(-self.init_logprob_clip, self.init_logprob_clip)
                        ratio = torch.exp((log_prob - old_lp).clamp(-10.0, 10.0))
                        clipped_ratio = ratio.clamp(
                            1.0 - self.init_ppo_clip,
                            1.0 + self.init_ppo_clip,
                        )
                        surrogate_1 = ratio * advantages_pg
                        surrogate_2 = clipped_ratio * advantages_pg
                        policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
                    else:
                        policy_loss = -(log_prob * advantages_pg).mean()
            else:
                x_pred = x_pred_all[:len(z_sampled)]

                denom = (1 - t_n_sampled[:, 0]).clamp_min(self.t_eps)
                denom_sq = denom.square()

                inv_denom_sq = denom_sq.reciprocal()
                mse_Loss=F.mse_loss(x_pred[:, 0] , x_sampled[:, 0] , reduction="none")
                l1_Loss=F.l1_loss(x_pred[:, 0] , x_sampled[:, 0] , reduction="none").mean(-1, keepdim=True).clip(min=0.00001).detach()

                sampled_match_loss=(mse_Loss/l1_Loss).mean(-1)
                #sampled_match_loss=sampled_match_loss*0.1

                non_ego = ~ego_mask
                advantages_pg = self._sanitize_init_advantages(
                    advantages,
                    ego_mask,
                    selected_agent_idx=None,
                )[non_ego]
                #advantages_pg = torch.exp(advantages_pg).detach() #.clamp_max(5)

                #advantages_pg=advantages_pg.clamp_min(0.0)

                logp_cur = -sampled_match_loss[non_ego]

                tokenized_agent["sampled_match_loss"]=sampled_match_loss

                # policy_loss=(-advantages_pg*logp_cur).mean()#.exp()
                #
                if self.use_ref:
                    mse_Loss = F.mse_loss(ref_prediction[:, 0], x_sampled[:, 0], reduction="none")
                    l1_Loss = F.l1_loss(ref_prediction[:, 0], x_sampled[:, 0], reduction="none").mean(-1, keepdim=True).clip(min=0.00001).detach()

                    sampled_match_loss = (mse_Loss /l1_Loss).mean(-1)

                    logp_old=-sampled_match_loss[non_ego]
                else:
                    logp_old = logp_cur.detach()

                log_ratio = logp_cur - logp_old
                ratio = torch.exp(log_ratio)

                clip_eps = 0.2
                ratio_clip = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)

                surrogate_1 = ratio * advantages_pg.detach()
                surrogate_2 = ratio_clip * advantages_pg.detach()

                policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean() #* 0.01

                tokenized_agent["policy_loss"]=policy_loss
                tokenized_agent["ratio"]=ratio

            #x_pred = x_pred_all[len(z_sampled):]

            x_pred = self.model(z, t, tokenized_agent, initial_map_feature)


        else:
            x_pred = self.model(z, t, tokenized_agent, initial_map_feature)

        if self.use_scale:
            x = out[:, None]

        if not self.model.pred_gmm:
            ego_mask_expand = ego_mask.view(
                ego_mask.shape[0],
                *([1] * (x_pred.ndim - 1)),
            )

            x_pred = torch.where(
                ego_mask_expand,
                x.detach(),
                x_pred,
            )

        match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
            tokenized_agent,
            x_pred[:, 0],
            x[:, 0],
            z[:, 0],
            e[:, 0],
            t[:, 0],
            base_t=base_t[:,0],
            t_dt=t_dt[:,0],
            t_eps=self.t_eps,
            use_col=True,  # not self.model.pred_gmm,
            x_pred=self.x_pred,
        )

        #self.debug_loss_vs_timestep(base_t[:,0,0],match_loss,ego_mask)

        if self.model.schedule_loss:
            with torch.no_grad():
                e = torch.randn_like(x)  # .clamp(min=-3,max=3) # base distribution N(0, I)

                base_t = torch.rand((num_graphs, 1, 1), device=x.device, dtype=torch.float32).repeat(1, 1,
                                                                                                     self.model.m_delta_dim)

                t = base_t[agent_batch]

                t = torch.where(
                    ego_mask[:, None, None],
                    torch.ones_like(t),
                    t,
                )

                z = (1 - t) * e + t * x  # large t, low noise        target velocity e-x = (z-x)/(1-t)

                x_pred = self.model(z, t, tokenized_agent, initial_map_feature)

                observed_group_loss, pos_loss1, heading_loss1, shape_loss1, vel_loss1, collision_loss1 = get_diff_loss(
                    tokenized_agent,
                    x_pred[:, 0],
                    x[:, 0],
                    z[:, 0],
                    e[:, 0],
                    t[:, 0],
                    use_col=False,  # not self.model.pred_gmm,
                    x_pred=self.x_pred,
                )

                observed_group_loss = torch.stack([pos_loss1, heading_loss1, shape_loss1, vel_loss1], dim=-1)

            policy_loss = self.model.schedule.loss(t[~ego_mask, 0, ::2], observed_group_loss)

        loss = (match_loss, collision_loss + policy_loss, pos_loss, heading_loss, shape_loss, vel_loss)

       # print(match_loss.mean(),sampled_match_loss.mean())

        return loss, x_pred[:, 0], z[:, 0], t[:, 0]  # ,denom[:,0]

    def debug_loss_vs_timestep(
            self,
            t,
            loss_per_agent,
            ego_mask=None,
            prefix="train/fm",
            num_bins=20,
    ):
        """
        Compute binned mean loss as a function of diffusion/flow timestep t.

        Args:
            t:
                Tensor with shape [N], [N, D], or [N, 1, D].
                For grouped schedules, we average over dimensions to get one scalar t per agent.

            loss_per_agent:
                Tensor with shape [N] or [N_non_ego].
                Usually the first output of get_diff_loss.

            ego_mask:
                Bool tensor with shape [N]. If loss_per_agent is non-ego-only,
                the code will automatically align t to non-ego agents.

            prefix:
                Name prefix for returned debug keys.

            num_bins:
                Number of timestep bins.
        """
        with torch.no_grad():
            if t.ndim == 3:
                # [N, 1, D] -> [N]
                t_scalar = t[:, 0].mean(dim=-1)
            elif t.ndim == 2:
                # [N, D] -> [N]
                t_scalar = t.mean(dim=-1)
            elif t.ndim == 1:
                t_scalar = t
            else:
                raise RuntimeError(f"Unsupported t shape: {tuple(t.shape)}")

            loss = loss_per_agent.detach()

            if loss.ndim > 1:
                loss = loss.flatten(1).mean(dim=-1)

            if ego_mask is not None:
                ego_mask = ego_mask.bool()

                if loss.shape[0] == t_scalar.shape[0]:
                    # loss is all-agent length
                    valid_mask = ~ego_mask
                    t_scalar = t_scalar[valid_mask]
                    loss = loss[valid_mask]
                elif loss.shape[0] == int((~ego_mask).sum().item()):
                    # loss is already non-ego-only
                    t_scalar = t_scalar[~ego_mask]
                else:
                    raise RuntimeError(
                        f"Cannot align loss and t: "
                        f"loss={tuple(loss.shape)}, t={tuple(t_scalar.shape)}, "
                        f"ego_mask={tuple(ego_mask.shape)}"
                    )

            finite = torch.isfinite(t_scalar) & torch.isfinite(loss)
            t_scalar = t_scalar[finite].clamp(0.0, 1.0)
            loss = loss[finite].cpu()

            if loss.numel() == 0:
                return {}

            bin_idx = torch.clamp(
                (t_scalar * num_bins).long(),
                min=0,
                max=num_bins - 1,
            ).cpu()


            self.bin_loss_sum.scatter_add_(0, bin_idx, loss)
            self.bin_count.scatter_add_(0, bin_idx, torch.ones_like(loss))

            bin_mean = self.bin_loss_sum / self.bin_count.clamp_min(1.0)

            print(bin_mean)
            #print(self.bin_count/self.bin_count.sum())

    @torch.no_grad()
    def _forward_sample(self, z, t_n, t_next, labels):
        tokenized_agent, initial_map_feature, eval_mask = labels
        num_agents = len(z)

        if self.model.use_return_conditioned:
            tokenized_agent["advantages"] = torch.ones_like(tokenized_agent["init_agent_batch"]).to(torch.float32)

        if self.use_cluster:
            t_n = t_n[:, None, None]
        else:

            t_n = torch.full((num_agents, 1, 1), t_n, device=z.device)
            t_next = torch.full((num_agents, 1, 1), t_next, device=z.device)

            t_n, t_dt = self.model.schedule(t_n, z, tokenized_agent)

            ego_mask = tokenized_agent["ego_mask"]
            if eval_mask is not None:
                ego_mask = ego_mask[eval_mask]
            t_n[ego_mask] = 1

            t_next, t_next_dt = self.model.schedule(t_next, z, tokenized_agent)

            t_next[ego_mask] = 1

        if self.use_scale:
            padding_mask = tokenized_agent["padding_mask"]

            t_n[padding_mask] = 0

        x_cond = self.model(z, t_n, tokenized_agent, initial_map_feature, eval_mask, mode=1)  # [...,:z.shape[-1]]

        if self.model.pred_gmm:
            K = 8
            x_cond = x_cond[:, 0]

            gm_means = x_cond[:, :8 * K].reshape(-1, K, 8)
            logstds = x_cond[:, 9 * K:]
            gm_logweights = x_cond[:, 8 * K:9 * K]  # .log_softmax(dim=1)

            inds = torch.multinomial(gm_logweights.softmax(dim=-1), 1, replacement=True)[:, :, None].repeat(1, 1, 8)

            means = gm_means.gather(dim=1, index=inds)

            stds = logstds.exp()  # (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)

            # (bs, *, n_samples, out_channels, h, w)
            x_cond = stds[:, None] * torch.randn_like(z) + means

        if self.x_pred:

            if x_cond.shape[-1] != z.shape[-1]:
                x_cond = x_cond[..., :z.shape[-1]] + torch.randn_like(z) * (x_cond[..., z.shape[-1]:])
            else:
                x_cond = x_cond[..., :z.shape[-1]]

            v_cond = (x_cond - z) / (1.0 - t_n).clamp_min(self.t_eps)

            # x_euler = z + 0.05 * v_cond*t_dt
            #
            # x_next = self.model(x_euler, t_next, tokenized_agent, initial_map_feature, eval_mask,mode=1)
            #
            # velocity_next = (x_next- x_euler) / (1.0 - t_next).clamp_min(self.t_eps)
            # v_cond=0.5*(v_cond*t_dt+velocity_next*t_next_dt)
        else:
            v_cond = x_cond

        if self.model.label_drop_prob > 0:
            x_cond_non = self.model(z, t_n, tokenized_agent, initial_map_feature, eval_mask, mode=0)
            v_pred_non = (x_cond_non - z) / (1.0 - t_n).clamp_min(self.t_eps)
            v_cond = v_cond + (v_cond - v_pred_non) * 3

        return v_cond, t_n, t_next, x_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, noise_level, sde_inspired=False):

        if sde_inspired:
            gamma = 1
            h = t_next - t

            alpha = torch.clamp(
                1.0 - gamma * h * (1 - t_next),  # t=0  alpha=1
                min=0.0,
                max=1.0,
            )

            e = torch.randn_like(z)

            e = self.model.denormalize(e)

            t = alpha * t
            z = alpha * z + (1 - alpha) * e

        v_pred, t_n, t_next, pred_x0 = self._forward_sample(z, t, t_next, labels)
        tokenized_agent, initial_map_feature, eval_mask = labels

        log_prob = z.new_zeros(z.shape[0])

        if self.use_cluster:
            increasing = tokenized_agent["increasing"]
            non_increasing = ~increasing
            z[non_increasing] = z[non_increasing] + (t_next - t_n)[non_increasing] * v_pred[non_increasing]
            z[increasing] = (
                    (1 - t_next[increasing]) * torch.randn_like(pred_x0[increasing])
                    + t_next[increasing] * pred_x0[increasing]
            )
        elif self.use_sde and torch.any(noise_level > 0):
            # Only the selected noisy scenes use stochastic SDE transition.
            # Non-selected scenes take deterministic Euler steps; otherwise
            # noise_level == 0 creates log(0)/0 in the SDE density.
            sde_mask = noise_level.reshape(noise_level.shape[0], -1).amax(dim=-1) > 0
            z_next = z + (t_next - t_n) * v_pred
            if torch.any(sde_mask):
                z_sde, log_prob_sde, prev_sample_mean, std_dev_t = self.sde_step_with_logprob(
                    1 - t_n[sde_mask],
                    1 - t_next[sde_mask],
                    -v_pred[sde_mask],
                    z[sde_mask],
                    noise_level[sde_mask],
                )
                z_next[sde_mask] = z_sde
                log_prob[sde_mask] = log_prob_sde
            z = z_next
        else:
            z = z + (t_next - t_n) * v_pred

        return z, pred_x0, t_n, log_prob

    @torch.no_grad()
    def sample(self, tokenized_agent, initial_map_feature, eval_mask, infer_steps=20, num_samples=1):

        agent_batch = tokenized_agent["init_agent_batch"]
        num_graphs = tokenized_agent["num_graphs"]
        init_agent_type = tokenized_agent["init_agent_type"]
        num_agents = len(agent_batch)

        if self.use_uniform:
            z = torch.rand(num_agents, num_samples, self.model.m_delta_dim, device=agent_batch.device) * 2 - 1
        else:
            z = torch.randn(num_agents, num_samples, self.model.m_delta_dim,
                            device=agent_batch.device)  # .clamp(min=-3,max=3)#*0.5#*0.9 #

        z = self.model.denormalize(z, init_agent_type)

        diff_input, diff_output = self.model.get_input(tokenized_agent)

        diff_input = diff_input[:, None]

        if self.learn_noise:
            t = torch.zeros((len(agent_batch), 1, self.model.m_delta_dim), device=z.device, dtype=torch.float32)

            x_pred_noise = self.noise_model(torch.zeros_like(z), t, tokenized_agent, initial_map_feature)

            tokenized_agent["x_pred_noise"] = x_pred_noise.detach()

            std = torch.clamp(x_pred_noise[:, :, 8:].exp(), max=50, min=1e-5)

            std[:, :, 2:6] = std[:, :, 2:6] * 2
            std[:, :, 4:6] = std[:, :, 4:6] * 2

            z = std * torch.randn_like(z) + x_pred_noise[:, :, :8]

        z[tokenized_agent["ego_mask"]] = diff_input[tokenized_agent["ego_mask"]]

        z_list = [z.clone()]
        x_list = []
        log_prob_list = []
        feat_list = []
        t_list = []

        if self.model.use_cfg_cond:
            tokenized_agent["cfg"] = torch.ones(num_graphs, device=agent_batch.device) * 2

        if self.use_scale:
            type_counts = tokenized_agent["type_counts"]

            agent_type = tokenized_agent["init_agent_type"]

            num_types = 3

            idx = agent_batch * num_types + agent_type

            mask = agent_type >= 0

            rank = torch.full_like(agent_batch, -1)

            valid_idx = idx[mask]

            # sort group ids
            sorted_idx, perm = torch.sort(valid_idx)

            # detect new groups
            group_change = torch.ones_like(sorted_idx, dtype=torch.bool)
            group_change[1:] = sorted_idx[1:] != sorted_idx[:-1]

            # position inside sorted array
            pos = torch.arange(sorted_idx.numel(), device=sorted_idx.device)

            # first position of each group
            group_start = torch.where(group_change, pos, 0)
            group_start = torch.cummax(group_start, dim=0)[0]

            # rank within group
            sorted_rank = pos - group_start

            # unsort back
            unsorted_rank = torch.empty_like(sorted_rank)
            unsorted_rank[perm] = sorted_rank

            rank[mask] = unsorted_rank

            counts = type_counts.sum(-1)

            schedule, noise_scedule = batch_increasing_schedule(counts, step_number=infer_steps)  # [agent_batch]

            steps = schedule.shape[1] - 1

        else:
            steps = infer_steps

        timesteps = torch.linspace(0, 1, steps + 1, device=agent_batch.device)  # .pow(2/3)

        noise_level = torch.zeros(num_agents, steps, 1, device=agent_batch.device)

        if self.use_sde:
            t_rand = torch.randint(0, steps, (num_graphs,), device=agent_batch.device)

            t_rand = t_rand[agent_batch]

            noise_level[
                torch.arange(num_agents, device=agent_batch.device), t_rand, 0
            ] = self.noise_level

            noise_mask = noise_level[:, :, 0] > 0

            noise_level = noise_level[:, :, None]

        for i in range(steps):  # - 1
            t = timesteps[i]
            t_next = timesteps[i + 1]

            if self.use_scale:
                schedule_i = schedule[:, i]
                schedule_i1 = schedule[:, i + 1]

                k = allocate_k_per_type(schedule_i, type_counts)[agent_batch, agent_type]
                k1 = allocate_k_per_type(schedule_i1, type_counts)[agent_batch, agent_type]

                eval_mask = rank <= k1

                padding_mask = (eval_mask & (rank > k))[eval_mask]

                tokenized_agent['padding_mask'] = padding_mask

                if self.use_cluster:
                    tokenized_agent["increasing"] = (schedule_i != schedule_i1)[agent_batch][eval_mask]

                    t = noise_scedule[:, i][agent_batch][eval_mask]
                    t_next = noise_scedule[:, i + 1][agent_batch][eval_mask][:, None, None]

                z[eval_mask], x_cond, t_n, log_prob = self._euler_step(
                    z[eval_mask],
                    t,
                    t_next,
                    (tokenized_agent, initial_map_feature, eval_mask),
                    noise_level[eval_mask, i],
                )
            else:
                z, x_cond, t_n, log_prob = self._euler_step(z, t, t_next,
                                                            (tokenized_agent, initial_map_feature, eval_mask),
                                                            noise_level[:, i])

            z[tokenized_agent["ego_mask"]] = diff_input[tokenized_agent["ego_mask"]]

            x_list.append(x_cond)
            z_list.append(z.clone())
            t_list.append(t_n.clone())
            log_prob_list.append(log_prob)

        t_list.append(torch.ones_like(t_n))

        if self.use_sde:
            z_list = torch.stack(z_list, dim=1)

            old_log_prob = torch.stack(log_prob_list, dim=1)[noise_mask].detach()
            selected_agent_idx = torch.arange(num_agents, device=agent_batch.device)[:, None].expand(
                num_agents, steps
            )[noise_mask]
            gen_z = (z_list[:, :-1][noise_mask], z_list[:, 1:][noise_mask], old_log_prob)
            t_list = torch.stack(t_list, dim=1)
            gen_t = (t_list[:, :-1][noise_mask], t_list[:, 1:][noise_mask])

            tokenized_agent["gen_z"] = gen_z
            tokenized_agent["gen_t"] = gen_t
            tokenized_agent["gen_agent_idx"] = selected_agent_idx
        else:
            tokenized_agent["pred_z_list"] = torch.cat(z_list, dim=1)
            tokenized_agent["gen_z"] = z

        if self.pred_all_pos:
            all_pred = self.pred_model(z, t_n * 0, tokenized_agent, initial_map_feature).reshape(z.shape[0], -1, 3)

            tokenized_agent["all_pred"] = all_pred

        return z[:, 0], x_list

    def sde_step_with_logprob(
            self,
            sigma,
            sigma_prev,
            model_output: torch.FloatTensor,
            sample: torch.FloatTensor,
            noise_level=0.7,
            prev_sample=None,
            sde_type: Optional[str] = 'sde',
            return_sqrt_dt: Optional[bool] = False,
    ):
        eps = 1e-5
        scale = self.model.normal_scale[None].clamp_min(eps)
        model_output = model_output / scale
        sample = self.model.normalize(sample)

        if prev_sample is not None:
            prev_sample = self.model.normalize(prev_sample)

        sigma = sigma.clamp(min=eps, max=1.0 - eps)
        sigma_prev = sigma_prev.clamp(min=eps, max=1.0 - eps)
        dt = sigma_prev - sigma
        sqrt_neg_dt = (-dt).clamp_min(eps).sqrt()

        if sde_type == 'sde':
            denom = (1.0 - sigma).clamp_min(eps)
            std_dev_t = torch.sqrt((sigma / denom).clamp_min(eps)) * noise_level
            std_dev_t = std_dev_t.clamp_min(eps)

            prev_sample_mean = sample * (
                    1.0 + std_dev_t.square() / (2.0 * sigma) * dt
            ) + model_output * (
                                       1.0 + std_dev_t.square() * (1.0 - sigma) / (2.0 * sigma)
                               ) * dt

            std = (std_dev_t * sqrt_neg_dt).clamp_min(eps)

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)
                prev_sample = prev_sample_mean + std * variance_noise

            log_prob = (
                    -((prev_sample.detach() - prev_sample_mean).square()) / (2.0 * std.square())
                    - torch.log(std)
                    - 0.5 * math.log(2.0 * math.pi)
            )

        elif sde_type == 'cps':
            std_dev_t = (sigma_prev * math.sin(noise_level * math.pi / 2)).clamp_min(eps)
            pred_original_sample = sample - sigma * model_output
            noise_estimate = sample + model_output * (1.0 - sigma)
            inside = (sigma_prev.square() - std_dev_t.square()).clamp_min(eps)
            prev_sample_mean = pred_original_sample * (1.0 - sigma_prev) + noise_estimate * inside.sqrt()

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            log_prob = -((prev_sample.detach() - prev_sample_mean).square()) / (2.0 * std_dev_t.square())
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        log_prob = log_prob.clamp(-self.init_logprob_clip, self.init_logprob_clip)

        prev_sample = self.model.denormalize(prev_sample)

        if return_sqrt_dt:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_neg_dt
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def repeat_input_copy(self, tokenized_agent, n_step):
        out = dict(tokenized_agent)#tokenized_agent#

        num_graphs = tokenized_agent["num_graphs"]
        batch = tokenized_agent["init_agent_batch"]

        out["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step)

        repeated_batch = torch.stack(
            [batch + num_graphs * k for k in range(n_step)],
            dim=1,
        ).transpose(0, 1).flatten(0, 1)

        out["init_agent_batch"] = repeated_batch
        out["init_agent_type"] = tokenized_agent["init_agent_type"][None].repeat(
            n_step, 1
        ).flatten(0, 1)
        out["num_graphs"] = num_graphs * n_step

        if self.model.use_rel_ego:
            out["ego_feat"] = tokenized_agent["ego_feat"][None].repeat(
                n_step, 1, 1
            ).flatten(0, 1)
        else:
            out["ego_embedding"] = tokenized_agent["ego_embedding"][None].repeat(
                n_step, 1, 1
            ).flatten(0, 1)

        return out


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
        uphold = 0.999
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type}")

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)