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
from typing import Dict, Mapping, Optional, Tuple


# TempFlow-GRPO adaptation:
#   1. deterministic flow with one time-indexed SDE branch per rollout;
#   2. noise-aware, mean-one weighting of the PPO objective;
#   3. optional seed-level groups sharing the same initial latent noise.
class ScaleFlow(nn.Module):

    def __init__(self, args, token_processor,gail):
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

        if gail and token_processor.learn_init:
            self.use_sde = True
        else:
            self.use_sde=False

        self.noise_level = float(getattr(args, "noise_level", 0.7))

        self.rationorm = False

        self.global_step = 0

        self.use_nft = False

        self.use_uniform = False

        self.learn_noise = False

        # self.info_sampler=InfoNoiseSampler()

        self.mc_num = 1

        self.init_adv_clip = float(getattr(args, "init_adv_clip", 3.0))
        self.init_logprob_clip = float(getattr(args, "init_logprob_clip", 50.0))
        self.init_ppo_clip = float(getattr(args, "init_ppo_clip", 0.2))

        # TempFlow-GRPO options. All options are backward compatible with old configs.
        # A rollout follows deterministic flow except for one designated SDE branch step.
        self.use_tempflow_grpo = bool(
            getattr(args, "use_tempflow_grpo", False)
        )
        self.use_init_ppo_ratio = bool(
            getattr(args, "use_init_ppo_ratio", False)
        )
        self.tempflow_group_advantages = bool(
            getattr(args, "tempflow_group_advantages", True)
        )
        self.tempflow_stratified_branching = bool(
            getattr(args, "tempflow_stratified_branching", True)
        )
        self.tempflow_seed_group_size = int(
            getattr(args, "tempflow_seed_group_size", 2)
        )
        self.tempflow_branch_min_step = int(
            getattr(args, "tempflow_branch_min_step", 1)
        )
        self.tempflow_branch_max_step = int(
            getattr(args, "tempflow_branch_max_step", -1)
        )
        self.tempflow_weight_min = float(
            getattr(args, "tempflow_weight_min", 0.25)
        )
        self.tempflow_weight_max = float(
            getattr(args, "tempflow_weight_max", 4.0)
        )
        self.tempflow_weight_eps = float(
            getattr(args, "tempflow_weight_eps", 1e-6)
        )

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

        self.use_ref = False

        if self.use_sde:
            self.use_ref=False

        if self.use_ref:
            self.ref_model = copy.deepcopy(self.model)

        num_bins=20

        self.bin_loss_sum = torch.zeros(num_bins)
        self.bin_count = torch.zeros(num_bins)

        self.apply(weight_init)


    def _sanitize_init_advantages(
            self,
            advantages,
            ego_mask,
            selected_agent_idx=None,
            group_ids=None,
    ):
        """Return finite, clipped advantages aligned to sampled SDE transitions.

        When ``group_ids`` is supplied, corresponding agents from repeated scenes are
        normalized within the same seed group. This is the seed-level GRPO grouping
        used by TempFlow-GRPO. Without valid groups, the original batch-level
        normalization is retained.
        """
        if advantages is None:
            return None

        advantages = advantages.detach().to(dtype=torch.float32)
        advantages = torch.nan_to_num(
            advantages,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if advantages.ndim > 1:
            advantages = advantages.reshape(advantages.shape[0], -1).mean(dim=-1)

        if selected_agent_idx is not None:
            if advantages.shape[0] != ego_mask.shape[0]:
                raise RuntimeError(
                    "learn_init advantages must be all-agent length before indexing: "
                    f"advantages={tuple(advantages.shape)}, ego_mask={tuple(ego_mask.shape)}"
                )
            advantages = advantages[selected_agent_idx]
            if group_ids is not None:
                group_ids = group_ids[selected_agent_idx]

        use_group_norm = (
            group_ids is not None
            and group_ids.numel() == advantages.numel()
            and torch.unique(group_ids).numel() < group_ids.numel()
        )

        if use_group_norm:
            _, inverse = torch.unique(group_ids.long(), return_inverse=True)
            num_groups = int(inverse.max().item()) + 1

            group_mean = scatter_mean(advantages, inverse, dim=0, dim_size=num_groups)
            centered = advantages - group_mean[inverse]
            group_var = scatter_mean(
                centered.square(),
                inverse,
                dim=0,
                dim_size=num_groups,
            )
            group_count = torch.bincount(inverse, minlength=num_groups)
            normalized = centered / group_var[inverse].sqrt().clamp_min(1e-6)

            # A singleton has no valid within-group comparison and receives zero
            # policy-gradient advantage rather than an unstable normalized value.
            advantages = torch.where(
                group_count[inverse] > 1,
                normalized,
                torch.zeros_like(normalized),
            )
        elif advantages.numel() > 1:
            advantages = advantages - advantages.mean()
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

        return advantages.clamp(-self.init_adv_clip, self.init_adv_clip)

    def _resolve_tempflow_seed_groups(self, tokenized_agent, num_graphs, device):
        """Resolve graph-level seed groups for repeated copies of the same scene."""
        supplied = tokenized_agent.get("tempflow_seed_group_id", None)
        if supplied is not None:
            group_ids = torch.as_tensor(supplied, device=device, dtype=torch.long)
            if group_ids.numel() != num_graphs:
                raise RuntimeError(
                    "tempflow_seed_group_id must contain one id per graph: "
                    f"got {group_ids.numel()} ids for {num_graphs} graphs"
                )
            return group_ids.reshape(num_graphs)

        group_size = max(self.tempflow_seed_group_size, 1)
        return torch.arange(num_graphs, device=device, dtype=torch.long) // group_size

    def _build_tempflow_agent_groups(
            self,
            agent_batch,
            init_agent_type,
            seed_group_ids,
            num_graphs,
    ):
        """Group corresponding agents across scene replicas and validate layout."""
        local_rank = torch.empty_like(agent_batch)
        graph_indices = []
        max_agents = 0

        for graph_idx in range(num_graphs):
            idx = torch.nonzero(agent_batch == graph_idx, as_tuple=False).flatten()
            graph_indices.append(idx)
            local_rank[idx] = torch.arange(idx.numel(), device=agent_batch.device)
            max_agents = max(max_agents, int(idx.numel()))

        for seed_group in torch.unique(seed_group_ids):
            members = torch.nonzero(
                seed_group_ids == seed_group,
                as_tuple=False,
            ).flatten()
            if members.numel() <= 1:
                continue

            reference = graph_indices[int(members[0].item())]
            reference_types = init_agent_type[reference]
            for member in members[1:]:
                idx = graph_indices[int(member.item())]
                if idx.numel() != reference.numel() or not torch.equal(
                    init_agent_type[idx],
                    reference_types,
                ):
                    raise RuntimeError(
                        "Graphs in a TempFlow seed group must have identical agent "
                        "counts, ordering, and types. Duplicate each scene before "
                        "collation and keep replica ordering unchanged."
                    )

        stride = max(max_agents, 1) + 1
        return seed_group_ids[agent_batch] * stride + local_rank, graph_indices

    @staticmethod
    def _share_tempflow_initial_noise(z, seed_group_ids, graph_indices):
        """Copy one initial latent seed to every scene replica in each seed group."""
        z = z.clone()
        for seed_group in torch.unique(seed_group_ids):
            members = torch.nonzero(
                seed_group_ids == seed_group,
                as_tuple=False,
            ).flatten()
            if members.numel() <= 1:
                continue

            source_idx = graph_indices[int(members[0].item())]
            source_noise = z[source_idx].clone()
            for member in members[1:]:
                target_idx = graph_indices[int(member.item())]
                z[target_idx] = source_noise
        return z

    def _sample_tempflow_branch_steps(
            self,
            tokenized_agent,
            seed_group_ids,
            num_graphs,
            steps,
            device,
    ):
        """Choose one SDE branch time per graph, stratified inside seed groups."""
        supplied = tokenized_agent.get("tempflow_branch_step_override", None)
        if supplied is not None:
            branch_steps = torch.as_tensor(supplied, device=device, dtype=torch.long)
            if branch_steps.numel() != num_graphs:
                raise RuntimeError(
                    "tempflow_branch_step_override must contain one step per graph: "
                    f"got {branch_steps.numel()} values for {num_graphs} graphs"
                )
            if torch.any((branch_steps < 0) | (branch_steps >= steps)):
                raise RuntimeError(
                    f"tempflow_branch_step_override must be in [0, {steps - 1}]"
                )
            return branch_steps.reshape(num_graphs)

        min_step = min(max(self.tempflow_branch_min_step, 0), steps - 1)
        configured_max = self.tempflow_branch_max_step
        max_step = steps - 1 if configured_max < 0 else configured_max
        max_step = min(max(max_step, min_step), steps - 1)
        candidates = torch.arange(min_step, max_step + 1, device=device)

        branch_steps = torch.empty(num_graphs, device=device, dtype=torch.long)
        if not self.tempflow_stratified_branching:
            random_idx = torch.randint(0, candidates.numel(), (num_graphs,), device=device)
            return candidates[random_idx]

        # Scene replicas sharing one initial noise seed cover the full time range
        # instead of clustering around adjacent branch steps.
        num_candidates = candidates.numel()
        for seed_group in torch.unique(seed_group_ids):
            members = torch.nonzero(
                seed_group_ids == seed_group,
                as_tuple=False,
            ).flatten()
            num_members = members.numel()

            if num_members <= num_candidates:
                edges = torch.linspace(
                    0,
                    num_candidates,
                    num_members + 1,
                    device=device,
                )
                low = edges[:-1].floor().long()
                high = (edges[1:].ceil().long() - 1).clamp_max(
                    num_candidates - 1
                )
                high = torch.maximum(high, low)
                width = high - low + 1
                sampled_idx = low + (
                    torch.rand(num_members, device=device) * width
                ).floor().long()
                sampled_idx = sampled_idx[torch.randperm(num_members, device=device)]
            else:
                sampled_idx = torch.arange(num_members, device=device) % num_candidates
                sampled_idx = sampled_idx[torch.randperm(num_members, device=device)]

            branch_steps[members] = candidates[sampled_idx]

        return branch_steps

    def _tempflow_noise_weights(self, std_dev_t):
        """Convert SDE diffusion coefficients into mean-one policy weights."""
        # if not self.use_tempflow_grpo:
        #     return std_dev_t.new_ones(std_dev_t.shape[0])

        # Dimension-specific flow schedules may produce a vector coefficient.
        # RMS gives one effective noise scale per sampled transition.
        raw_weight = std_dev_t.detach().reshape(std_dev_t.shape[0], -1)
        raw_weight = raw_weight.square().mean(dim=-1).sqrt()
        raw_weight = torch.nan_to_num(
            raw_weight,
            nan=0.0,
            posinf=self.tempflow_weight_max,
            neginf=0.0,
        )
        normalized = raw_weight / raw_weight.mean().clamp_min(
            self.tempflow_weight_eps
        )
        bounded = normalized.clamp(
            min=self.tempflow_weight_min,
            max=self.tempflow_weight_max,
        )
        # Preserve the TempFlow convention that the average weight is one, so
        # enabling temporal weighting does not silently change the PG loss scale.
        return bounded / bounded.mean().clamp_min(self.tempflow_weight_eps)

    def adaptive_x0_loss_per_sample(self,pred_x0, target_x0):
        """
        pred_x0:   [B, A, T, D]
        target_x0: [B, A, T, D]
        valid_mask:[B, A, T]
        return:    [B]
        """
        err = pred_x0 - target_x0

        mse = err.square()  # [B, A, T, D]
        l1 = err.abs().mean(dim=-1, keepdim=True)
        l1 = l1.clamp_min(1e-5).detach()

        loss = (mse / l1).mean(dim=-1)  # [B, A, T]
        # denom = (1 - t_n_sampled[:, 0]).clamp_min(self.t_eps)
        # denom_sq = denom.square()
        # sampled_match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
        #     tokenized_agent,
        #     x_pred[:, 0],
        #     x_sampled[:, 0],
        #     z_sampled[:, 0],
        #     e_sampled[:, 0],
        #     t_n_sampled[:, 0],
        #     use_col=False,
        #     x_pred=self.x_pred
        # )

        # inv_denom_sq = denom_sq.reciprocal()
        # mse_Loss = F.mse_loss(x_pred[:, 0] / scale, x_sampled[:, 0] / scale, reduction="none")
        # # l1_Loss=F.l1_loss(x_pred[:, 0] /scale, x_sampled[:, 0]/scale , reduction="none").mean(-1, keepdim=True).clip(min=0.00001).detach()
        # #
        # sampled_match_loss = (mse_Loss * inv_denom_sq).mean(-1).reshape(self.mc_num, -1).mean(0)
        #
        return -loss.mean(dim=1)*0.1

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

        if "advantages" in tokenized_agent.keys():
            raw_advantages = tokenized_agent["advantages"]

            if self.use_sde:

                z_sampled, prev_sample, old_log_prob = tokenized_agent["sde_z"]
                t_n_sampled, t_next_sampled = tokenized_agent["sde_t"]
                selected_agent_idx = tokenized_agent.get(
                    "gen_agent_idx",
                    torch.arange(z_sampled.shape[0], device=z_sampled.device),
                )
                tempflow_group_ids = None
                if self.use_tempflow_grpo and self.tempflow_group_advantages:
                    tempflow_group_ids = tokenized_agent.get(
                        "tempflow_agent_group_id",
                        None,
                    )

                advantages = self._sanitize_init_advantages(
                    raw_advantages,
                    ego_mask,
                    selected_agent_idx=selected_agent_idx,
                    group_ids=tempflow_group_ids,
                )
            else:
                advantages = raw_advantages
                gen_z=tokenized_agent["gen_z"]
                # gen_z[:, :, 2:4] = gen_z[:, :, 2:4] / gen_z[:, :, 2:4].norm(dim=-1, keepdim=True)
                # gen_z[:, 0, 6:8] = tokenized_agent["initial_local_vel"]

                # gen_z,gen_z=self.model.get_input(tokenized_agent,expert_data=False)
                # gen_z=gen_z[:,None]

                x_sampled = gen_z[None].repeat(self.mc_num, 1, 1, 1).flatten(0, 1)
                e_sampled = torch.randn_like(e[None].repeat(self.mc_num, 1, 1, 1).flatten(0, 1))

                agent_batch = torch.stack(
                    [
                        agent_batch + num_graphs * t0
                        for t0 in range(self.mc_num)
                    ],
                    dim=1,
                ).transpose(0, 1).flatten(0, 1)  # [n_agent*n_step]

                sampled_base_t = torch.rand_like(base_t[None].repeat(self.mc_num, 1, 1, 1).flatten(0, 1))[agent_batch]
                t_n_sampled, _ = self.model.schedule(
                    sampled_base_t, x_sampled, tokenized_agent
                )

                t_n_sampled = torch.where(
                    ego_mask.repeat(self.mc_num, 1).flatten(0, 1)[:, None, None],
                    torch.ones_like(t_n_sampled),
                    t_n_sampled,
                )

                z_sampled = (1 - t_n_sampled) * e_sampled + t_n_sampled * x_sampled

            denom = (1.0 - t_n_sampled).clamp_min(self.t_eps)

            with torch.no_grad():
                if self.use_ref:
                    decay = return_decay(self.global_step, 2)
                    for src_param, tgt_param in zip(self.model.parameters(), self.ref_model.parameters(), strict=True):
                        tgt_param.data.copy_(
                            tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

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

            t_all = torch.cat((t_n_sampled, t), dim=0)
            z_all =  torch.cat((z_sampled, z), dim=0)
            model_tokenized_agent = self.repeat_input_copy(
                tokenized_agent,
                self.mc_num + 1,
            )

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

            if self.use_nft:
                adv_clip_max = 3

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
                    advantages_pg = advantages[sampled_non_ego]
                    time_weight = self._tempflow_noise_weights(std_dev_t)

                    tokenized_agent["tempflow_weight_mean"] = time_weight.mean().detach()
                    tokenized_agent["tempflow_weight_min"] = time_weight.min().detach()
                    tokenized_agent["tempflow_weight_max"] = time_weight.max().detach()

                    if self.use_init_ppo_ratio and old_log_prob is not None:
                        old_lp = old_log_prob[sampled_non_ego].detach()
                        old_lp = torch.nan_to_num(
                            old_lp,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        old_lp = old_lp.clamp(
                            -self.init_logprob_clip,
                            self.init_logprob_clip,
                        )
                        log_ratio = (log_prob - old_lp).clamp(-10.0, 10.0)
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = ratio.clamp(
                            1.0 - self.init_ppo_clip,
                            1.0 + self.init_ppo_clip,
                        )
                        surrogate_1 = ratio * advantages_pg
                        surrogate_2 = clipped_ratio * advantages_pg
                        clipped_surrogate = torch.minimum(
                            surrogate_1,
                            surrogate_2,
                        )
                        policy_loss = -(time_weight * clipped_surrogate).mean()

                        tokenized_agent["tempflow_ratio_mean"] = ratio.mean().detach()
                        tokenized_agent["tempflow_clip_fraction"] = (
                            (ratio < 1.0 - self.init_ppo_clip)
                            | (ratio > 1.0 + self.init_ppo_clip)
                        ).float().mean().detach()
                    else:
                        policy_loss = -(
                            1 * log_prob * advantages_pg
                        ).mean()
            else:
                new_pred_x0 = x_pred_all[:len(z_sampled)]

                scale=self.model.normal_scale[:,None]

                logp_cur = self.adaptive_x0_loss_per_sample(
                    new_pred_x0/scale, x_sampled.detach()/scale).reshape(self.mc_num, -1).mean(0)

                tokenized_agent["sampled_match_loss"]=-logp_cur

                non_ego = ~ego_mask
                advantages_pg = self._sanitize_init_advantages(
                    advantages,
                    ego_mask,
                    selected_agent_idx=None,
                )[non_ego]

                if self.use_ref:
                    logp_old= self.adaptive_x0_loss_per_sample(
                    ref_prediction/scale, x_sampled.detach()/scale).reshape(self.mc_num, -1).mean(0)
                else:
                    logp_old = logp_cur.detach()

                log_ratio = logp_cur - logp_old

                # scale = log_ratio.detach().std().clamp_min(1e-6)
                # log_ratio = -log_ratio / scale
                #
                ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))[non_ego]#

                clip_eps = 0.1
                tokenized_agent["clip_ratio"]=((ratio<1-clip_eps) | (ratio>1+clip_eps) ).float()

                ratio_clip = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)


                surrogate_1 = ratio * advantages_pg.detach()
                surrogate_2 = ratio_clip * advantages_pg.detach()

                policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()*100
                # smooth proximal correction
                # pepg_loss#advantages_pg = advantages_pg -  log_ratio.detach()

                # coef = (ratio.detach() * advantages_pg).detach()
                #
                # policy_loss = -(coef * logp_cur).mean()
                # advantages_pg = torch.exp(advantages_pg/2).detach()#.clamp_max(5)

                #policy_loss=(-advantages_pg*logp_cur).mean()*0.1##.exp()

                beta_kl=0

                if beta_kl > 0:
                    delta = (logp_old.detach() - logp_cur).clamp(-10.0, 10.0)
                    kl = torch.exp(delta) - delta - 1.0

                    policy_loss = policy_loss + beta_kl * kl.mean()

            tokenized_agent["pg_loss"]=policy_loss

            x_pred = x_pred_all[len(z_sampled):]

            #x_pred = self.model(z, t, tokenized_agent, initial_map_feature)


        else:
            x_pred = self.model(z, t, tokenized_agent, initial_map_feature)

        if self.use_scale:
            x = out[:, None]

        if not self.model.pred_gmm:
            ego_mask_expand = ego_mask.view(
                ego_mask.shape[0],
                *([1] * (x_pred.ndim - 1)),
            )

            x_pred[:,:,:x.shape[-1]] = torch.where(
                ego_mask_expand,
                x.detach(),
                x_pred[:,:,:x.shape[-1]]
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
                x_cond = x_cond[..., :z.shape[-1]] + torch.randn_like(z) * (x_cond[..., z.shape[-1]:].exp())
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
                            device=agent_batch.device) #*0.9 # .clamp(min=-3,max=3)#*0.5# #

        # clip_min = torch.tensor(
        #     [-83.783, -79.270, -1.000, -1.000, 0.564, 0.593, -4.535, -9.342],
        #     device=z.device,
        #     dtype=z.dtype,
        # )
        #
        # clip_max = torch.tensor(
        #     [100.118, 81.043, 1.000, 1.000, 20.335, 3.835, 31.341, 3.648],
        #     device=z.device,
        #     dtype=z.dtype,
        # )

        z = self.model.denormalize(z, init_agent_type)

       # z = torch.clamp(z, min=clip_min, max=clip_max)

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

        if self.use_sde and self.use_tempflow_grpo:
            seed_group_ids = self._resolve_tempflow_seed_groups(
                tokenized_agent,
                num_graphs,
                agent_batch.device,
            )
            tempflow_agent_group_ids, graph_indices = (
                self._build_tempflow_agent_groups(
                    agent_batch,
                    init_agent_type,
                    seed_group_ids,
                    num_graphs,
                )
            )
            z = self._share_tempflow_initial_noise(
                z,
                seed_group_ids,
                graph_indices,
            )
        else:
            # Preserve legacy behavior when TempFlow-GRPO is disabled.
            seed_group_ids = torch.arange(
                num_graphs,
                device=agent_batch.device,
                dtype=torch.long,
            )
            tempflow_agent_group_ids = torch.arange(
                num_agents,
                device=agent_batch.device,
                dtype=torch.long,
            )

        tokenized_agent["tempflow_seed_group_id"] = seed_group_ids
        tokenized_agent["tempflow_agent_group_id"] = tempflow_agent_group_ids

        z[tokenized_agent["ego_mask"]] = diff_input[tokenized_agent["ego_mask"]]

        z_list = [z.clone()]
        x_list = []
        log_prob_list = []
        feat_list = []
        t_list = []

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

        timesteps = torch.linspace(0, 1, steps + 1, device=agent_batch.device).pow(3/2)

        noise_level = torch.zeros(num_agents, steps, 1, device=agent_batch.device)

        if self.use_sde:
            if self.use_tempflow_grpo:
                branch_step_graph = self._sample_tempflow_branch_steps(
                    tokenized_agent,
                    seed_group_ids,
                    num_graphs,
                    steps,
                    agent_batch.device,
                )
            else:
                branch_step_graph = torch.randint(
                    0,
                    steps,
                    (num_graphs,),
                    device=agent_batch.device,
                )

            branch_step_agent = branch_step_graph[agent_batch]
            noise_level[
                torch.arange(num_agents, device=agent_batch.device),
                branch_step_agent,
                0,
            ] = self.noise_level

            noise_mask = noise_level[:, :, 0] > 0
            noise_level = noise_level[:, :, None]

            tokenized_agent["tempflow_branch_step"] = branch_step_graph
            tokenized_agent["tempflow_branch_time"] = (
                branch_step_graph.to(dtype=timesteps.dtype) / float(steps)
            )

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

            #z = torch.clamp(z, min=clip_min, max=clip_max)

            z[tokenized_agent["ego_mask"]] = diff_input[tokenized_agent["ego_mask"]]

            feat_list.append(tokenized_agent["noise_feat"])

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

            tokenized_agent["sde_z"] = gen_z
            tokenized_agent["sde_t"] = gen_t
            tokenized_agent["gen_agent_idx"] = selected_agent_idx
            tokenized_agent["tempflow_selected_branch_step"] = (
                branch_step_agent[selected_agent_idx]
            )
            tokenized_agent["tempflow_selected_seed_group"] = (
                seed_group_ids[agent_batch[selected_agent_idx]]
            )
            tokenized_agent["noise_feat"] = torch.stack(feat_list, dim=1)[noise_mask]
            #tokenized_agent["z_log_prob"]=z_log_prob
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
        sigma_max=0.95

        if prev_sample is not None:
            prev_sample = self.model.normalize(prev_sample)

        dt = sigma_prev - sigma

        if sde_type == 'sde':
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

            prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + model_output * (
                        1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt


            std = (std_dev_t * torch.sqrt(-1*dt))

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

        prev_sample = self.model.denormalize(prev_sample)

        if return_sqrt_dt:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_neg_dt
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def repeat_input_copy(self, tokenized_agent, n_step):
        out =tokenized_agent# dict(tokenized_agent)#tokenized_agent#

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

    @torch.no_grad()
    def self_resample_initial_state(
            self,
            tokenized_agent,
            initial_map_feature: Mapping[str, torch.Tensor],
            training_step: int,
            warmup_steps: int = 0,
            apply_prob: float = 1,
            # timestep_shift: float = 0.6,
            # min_strength: float = 0.03,
            # max_strength: float = 0.45,
            # max_pos_delta: float = 2.0,
            # max_vel_delta: float = 3.0,
            # max_shape_ratio: float = 0.20,
            # timestep_shift=0.35,
            # min_strength=0.01,
            # max_strength=0.20,
            # max_pos_delta=0.75,
            # max_vel_delta=1.00,
            # max_shape_ratio=0.05,
            timestep_shift=0.25/2,
            min_strength = 0.005/2,
            max_strength = 0.12/2,
            max_pos_delta = 0.40/2,
            max_vel_delta = 0.60/2,
            max_shape_ratio = 0.03/2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate model-error-aware initial states by self-resampling.

        Returns:
            resampled_x:
                Local initial state [N_agent, 8]:
                [x, y, heading_cos, heading_sin,
                 length, width, vel_x, vel_y].

            info:
                Diagnostics including per-scene resampling strength.
        """
        clean_x, _ = self.model.get_input(tokenized_agent)
        clean_x = clean_x[:, None]  # [N, 1, 8]

        num_graphs = int(tokenized_agent["num_graphs"])
        agent_batch = tokenized_agent["init_agent_batch"]
        agent_type = tokenized_agent["init_agent_type"]
        ego_mask = tokenized_agent["ego_mask"].bool()

        zero_info = {
            "applied": torch.tensor(False, device=clean_x.device),
            "strength": clean_x.new_zeros(num_graphs),
        }

        # --------------------------------------------------------------
        # 1. Teacher-forcing warmup.
        # --------------------------------------------------------------
        if training_step < warmup_steps:
            return clean_x[:, 0], zero_info

        if torch.rand((), device=clean_x.device) >= apply_prob:
            return clean_x[:, 0], zero_info

        # --------------------------------------------------------------
        # 2. Sample resampling strength from shifted logit-normal.
        #
        # strength near 0:
        #     sample remains close to ground truth.
        #
        # strength near 1:
        #     sample approaches full generation.
        # --------------------------------------------------------------
        logit_noise = torch.randn(
            num_graphs,
            1,
            1,
            device=clean_x.device,
            dtype=clean_x.dtype,
        )

        strength_scene = torch.sigmoid(logit_noise)

        shift = float(timestep_shift)
        strength_scene = (
                shift * strength_scene
                / (1.0 + (shift - 1.0) * strength_scene)
        )

        strength_scene = strength_scene.clamp(
            min=min_strength,
            max=max_strength,
        )

        strength_agent = strength_scene[agent_batch]

        # Ego remains a clean conditioning state.
        strength_agent = torch.where(
            ego_mask[:, None, None],
            torch.zeros_like(strength_agent),
            strength_agent,
        )

        # Your model uses t=1 for data and t=0 for noise.
        base_start_t = 1.0 - strength_agent

        # Apply the model's dimension-specific schedule.
        start_t, _ = self.model.schedule(
            base_start_t,
            clean_x,
            tokenized_agent,
        )

        start_t = torch.where(
            ego_mask[:, None, None],
            torch.ones_like(start_t),
            start_t,
        )

        # --------------------------------------------------------------
        # 3. Partially corrupt the clean initial state.
        # --------------------------------------------------------------
        noise = torch.randn_like(clean_x)
        noise = self.model.denormalize(noise, agent_type)

        z_start = (
                (1.0 - start_t) * noise
                + start_t * clean_x
        )

        # --------------------------------------------------------------
        # 4. Online-model self-resampling.
        #
        # For x-prediction, one model call directly predicts the endpoint.
        # For velocity prediction, one Euler step advances from start_t to 1.
        # --------------------------------------------------------------
        prediction = self.model(
            z_start,
            start_t,
            tokenized_agent,
            initial_map_feature,
        )

        if self.x_pred:
            resampled_x = prediction
        else:
            resampled_x = (
                    z_start
                    + (1.0 - start_t) * prediction
            )

        # Detach explicitly: no gradient through the self-resampling stage.
        resampled_x = resampled_x.detach()

        # Ego must stay exactly equal to logged ego.
        resampled_x[ego_mask] = clean_x[ego_mask]

        # --------------------------------------------------------------
        # 5. Safety stabilization.
        #
        # Prevent immature models from producing catastrophic initial states.
        # These bounds can be relaxed after the model becomes strong.
        # --------------------------------------------------------------

        # Position: clamp vector magnitude, not each axis independently.
        pos_delta = resampled_x[..., :2] - clean_x[..., :2]
        pos_norm = pos_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pos_scale = (
                max_pos_delta / pos_norm
        ).clamp(max=1.0)

        resampled_x[..., :2] = (
                clean_x[..., :2]
                + pos_delta * pos_scale
        )

        # Heading: enforce unit cosine/sine vector.
        heading_vec = resampled_x[..., 2:4]
        heading_vec = heading_vec / heading_vec.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        resampled_x[..., 2:4] = heading_vec

        # Shape: constrain relative change.
        clean_shape = clean_x[..., 4:6]

        shape_min = (
                clean_shape * (1.0 - max_shape_ratio)
        ).clamp_min(0.1)

        shape_max = (
                clean_shape * (1.0 + max_shape_ratio)
        ).clamp_min(0.2)

        resampled_shape = resampled_x[..., 4:6]
        resampled_shape = torch.maximum(resampled_shape, shape_min)
        resampled_shape = torch.minimum(resampled_shape, shape_max)

        resampled_x[..., 4:6] = resampled_shape

        # Velocity: clamp change magnitude.
        vel_delta = resampled_x[..., 6:8] - clean_x[..., 6:8]
        vel_norm = vel_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        vel_scale = (
                max_vel_delta / vel_norm
        ).clamp(max=1.0)

        resampled_x[..., 6:8] = (
                clean_x[..., 6:8]
                + vel_delta * vel_scale
        )

        resampled_x[ego_mask] = clean_x[ego_mask]

        info = {
            "applied": torch.tensor(True, device=clean_x.device),
            "strength": strength_scene[:, 0, 0],
            "mean_position_error": (
                    resampled_x[:, 0, :2] - clean_x[:, 0, :2]
            ).norm(dim=-1).mean(),
            "mean_velocity_error": (
                    resampled_x[:, 0, 6:8] - clean_x[:, 0, 6:8]
            ).norm(dim=-1).mean(),
        }

        return resampled_x[:, 0], info


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