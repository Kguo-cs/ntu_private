"""Rectified Flow with timestep-adaptive SDE noise.

State convention:
    [x, y, heading_cos, heading_sin, length, width, vx, vy]

Rectified-Flow convention:
    x0 ~ data, x1 ~ noise
    x_t = (1 - t) * x0 + t * x1
    velocity = dx_t/dt = x1 - x0

Training uses t in [0, 1]. Generation starts from noise at t=1 and
integrates backward to data at t=0. The SDE/PPO transition uses the same
Rectified-Flow time directly; no complementary ``1 - t`` variable is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.diffusion.diffusion_utils import (
    get_closest_sum_idx_fast,
    get_diff_loss,
)
from src.smart.utils import weight_init
import copy
from .denoiser import InitDenoiser


class ScaleFlow(nn.Module):
    """Initial-state flow with simple timestep-adaptive branch noise."""

    def __init__(
        self,
        args,
        token_processor,
        gail: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(args.hidden_dim)

        # Standard x0-prediction flow. No Gaussian or MeanFlow output.
        self.x_pred = True
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
            x_pred=True,
        )

        self.use_all_type = self.model.use_all_type
        self.t_eps = 0.05

        # --------------------------------------------------------------
        # Initial-state policy optimization
        # --------------------------------------------------------------
        self.use_sde = bool(
            gail and getattr(token_processor, "learn_init", False)
        )

        self.use_init_ppo_ratio = bool(
            getattr(args, "use_init_ppo_ratio", False)
        )
        self.init_adv_clip = float(
            getattr(args, "init_adv_clip", 10.0)
        )
        self.init_logprob_clip = float(
            getattr(args, "init_logprob_clip", 50.0)
        )
        self.init_ppo_clip = float(
            getattr(args, "init_ppo_clip", 0.1)
        )

        # --------------------------------------------------------------
        # Multi-branch sampling
        # --------------------------------------------------------------
        self.num_branch_steps = int(
            getattr(args, "num_branch_steps", 1)
        )
        if self.num_branch_steps <= 0:
            raise ValueError(
                "num_branch_steps must be positive."
            )

        self.fixed_branch_steps = self._parse_fixed_branch_steps(
            getattr(args, "branch_steps", None)
        )

        # Simple timestep-adaptive noise. target_step_std is the desired
        # transition standard deviation in normalized state space at t=1.
        self.target_step_std = float(
            getattr(args, "target_step_std", 0.1)
        )

        # self.register_buffer(
        #     "noise_dim_weights",
        #     torch.zeros(
        #         [8],
        #         dtype=torch.float32,
        #     )[None],
        # )
        self.use_ref = False
        self.apply(weight_init)

    @staticmethod
    def _parse_fixed_branch_steps(
        value,
    ) -> Optional[tuple[int, ...]]:
        if value is None:
            return None

        if torch.is_tensor(value):
            value = (
                value.detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
        ):
            raise TypeError(
                "branch_steps must be a sequence "
                "of timestep indices."
            )

        steps = tuple(
            sorted(
                {
                    int(step)
                    for step in value
                }
            )
        )

        if not steps:
            raise ValueError(
                "branch_steps cannot be empty."
            )

        return steps

    def _schedule_time(
        self,
        time: Tensor,
    ) -> Tensor:
        """Apply the optional grouped learnable schedule.

        Supported results:
            [N, 1]
            [N, state_dim]
        """
        schedule = getattr(
            self.model,
            "schedule",
            None,
        )

        scheduled = (
            schedule(time)
            if callable(schedule)
            else time
        )

        if (
            scheduled.ndim != 2
            or scheduled.shape[0] != time.shape[0]
        ):
            raise ValueError(
                "Time schedule must return "
                "[N, 1] or [N, state_dim], got "
                f"{tuple(scheduled.shape)}."
            )

        if scheduled.shape[-1] not in (
            1,
            self.model.m_delta_dim,
        ):
            raise ValueError(
                "Scheduled time must have either one "
                "channel or state_dim channels."
            )

        return scheduled

    # ==================================================================
    # Standard flow helpers
    # ==================================================================
    def _sample_noise(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        #x = self.model.normalize(x)
        noise = torch.randn_like(x)
        noise=self.model.denormalize(
            noise
        )

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()
        noise[ego_mask] = x[ego_mask]

        matched_index = get_closest_sum_idx_fast(
            noise,
            x,
            tokenized_agent,
            all_state=True,
        )

        return noise[matched_index]

        # return self.model.denormalize(
        #     noise[matched_index]
        # )
        #
    def _sample_time(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        batch = tokenized_agent["batch"].long()
        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        scene_time = torch.rand(
            (num_graphs, 1),
            device=x.device,
            dtype=x.dtype,
        )

        return scene_time[batch]

    @staticmethod
    def _fix_conditioned_agents(
        clean: Tensor,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
    ) -> None:
        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()

        latent[ego_mask] = clean[ego_mask]
        # Under x_t=(1-t)x0+t*x1, clean data corresponds to t=0.
        time[ego_mask] = 0.0

    def _prepare_supervised_batch(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        noise = self._sample_noise(
            x,
            tokenized_agent,
        )

        time = self._sample_time(
            x,
            tokenized_agent,
        )

        self._fix_conditioned_agents(
            x,
            noise,
            time,
            tokenized_agent,
        )

        # Rectified Flow: x0=data, x1=noise.
        latent = (
            (1.0 - time) * x
            + time * noise
        )

        return noise, time, latent

    def _model_velocity(
        self,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        eval_mask: Optional[Tensor] = None,
        mode: int = 1,
        use_map_condition: bool = True,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.model(
            latent,
            time,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )

        if (
            prediction.ndim != 2
            or prediction.shape[0] != latent.shape[0]
        ):
            raise ValueError(
                "Denoiser prediction must have shape "
                f"[N, D], got {tuple(prediction.shape)}."
            )

        if prediction.shape[-1] < latent.shape[-1]:
            raise ValueError(
                "Denoiser output has fewer dimensions "
                "than the latent state."
            )

        x0 = prediction[:, : latent.shape[-1]]

        # x_t = (1-t)x0 + t*x1 and v = x1-x0 imply
        # v = (x_t-x0) / t.
        velocity = (
            latent - x0
        ) / time.clamp_min(
            self.t_eps
        )

        return velocity, x0

    def get_adaptive_noise_level(
        self,
        time: Tensor,
        next_time: Tensor,
    ) -> Tensor:
        """Return SDE noise level in Rectified-Flow time.

        Convention:
            x_t = (1 - t) * x0 + t * x1

        Generation runs backward, so ``next_time <= time``. The transition
        standard deviation is

            std = sqrt(t / (1-t)) * noise_level * sqrt(t-next_t),

        apart from the same t=1 boundary stabilization used previously.
        """
        eps = 1e-5

        if time.shape != next_time.shape:
            raise ValueError("time and next_time must have the same shape.")

        delta_t = (time - next_time).clamp_min(0.0)
        time_mid = (0.5 * (time + next_time)).clamp(0.0, 1.0)

        # Desired transition std in normalized state space.
        target_std = 0.05  # e.g. self.target_step_std * (1.0 - time_mid)

        time_for_ratio = torch.where(
            time >= 1.0,
            torch.full_like(time, 0.95),
            time,
        )

        base_std = torch.sqrt(
            time.clamp_min(0.0)
            / (1.0 - time_for_ratio).clamp_min(eps)
        ) * torch.sqrt(delta_t.clamp_min(eps))

        noise_level = target_std / base_std.clamp_min(eps)
        return noise_level

    def get_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, Tensor],
    ):
        loss=self._supervised_loss(
            x,
            tokenized_agent,
            initial_map_feature,
        )

        if "advantages" in tokenized_agent:
            if self.use_sde:
                rl_loss = self._sde_advantage_loss(
                    tokenized_agent,
                    initial_map_feature,
                )
            else:
                rl_loss = self._direct_advantage_loss(
                    tokenized_agent,
                    initial_map_feature,
                )

            tokenized_agent["rl_loss"] = rl_loss

        return loss

    def _supervised_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):
        _, time, latent = (
            self._prepare_supervised_batch(
                x,
                tokenized_agent,
            )
        )

        _, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()[:, None]

        # Avoid in-place modification of model output.
        x0 = torch.where(
            ego_mask,
            x.detach(),
            x0,
        )

        loss = get_diff_loss(
            tokenized_agent,
            x0,
            x,
            time,
            self.t_eps,
            scale=self.model.normal_scale,
            use_col=True,
            x_pred=True,
        )

        return loss, x0, latent, time

    def _sde_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        (
            current,
            next_sample,
            old_log_prob,
        ) = tokenized_agent["sde_z"]

        (
            time,
            next_time,
        ) = tokenized_agent["sde_t"]

        saved_noise_level = tokenized_agent.get(
            "sde_noise_level"
        )

        if current.ndim == 2:
            current = current[:, None]
            next_sample = next_sample[:, None]
            time = time[:, None]
            next_time = next_time[:, None]

            if old_log_prob is not None:
                old_log_prob = old_log_prob[:, None]

            if saved_noise_level is not None:
                saved_noise_level = saved_noise_level[:, None]

        num_agents, num_branches, _ = current.shape

        non_ego = ~tokenized_agent[
            "ego_mask"
        ].bool()

        if not torch.any(non_ego):
            return current.new_zeros(())

        # valid_transition = torch.zeros(
        #     num_agents,
        #     num_branches,
        #     device=current.device,
        #     dtype=torch.bool,
        # )

        tokenized_agent=self.repeat_input_copy(tokenized_agent,num_branches)

        velocities, pred_x0 = self._model_velocity(
            current.transpose(0, 1).flatten(0, 1),
            time.transpose(0, 1).flatten(0, 1),
            tokenized_agent,
            map_feature,
        )
        # pred_x0 = pred_x0.reshape(
        #     num_branches,
        #     num_agents,
        #     -1,
        # ).transpose(0, 1)

        velocities = velocities.reshape(
            num_branches,
            num_agents,
            -1,
        ).transpose(0, 1)  # [A, B, D]

        branch_noise = saved_noise_level.detach()

        delta_t = time - next_time

        active = (
                non_ego[:, None]
                & (branch_noise.amax(dim=-1) > 0)
                & (delta_t.amax(dim=-1) > 1e-7)
        )

        if not torch.any(active):
            return current.new_zeros(())

        log_prob = current.new_zeros(
            num_agents,
            num_branches,
        )

        (
            _,
            active_log_prob,
            _,
            _,
        ) = self.sde_step_with_logprob(
            time=time[active],
            next_time=next_time[active],
            model_output=velocities[active],
            sample=current[active],
            noise_level=branch_noise[active],
            prev_sample=next_sample[active],
        )

        # (
        #     _,
        #     active_log_prob,
        #     _,
        #     _,
        # ) = self.precise_step_with_logprob(
        #     time=time[active],
        #     next_time=next_time[active],
        #     pred_x0=pred_x0[active],
        #     sample=current[active],
        #     eta=branch_noise[active],
        #     prev_sample=next_sample[active],
        # )

        log_prob[active] = active_log_prob

        # for branch in range(num_branches):
        #     # velocity, _ = self._model_velocity(
        #     #     current[:, branch],
        #     #     time[:, branch],
        #     #     tokenized_agent,
        #     #     map_feature,
        #     # )
        #
        #     #velocity=velocities.reshape(num_branches,len(current),-1)[branch]
        #
        #     branch_noise = (
        #         saved_noise_level[:, branch]
        #         .detach()
        #     )
        #
        #     active = (
        #         non_ego
        #         & (
        #             branch_noise.amax(dim=-1)
        #             > 0
        #         )
        #     )
        #
        #     # if not torch.any(active):
        #     #     continue
        #     #
        #     # (
        #     #     _,
        #     #     branch_log_prob,
        #     #     _,
        #     #     _,
        #     # ) = self.sde_step_with_logprob(
        #     #     time=time[active, branch],
        #     #     next_time=next_time[active, branch],
        #     #     model_output=velocity[active],
        #     #     sample=current[active, branch],
        #     #     noise_level=branch_noise[active],
        #     #     prev_sample=next_sample[
        #     #         active,
        #     #         branch,
        #     #     ],
        #     # )
        #     #
        #     # log_prob[
        #     #     active,
        #     #     branch,
        #     # ] = branch_log_prob
        #
        #     valid_transition[
        #         active,
        #         branch,
        #     ] = True

        valid_transition = ( non_ego[:, None]
                & (saved_noise_level.amax(dim=-1) > 0)
        )  # [num_agents, num_branches]


        if not torch.any(valid_transition):
            return current.new_zeros(())

        advantages = tokenized_agent["advantages"].transpose(0, 1) #a,t

        #advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / advantages.std(dim=0, keepdim=True)

        selected_log_prob = log_prob[
            valid_transition
        ]
        selected_advantage = advantages[
            valid_transition
        ]


        if (
            self.use_init_ppo_ratio
            and old_log_prob is not None
        ):
            selected_old_log_prob = (
                old_log_prob.detach()[
                    valid_transition
                ]
            )

            selected_old_log_prob = torch.nan_to_num(
                selected_old_log_prob,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(
                -self.init_logprob_clip,
                self.init_logprob_clip,
            )

            log_ratio = (
                selected_log_prob
                - selected_old_log_prob
            ).clamp(
                -10.0,
                10.0,
            )

            ratio = log_ratio.exp()

            clipped_ratio = ratio.clamp(
                1.0 - self.init_ppo_clip,
                1.0 + self.init_ppo_clip,
            )

            policy_loss = -torch.minimum(
                ratio * selected_advantage,
                clipped_ratio * selected_advantage,
            ).mean()

            tokenized_agent[
                "init_ratio_mean"
            ] = ratio.detach().mean()

            tokenized_agent[
                "init_clip_fraction"
            ] = (
                (
                    ratio
                    < 1.0 - self.init_ppo_clip
                )
                | (
                    ratio
                    > 1.0 + self.init_ppo_clip
                )
            ).float().mean().detach()

            return policy_loss

        return -(
            selected_log_prob
            * selected_advantage
        ).mean()

    def _direct_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        sampled_x0 = tokenized_agent["gen_z"]

        _, time, latent = (
            self._prepare_supervised_batch(
                sampled_x0,
                tokenized_agent,
            )
        )

        _, current_x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        scale = self.model.normal_scale.clamp_min(
            1e-6
        )

        log_prob = self.adaptive_x0_logprob(
            current_x0 / scale,
            sampled_x0.detach() / scale,
        )

        num_agents = sampled_x0.shape[0]

        non_ego = ~tokenized_agent[
            "ego_mask"
        ].bool()

        if not torch.any(non_ego):
            return sampled_x0.new_zeros(())

        advantages = self._prepare_advantages(
            tokenized_agent["advantages"],
            num_agents,
        )[non_ego]

        old_log_prob = log_prob.detach()

        ratio = (
            log_prob - old_log_prob
        ).clamp(
            -10.0,
            10.0,
        ).exp()[non_ego]

        clipped_ratio = ratio.clamp(
            1.0 - self.init_ppo_clip,
            1.0 + self.init_ppo_clip,
        )

        tokenized_agent[
            "sampled_match_loss"
        ] = -log_prob.detach()

        tokenized_agent[
            "clip_ratio"
        ] = (
            (
                ratio
                < 1.0 - self.init_ppo_clip
            )
            | (
                ratio
                > 1.0 + self.init_ppo_clip
            )
        ).float().detach()

        return -torch.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean() * 100.0

    @staticmethod
    def adaptive_x0_logprob(
        pred_x0: Tensor,
        target_x0: Tensor,
    ) -> Tensor:
        error = pred_x0 - target_x0

        l1_scale = error.abs().mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(
            1e-5
        ).detach()

        return -0.1 * (
            error.square()
            / l1_scale
        ).mean(dim=-1)

    # ==================================================================
    # Multi-branch sampling
    # ==================================================================
    def _choose_branch_steps(
        self,
        num_graphs: int,
        total_steps: int,
        device: torch.device,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ],
    ) -> Tensor:
        if total_steps <= 0:
            raise ValueError(
                "steps must be positive."
            )

        if (
            branch_steps is None
            and self.fixed_branch_steps is not None
        ):
            branch_steps = self.fixed_branch_steps

        if torch.is_tensor(branch_steps):
            if branch_steps.ndim == 0:
                branch_steps = int(
                    branch_steps.item()
                )
            else:
                branch_steps = (
                    branch_steps.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

        if (
            isinstance(branch_steps, Sequence)
            and not isinstance(
                branch_steps,
                (str, bytes),
            )
        ):
            selected = sorted(
                {
                    int(step)
                    for step in branch_steps
                }
            )

            if not selected:
                raise ValueError(
                    "branch_steps cannot be empty."
                )

            if (
                selected[0] < 0
                or selected[-1] >= total_steps
            ):
                raise ValueError(
                    "branch step indices must be in "
                    f"[0, {total_steps - 1}], "
                    f"got {selected}."
                )

            return torch.tensor(
                selected,
                device=device,
                dtype=torch.long,
            )[None].expand(
                num_graphs,
                -1,
            )

        count = (
            self.num_branch_steps
            if branch_steps is None
            else int(branch_steps)
        )

        if count <= 0:
            raise ValueError(
                "The number of branch steps "
                "must be positive."
            )

        count = min(
            count,
            total_steps,
        )

        # Sampling without replacement for every scene.
        random_score = torch.rand(
            num_graphs,
            total_steps,
            device=device,
        )

        selected = random_score.argsort(
            dim=1
        )[:, :count]

        return selected.sort(
            dim=1
        ).values

    @staticmethod
    def _gather_steps(
        values: Tensor,
        step_index: Tensor,
    ) -> Tensor:
        """Gather [N, T, ...] at per-agent indices [N, B]."""
        if values.ndim < 2:
            raise ValueError(
                "values must have shape [N, T, ...]."
            )

        if (
            step_index.ndim != 2
            or step_index.shape[0] != values.shape[0]
        ):
            raise ValueError(
                "step_index must have shape [N, B]."
            )

        index = step_index

        for _ in range(values.ndim - 2):
            index = index.unsqueeze(-1)

        index = index.expand(
            *step_index.shape,
            *values.shape[2:],
        )

        return values.gather(
            dim=1,
            index=index,
        )

    @torch.no_grad()
    def _sample_step(
        self,
        latent: Tensor,
        time_scalar: Tensor,
        next_time_scalar: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        branch_mask: Optional[Tensor] = None,
    ):
        num_agents = latent.shape[0]

        base_time = torch.full(
            (num_agents, 1),
            time_scalar,
            device=latent.device,
            dtype=latent.dtype,
        )
        base_next_time = torch.full_like(base_time, next_time_scalar)

        time = self._schedule_time(base_time)
        next_time = self._schedule_time(base_next_time)

        self._fix_conditioned_agents(
            tokenized_agent["expert_input"],
            latent,
            time,
            tokenized_agent,
        )

        ego_mask = tokenized_agent["ego_mask"].bool()
        next_time[ego_mask] = 0.0

        velocity, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        next_latent = latent + (next_time - time) * velocity
        log_prob = latent.new_zeros(num_agents)
        used_noise_level = latent.new_zeros(latent.shape)

        if (
            self.use_sde
            and branch_mask is not None
            and branch_mask.any()
           #and "gt_z_raw" not in tokenized_agent
        ):
            noise_level = self.get_adaptive_noise_level(time, next_time)
            noise_level = noise_level * branch_mask[:, None].to(latent.dtype)

            stochastic = branch_mask.bool() & (~ego_mask)
            stochastic &= noise_level.amax(dim=-1) > 0

            if torch.any(stochastic):
                (
                    stochastic_next,
                    stochastic_log_prob,
                    _,
                    _,
                ) = self.sde_step_with_logprob(
                    time=time[stochastic],
                    next_time=next_time[stochastic],
                    model_output=velocity[stochastic],
                    sample=latent[stochastic],
                    noise_level=noise_level[stochastic],
                )
                # (
                #     stochastic_next1,
                #     stochastic_log_prob1,
                #     new_next_mean,
                #     new_std,
                # ) = self.sde_step_with_logprob1(
                #     time=time[stochastic],
                #     next_time=next_time[stochastic],
                #     model_output=velocity[stochastic],
                #     sample=latent[stochastic],
                #     noise_level=noise_level[stochastic],
                #
                #     # 关键：使用完全相同的 transition sample
                #     prev_sample=stochastic_next.detach(),
                # )

                next_latent[stochastic] = stochastic_next
                log_prob[stochastic] = stochastic_log_prob

                # Expand [N,1] scheduled noise to [N,D] for replay storage.
                used_noise_level[stochastic] = noise_level[stochastic]

        return (
            next_latent,
            x0,
            time,
            next_time,
            log_prob,
            used_noise_level,
        )

    @torch.no_grad()
    def sample(
        self,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, Tensor],
        steps: int = 20,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ] = None,
    ) -> Tensor:

        agent_batch = tokenized_agent[
            "batch"
        ].long()

        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        num_agents = agent_batch.numel()

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()

        latent = torch.randn(
            num_agents,
            self.model.m_delta_dim,
            device=agent_batch.device,
            dtype=self.model.normal_scale.dtype,
        )

        latent = self.model.denormalize(
            latent
        )

        if "expert_input" not in tokenized_agent:
            expert_input, _ = self.model.get_input(
                tokenized_agent
            )
            tokenized_agent[
                "expert_input"
            ] = expert_input

        # Generation: start from x1~noise at t=1 and integrate to x0 at t=0.
        timesteps = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=agent_batch.device,
            dtype=latent.dtype,
        )

        if self.use_sde:
            graph_branch_steps = (
                self._choose_branch_steps(
                    num_graphs,
                    steps,
                    agent_batch.device,
                    branch_steps,
                )
            )

            agent_branch_steps = (
                graph_branch_steps[
                    agent_batch
                ]
            )

            step_branch_mask = torch.zeros(
                num_agents,
                steps,
                device=latent.device,
                dtype=torch.bool,
            )
            step_branch_mask.scatter_(
                dim=1,
                index=agent_branch_steps,
                value=True,
            )
            step_branch_mask[ego_mask] = False

            latent_history = [
                latent.clone()
            ]
            time_history = []
            next_time_history = []
            log_prob_history = []
            noise_level_history = []
            feature_history = []

        # Rollout sampling should not use dropout. This also ensures that
        # denoisers which expose noise_feat_cur only in eval mode work.
        model_was_training = self.model.training
        self.model.eval()

        try:
            for step in range(steps):
                (
                    latent,
                    _,
                    time,
                    next_time,
                    log_prob,
                    used_noise_level,
                ) = self._sample_step(
                    latent,
                    timesteps[step],
                    timesteps[step + 1],
                    tokenized_agent,
                    initial_map_feature,
                    branch_mask=(
                        step_branch_mask[:, step]
                        if self.use_sde
                        else None
                    ),
                )

                if self.use_sde:
                    latent_history.append(
                        latent.clone()
                    )
                    time_history.append(time)
                    next_time_history.append(
                        next_time
                    )
                    log_prob_history.append(
                        log_prob
                    )
                    noise_level_history.append(
                        used_noise_level
                    )

                    feature_history.append(
                        tokenized_agent[
                            "noise_feat_cur"
                        ].clone()
                    )

                elif (
                    step == 0
                    and "noise_feat_cur"
                    in tokenized_agent
                ):
                    tokenized_agent[
                        "noise_feat"
                    ] = tokenized_agent[
                        "noise_feat_cur"
                    ]

        finally:
            self.model.train(
                model_was_training
            )

        latent[
            ego_mask
        ] = tokenized_agent[
            "expert_input"
        ][ego_mask]

        if not self.use_sde:
            tokenized_agent["gen_z"] = latent
            return latent

        latent_stack = torch.stack(
            latent_history,
            dim=1,
        )

        time_stack = torch.stack(
            time_history,
            dim=1,
        )

        next_time_stack = torch.stack(
            next_time_history,
            dim=1,
        )

        log_prob_stack = torch.stack(
            log_prob_history,
            dim=1,
        )

        noise_level_stack = torch.stack(
            noise_level_history,
            dim=1,
        )

        feature_stack = torch.stack(
            feature_history,
            dim=1,
        )

        selected_current = self._gather_steps(
            latent_stack[:, :-1],
            agent_branch_steps,
        )

        selected_next = self._gather_steps(
            latent_stack[:, 1:],
            agent_branch_steps,
        )

        selected_time = self._gather_steps(
            time_stack,
            agent_branch_steps,
        )

        selected_next_time = self._gather_steps(
            next_time_stack,
            agent_branch_steps,
        )

        selected_log_prob = self._gather_steps(
            log_prob_stack,
            agent_branch_steps,
        )

        selected_noise_level = self._gather_steps(
            noise_level_stack,
            agent_branch_steps,
        )

        selected_features = self._gather_steps(
            feature_stack,
            agent_branch_steps,
        )

        tokenized_agent["sde_z"] = (
            selected_current,
            selected_next,
            selected_log_prob.detach(),
        )

        tokenized_agent["sde_t"] = (
            selected_time,
            selected_next_time,
        )

        tokenized_agent[ "sde_noise_level"] = selected_noise_level.detach()

        tokenized_agent["noise_feat"] = selected_features

        return latent

    def sde_step_with_logprob(
            self,
            time: Tensor,
            next_time: Tensor,
            model_output: Tensor,
            sample: Tensor,
            noise_level=0.7,
            prev_sample: Optional[Tensor] = None,
    ):
        """SDE transition directly in raw data space."""

        eps = 1e-5

        scale = self.model.normal_scale.to(
            device=sample.device,
            dtype=sample.dtype,
        ).clamp_min(eps)

        mean = self.model.normal_mean.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        dt = next_time - time

        time_for_ratio = torch.where(
            time >= 1.0,
            torch.full_like(time, 0.9),
            time,
        )

        diffusion = (
                torch.sqrt(
                    time.clamp_min(0.0)
                    / (1.0 - time_for_ratio).clamp_min(eps)
                )
                * 1#noise_level
        )
        #diffusion=0.05/torch.sqrt(-dt)

        safe_time = time.clamp_min(eps)

        # Normalized-space transition coefficients.
        A = (
                1.0
                + diffusion.square()
                / (2.0 * safe_time)
                * dt
        )

        B = (
                    1.0
                    + diffusion.square()
                    * (1.0 - time)
                    / (2.0 * safe_time)
            ) * dt

        # Equivalent mean directly in raw space.
        next_mean = (
                mean
                + A * (sample - mean)
                + B * model_output
        )

        # Raw-space transition std.
        transition_std = (
                diffusion
                * torch.sqrt((-dt).clamp_min(eps))
                * scale
        ).clamp_min(eps)

        # Sample transition.
        if prev_sample is None:
            next_sample = (
                    next_mean
                    + transition_std
                    * torch.randn_like(sample)
            )
        else:
            next_sample = prev_sample

        # Raw-space Gaussian log probability.
        residual = (
                next_sample.detach()
                - next_mean
        )

        log_prob_element = (
                -0.5
                * (
                        residual.square()
                        / transition_std.square()
                        + 2.0 * torch.log(transition_std)
                        + math.log(2.0 * math.pi)
                )
        )

        log_prob = log_prob_element.sum(
            dim=tuple(range(1, log_prob_element.ndim))
        )

        return (
            next_sample,
            log_prob,
            next_mean,
            transition_std,
        )

    def repeat_input_copy(self, tokenized_agent, n_step):
        out = copy.copy(tokenized_agent)

        num_graphs = tokenized_agent["num_graphs"]
        batch = tokenized_agent["batch"]

        out["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step)

        repeated_batch = torch.stack(
            [batch + num_graphs * k for k in range(n_step)],
            dim=1,
        ).transpose(0, 1).flatten(0, 1)

        out["batch"] = repeated_batch
        out["agent_type_embed"] = tokenized_agent["agent_type_embed"][None].repeat(
            n_step, 1,1
        ).flatten(0, 1)
        out["num_graphs"] = num_graphs * n_step

        out["ego_feat"] = tokenized_agent["ego_feat"][None].repeat(
            n_step, 1, 1
        ).flatten(0, 1)

        return out

    def precise_step_with_logprob(
        self,
        time: Tensor,
        next_time: Tensor,
        pred_x0: Tensor,
        sample: Tensor,
        eta=0.2,
        prev_sample: Optional[Tensor] = None,
    ):
        """Exact finite-step Gaussian transition in Rectified-Flow time.

        Convention:
            x_t = (1 - t) * x0 + t * x1

        Generation direction:
            next_time <= time, with t decreasing from 1 to 0.
        """
        eps = 1e-6

        sample_normalized = self.model.normalize(sample)
        x0_normalized = self.model.normalize(pred_x0)

        next_normalized = (
            None
            if prev_sample is None
            else self.model.normalize(prev_sample)
        )

        time = torch.as_tensor(
            time,
            device=sample.device,
            dtype=sample.dtype,
        )

        next_time = torch.as_tensor(
            next_time,
            device=sample.device,
            dtype=sample.dtype,
        )

        eta = torch.as_tensor(
            eta,
            device=sample.device,
            dtype=sample.dtype,
        )

        if torch.any(next_time > time + 1e-7):
            raise ValueError(
                "Expected next_time <= time."
            )

        safe_time = time.clamp_min(eps)

        # x1_hat implied by current x_t and predicted x0:
        # x_t = (1-t)x0 + t*x1.
        x1_hat = (
            sample_normalized
            - (1.0 - time) * x0_normalized
        ) / safe_time

        # rho = [ next_t(1-t) / (t(1-next_t)) ]^(eta^2/2)
        numerator = (
            next_time.clamp_min(0.0)
            * (1.0 - time).clamp_min(0.0)
        )

        denominator = (
            time.clamp_min(eps)
            * (1.0 - next_time).clamp_min(eps)
        )

        base_ratio = (
            numerator / denominator
        ).clamp(
            min=0.0,
            max=1.0,
        )

        log_base = torch.log(
            base_ratio.clamp_min(1e-12)
        )

        rho = torch.exp(
            0.5 * eta.square() * log_base
        )

        rho = torch.where(
            numerator <= eps,
            torch.zeros_like(rho),
            rho,
        )

        # eta=0 recovers deterministic Rectified Flow.
        rho = torch.where(
            eta.abs() <= eps,
            torch.ones_like(rho),
            rho,
        )

        rho = rho.clamp(0.0, 1.0)

        next_mean = (
            (1.0 - next_time) * x0_normalized
            + next_time * rho * x1_hat
        )

        transition_std = (
            next_time
            * torch.sqrt(
                (1.0 - rho.square()).clamp_min(0.0)
            )
        )

        if next_normalized is None:
            next_normalized = (
                next_mean
                + transition_std
                * torch.randn_like(next_mean)
            )

        safe_std = transition_std.clamp_min(eps)

        log_prob_element = (
            -0.5
            * (
                (
                    next_normalized.detach()
                    - next_mean
                )
                / safe_std
            ).square()
            - torch.log(safe_std)
            - 0.5 * math.log(2.0 * math.pi)
        )

        log_prob = log_prob_element.mean(
            dim=tuple(
                range(
                    1,
                    log_prob_element.ndim,
                )
            )
        )

        next_sample = self.model.denormalize(
            next_normalized
        )

        return (
            next_sample,
            log_prob,
            next_mean,
            transition_std,
        )

