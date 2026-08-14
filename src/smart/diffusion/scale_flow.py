"""Advantage-weighted initial-state flow matching.

State convention:
    [x, y, heading_cos, heading_sin, length, width, vx, vy]

Flow convention:
    z_t = (1 - t) * noise + t * data
    velocity = data - noise

For the equivalent noisy-time convention

    sigma = 1 - t,
    z_sigma = (1 - sigma) * data + sigma * noise,

the stochastic sampler freezes the predicted clean state during one step and
integrates the resulting reverse SDE in closed form.  ``sde_eta`` controls the
log-SNR exploration schedule.

The endpoints are deterministic: stochastic branches are restricted to
interior flow steps, so the transition into the clean state has zero variance.

Sampling and training are deliberately decoupled.  SDE branches provide
exploration, while the generated clean endpoint is optimized with the same
clean-target flow-matching objective used during pretraining, weighted by its
initial-state advantage.  No reverse-transition log-probability or Initial
SDE-PPO ratio is used.
"""

from __future__ import annotations

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
from .denoiser import InitDenoiser


class ScaleFlow(nn.Module):
    """Initial-state AWM with an SDE-consistent exploration sampler."""

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
        # Initial-state Advantage-Weighted Flow Matching
        # --------------------------------------------------------------
        self.use_sde = bool(
            gail and getattr(token_processor, "learn_init", False)
        )

        self.init_adv_clip = float(
            getattr(args, "init_adv_clip", 5.0)
        )
        if self.init_adv_clip < 0.0:
            raise ValueError("init_adv_clip must be non-negative.")

        self.awm_coef = float(
            getattr(args, "awm_coef", 0.1)
        )
        if self.awm_coef < 0.0:
            raise ValueError("awm_coef must be non-negative.")

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

        # Exploration coefficient in
        #   eps(sigma) = eta * sqrt(sigma / (1 - sigma)).
        # ``target_step_std`` is accepted as a backward-compatible fallback,
        # but ``sde_eta`` is the preferred configuration name.
        self.sde_eta = float(
            getattr(
                args,
                "sde_eta",
                getattr(args, "target_step_std", 0.1),
            )
        )
        if self.sde_eta < 0.0:
            raise ValueError("sde_eta must be non-negative.")

        # The log-SNR schedule is singular at pure noise and the clean state.
        # Those endpoint transitions are therefore kept deterministic.
        self.sde_endpoint_eps = float(
            getattr(args, "sde_endpoint_eps", 1e-5)
        )
        if self.sde_endpoint_eps <= 0.0:
            raise ValueError("sde_endpoint_eps must be positive.")

        # self.register_buffer(
        #     "noise_dim_weights",
        #     torch.zeros(
        #         [8],
        #         dtype=torch.float32,
        #     )[None],
        # )
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
        normalized_x = self.model.normalize(x)
        noise = torch.randn_like(normalized_x)

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()
        noise[ego_mask] = normalized_x[ego_mask]

        matched_index = get_closest_sum_idx_fast(
            noise,
            normalized_x,
            tokenized_agent,
            all_state=True,
        )

        return self.model.denormalize(
            noise[matched_index]
        )

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

        return self._schedule_time(
            scene_time
        )[batch]

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
        time[ego_mask] = 1.0

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

        latent = (
            (1.0 - time) * noise
            + time * x
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

        velocity = (
            x0 - latent
        ) / (
            1.0 - time
        ).clamp_min(
            self.t_eps
        )

        return velocity, x0

    def get_adaptive_noise_level(
        self,
        time: Tensor,
        next_time: Tensor,
    ) -> Tensor:
        """Return the eta used by the closed-form reverse-SDE step.

        ``time`` follows the code convention (0=noise, 1=data).  Stochastic
        exploration is disabled at both endpoints and for zero-length steps.
        The actual finite-step variance is computed analytically inside
        :meth:`_sde_consistent_step`; eta is not an Euler noise multiplier.
        """
        if time.shape != next_time.shape:
            raise ValueError(
                "time and next_time must have the same shape."
            )

        eps = self.sde_endpoint_eps
        delta_t = next_time - time

        if torch.any(delta_t < -eps):
            raise ValueError(
                "Expected non-decreasing flow time."
            )

        sigma = 1.0 - time
        sigma_prev = 1.0 - next_time

        interior = (
            (delta_t > eps)
            & (sigma < 1.0 - eps)
            & (sigma_prev > eps)
        )

        eta = torch.full_like(time, self.sde_eta)
        return torch.where(
            interior,
            eta,
            torch.zeros_like(eta),
        )

    def get_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, Tensor],
    ):
        loss = self._supervised_loss(
            x,
            tokenized_agent,
            initial_map_feature,
        )

        if "advantages" in tokenized_agent:
            tokenized_agent["rl_loss"] = (
                self._advantage_weighted_flow_loss(
                    tokenized_agent,
                    initial_map_feature,
                )
            )

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
            use_col=True,
            x_pred=True,
        )

        return loss, x0, latent, time

    def _aggregate_initial_advantages(
        self,
        advantages: Tensor,
        num_agents: int,
    ) -> Tensor:
        """Reduce rollout/branch advantages to one value per agent."""
        advantages = advantages.detach()

        if advantages.ndim == 0:
            advantages = advantages.expand(num_agents)
        elif advantages.ndim == 1:
            if advantages.numel() != num_agents:
                raise ValueError(
                    "One-dimensional advantages must have one value "
                    f"per agent; got {advantages.numel()} for "
                    f"{num_agents} agents."
                )
        elif advantages.shape[-1] == num_agents:
            # Rollout advantages convention: [branch/time, agent].
            advantages = advantages.reshape(-1, num_agents).mean(dim=0)
        elif advantages.shape[0] == num_agents:
            advantages = advantages.reshape(num_agents, -1).mean(dim=1)
        else:
            raise ValueError(
                "advantages must contain an agent axis of length "
                f"{num_agents}, got {tuple(advantages.shape)}."
            )

        return torch.nan_to_num(
            advantages,
            nan=0.0,
            posinf=self.init_adv_clip,
            neginf=-self.init_adv_clip,
        ).clamp(
            -self.init_adv_clip,
            self.init_adv_clip,
        )

    def _advantage_weighted_flow_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        """Match sampled clean endpoints with signed advantage weights.

        A fresh forward-flow time and noise are sampled for every update.
        Positive advantages pull the denoiser toward the generated endpoint;
        negative advantages push it away. The endpoint and advantage are
        treated as fixed targets.
        """
        if "gen_z" not in tokenized_agent:
            raise KeyError(
                "Advantage-weighted flow matching requires sample() to "
                "store tokenized_agent['gen_z'] before get_loss()."
            )

        sampled_x0 = tokenized_agent["gen_z"].detach()
        _, time, latent = self._prepare_supervised_batch(
            sampled_x0,
            tokenized_agent,
        )

        _, predicted_x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        scale = self.model.normal_scale.to(
            device=sampled_x0.device,
            dtype=sampled_x0.dtype,
        ).clamp_min(1e-6)

        matching_loss = (
            (predicted_x0 - sampled_x0)
            / scale
        ).square().mean(dim=-1)

        advantages = self._aggregate_initial_advantages(
            tokenized_agent["advantages"],
            sampled_x0.shape[0],
        ).to(
            device=sampled_x0.device,
            dtype=sampled_x0.dtype,
        )

        valid = (
            ~tokenized_agent["ego_mask"].bool()
            & torch.isfinite(matching_loss)
            & torch.isfinite(advantages)
        )

        if not torch.any(valid):
            return sampled_x0.new_zeros(())

        valid_advantage = advantages[valid]
        valid_matching_loss = matching_loss[valid]

        # Same on-policy gradient as the AWM pseudo-likelihood ratio
        # exp(-L + stopgrad(L)), without materializing an always-one ratio.
        weighted_loss = (
            valid_advantage.detach()
            * valid_matching_loss
        ).mean()

        tokenized_agent["sampled_match_loss"] = matching_loss.detach()
        tokenized_agent["awm_matching_loss"] = (
            valid_matching_loss.detach().mean()
        )
        tokenized_agent["awm_weighted_matching_loss"] = (
            weighted_loss.detach()
        )
        tokenized_agent["awm_advantage_mean"] = (
            valid_advantage.detach().mean()
        )
        tokenized_agent["awm_positive_fraction"] = (
            (valid_advantage > 0.0).float().mean().detach()
        )

        return self.awm_coef * weighted_loss

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

        # The log-SNR SDE is singular at pure noise and has zero transition
        # variance at the clean endpoint.  Only interior transitions define
        # valid stochastic exploration steps.
        if total_steps < 3:
            raise ValueError(
                "SDE-consistent branching requires at least 3 flow steps."
            )

        first_valid_step = 1
        last_valid_step = total_steps - 2

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
                selected[0] < first_valid_step
                or selected[-1] > last_valid_step
            ):
                raise ValueError(
                    "SDE-consistent branch step indices must be interior: "
                    f"[{first_valid_step}, {last_valid_step}], "
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

        num_valid_steps = last_valid_step - first_valid_step + 1
        count = min(count, num_valid_steps)

        # Sampling without replacement for every scene.
        random_score = torch.rand(
            num_graphs,
            num_valid_steps,
            device=device,
        )

        selected = random_score.argsort(
            dim=1
        )[:, :count] + first_valid_step

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
        next_time[ego_mask] = 1.0

        velocity, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        next_latent = latent + (next_time - time) * velocity
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
                    _,
                    _,
                ) = self._sde_consistent_step(
                    sigma=1.0 - time[stochastic],
                    sigma_prev=1.0 - next_time[stochastic],
                    model_output=-velocity[stochastic],
                    sample=latent[stochastic],
                    noise_level=noise_level[stochastic],
                )

                next_latent[stochastic] = stochastic_next

        return (
            next_latent,
            x0,
            time,
            next_time,
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

        timesteps = torch.linspace(
            0.0,
            1.0,
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
                    _,
                    _,
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

        # AWM trains from the final clean endpoint irrespective of sampler.
        tokenized_agent["gen_z"] = latent.detach()

        if not self.use_sde:
            return latent

        feature_stack = torch.stack(
            feature_history,
            dim=1,
        )

        selected_features = self._gather_steps(
            feature_stack,
            agent_branch_steps,
        )
        tokenized_agent["noise_feat"] = selected_features

        return latent

    # ==================================================================
    # SDE-consistent finite transition
    # ==================================================================
    def _sde_consistent_step(
        self,
        sigma: Tensor,
        sigma_prev: Tensor,
        model_output: Tensor,
        sample: Tensor,
        noise_level=0.1,
    ):
        """Apply a frozen-clean-state, SDE-consistent finite transition.

        Args:
            sigma: Current noisy time, 1 - flow_time.
            sigma_prev: Next noisy time; must not exceed sigma.
            model_output: Flow velocity in noisy-time convention
                (noise - data), in physical state units.
            sample: Current state in physical units.
            noise_level: The log-SNR exploration coefficient eta in
                eps(s) = eta * sqrt(s / (1 - s)). This may be
                dimension-dependent. Zero gives a deterministic transition.

        Under a locally frozen clean-state posterior mean, the exact step is

            z_{s'} = (1-s') x0_hat
                     + s' rho eps_hat
                     + s' sqrt(1-rho^2) w,

        where

            rho = [s'(1-s) / (s(1-s'))] ** (eta^2 / 2).

        This avoids the excess finite-step variance introduced by
        Euler--Maruyama discretization of a flow-matching reverse SDE.
        """
        eps = self.sde_endpoint_eps

        scale = self.model.normal_scale.to(
            device=sample.device,
            dtype=sample.dtype,
        ).clamp_min(eps)

        model_output = model_output / scale
        sample_normalized = self.model.normalize(sample)

        eta = torch.as_tensor(
            noise_level,
            device=sample.device,
            dtype=sample.dtype,
        ).clamp_min(0.0)

        sigma = torch.as_tensor(
            sigma,
            device=sample.device,
            dtype=sample.dtype,
        ).clamp(0.0, 1.0)

        sigma_prev = torch.as_tensor(
            sigma_prev,
            device=sample.device,
            dtype=sample.dtype,
        ).clamp(0.0, 1.0)

        if torch.any(sigma_prev - sigma > eps):
            raise ValueError(
                "Expected non-increasing sigma, "
                "but sigma_prev > sigma."
            )

        # For z_sigma=(1-sigma)*x0+sigma*epsilon, recover the
        # locally frozen clean and noise predictions.
        predicted_x0 = (
            sample_normalized
            - sigma * model_output
        )
        estimated_noise = (
            sample_normalized
            + (1.0 - sigma) * model_output
        )

        # Compute rho in log-space for stable small finite steps.
        log_base_ratio = (
            torch.log(sigma_prev.clamp_min(eps))
            + torch.log((1.0 - sigma).clamp_min(eps))
            - torch.log(sigma.clamp_min(eps))
            - torch.log((1.0 - sigma_prev).clamp_min(eps))
        ).clamp_max(0.0)

        log_rho = 0.5 * eta.square() * log_base_ratio
        rho = torch.exp(log_rho).clamp(0.0, 1.0)

        active_step = (sigma - sigma_prev) > eps
        stochastic_dim = (
            active_step
            & (eta > 0.0)
            & (sigma < 1.0 - eps)
            & (sigma_prev > eps)
        )

        # eta=0 and both endpoints use the deterministic probability flow.
        rho = torch.where(
            stochastic_dim,
            rho,
            torch.ones_like(rho),
        )

        previous_mean = (
            (1.0 - sigma_prev) * predicted_x0
            + sigma_prev * rho * estimated_noise
        )
        transition_std = (
            sigma_prev
            * torch.sqrt(
                (1.0 - rho.square()).clamp_min(0.0)
            )
        )
        previous_normalized = (
            previous_mean
            + transition_std
            * torch.randn_like(model_output)
        )
        previous_sample = self.model.denormalize(
            previous_normalized
        )

        return (
            previous_sample,
            previous_mean,
            transition_std,
        )
