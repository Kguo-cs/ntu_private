"""Initial-state flow model with multi-branch SDE sampling.

MeanFlow/iMF support has been removed. The model now uses standard flow
matching only:

    z_t = (1 - t) * noise + t * data
    v_gt = data - noise

During SDE sampling, each scene may select several unique branch timesteps.
The selected transitions are stored as [agent, branch, ...] so advantage
training can use every branch without duplicating agent metadata.
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

from .denoiser import InitDenoiser


class ScaleFlow(nn.Module):
    """Initial-state flow model with optional SDE advantage finetuning."""

    def __init__(self, args, token_processor, gail: bool = False):
        super().__init__()

        self.hidden_dim = int(args.hidden_dim)
        self.use_dit = bool(getattr(args, "use_dit", False))

        if self.use_dit:
            from src.smart.diffusion.dit.dit import DiT

            self.x_pred = False
            self.model = DiT(self.hidden_dim)
        else:
            # Standard flow only. The denoiser predicts x0 by default.
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
                x_pred=self.x_pred,
            )

        self.use_all_type = self.model.use_all_type
        self.t_eps = 0.05 if self.x_pred else 0.0

        self.use_sde = bool(
            gail and getattr(token_processor, "learn_init", False)
        )
        self.noise_level = float(getattr(args, "noise_level", 0.7))
        self.use_init_ppo_ratio = bool(
            getattr(args, "use_init_ppo_ratio", False)
        )
        self.init_adv_clip = float(getattr(args, "init_adv_clip", 3.0))
        self.init_logprob_clip = float(
            getattr(args, "init_logprob_clip", 50.0)
        )
        self.init_ppo_clip = float(getattr(args, "init_ppo_clip", 0.2))

        # Number of random, unique SDE branch timesteps selected per scene.
        self.num_branch_steps = int(
            getattr(args, "num_branch_steps", 1)
        )
        if self.num_branch_steps <= 0:
            raise ValueError("num_branch_steps must be positive.")

        configured_steps = getattr(args, "branch_steps", None)
        self.fixed_branch_steps = self._parse_fixed_branch_steps(
            configured_steps
        )

        self.use_ref = False
        self.apply(weight_init)

    # ------------------------------------------------------------------
    # Flow helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_fixed_branch_steps(value) -> Optional[tuple[int, ...]]:
        if value is None:
            return None

        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()

        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(
                "branch_steps must be a sequence of integer timestep indices."
            )

        steps = tuple(sorted({int(step) for step in value}))
        if not steps:
            raise ValueError("branch_steps cannot be empty.")
        return steps

    def _sample_noise(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        """Sample matched Gaussian noise and keep conditioned agents clean."""
        normalized_x = self.model.normalize(x)
        noise = torch.randn_like(normalized_x)

        ego_mask = tokenized_agent["ego_mask"]
        noise[ego_mask] = normalized_x[ego_mask]

        fake_idx = get_closest_sum_idx_fast(
            noise,
            normalized_x,
            tokenized_agent,
            all_state=True,
        )
        return self.model.denormalize(noise[fake_idx])

    def _sample_time(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        """Sample one flow time per scene and broadcast it to its agents."""
        batch = tokenized_agent["batch"].long()
        num_graphs = int(tokenized_agent["num_graphs"])

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
        """Keep ego/conditioned agents on a constant clean trajectory."""
        ego_mask = tokenized_agent["ego_mask"]
        latent[ego_mask] = clean[ego_mask]
        time[ego_mask] = 1.0

    def _prepare_supervised_batch(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        noise = self._sample_noise(x, tokenized_agent)
        time = self._sample_time(x, tokenized_agent)
        self._fix_conditioned_agents(x, noise, time, tokenized_agent)
        latent = (1.0 - time) * noise + time * x
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
        """Return flow velocity and the corresponding x0 estimate."""
        prediction = self.model(
            latent,
            time,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )

        if self.x_pred:
            x0 = prediction
            velocity = (
                x0 - latent
            ) / (1.0 - time).clamp_min(self.t_eps)
        else:
            velocity = prediction
            x0 = latent + (1.0 - time) * velocity

        return velocity, x0

    def _prepare_advantages(
        self,
        advantages: Tensor,
        num_agents: int,
    ) -> Tensor:
        """Return finite, detached, clipped per-agent advantages."""
        if advantages.ndim > 1:
            advantages = advantages.reshape(num_agents, -1).mean(dim=-1)

        return advantages.clamp(
            -self.init_adv_clip,
            self.init_adv_clip,
        )

    # ------------------------------------------------------------------
    # Supervised and advantage losses
    # ------------------------------------------------------------------
    def get_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, Tensor],
    ):
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

        return self._supervised_loss(
            x,
            tokenized_agent,
            initial_map_feature,
        )

    def _supervised_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):
        noise, time, latent = self._prepare_supervised_batch(
            x,
            tokenized_agent,
        )
        _, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        ego_mask = tokenized_agent["ego_mask"][:, None]
        x0 = torch.where(ego_mask, x.detach(), x0)

        loss = get_diff_loss(
            tokenized_agent,
            x0,
            x,
            time,
            self.t_eps,
            use_col=True,
            x_pred=self.x_pred,
        )
        return loss, x0, latent, time

    @staticmethod
    def _ensure_branch_dimension(
        current: Tensor,
        next_sample: Tensor,
        old_log_prob: Optional[Tensor],
        time: Tensor,
        next_time: Tensor,
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """Accept both old [N,...] and new [N,B,...] SDE storage."""
        if current.ndim == 2:
            current = current[:, None]
            next_sample = next_sample[:, None]
            time = time[:, None]
            next_time = next_time[:, None]
            if old_log_prob is not None:
                old_log_prob = old_log_prob[:, None]

        if current.ndim != 3:
            raise ValueError(
                "Stored SDE states must have shape [N,D] or [N,B,D]."
            )
        if next_sample.shape != current.shape:
            raise ValueError("Current and next SDE states must have the same shape.")
        if time.shape[:2] != current.shape[:2] or time.shape[-1] != 1:
            raise ValueError("SDE time must have shape [N,B,1].")
        if next_time.shape != time.shape:
            raise ValueError("Current and next SDE times must have the same shape.")
        if old_log_prob is not None and old_log_prob.shape != current.shape[:2]:
            raise ValueError("old_log_prob must have shape [N,B].")

        return current, next_sample, old_log_prob, time, next_time

    def _sde_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        current, next_sample, old_log_prob = tokenized_agent["sde_z"]
        time, next_time = tokenized_agent["sde_t"]

        (
            current,
            next_sample,
            old_log_prob,
            time,
            next_time,
        ) = self._ensure_branch_dimension(
            current,
            next_sample,
            old_log_prob,
            time,
            next_time,
        )

        num_agents, num_branches, _ = current.shape
        ego_mask = tokenized_agent["ego_mask"]
        non_ego = ~ego_mask

        if not torch.any(non_ego):
            return current.new_zeros(())

        branch_log_prob = []
        for branch in range(num_branches):
            velocity, _ = self._model_velocity(
                current[:, branch],
                time[:, branch],
                tokenized_agent,
                map_feature,
            )

            _, log_prob, _, _ = self.sde_step_with_logprob(
                sigma=1.0 - time[non_ego, branch],
                sigma_prev=1.0 - next_time[non_ego, branch],
                model_output=-velocity[non_ego],
                sample=current[non_ego, branch],
                noise_level=self.noise_level,
                prev_sample=next_sample[non_ego, branch],
            )
            branch_log_prob.append(log_prob)

        # [num_non_ego, num_branches]
        log_prob = torch.stack(branch_log_prob, dim=1)
        advantages = self._prepare_advantages(
            tokenized_agent["advantages"],
            num_agents,
        )[non_ego, None].expand_as(log_prob)

        if self.use_init_ppo_ratio and old_log_prob is not None:
            old_lp = old_log_prob[non_ego].detach()
            old_lp = torch.nan_to_num(
                old_lp,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(
                -self.init_logprob_clip,
                self.init_logprob_clip,
            )

            log_ratio = (log_prob - old_lp).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            clipped_ratio = ratio.clamp(
                1.0 - self.init_ppo_clip,
                1.0 + self.init_ppo_clip,
            )
            loss = -torch.minimum(
                ratio * advantages,
                clipped_ratio * advantages,
            ).mean()

            tokenized_agent["init_ratio_mean"] = ratio.detach().mean()
            tokenized_agent["init_clip_fraction"] = (
                (ratio < 1.0 - self.init_ppo_clip)
                | (ratio > 1.0 + self.init_ppo_clip)
            ).float().mean().detach()
            return loss

        return -(log_prob * advantages).mean()

    def _direct_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        sampled_x0 = tokenized_agent["gen_z"]
        _, time, latent = self._prepare_supervised_batch(
            sampled_x0,
            tokenized_agent,
        )
        _, current_x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        scale = self.model.normal_scale.clamp_min(1e-6)
        log_prob = self.adaptive_x0_logprob(
            current_x0 / scale,
            sampled_x0.detach() / scale,
        )

        num_agents = sampled_x0.shape[0]
        ego_mask = tokenized_agent["ego_mask"]
        non_ego = ~ego_mask
        if not torch.any(non_ego):
            return sampled_x0.new_zeros(())

        advantages = self._prepare_advantages(
            tokenized_agent["advantages"],
            num_agents,
        )[non_ego]

        # Detached-current baseline: forward ratio is one, while gradients still
        # flow through the current pseudo log-probability.
        old_log_prob = log_prob.detach()
        ratio = (log_prob - old_log_prob).clamp(-10.0, 10.0).exp()[non_ego]
        clipped_ratio = ratio.clamp(
            1.0 - self.init_ppo_clip,
            1.0 + self.init_ppo_clip,
        )

        tokenized_agent["sampled_match_loss"] = -log_prob.detach()
        tokenized_agent["clip_ratio"] = (
            (ratio < 1.0 - self.init_ppo_clip)
            | (ratio > 1.0 + self.init_ppo_clip)
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
        """Return a per-agent pseudo log-probability from normalized x0 error."""
        error = pred_x0 - target_x0
        l1_scale = error.abs().mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-5).detach()
        loss = (error.square() / l1_scale).mean(dim=-1)
        return -0.1 * loss

    # ------------------------------------------------------------------
    # Multi-branch sampling
    # ------------------------------------------------------------------
    def _choose_branch_steps(
        self,
        num_graphs: int,
        total_steps: int,
        device: torch.device,
        branch_steps: Optional[int | Sequence[int] | Tensor],
    ) -> Tensor:
        """Return branch timestep indices with shape [num_graphs, num_branches].

        ``branch_steps`` may be:
            None:
                use configured fixed steps, otherwise num_branch_steps random
                unique steps per scene;
            int:
                select this many random unique steps per scene;
            sequence/tensor:
                use these exact timestep indices for every scene.
        """
        if total_steps <= 0:
            raise ValueError("steps must be positive.")

        if branch_steps is None and self.fixed_branch_steps is not None:
            branch_steps = self.fixed_branch_steps

        if torch.is_tensor(branch_steps):
            if branch_steps.ndim == 0:
                branch_steps = int(branch_steps.item())
            else:
                branch_steps = branch_steps.detach().cpu().reshape(-1).tolist()

        if isinstance(branch_steps, Sequence) and not isinstance(
            branch_steps,
            (str, bytes),
        ):
            selected = sorted({int(step) for step in branch_steps})
            if not selected:
                raise ValueError("branch_steps cannot be empty.")
            if selected[0] < 0 or selected[-1] >= total_steps:
                raise ValueError(
                    f"branch step indices must lie in [0, {total_steps - 1}], "
                    f"got {selected}."
                )
            return torch.tensor(
                selected,
                dtype=torch.long,
                device=device,
            )[None].expand(num_graphs, -1)

        count = (
            self.num_branch_steps
            if branch_steps is None
            else int(branch_steps)
        )
        if count <= 0:
            raise ValueError("The number of branch steps must be positive.")
        count = min(count, total_steps)

        # argsort of random scores samples without replacement per scene.
        random_score = torch.rand(
            num_graphs,
            total_steps,
            device=device,
        )
        selected = random_score.argsort(dim=1)[:, :count]
        return selected.sort(dim=1).values

    @staticmethod
    def _gather_steps(values: Tensor, step_index: Tensor) -> Tensor:
        """Gather [N,T,...] values at per-agent indices [N,B]."""
        if values.ndim < 2:
            raise ValueError("values must have shape [N,T,...].")
        if step_index.ndim != 2 or step_index.shape[0] != values.shape[0]:
            raise ValueError(
                "step_index must have shape [N,B] and match values."
            )

        index = step_index
        for _ in range(values.ndim - 2):
            index = index.unsqueeze(-1)

        index = index.expand(
            *step_index.shape,
            *values.shape[2:],
        )
        return values.gather(dim=1, index=index)

    @torch.no_grad()
    def _sample_step(
        self,
        latent: Tensor,
        time_scalar: Tensor,
        next_time_scalar: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        noise_level: Optional[Tensor] = None,
    ):
        num_agents = latent.shape[0]
        time = torch.ones(
            (num_agents, 1),
            device=latent.device,
            dtype=latent.dtype,
        ) * time_scalar
        next_time = torch.ones_like(time) * next_time_scalar

        self._fix_conditioned_agents(
            tokenized_agent["expert_input"],
            latent,
            time,
            tokenized_agent,
        )
        next_time[tokenized_agent["ego_mask"]] = 1.0

        velocity, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        next_latent = latent + (next_time - time) * velocity
        log_prob = latent.new_zeros(num_agents)

        if self.use_sde and noise_level is not None and "gt_z_raw" not in tokenized_agent:
            noise_level = noise_level.reshape(num_agents, -1)
            stochastic = noise_level.amax(dim=-1) > 0
            stochastic &= ~tokenized_agent["ego_mask"]

            if torch.any(stochastic):
                stochastic_next, stochastic_log_prob, _, _ = (
                    self.sde_step_with_logprob(
                        sigma=1.0 - time[stochastic],
                        sigma_prev=1.0 - next_time[stochastic],
                        model_output=-velocity[stochastic],
                        sample=latent[stochastic],
                        noise_level=noise_level[stochastic],
                    )
                )
                next_latent[stochastic] = stochastic_next
                log_prob[stochastic] = stochastic_log_prob

        return next_latent, x0, time, next_time, log_prob

    @torch.no_grad()
    def sample(
        self,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, Tensor],
        steps: int = 20,
        branch_steps: Optional[int | Sequence[int] | Tensor] = None,
    ) -> Tensor:
        """Generate one state per agent with optional multi-branch SDE steps.

        Examples:
            sample(..., steps=20, branch_steps=4)
                Four random unique branch steps per scene.

            sample(..., steps=20, branch_steps=[3, 7, 12])
                The same three explicit branch steps in every scene.

            sample(..., steps=20)
                Uses args.branch_steps when configured, otherwise
                args.num_branch_steps random steps per scene.
        """
        if steps <= 0:
            raise ValueError("steps must be positive.")

        agent_batch = tokenized_agent["batch"].long()
        num_graphs = int(tokenized_agent["num_graphs"])
        num_agents = agent_batch.numel()
        ego_mask = tokenized_agent["ego_mask"]

        latent = torch.randn(
            num_agents,
            self.model.m_delta_dim,
            device=agent_batch.device,
            dtype=self.model.normal_scale.dtype,
        )
        latent = self.model.denormalize(latent)

        if "expert_input" not in tokenized_agent:
            expert_input, _ = self.model.get_input(tokenized_agent)
            tokenized_agent["expert_input"] = expert_input

        timesteps = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=agent_batch.device,
            dtype=latent.dtype,
        )

        if self.use_sde:
            graph_branch_steps = self._choose_branch_steps(
                num_graphs=num_graphs,
                total_steps=steps,
                device=agent_batch.device,
                branch_steps=branch_steps,
            )
            agent_branch_steps = graph_branch_steps[agent_batch]

            graph_branch_mask = torch.zeros(
                num_graphs,
                steps,
                dtype=torch.bool,
                device=agent_batch.device,
            )
            graph_branch_mask.scatter_(
                dim=1,
                index=graph_branch_steps,
                value=True,
            )
            agent_branch_mask = graph_branch_mask[agent_batch]

            step_noise_level = (
                agent_branch_mask[..., None].to(latent.dtype)
                * self.noise_level
            )
            step_noise_level[ ego_mask ] = 0.0

            latent_history = [latent.clone()]
            time_history = []
            next_time_history = []
            log_prob_history = []
            feature_history = []

        for step in range(steps):
            latent, x0, time, next_time, log_prob = self._sample_step(
                latent,
                timesteps[step],
                timesteps[step + 1],
                tokenized_agent,
                initial_map_feature,
                noise_level=(
                    step_noise_level[:, step]
                    if self.use_sde
                    else None
                ),
            )

            if self.use_sde:
                latent_history.append(latent.clone())
                time_history.append(time)
                next_time_history.append(next_time)
                log_prob_history.append(log_prob)

                feature_history.append(
                    tokenized_agent["noise_feat_cur"].clone()
                )
            elif step == 0:
                tokenized_agent["noise_feat"] = tokenized_agent[ "noise_feat_cur" ]

        latent[ego_mask] = tokenized_agent["expert_input"][ego_mask]

        if not self.use_sde:
            tokenized_agent["gen_z"] = latent
            return latent

        latent_stack = torch.stack(
            latent_history,
            dim=1,
        )  # [N, steps + 1, D]
        time_stack = torch.stack(
            time_history,
            dim=1,
        )  # [N, steps, 1]
        next_time_stack = torch.stack(
            next_time_history,
            dim=1,
        )
        log_prob_stack = torch.stack(
            log_prob_history,
            dim=1,
        )  # [N, steps]
        feature_stack = torch.stack(
            feature_history,
            dim=1,
        )  # [N, steps, ...]

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
        tokenized_agent["sde_branch_steps"] = agent_branch_steps
        tokenized_agent["sde_branch_steps_graph"] = graph_branch_steps
        tokenized_agent["sde_noise_feat"] = selected_features

        # Keep the old [N,...] value-network interface. The first selected
        # branch is the earliest one because branch indices are sorted.
        tokenized_agent["noise_feat"] = selected_features[:, 0]

        return latent

    # ------------------------------------------------------------------
    # SDE transition and log-probability
    # ------------------------------------------------------------------
    def sde_step_with_logprob(
        self,
        sigma: Tensor,
        sigma_prev: Tensor,
        model_output: Tensor,
        sample: Tensor,
        noise_level=0.7,
        prev_sample: Optional[Tensor] = None,
        sde_type: str = "sde",
    ):
        """Apply one stochastic reverse transition and compute its log-prob."""
        eps = 1e-5
        scale = self.model.normal_scale.clamp_min(eps)

        model_output = model_output / scale
        sample_normalized = self.model.normalize(sample)
        previous_normalized = (
            None
            if prev_sample is None
            else self.model.normalize(prev_sample)
        )

        dt = sigma_prev - sigma
        if torch.any(dt > 1e-7):
            raise ValueError(
                "Expected non-increasing sigma, but sigma_prev > sigma."
            )

        if sde_type == "sde":
            sigma_for_ratio = torch.where(
                sigma >= 1.0,
                torch.full_like(sigma, 0.95),
                sigma,
            )
            diffusion = torch.sqrt(
                sigma / (1.0 - sigma_for_ratio).clamp_min(eps)
            ) * noise_level

            safe_sigma = sigma.clamp_min(eps)
            previous_mean = sample_normalized * (
                1.0
                + diffusion.square() / (2.0 * safe_sigma) * dt
            )
            previous_mean = previous_mean + model_output * (
                1.0
                + diffusion.square() * (1.0 - sigma)
                / (2.0 * safe_sigma)
            ) * dt

            transition_std = diffusion * torch.sqrt(
                (-dt).clamp_min(eps)
            )

            if previous_normalized is None:
                previous_normalized = (
                    previous_mean
                    + transition_std * torch.randn_like(model_output)
                )

            log_prob = (
                -(
                    previous_normalized.detach() - previous_mean
                ).square()
                / (2.0 * transition_std.square().clamp_min(eps))
                - torch.log(transition_std.clamp_min(eps))
                - 0.5 * math.log(2.0 * math.pi)
            )

        elif sde_type == "cps":
            transition_std = (
                sigma_prev
                * math.sin(float(noise_level) * math.pi / 2.0)
            ).clamp_min(eps)

            predicted_x0 = sample_normalized - sigma * model_output
            estimated_noise = (
                sample_normalized + model_output * (1.0 - sigma)
            )
            remaining_variance = (
                sigma_prev.square() - transition_std.square()
            ).clamp_min(eps)
            previous_mean = (
                predicted_x0 * (1.0 - sigma_prev)
                + estimated_noise * remaining_variance.sqrt()
            )

            if previous_normalized is None:
                previous_normalized = (
                    previous_mean
                    + transition_std * torch.randn_like(model_output)
                )

            log_prob = (
                -(
                    previous_normalized.detach() - previous_mean
                ).square()
                / (2.0 * transition_std.square().clamp_min(eps))
                - torch.log(transition_std)
                - 0.5 * math.log(2.0 * math.pi)
            )
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type!r}.")

        reduce_dims = tuple(range(1, log_prob.ndim))
        log_prob = log_prob.mean(dim=reduce_dims)

        previous_sample = self.model.denormalize(previous_normalized)
        return (
            previous_sample,
            log_prob,
            previous_mean,
            transition_std,
        )