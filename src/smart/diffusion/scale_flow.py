import math
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import jvp
from torch_geometric.data import HeteroData

from src.smart.utils import weight_init
from src.smart.diffusion.diffusion_utils import get_diff_loss, get_closest_sum_idx_fast
from src.smart.diffusion.dit.dit import DiT
from .denoiser import InitDenoiser


class ScaleFlow(nn.Module):
    """Initial-state Flow / MeanFlow module with optional SDE advantage training.

    This is a simplified version of the old ScaleFlow:
        - keeps supervised flow / iMF training;
        - keeps the SDE branch used by advantage-based finetuning;
        - keeps the non-SDE advantage branch;
        - removes TempFlow-GRPO seed grouping / temporal weighting;
        - removes num_samples / mc_num repetition. The module always uses one
          latent sample per agent with shape [N, D].

    Time convention:
        z_t = (1 - t) * noise + t * data
        t = 0 -> noise, t = 1 -> data, v_gt = data - noise.

    In iMF mode, the denoiser output is interpreted as interval-average velocity
    u_theta(z_t, t, r), and the update is
        z_r = z_t + (r - t) * u_theta(z_t, t, r).
    """

    def __init__(self, args, token_processor, gail: bool = False):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.use_dit = bool(getattr(args, "use_dit", False))

        # MeanFlow / iMF options.
        self.use_imf = bool(getattr(args, "use_imf", False))
        self.imf_interval_ratio = float(getattr(args, "imf_interval_ratio", 0.30))
        self.imf_global_ratio = float(getattr(args, "imf_global_ratio", 0.0))
        self.imf_jvp_detach = bool(getattr(args, "imf_jvp_detach", True))

        if self.use_dit:
            self.x_pred = False
            self.model = DiT(self.hidden_dim)
        else:
            # In iMF, the model predicts average velocity u, not x0.
            self.x_pred = not self.use_imf
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
                mean_flow=self.use_imf,
                x_pred=self.x_pred,
            )

        # Compatibility flags used elsewhere in the project.
        self.use_all_type = self.model.use_all_type
        self.t_eps = 0.05 if self.x_pred else 0.0

        # Keep SDE + advantage finetuning, but remove TempFlow-specific parts.
        self.use_sde = bool(gail and getattr(token_processor, "learn_init", False))
        self.noise_level = float(getattr(args, "noise_level", 0.1))
        self.use_init_ppo_ratio = bool(getattr(args, "use_init_ppo_ratio", False))
        self.init_adv_clip = float(getattr(args, "init_adv_clip", 3.0))
        self.init_logprob_clip = float(getattr(args, "init_logprob_clip", 50.0))
        self.init_ppo_clip = float(getattr(args, "init_ppo_clip", 0.2))

        self.use_ref=False
        self.apply(weight_init)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _sample_noise(self, x: torch.Tensor, tokenized_agent: HeteroData) -> torch.Tensor:
        """Sample base noise and denormalize it to the data scale."""
        noise = torch.randn_like(x)
        norm_x=self.model.normalize(x)
        noise[tokenized_agent["ego_mask"]] = norm_x[tokenized_agent["ego_mask"]]
        fake_idx = get_closest_sum_idx_fast(
            noise,
            norm_x,
            tokenized_agent,
            all_state=True,
        )
        return self.model.denormalize(noise[fake_idx])

    def _sample_time(
        self,
        x: torch.Tensor,
        tokenized_agent: HeteroData,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one training time per scene, then broadcast by agent."""
        device = x.device
        batch = tokenized_agent["batch"]
        num_graphs = tokenized_agent["num_graphs"]

        base_t = torch.rand((num_graphs, 1), device=device, dtype=torch.float32)#.sqrt()
        t = base_t[batch]
        t_dt = torch.ones_like(t)
        return t, t_dt, base_t

    @staticmethod
    def _fix_conditioned_agents(
        x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        tokenized_agent: HeteroData,
    ) -> None:
        """Make ego / conditioned agents follow a constant clean path."""
        ego_mask = tokenized_agent["ego_mask"]
        noise[ego_mask] = x[ego_mask]
        t[ego_mask] = 1.0

    def _sanitize_advantages(
        self,
        advantages: torch.Tensor,
        ego_mask: torch.Tensor,
        selected_agent_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return finite, centered, clipped advantages.

        TempFlow seed-group normalization is intentionally removed. If a selected
        subset is supplied, advantages are first aligned to that subset.
        """
        advantages = advantages.detach().to(dtype=torch.float32)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        if advantages.ndim > 1:
            advantages = advantages.reshape(advantages.shape[0], -1).mean(dim=-1)

        if selected_agent_idx is not None:
            if advantages.shape[0] != ego_mask.shape[0]:
                raise RuntimeError(
                    "advantages must be all-agent length before selected_agent_idx: "
                    f"advantages={tuple(advantages.shape)}, ego={tuple(ego_mask.shape)}"
                )
            advantages = advantages[selected_agent_idx]

        if advantages.numel() > 1:
            advantages = advantages - advantages.mean()
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

        return advantages.clamp(-self.init_adv_clip, self.init_adv_clip)

    def _meanflow_u(
        self,
        z: torch.Tensor,
        t_cur: torch.Tensor,
        t_ref: torch.Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, torch.Tensor],
        eval_mask: Optional[torch.Tensor] = None,
        mode: int = 1,
        use_map_condition: bool = True,
    ) -> torch.Tensor:
        """Return u_theta(z_t, t, r), the interval-average velocity."""

        tokenized_agent["meanflow_h"] = (t_ref - t_cur).clamp_min(0.0)
        out = self.model(
            z,
            t_cur,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )
        return out

    def _sample_ref_time(self, t: torch.Tensor, ego_mask: torch.Tensor) -> torch.Tensor:
        """Sample r in [t, 1] for iMF identity training."""
        r = t + (1.0 - t) * torch.rand_like(t)

        if self.imf_interval_ratio < 1.0:
            use_interval = torch.rand_like(t[..., :1]) < self.imf_interval_ratio
            r = torch.where(use_interval, r, t)

        if self.imf_global_ratio > 0.0:
            use_global = torch.rand_like(t[..., :1]) < self.imf_global_ratio
            r = torch.where(use_global, torch.ones_like(r), r)

        return r

    # ------------------------------------------------------------------
    # Model-output interpretation
    # ------------------------------------------------------------------
    def _model_velocity(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, torch.Tensor],
        eval_mask: Optional[torch.Tensor] = None,
        t_next: Optional[torch.Tensor] = None,
        mode: int = 1,
        use_map_condition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (velocity_or_average_velocity, x0_estimate).

        If use_imf=True, the first output is u(z_t,t,t_next). If t_next is omitted,
        it defaults to 1, giving an endpoint average velocity.
        """
        if self.use_imf:
            u = self._meanflow_u(
                z,
                t,
                t_next,
                tokenized_agent,
                map_feature,
                eval_mask=eval_mask,
                mode=mode,
                use_map_condition=use_map_condition,
            )
            return u, None

        pred = self.model(
            z,
            t,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )

        if self.x_pred:
            velocity = (pred - z) / (1.0 - t).clamp_min(self.t_eps)
            x0 = pred
        else:
            velocity = pred
            x0 = z + (1.0 - t) * velocity
        return velocity, x0

    # ------------------------------------------------------------------
    # Public loss entry
    # ------------------------------------------------------------------
    def get_loss(
        self,
        x: torch.Tensor,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, torch.Tensor],
    ):
        """Compute supervised loss plus optional advantage policy loss.

        num_samples has been removed. Input x is used once as [N, D].
        """
        if "advantages" in tokenized_agent:
            if self.use_sde:
                policy_loss = self._sde_advantage_loss(tokenized_agent, initial_map_feature)
            else:
                policy_loss = self._direct_advantage_loss(tokenized_agent, initial_map_feature)

            tokenized_agent["rl_loss"] = policy_loss

        if self.use_imf:
            return self._meanflow_supervised_loss(x,  tokenized_agent, initial_map_feature)
        return self._standard_supervised_loss(x,  tokenized_agent, initial_map_feature)

    def _prepare_supervised_batch(self, x,  tokenized_agent):
        noise = self._sample_noise(x, tokenized_agent)
        t, t_dt, base_t = self._sample_time(x, tokenized_agent)
        self._fix_conditioned_agents(x, noise, t, tokenized_agent)
        z = (1.0 - t) * noise + t * x
        return x, noise, t, t_dt, base_t, z

    def _standard_supervised_loss(self, x,  tokenized_agent, map_feature):
        x, noise, t, t_dt, base_t, z = self._prepare_supervised_batch(x,  tokenized_agent)
        ego_mask = tokenized_agent["ego_mask"]

        velocity, x0 = self._model_velocity(z, t, tokenized_agent, map_feature)
        ego_view = ego_mask.view(ego_mask.shape[0], 1)
        x_pred = torch.where(ego_view, x.detach(), x0)

        loss = get_diff_loss(
            tokenized_agent,
            x_pred,
            x,
            t,
            self.t_eps,
            use_col=True,
            x_pred=self.x_pred,
        )

        return loss, x_pred, z, t

    def _meanflow_supervised_loss(self, x,  tokenized_agent, map_feature):

        x, noise, t, t_dt, base_t, z = self._prepare_supervised_batch(x,  tokenized_agent)
        ego_mask = tokenized_agent["ego_mask"]

        # A small fraction of true one-step points: t=0, r=1.
        if self.imf_global_ratio > 0.0:
            batch = tokenized_agent["batch"]
            num_graphs = tokenized_agent["num_graphs"]
            global_mask = torch.rand((num_graphs, 1), device=x.device) < self.imf_global_ratio
            global_mask = global_mask[batch] & (~ego_mask[:, None])
            t = torch.where(global_mask, torch.zeros_like(t), t)
            z = (1.0 - t) * noise + t * x

        r = self._sample_ref_time(t, ego_mask)

        def u_func(z_in, t_in, r_in):
            return self._meanflow_u(z_in, t_in, r_in, tokenized_agent, map_feature)

        u_pred = u_func(z, t, r)
        with torch.no_grad():
            v_boundary = u_func(z, t, t)

        tangent = (v_boundary.detach(), torch.ones_like(t), torch.zeros_like(r))
        if self.imf_jvp_detach:
            with torch.no_grad():
                #_, du_dt = jvp(u_func, (z, t, r), tangent)
                eps = float(getattr(self, "imf_fd_eps", 1e-3))
                dt_eps = torch.full_like(t, eps)

                # Avoid stepping beyond t=1 too much.
                max_dt = (1.0 - t).clamp_min(0.0)
                dt_eps = torch.minimum(dt_eps, max_dt)

                valid_fd = dt_eps > 1e-8

                # Total derivative direction:
                #   dz/dt = v_boundary
                #   dt/dt = 1
                #   dr/dt = 0
                z_plus = z.detach() + dt_eps * v_boundary
                t_plus = (t.detach() + dt_eps).clamp_max(1.0)
                r_plus = r.detach()

                u_plus = u_func(z_plus, t_plus, r_plus).detach()

                denom = dt_eps.clamp_min(1e-6)
                du_dt = (u_plus - u_pred.detach()) / denom
                du_dt = torch.where(valid_fd, du_dt, torch.zeros_like(du_dt))
        else:
            _, du_dt = jvp(u_func, (z, t, r), tangent)

        # Forward-time iMF identity: v = u - (r - t) * d_t u.
        v_reparam = u_pred - (r - t) * du_dt

        loss = get_diff_loss(
            tokenized_agent,
            v_reparam,
            x-noise,
            t,
            self.t_eps,
            use_col=False,
            x_pred=False,
        )

        return loss, None, z, t

    def _sde_advantage_loss(self, tokenized_agent, map_feature):

        ego_mask=tokenized_agent["ego_mask"]

        z_sampled, prev_sample, old_log_prob = tokenized_agent["sde_z"]
        t_sampled, t_next_sampled = tokenized_agent["sde_t"]

        velocity, _ = self._model_velocity(
            z_sampled,
            t_sampled,
            tokenized_agent,
            map_feature,
            t_next=t_next_sampled if self.use_imf else None,
        )

        non_ego = ~ego_mask

        _, log_prob, _, _ = self.sde_step_with_logprob(
            1.0 - t_sampled[non_ego],
            1.0 - t_next_sampled[non_ego],
            -velocity[non_ego],
            z_sampled[non_ego],
            noise_level=self.noise_level,
            prev_sample=prev_sample[non_ego],
        )

        advantages =tokenized_agent["advantages"][non_ego]

        # advantages = self._sanitize_advantages(
        #     tokenized_agent["advantages"],
        #     ego_mask,
        #     selected_agent_idx=selected_agent_idx,
        # )[non_ego]

        if self.use_init_ppo_ratio and old_log_prob is not None:
            old_lp = old_log_prob[non_ego].detach()
            old_lp = torch.nan_to_num(old_lp, nan=0.0, posinf=0.0, neginf=0.0)
            old_lp = old_lp.clamp(-self.init_logprob_clip, self.init_logprob_clip)

            log_ratio = (log_prob - old_lp).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = ratio.clamp(1.0 - self.init_ppo_clip, 1.0 + self.init_ppo_clip)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()

            tokenized_agent["init_ratio_mean"] = ratio.detach().mean()
            tokenized_agent["init_clip_fraction"] = (
                (ratio < 1.0 - self.init_ppo_clip) | (ratio > 1.0 + self.init_ppo_clip)
            ).float().mean().detach()
            return policy_loss

        return -(log_prob * advantages).mean()

    def _direct_advantage_loss(self, tokenized_agent, map_feature):
        """Non-SDE advantage loss based on x0 matching quality.

        This is the old lightweight PPO-like branch, but without mc_num/num_samples.
        """
        ego_mask=tokenized_agent["ego_mask"]

        x_sampled = tokenized_agent["gen_z"]

        x, noise, t, t_dt, base_t, z = self._prepare_supervised_batch(x_sampled,  tokenized_agent)

        _, x0_pred = self._model_velocity(z, t, tokenized_agent, map_feature)
        scale = self.model.normal_scale.clamp_min(1e-6)

        logp_cur = self.adaptive_x0_logprob(
            x0_pred / scale,
            x_sampled.detach() / scale,
        )

        advantages = tokenized_agent["advantages"]
        non_ego = ~ego_mask
        advantages = advantages[non_ego]

        # Reference-free PPO-like ratio. logp_old is detached current logp, so the
        # gradient remains through logp_cur while the baseline is fixed.
        logp_old = logp_cur.detach()
        log_ratio = (logp_cur - logp_old).clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio)[non_ego]
        ratio_clip = ratio.clamp(1.0 - self.init_ppo_clip, 1.0 + self.init_ppo_clip)

        tokenized_agent["sampled_match_loss"] = -logp_cur.detach()
        tokenized_agent["clip_ratio"] = (
            (ratio < 1.0 - self.init_ppo_clip) | (ratio > 1.0 + self.init_ppo_clip)
        ).float().detach()

        return -torch.minimum(ratio * advantages, ratio_clip * advantages).mean() * 100.0

    @staticmethod
    def adaptive_x0_logprob(pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
        """Return a per-agent pseudo log-prob from normalized x0 error.

        Shapes are [N, D]. Higher is better.
        """
        err = pred_x0 - target_x0
        mse = err.square()
        l1 = err.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5).detach()
        loss = (mse / l1).mean(dim=-1)  # [N]
        return -loss * 0.1  # [N]

    # ------------------------------------------------------------------
    # Sampling with optional SDE branch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_step(
        self,
        z: torch.Tensor,
        t_scalar: torch.Tensor,
        t_next_scalar: torch.Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, torch.Tensor],
        noise_level: Optional[torch.Tensor] = None,
    ):
        n_agent = z.shape[0]
        t = torch.full((n_agent, 1), t_scalar, device=z.device, dtype=z.dtype)
        t_next = torch.full((n_agent, 1), t_next_scalar, device=z.device, dtype=z.dtype)

        self._fix_conditioned_agents(tokenized_agent["expert_input"], z, t, tokenized_agent)

        velocity, x_pred = self._model_velocity(
            z,
            t,
            tokenized_agent,
            map_feature,
            t_next=t_next if self.use_imf else None,
        )

        log_prob = z.new_zeros(n_agent)
        z_next = z + (t_next - t) * velocity

        if self.use_sde and noise_level is not None and torch.any(noise_level > 0):
            sde_mask = noise_level.reshape(noise_level.shape[0], -1).amax(dim=-1) > 0
            if torch.any(sde_mask):
                z_sde, log_prob_sde, _, _ = self.sde_step_with_logprob(
                    1.0 - t[sde_mask],
                    1.0 - t_next[sde_mask],
                    -velocity[sde_mask],
                    z[sde_mask],
                    noise_level=noise_level[sde_mask],
                )
                z_next[sde_mask] = z_sde
                log_prob[sde_mask] = log_prob_sde

        return z_next, x_pred, t, log_prob

    @torch.no_grad()
    def sample(
        self,
        tokenized_agent: HeteroData,
        initial_map_feature: Mapping[str, torch.Tensor],
        steps: int = 20,
    ):
        """Generate one initial state sample per agent.

        num_samples has been removed. The latent has shape [N, D].
        """
        agent_batch = tokenized_agent["batch"]
        num_graphs = tokenized_agent["num_graphs"]
        n_agent = len(agent_batch)

        z = torch.randn(n_agent, self.model.m_delta_dim, device=agent_batch.device)
        z = self.model.denormalize(z)

        if "expert_input" not in tokenized_agent.keys():
            diff_input, _ = self.model.get_input(tokenized_agent)
            tokenized_agent["expert_input"]=diff_input

        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=agent_batch.device)#.sqrt()

        if self.use_sde:
            z_list = [z]
            x_list = []
            t_list = []
            log_prob_list = []
            feat_list=[]

            noise_level = torch.zeros(n_agent, steps, 1, device=agent_batch.device)

            branch_step_graph = torch.randint(0, steps, (num_graphs,), device=agent_batch.device)
            branch_step_agent = branch_step_graph[agent_batch]
            noise_level[torch.arange(n_agent, device=agent_batch.device), branch_step_agent, 0] = self.noise_level
            noise_mask = noise_level[:, :, 0] > 0

        for i in range(steps):
            z, x_pred, t, log_prob = self._sample_step(
                z,
                timesteps[i],
                timesteps[i + 1],
                tokenized_agent,
                initial_map_feature,
                noise_level=noise_level[:, i] if self.use_sde else None,
            )

            if self.use_sde:
                z_list.append(z)
                x_list.append(x_pred)
                t_list.append(t)
                log_prob_list.append(log_prob)
                feat_list.append( tokenized_agent["noise_feat_cur"])
            elif i==0:
                tokenized_agent["noise_feat"]=tokenized_agent["noise_feat_cur"]

        z[tokenized_agent["ego_mask"]]=tokenized_agent["expert_input"][tokenized_agent["ego_mask"]]


        if self.use_sde:
            # tokenized_agent["pred_z_list"] = torch.stack(z_list, dim=1)

            z_stack = torch.stack(z_list, dim=1)          # [N, steps+1, D]
            t_stack = torch.stack(t_list + [torch.ones_like(t_list[-1])], dim=1)
            logp_stack = torch.stack(log_prob_list, dim=1)

            tokenized_agent["sde_z"] = (
                z_stack[:, :-1][noise_mask],
                z_stack[:, 1:][noise_mask],
                logp_stack[noise_mask].detach(),
            )
            tokenized_agent["sde_t"] = (
                t_stack[:, :-1][noise_mask],
                t_stack[:, 1:][noise_mask],
            )
            tokenized_agent["noise_feat"] = torch.stack(feat_list, dim=1)[noise_mask]
        else:
            tokenized_agent["gen_z"] = z

        return z

    # ------------------------------------------------------------------
    # SDE transition and log-prob
    # ------------------------------------------------------------------
    def sde_step_with_logprob(
        self,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        noise_level=0.7,
        prev_sample: Optional[torch.Tensor] = None,
        sde_type: str = "sde",
    ):
        """One stochastic reverse transition with log-prob.

        This is the old SDE branch kept without TempFlow-specific weighting.
        The inputs use sigma=1-t so that the existing flow-time convention is
        compatible with the previous implementation.
        """
        eps = 1e-5
        scale = self.model.normal_scale.clamp_min(eps)
        model_output = model_output / scale
        sample_norm = self.model.normalize(sample)
        sigma_max = 0.95

        if prev_sample is not None:
            prev_sample_norm = self.model.normalize(prev_sample)
        else:
            prev_sample_norm = None

        dt = sigma_prev - sigma

        if sde_type == "sde":
            safe_sigma = torch.where(sigma == 1, torch.full_like(sigma, sigma_max), sigma)
            std_dev_t = torch.sqrt(sigma / (1.0 - safe_sigma).clamp_min(eps)) * noise_level
            prev_mean = sample_norm * (1.0 + std_dev_t.square() / (2.0 * sigma.clamp_min(eps)) * dt)
            prev_mean = prev_mean + model_output * (
                1.0 + std_dev_t.square() * (1.0 - sigma) / (2.0 * sigma.clamp_min(eps))
            ) * dt
            std = std_dev_t * torch.sqrt((-dt).clamp_min(eps))

            if prev_sample_norm is None:
                prev_sample_norm = prev_mean + std * torch.randn_like(model_output)

            log_prob = (
                -((prev_sample_norm.detach() - prev_mean).square()) / (2.0 * std.square().clamp_min(eps))
                - torch.log(std.clamp_min(eps))
                - 0.5 * math.log(2.0 * math.pi)
            )
        elif sde_type == "cps":
            std_dev_t = (sigma_prev * math.sin(noise_level * math.pi / 2.0)).clamp_min(eps)
            pred_original_sample = sample_norm - sigma * model_output
            noise_estimate = sample_norm + model_output * (1.0 - sigma)
            inside = (sigma_prev.square() - std_dev_t.square()).clamp_min(eps)
            prev_mean = pred_original_sample * (1.0 - sigma_prev) + noise_estimate * inside.sqrt()

            if prev_sample_norm is None:
                prev_sample_norm = prev_mean + std_dev_t * torch.randn_like(model_output)

            log_prob = -((prev_sample_norm.detach() - prev_mean).square()) / (2.0 * std_dev_t.square().clamp_min(eps))
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        prev_sample = self.model.denormalize(prev_sample_norm)
        return prev_sample, log_prob, prev_mean, std_dev_t
