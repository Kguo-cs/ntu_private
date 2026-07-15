# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.smart.modules.agent_token_encoder import AgentTokenEncoder
from src.smart.modules.interative_decoder import InterativeDecoder
from src.smart.utils import infer_prev_pose, transform_to_global
from src.smart.utils.edge_utils import build_batch


class SMARTAgentDecoder(nn.Module):
    """Token-based agent decoder with autoregressive rollout."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        token_processor,
        alpha,
        dis_weight,
        dist_decay,
        reward_weight,
        reward_decay,
        use_gail: bool = False,
        discriminator=False,
        traj_diffusion: bool = False,
    ) -> None:
        super().__init__()
        del use_gail  # Kept in the signature for configuration compatibility.

        self.num_historical_steps = int(num_historical_steps)
        self.num_future_steps = int(num_future_steps)
        self.token_processor = token_processor
        self.discriminator = discriminator
        self.alpha = alpha
        self.shift = int(token_processor.shift)

        if self.shift <= 0:
            raise ValueError(f"token shift must be positive, got {self.shift}")
        if isinstance(alpha, (int, float)) and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.agent_token_embedding = AgentTokenEncoder(
            hidden_dim,
            num_freq_bands,
            token_processor,
            discriminator,
            traj_diffusion,
        )
        self.interative_decoder = InterativeDecoder(
            hidden_dim,
            time_span,
            pl2a_radius,
            a2a_radius,
            num_freq_bands,
            num_layers,
            num_heads,
            head_dim,
            dropout,
            hist_drop_prob,
            n_token_agent,
            pt2a_neighbor,
            a2a_neighbor,
            token_processor,
            dis_weight,
            dist_decay,
            reward_weight,
            reward_decay,
            discriminator=discriminator,
        )

        self.pred_init = bool(token_processor.pred_init and not discriminator)
        self.learn_init = bool(token_processor.learn_init)

    @staticmethod
    def _validate_time_inputs(sampled_idx, token_mask, valid_mask, pos, heading):
        if heading.ndim != 2:
            raise ValueError(f"heading must be [N, T], got {tuple(heading.shape)}")

        num_agents, num_steps = heading.shape
        expected = (num_agents, num_steps)

        for name, value in (
            ("sampled_idx", sampled_idx),
            ("token_mask", token_mask),
            ("valid_mask", valid_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")

        if pos.ndim != 3 or pos.shape[0] != num_agents or pos.shape[-1] != 2:
            raise ValueError(f"pos must be [N, T, 2], got {tuple(pos.shape)}")
        if pos.shape[1] < num_steps:
            raise ValueError("pos contains fewer timesteps than heading")

        # Legacy incremental rollout passed two positions with one heading.
        # Align all decoder inputs before token encoding.
        return pos[:, -num_steps:]

    def predict_agent(
        self,
        sampled_idx,
        token_mask,
        mask_a,
        pos_a,
        head_a,
        tokenized_agent,
        map_feature,
        shape,
        n_current=0,
    ):
        """Encode a temporal window and predict its next-token logits."""
        num_agents, num_steps = head_a.shape

        batch_a = tokenized_agent["batch"]

        head_vector_a = torch.stack((head_a.cos(), head_a.sin()), dim=-1)
        feat_a_token, agent_token_emb, _ = self.agent_token_embedding(
            agent_token_index=sampled_idx,
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            mask_a=mask_a,
            agent_type=tokenized_agent["type"],
            agent_shape=shape,
            token_mask=token_mask,
            goal_pos=tokenized_agent.get("goal_pos"),
            goal_mask=tokenized_agent.get("goal_mask"),
        )

        pos_a=pos_a[:, -num_steps:]

        batch_by_agent = batch_a[:, None].expand(-1, num_steps)
        batch_by_time = build_batch(
            batch_a,
            tokenized_agent["num_graphs"],
            num_steps,
        ).reshape(num_steps, num_agents).transpose(0, 1)

        features = [
            pos_a,
            head_a,
            head_vector_a,
            mask_a,
            batch_by_agent,
            batch_by_time,
        ]
        train_mask = tokenized_agent.get("train_mask") if self.training else None

        logits, feat_a, rewards, weight, a2a_feature = self.interative_decoder(
            features,
            feat_a_token,
            agent_token_emb,
            map_feature,
            train_mask,
            n_current,
            tokenized_agent,
        )
        return logits, a2a_feature, rewards, weight, feat_a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict all available next-token targets."""
        if tokenized_agent["sampled_idx"].shape[1] <= 1:
            return {}

        logits, _, _, _, feat_a = self.predict_agent(
            sampled_idx=tokenized_agent["sampled_idx"][:, :-1],
            token_mask=tokenized_agent["token_mask"][:, :-1],
            mask_a=tokenized_agent["valid_mask"][:, :-1],
            pos_a=tokenized_agent["sampled_pos"][:, :-1],
            head_a=tokenized_agent["sampled_heading"][:, :-1],
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            shape=tokenized_agent["shape"],
        )

        # These side effects are retained because rollout/value code consumes them.
        tokenized_agent["next_token_logits"] = logits
        tokenized_agent["feat_a"] = feat_a
        return {"next_token_logits": logits}

    @staticmethod
    def _latest_active_logits(
        logits: torch.Tensor,
        active_mask: torch.Tensor,
        num_decoded_steps: int,
    ):
        """Return latest-step logits in active-agent order."""
        num_agents = active_mask.numel()
        num_active = int(active_mask.sum().item())

        if num_active == 0:
            return logits.new_empty((0, logits.shape[-1]))

        if logits.ndim == 3:
            if logits.shape[1] == num_agents:       # [T, N, K]
                return logits[-1, active_mask]
            if logits.shape[0] == num_agents:       # [N, T, K]
                return logits[active_mask, -1]
            raise ValueError(f"Unknown logit layout: {tuple(logits.shape)}")

        if logits.ndim != 2:
            raise ValueError(f"Expected 2-D or 3-D logits, got {tuple(logits.shape)}")

        full_rows = num_decoded_steps * num_agents
        active_rows = num_decoded_steps * num_active

        if logits.shape[0] == full_rows:
            return logits.reshape(num_decoded_steps, num_agents, -1)[-1, active_mask]
        if logits.shape[0] == active_rows:
            return logits.reshape(num_decoded_steps, num_active, -1)[-1]
        if logits.shape[0] == num_active:           # already one selected block
            return logits

        raise ValueError(
            "Cannot infer flattened logit layout: "
            f"rows={logits.shape[0]}, decoded_steps={num_decoded_steps}, "
            f"agents={num_agents}, active_agents={num_active}"
        )

    @staticmethod
    def _cached_logits(logits, active_mask, current_step):
        """Use cached logits only when their flattened layout is unambiguous."""
        num_agents = active_mask.numel()
        num_active = int(active_mask.sum().item())
        step = current_step - 1

        if num_active == 0:
            return logits.new_empty((0, logits.shape[-1]))

        if logits.ndim == 3:
            if logits.shape[1] == num_agents and step < logits.shape[0]:
                return logits[step, active_mask]
            if logits.shape[0] == num_agents and step < logits.shape[1]:
                return logits[active_mask, step]
            return None

        if logits.ndim != 2:
            return None

        for agents_per_step, use_mask in (
            (num_agents, True),
            (num_active, False),
        ):
            if logits.shape[0] % agents_per_step != 0:
                continue
            num_steps = logits.shape[0] // agents_per_step
            if step >= num_steps:
                continue
            block = logits.reshape(num_steps, agents_per_step, -1)[step]
            return block[active_mask] if use_mask else block

        return None

    def _trim_cache(self, current_step):
        for name in ("pos_cache", "head_cache", "mask_cache", "head_vector_cache"):
            value = getattr(self.interative_decoder, name, None)
            if torch.is_tensor(value):
                setattr(self.interative_decoder, name, value[:, :current_step])

        value = getattr(self.interative_decoder, "feat_a_cache", None)
        if isinstance(value, list):
            self.interative_decoder.feat_a_cache = [
                feature[:current_step] for feature in value
            ]

    def _sample_token(
        self,
        logits,
        previous_idx,
        active_mask,
        num_decoded_steps,
    ):
        next_idx = previous_idx.clone()
        if not active_mask.any():
            return next_idx

        logits = self._latest_active_logits(
            logits,
            active_mask,
            num_decoded_steps,
        )

        next_idx[active_mask] = Categorical(logits=logits / self.alpha).sample()
        return next_idx

    def _run_init_decoder(
        self,
        init_decoder,
        tokenized_agent,
        pred_traj_10hz,
        pred_head_10hz,
    ):
        pos, heading, sampled_idx, shape, local_vel = init_decoder(tokenized_agent)

        if pos.ndim != 3 or pos.shape[-1] != 2:
            raise ValueError(f"init position must be [N, T, 2], got {tuple(pos.shape)}")
        if heading.shape != pos.shape[:2] or sampled_idx.shape != pos.shape[:2]:
            raise ValueError("init heading/index must match init position timesteps")
        if pos.shape[1] == 0:
            raise ValueError("init_decoder returned no state")

        if "gt_z_raw" in tokenized_agent:
            # The original code inferred the previous pose using the last token,
            # but reconstructed it using the first token. Both must use the first.
            first_idx = sampled_idx[:, :1]
            prev_pos, prev_heading = infer_prev_pose(
                pos[:, :1],
                heading[:, :1],
                first_idx,
                tokenized_agent["token_traj_all"],
            )
            pred_traj_10hz.append(prev_pos)
            pred_head_10hz.append(prev_heading)

            # Populate the 10 Hz reconstruction lists. Returned token states are
            # intentionally ignored because `pos`/`heading` are the init states.
            self.get_next(
                first_idx,
                prev_pos,
                prev_heading,
                pred_traj_10hz,
                pred_head_10hz,
                tokenized_agent,
            )

        return pos, heading, sampled_idx, shape, local_vel

    def autoregressive_agent(
        self,
        init_decoder,
        tokenized_agent,
        map_feature,
        current_step,
        max_step,
    ):
        """Append ``max_step`` autoregressive token states."""
        gt_pos = tokenized_agent["sampled_pos"]
        gt_head = tokenized_agent["sampled_heading"]
        gt_valid = tokenized_agent["valid_mask"].bool()
        gt_idx = tokenized_agent["sampled_idx"]

        current_step = int(current_step)
        num_steps = int(max_step)
        if current_step <= 0 and not self.pred_init:
            raise ValueError(f"current_step must be positive, got {current_step}")
        if num_steps < 0:
            raise ValueError(f"max_step must be non-negative, got {num_steps}")

        pred_traj_10hz, pred_head_10hz = [], []
        initial_local_vel = None

        if self.pred_init:
            pos_a, head_a, sampled_idx, shape, initial_local_vel = (
                self._run_init_decoder(
                    init_decoder,
                    tokenized_agent,
                    pred_traj_10hz,
                    pred_head_10hz,
                )
            )
            current_step = pos_a.shape[1]

            if "gt_z_raw" in tokenized_agent:
                total_steps = (
                    self.num_historical_steps - 1 + self.num_future_steps
                ) // self.shift
                num_steps = max(total_steps - current_step, 0)
            else:
                num_steps = max(gt_valid.shape[1] - current_step, 0)

            mask = torch.ones(
                pos_a.shape[:2],
                dtype=torch.bool,
                device=pos_a.device,
            )
            token_mask = mask.clone()
        else:
            if current_step > gt_pos.shape[1]:
                raise ValueError("current_step exceeds available token states")

            pos_a = gt_pos[:, :current_step].clone()
            head_a = gt_head[:, :current_step].clone()
            sampled_idx = gt_idx[:, :current_step].clone()
            shape = tokenized_agent["shape"]
            mask = gt_valid[:, :current_step].clone()
            token_mask = tokenized_agent["token_mask"][:, :current_step].bool().clone()

        # Agents valid at the rollout boundary remain active for the rollout.
        active_mask = mask[:, -1].clone()
        cached_logits = tokenized_agent.get("next_token_logits")

        for rollout_step in range(num_steps):
            logits = None
            decoded_steps = 1

            if rollout_step == 0 and cached_logits is not None and not self.pred_init:
                logits = self._cached_logits(
                    cached_logits,
                    active_mask,
                    current_step,
                )
                if logits is not None:
                    self._trim_cache(current_step)

            if logits is None:
                if rollout_step == 0:
                    idx_in, token_mask_in, mask_in = sampled_idx, token_mask, mask
                    pos_in, head_in = pos_a, head_a
                else:
                    idx_in = sampled_idx[:, -1:]
                    token_mask_in = token_mask[:, -1:]
                    mask_in = mask[:, -1:]
                    pos_in = pos_a[:, -2:]       # fixed: was two position steps
                    head_in = head_a[:, -1:]

                decoded_steps = idx_in.shape[1]
                logits, _, _, _, _ = self.predict_agent(
                    idx_in,
                    token_mask_in,
                    mask_in,
                    pos_in,
                    head_in,
                    tokenized_agent,
                    map_feature,
                    shape,
                    n_current=current_step + rollout_step - 1,
                )

            next_idx = self._sample_token(
                logits,
                sampled_idx[:, -1],
                active_mask,
                decoded_steps,
            )
            sampled_idx = torch.cat((sampled_idx, next_idx[:, None]), dim=1)

            pos_a, head_a = self.get_next(
                sampled_idx,
                pos_a,
                head_a,
                pred_traj_10hz,
                pred_head_10hz,
                tokenized_agent,
                active_mask=active_mask,
            )
            mask = torch.cat((mask, active_mask[:, None]), dim=1)
            token_mask = torch.cat((token_mask, active_mask[:, None]), dim=1)

        output = {
            "num_graphs": tokenized_agent["num_graphs"],
            "type": tokenized_agent["type"],
            "shape": shape,
            "batch": tokenized_agent["batch"],
            "sampled_pos": pos_a,
            "sampled_heading": head_a,
            "valid_mask": mask,
            "token_mask": token_mask,
            "sampled_idx": sampled_idx,
        }

        if "gt_z_raw" in tokenized_agent:
            if self.pred_init:
                output["initial_local_vel"] = initial_local_vel

            if pred_traj_10hz:
                output["pred_traj_10hz"] = torch.cat(pred_traj_10hz, dim=1)
                output["pred_head_10hz"] = torch.cat(pred_head_10hz, dim=1)
            else:
                output["pred_traj_10hz"] = pos_a.new_empty((len(pos_a), 0, 2))
                output["pred_head_10hz"] = head_a.new_empty((len(head_a), 0))

            output["pred_z_10hz"] = tokenized_agent["gt_z_raw"][:, None].expand(
                -1,
                output["pred_traj_10hz"].shape[1],
            )

        return output

    def get_next(
        self,
        sampled_idx,
        pos_a,
        head_a,
        pred_traj_10hz,
        pred_head_10hz,
        tokenized_agent,
        active_mask=None,
    ):
        """Convert the latest token to global motion and append its endpoint."""
        num_agents = len(sampled_idx)
        if active_mask is None:
            active_mask = torch.ones(
                num_agents,
                dtype=torch.bool,
                device=sampled_idx.device,
            )
        else:
            active_mask = active_mask.to(sampled_idx.device, dtype=torch.bool)

        token_library = tokenized_agent["token_traj_all"]
        agent_idx = torch.arange(num_agents, device=sampled_idx.device)
        token_local = token_library[agent_idx, sampled_idx[:, -1]]

        token_global = transform_to_global(
            pos_local=token_local.flatten(1, 2),
            head_local=None,
            pos_now=pos_a[:, -1],
            head_now=head_a[:, -1],
        )[0].view_as(token_local)

        centers = token_global.mean(dim=2)
        direction = token_global[:, :, 0] - token_global[:, :, 3]
        headings = torch.atan2(direction[..., 1], direction[..., 0])

        # Do not move agents that were invalid at the rollout boundary.
        if not active_mask.all():
            centers = centers.clone()
            headings = headings.clone()
            centers[~active_mask] = pos_a[~active_mask, -1:, :]
            headings[~active_mask] = head_a[~active_mask, -1:]

        if "gt_z_raw" in tokenized_agent:
            pred_traj_10hz.append(centers)
            pred_head_10hz.append(headings)

        pos_a = torch.cat((pos_a, centers[:, -1:]), dim=1)
        head_a = torch.cat((head_a, headings[:, -1:]), dim=1)
        return pos_a, head_a

    def inference(
        self,
        init_decoder,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme=None,
        post_sampling=False,
        step_current_10hz=None,
        n_step_future_10hz=None,
    ) -> Dict[str, torch.Tensor]:
        del sampling_scheme, post_sampling  # Preserved API; not used here.

        future_10hz = (
            self.num_future_steps
            if n_step_future_10hz is None
            else int(n_step_future_10hz)
        )
        current_10hz = (
            self.num_historical_steps - 1
            if step_current_10hz is None
            else int(step_current_10hz)
        )

        if future_10hz < 0 or current_10hz < 0:
            raise ValueError("10 Hz step counts must be non-negative")
        if future_10hz % self.shift != 0 or current_10hz % self.shift != 0:
            raise ValueError(
                f"10 Hz step counts must be divisible by token shift={self.shift}"
            )

        future_tokens = future_10hz // self.shift
        current_tokens = current_10hz // self.shift

        if current_tokens <= 0 and not self.pred_init:
            raise ValueError("At least one current token state is required")

        return self.autoregressive_agent(
            init_decoder,
            tokenized_agent,
            map_feature,
            current_tokens,
            future_tokens,
        )