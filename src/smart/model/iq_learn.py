"""Cleaner IQ/GAIL Lightning training module.

The public class and main method names are kept compatible with the original
implementation, while the training stages are split into small helpers.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from lightning import LightningModule
from torch import Tensor

from src.smart.loss.edge_gp import ZeroCenteredGradientPenalty_edge
from src.smart.loss.gp_penalty import (
    _has_elements,
    _reshape_valid_rewards,
    _select_ego_logits,
    _weighted_bce_with_logits,
    ZeroCenteredGradientPenalty
)
from src.smart.loss.rollout_buffer import (
    RunningMeanStdTorch,
    compute_advantages,
    get_train_mask,
    rollout,
)


TensorDict = MutableMapping[str, Any]


def _zero(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _safe_mean(value: Any, reference: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.mean() if value.numel() else _zero(reference)
    return reference.new_tensor(float(value))


class IQ_SoftQ(LightningModule):
    """Joint supervised, GAIL, value, and initial-state trainer.

    ``model_config`` must expose ``encoder`` and ``token_processor`` either as
    attributes or mapping entries. This explicit contract replaces the invalid
    ``LightningModule.__init__(model_config)`` call in the original code.
    """

    def __init__(self, model_config: Any) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.gamma = 0.99
        self.gail = bool(self.encoder.gail)
        self.alpha = float(self.encoder.alpha)

        self.use_kl_penalty = bool(self.encoder.use_kl_penalty)
        self.use_lcf = bool(self.encoder.use_lcf)
        self.use_gradient_penalty = bool(self.token_processor.use_gradient_penalty)
        self.pred_init = bool(self.token_processor.pred_init)

        decoder = self.encoder.agent_encoder.interative_decoder
        self.gail_start_step = int(decoder.gail_start_step)
        self.dis_start_step = int(decoder.dis_start_step)

        if self.use_kl_penalty:
            self.bc_net = self._frozen_copy(self.encoder.agent_encoder)
            if self.encoder.map_encoder.type_pt_emb.weight.requires_grad:
                self.bc_map_net = self._frozen_copy(self.encoder.map_encoder)
            else:
                self.bc_map_net = None

        if self.gail:
            self.return_meanstd = RunningMeanStdTorch(shape=(1,))

        init_uses_gan = self.pred_init and self.encoder.init_decoder.use_gan
        if self.gail or init_uses_gan:
            self.automatic_optimization = False

    @staticmethod
    def _frozen_copy(module: torch.nn.Module) -> torch.nn.Module:
        frozen = copy.deepcopy(module)
        if hasattr(frozen, "target_net"):
            frozen.target_net = True
        frozen.requires_grad_(False)
        frozen.eval()
        return frozen

    def _log_train(self, name: str, value: Any) -> None:
        """Log every 50 optimizer steps and safely handle empty tensors."""

        if self.global_step % 50 != 0:
            return
        if torch.is_tensor(value) and value.numel() == 0:
            return
        self.log(name, value, on_step=True, batch_size=1)

    # ------------------------------------------------------------------
    # Supervised policy / initialization loss
    # ------------------------------------------------------------------
    def get_pred(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
        key: str = "expert",
    ) -> tuple[Tensor, Tensor]:
        """Compute policy NLL and optional initial-state loss.

        Returns:
            total_loss: scalar tensor.
            selected_log_prob: one-dimensional tensor; empty when unavailable.
        """
        prediction = self.encoder(tokenized_map, tokenized_agent)
        reference = self._prediction_reference(prediction, tokenized_agent)

        policy_loss, log_prob = self._policy_nll(prediction, tokenized_agent, key, reference)
        init_loss = self._initial_prediction_loss(prediction, tokenized_agent, reference)
        return policy_loss + init_loss, log_prob

    @staticmethod
    def _prediction_reference(prediction: Mapping[str, Any], agent: Mapping[str, Any]) -> Tensor:
        for value in prediction.values():
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    if torch.is_tensor(item):
                        return item
        for value in agent.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("Could not find a tensor to determine device and dtype.")

    def _policy_nll(
        self,
        prediction: Mapping[str, Any],
        agent: TensorDict,
        key: str,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = prediction.get("next_token_logits")
        if logits is None:
            return _zero(reference), reference.new_empty((0,))

        start_step = 0 if key == "expert" else self.gail_start_step
        valid_mask = agent["valid_mask"][:, start_step:].bool()
        actions = agent["sampled_idx"][:, start_step + 1 :].long()
        state_action_mask = get_train_mask(agent, start_step).bool()

        explicit_agent_mask = agent.get("train_mask")
        if explicit_agent_mask is not None:
            explicit_agent_mask = explicit_agent_mask.bool()
            valid_mask = valid_mask[explicit_agent_mask]
            actions = actions[explicit_agent_mask]

        flat_sa_mask = state_action_mask.reshape(-1)
        flat_actions = actions.transpose(0, 1).reshape(-1)

        if key == "expert":
            # A transition is valid only when its source frame is valid.
            state_valid = valid_mask[:, :-1].transpose(0, 1).reshape(-1)
            logit_mask = flat_sa_mask[state_valid]
        else:
            logit_mask = flat_sa_mask

        selected_logits = logits[logit_mask] / self.alpha
        selected_actions = flat_actions[flat_sa_mask]

        log_policy = selected_logits.log_softmax(dim=-1)
        policy = log_policy.exp()
        log_prob = log_policy.gather(1, selected_actions[:, None]).squeeze(1)
        entropy = -(policy * log_policy).sum(dim=-1)
        nll = -log_prob.mean()

        self._log_train(f"train/{key}_nll", nll)
        self._log_train(f"train/{key}_entropy", entropy.mean())
        return nll, log_prob

    def _initial_prediction_loss(
        self,
        prediction: Mapping[str, Any],
        agent: TensorDict,
        reference: Tensor,
    ) -> Tensor:
        result = prediction.get("initial_logit")
        if result is None:
            return _zero(reference)

        init_decoder = self.encoder.init_decoder
        if init_decoder.use_gan:
            return init_decoder.D.gan_update(
                self.log,
                self.optimizers(),
                init_decoder.G1,
                result,
            )

        if init_decoder.learn_autoencoder:
            (
                reconstruction_loss,
                agent_loss,
                kl_loss,
                pos_loss,
                heading_loss,
                shape_loss,
                vel_loss,
                _,
            ) = result
            metrics = {
                "rec_loss": reconstruction_loss,
                "agent_loss": agent_loss,
                "kl_loss": kl_loss,
                "pos_loss": pos_loss,
                "heading_loss": heading_loss,
                "shape_loss": shape_loss,
                "vel_loss": vel_loss,
            }
            for name, value in metrics.items():
                self._log_train(f"train/{name}", _safe_mean(value, reference))
            return reconstruction_loss

        match_loss, collision_loss, pos_loss, heading_loss, shape_loss, vel_loss = result
        metrics = {
            "match_loss": match_loss,
            "pos_loss": pos_loss,
            "heading_loss": heading_loss,
            "shape_loss": shape_loss,
            "vel_loss": vel_loss,
            "col_loss": collision_loss,
        }
        for name, value in metrics.items():
            self._log_train(f"train/{name}", _safe_mean(value, reference))

        noise_std = agent.get("noise_std")
        if noise_std is not None:
            std = noise_std[:, 0].mean(0)
        else:
            std = init_decoder.G1.model.normal_scale[0]
        for name, value in zip(
            ("pos_std", "heading_std", "shape_std", "vel_std"),
            std.reshape(-1, 2).mean(-1),
        ):
            self._log_train(f"train/{name}", value)

        return match_loss + collision_loss

    # ------------------------------------------------------------------
    # Discriminator
    # ------------------------------------------------------------------
    def get_reward(
        self,
        tokenized_agent: TensorDict,
        key: str,
        dis_mask: Optional[Tensor] = None,
    ):
        """Compute discriminator loss and rewards.

        Rewards returned for policy optimization are detached, preventing actor
        backward from entering the discriminator graph.
        """

        if key not in {"expert", "agent"}:
            raise ValueError(f"Unsupported discriminator sample type: {key!r}")

        discriminator = self.encoder.discriminator
        agent = tokenized_agent
        mask_t = self._discriminator_mask(agent)

        if dis_mask is None:
            dis_mask = mask_t.reshape(-1)
        else:
            dis_mask = dis_mask.bool()
        agent["dis_mask"] = dis_mask

        sampled_pos = agent["sampled_pos"]
        sampled_heading = agent["sampled_heading"]
        shape = agent["shape"][..., :2]

        use_gp = self.use_gradient_penalty and key == "expert" and discriminator.training
        if use_gp:
            sampled_pos = sampled_pos.detach().requires_grad_(True)
            sampled_heading = sampled_heading.detach().requires_grad_(True)
            shape = shape.detach().requires_grad_(True)

        disc_out = discriminator.predict_agent(
            agent["sampled_idx"],
            agent["token_mask"],
            agent["valid_mask"],
            sampled_pos,
            sampled_heading,
            agent,
            agent["map_feature"],
            shape,
        )

        (ego_logits, interaction_logits, map_logits) = disc_out[0]
        ego_rewards, neighbour_rewards, scene_reward, interaction_reward = disc_out[2]

        if not discriminator.training:
            return _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards").detach()

        target = 1.0 if key == "expert" else 0.0
        ego_logits = _select_ego_logits(ego_logits, dis_mask, mask_t)
        ego_loss = _weighted_bce_with_logits(
            logits=ego_logits,
            target=target,
            weight=torch.ones_like(ego_logits),
        )
        combined_logits = [ego_logits.reshape(-1)]

        self._log_train(f"train/{key}_ego_score", torch.sigmoid(ego_logits).mean())

        if _has_elements(map_logits):
            map_logits = _select_ego_logits(map_logits, dis_mask, mask_t)
            map_loss = _weighted_bce_with_logits(
                logits=map_logits,
                target=target,
                weight=torch.ones_like(map_logits),
            )
            ego_loss = ego_loss + map_loss
            combined_logits.append(map_logits.reshape(-1))
            self._log_train(f"train/{key}_map_score", torch.sigmoid(map_logits).mean())

        interaction_weight = None
        if _has_elements(interaction_logits):
            interaction_weight = disc_out[3].detach()
            interaction_loss = _weighted_bce_with_logits(
                logits=interaction_logits,
                target=target,
                weight=interaction_weight,
            )
            combined_logits.append(interaction_logits.reshape(-1))
            self._log_train(
                f"train/{key}_inter_score",
                torch.sigmoid(interaction_logits).mean(),
            )
        else:
            interaction_loss = _zero(ego_logits)

        discriminator_loss = ego_loss + interaction_loss
        combined = torch.cat(combined_logits)
        probabilities = combined.sigmoid()
        self._log_train(f"train/{key}_disc_val", probabilities.mean())
        self._log_train(
            f"train/{key}_disc_val_std",
            probabilities.std(unbiased=False),
        )

        ego_reward_grid = _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards")
        neighbour_reward_grid = None
        if _has_elements(neighbour_rewards):
            neighbour_reward_grid = _reshape_valid_rewards(
                neighbour_rewards,
                mask_t,
                "nei_rewards",
            )

        self._log_train(f"train/{key}_rewards", ego_reward_grid.mean())
        if neighbour_reward_grid is not None:
            self._log_train(f"train/{key}_nei_rewards", neighbour_reward_grid.mean())
            self._log_train(
                f"train/{key}_all_rewards",
                (ego_reward_grid + neighbour_reward_grid).mean(),
            )
        if _has_elements(scene_reward):
            self._log_train(f"train/{key}_scene_reward", scene_reward.mean())
        if _has_elements(interaction_reward):
            self._log_train(f"train/{key}_interact_reward", interaction_reward.mean())

        gp_loss = _zero(ego_logits)
        if use_gp:
            gp_valid_mask = agent["valid_mask"].bool().clone()
            gp_valid_mask[:, : self.dis_start_step] = False

            # (
            #     regularization_loss,
            #     penalty_pos,
            #     penalty_head,
            #     penalty_shape,
            # ) = ZeroCenteredGradientPenalty(
            #     sampled_pos=sampled_pos,
            #     sampled_heading=sampled_heading,
            #     shape=shape,
            #     critic_score=combined.sum(),#[combined_logits.abs()>1]
            #     valid_mask=gp_valid_mask,
            #     gamma=0.01,
            # )
            # #
            # self._log_train(f"train/{key}_gp", regularization_loss)
            # self._log_train(f"train/{key}_pos_gp", penalty_pos)
            # self._log_train(f"train/{key}_head_gp", penalty_head)
            # self._log_train(f"train/{key}_shape_gp", penalty_shape)
            edge_index = disc_out[1][0]
            destination_index = edge_index[1]

            gp_loss, scene_gp, interaction_gp, _ = ZeroCenteredGradientPenalty_edge(
                sampled_pos=sampled_pos,
                sampled_heading=sampled_heading,
                shape=shape,
                critic_score=(
                    ego_logits,
                    interaction_logits,
                    interaction_weight,
                    destination_index,
                ),
                valid_mask=gp_valid_mask,
                gamma=0.01,
                interaction_gamma=0.01,
                position_scale=1.0,
                heading_scale=1.0,
                shape_scale=1.0,
                interaction_min_mass=1,
                detach_edge_weight=True,
            )
            self._log_train(f"train/{key}_gp", gp_loss)
            self._log_train("train/scene_gp", scene_gp)
            self._log_train("train/interaction_gp", interaction_gp)

        # Rewards are targets, never differentiable critic outputs for the actor.
        return (
            discriminator_loss,
            ego_reward_grid.detach(),
            None if neighbour_reward_grid is None else neighbour_reward_grid.detach(),
            gp_loss,
            dis_mask,
        )

    def _discriminator_mask(self, agent: Mapping[str, Any]) -> Tensor:
        mask_t = agent["valid_mask"].bool().transpose(0, 1)[self.dis_start_step :]
        if not self.pred_init:
            train_mask = agent.get("train_mask")
            if train_mask is not None:
                mask_t = mask_t[:, train_mask.bool()]
        return mask_t

    # ------------------------------------------------------------------
    # Complete update
    # ------------------------------------------------------------------
    def update(self, tokenized_map: TensorDict, tokenized_agent: TensorDict) -> Tensor:
        self._initialize_reference_model_once()

        expert_agent = self._prepare_expert_agent(tokenized_agent)
        expert_nll = self._expert_loss(tokenized_map, expert_agent)
        if not self.gail:
            return expert_nll

        if self.pred_init:
            expert_agent["pred_mask"] = None
        else:
            expert_agent["train_mask"] = (
                expert_agent["pred_mask"].bool()
                & expert_agent["token_mask"][:, self.start_step :].all(dim=1)
            )

        expert_dis_loss, _, _, expert_gp, expert_dis_mask = self.get_reward(
            expert_agent,
            "expert",
        )

        rollout_agent = rollout(
            self.encoder,
            tokenized_map,
            expert_agent,
            self.validation_rollout_sampling,
        )
        agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
            rollout_agent,
            "agent",
            expert_dis_mask,
        )

        critic_loss = expert_dis_loss + agent_dis_loss + expert_gp + agent_gp
        self._log_train("train/critic_loss", critic_loss)

        actor_optimizer, discriminator_optimizer, init_optimizer = self._optimizers()
        self._optimizer_step(discriminator_optimizer, critic_loss)

        policy_loss, advantages_flat, advantages_2d = self._actor_value_loss(
            tokenized_map,
            rollout_agent,
            agent_rewards,
            expert_nll,
        )
        self._optimizer_step(actor_optimizer, policy_loss)

        if init_optimizer is not None:
            self._initial_state_update(
                tokenized_agent,
                advantages_flat,
                advantages_2d,
                init_optimizer,
            )

        # Returned only for progress reporting. Both optimizers already stepped.
        return critic_loss.detach() + policy_loss.detach()

    def _initialize_reference_model_once(self) -> None:
        generator = self.encoder.init_decoder.G1
        if self.global_step != 0 or not generator.use_ref:
            return
        with torch.no_grad():
            for source, target in zip(
                generator.model.parameters(),
                generator.ref_model.parameters(),
                strict=True,
            ):
                target.copy_(source)
        generator.ref_model.requires_grad_(False)
        generator.ref_model.eval()

    def _prepare_expert_agent(self, tokenized_agent: TensorDict) -> TensorDict:
        if self.gail:
            length = int(self.training_rollout_len)
            for key in (
                "sampled_pos",
                "sampled_heading",
                "sampled_idx",
                "valid_mask",
                "token_mask",
            ):
                tokenized_agent[key] = tokenized_agent[key][:, :length]
        return tokenized_agent

    def _expert_loss(self, tokenized_map: TensorDict, agent: TensorDict) -> Tensor:
        if self.use_kl_penalty:
            map_feature = self.encoder.map_encoder(tokenized_map)
            agent["map_feature"] = map_feature
            agent["detach_map_feature"] = {
                key: value.detach() for key, value in map_feature.items()
            }
            reference = next(iter(map_feature.values()))
            return _zero(reference)
        expert_nll, _ = self.get_pred(tokenized_map, agent, key="expert")
        return expert_nll

    def _optimizers(self):
        optimizers = tuple(self.optimizers())
        expected = 3 if self.token_processor.learn_init else 2
        if len(optimizers) != expected:
            raise RuntimeError(
                f"Expected {expected} optimizers, received {len(optimizers)}."
            )
        if expected == 3:
            actor, discriminator, initial = optimizers
        else:
            actor, discriminator = optimizers
            initial = None
        return actor, discriminator, initial

    def _optimizer_step(self, optimizer, loss: Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        optimizer.step()

    def _actor_value_loss(
        self,
        tokenized_map: TensorDict,
        rollout_agent: TensorDict,
        rewards: Tensor,
        expert_nll: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        edge_encoder = self.encoder.agent_encoder.interative_decoder.edge_encoder
        old_rollout_flag = edge_encoder.rollout_traj
        edge_encoder.rollout_traj = True
        try:
            _, agent_log_prob = self.get_pred(tokenized_map, rollout_agent, key="agent")
        finally:
            edge_encoder.rollout_traj = old_rollout_flag

        value = self._value_predictions(rollout_agent)
        advantages_2d, value_loss_elements = compute_advantages(
            rewards[-len(value) :],
            value,
        )

        advantages_flat = advantages_2d.reshape(-1)
        self.return_meanstd.update(advantages_flat.detach())
        advantages_flat = self.return_meanstd.normalize(advantages_flat)

        if self.training_rollout_len > 1 and agent_log_prob.numel():
            ppo_advantages = advantages_flat[-agent_log_prob.numel() :].detach()
            ppo_loss = -(agent_log_prob * ppo_advantages).mean()
        else:
            ppo_loss = _zero(value)

        if self.token_processor.learn_init:
            value_loss = value_loss_elements[1:].mean()
            init_value_loss = value_loss_elements[:1].mean()
            self._log_train("train/init_value_loss", init_value_loss)
        else:
            value_loss = value_loss_elements.mean()
            init_value_loss = _zero(value)

        self._log_train("train/ppo_loss", ppo_loss)
        self._log_train("train/running_mean", self.return_meanstd.mean)
        self._log_train("train/running_var", self.return_meanstd.var)
        self._log_train("train/value_loss", value_loss)

        policy_loss = expert_nll + ppo_loss + 1e-3 * value_loss + init_value_loss
        return policy_loss, advantages_flat, advantages_2d

    def _value_predictions(self, rollout_agent: TensorDict) -> Tensor:
        initial_value = None
        if self.token_processor.learn_init:
            initial_value = self.encoder.init_value_network(
                rollout_agent["noise_feat"]
            )[:, 0]

        if "feat_a" in rollout_agent:
            batch_size = len(rollout_agent["batch"])
            values = self.encoder.value_network(rollout_agent["feat_a"])[..., 0]
            values = values.reshape(-1, batch_size)
            if initial_value is not None:
                values = torch.cat((initial_value[None], values), dim=0)
            return values

        if initial_value is None:
            raise KeyError("Rollout must contain 'feat_a' when learn_init is disabled.")
        return initial_value

    def _initial_state_update(
        self,
        tokenized_agent: TensorDict,
        advantages_flat: Tensor,
        advantages_2d: Tensor,
        optimizer,
    ) -> None:
        normalized = advantages_flat.view_as(advantages_2d)
        tokenized_agent["advantages"] = normalized[0].detach()

        match_loss, col_loss, pos_loss, heading_loss, shape_loss, vel_loss = (
            self.encoder.init_decoder(tokenized_agent)
        )
        rl_loss = tokenized_agent["rl_loss"]
        reference = match_loss
        metrics = {
            "match_loss": match_loss,
            "pos_loss": pos_loss,
            "heading_loss": heading_loss,
            "shape_loss": shape_loss,
            "vel_loss": vel_loss,
            "rl_loss": rl_loss,
            "col_loss": col_loss,
        }
        for name, value in metrics.items():
            self._log_train(f"train/{name}", _safe_mean(value, reference))

        self._optimizer_step(optimizer, match_loss + rl_loss + col_loss)

    # ------------------------------------------------------------------
    # Lightning entry point
    # ------------------------------------------------------------------
    def training_step(self, data: Any, batch_idx: int) -> Tensor:
        tokenized_map, tokenized_agent = self.token_processor(data)

        loss = self.update(tokenized_map, tokenized_agent)
        self._log_train("train/loss", loss)

        for key in ("sampled_match_loss", "clip_ratio", "noncol_rate"):
            if key in tokenized_agent:
                self._log_train(f"train/{key}", tokenized_agent[key].mean())
        if "pg_loss" in tokenized_agent:
            self._log_train("train/pg_loss", tokenized_agent["pg_loss"])
        return loss