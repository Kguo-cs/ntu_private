from lightning import LightningModule
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from collections import deque
import random
import copy

from src.smart.loss.rollout_buffer import RunningMeanStdTorch,rollout, compute_advantages,get_train_mask
from src.smart.loss.gp_penalty import compute_gp,_select_ego_logits,_weighted_bce_with_logits,_has_elements
from src.smart.loss.earth_match import get_matching_loss,multi_circle_collision_loss_mem_efficient,get_scale,get_col_rate
from torch_scatter import scatter_sum,scatter_mean

def _masked_square_norm_mean(gradient, mask, reference):
    """Mean squared gradient norm over valid entries only."""
    if gradient is None or mask is None or not torch.any(mask):
        return reference.new_zeros(())

    valid_gradient = gradient[mask]
    return valid_gradient.reshape(valid_gradient.shape[0], -1).square().sum(dim=-1).mean()


def ZeroCenteredGradientPenalty(
        sampled_pos,
        sampled_heading,
        shape,
        critic_score,
        valid_mask,
        gamma=0.01,
):
    """Zero-centered R2 penalty on generated trajectories.

    The discriminator locally differentiates through the fixed graph topology.
    Padding entries are excluded from the normalization.
    """
    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=(sampled_pos, sampled_heading, shape),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    grad_pos, grad_heading, grad_shape = gradients
    valid_mask = valid_mask.bool()
    valid_agent_mask = valid_mask.any(dim=-1)

    pos_penalty = _masked_square_norm_mean(grad_pos, valid_mask, critic_score)
    heading_penalty = _masked_square_norm_mean(grad_heading, valid_mask, critic_score)
    shape_penalty = _masked_square_norm_mean(grad_shape, valid_agent_mask, critic_score)

    scale = gamma / 2.0
    return (
        scale * (pos_penalty + heading_penalty + shape_penalty*10),
        scale * pos_penalty,
        scale * heading_penalty,
        scale * shape_penalty,
    )


def _reshape_valid_rewards(rewards, mask_t, name):
    """Restore compressed valid-node rewards to a dense [T, A] tensor."""
    if not torch.is_tensor(rewards):
        raise TypeError(f"{name} must be a tensor, got {type(rewards)}")

    if rewards.numel() == mask_t.numel():
        return rewards.reshape(mask_t.shape)

    valid_count = int(mask_t.sum().item())
    if rewards.numel() != valid_count:
        raise ValueError(
            f"Cannot align {name} with mask_t: rewards.numel()={rewards.numel()}, "
            f"mask_t.numel()={mask_t.numel()}, mask_t.sum()={valid_count}."
        )

    dense_rewards = rewards.new_zeros(mask_t.shape)
    dense_rewards[mask_t] = rewards
    return dense_rewards

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)
        self.gamma = 0.99
        self.gail = self.encoder.gail
        self.alpha = self.encoder.alpha

        self.buffer_len = 1

        self.replay_buffer = deque(maxlen=self.buffer_len)

        self.use_kl_penalty = self.encoder.use_kl_penalty

        if self.use_kl_penalty:
            self.bc_net = copy.deepcopy(self.encoder.agent_encoder)

            self.bc_net.target_net = True
            for param in self.bc_net.parameters():
                param.requires_grad = False
            self.bc_net.eval()

            if self.encoder.map_encoder.type_pt_emb.weight.requires_grad:
                self.bc_map_net = copy.deepcopy(self.encoder.map_encoder)

                for param in self.bc_map_net.parameters():
                    param.requires_grad = False

                self.bc_map_net.eval()
            else:
                self.bc_map_net = None

        if self.gail:
            self.return_meanstd = RunningMeanStdTorch(shape=(1))
            # self.ego_return_meanstd = RunningMeanStdTorch(shape=(1))
            # self.global_return_meanstd = RunningMeanStdTorch(shape=(1))

        self.use_lcf = self.encoder.use_lcf
        self.use_gradient_penalty = self.token_processor.use_gradient_penalty
        self.pred_init=self.token_processor.pred_init

        self.gail_start_step= self.encoder.agent_encoder.interative_decoder.gail_start_step
        self.dis_start_step = self.encoder.agent_encoder.interative_decoder.dis_start_step


        if self.gail or (self.pred_init and self.encoder.agent_encoder.init_decoder.use_gan):

            self.automatic_optimization=False


    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        if not self.pred_init:
            tokenized_agent["train_mask"] = None

        pred = self.encoder(tokenized_map, tokenized_agent)

        if key=="agent" and not self.pred_init:
            tokenized_agent["train_mask"]=tokenized_agent["pred_mask"]

        if pred["next_token_logits"] is not None:
            if key=='expert':
                start_step=0
            else:
                start_step=self.gail_start_step

            valid_mask = tokenized_agent["valid_mask"][:, start_step:]
            action = tokenized_agent["sampled_idx"][:, start_step + 1:]

            train_mask = get_train_mask(tokenized_agent, start_step)  # t,a

            if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
                agent_train_mask=tokenized_agent["train_mask"]
                valid_mask=valid_mask[agent_train_mask]
                action=action[agent_train_mask]

            train_mask=train_mask.flatten(0, 1)

            action=action.transpose(0, 1).flatten(0, 1)[train_mask]

            next_token_logits=pred["next_token_logits"]

            if key == "expert":
                action_valid=train_mask[valid_mask[:,:-1].transpose(0, 1).flatten(0, 1)]
            else:
                action_valid=train_mask

            next_token_logits=next_token_logits[action_valid] / self.alpha

            pi = torch.softmax(next_token_logits, dim=-1)

            logpi = torch.log(pi + 1e-10)

            log_prob = torch.gather(logpi, dim=-1, index=action.unsqueeze(-1)).squeeze(-1) #t,a

            entropy = -torch.sum(pi * logpi, dim=-1)

            action_nll=-log_prob.mean()

            self.log("train/" + key + "_nll", action_nll, on_step=True, batch_size=1)
            self.log("train/" + key + "_entropy", entropy.mean(), on_step=True, batch_size=1)
        else:
            action_nll=log_prob=0

        if pred["initial_logit"] is not None:

            if self.encoder.agent_encoder.init_decoder.use_gan:
                # noncol_rate = get_col_rate(tokenized_agent, pred["initial_logit"][1])
                #
                # self.log('train/noncol_rate', noncol_rate.mean(), on_step=True, batch_size=1)

                loss=self.encoder.agent_encoder.init_decoder.D.gan_update( self.log,self.optimizers(), self.encoder.agent_encoder.init_decoder.G1,pred["initial_logit"])
            elif self.encoder.agent_encoder.init_decoder.learn_autoencoder:
                rec_loss, agent_loss, kl_loss,pos_loss,heading_loss,shape_loss,vel_loss,_=pred[   "initial_logit"]
                self.log('train/rec_loss', rec_loss, on_step=True, batch_size=1)
                self.log('train/agent_loss', agent_loss, on_step=True, batch_size=1)
                self.log('train/kl_loss', kl_loss, on_step=True, batch_size=1)
                self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
                self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)
                self.log('train/shape_loss', shape_loss, on_step=True, batch_size=1)
                self.log('train/vel_loss', vel_loss, on_step=True, batch_size=1)

                loss=rec_loss
            else:
                match_loss,  collision_loss, pos_loss, heading_loss, shape_loss, vel_loss = pred[
                    "initial_logit"]
                self.log('train/match_loss', match_loss, on_step=True, batch_size=1)
                self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
                self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)
                self.log('train/shape_loss', shape_loss, on_step=True, batch_size=1)
                self.log('train/vel_loss', vel_loss, on_step=True, batch_size=1)
                self.log('train/collision_loss', collision_loss, on_step=True, batch_size=1)

                if "noncol_rate" in tokenized_agent.keys():
                    self.log('train/noncol_rate', tokenized_agent["noncol_rate"].mean(), on_step=True, batch_size=1)

                # if self.encoder.agent_encoder.init_decoder.G1.model.learn_schedule:
                gamma_groups=self.encoder.agent_encoder.init_decoder.G1.model.schedule.gamma_groups
                self.log('train/pos_gamma', gamma_groups[0], on_step=True, batch_size=1)
                self.log('train/heading_gamma', gamma_groups[1], on_step=True, batch_size=1)
                self.log('train/shape_gamma', gamma_groups[2], on_step=True, batch_size=1)
                self.log('train/vel_gamma', gamma_groups[3], on_step=True, batch_size=1)

                if "noise_std" in tokenized_agent.keys():
                    std=tokenized_agent["noise_std"][:,0].mean(0)
                else:
                    std=self.encoder.agent_encoder.init_decoder.G1.model.normal_scale[0]

                self.log('train/pos_std', std[0:2].mean(), on_step=True, batch_size=1)
                self.log('train/heading_std', std[2:4].mean(), on_step=True, batch_size=1)
                self.log('train/shape_std', std[4:6].mean(), on_step=True, batch_size=1)
                self.log('train/vel_std', std[6:8].mean(), on_step=True, batch_size=1)


                loss = match_loss + collision_loss

            action_nll = action_nll + loss

        if "pos_loss" in pred.keys():
            pos_loss=pred["pos_loss"]
            heading_loss=pred["heading_loss"]

            self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
            self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)

            action_nll=pos_loss+heading_loss

        return action_nll,log_prob

    def get_reward(
            self,
            tokenized_agent: dict,
            key: str,
            dis_mask: torch.Tensor | None = None,
    ):
        discriminator = self.encoder.discriminator

        # valid_mask: [A, T]
        # mask_t:     [T_selected, A_selected]
        mask_t = tokenized_agent["valid_mask"].transpose(0, 1)[
            self.dis_start_step:
        ]

        if not self.pred_init:
            train_mask = tokenized_agent.get("train_mask")

            if train_mask is not None:
                train_mask = train_mask.bool()
                mask_t = mask_t[:, train_mask]

            if dis_mask is None or (self.pred_init and key == "agent"):
                dis_mask = mask_t.flatten()

            if dis_mask is not None:
                dis_mask = dis_mask.bool()
                tokenized_agent["dis_mask"] = dis_mask

        sampled_pos = tokenized_agent["sampled_pos"]
        sampled_heading = tokenized_agent["sampled_heading"]
        shape = tokenized_agent["shape"][..., :2]

        use_gp_this_step = (
                self.use_gradient_penalty
                and key == "expert"
               # and self.global_step % 4 == 0
        )

        # The critic update must not backpropagate through rollout generation.
        if use_gp_this_step:
            sampled_pos = sampled_pos.detach().requires_grad_(True)
            sampled_heading = sampled_heading.detach().requires_grad_(True)
            shape = shape.detach().requires_grad_(True)

        disc_out = discriminator.predict_agent(
            tokenized_agent["sampled_idx"],
            tokenized_agent["token_mask"],
            tokenized_agent["valid_mask"],
            sampled_pos,
            sampled_heading,
            tokenized_agent,
            tokenized_agent["map_feature"],
            shape,
        )

        ego_logits, interact_logits = disc_out[0]

        (
            ego_rewards,
            nei_rewards,
            valid_ego_reward,
            valid_interact_reward,
        ) = disc_out[2]

        # During validation or rollout, only return the policy reward grid.
        if not discriminator.training:
            return _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards")

        # Apply the same ego-logit mask to both expert and policy samples.
        # This avoids training expert and generated samples on different subsets.
        ego_logits = _select_ego_logits(
            ego_logits=ego_logits,
            dis_mask=dis_mask,
            mask_t=mask_t,
        )

        target = 1.0 if key == "expert" else 0.0

        ego_bce_loss = _weighted_bce_with_logits(
            logits=ego_logits,
            target=target,
        )

        has_nei_rewards = _has_elements(nei_rewards)
        has_interact_logits = _has_elements(interact_logits)

        if has_interact_logits:
            interaction_weight = disc_out[3].detach()

            interact_bce_loss = _weighted_bce_with_logits(
                logits=interact_logits,
                target=target,
                weight=interaction_weight,
            )

            combined_logits = torch.cat(
                [ego_logits.reshape(-1), interact_logits.reshape(-1)],
                dim=0,
            )
        else:
            interact_bce_loss = ego_logits.new_zeros(())
            combined_logits = ego_logits.reshape(-1)

        disc_loss = ego_bce_loss + interact_bce_loss

        # Reshape only policy rewards because these are consumed by the policy
        # update as a [time, agent] reward matrix.
        if key == "agent":
            ego_rewards = _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards")

            if has_nei_rewards:
                nei_rewards = _reshape_valid_rewards(nei_rewards, mask_t, "nei_rewards")

        # ----------------------------
        # Logging
        # ----------------------------
        self.log(
            f"train/{key}_rewards",
            ego_rewards.mean(),
            on_step=True,
            batch_size=1,
        )

        if _has_elements(valid_ego_reward):
            self.log(
                f"train/{key}_valid_ego_reward",
                valid_ego_reward.mean(),
                on_step=True,
                batch_size=1,
            )

        if _has_elements(valid_interact_reward):
            self.log(
                f"train/{key}_valid_interact_reward",
                valid_interact_reward.mean(),
                on_step=True,
                batch_size=1,
            )

        if has_nei_rewards:
            all_rewards = ego_rewards + nei_rewards

            self.log(
                f"train/{key}_all_rewards",
                all_rewards.mean(),
                on_step=True,
                batch_size=1,
            )

            self.log(
                f"train/{key}_nei_rewards",
                nei_rewards.mean(),
                on_step=True,
                batch_size=1,
            )

        ego_score = torch.sigmoid(ego_logits)

        self.log(
            f"train/{key}_ego_score",
            ego_score.mean(),
            on_step=True,
            batch_size=1,
        )

        if has_interact_logits:
            interact_score = torch.sigmoid(interact_logits)

            self.log(
                f"train/{key}_inter_score",
                interact_score.mean(),
                on_step=True,
                batch_size=1,
            )

            self.log(
                f"train/{key}_interact_logits",
                interact_logits.mean(),
                on_step=True,
                batch_size=1,
            )

        disc_val = torch.sigmoid(combined_logits)

        self.log(
            f"train/{key}_disc_val",
            disc_val.mean(),
            on_step=True,
            batch_size=1,
        )

        self.log(
            f"train/{key}_disc_val_std",
            disc_val.std(unbiased=False),
            on_step=True,
            batch_size=1,
        )
        if use_gp_this_step:
            gamma = 0.01

            # critic_score = ego_logits.sum()
            # if has_interact_logits:
            #     critic_score = critic_score + (
            #         interact_logits * interaction_weight
            #     ).sum()

            (
                regularization_loss,
                penalty_pos,
                penalty_head,
                penalty_shape,
            ) = ZeroCenteredGradientPenalty(
                sampled_pos=sampled_pos,
                sampled_heading=sampled_heading,
                shape=shape,
                critic_score=combined_logits.sum(),
                valid_mask=tokenized_agent["valid_mask"],
                gamma=gamma,
            )

            self.log(f"train/{key}_gp", regularization_loss, on_step=True, batch_size=1)
            self.log(f"train/{key}_pos_gp", penalty_pos, on_step=True, batch_size=1)
            self.log(f"train/{key}_head_gp", penalty_head, on_step=True, batch_size=1)
            self.log(f"train/{key}_shape_gp", penalty_shape, on_step=True, batch_size=1)

            ego_rewards=ego_rewards.detach()
        else:
            regularization_loss = ego_logits.new_zeros(())

        return (
            disc_loss,
            ego_rewards,
            nei_rewards,
            regularization_loss,
            dis_mask,
        )

    def iq_update(self, tokenized_map, tokenized_agent):
        if self.use_kl_penalty:
            expert_nll= 0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            if self.gail:
                for key in ["sampled_pos", "sampled_heading", "sampled_idx", "valid_mask", "token_mask"]:
                    tokenized_agent[key] = tokenized_agent[key][:, :10]

            expert_nll, expert_log_prob= self.get_QV(tokenized_map, tokenized_agent)

        if not self.gail:
            return expert_nll


       # if self.pred_init:
       #  tokenized_agent["train_mask"] = tokenized_agent["token_mask"].all(1)
       #  tokenized_agent["pred_mask"] =tokenized_agent["token_mask"].all(1)
        # else:
        #     tokenized_agent["train_mask"]=tokenized_agent["pred_mask"] #& tokenized_agent["token_mask"][:,self.start_step:].all(1)
        if self.encoder.learn_dis: #and (self.global_step%4==0) :
            expert_dis_loss,_,_,expert_gp,expert_dis_mask = self.get_reward(tokenized_agent, "expert")
        else:
            expert_dis_loss=expert_gp=0
            expert_dis_mask=None


        tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent,  self.validation_rollout_sampling)

        # agent_train_mask= get_train_mask(tokenized_agent_rollout,self.gail_start_step)

        if self.encoder.learn_dis:
           # if  (self.global_step%4==0) :
            agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
                tokenized_agent_rollout, "agent", expert_dis_mask
            )
            # else:
            #     with torch.no_grad():
            #         agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
            #             tokenized_agent_rollout, "agent", expert_dis_mask
            #         )
            critic_loss = expert_dis_loss + agent_dis_loss + agent_gp+expert_gp
        else:
            critic_loss = tokenized_agent_rollout["sampled_pos"].new_zeros(())

        self.log("train/critic_loss", critic_loss, on_step=True, batch_size=1)

        if self.token_processor.learn_init:
            actor_optimizer, discriminator_optimizer, init_optimizer = self.optimizers()
        else:
            actor_optimizer, discriminator_optimizer = self.optimizers()

        if self.encoder.learn_dis :#and (self.global_step%4==0):
           # print(self.global_step)
            discriminator_optimizer.zero_grad()
            critic_loss.backward()
            discriminator_optimizer.step()
        #
        # if not self.use_gradient_penalty:
        #     with torch.no_grad():
        #         discriminator_was_training = self.encoder.discriminator.training
        #         self.encoder.discriminator.eval()
        #         agent_rewards = self.get_reward(
        #             tokenized_agent_rollout, "agent", expert_dis_mask
        #         )
        #         self.encoder.discriminator.train(discriminator_was_training)
        #
        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = True

        agent_nll, agent_log_prob = self.get_QV(tokenized_map, tokenized_agent_rollout, key='agent')

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = False

        feat_a = tokenized_agent_rollout["feat_a"].detach()

        value = self.encoder.value_network(feat_a)[..., 0].view(-1,len(tokenized_agent_rollout["batch"]))

        if self.token_processor.learn_init:
            intial_value=self.encoder.init_value_network(tokenized_agent_rollout["noise_feat"])[:,0]

            value=torch.cat([intial_value[None],value],dim=0)

        advantages_2d, all_value_loss = compute_advantages(
            agent_rewards[-len(value):],
            value,
        )

        # Normalize dense advantages.
        advantages_flat = advantages_2d.reshape(-1)
        self.return_meanstd.update(advantages_flat.detach())
        advantages_flat = self.return_meanstd.normalize(advantages_flat)

        # For PPO action-token update.
        ppo_advantages = advantages_flat[-len(agent_log_prob):]
        ppo_loss = -(agent_log_prob * ppo_advantages).mean()

        if self.token_processor.learn_init:
            value_loss=all_value_loss[1:].mean()
            init_value_loss=all_value_loss[:1].mean()
            self.log("train/init_value_loss", init_value_loss, on_step=True, batch_size=1)
        else:
            value_loss=all_value_loss.mean()
            init_value_loss=0

        self.log("train/running_mean", self.return_meanstd.mean, on_step=True, batch_size=1)
        self.log("train/running_var", self.return_meanstd.var, on_step=True, batch_size=1)
        self.log("train/ppo_loss", ppo_loss, on_step=True, batch_size=1)
        self.log("train/advantages", ppo_advantages.mean(), on_step=True, batch_size=1)
        self.log("train/value_loss", value_loss, on_step=True, batch_size=1)

        policy_loss = expert_nll + ppo_loss + value_loss+init_value_loss  #  1e-3 * agent_entropy.mean()

        actor_optimizer.zero_grad()

        policy_loss.backward()

        actor_optimizer.step()

        if self.token_processor.learn_init:
            # noncol_rate=get_col_rate(tokenized_agent,tokenized_agent["pred_init"])
            #
            # self.log('train/noncol_rate', noncol_rate.mean(), on_step=True, batch_size=1)

            # For learn_init update: use the initial-state row explicitly.
            # advantages_2d_norm = advantages_flat.view_as(advantages_2d)
            # init_advantages = advantages_2d_norm[0].detach()

            init_return = agent_rewards.sum(dim=0).detach()

            non_ego = ~tokenized_agent_rollout["ego_mask"]
            init_advantages = torch.zeros_like(init_return)

            ret = init_return[non_ego]
            init_advantages[non_ego] = (
                                               ret - ret.mean()
                                       ) / ret.std(unbiased=False).clamp_min(1e-8)
            # Optional: normalize only non-ego init advantages.
            # non_ego = ~tokenized_agent_rollout["ego_mask"]
            # init_adv_non_ego = init_advantages[non_ego]
            # init_advantages = init_advantages.clone()
            # init_advantages[non_ego] = (
            #                                    init_adv_non_ego - init_adv_non_ego.mean()
            #                            ) / init_adv_non_ego.std(unbiased=False).clamp_min(1e-8)

            tokenized_agent["advantages"] = init_advantages

            match_loss, g_loss, pos_loss, heading_loss, shape_loss, vel_loss=self.encoder.agent_encoder.init_decoder(tokenized_agent)

            self.log('train/match_loss', match_loss, on_step=True, batch_size=1)
            self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
            self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)
            self.log('train/shape_loss', shape_loss, on_step=True, batch_size=1)
            self.log('train/vel_loss', vel_loss, on_step=True, batch_size=1)
            self.log('train/g_loss', g_loss, on_step=True, batch_size=1)

            init_loss=match_loss+g_loss

            init_optimizer.zero_grad()

            init_loss.backward()

            init_optimizer.step()

        loss = critic_loss + policy_loss

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        if "max_dist"  in   tokenized_agent.keys():
            max_dist=tokenized_agent["max_dist"]
            reset_mask=tokenized_agent["reset_mask"]
            token_mask=tokenized_agent["token_mask"]

            self.log("train/mean_token_error", max_dist.mean().detach(), on_step=True, batch_size=1)
            self.log("train/reset_mask", reset_mask[token_mask].float().mean().detach(), on_step=True, batch_size=1)

        if "entry_token_invalid_mask"  in   tokenized_agent.keys():
            entry_token_invalid_mask=tokenized_agent["entry_token_invalid_mask"]

            self.log("train/entry_token_invalid", entry_token_invalid_mask.float().mean().detach(), on_step=True, batch_size=1)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss