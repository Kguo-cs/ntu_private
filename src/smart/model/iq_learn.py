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
from src.smart.loss.gp_penalty import _select_ego_logits,_weighted_bce_with_logits,_has_elements,_reshape_valid_rewards,ZeroCenteredGradientPenalty


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

        if self.gail or (self.pred_init and self.encoder.init_decoder.use_gan):

            self.automatic_optimization=False

    def _log_train(self, name, value):
        """Throttle train logging to avoid W&B broken-pipe / socket overload."""
        if self.global_step % 50 == 0:
            return self.log(name, value, on_step=True,batch_size=1)

    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        if not self.pred_init:
            tokenized_agent["train_mask"] = None

        pred = self.encoder(tokenized_map, tokenized_agent)

        if key=="agent" and not self.pred_init:
            tokenized_agent["train_mask"]=tokenized_agent["pred_mask"]

        if "next_token_logits" in pred.keys():
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

            self._log_train("train/" + key + "_nll", action_nll)
            self._log_train("train/" + key + "_entropy", entropy.mean())
        else:
            action_nll=log_prob=0

        if "initial_logit" in pred.keys():

            if self.encoder.init_decoder.use_gan:
                # noncol_rate = get_col_rate(tokenized_agent, pred["initial_logit"][1])
                #
                # self._log_train('train/noncol_rate', noncol_rate.mean())

                loss=self.encoder.init_decoder.D.gan_update( self.log,self.optimizers(), self.encoder.init_decoder.G1,pred["initial_logit"])
            elif self.encoder.init_decoder.learn_autoencoder:
                rec_loss, agent_loss, kl_loss,pos_loss,heading_loss,shape_loss,vel_loss,_=pred[   "initial_logit"]
                self._log_train('train/rec_loss', rec_loss)
                self._log_train('train/agent_loss', agent_loss)
                self._log_train('train/kl_loss', kl_loss)
                self._log_train('train/pos_loss', pos_loss)
                self._log_train('train/heading_loss', heading_loss)
                self._log_train('train/shape_loss', shape_loss)
                self._log_train('train/vel_loss', vel_loss)

                loss=rec_loss
            else:
                match_loss,  collision_loss, pos_loss, heading_loss, shape_loss, vel_loss = pred[
                    "initial_logit"]
                self._log_train('train/match_loss', match_loss)
                self._log_train('train/pos_loss', pos_loss)
                self._log_train('train/heading_loss', heading_loss)
                self._log_train('train/shape_loss', shape_loss)
                self._log_train('train/vel_loss', vel_loss)
                self._log_train('train/collision_loss', collision_loss)

                if "noncol_rate" in tokenized_agent.keys():
                    self._log_train('train/noncol_rate', tokenized_agent["noncol_rate"].mean())

                # if self.encoder.init_decoder.G1.model.learn_schedule:
                # gamma_groups=self.encoder.init_decoder.G1.model.schedule.gamma_groups
                # self._log_train('train/pos_gamma', gamma_groups[0])
                # self._log_train('train/heading_gamma', gamma_groups[1])
                # self._log_train('train/shape_gamma', gamma_groups[2])
                # self._log_train('train/vel_gamma', gamma_groups[3])

                if "noise_std" in tokenized_agent.keys():
                    std=tokenized_agent["noise_std"][:,0].mean(0)
                else:
                    std=self.encoder.init_decoder.G1.model.normal_scale[0]

                self._log_train('train/pos_std', std[0:2].mean())
                self._log_train('train/heading_std', std[2:4].mean())
                self._log_train('train/shape_std', std[4:6].mean())
                self._log_train('train/vel_std', std[6:8].mean())


                loss = match_loss + collision_loss

            action_nll = action_nll + loss

        if "pos_loss" in pred.keys():
            pos_loss=pred["pos_loss"]
            heading_loss=pred["heading_loss"]

            self._log_train('train/pos_loss', pos_loss)
            self._log_train('train/heading_loss', heading_loss)

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
            weight=torch.ones_like(ego_logits)
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

        disc_loss =  ego_bce_loss +interact_bce_loss

        # Reshape only policy rewards because these are consumed by the policy
        # update as a [time, agent] reward matrix.
        if key == "agent":
            ego_rewards = _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards")

            if has_nei_rewards:
                nei_rewards = _reshape_valid_rewards(nei_rewards, mask_t, "nei_rewards")

        # ----------------------------
        # Logging
        # ----------------------------
        self._log_train(   f"train/{key}_rewards",    ego_rewards.mean() )

        if _has_elements(valid_ego_reward):
            self._log_train(  f"train/{key}_valid_ego_reward", valid_ego_reward.mean() )

        if _has_elements(valid_interact_reward):
            self._log_train(  f"train/{key}_valid_interact_reward", valid_interact_reward.mean()  )

        if has_nei_rewards:
            all_rewards = ego_rewards + nei_rewards

            self._log_train(   f"train/{key}_all_rewards",all_rewards.mean() )

            self._log_train(   f"train/{key}_nei_rewards", nei_rewards.mean())

        ego_score = torch.sigmoid(ego_logits)

        self._log_train(f"train/{key}_ego_score",ego_score.mean()  )

        if has_interact_logits:
            interact_score = torch.sigmoid(interact_logits)

            self._log_train(f"train/{key}_inter_score", interact_score.mean() )

            self._log_train(  f"train/{key}_interact_logits", interact_logits.mean() )

        disc_val = torch.sigmoid(combined_logits)

        self._log_train(f"train/{key}_disc_val",   disc_val.mean() )

        self._log_train( f"train/{key}_disc_val_std", disc_val.std(unbiased=False) )
        if use_gp_this_step:
            gamma = 0.1 #l1 loss

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
                critic_score=combined_logits.sum(),#[combined_logits.abs()>1]
                valid_mask=tokenized_agent["valid_mask"],
                gamma=gamma,
            )

            self._log_train(f"train/{key}_gp", regularization_loss)
            self._log_train(f"train/{key}_pos_gp", penalty_pos)
            self._log_train(f"train/{key}_head_gp", penalty_head)
            self._log_train(f"train/{key}_shape_gp", penalty_shape)

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
                    tokenized_agent[key] = tokenized_agent[key][:, :self.training_rollout_len]

            expert_nll, expert_log_prob= self.get_QV(tokenized_map, tokenized_agent)

        if not self.gail:
            return expert_nll

       # if self.pred_init:
       #  tokenized_agent["train_mask"] = tokenized_agent["token_mask"].all(1)
       #  tokenized_agent["pred_mask"] =tokenized_agent["token_mask"].all(1)
        # else:
        #     tokenized_agent["train_mask"]=tokenized_agent["pred_mask"] #& tokenized_agent["token_mask"][:,self.start_step:].all(1)
        expert_dis_loss, _, _, expert_gp, expert_dis_mask = self.get_reward(tokenized_agent, "expert")

        tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent,  self.validation_rollout_sampling)

        # agent_train_mask= get_train_mask(tokenized_agent_rollout,self.gail_start_step)

        agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
            tokenized_agent_rollout, "agent", expert_dis_mask
        )
        # else:
        #     with torch.no_grad():
        #         agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
        #             tokenized_agent_rollout, "agent", expert_dis_mask
        #         )
        critic_loss = expert_dis_loss + agent_dis_loss + agent_gp + expert_gp

        self._log_train("train/critic_loss", critic_loss)

        if self.token_processor.learn_init:
            actor_optimizer, discriminator_optimizer, init_optimizer = self.optimizers()
        else:
            actor_optimizer, discriminator_optimizer = self.optimizers()

        discriminator_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        discriminator_optimizer.step()
        #
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

        if self.token_processor.learn_init:
            intial_value=self.encoder.init_value_network(tokenized_agent_rollout["noise_feat"])[:,0]


        if 'feat_a' in tokenized_agent_rollout.keys():
            feat_a = tokenized_agent_rollout["feat_a"]#.detach()

            value = self.encoder.value_network(feat_a)[..., 0].view(-1,len(tokenized_agent_rollout["batch"]))

            value=torch.cat([intial_value[None],value],dim=0)
        else:
            value = intial_value


        advantages_2d, all_value_loss = compute_advantages(
            agent_rewards[-len(value):],
            value,
            #infinite_horizon=True
        )

        advantages_flat = advantages_2d.reshape(-1)
        self.return_meanstd.update(advantages_flat.detach())
        advantages_flat = self.return_meanstd.normalize(advantages_flat)

        if self.training_rollout_len>1:
            ppo_advantages = advantages_flat[-len(agent_log_prob):]
            ppo_loss = -(agent_log_prob * ppo_advantages).mean()
            self._log_train("train/ppo_loss", ppo_loss)
            self._log_train("train/advantages", ppo_advantages.mean())
        else:
            ppo_loss=0

        if self.token_processor.learn_init:
            value_loss=all_value_loss[1:].mean()
            init_value_loss=all_value_loss[:1].mean()
            self._log_train("train/init_value_loss", init_value_loss)
        else:
            value_loss=all_value_loss.mean()
            init_value_loss=0

        self._log_train("train/running_mean", self.return_meanstd.mean)
        self._log_train("train/running_var", self.return_meanstd.var)
        self._log_train("train/value_loss", value_loss)

        policy_loss = expert_nll + ppo_loss + 1e-3 * value_loss+init_value_loss  #  agent_entropy.mean()

        actor_optimizer.zero_grad()

        self.manual_backward(policy_loss)

        actor_optimizer.step()

        if self.token_processor.learn_init:
            # noncol_rate=get_col_rate(tokenized_agent,tokenized_agent["pred_init"])
            #
            # self._log_train('train/noncol_rate', noncol_rate.mean())

            # For learn_init update: use the initial-state row explicitly.
            advantages_2d_norm = advantages_flat.view_as(advantages_2d)
            init_advantages = advantages_2d_norm[0].detach()

            # init_return = agent_rewards.sum(dim=0).detach()
            #
            # non_ego = ~tokenized_agent_rollout["ego_mask"]
            # init_advantages = torch.zeros_like(init_return)
            #
            # ret = init_return[non_ego]
            # init_advantages[non_ego] = (
            #                                    ret - ret.mean()
            #                            ) / ret.std(unbiased=False).clamp_min(1e-8)
            # Optional: normalize only non-ego init advantages.
            # non_ego = ~tokenized_agent_rollout["ego_mask"]
            # init_adv_non_ego = init_advantages[non_ego]
            # init_advantages = init_advantages.clone()
            # init_advantages[non_ego] = (
            #                                    init_adv_non_ego - init_adv_non_ego.mean()
            #                            ) / init_adv_non_ego.std(unbiased=False).clamp_min(1e-8)

            tokenized_agent["advantages"] = init_advantages

            match_loss, g_loss, pos_loss, heading_loss, shape_loss, vel_loss=self.encoder.init_decoder(tokenized_agent)

            self._log_train('train/match_loss', match_loss)
            self._log_train('train/pos_loss', pos_loss)
            self._log_train('train/heading_loss', heading_loss)
            self._log_train('train/shape_loss', shape_loss)
            self._log_train('train/vel_loss', vel_loss)
            self._log_train('train/g_loss', g_loss)
            self._log_train('train/sampled_match_loss', tokenized_agent["sampled_match_loss"].mean())
            self._log_train('train/policy_loss', tokenized_agent["policy_loss"])
            if "ratio" in tokenized_agent.keys():
                self._log_train('train/ratio_std', tokenized_agent["ratio"].std())

            init_loss=match_loss+g_loss

            init_optimizer.zero_grad()

            self.manual_backward(init_loss)

            init_optimizer.step()

        loss = critic_loss + policy_loss

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        if "max_dist"  in   tokenized_agent.keys():
            max_dist=tokenized_agent["max_dist"]
            reset_mask=tokenized_agent["reset_mask"]
            token_mask=tokenized_agent["token_mask"]

            self._log_train("train/mean_token_error", max_dist.mean().detach())
            self._log_train("train/reset_mask", reset_mask[token_mask].float().mean().detach())

        if "entry_token_invalid_mask"  in   tokenized_agent.keys():
            entry_token_invalid_mask=tokenized_agent["entry_token_invalid_mask"]

            self._log_train("train/entry_token_invalid", entry_token_invalid_mask.float().mean().detach())

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self._log_train("train/loss", loss)

        return loss