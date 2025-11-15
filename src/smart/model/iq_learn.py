from jax.example_libraries.stax import logsoftmax
from lightning import LightningModule
import numpy as np
import torch

from src.smart.loss.iq_loss import get_iqloss, soft_update, eval_light, get_proposal_loss, get_gaussian_loss
from src.smart.loss.rollout_buffer import rollout, compute_advantages
import torch.nn.functional as F
import torch.nn as nn
import time
from collections import deque
import random
import copy
from src.smart.loss.rollout_buffer import RunningMeanStdTorch, get_reward, get_nei_returns, get_return, \
    get_near_returns, per_scene_zscore_clip
from torch_scatter import scatter_mean
from torch.distributions import Categorical, Normal, Independent
from src.smart.loss.kl_loss import BalancedKL
from src.smart.loss.collision_check import oriented_box_collision, signed_distance_boxes_sat_fast, value_to_hist_class
from src.smart.loss.offroad_check import corners_offroad_signed_distance_per_batch
from torch_scatter import scatter_max
from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from torch.nn.functional import cross_entropy


class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)
        self.gamma = 0.99
        self.iq_learn = self.encoder.iq_learn
        self.alpha = self.encoder.alpha

        self.use_target_q = False

        self.start_step = self.encoder.agent_encoder.start_step

        self.use_gail = self.encoder.use_gail
        self.bce_loss = nn.BCELoss()

        self.buffer_len = 1

        self.replay_buffer = deque(maxlen=self.buffer_len)

        self.rollout_freq = 1

        if self.use_target_q:
            self.target_net = copy.deepcopy(self.encoder.critic)
            self.target_net.load_state_dict(self.encoder.critic.state_dict())
            for param in self.target_net.parameters():
                param.requires_grad = False

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

        if self.iq_learn and self.use_gail:
            # self.running_meanstd=RunningMeanStdTorch(shape=(1))

            self.return_meanstd = RunningMeanStdTorch(shape=(1))
            self.ego_return_meanstd = RunningMeanStdTorch(shape=(1))
            self.global_return_meanstd = RunningMeanStdTorch(shape=(1))

        self.use_lcf = self.encoder.use_lcf

        self.dis_loss = "gail"

        self.learn_lcf = self.encoder.learn_lcf

        if self.use_lcf and self.iq_learn:

            if self.learn_lcf:
                self.lcf_parameters = MLPLayer(128, 128, 1)  # [0.0, np.log(0.1)]

                self.automatic_optimization = False

        self.use_distance = False

        # self.automatic_optimization=False

        if self.encoder.use_vae:
            self.l_vae_kl = BalancedKL(kl_balance_scale=0.2, kl_free_nats=1.0)

        self.use_ce = False

        if self.use_ce:
            self.training_loss = CrossEntropy(**model_config.training_loss)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Unused parameter: {name}")

    def get_QV(self, tokenized_map, tokenized_agent, train_mask, key='expert'):
        valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]
        action = tokenized_agent["sampled_idx"][:, self.start_step + 1:]

        pred = self.encoder(tokenized_map, tokenized_agent)

        train_mask=train_mask.flatten(0, 1)

        action=action.transpose(0, 1).flatten(0, 1)[train_mask]

        next_token_logits=pred["agent_q"]

        if key == "expert":
            action_valid=train_mask[valid_mask[:,:-1].transpose(0, 1).flatten(0, 1)]
        else:
            action_valid=train_mask

        next_token_logits=next_token_logits[action_valid]

        pi = torch.softmax(next_token_logits / self.alpha, dim=-1)

        logpi = torch.log(pi + 1e-10)

        log_prob = torch.gather(logpi, dim=-1, index=action.unsqueeze(-1)).squeeze(-1) #t,a

        entropy = -torch.sum(pi * logpi, dim=-1)

        exit_logit = pred["exit_logit"]

        if exit_logit is not None:
            action_valid=valid_mask[:,1:].transpose(0, 1).flatten(0, 1)[train_mask]

            action_nll=-log_prob[action_valid].mean()

            self.log("train/" + key + "_nll", action_nll.item(), on_step=True, batch_size=1)

            exit_log_p = torch.log_softmax(exit_logit, dim=-1)

            exit_idx = (~action_valid).to(int)

            exit_log_prob = torch.gather(exit_log_p, dim=-1, index=exit_idx.unsqueeze(-1)).squeeze(-1)

            exit_nll = -exit_log_prob.mean()

            log_prob = log_prob + exit_log_prob

            action_nll = action_nll + 0.1 * exit_nll

        else:
            action_nll=-log_prob.mean()

            self.log("train/" + key + "_nll", action_nll.item(), on_step=True, batch_size=1)

            if self.token_processor.pred_exit:
                exit_mask=action==self.token_processor.n_token_agent-1

                exit_nll = -log_prob[exit_mask].mean()

                self.log("train/" + key +"_exit_nll", exit_nll.item(), on_step=True, batch_size=1)

        self.log("train/" + key + "_entropy", entropy.mean().item(), on_step=True, batch_size=1)

        if self.token_processor.pred_entry:
            entry_idx=tokenized_agent["entry_idx"][:,self.start_step + 1:].transpose(0, 1).flatten(0, 1)

            pred_entry_logit,pred_entry_head_logit=pred["entry_logit"]

            entry_log_p=torch.log_softmax(pred_entry_logit, dim=-1)

            entry_nll = -torch.gather(entry_log_p, dim=-1, index=entry_idx[train_mask].unsqueeze(-1)).mean()

            head_mask=(entry_idx!=(pred_entry_logit.shape[-1]-1)) & train_mask

            entry_head_idx=tokenized_agent["entry_head_idx"][:,self.start_step + 1:].transpose(0, 1).flatten(0, 1)[head_mask]

            entry_head_log_p=torch.log_softmax(pred_entry_head_logit, dim=-1)

            entry_head_nll = -torch.gather(entry_head_log_p, dim=-1, index=entry_head_idx.unsqueeze(-1)).mean()

            self.log("train/entry_nll", entry_nll.item(), on_step=True, batch_size=1)
            self.log("train/entry_head_nll", entry_head_nll.item(), on_step=True, batch_size=1)

            action_nll=0.1*entry_nll+0.1*entry_head_nll+action_nll

        return action_nll,log_prob

    def get_train_mask(self,tokenized_agent):
        valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]

        if self.token_processor.pred_exit:
            train_mask = valid_mask[:, :-1]
        else:
            train_mask = valid_mask[:, 1:] & valid_mask[:, :-1]

        if "pred_mask" in tokenized_agent.keys():
            pred_mask =tokenized_agent["pred_mask"]

            train_mask[pred_mask]=(valid_mask[:, 1:] & valid_mask[:, :-1])[pred_mask]

        return train_mask.transpose(0, 1) #t,a


    def get_reward(self, tokenized_agent, key):

        disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                            tokenized_agent["token_mask"],
                                                            tokenized_agent["valid_mask"],
                                                            tokenized_agent["sampled_pos"] ,
                                                            tokenized_agent["sampled_heading"],
                                                            tokenized_agent,
                                                            tokenized_agent["detach_map_feature"],
                                                            abs_time=tokenized_agent["abs_time"]  )


        if key == "agent" and self.use_kl_penalty:
            with torch.no_grad():

                logp_ref = (torch.softmax(target_q / self.alpha, dim=-1) + 1e-10).log()

                actions = tokenized_agent["sampled_idx"][:, 2:][tokenized_agent["train_mask"]]

                logp_a_ref = torch.gather(logp_ref, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

                kl_penalty = torch.sum(agent_pi * ((agent_pi + 1e-10).log() - logp_ref), dim=-1).mean()  # (B,T)

                self.log("train/kl_penalty", kl_penalty.item(), on_step=True, batch_size=1)

                kl_coef = 4  # np.power(0.9999,self.global_step)
                kl_taken = logp_a_ref - agent_log_prob

                # kl_taken=-logr.exp()+1+logr
                # kl_per_token = -kl_coef * ((logr).exp() - 1 - logr)
                kl_per_token = kl_coef * kl_taken

        else:
            kl_per_token = 0

        ego_logits, interact_logits = disc_out[0]

        ego_rewards, nei_rewards,valid_ego_reward,valid_interact_reward = disc_out[2]

        weight = disc_out[3]

        rewards = ego_rewards + nei_rewards + kl_per_token

        self.log("train/" + key + "_rewards", ego_rewards.mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_nei_rewards", nei_rewards.mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_all_rewards", rewards.mean().item(), on_step=True, batch_size=1)

        mask_s = tokenized_agent["valid_mask"].transpose(0,1)

        after_any= torch.cumsum(mask_s, dim=0)

        #exit_mask =~mask_s[:, 1 + self.start_step:] & mask_s[:,  self.start_step:-1]

        exit_mask = ~mask_s & (after_any >0)
        present_mask = exit_mask | mask_s
        present_flatten=present_mask.flatten(0,1)

        self.log("train/" + key + "_exit_rewards", ego_rewards[exit_mask.flatten(0,1)].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_ego_reward", valid_ego_reward[present_flatten].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_interact_reward", valid_interact_reward.mean().item(), on_step=True, batch_size=1)

        if key=="agent":
            #self.ego_return_meanstd.update(ego_rewards.reshape(-1))
            #ego_rewards=self.ego_return_meanstd.normalize(ego_rewards)#
            #ego_rewards = (ego_rewards-ego_rewards.mean())/ego_rewards.std()#

            #ego_rewards=ego_rewards-ego_rewards.mean()
            #ego_rewards=ego_rewards/ego_rewards.std()
            ego_rewards=ego_rewards.reshape(mask_s.shape[0],mask_s.shape[1])[self.start_step+1:] #t,a
            #ego_rewards[~present_mask]=0

            # mask_s = tokenized_agent["valid_mask"][:, 1 + self.start_step:].transpose(0, 1)
            # batch_rewards = torch.zeros_like(mask_s, dtype=rewards.dtype)
            # batch_rewards = batch_rewards.masked_scatter(mask_s, ego_rewards)
            # ego_rewards = batch_rewards.transpose(0, 1)

            if self.use_lcf:
               # self.global_return_meanstd.update(nei_rewards.reshape(-1))
               # nei_rewards = self.global_return_meanstd.normalize(nei_rewards)
                #nei_rewards = (nei_rewards - nei_rewards.mean()) / nei_rewards.std()  #
                #nei_rewards=nei_rewards-nei_rewards.mean()
                #nei_rewards=nei_rewards/nei_rewards.std()
               nei_rewards = nei_rewards.reshape(mask_s.shape[0], mask_s.shape[1])[self.start_step+1:]#t,a

               # batch_nei_rewards = torch.zeros_like(mask_s, dtype=rewards.dtype)#+nei_rewards.mean()
                # batch_nei_rewards = batch_nei_rewards.masked_scatter(mask_s, nei_rewards)
                # nei_rewards = batch_nei_rewards.transpose(0, 1)

        if self.dis_loss == "pugail":
            positive_class_prior = 0.7
            pugail_beta = None

            if key == "expert":
                # positive loss: prior * -ln(D(expert)) = prior * -logsigmoid(logits)
                bce_loss = positive_class_prior * -disc_val.log()
            else:
                bce_loss = -(1 - disc_val).log() - positive_class_prior * -(1 - expert_disc_val).log()

                # negative loss: -ln(1 - D(policy)) - prior * -ln(1 - D(expert))
                if pugail_beta is not None:
                    bce_loss = torch.clamp(bce_loss, min=-1.0 * pugail_beta)

            bce_loss = bce_loss.mean()
        elif self.dis_loss == 'rpgan':
            bce_loss = logit[:, :, 0]
        elif self.dis_loss == "wgan":
            if key == "expert":
                bce_loss = -logit[:, :, 0].mean()  # self.bce_loss(disc_val, torch.ones_like(disc_val)) #-disc_val.log()
            else:
                bce_loss = logit[:, :,
                           0].mean()  # self.bce_loss(disc_val, torch.zeros_like(disc_val)) # -(1 - disc_val).log()
        else:
            if key == "expert":
                target=1
            else:
                target=0

            ego_logits=ego_logits[present_flatten]

            bce_loss = F.binary_cross_entropy_with_logits(ego_logits, torch.zeros_like(ego_logits)+target, weight=None,
                                              reduction='mean')
            if len(interact_logits) > 0:

                bce_loss = bce_loss + F.binary_cross_entropy_with_logits(interact_logits, torch.zeros_like(interact_logits) + target,
                                                             weight=weight, reduction='mean') #/ego_num

                logit=torch.cat([ego_logits, interact_logits], dim=0)

                disc_val = torch.sigmoid(logit)

                self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
               # self.log("train/"+key+"_disc_val_std", disc_val.std().item(), on_step=True, batch_size=1)

        return bce_loss, ego_rewards, nei_rewards,present_mask[self.start_step:-1]

    def iq_update(self, tokenized_map, tokenized_agent):

        expert_train_mask= self.get_train_mask(tokenized_agent)

        if self.use_kl_penalty:
            expert_nll = 0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            if self.iq_learn and self.encoder.use_roformer:
                self.encoder.agent_encoder.a_t_roformer.attn.caching = True
                if self.encoder.agent_encoder.pred_light and not self.encoder.agent_encoder.light_encoder.share:
                    self.encoder.agent_encoder.light_encoder.lg_t_roformer.attn.caching = True

            expert_nll, expert_log_prob= self.get_QV(tokenized_map, tokenized_agent, expert_train_mask)

        if not self.iq_learn:
            return expert_nll

        expert_dis_loss = self.get_reward(tokenized_agent, "expert")[0]

        tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent,  self.validation_rollout_sampling)

        agent_train_mask= self.get_train_mask(tokenized_agent_rollout)

        if self.use_kl_penalty:
            with torch.no_grad():
                if self.bc_map_net is not None:
                    map_feature = self.bc_map_net(tokenized_map)
                else:
                    map_feature = tokenized_agent["map_feature"]

                target_q = self.bc_net(tokenized_agent_rollout, map_feature)["agent_q"]
        else:
            target_q = None

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = True

        agent_nll, agent_log_prob = self.get_QV(tokenized_map, tokenized_agent_rollout, agent_train_mask, key='agent')#current valid

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = False

        agent_dis_loss, agent_rewards, nei_rewards,agent_present_mask = self.get_reward(tokenized_agent_rollout, "agent")

        critic_loss = expert_dis_loss + agent_dis_loss  # + expert_gp + agent_gp

        feat_a = tokenized_agent_rollout["feat_a"]

        value = self.encoder.value_network(feat_a)[..., 0]

        advantages, value_loss=compute_advantages(agent_rewards, value, agent_present_mask)

        if self.use_lcf:
            if not self.encoder.agent_encoder.interative_decoder.use_edge_feature:
                nei_rewards = get_nei_returns(tokenized_agent, agent_rewards, train_mask=all_valid)

            nei_value = self.encoder.nei_value_network(feat_a)[..., 0]

            nei_advantages, nei_value_loss = compute_advantages(nei_rewards, nei_value, agent_present_mask)

            value_loss = nei_value_loss + value_loss

            advantages = 1 / 3 * advantages + 2 / 3 * nei_advantages

        self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

        advantages = advantages[agent_train_mask]#t,a

        self.return_meanstd.update(advantages)
        advantages = self.return_meanstd.normalize(advantages)
        self.log("train/running_mean", self.return_meanstd.mean.mean(), on_step=True, batch_size=1)
        self.log("train/running_var", self.return_meanstd.var.mean(), on_step=True, batch_size=1)

        agent_wNLL = -(agent_log_prob * advantages).mean()

        self.log("train/agent_wNLL", agent_wNLL.item(), on_step=True, batch_size=1)
        self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)

        expert_nll = expert_nll + agent_wNLL + 1e-3 * value_loss  # - 0.01 * agent_entropy.mean()

        self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

        loss = critic_loss + expert_nll

        if self.automatic_optimization == False:
            policy_optimizer, discriminator_optimizer = self.optimizers()
            discriminator_optimizer.zero_grad()
            self.manual_backward(critic_loss)
            discriminator_optimizer.step()

        if self.automatic_optimization == False:
            policy_optimizer.zero_grad()
            self.manual_backward(expert_nll)
            policy_optimizer.step()

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        if "max_dist"  in   tokenized_agent.keys():
            max_dist=tokenized_agent["max_dist"]
            reset_mask=tokenized_agent["reset_mask"]
            token_mask=tokenized_agent["token_mask"]

            self.log("train/mean_token_error", max_dist.mean().item(), on_step=True, batch_size=1)
            self.log("train/reset_mask", reset_mask[token_mask].float().mean().item(), on_step=True, batch_size=1)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q:
            soft_update(self.encoder.critic, self.target_net, tau=2e-4)

        return loss
