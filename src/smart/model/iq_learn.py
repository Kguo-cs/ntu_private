from jax.example_libraries.stax import logsoftmax
from lightning import LightningModule
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from collections import deque
import random
import copy
from src.smart.loss.rollout_buffer import RunningMeanStdTorch, get_reward, get_nei_returns, get_return, \
    get_near_returns, per_scene_zscore_clip,rollout, compute_advantages,get_train_mask
from src.smart.loss.gp_penalty import compute_gp

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)
        self.gamma = 0.99
        self.gail = self.encoder.gail
        self.alpha = self.encoder.alpha

        self.use_target_q = False

        self.start_step = self.encoder.agent_encoder.start_step

        self.bce_loss = nn.BCELoss()

        self.buffer_len = 1

        self.replay_buffer = deque(maxlen=self.buffer_len)

        self.rollout_freq = 1

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
            self.ego_return_meanstd = RunningMeanStdTorch(shape=(1))
            self.global_return_meanstd = RunningMeanStdTorch(shape=(1))

        self.use_lcf = self.encoder.use_lcf
        self.use_gradient_penalty = False

        # self.automatic_optimization=False
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Unused parameter: {name}")

    def get_QV(self, tokenized_map, tokenized_agent, train_mask, key='expert'):
        valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]
        action = tokenized_agent["sampled_idx"][:, self.start_step + 1:]

        pred = self.encoder(tokenized_map, tokenized_agent)

        if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
            agent_train_mask=tokenized_agent["train_mask"]
            valid_mask=valid_mask[agent_train_mask]
            action=action[agent_train_mask]

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

    def get_reward(self, tokenized_agent, key,dis_mask=None):

        disc_out = self.encoder.discriminator.predict_agent(None,
                                                            tokenized_agent["token_mask"],
                                                            tokenized_agent["valid_mask"],
                                                            tokenized_agent["sampled_pos"] ,
                                                            tokenized_agent["sampled_heading"],
                                                            tokenized_agent,
                                                            tokenized_agent["detach_map_feature"],
                                                            abs_time=tokenized_agent["abs_time"]  )

        ego_logits, interact_logits = disc_out[0]

        ego_rewards, nei_rewards,valid_ego_reward,valid_interact_reward = disc_out[2]

        if len(nei_rewards)>0:
            all_rewards = ego_rewards + nei_rewards
            self.log("train/" + key + "_all_rewards", all_rewards.mean().item(), on_step=True, batch_size=1)

        self.log("train/" + key + "_rewards", ego_rewards.mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_nei_rewards", nei_rewards.mean().item(), on_step=True, batch_size=1)

        mask_s = tokenized_agent["valid_mask"].transpose(0,1)#[2:]

        if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
            mask_s=mask_s[:,tokenized_agent["train_mask"]]

        after_any= torch.cumsum(mask_s, dim=0)

        #exit_mask =~mask_s[:, 1 + self.start_step:] & mask_s[:,  self.start_step:-1]

        exit_mask = ~mask_s & (after_any >0)
        present_mask = exit_mask | mask_s
        present_flatten=present_mask.flatten(0,1)

        self.log("train/" + key + "_exit_rewards", ego_rewards[exit_mask.flatten(0,1)].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_ego_reward", valid_ego_reward[present_flatten].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_interact_reward", valid_interact_reward.mean().item(), on_step=True, batch_size=1)

        if key == "expert":
            target=1
        else:
            target=0
            ego_rewards=ego_rewards.reshape(mask_s.shape[0],mask_s.shape[1])[self.start_step+1:] #t,a
            if len(nei_rewards):
               nei_rewards = nei_rewards.reshape(mask_s.shape[0], mask_s.shape[1])[self.start_step+1:]#t,a

        if dis_mask is not None:
            ego_logits=ego_logits[dis_mask]
        else:
            ego_logits=ego_logits[mask_s.flatten(0,1)]

            #ego_logits=ego_logits[present_flatten]

        bce_loss = F.binary_cross_entropy_with_logits(ego_logits, torch.zeros_like(ego_logits)+target, weight=None,
                                          reduction='mean')
        if len(interact_logits) > 0:
            weight = disc_out[3]

            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(interact_logits, torch.zeros_like(interact_logits) + target,
                                                         weight=weight, reduction='sum') /mask_s.sum()

            ego_logits=torch.cat([ego_logits, interact_logits], dim=0)
            self.log("train/"+key+"_interact_logits", interact_logits.mean().item(), on_step=True, batch_size=1)

        disc_val = torch.sigmoid(ego_logits)

        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val_std", disc_val.std().item(), on_step=True, batch_size=1)

        if self.use_gradient_penalty:
            gp=compute_gp(key, tokenized_agent, dis_mask, self.encoder.discriminator)
            self.log("train/" + key + "_gp", gp, on_step=True, batch_size=1)
        else:
            gp=0

        return bce_loss, ego_rewards, nei_rewards,present_mask[self.start_step:-1],mask_s.flatten(0,1),gp #

    def iq_update(self, tokenized_map, tokenized_agent):

        expert_train_mask= get_train_mask(tokenized_agent,self.start_step,self.token_processor.pred_exit)#t,a

        if self.use_kl_penalty:
            expert_nll = 0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            expert_nll, expert_log_prob= self.get_QV(tokenized_map, tokenized_agent, expert_train_mask)

        if not self.gail:
            return expert_nll

       # tokenized_agent["train_mask"]=tokenized_agent["pred_mask"] #& expert_train_mask.all(0)

        expert_dis_loss,_,_,_,expert_dis_mask,expert_gp = self.get_reward(tokenized_agent, "expert")

        tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent,  self.validation_rollout_sampling)

        agent_train_mask= get_train_mask(tokenized_agent_rollout,self.start_step,self.token_processor.pred_exit)

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = True

        agent_nll, agent_log_prob = self.get_QV(tokenized_map, tokenized_agent_rollout, agent_train_mask, key='agent')

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = False

        agent_dis_loss, agent_rewards, nei_rewards,agent_present_mask,_ ,agent_gp= self.get_reward(tokenized_agent_rollout, "agent",expert_dis_mask)

        critic_loss = expert_dis_loss + agent_dis_loss + agent_gp

        feat_a = tokenized_agent_rollout["feat_a"]

        value = self.encoder.value_network(feat_a)[..., 0]

        advantages, value_loss=compute_advantages(agent_rewards, value, agent_present_mask)

        if len(nei_rewards) and self.use_lcf:
            nei_value = self.encoder.nei_value_network(feat_a)[..., 0]

            nei_advantages, nei_value_loss = compute_advantages(nei_rewards, nei_value, agent_present_mask)

            value_loss =value_loss+nei_value_loss

            advantages = 1/2 * advantages + 1/2 * nei_advantages

        self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

        advantages = advantages[agent_train_mask]#t,a  # only train at expert valid

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
        return loss
