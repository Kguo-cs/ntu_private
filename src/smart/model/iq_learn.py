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
    get_near_returns, per_scene_zscore_clip,rollout, compute_advantages,get_train_mask,get_reduce_loss
from src.smart.loss.gp_penalty import compute_gp
import torch.distributed as dist

from src.smart.utils import wrap_angle


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
        self.token_cls_loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        if self.encoder.agent_encoder.learn_init and (self.encoder.agent_encoder.use_gan or self.encoder.agent_encoder.init_decoder.use_gan):
            self.automatic_optimization=False

    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        pred = self.encoder(tokenized_map, tokenized_agent)

        if pred["next_token_logits"] is not None:
            valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]
            action = tokenized_agent["sampled_idx"][:, self.start_step + 1:]

            train_mask = get_train_mask(tokenized_agent, self.start_step, self.token_processor.pred_exit)  # t,a

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

            self.log("train/" + key + "_nll", action_nll.item(), on_step=True, batch_size=1)
            self.log("train/" + key + "_entropy", entropy.mean().item(), on_step=True, batch_size=1)

            if self.token_processor.pred_exit:
                exit_mask=action==self.token_processor.n_token_agent-1

                exit_nll = -log_prob[exit_mask].mean()

                self.log("train/" + key +"_exit_nll", exit_nll.item(), on_step=True, batch_size=1)
        else:
            action_nll=log_prob=0

        if pred["initial_logit"] is not None:

            if not self.token_processor.token_initial:

                if  self.encoder.agent_encoder.use_gan or self.encoder.agent_encoder.init_decoder.use_gan:
                    opt_G,opt_D=self.optimizers()

                    if len(pred["initial_logit"]) == 3:
                        real_loss, fake_loss,gp=pred["initial_logit"]
                        loss=real_loss+fake_loss+gp
                        self.log("train/real_loss", real_loss.item(), on_step=True, batch_size=1)
                        self.log("train/fake_loss", fake_loss.item(), on_step=True, batch_size=1)
                        self.log("train/d_loss", loss.item(), on_step=True, batch_size=1)
                        self.log("train/gp", gp.item(), on_step=True, batch_size=1)

                        opt_D.zero_grad()
                        loss.backward()
                        opt_D.step()

                    else:
                        g_loss, match_loss,pos_loss,heading_loss,shape_loss,vel_loss = pred["initial_logit"]

                        loss=g_loss+match_loss+action_nll

                        self.log("train/g_loss", g_loss.item(), on_step=True, batch_size=1)
                        self.log("train/pos_loss", pos_loss.item(), on_step=True, batch_size=1)
                        self.log("train/heading_loss", heading_loss.item(), on_step=True, batch_size=1)
                        self.log("train/shape_loss", shape_loss.item(), on_step=True, batch_size=1)
                        self.log("train/match_loss", match_loss.item(), on_step=True, batch_size=1)
                        self.log("train/vel_loss", vel_loss.item(), on_step=True, batch_size=1)

                        opt_G.zero_grad()
                        loss.backward()
                        opt_G.step()
                else:
                    if self.encoder.agent_encoder.init_decoder.learn_autoencoder:
                        loss,agent_loss,kl_loss=pred["initial_logit"]
                        self.log('train/loss_diff_init', loss, on_step=True,  batch_size=1  )
                        self.log('train/agent_loss', agent_loss,  on_step=True, batch_size=1)
                        self.log('train/kl_loss', kl_loss,  on_step=True,  batch_size=1)
                    else:
                        loss,loss_diff_init,match_loss,pos_loss,heading_loss,shape_loss,vel_loss=pred["initial_logit"]
                        self.log('train/loss_diff_init', loss_diff_init,  on_step=True, batch_size=1)
                        self.log('train/match_loss', match_loss,  on_step=True,  batch_size=1)
                        self.log('train/pos_loss', pos_loss,  on_step=True,  batch_size=1)
                        self.log('train/heading_loss', heading_loss,  on_step=True,  batch_size=1)
                        self.log('train/shape_loss', shape_loss,  on_step=True,  batch_size=1)
                        self.log('train/vel_loss', vel_loss,  on_step=True,  batch_size=1)

                action_nll = action_nll +loss
            else:
                pos_logit, entry_head_logit, entry_offset, pred_shape=pred["initial_logit"]

                non_ego_mask=~tokenized_agent["initial_ego_mask"]
                initial_pos_token = tokenized_agent["initial_pos_token"][non_ego_mask]
                initial_offset_token = tokenized_agent["initial_offset_token"][non_ego_mask]
                initial_heading_token = tokenized_agent["initial_heading_token"][non_ego_mask]
                initial_shape = tokenized_agent["initial_shape_token"][non_ego_mask]

                pos_nll=self.token_cls_loss(pos_logit, initial_pos_token)
                head_nll=self.token_cls_loss(entry_head_logit, initial_heading_token)
                offset_nll=self.token_cls_loss(entry_offset, initial_offset_token)
                #offset_mse=self.token_cls_loss(entry_offset, initial_offset_token)
                shape_nll=self.token_cls_loss(pred_shape, initial_shape)

                self.log("train/pos_nll", pos_nll.item(), on_step=True, batch_size=1)
                self.log("train/head_nll", head_nll.item(), on_step=True, batch_size=1)
                self.log("train/offset_nll", offset_nll.item(), on_step=True, batch_size=1)
                # self.log("train/offset_mse", offset_mse.item(), on_step=True, batch_size=1)
                self.log("train/shape_nll", shape_nll.item(), on_step=True, batch_size=1)

                action_nll=action_nll+pos_nll+head_nll+0.1*shape_nll#+0.1*offset_nll

        if pred["entry_logit"] is not None:

            if self.token_processor.autoregressive_entry:

                pred_entry_logit,pred_entry_head_logit,pred_offset,type_logit,pred_shape=pred["entry_logit"]

                pos_idx=tokenized_agent["pos_idx"]

                entry_head_idx=tokenized_agent["head_idx"]

                entry_pos_offset=tokenized_agent["offset"]

                entry_mask =pos_idx!=pred_entry_logit.shape[-1]-1

                pos_mask=torch.cat([torch.ones_like(entry_mask[:,:1]),entry_mask],dim=1)

                pos_idx=torch.cat([pos_idx, torch.zeros_like(pos_idx[:,:1])+pred_entry_logit.shape[-1]-1], dim=1)[pos_mask]

                entry_log_p=torch.log_softmax(pred_entry_logit[pos_mask], dim=-1)

                entry_nll = -torch.gather(entry_log_p, dim=-1, index=pos_idx.unsqueeze(-1)).mean()

                self.log("train/entry_nll", entry_nll.item(), on_step=True, batch_size=1)

                entry_head_log_p=torch.log_softmax(pred_entry_head_logit[entry_mask], dim=-1)

                entry_head_nll = -torch.gather(entry_head_log_p, dim=-1, index=entry_head_idx[entry_mask].unsqueeze(-1)).mean()

                self.log("train/entry_head_nll", entry_head_nll.item(), on_step=True, batch_size=1)

                entry_pos_offset=entry_pos_offset[entry_mask]
                pred_offset=pred_offset[entry_mask]

                action_nll=action_nll+0.01*entry_nll+0.01*entry_head_nll

                if self.token_processor.token_offset:
                    pred_offset=torch.log_softmax(pred_offset, dim=-1)

                    entry_offset_nll = -torch.gather(pred_offset, dim=-1, index=entry_pos_offset.unsqueeze(-1)).mean()

                    self.log("train/entry_offset_nll", entry_offset_nll.item(), on_step=True, batch_size=1)

                    action_nll = action_nll+entry_offset_nll
                else:
                    offset_l1=(entry_pos_offset[...,:-1]-pred_offset[...,:-1]).abs().mean()

                    self.log("train/offset_l1", offset_l1.item(), on_step=True, batch_size=1)

                    offset_head = wrap_angle(entry_pos_offset[..., -1] - pred_offset[..., -1]).abs().mean()

                    self.log("train/offset_head", offset_head.item(), on_step=True, batch_size=1)

                    action_nll=action_nll+0.01*offset_l1+0.01*offset_head

                if not self.token_processor.use_bird:
                    entry_type = tokenized_agent["entry_type"]

                    entry_type_log_p = torch.log_softmax(type_logit[entry_mask], dim=-1)

                    entry_type_nll = -torch.gather(entry_type_log_p, dim=-1,
                                                   index=entry_type[entry_mask].unsqueeze(-1)).mean()

                    self.log("train/entry_type_nll", entry_type_nll.item(), on_step=True, batch_size=1)

                    entry_shape = tokenized_agent["entry_shape"]

                    shape_l1 = (entry_shape[entry_mask] - pred_shape[entry_mask]).abs().mean()

                    self.log("train/shape_l1", shape_l1.item(), on_step=True, batch_size=1)

                    action_nll=action_nll+0.01*entry_type_nll+0.01*shape_l1

            else:
                entry_idx=tokenized_agent["entry_idx"][:,self.start_step + 1:].transpose(0, 1).flatten(0, 1)

                pred_entry_logit,pred_entry_head_logit,pred_offset,type_logit,pred_shape=pred["entry_logit"]

                entry_log_p=torch.log_softmax(pred_entry_logit, dim=-1)

                entry_nll = -torch.gather(entry_log_p, dim=-1, index=entry_idx[train_mask].unsqueeze(-1)).mean()

                self.log("train/entry_type_nll", entry_nll.item(), on_step=True, batch_size=1)

                entry_head_idx=tokenized_agent["entry_head_idx"]#[:,self.start_step + 1:].transpose(0, 1).flatten(0, 1)[head_mask]#t,a

                entry_head_log_p=torch.log_softmax(pred_entry_head_logit, dim=-1)

                entry_head_nll = -torch.gather(entry_head_log_p, dim=-1, index=entry_head_idx.unsqueeze(-1)).squeeze(-1).mean()

                self.log("train/entry_head_nll", entry_head_nll.item(), on_step=True, batch_size=1)

                action_nll = action_nll +0.01* entry_nll + 0.01*entry_head_nll

                entry_pos_offset=tokenized_agent["entry_pos_offset"]

                offset_l1=(entry_pos_offset[...,:-1]-pred_offset[...,:-1]).abs().mean()

                self.log("train/offset_l1", offset_l1.item(), on_step=True, batch_size=1)

                offset_head = wrap_angle(entry_pos_offset[..., -1] - pred_offset[..., -1]).abs().mean()

                self.log("train/offset_head", offset_head.item(), on_step=True, batch_size=1)

                action_nll=action_nll+0.01*offset_l1+0.01*offset_head

                if not self.token_processor.use_bird:
                    entry_type = tokenized_agent["entry_pos"]
                    type_log_p = torch.log_softmax(type_logit, dim=-1)
                    entry_type_nll = -torch.gather(type_log_p, dim=-1,
                                                   index=entry_type.unsqueeze(-1)).mean()

                    self.log("train/entry_nll", entry_type_nll.item(), on_step=True, batch_size=1)

                    entry_shape = tokenized_agent["entry_shape"]

                    shape_l1 = (entry_shape - pred_shape).abs().mean()

                    self.log("train/shape_l1", shape_l1.item(), on_step=True, batch_size=1)

                    action_nll=action_nll+0.01*entry_type_nll+0.01*shape_l1

        return action_nll,log_prob

    def get_reward(self, tokenized_agent, key,dis_mask=None):

        disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                            tokenized_agent["token_mask"],
                                                            tokenized_agent["valid_mask"],
                                                            tokenized_agent["sampled_pos"] ,
                                                            tokenized_agent["sampled_heading"],
                                                            tokenized_agent,
                                                            tokenized_agent["detach_map_feature"])


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

        if self.encoder.agent_encoder.agent_token_embedding.use_state_action:
            exit_mask=exit_mask[1:]
            present_flatten=present_mask[1:].flatten(0,1)
            start_step=self.start_step
        else:
            present_flatten=present_mask.flatten(0,1)
            start_step=self.start_step+1

        self.log("train/" + key + "_exit_rewards", ego_rewards[exit_mask.flatten(0,1)].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_ego_reward", valid_ego_reward[present_flatten].mean().item(), on_step=True, batch_size=1)
        self.log("train/" + key + "_valid_interact_reward", valid_interact_reward.mean().item(), on_step=True, batch_size=1)

        if key == "expert":
            target=1
        else:
            target=0
            ego_rewards=ego_rewards.reshape(exit_mask.shape)[start_step:] #t,a
            if len(nei_rewards):
               nei_rewards = nei_rewards.reshape(exit_mask.shape)[start_step:]#t,a

        if dis_mask is None:
            if self.token_processor.use_bird:
                dis_mask=present_flatten
            else:
                dis_mask=mask_s.flatten(0, 1)

        if len(interact_logits)==len(ego_logits):
            interact_logits=interact_logits[dis_mask]

        ego_logits=ego_logits[dis_mask]

        bce_loss = F.binary_cross_entropy_with_logits(ego_logits, torch.zeros_like(ego_logits)+target, reduction='mean')
        if len(interact_logits) > 0:
            weight = disc_out[3]

            interact_bce_loss=F.binary_cross_entropy_with_logits(interact_logits, torch.zeros_like(interact_logits) + target,
                                                         weight=weight, reduction='sum')/dis_mask.sum()

            ego_logits=torch.cat([ego_logits, interact_logits], dim=0)
            self.log("train/"+key+"_interact_logits", interact_logits.mean().item(), on_step=True, batch_size=1)
        else:
            interact_bce_loss=None

        disc_val = torch.sigmoid(ego_logits)

        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val_std", disc_val.std().item(), on_step=True, batch_size=1)

        if self.use_gradient_penalty:
            gp=compute_gp(key, tokenized_agent, dis_mask, self.encoder.discriminator)
            self.log("train/" + key + "_gp", gp, on_step=True, batch_size=1)
        else:
            gp=0

        return bce_loss,interact_bce_loss, ego_rewards, nei_rewards,present_mask[self.start_step:-1],gp,dis_mask #,mask_s.flatten(0,1)

    def iq_update(self, tokenized_map, tokenized_agent):
        if self.use_kl_penalty:
            expert_nll= 0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            expert_nll, expert_log_prob= self.get_QV(tokenized_map, tokenized_agent)

        if not self.gail:
            return expert_nll

        tokenized_agent["train_mask"]=tokenized_agent["pred_mask"] & tokenized_agent["token_mask"][:,self.start_step:].all(1)

        expert_dis_loss,expert_dis_loss1,_,_,expert_present_mask,expert_gp,expert_dis_mask = self.get_reward(tokenized_agent, "expert")

        tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent,  self.validation_rollout_sampling)

        agent_train_mask= get_train_mask(tokenized_agent_rollout,self.start_step,self.token_processor.pred_exit)

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = True

        agent_nll, agent_log_prob = self.get_QV(tokenized_map, tokenized_agent_rollout, key='agent')

        self.encoder.agent_encoder.interative_decoder.edge_encoder.rollout_traj = False

        agent_dis_loss,agent_dis_loss1, agent_rewards, nei_rewards,agent_present_mask,agent_gp,_= self.get_reward(tokenized_agent_rollout, "agent",expert_dis_mask)

        feat_a = tokenized_agent_rollout["feat_a"]

        value = self.encoder.value_network(feat_a)[..., 0]

        advantages, value_loss=compute_advantages(agent_rewards, value, agent_present_mask)

        if len(nei_rewards) and self.use_lcf:
            nei_value = self.encoder.nei_value_network(feat_a)[..., 0]

            nei_advantages, nei_value_loss = compute_advantages(nei_rewards, nei_value, agent_present_mask)

            value_loss =value_loss+nei_value_loss

            advantages = 1/2 * advantages + 1/2 * nei_advantages

        advantages = advantages[agent_train_mask]#t,a  # only train at expert valid

        self.return_meanstd.update(advantages)

        advantages = self.return_meanstd.normalize(advantages)

        ppo_loss = -(agent_log_prob * advantages).mean()

        critic_loss = expert_dis_loss + agent_dis_loss + agent_gp

        self.log("train/running_mean", self.return_meanstd.mean, on_step=True, batch_size=1)
        self.log("train/running_var", self.return_meanstd.var, on_step=True, batch_size=1)
        self.log("train/ppo_loss", ppo_loss.item(), on_step=True, batch_size=1)
        self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)
        self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)
        self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

        policy_loss = expert_nll + ppo_loss + 1e-3 * value_loss  # - 0.01 * agent_entropy.mean()

        loss = critic_loss + policy_loss

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        if "max_dist"  in   tokenized_agent.keys():
            max_dist=tokenized_agent["max_dist"]
            reset_mask=tokenized_agent["reset_mask"]
            token_mask=tokenized_agent["token_mask"]

            self.log("train/mean_token_error", max_dist.mean().item(), on_step=True, batch_size=1)
            self.log("train/reset_mask", reset_mask[token_mask].float().mean().item(), on_step=True, batch_size=1)

        if "entry_token_invalid_mask"  in   tokenized_agent.keys():
            entry_token_invalid_mask=tokenized_agent["entry_token_invalid_mask"]

            self.log("train/entry_token_invalid", entry_token_invalid_mask.float().mean().item(), on_step=True, batch_size=1)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        if self.token_processor.use_infgen:
            for key in tokenized_agent["map_feature"].keys():
                data["pt_token"][key]=tokenized_agent["map_feature"][key]

            loss1=self.encoder.agent_encoder.inf_decoder(data,tokenized_agent["map_feature"])

        self.log("train/loss", loss, on_step=True, batch_size=1)
        return loss
