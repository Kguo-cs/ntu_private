from lightning import LightningModule

import torch.optim as optim
import random
from collections import deque
import torch.nn as nn
from sklearn.covariance import log_likelihood
from tensorflow_probability.substrates.jax.distributions.student_t import entropy
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
from torch_geometric.profile.benchmark import require_grad

from src.smart.model.rollout_buffer import ReplayBuffer
from torch.autograd import Variable, grad
import numpy as np
# import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy


class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.replay_buffer = deque(maxlen=10)
        self.critic_tau = 0.005
        self.critic_target_update_frequency = 1
        self.gamma = 0.99
        self.alpha = self.encoder.agent_encoder.alpha

        self.reg_mult = 0.5

        self.Q_max = 1.0 / (self.reg_mult * (1 - self.gamma))
        self.Q_min = - 1.0 / (self.reg_mult * (1 - self.gamma))

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def rollout(self, tokenized_map, tokenized_agent,train_mask):
        pred = self.encoder.inference(
            tokenized_map,
            tokenized_agent,
            sampling_scheme=self.training_rollout_sampling,
        )

        tokenized_agent_rollout = {}

        tokenized_agent_rollout['sampled_pos'] = pred["pred_pos"]
        tokenized_agent_rollout['sampled_heading'] = pred['pred_head']
        tokenized_agent_rollout['sampled_idx'] = pred["pred_idx"]
        tokenized_agent_rollout['valid_mask'] = tokenized_agent['valid_mask']
        tokenized_agent_rollout['trajectory_token_veh'] = tokenized_agent['trajectory_token_veh']
        tokenized_agent_rollout['trajectory_token_ped'] = tokenized_agent['trajectory_token_ped']
        tokenized_agent_rollout['trajectory_token_cyc'] = tokenized_agent['trajectory_token_cyc']
        tokenized_agent_rollout['type'] = tokenized_agent['type']
        tokenized_agent_rollout['shape'] = tokenized_agent['shape']
        tokenized_agent_rollout['batch'] = tokenized_agent['batch']
        tokenized_agent_rollout['num_graphs'] = tokenized_agent['num_graphs']
        # tokenized_agent_rollout['next_route'] = tokenized_agent['next_route']
        # tokenized_agent_rollout['light'] = tokenized_agent['light']
        # pred_dict = self.encoder(tokenized_map, tokenized_agent_rollout)

        # pred_pos=pred["pred_pos"].cpu().numpy()
        #
        # np.save('p.npy',pred_pos)

        # for i in range(1):
        #     plt.plot(pred_pos[i][:,0],pred_pos[i][:,1])
        #
        # plt.show()
        #
        # for i in range(tokenized_agent["num_graphs"]):
        #     agent_mask= tokenized_agent['batch']==i
        #     tokenized_agent_rollout = {}
        #
        #     tokenized_agent_rollout['sampled_pos'] = pred["pred_pos"][agent_mask]
        #     tokenized_agent_rollout['sampled_heading'] = pred['pred_head'][agent_mask]
        #     tokenized_agent_rollout['sampled_idx'] = pred["pred_idx"][agent_mask]
        #     tokenized_agent_rollout['valid_mask'] = pred[ "pred_valid"][agent_mask]  # tokenized_agent['valid_mask']
        #     tokenized_agent_rollout['type'] = tokenized_agent['type'][agent_mask]
        #     tokenized_agent_rollout['shape'] = tokenized_agent['shape'][agent_mask]
        #
        #     map_mask=tokenized_map["batch"]==i
        #
        #     tokenized_map_rollout={}
        #
        #     tokenized_map_rollout["position"]=tokenized_map["position"][map_mask]
        #     tokenized_map_rollout["orientation"]=tokenized_map["orientation"][map_mask]
        #     tokenized_map_rollout["token_idx"]=tokenized_map["token_idx"][map_mask]
        #     tokenized_map_rollout["type"]=tokenized_map["type"][map_mask]
        #     tokenized_map_rollout["pl_type"]=tokenized_map["pl_type"][map_mask]
        #     tokenized_map_rollout["light_type"]=tokenized_map["light_type"][map_mask]
        #
        self.replay_buffer.append((tokenized_map, tokenized_agent_rollout))


    def get_QV(self, tokenized_map, tokenized_agent,div='rkl', key='expert'):

        pred_dict = self.encoder(tokenized_map, tokenized_agent)

        q_value =pred_dict["q_value"]

        q = q_value[:, :-1]

        action = tokenized_agent["sampled_idx"][:, 2:].reshape(-1)

        current_Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        v=  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

        current_V = v[:, :-1]

        target_v=v[:, 1:].detach()

        pi = torch.softmax(q / self.alpha, dim=-1)  # Compute policy
        logpi= torch.log(pi + 1e-10)
        entropy = -torch.sum(pi * logpi, dim=-1)

        # pred_logprob=self.logsoftmax(q/self.alpha)
        # with torch.no_grad():
            # next_q = q_value[:, 1:]#self.target_net(tokenized_map, tokenized_agent,kl_loss=False)["q_value"][:, 1:]
            # target_v = self.alpha * torch.logsumexp(next_q / self.alpha, dim=-1, keepdim=False)
            # pi = torch.softmax(next_q / self.alpha, dim=-1)  # Compute policy
            # target_v= torch.sum(pi * next_q, dim=-1)


        action_logprob = logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        done = torch.zeros_like(target_v)

        done[:, -1] = 1

        rewards = current_Q - (1 - done) * self.gamma * target_v

        valid_mask = tokenized_agent["valid_mask"]

        state_mask=valid_mask[:, 1:-1]

        state_action_mask = valid_mask[:, 2:] & state_mask

        reward=rewards[state_action_mask]

        if div=="kl":
            alpha=1
            reward_loss= alpha*(torch.log(alpha/reward)-1)
        elif div == "rkl":
            alpha=10
            reward_loss= (-reward/alpha-1).exp() * alpha
        else:
            alpha = 0.025

            reward_loss= -reward+reward.square()/(4*alpha)

        self.log("train/"+key+"_V", current_V[state_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q[state_action_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy[state_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward_loss", reward_loss.mean().item(), on_step=True, batch_size=1)

        return  reward,reward_loss, state_action_mask,action_logprob

    def collate_agent(self,batch_list):

        map_batch,agent_batch=batch_list[0]

        # sampled_pos=[]
        # sampled_heading=[]
        # sampled_idx=[]
        # valid_mask=[]
        # agent_type=[]
        # shape=[]
        # agent_batchid=[]
        #
        # position=[]
        # orientation=[]
        # token_idx=[]
        # map_type=[]
        # pl_type=[]
        # light_type=[]
        # map_batchid=[]
        #
        # for i,(tokenized_map_rollout,tokenized_agent_rollout) in enumerate(batch_list):
        #
        #     sampled_pos.append(tokenized_agent_rollout["sampled_pos"])
        #     sampled_heading.append(tokenized_agent_rollout["sampled_heading"])
        #     sampled_idx.append(tokenized_agent_rollout["sampled_idx"])
        #     valid_mask.append(tokenized_agent_rollout["valid_mask"])
        #     agent_type.append(tokenized_agent_rollout["type"])
        #     shape.append(tokenized_agent_rollout["shape"])
        #     agent_batchid.append(torch.zeros_like(tokenized_agent_rollout["type"])+i)
        #
        #     position.append(tokenized_map_rollout["position"])
        #     orientation.append(tokenized_map_rollout["orientation"])
        #     token_idx.append(tokenized_map_rollout["token_idx"])
        #     map_type.append(tokenized_map_rollout["type"])
        #     pl_type.append(tokenized_map_rollout["pl_type"])
        #     light_type.append(tokenized_map_rollout["light_type"])
        #     map_batchid.append(torch.zeros_like(tokenized_map_rollout["type"])+i)
        #
        #
        # agent_batch['sampled_pos'] =torch.cat(sampled_pos)
        # agent_batch['sampled_heading'] =torch.cat(sampled_heading)
        # agent_batch['sampled_idx'] = torch.cat(sampled_idx)
        # agent_batch['valid_mask'] = torch.cat(valid_mask)
        # agent_batch['type'] = torch.cat(agent_type)
        # agent_batch['shape'] = torch.cat(shape)
        # agent_batch["batch"]=torch.cat(agent_batchid)
        #
        agent_batch["trajectory_token_veh"]=self.token_processor.trajectory_token_veh
        agent_batch["trajectory_token_ped"]=self.token_processor.trajectory_token_ped
        agent_batch["trajectory_token_cyc"]=self.token_processor.trajectory_token_cyc
        # agent_batch['num_graphs'] = len(batch_list)
        #
        #
        # map_batch["position"]= torch.cat(position)
        # map_batch["orientation"]= torch.cat(orientation)
        # map_batch["token_idx"]= torch.cat(token_idx)
        # map_batch["type"]=torch.cat(map_type)
        # map_batch["pl_type"]=torch.cat(pl_type)
        # map_batch["light_type"]=torch.cat(light_type)
        # map_batch["batch"]=torch.cat(map_batchid)
        # map_batch["token_traj_src"]=self.token_processor.map_token_traj_src

        return map_batch,agent_batch

    def iq_update(self, tokenized_map, tokenized_agent,train_mask):

        expert_reward,expert_reward_loss, expert_valid,expert_logprob = self.get_QV(tokenized_map, tokenized_agent)

        agent_tokenized_map, agent_tokenized_agent = random.sample(self.replay_buffer,1)[0]

        agent_reward,agent_reward_loss ,agent_valid,_ = self.get_QV(agent_tokenized_map, agent_tokenized_agent, key='agent')

        expert_nll=-expert_logprob[expert_valid].mean()

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        agent_ratio=0
        reward_w=0.1

        reward_loss= (expert_reward_loss.sum()*(1-agent_ratio)+agent_reward_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

        self.log("train/reward_loss", reward_loss.item(), on_step=True, batch_size=1)

        agent_mean_reward = agent_reward.mean()

        critic_loss = expert_nll+reward_w*(reward_loss+agent_mean_reward)

        # expert_Q_loss = F.mse_loss(expert_Q[expert_valid], torch.ones_like(expert_Q[expert_valid]) * self.Q_max)
        # r_max = (1 - expert_done) * ((1 / self.reg_mult)) \
        #         + expert_done * (1 / (1 - self.gamma)) * ((1 / self.reg_mult))
        #
        # expert_Q_loss = torch.square(expert_Q - (r_max + expert_target_v))[expert_valid].mean()

        # self.log("train/expert_Q_loss", expert_Q_loss.item(), on_step=True, batch_size=1)


       # self.log("train/agent_kl_loss", agent_kl_loss.item(), on_step=True, batch_size=1)
        #self.log("train/expert_kl_loss", expert_kl_loss.item(), on_step=True, batch_size=1)
        # expert_value_loss = (expert_V - expert_target_v)[expert_valid].mean()
        #
        # agent_value_loss = (agent_V - agent_target_v)[agent_valid].mean()
        #
        # value_loss = (expert_value_loss * expert_valid.sum() + agent_value_loss * agent_valid.sum()) / (
        #             expert_valid.sum() + agent_valid.sum())
        #
        # self.log("train/expert_value_loss", expert_value_loss.item(), on_step=True, batch_size=1)
        # self.log("train/agent_value_loss", agent_value_loss.item(), on_step=True, batch_size=1)
        # self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

        #expert_r_min = -(1 / self.reg_mult)
        # expert_r_max  = (1 - expert_done) * ((1 / self.reg_mult)) \
        #         + expert_done * (1 / (1 - self.gamma)) * ((1 / self.reg_mult))
        #expert_r_max=1 / self.reg_mult

        #expert_value_mse_loss = torch.square(expert_V - (expert_r_min + expert_target_v))[expert_valid].mean()

        # policy_r_min = (1 - done) * (-(1 / self.reg_mult)) \
        #                + done * (1 / (1 - self.gamma)) * (-(1 / self.reg_mult))
        # policy_r_min = -(1 / self.reg_mult)
        #
        # agent_value_mse_loss = torch.square(agent_V - (policy_r_min + agent_target_v))[agent_valid].mean()
        #
        # value_mse_loss = (expert_value_mse_loss * expert_valid.sum() + agent_value_mse_loss * agent_valid.sum()) / (
        #             expert_valid.sum() + agent_valid.sum())


        return critic_loss

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def training_step(self, data, batch_idx):
        # print(self.global_step)
        # time1=time.time()
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)

            pred_prob=self.logsoftmax(pred["next_token_logits"])
            gt_id = tokenized_agent['gt_idx'][:,2:].reshape(-1)

            log_likelihood=pred_prob.reshape(len(gt_id),-1)[torch.arange(len(gt_id)),gt_id]

            valid_mask=tokenized_agent['valid_mask'][:,2:].reshape(-1)
            loss=-log_likelihood[valid_mask].mean()

            # loss = self.training_loss(
            #     **pred,
            #     token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            #     token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            #     train_mask=data["agent"]["train_mask"],  # [n_agent]
            #     gt_id=tokenized_agent['gt_idx'],  # [n_agent]
            #     current_epoch=self.current_epoch,
            # )

            loss=loss+pred["kl_loss"]
            pi = torch.softmax(pred['next_token_logits'] , dim=-1)  # Compute policy
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)

            expert_valid = tokenized_agent['valid_mask'][:, 1:-1]

            self.log("train/expert_kl_loss", pred["kl_loss"].item(), on_step=True, batch_size=1)
            self.log("train/expert_entropy", entropy[expert_valid].mean().item(), on_step=True, batch_size=1)

        else:
            if len(self.replay_buffer) < self.replay_buffer.maxlen or self.global_step % 10 == 0:
                with torch.no_grad():
                    self.encoder.eval()
                    self.rollout(tokenized_map, tokenized_agent,data["agent"]["train_mask"])
                    self.encoder.train()
            # print(time.time() - time1)

            loss = self.iq_update(tokenized_map, tokenized_agent,data["agent"]["train_mask"])
            # print(time.time() - time1)

            # if self.global_step % self.critic_target_update_frequency == 0:
            #     self.soft_update(self.encoder, self.target_net, self.critic_tau)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            # if not self.wosac_submission.is_active:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
            if self.global_rank == 0:
                # epoch_wosac_metrics["epoch"] = (
                #     self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                # )
                # self.logger.log_metrics(epoch_wosac_metrics)
                for key, value in epoch_wosac_metrics.items():
                    self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.wosac_metrics.reset()
            self.minADE.reset()


