import copy

from lightning import LightningModule

import random
from collections import deque
import torch.nn as nn
import torch
import numpy as np
from src.smart.tokens.my_token_processor import TokenProcessor
from src.smart.modules.smart_decoder import SMARTDecoder


class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.gamma = 0.99
        self.alpha = self.encoder.agent_encoder.alpha

        self.reg_mult = 0.5

        self.Q_max = 1.0 / (self.reg_mult * (1 - self.gamma))
        self.Q_min = - 1.0 / (self.reg_mult * (1 - self.gamma))

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.batch_replay=False

        if self.batch_replay:
            self.replay_buffer = deque(maxlen=4000)
        else:
            self.replay_buffer = deque(maxlen=100)

        self.reward_w= 1
        self.use_target_q=True
        self.soft_update=True

        if self.reward_w and self.use_target_q:
            self.target_net=SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
            )
            self.target_net.load_state_dict(self.encoder.state_dict())

            if self.soft_update:
                self.critic_tau = 0.005
                self.critic_target_update_frequency = 1

            else:
                self.critic_target_update_frequency = 4

    def rollout(self, tokenized_map, tokenized_agent):
        pred = self.encoder.inference(
            tokenized_map,
            tokenized_agent,
            sampling_scheme=self.training_rollout_sampling,
        )

        if self.batch_replay:
            for i in range(tokenized_agent["num_graphs"]):
                tokenized_agent_rollout={}
                agent_mask= tokenized_agent['batch']==i
                for key in ["sampled_pos","sampled_heading","sampled_idx","valid_mask","type","shape"]:
                    tokenized_agent_rollout[key]=pred[key][agent_mask]

                map_mask=tokenized_map["batch"]==i
                tokenized_map_rollout = {}
                for key in ["position","orientation","token_idx","type","pl_type","light_type"]:
                    tokenized_map_rollout[key]=tokenized_map[key][map_mask]
                self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

        else:
            tokenized_agent_rollout = {}
            for key in ["sampled_pos", "sampled_heading", "sampled_idx", "valid_mask", "type", "shape"]:
                tokenized_agent_rollout[key] = pred[key]

            tokenized_agent_rollout['batch'] = tokenized_agent['batch']
            tokenized_map_rollout = {}

            for key in ["position", "orientation", "token_idx", "type", "pl_type", "light_type","batch"]:
                tokenized_map_rollout[key] = tokenized_map[key]

            self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        pred_dict = self.encoder(tokenized_map, tokenized_agent)

        q_value =pred_dict["q_value"]

        q = q_value[:, :-1]

        pi = torch.softmax(q / self.alpha, dim=-1)  # Compute policy
        logpi= torch.log(pi + 1e-10)
        entropy = -torch.sum(pi * logpi, dim=-1)

        action = tokenized_agent["sampled_idx"][:, 2:].reshape(-1)

        current_Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        v=  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

        current_V = v[:, :-1]

        if self.use_target_q:
            with torch.no_grad():
                next_q = self.target_net(tokenized_map, tokenized_agent,kl_loss=False)["q_value"][:, 1:]
                target_v = self.alpha * torch.logsumexp(next_q / self.alpha, dim=-1, keepdim=False)
        else:
            target_v=v[:, 1:].detach()


        action_logprob = logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        done = torch.zeros_like(target_v)

        done[:, -1] = 1

        rewards = current_Q - (1 - done) * self.gamma * target_v

        valid_mask = tokenized_agent["valid_mask"]

        state_mask=valid_mask[:, 1:-1]

        state_action_mask = valid_mask[:, 2:] & state_mask

        reward=rewards[state_action_mask]

        constraint_loss = torch.relu(-reward).mean()

        div = 'kl'
        #TO DO: detach gradient, clip reward

        if div=="kl":
            alpha=1e-3#*(self.global_step/10000+1e-2)
            reward=torch.clamp_min(reward,min=alpha*1e-6)
            reward_loss= -alpha*((reward/alpha).log()+1)
        elif div == "rkl":
            alpha=0.1
            reward=torch.clamp_min(reward,min=alpha*(-1+np.log(1e-2)))
            reward_loss= alpha*(-reward/alpha-1).exp()
            # with torch.no_grad():
            #     phi_grad = torch.exp(-reward)
            # reward_loss = -(phi_grad * reward)

           # reward_loss= reward_loss.detach()*reward
        elif div=="sh":
            alpha=1
            reward_loss= -1/(1/reward+1/alpha)
        elif div =='js':
            alpha=0.1
            reward=torch.clamp_min(reward,min=alpha*(-np.log(2-1e-4)))
            reward_loss= -alpha*(2-(-reward/alpha).exp()).log()
            # with torch.no_grad():
            #     phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            # reward_loss = -(phi_grad * reward)
        elif div=="tv":
            if key == 'expert':
                reward = torch.clamp_max(reward,max=1)
            else:
                reward=torch.clamp_min(reward,min=-1)
            reward_loss= -reward
        else:
            alpha = 0.025

            reward_loss= -reward+reward.square()/(4*alpha)

        entropy =entropy[state_mask].mean()

        value_loss=(current_V-target_v)[state_action_mask]

        self.log("train/"+key+"_V", current_V[state_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q[state_action_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward_loss", reward_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

        return  reward,reward_loss,value_loss, state_action_mask,action_logprob,entropy,constraint_loss

    def collect_agent(self,num_graphs):

        if self.batch_replay:
            tokenized_agent_rollout={}
            tokenized_map_rollout = {}

            for key in ["sampled_pos", "sampled_heading", "sampled_idx", "valid_mask", "type", "shape","batch"]:
                tokenized_agent_rollout[key]=[]
            for key in ["position", "orientation", "token_idx", "type", "pl_type", "light_type","batch"]:
                tokenized_map_rollout[key]=[]

            batch_list=random.sample(self.replay_buffer, num_graphs)

            for i,(map,agent) in enumerate(batch_list):
                for key in ["sampled_pos", "sampled_heading", "sampled_idx", "valid_mask", "type", "shape"]:
                    tokenized_agent_rollout[key].append(agent[key])
                tokenized_agent_rollout["batch"].append(torch.zeros_like(agent["type"])+i)

                for key in ["position", "orientation", "token_idx", "type", "pl_type", "light_type"]:
                    tokenized_map_rollout[key].append(map[key])
                tokenized_map_rollout["batch"].append(torch.zeros_like(map["type"])+i)

            for key in ["sampled_pos", "sampled_heading", "sampled_idx", "valid_mask", "type", "shape","batch"]:
                tokenized_agent_rollout[key]=torch.cat(tokenized_agent_rollout[key])
            for key in ["position", "orientation", "token_idx", "type", "pl_type", "light_type","batch"]:
                tokenized_map_rollout[key]=torch.cat(tokenized_map_rollout[key])
        else:
            batch_list=random.sample(self.replay_buffer, 1)
            tokenized_map_rollout,tokenized_agent_rollout=batch_list[0]

        tokenized_agent_rollout["trajectory_token_veh"]=self.token_processor.trajectory_token_veh
        tokenized_agent_rollout["trajectory_token_ped"]=self.token_processor.trajectory_token_ped
        tokenized_agent_rollout["trajectory_token_cyc"]=self.token_processor.trajectory_token_cyc
        tokenized_agent_rollout['num_graphs'] = num_graphs
        tokenized_map_rollout["token_traj_src"]=self.token_processor.map_token_traj_src

        return tokenized_map_rollout,tokenized_agent_rollout

    def iq_update(self, tokenized_map, tokenized_agent):

        expert_reward,expert_reward_loss,expert_value_loss, expert_valid,expert_logprob,_,expert_constraint_loss = self.get_QV(tokenized_map, tokenized_agent)

        expert_nll=-expert_logprob[expert_valid].mean()

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        if self.reward_w==0:
            loss =expert_nll
        else:
            tokenized_map_rollout,tokenized_agent_rollout = self.collect_agent(tokenized_agent['num_graphs'])

            agent_reward,agent_reward_loss ,agent_value_loss,agent_valid,_,agent_entropy,agent_constraint_loss = self.get_QV(tokenized_map_rollout,tokenized_agent_rollout, key='agent')

            agent_ratio=0

            reward_loss= (expert_reward_loss.sum()*(1-agent_ratio)+agent_reward_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

            self.log("train/reward_loss", reward_loss.item(), on_step=True, batch_size=1)

            agent_ratio=1

            value_loss= (expert_value_loss.sum()*(1-agent_ratio)+agent_value_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

            self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

            reward_mean= (expert_reward.sum()*(1-agent_ratio)+agent_reward.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

            self.log("train/reward_mean", reward_mean.item(), on_step=True, batch_size=1)

            critic_loss=self.reward_w*(reward_loss+reward_mean)#self.global_step/10000*+expert_constraint_loss+agent_constraint_loss

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            loss = expert_nll+ critic_loss

        return loss

    def process_data(self,data):
        map=data["tokenized_map"]
        agent=data["tokenized_agent"]

        tokenized_agent={}

        tokenized_agent['sampled_pos'] = agent["sampled_pos"]
        tokenized_agent['sampled_heading'] = agent['sampled_heading']
        tokenized_agent['sampled_idx'] = agent["sampled_idx"]

        tokenized_agent["gt_pos"] = agent["sampled_pos"]
        tokenized_agent["gt_heading"]  = agent['sampled_heading']
        tokenized_agent["gt_idx"] = agent["sampled_idx"]

        tokenized_agent['valid_mask'] = agent['valid_mask']
        tokenized_agent['type'] = agent['type']
        tokenized_agent['batch'] = agent['batch']
        tokenized_agent['num_graphs'] = data.num_graphs
        tokenized_agent['shape'] = agent['shape']

        agent_shape, token_traj_all, token_traj = self.token_processor._get_agent_shape_and_token_traj(
            agent['type']
        )
        tokenized_agent['token_traj'] = token_traj
        tokenized_agent['token_traj_all'] = token_traj_all
        tokenized_agent['token_agent_shape'] = agent_shape
        tokenized_agent['trajectory_token_veh'] = self.token_processor.trajectory_token_veh
        tokenized_agent['trajectory_token_ped'] = self.token_processor.trajectory_token_ped
        tokenized_agent['trajectory_token_cyc'] = self.token_processor.trajectory_token_cyc

        tokenized_map={}

        tokenized_map["position"]= map["position"]
        tokenized_map["orientation"]=  map["orientation"]
        tokenized_map["token_idx"]=  map["token_idx"].long()
        tokenized_map["type"]= map["type"].long()
        tokenized_map["pl_type"]= map["pl_type"].long()
        tokenized_map["light_type"]= map["light_type"].long()
        tokenized_map["batch"]= map["batch"]
        tokenized_map["token_traj_src"]=self.token_processor.map_token_traj_src

        return tokenized_map, tokenized_agent

    def training_step(self, data, batch_idx):

        if "traj_pos" in data.keys():
            tokenized_map, tokenized_agent = self.token_processor(data)
        else:
            tokenized_map, tokenized_agent = self.process_data(data)

        if self.reward_w!=0 and (len(self.replay_buffer) < self.replay_buffer.maxlen or self.global_step % 10 == 0):
            with torch.no_grad():
                self.encoder.eval()
                self.rollout(tokenized_map, tokenized_agent)
                self.encoder.train()

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.reward_w!=0 and self.use_target_q and self.global_step % self.critic_target_update_frequency == 0  :

            if self.soft_update:
                soft_update(self.encoder,self.target_net,self.critic_tau)
            else:
                hard_update(self.encoder,self.target_net)

        return loss

def soft_update( net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

