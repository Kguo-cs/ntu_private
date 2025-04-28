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

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.batch_replay=False

        if self.batch_replay:
            self.replay_buffer = deque(maxlen=4000)
        else:
            self.replay_buffer = deque(maxlen=1)

        self.reward_w = 1
        self.use_target_q=True
        self.soft_update=True

        self.rollout_freq=1
        self.target_net = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        self.target_net.load_state_dict(self.encoder.state_dict())

        #self.automatic_optimization=False

        if self.reward_w and self.use_target_q:

            if self.soft_update:
                self.critic_tau = 1
                self.critic_target_update_frequency = 1
            else:
                self.critic_target_update_frequency = 1

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

    def compute_reward_loss(self,reward):

        div = 'rkl'
        # TO DO: detach gradient, clip reward, gmm, refine by KL constrained

        eps = 1e-1

        if div == "kl":
            alpha = 1  # *(self.global_step/10000+1e-2)
            reward = torch.clamp(reward, max=alpha / eps, min=alpha * eps)
            reward_loss = -alpha * ((reward / alpha).log() + 1)
        elif div == "rkl":
            alpha = 10
            # reward=torch.clamp(reward,max=alpha*(-1-np.log(eps)),min=alpha*(-1+np.log(eps)))
            reward_loss = alpha * (-reward / alpha - 1).exp()
            # with torch.no_grad():
            #     phi_grad = torch.exp(-reward)
            # reward_loss = -(phi_grad * reward)
        # reward_loss= reward_loss.detach()*reward
        elif div == "sh":
            alpha = 1
            reward_loss = -1 / (1 / reward + 1 / alpha)
        elif div == 'js':
            alpha = 1
            reward = torch.clamp_min(reward, min=alpha * (np.log(1 / 2 + eps)))  # ,max=alpha*(np.log(1/2+1/eps))
            reward_loss = -alpha * (2 - (-reward / alpha).exp()).log()
            # with torch.no_grad():
            #     phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            # reward_loss = -(phi_grad * reward)
        elif div == "tv":
            reward_loss = -reward
        elif div == 'x2':
            alpha = 1
            reward = torch.clamp(reward, min=2 * (1 - 1 / eps), max=2 * (1 - eps))

            reward_loss = -reward + reward.square() / (4 * alpha)
        else:
            reward_loss = -reward

        return reward_loss

    def get_network_QV(self,network,tokenized_map, tokenized_agent,action):

        q_value = network(tokenized_map, tokenized_agent)["q_value"]

        v_value =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

        q = q_value[:, :-1]

        current_Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        current_V = v_value[:, :-1]

        next_V = v_value[:, 1:]

        done = torch.zeros_like(next_V)

        done[:, -1] = 1

        next_V = (1 - done) * next_V

        reward = current_Q - self.gamma * next_V.detach()  # next_V#

        return q,current_Q,current_V,next_V,reward

    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        action = tokenized_agent["sampled_idx"][:, 2:].reshape(-1)

        valid_mask = tokenized_agent["valid_mask"]

        state_mask=valid_mask[:, 1:-1]

        action_mask= valid_mask[:, 2:]

        state_action_mask = action_mask & state_mask

        q,current_Q,current_V,next_V,reward=self.get_network_QV(self.encoder, tokenized_map, tokenized_agent,action)

        with torch.no_grad():
            target_q, target_current_Q, target_current_V,target_next_V, target_reward = self.get_network_QV(self.target_net, tokenized_map, tokenized_agent,action)

        reward = current_Q - self.gamma * target_next_V  # next_V#

        pi = torch.softmax(q / self.alpha, dim=-1)

        logpi= torch.log(pi + 1e-10)

        action_nll = -logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])[state_action_mask].mean()

        entropy = -torch.sum(pi * logpi, dim=-1)

        reward = reward[state_action_mask]

        current_Q=current_Q[state_action_mask]

        current_V=current_V[state_mask]

        entropy =entropy[state_mask]

        current_V_diff=current_V-target_current_V[state_mask]

        next_V_diff=(next_V-target_next_V)[action_mask]

        reward_diff=reward-target_reward[state_action_mask]

        Q_diff=current_Q-target_current_Q[state_action_mask]

        self.log("train/"+key+"_V", current_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_NextV", next_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_V_diff", current_V_diff.abs().mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_NextV_diff", next_V_diff.abs().mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward_diff", reward_diff.abs().mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q_diff", Q_diff.abs().mean().item(), on_step=True, batch_size=1)

        return  reward,current_V,current_Q,next_V,current_V_diff,next_V_diff,Q_diff,action_nll

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

        expert_reward,expert_V,expert_Q,expert_next_V,expert_current_V_diff,expert_next_V_diff,expert_Q_diff,expert_nll= self.get_QV(tokenized_map, tokenized_agent)

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        if self.reward_w==0:
            loss =expert_nll
        else:
            tokenized_map_rollout,tokenized_agent_rollout = self.collect_agent(tokenized_agent['num_graphs'])

            agent_reward,agent_V,agent_Q,agent_next_V,agent_current_V_diff,agent_next_V_diff,agent_Q_diff ,_= self.get_QV(tokenized_map_rollout,tokenized_agent_rollout, key='agent')

            # agent_ratio=0
            #
            # reward_loss= (expert_reward_loss.sum()*(1-agent_ratio)+agent_reward_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)
            #
            # self.log("train/reward_loss", reward_loss.item(), on_step=True, batch_size=1)
            # reward_loss=self.compute_reward_loss(reward)

            # agent_ratio=1
            #
            # reward_mean= (expert_reward.sum()*(1-agent_ratio)+agent_reward.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)
            #
            # self.log("train/reward_mean", reward_mean.item(), on_step=True, batch_size=1)

            #critic_loss=self.reward_w*(reward_loss+reward_mean)#self.global_step/10000*+expert_constraint_loss+agent_constraint_loss

            div='tv'
            alpha=1
            eps=1e-3

            if div=="lsif":
                critic_loss=-expert_reward.exp().mean()+1/2*(2*agent_reward).exp().mean()
            elif div == 'bce':
                critic_loss=((-expert_reward/1).exp()+1).log().mean()+((agent_reward/1).exp()+1).log().mean()
            elif div=='ukl':
                critic_loss = -expert_reward.mean() + agent_reward.exp().mean()
            elif div=='rkl':
                # phi_grad = torch.exp(-expert_reward).detach()
                # critic_loss =  -(phi_grad*expert_reward).mean()+agent_reward.mean()
                critic_loss= alpha *(-expert_reward / alpha  ).exp().mean()+agent_reward.mean()
            elif div=='tv':
                critic_loss= (-expert_reward ).mean()+agent_reward.mean()
            elif div=='x2':
                critic_loss= (-expert_reward +expert_reward.square()/ (4 * alpha)).mean()+agent_reward.mean()
            elif div=='kl':
                # expert_reward = torch.clamp_min(expert_reward, min=alpha * eps)
                critic_loss = -alpha * ((expert_reward / alpha).log().mean() + 1)+agent_reward.mean()
            elif div=='sh':
                critic_loss = - (expert_reward / (1+expert_reward)).mean() +agent_reward.mean()
            elif div=='js':
                phi_grad = torch.exp(-expert_reward) / (2 - torch.exp(-expert_reward))
                critic_loss =  -(phi_grad.detach()*expert_reward).mean()+agent_reward.mean()
                # expert_reward = torch.clamp_min(expert_reward, min=alpha * (np.log(1 / 2 + eps)))  # ,max=alpha*(np.log(1/2+1/eps))
                #critic_loss = -(2 - (-expert_reward / alpha).exp()).log().mean()+agent_reward.mean()
            else:
                critic_loss= (expert_reward-1 ).square().mean()+(agent_reward+1).square().mean()

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            constraint_loss=10*(expert_current_V_diff.square().mean()+agent_current_V_diff.square().mean() )#10*000/(self.global_step+1)10*

            constraint_ratio=critic_loss/constraint_loss

            self.log("train/constraint_ratio", constraint_ratio.item(), on_step=True, batch_size=1)

            # constraint_loss=constraint_ratio.detach()*0.02*constraint_loss

            self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss =  critic_loss#+constraint_loss #expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1

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

        if self.reward_w!=0 and (self.global_step % self.rollout_freq == 0 or len(self.replay_buffer)<self.replay_buffer.maxlen):
            with torch.no_grad():
                #self.encoder.eval()
                self.rollout(tokenized_map, tokenized_agent)
                #self.encoder.train()

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        # Get the optimizers
        # opt1, opt2 = self.optimizers()
        # opt1.zero_grad()
        # opt2.zero_grad()
        # self.manual_backward(loss)
        # opt1.step()
        # opt2.step()

        if self.reward_w!=0 and self.use_target_q and self.global_step % self.critic_target_update_frequency == 0  :

            if self.soft_update:
                tau=1e-4 #self.critic_tau/(self.global_step+1)
                soft_update(self.encoder,self.target_net,tau)
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

