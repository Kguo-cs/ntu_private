import numpy as np
import torch
from collections import deque

import random



class ReplayBuffer:
    def __init__(self, max_len=1):

        self.state_action_list=deque(maxlen=max_len)

        self.value_list=deque(maxlen=max_len)
        self.action_log_probs_list=deque(maxlen=max_len)

    def __len__(self):
        return len(self.state_action_list)

    def initialize(self,map, agent_num,device):

        self.rewards = torch.zeros(self.num_steps, agent_num).to(device)
        self.value_preds = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.returns = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.action_log_probs = torch.zeros(self.num_steps, agent_num).to(device)
        self.actions = torch.zeros(self.num_steps, agent_num).to(device).to(torch.int)
        self.masks = torch.ones(self.num_steps + 1, agent_num).to(device)
        self.masks[-1]=0
        self.map=map

        self.smaple_idx=list(range(self.num_steps))

        random.shuffle(self.smaple_idx)

    def insert(self, sample,step):
        self.state.append(sample["state"])
        self.value_preds[step]=sample["value"]

        if step<self.num_steps:
            self.action_log_probs[step]=sample["prev_log_prob"]
            self.actions[step]=sample["action"]

    def sample(self):
        return {"state_action": self.state_action_list[-1],
                "prev_log_prob":self.action_log_probs_list[-1],
                "adv":self.advantages,
                "value":self.value_preds[:,:-1],
                "return":self.returns
                }


    def compute_advantages(self,gamma=0.99,gae_lambda=0.95):
        self.value_preds=torch.cat([self.value_list[-1],torch.zeros_like(self.value_list[-1][:,:1])],dim=1)

        self.masks = torch.ones_like(self.value_preds)
        self.masks[:,-1]=0

        self.returns = []

        gae = 0
        for step in reversed(range(self.rewards.size(1))):
            delta = (
                self.rewards[:,step]
                + gamma * self.value_preds[:,step + 1] * self.masks[:,step + 1]
                - self.value_preds[:,step]
            )
            gae = (
                delta
                + gamma * gae_lambda * self.masks[:,step + 1] * gae
            )
            #self.returns[step] = gae + self.value_preds[step]
            self.returns.insert(0, gae + self.value_preds[:,step])

        self.returns=torch.stack(self.returns,dim=1)

        advantages = self.returns - self.value_preds[:,:-1]
        # Normalize the advantages
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)





def rollout(encoder, tokenized_map, tokenized_agent,post_sampling=False):
    encoder.eval()
    with torch.no_grad():
        pred = encoder.inference(
            tokenized_map,
            tokenized_agent,
           post_sampling=post_sampling
        )
    encoder.train()

    tokenized_agent.update(pred)
    # tokenized_agent_rollout = tokenized_agent
    # tokenized_agent_rollout['num_graphs'] = tokenized_agent['num_graphs']
    #
    # if "sampled_idx" in pred.keys():
    #     for key in ["sampled_idx","sampled_pos", "sampled_heading", "valid_mask","batch", "type", "shape"]:
    #         tokenized_agent_rollout[key] = pred[key]
    #
    #     #tokenized_agent_rollout['sampled_idx'] = pred['sampled_idx'].to(torch.int16)
    #
    # if "light_idx" in tokenized_agent.keys():
    #     tokenized_agent_rollout['light_idx'] = pred['light_idx']
    #     for key in ["lengths_lg", "pos_lg","orient_lg", "batch_lg"]:
    #         tokenized_agent_rollout[key] = tokenized_agent[key]

    # if self.rollout_freq > 1:
    #     tokenized_map_rollout = {}
    #
    #     for key in tokenized_map.keys():
    #         if key !="map_feature":
    #             tokenized_map_rollout[key]=tokenized_map[key]
    #
    #     self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

    return tokenized_agent


def get_return_diff(reward,log_prob,current_Q,V,alpha,gamma):
    rewards=reward - alpha * log_prob
    returns = torch.zeros_like(V)
    running_return=returns[:,-1]

    for i in range(rewards.size(1)-1,-1,-1):
        running_return = rewards[:, i] + gamma *running_return
        returns[:, i] = running_return

    current_Q_diff = (current_Q - returns[:,:-1])
    V_diff=(V[:,:-1]-returns[:,:-1])

    return current_Q_diff, V_diff

def get_return(s,gamma,eps = 1e-20,reward_type="airl"):

    s=s.detach()

    if reward_type == 'airl':
        rewards = (s + eps).log() - (1 - s + eps).log()
    elif reward_type == 'gail':
        rewards = (s + eps).log()
    elif reward_type == 'raw':
        rewards = s
    elif reward_type == 'airl-positive':
        rewards = (s + eps).log() - (1 - s + eps).log() + 20
    elif reward_type == 'revise':
        d_x = (s + eps).log()
        rewards = d_x + (-1 - (-d_x).log())

    returns = torch.zeros_like(rewards)
    running_return=returns[:,-1]

    for i in reversed(range(rewards.shape[1])):
        running_return = rewards[:, i] + gamma *running_return
        returns[:, i] = running_return

    # dones = torch.zeros_like(rewards)
    # dones[:,-1]=1
    #* (1.0 - dones[:, i])

    # returns1 = torch.zeros_like(rewards)
    # R = 0
    # for t in reversed(range(len(rewards))):
    #     R = rewards[t] + gamma * R * (1.0 - dones[t])
    #     returns1[t] = R

    return returns,rewards


def compute_advantages(rewards, values,gamma=0.99,lam=0.95):#0.95

    dones = torch.zeros_like(rewards)
    dones[:,-1]=1

    # returns1 = torch.zeros_like(rewards)
    # R = 0
    # for t in reversed(range(len(rewards))):
    #     R = rewards[t] + gamma * R * (1.0 - dones[t])
    #     returns1[t] = R

    advantages = torch.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(rewards.shape[1])):
        if t == rewards.shape[1] - 1:
            next_value = 0
            next_non_terminal = 1.0 - dones[:,t]
        else:
            next_value = values[:,t + 1]
            next_non_terminal = 1.0 - dones[:,t]
        delta = rewards[:,t] + gamma * next_value * next_non_terminal - values[:,t]
        advantages[:,t] = last_adv = delta + gamma * lam * next_non_terminal * last_adv
    returns = advantages + values

    # advantages = returns - value_preds[:,:-1]
    # Normalize the advantages
    #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    #
    # returns = []
    #
    # gae = 0
    # for step in reversed(range(rewards.size(1))):
    #     delta = (
    #         rewards[:,step]
    #         + gamma * value_preds[:,step + 1] * dones[:,step + 1]
    #         - value_preds[:,step]
    #     )
    #     gae = (
    #         delta
    #         + gamma * gae_lambda * dones[:,step + 1] * gae
    #     )
    #     #self.returns[step] = gae + self.value_preds[step]
    #     returns.insert(0, gae + value_preds[:,step])
    #
    # returns=torch.stack(returns,dim=1)

    return advantages,returns
