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





def rollout(encoder, tokenized_map, tokenized_agent):
    encoder.eval()
    with torch.no_grad():
        pred = encoder.inference(
            tokenized_map,
            tokenized_agent,
            None
        )
    encoder.train()

    tokenized_agent_rollout = {}
    tokenized_agent_rollout['num_graphs'] = tokenized_agent['num_graphs']

    if "sampled_idx" in pred.keys():
        for key in ["sampled_idx","sampled_pos", "sampled_heading", "valid_mask","batch", "type", "shape"]:
            tokenized_agent_rollout[key] = pred[key]

        #tokenized_agent_rollout['sampled_idx'] = pred['sampled_idx'].to(torch.int16)

    if "light_idx" in tokenized_agent.keys():
        tokenized_agent_rollout['light_idx'] = pred['light_idx']
        for key in ["lengths_lg", "pos_lg","orient_lg", "batch_lg"]:
            tokenized_agent_rollout[key] = tokenized_agent[key]

    # if self.rollout_freq > 1:
    #     tokenized_map_rollout = {}
    #
    #     for key in tokenized_map.keys():
    #         if key !="map_feature":
    #             tokenized_map_rollout[key]=tokenized_map[key]
    #
    #     self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

    return tokenized_map,tokenized_agent_rollout

