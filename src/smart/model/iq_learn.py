import copy

from lightning import LightningModule

import random
from collections import deque
import torch.nn as nn
import torch
import numpy as np
from src.smart.tokens.my_token_processor import TokenProcessor


class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.replay_buffer = deque(maxlen=100)
        self.critic_tau = 0.005
        self.critic_target_update_frequency = 1
        self.gamma = 0.99
        self.alpha = self.encoder.agent_encoder.alpha

        self.reg_mult = 0.5

        self.Q_max = 1.0 / (self.reg_mult * (1 - self.gamma))
        self.Q_min = - 1.0 / (self.reg_mult * (1 - self.gamma))

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def rollout(self, tokenized_map, tokenized_agent):
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
        tokenized_map_rollout={}

        tokenized_map_rollout["position"]=tokenized_map["position"]
        tokenized_map_rollout["orientation"]=tokenized_map["orientation"]
        tokenized_map_rollout["token_idx"]=tokenized_map["token_idx"]
        tokenized_map_rollout["type"]=tokenized_map["type"]
        tokenized_map_rollout["pl_type"]=tokenized_map["pl_type"]
        tokenized_map_rollout["light_type"]=tokenized_map["light_type"]
        tokenized_map_rollout["batch"] = tokenized_map["batch"]

        self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))


    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

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
        div = 'js'

        if div=="kl":
            alpha=1
            reward_loss= -alpha*((reward/alpha).log()+1)
        elif div == "rkl":
            alpha=0.1
            reward_loss= (-reward/alpha-1).exp() * alpha
           # reward_loss= reward_loss.detach()*reward
        elif div=="sh":
            alpha=1
            reward_loss= -1/(1/reward+1/alpha)
        elif div =='js':
            alpha=1
            reward=torch.clamp_min(reward,min=alpha*(-np.log(2)+0.01))
            reward_loss= -alpha*(2-(-reward/alpha).exp()).log()
        else:
            alpha = 0.01

            reward_loss= -reward+reward.square()/(4*alpha)

        entropy =entropy[state_mask].mean()

        value_loss=(current_V-target_v)[state_action_mask]

        self.log("train/"+key+"_V", current_V[state_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q[state_action_mask].mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward_loss", reward_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)

        return  reward,reward_loss,value_loss, state_action_mask,action_logprob,entropy

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
        map_batch["token_traj_src"]=self.token_processor.map_token_traj_src

        return map_batch,agent_batch

    def iq_update(self, tokenized_map, tokenized_agent):

        agent_tokenized_map, agent_tokenized_agent = self.collate_agent(random.sample(self.replay_buffer,1)) #random.sample(self.replay_buffer,1)[0]

        expert_reward,expert_reward_loss,expert_value_loss, expert_valid,expert_logprob,_ = self.get_QV(tokenized_map, tokenized_agent)

        agent_reward,agent_reward_loss ,agent_value_loss,agent_valid,_,agent_entropy = self.get_QV(agent_tokenized_map, agent_tokenized_agent, key='agent')

        expert_nll=-expert_logprob[expert_valid].mean()

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        agent_ratio=0.5
        reward_w=0.1

        reward_loss= (expert_reward_loss.sum()*(1-agent_ratio)+agent_reward_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

        self.log("train/reward_loss", reward_loss.item(), on_step=True, batch_size=1)

        #agent_mean_reward = agent_reward.mean()
        agent_ratio=1 #0.5

        value_loss= (expert_value_loss.sum()*(1-agent_ratio)+agent_value_loss.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

        self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

        reward_mean= (expert_reward.sum()*(1-agent_ratio)+agent_reward.sum()*agent_ratio)/(expert_valid.sum()*(1-agent_ratio)+agent_valid.sum()*agent_ratio)

        self.log("train/reward_mean", reward_mean.item(), on_step=True, batch_size=1)

        critic_loss = expert_nll+reward_w*(reward_loss+value_loss)#-self.alpha*agent_entropy

        return critic_loss

    def process_data(self,data):
        map=data["tokenized_map"]
        agent=data["tokenized_agent"]

        tokenized_agent={}

        tokenized_agent['sampled_pos'] = agent["sampled_pos"]
        tokenized_agent['sampled_heading'] = agent['sampled_heading']
        tokenized_agent['sampled_idx'] = agent["sampled_idx"]

        tokenized_agent["gt_pos"] = copy.deepcopy(agent["sampled_pos"])##.clone()
        tokenized_agent["gt_heading"]  =  copy.deepcopy(agent['sampled_heading'])#.clone()
        tokenized_agent["gt_idx"] =  copy.deepcopy(agent["sampled_idx"])#.clone()

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
        tokenized_map["token_idx"]=  map["token_idx"].to(torch.long)
        tokenized_map["type"]= map["type"].to(torch.long)
        tokenized_map["pl_type"]= map["pl_type"].to(torch.long)
        tokenized_map["light_type"]= map["light_type"].to(torch.long)
        tokenized_map["batch"]= map["batch"]
        tokenized_map["token_traj_src"]=self.token_processor.map_token_traj_src

        return tokenized_map, tokenized_agent


    def training_step(self, data, batch_idx):
        # time1=time.time()
        #tokenized_map, tokenized_agent = self.token_processor(data)
        tokenized_map, tokenized_agent = self.process_data(data)

        if len(self.replay_buffer) < self.replay_buffer.maxlen or self.global_step % 10 == 0:
            with torch.no_grad():
                self.encoder.eval()
                self.rollout(tokenized_map, tokenized_agent)
                self.encoder.train()

        loss = self.iq_update(tokenized_map, tokenized_agent)

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


# def soft_update( net, target_net, tau):
#     for param, target_param in zip(net.parameters(), target_net.parameters()):
#         target_param.data.copy_(tau * param.data +
#                                 (1 - tau) * target_param.data)
