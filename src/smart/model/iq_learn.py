import copy
from lightning import LightningModule
import random
from collections import deque
import torch.nn as nn
import torch
import numpy as np
from src.smart.modules.smart_decoder import SMARTDecoder
import pickle
from torch_scatter import scatter_mean,scatter_max

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
        self.use_target_q=False
        self.soft_update=True

        self.rollout_freq=1

        if self.reward_w and self.use_target_q:
            self.target_net = SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
            )
            self.target_net.load_state_dict(self.encoder.state_dict())

            if self.soft_update:
                self.critic_tau = 1
                self.critic_target_update_frequency = 1
            else:
                self.critic_target_update_frequency = 1

    def rollout(self, tokenized_map, tokenized_agent):
        self.encoder.eval()
        with torch.no_grad():
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.validation_rollout_sampling,
            )
        self.encoder.train()

        #self.log("rollout_entropy",pred["rollout_entropy"].mean().item(), on_step=True, batch_size=1)

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
            tokenized_map_rollout = tokenized_map#{"map_feature":tokenized_map["map_feature"]}

            # for key in ["position", "orientation", "token_idx", "type", "pl_type", "light_type","batch"]:
            #     tokenized_map_rollout[key] = tokenized_map[key]

            self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

    def get_network_QV(self,network,tokenized_map, tokenized_agent,action,key,cumulative_mask):

        q_value = network(tokenized_map, tokenized_agent)["q_value"]

        q = q_value[:, :-1]

        current_Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        # if key=='expert':
        #     # pi = torch.softmax(q / self.alpha, dim=-1)
        #     #
        #     # logpi = torch.log(pi + 1e-10)
        #     #
        #     # log_prob = logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])
        #
        #     current_V= current_Q - self.alpha * log_prob
        #
        #     v_value=torch.cat([current_V,torch.zeros_like(current_V[:,:1])],dim=1)
        # else:
        v_value =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

        current_V = v_value[:, :-1]

        next_V = v_value[:, 1:]

        dones = torch.zeros_like(next_V)

        dones[~cumulative_mask[:,1:]] = 1

        dones[:, -1] = 1

        next_V=(1 - dones) * next_V

        reward = current_Q - self.gamma * next_V

        return q,current_Q,v_value,current_V,next_V,reward,dones

    def get_QV(self, tokenized_map, tokenized_agent, key='expert'):

        action = tokenized_agent["sampled_idx"][:, 2:].reshape(-1)

        valid_mask = tokenized_agent["valid_mask"][:, 1:]

        state_mask = valid_mask[:, :-1]

        action_mask= valid_mask[:, 1:]

        cumulative_mask = valid_mask.float().cumsum(dim=1) == torch.arange(1, valid_mask.shape[1] + 1,device=valid_mask.device).float()

        state_action_mask = action_mask & state_mask

        q,current_Q,V,current_V,next_V,reward,dones=self.get_network_QV(self.encoder, tokenized_map, tokenized_agent,action,key,cumulative_mask)

        pi = torch.softmax(q / self.alpha, dim=-1)

        logpi= torch.log(pi + 1e-10)

        log_prob=logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

        action_nll = -log_prob[state_action_mask].mean()

        entropy = -torch.sum(pi * logpi, dim=-1)

        rewards=reward - self.alpha * log_prob

        returns = torch.zeros_like(V)

        running_return=returns[:,-1]

        # Convert done mask to 1s and 0s if needed
        for i in range(rewards.size(1)-1,-1,-1):
            running_return = (rewards[:, i] + self.gamma *running_return)*cumulative_mask[:,i+1] #* (1.0 - dones[:, i])
            returns[:, i] = running_return

        # if key=="expert":
        #     returns += 0.1

        current_returns=returns[:,:-1]
        next_returns=returns[:,1:]

        if self.use_target_q and key=="expert":
            with torch.no_grad():
                target_q, target_current_Q, target_V,target_current_V,target_next_V, target_reward,_ = self.get_network_QV(self.target_net, tokenized_map, tokenized_agent,action,key,cumulative_mask)
        else:
            target_V = 1 #returns#cannot .detach()
            #reward= current_Q - self.gamma * next_returns

        reward = reward[cumulative_mask[:,1:]]

        current_Q=current_Q[state_action_mask]

        current_Q_diff=(current_Q-current_returns)[cumulative_mask[:,:-1]]

        current_V_diff=(current_V-current_returns)[cumulative_mask[:,:-1]]

        last_V=V[:,-1][valid_mask[:,-1]]

        V_diff=(V-target_V)[cumulative_mask]#last_V#(V-target_V)[:,-1][valid_mask[:,-1]]

        current_V=current_V[state_mask]

        entropy =entropy[state_mask]

        self.log("train/"+key+"_V", current_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q_diff", current_Q_diff.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_V_diff", current_V_diff.mean().item(), on_step=True, batch_size=1)

        return  reward,current_V,current_Q,next_V,V_diff,current_V_diff,next_V_diff,action_nll,entropy

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
        #tokenized_map_rollout["token_traj_src"]=self.token_processor.map_token_traj_src

        return tokenized_map_rollout,tokenized_agent_rollout

    def iq_update(self, tokenized_map, tokenized_agent):

        expert_reward,expert_V,expert_Q,expert_next_V,expert_V_diff,expert_current_V_diff,expert_next_V_diff,expert_nll,_= self.get_QV(tokenized_map, tokenized_agent)

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)


        if self.reward_w==0:
            loss =expert_nll
        else:
            # if (self.global_step % self.rollout_freq == 0 or len(self.replay_buffer) < self.replay_buffer.maxlen):
            #     self.rollout(tokenized_map, tokenized_agent)

            # tokenized_map_rollout,tokenized_agent_rollout = self.collect_agent(tokenized_agent['num_graphs'])
            #
            # agent_reward,agent_V,agent_Q,agent_next_V,agent_V_diff,agent_current_V_diff,agent_next_V_diff ,_,agent_entropy= self.get_QV(tokenized_map_rollout,tokenized_agent_rollout, key='agent')
            agent_reward=torch.zeros_like(expert_reward)

            div='x2'
            alpha=0.1
            eps=1e-3

            if div=="lsif":
                critic_loss=-expert_reward.exp().mean()+1/2*(2*agent_reward).exp().mean()
            elif div == 'bce':
                critic_loss=((-expert_reward/1).exp()+1).log().mean()+((agent_reward/1).exp()+1).log().mean()
            elif div=='ukl':
                critic_loss = -expert_reward.mean() + agent_reward.exp().mean()
            elif div=='exp':
                critic_loss = (-expert_reward ).exp().mean()+ agent_reward.exp().mean()
            elif div=='rkl':
                # phi_grad = torch.exp(-expert_reward).detach()
                # critic_loss =  -(phi_grad*expert_reward).mean()+agent_reward.mean()
                critic_loss= alpha *(-expert_reward / alpha  ).exp().mean()+agent_reward.mean()
            elif div=='tv':
                critic_loss= (-expert_reward ).mean()+agent_reward.mean()
            elif div=='x2':
                critic_loss= (-expert_reward +expert_reward.square()/ (4 * alpha)).mean()+agent_reward.mean()
            elif div=='kl':
                expert_reward = torch.clamp_min(expert_reward, min=alpha * eps)
                critic_loss = -alpha * ((expert_reward / alpha).log().mean() + 1)+agent_reward.mean()
            elif div=='sh':
                critic_loss = - (expert_reward / (1+expert_reward)).mean() +agent_reward.mean()
            elif div=='js':
                # phi_grad = torch.exp(-expert_reward) / (2 - torch.exp(-expert_reward))
                # critic_loss =  -(phi_grad.detach()*expert_reward).mean()+agent_reward.mean()
                expert_reward = torch.clamp_min(expert_reward, min=alpha * (np.log(1 / 2 + eps)))  # ,max=alpha*(np.log(1/2+1/eps))
                critic_loss = -(2 - (-expert_reward / alpha).exp()).log().mean()+agent_reward.mean()
            else:
                critic_loss= (expert_reward-1 ).square().mean()+(agent_reward+1).square().mean()

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            constraint_loss=0*(expert_V_diff.square()).mean() #(expert_V_diff.square().mean() +agent_V_diff.square().mean())#10*000/(self.global_step+1)10*

            constraint_ratio=critic_loss/constraint_loss

            self.log("train/constraint_ratio", constraint_ratio.item(), on_step=True, batch_size=1)

            # constraint_loss=constraint_ratio.detach()*0.02*constraint_loss

            self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss =  critic_loss+constraint_loss #-0.01*agent_entropy.mean() #expert_nll+expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1

        return loss

    def process_data(self,data):
        map=data["tokenized_map"]
        agent=data["tokenized_agent"]

        tokenized_agent={}

        tokenized_agent['sampled_pos'] = agent["sampled_pos"]
        tokenized_agent['sampled_heading'] = agent['sampled_heading']
        tokenized_agent['sampled_idx'] = agent["sampled_idx"].long()

        tokenized_agent["gt_pos"] = tokenized_agent["sampled_pos"]
        tokenized_agent["gt_heading"]  =tokenized_agent["sampled_heading"]
        tokenized_agent["gt_idx"] = tokenized_agent['sampled_idx']

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

        if "ln_id" in map._mapping.keys():
            ln_id = map["ln_id"]
            batch=map["batch"]
            #unique_ids, ln_id = torch.unique(ln_id, return_inverse=True)
            # Step 1: compute per-graph max ln_id
            #ln_id_max, _ = scatter_max(ln_id, batch, dim=0, dim_size=data.num_graphs)

            ln_id_max=ln_id[torch.where(ln_id[1:]<ln_id[:-1])]


            # Step 2: compute cumulative offset for each graph (ln_num per graph)
            offsets = torch.zeros([data.num_graphs],device=ln_id_max.device)
            offsets[1:] = torch.cumsum(ln_id_max + 1, dim=0)
            batch_ln_id = (ln_id + offsets[batch]).long()
            batch=scatter_mean(batch,batch_ln_id,dim=0)

            # Step 3: gather offsets using batch
            tokenized_map["ln_id"] = batch_ln_id
            tokenized_map["rel_position"]=map["rel_position"]
            tokenized_map["batch"]=batch

        return tokenized_map, tokenized_agent

    def training_step(self, data, batch_idx):

        if "traj_pos" in data.keys():
            tokenized_map, tokenized_agent = self.token_processor(data)
           # tokenized_agent["dist_mask"]=data["agent"]["dist_mask"]
        else:
            tokenized_map, tokenized_agent = self.process_data(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.reward_w!=0 and self.use_target_q and self.global_step % self.critic_target_update_frequency == 0  :

            if self.soft_update:
                tau=2e-4 #self.critic_tau/(self.global_step+1)
                soft_update(self.encoder.agent_encoder,self.target_net.agent_encoder,tau)
            else:
                hard_update(self.encoder.agent_encoder,self.target_net.agent_encoder)

        return loss

def soft_update( net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

# def compute_reward_loss(self,reward):
#
#     div = 'rkl'
#     # TO DO: detach gradient, clip reward, gmm, refine by KL constrained
#
#     eps = 1e-1
#
#     if div == "kl":
#         alpha = 1  # *(self.global_step/10000+1e-2)
#         reward = torch.clamp(reward, max=alpha / eps, min=alpha * eps)
#         reward_loss = -alpha * ((reward / alpha).log() + 1)
#     elif div == "rkl":
#         alpha = 10
#         # reward=torch.clamp(reward,max=alpha*(-1-np.log(eps)),min=alpha*(-1+np.log(eps)))
#         reward_loss = alpha * (-reward / alpha - 1).exp()
#         # with torch.no_grad():
#         #     phi_grad = torch.exp(-reward)
#         # reward_loss = -(phi_grad * reward)
#     # reward_loss= reward_loss.detach()*reward
#     elif div == "sh":
#         alpha = 1
#         reward_loss = -1 / (1 / reward + 1 / alpha)
#     elif div == 'js':
#         alpha = 1
#         reward = torch.clamp_min(reward, min=alpha * (np.log(1 / 2 + eps)))  # ,max=alpha*(np.log(1/2+1/eps))
#         reward_loss = -alpha * (2 - (-reward / alpha).exp()).log()
#         # with torch.no_grad():
#         #     phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
#         # reward_loss = -(phi_grad * reward)
#     elif div == "tv":
#         reward_loss = -reward
#     elif div == 'x2':
#         alpha = 1
#         reward = torch.clamp(reward, min=2 * (1 - 1 / eps), max=2 * (1 - eps))
#
#         reward_loss = -reward + reward.square() / (4 * alpha)
#     else:
#         reward_loss = -reward
#
#     return reward_loss
