import copy

from keras.integration_test.models.bert import loss_fn
from lightning import LightningModule
import random
from collections import deque
import torch.nn as nn
import torch
import numpy as np
from tensorflow_probability.substrates.jax.distributions.student_t import log_prob
from tensorflow_probability.substrates.numpy.distributions.student_t import entropy

from src.smart.modules.smart_decoder import SMARTDecoder
import pickle
from torch_scatter import scatter_mean,scatter_max,scatter_sum
from torch.nn.utils.rnn import pad_sequence
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding,RoFormerBlock,general_rope
from typing import Optional

import torch
from torch import Tensor, tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torchmetrics.metric import Metric

from src.smart.metrics.utils import get_euclidean_targets
from ..modules.gmm_dist import  GMM_Dist,get_entropy
from .iq_loss import get_iqloss,soft_update

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.gamma = 0.99
        self.alpha = self.encoder.alpha

        self.batch_replay=False

        if self.batch_replay:
            self.replay_buffer = deque(maxlen=4000)
        else:
            self.replay_buffer = deque(maxlen=1)

        self.use_target_q=False
        self.rollout_freq=1
        
        if self.encoder.iq_learn and self.encoder.output_gmm:
            self.automatic_optimization = False

        if  self.use_target_q:
            self.target_net = SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
            )
            self.target_net.load_state_dict(self.encoder.state_dict())

    def rollout(self, tokenized_map, tokenized_agent):
        self.encoder.eval()
        with torch.no_grad():
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.validation_rollout_sampling,
            )
        self.encoder.train()

        tokenized_agent_rollout = {}
        tokenized_agent_rollout['num_graphs'] = tokenized_agent['num_graphs']

        if "sampled_idx" in pred.keys():
            for key in ["sampled_pos", "sampled_heading", "valid_mask","batch", "type", "shape"]:
                tokenized_agent_rollout[key] = pred[key]

            tokenized_agent_rollout['sampled_idx'] = pred['sampled_idx'].to(torch.int16)

        if "light_idx" in pred.keys():
            tokenized_agent_rollout['light_idx'] = pred['light_idx']
            for key in ["lengths_lg", "sinusoidal_lg", "batch_lg"]:
                tokenized_agent_rollout[key] = tokenized_agent[key]

        if self.rollout_freq > 1:
            tokenized_map_rollout = {}

            for key in tokenized_map.keys():
                if key !="map_feature":
                    tokenized_map_rollout[key]=tokenized_map[key]

            self.replay_buffer.append((tokenized_map_rollout, tokenized_agent_rollout))

        return tokenized_map,tokenized_agent_rollout

    def get_network_QV(self,network,tokenized_map, tokenized_agent,action,key,action_mask):

        pred = network(tokenized_map, tokenized_agent)

        q_value = pred["q_value"]

        if self.encoder.output_gmm:
            dist =  GMM_Dist(q_value)

            entropy=get_entropy(q_value)

            if "gt_pos_raw" in tokenized_agent.keys():
                gt_pos=tokenized_agent["gt_pos_raw"]
                gt_head=tokenized_agent["gt_head_raw"]
                gt_valid=tokenized_agent["gt_valid_raw"]
            else:
                gt_pos=tokenized_agent["sampled_pos"]
                gt_head=tokenized_agent["sampled_heading"]
                gt_valid=tokenized_agent["valid_mask"]

            target, target_valid = get_euclidean_targets(
                pred_pos=tokenized_agent["sampled_pos"],
                pred_head=tokenized_agent["sampled_heading"],
                pred_valid=tokenized_agent["valid_mask"],
                gt_pos=gt_pos,
                gt_head=gt_head,
                gt_valid=gt_valid
            )
            if self.encoder.agent_encoder.output_dim == 4:
                target = torch.cat(
                    [target[..., :2], target[..., [-1]].cos(), target[..., [-1]].sin()], dim=-1
                )  # [n_batch, n_step, 4]

            target1=torch.cat([target, torch.zeros_like(target[:,:1])],dim=1)

            log_prob = dist.log_prob(target1)[:,:-1]#.clamp_min(min=np.log(1e-3))

            if self.encoder.iq_learn:

                sampled_action=dist.sample([2])

                sampled_log_prob=dist.log_prob(sampled_action)

                pred=network(tokenized_map, tokenized_agent,True)

                Q=network.get_Q(pred["feat_a"],sampled_action[0])

                v_value=Q-self.alpha*sampled_log_prob[0].detach()

                actor_loss=self.alpha * sampled_log_prob[0][:,:-1] - Q[:,:-1].detach()

                current_V= v_value[:,:-1]

                next_V = network.get_Q(pred["feat_a"][:,1:],sampled_action[1][:,1:])-self.alpha*sampled_log_prob[1][:,1:].detach()

                current_Q = network.get_Q(pred["feat_a"][:, :-1], target)
            else:
                next_V = torch.zeros_like(log_prob)
                current_V = torch.zeros_like(log_prob)
                current_Q = torch.zeros_like(log_prob)
                actor_loss = torch.zeros_like(log_prob)
                v_value = torch.zeros_like(torch.cat([log_prob, torch.zeros_like(log_prob[:,:1])],dim=1))
        else:

            q = q_value[:, :-1]

            current_Q = q.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

            v_value =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

            current_V = v_value[:, :-1]

            next_V = v_value[:, 1:]

            pi = torch.softmax( q / self.alpha, dim=-1)

            logpi= torch.log(pi + 1e-10)

            log_prob=logpi.reshape(len(action), -1)[torch.arange(len(action)), action].reshape(q.shape[0], q.shape[1])

            entropy = -torch.sum(pi * logpi, dim=-1)

            actor_loss=self.alpha * log_prob - current_Q.detach()

        dones = torch.zeros_like(next_V)

        dones[:, -1] = 1

        y=self.gamma *(1 - dones) * next_V

        reward = current_Q - y
        value_loss = current_V - y

        if self.encoder.agent_encoder.mixing:
            total_q=result["total_q"]
            total_v=result["total_v"]

            dones = torch.zeros_like(total_v[:, 1:])

            dones[:, -1] = 1

            y = self.gamma * (1 - dones) *  total_v[:, 1:]
            total_reward=total_q-y

            total_value_loss= total_v[:, :-1]-y
        else:
            total_reward=0
            total_value_loss=0

        return actor_loss,log_prob,entropy,current_Q,v_value,value_loss,reward,dones,total_reward,total_value_loss,pred

    def get_return(self,reward,log_prob,current_Q,V,all_valid_mask,key):
        rewards=reward - self.alpha * log_prob
        returns = torch.zeros_like(V)
        running_return=returns[:,-1]

        for i in range(rewards.size(1)-1,-1,-1):
            running_return = rewards[:, i] + self.gamma *running_return
            returns[:, i] = running_return

        current_Q_diff = (current_Q - returns[:,:-1])[all_valid_mask]
        V_diff=(V[:,:-1]-returns[:,:-1])[all_valid_mask]

        self.log("train/"+key+"_Q_diff", current_Q_diff.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_V_diff", V_diff.mean().item(), on_step=True, batch_size=1)

    def get_QV(self, tokenized_map, tokenized_agent,train_mask, key='expert'):

        if self.encoder.agent_encoder.pred_agent:
            valid_mask = tokenized_agent["valid_mask"][:, 1:]
            action = tokenized_agent["sampled_idx"][:, 2:]
            state_mask = valid_mask[:, :-1]

        if self.encoder.agent_encoder.pred_light:
            light_valid_mask = tokenized_agent["light_idx"] < self.encoder.agent_encoder.light_type

            light_action=torch.clamp_max(tokenized_agent["light_idx"][:, 2:],max=2)

            if self.encoder.agent_encoder.pred_agent:
                agent_num=len(valid_mask)

                action = torch.cat([action,light_action])#tokenized_agent["light_idx"][:, 2:] #

                valid_mask =  torch.cat([valid_mask, light_valid_mask[:,1:]])#light_valid_mask[:,1:] #
                state_mask= torch.cat([state_mask,light_valid_mask[:,1:-1]])#light_valid_mask[:,1:-1]#

                all_valid_mask=torch.cat([tokenized_agent["valid_mask"].all(-1) ,light_valid_mask.all(-1)])
            else:
                action = light_action
                all_valid_mask=light_valid_mask.all(-1)

                valid_mask = light_valid_mask[:, 1:]
                state_mask = light_valid_mask[:, 1:-1]
                agent_num=0

        action = action.reshape(-1).long()
        action_mask= valid_mask[:, 1:]
        all_valid_mask=valid_mask.all(-1)

        actor_loss,log_prob,entropy, current_Q, V,  value_loss, reward,dones,total_reward,total_value_loss,pred=self.get_network_QV(self.encoder, tokenized_map, tokenized_agent,action,key,action_mask)

        if self.encoder.agent_encoder.pred_light and key=="expert":
            light_pred=torch.argmax(logpi[agent_num:] , dim=-1)
            real_light=tokenized_agent["light_idx"][:, 2:]

            light_acc=(light_pred==real_light)[state_action_mask[agent_num:]]

            self.log("train/"+key+"_light_acc", light_acc.float().mean().item(), on_step=True, batch_size=1)

        action_nll = -log_prob[train_mask].mean()

        if self.use_target_q:
            with torch.no_grad():
                target_q, target_current_Q, target_V,target_current_V,target_next_V, target_reward,_ = self.get_network_QV(self.target_net, tokenized_map, tokenized_agent,action,key,action_mask)

            reward = current_Q - self.gamma * target_next_V
            value_loss=current_V-self.gamma *target_next_V

        init_V = V[:, 0]
        last_V=V[:,-1]

        self.get_return(reward,log_prob,current_Q,V,all_valid_mask,key)

        actor_loss = actor_loss[train_mask]

        reward = reward[train_mask]

        value_loss=value_loss[train_mask]

        V=V[all_valid_mask]

        current_Q=current_Q[all_valid_mask]

        entropy =entropy[all_valid_mask]

        init_V=init_V[valid_mask[:,0]]

        last_V=last_V[valid_mask[:,-1]]


        self.log("train/"+key+"_V", V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_initV", init_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_actor_loss", actor_loss.mean().item(), on_step=True, batch_size=1)

        if self.encoder.agent_encoder.mixing:
            reward=total_reward
            value_loss=total_value_loss

        return  reward,value_loss,init_V-1,action_nll,actor_loss

    def iq_update(self, tokenized_map, tokenized_agent):
        train_mask= tokenized_agent["valid_mask"][:, 1:].all(-1)

        expert_reward,expert_value_loss,expert_V_diff,expert_nll,expert_actor_loss= self.get_QV(tokenized_map, tokenized_agent,train_mask)

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        if self.encoder.agent_encoder.pred_light:
            real_light=tokenized_agent["light_idx"][:, 2:]

            batch_lg=tokenized_agent["batch_lg"]

            batch_mask=batch_lg[:,None]==batch_lg[None]

            real_light_mask=(real_light<self.encoder.agent_encoder.light_type).all(-1)#[:,None].repeat(1,real_light.shape[1])#torch.ones_like(tokenized_agent["light_mask"][:, 2:]).to(bool)

            repeat_pred=tokenized_agent["light_idx"][:, 1:2].repeat(1,real_light.shape[1])

            repeat_light_acc=(repeat_pred==real_light)[real_light_mask].float().mean()

            self.log("train/repeat_light_acc", repeat_light_acc.item(), on_step=True, batch_size=1)

            real_relation=real_light[:,None]==real_light[None][batch_mask]

            real_relation_mask=(real_light_mask[:,None] & real_light_mask[None]) [batch_mask]

            repeat_relation=repeat_pred[:,None]==repeat_pred[None][batch_mask]

            repeat_relation_acc=(real_relation==repeat_relation)[real_relation_mask].float().mean()

            self.log("train/repeat_relation_acc", repeat_relation_acc.item(), on_step=True, batch_size=1)

        if not self.encoder.iq_learn:
            loss =expert_nll
        else:
            if self.global_step % self.rollout_freq== 0:
                tokenized_map_rollout, tokenized_agent_rollout = self.rollout(tokenized_map, tokenized_agent)
            else:
                tokenized_map_rollout, tokenized_agent_rollout =random.sample(self.replay_buffer,1)[0]

            if self.encoder.agent_encoder.pred_light:

                light_rollout = tokenized_agent_rollout["light_idx"][:, 2:]

                light_acc = (light_rollout == real_light)[real_light_mask].float().mean()

                self.log("train/agent_light_acc", (light_acc-repeat_light_acc).item(), on_step=True, batch_size=1)

                agent_relation = light_rollout[:,None]==light_rollout[None][batch_mask]

                agent_relation_acc = (real_relation == agent_relation)[real_relation_mask].float().mean()

                self.log("train/agent_relation_acc",(agent_relation_acc-repeat_relation_acc).item(), on_step=True, batch_size=1)

            agent_reward, agent_value_loss, agent_V_diff, _,agent_actor_loss = self.get_QV(
                tokenized_map_rollout, tokenized_agent_rollout, train_mask,key='agent')

            critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            constraint_loss=expert_V_diff.square().mean()#

            self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss#+constraint_loss#critic_loss+constraint_loss #expert_nll #-0.01*agent_entropy.mean() #expert_nll+expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1

            if self.automatic_optimization==False:
                actor_optimizer,critic_optimizer=self.optimizers()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss=expert_actor_loss.mean()/2+agent_actor_loss.mean()/2

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        return loss

    def process_data(self,data):

        tokenized_agent={}
        tokenized_map={}
        tokenized_agent['num_graphs'] = data.num_graphs

        if self.encoder.agent_encoder.pred_agent:
            map=data["tokenized_map"]
            agent=data["tokenized_agent"]

            for key in ["position", "orientation", "batch","token_idx", "type", "pl_type","light_type"]:
                tokenized_map[key] = map[key]

            agent_shape, token_traj_all, token_traj = self.token_processor._get_agent_shape_and_token_traj(
                agent['type']
            )
            tokenized_agent['token_traj_all'] = token_traj_all
            tokenized_agent["token_agent_shape"]=agent_shape  # [n_token, 2]

            if "col_mask" in agent.keys():
                tokenized_agent["col_mask"] = agent["col_mask"]

            if "gt_pos_raw" in agent.keys():
                for key in ["gt_pos_raw", "gt_head_raw", "gt_valid_raw"]:
                    tokenized_agent[key] = agent[key]
                #if self.encoder.agent_encoder.use_GT:
                pred=self.token_processor.my_match_agent_token(agent["gt_valid_raw"],agent["gt_pos_raw"],
                                                                agent["gt_head_raw"],
                                                                agent_shape,token_traj,not self.encoder.iq_learn
                                                                )

                tokenized_agent["valid_mask"]=pred["valid_mask"]
                tokenized_agent["sampled_idx"]=pred["sampled_idx"]
                tokenized_agent["sampled_pos"]=pred["sampled_pos"]
                tokenized_agent["sampled_heading"]=pred["sampled_heading"]
                tokenized_agent["gt_pos_raw"]=agent["gt_pos_raw"][:,5::5]
                tokenized_agent["gt_head_raw"]=agent["gt_head_raw"][:,5::5]
                tokenized_agent["gt_valid_raw"]=agent["gt_valid_raw"][:,5::5]

                for key in [ "type", "batch", "shape"]:
                    tokenized_agent[key] = agent[key]
            else:
                for key in ["sampled_pos", "sampled_heading", "type", "batch", "shape", "sampled_idx", "valid_mask"]:
                    tokenized_agent[key] = agent[key]

        if self.encoder.agent_encoder.pred_light:

            tokenized_light=data["tokenized_light"]

            light_idx=tokenized_light["light_idx"]

            #light_mask=light_idx<self.encoder.agent_encoder.light_type

            # light_pred_mask=light_mask.all(-1)#torch.ones_like(light_idx[:,0]).to(torch.bool)
            #light_idx[light_idx>2]=0

            #light_pred_mask=torch.ones_like(light_pred_mask)
            #pos_lg, orient_lg=self.rotate(pos_lg, orient_lg, batch_lg)

            tokenized_agent["light_idx"]=light_idx.long()#[light_pred_mask]
            pos_lg=tokenized_light["pos_lg"]#[light_pred_mask]
            orient_lg=tokenized_light["orient_lg"]#[light_pred_mask]
            batch_lg=tokenized_light["batch"]#[light_pred_mask]

            lengths_lg = torch.bincount(batch_lg, minlength=data.num_graphs).tolist()

            sinusoidal_lg = general_rope(pos_lg, self.encoder.agent_encoder.head_dim, orient_lg)    
            sinusoidal_lg=self.encoder.agent_encoder.padding(sinusoidal_lg, lengths_lg)
            tokenized_agent["lengths_lg"] = lengths_lg
            tokenized_agent["batch_lg"]=batch_lg
            tokenized_agent["sinusoidal_lg"] = sinusoidal_lg

        if self.encoder.agent_encoder.pred_route:
            route_idx= agent["route_idx"]//(120//self.encoder.agent_encoder.route_type)
            route_idx[:, :2] = -1

            route_idx[route_idx==-1]=self.encoder.agent_encoder.route_type

            tokenized_agent["route_idx"] = route_idx.long()
            tokenized_agent["route_valid_mask"]=route_idx!=self.encoder.agent_encoder.route_type

        return tokenized_map, tokenized_agent

    def training_step(self, data, batch_idx):

        if self.encoder.tokenizer_training:

            commit_loss,rec_loss,dist,smapled_dist=self.encoder.vq_vae(data)

            loss=commit_loss+rec_loss

            self.log("train/loss", loss, on_step=True, batch_size=1)
            self.log("train/commit_loss", commit_loss, on_step=True, batch_size=1)
            self.log("train/rec_loss", rec_loss, on_step=True, batch_size=1)
            self.log("train/dist", dist, on_step=True, batch_size=1)
            self.log("train/smapled_dist", smapled_dist, on_step=True, batch_size=1)

            return loss

        if "traj_pos" in data.keys():
            tokenized_map, tokenized_agent = self.token_processor(data)
        else:
            tokenized_map, tokenized_agent = self.process_data(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q :
            soft_update(self.encoder.agent_encoder, self.target_net.agent_encoder, tau = 1e-4)

        return loss
