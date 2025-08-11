from lightning import LightningModule
import numpy as np
import torch
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.loss.iq_loss import get_iqloss,soft_update,eval_light,get_proposal_loss,get_gaussian_loss
from src.smart.loss.rollout_buffer import rollout, compute_advantages, ReplayBuffer
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import torch.nn.functional as F
import torch.nn as nn
import time
from collections import deque
import random
import copy
from src.smart.loss.rollout_buffer import RunningMeanStdTorch

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)
        self.gamma = 0.99
        self.iq_learn=self.encoder.iq_learn
        self.alpha = self.encoder.alpha
        self.n_token_agent=self.encoder.agent_encoder.n_token_agent

        self.use_target_q=False

        self.start_step=10//self.token_processor.shift-1

        self.use_gail=self.encoder.use_gail
        self.bce_loss = nn.BCELoss()


        self.buffer_len=1

        self.replay_buffer = deque(maxlen=self.buffer_len)

        self.rollout_freq=1

        # if self.use_gail:
        #     self.automatic_optimization = False
       # self.dis_freq=2
        #with torch.no_grad():
        if  self.use_target_q:
            self.target_net = copy.deepcopy(self.encoder.critic)
            self.target_net.load_state_dict(self.encoder.critic.state_dict())
            for param in self.target_net.parameters():
                param.requires_grad = False

        self.reward_type='airl'

        if self.use_gail:
            self.running_meanstd=RunningMeanStdTorch(shape=(1))

    def get_network_QV(self,q_value,tokenized_map, tokenized_agent,action,key):

        action = action.unsqueeze(-1)  # .reshape(-1)

        q = q_value#[:, :-1]

        current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

        current_V =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

        V=torch.cat([current_V,torch.zeros_like(current_V[:,:1])],dim=-1)

        current_V = V[:, :-1]
        next_V = V[:, 1:]

        pi = torch.softmax( q / self.alpha, dim=-1)

        logpi= torch.log(pi+ 1e-10)#.clamp_min(min=1e-10)

        log_prob=torch.gather(logpi, dim=-1, index=action).squeeze(-1)
        entropy = -torch.sum(pi * logpi, dim=-1)

        actor_loss = self.alpha * log_prob - current_Q

        dones = torch.zeros_like(next_V)
        dones[:, -1] = 1
        y=self.gamma *(1 - dones) * next_V
        reward = current_Q - y
        value_loss = current_V - y

        return log_prob,pi,actor_loss,entropy,current_Q,V,value_loss,reward

    def get_QV(self, tokenized_map, tokenized_agent,train_mask, key='expert'):
        valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]

        pred = self.encoder(tokenized_map, tokenized_agent)#,post_sampling=(key=='expert')

        if pred["proposal"] is not None:
            if key=="expert":
                proposal_loss, pos_dist, head_diff,min_idx=get_proposal_loss(pred["proposal"],tokenized_agent,self.start_step )

                all_valid_mask = valid_mask.all(-1)

                self.log("train/" + key + "_head_diff", head_diff[all_valid_mask].mean().item(), on_step=True, batch_size=1)
                self.log("train/" + key + "_pos_dist", pos_dist[all_valid_mask].mean().item(), on_step=True, batch_size=1)
                self.log("train/" + key + "_proposal_loss", proposal_loss.item(), on_step=True, batch_size=1)

                if 'pos' in tokenized_agent.keys():
                    sampled_pos=tokenized_agent['sampled_pos']
                    sampled_heading=tokenized_agent['sampled_heading']

                    gt_pos = tokenized_agent['pos']
                    heading = tokenized_agent['heading']

                    head_diff=wrap_angle(heading-sampled_heading).abs()
                    pos_dist=torch.linalg.norm(gt_pos-sampled_pos,dim=-1)

                    self.log("train/" + key + "_sample_head_diff", head_diff[all_valid_mask].mean().item(), on_step=True, batch_size=1)
                    self.log("train/" + key + "_sample_pos_dist", pos_dist[all_valid_mask].mean().item(), on_step=True, batch_size=1)
            else:
                proposal_loss=0
        else:
            proposal_loss=0

        action = tokenized_agent["sampled_idx"][:, self.start_step+1:]

        if pred["agent_q"] is None:
            return 0,0,0,0,0,proposal_loss

        # if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
        #     valid_mask=valid_mask[train_mask]
        #     action=action[train_mask]

        all_valid_mask=valid_mask.all(-1)

        log_prob,pi,actor_loss,entropy, current_Q, V,  value_loss, reward=self.get_network_QV(pred["agent_q"], tokenized_map, tokenized_agent,action,key)

        #current_Q_diff, V_diff = get_return_diff(reward,log_prob,current_Q,V,self.alpha,self.gamma)

        # if self.use_target_q and key=="expert":
        #     with torch.no_grad():
        #         pred = self.target_net(tokenized_map, tokenized_agent)
        #
        #         target_V = self.get_network_QV( pred["agent_q"], tokenized_map, tokenized_agent, action, key)[5]
        #     self.log("train/" + key + "_target_V", target_V.mean().item(), on_step=True, batch_size=1)
        # else:
        #     target_V=0

        init_V = V[:, 0]
        last_V= V[:,-1]

        if train_mask is not None:
            log_prob=log_prob[train_mask]

            reward=reward[train_mask]

            value_loss=value_loss[train_mask]

            V=V[all_valid_mask]

            current_Q=current_Q[all_valid_mask]

            entropy =entropy[all_valid_mask]

            init_V=init_V[all_valid_mask]

            last_V=last_V[all_valid_mask]

            off_ratio=(action==self.token_processor.agent_token_all_veh.shape[0])[train_mask].float().mean()
            self.log("train/"+key+"_off_ratio", off_ratio.item(), on_step=True, batch_size=1)

        action_nll = -log_prob.mean()

        self.log("train/"+key+"_V", V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_initV", init_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_nll", action_nll.item(), on_step=True, batch_size=1)

        if self.iq_learn and not self.use_gail:
            action_nll=0

        # if self.use_target_q:
        #     V=(V-target_V)[all_valid_mask]

        # if pred["visibility"] is None:
        #     vis_nll=0
        # else:
        #     visibility=pred["visibility"][:,:-1,0]
        #
        #     if key=="expert":
        #         vis_mask=valid_mask.to(torch.float)[:,1:]
        #     else:
        #         vis_mask=tokenized_agent["vis_mask"][:, self.start_step+1:][train_mask]
        #
        #     vis_log_prob = F.binary_cross_entropy_with_logits(visibility, vis_mask.float(), reduction='none')
        #
        #     if key=="expert": #state =True
        #         state_mask=valid_mask[:,:-1]
        #         vis_log_prob=vis_log_prob[state_mask]
        #     else:
        #         log_prob=log_prob+vis_log_prob
        #
        #     vis_nll=-vis_log_prob.mean()
        #     self.log("train/"+key+"_vis_nll", vis_nll.item(), on_step=True, batch_size=1)

        if len(pred["goal_q"]):
            goal_idx=tokenized_agent["goal_idx"][:, 2:]

            log_prob_goal,goal_logpi=self.get_network_QV(pred["goal_q"], tokenized_map, tokenized_agent,goal_idx,key)[:2]

            if key == "expert":
                goal_nll=-log_prob_goal[train_mask].mean()
                self.log("train/" + key + "_goal_nll", goal_nll.item(), on_step=True, batch_size=1)
            else:
                log_prob=log_prob_goal+log_prob
                goal_nll=0

        else:
            goal_nll=0

        if len(pred["light_q"]) and key=="expert":
            light_idx=tokenized_agent["light_idx"][:, 2:]
            light_action=torch.clamp_max(light_idx,max=self.token_processor.light_type-1)

            log_prob_light,light_logpi=self.get_network_QV(pred["light_q"], tokenized_map, tokenized_agent,light_action,key)[:2]

            light_mask=light_idx<self.token_processor.light_type
            light_nll=-log_prob_light[light_mask].mean()
            light_acc = (torch.argmax(light_logpi, dim=-1) == light_idx)#[train_mask[agent_num:]]
            self.log("train/" + key + "_light_acc", light_acc.float().mean().item(), on_step=True, batch_size=1)
            self.log("train/" + key + "_light_nll", light_nll.item(), on_step=True, batch_size=1)
        else:
            light_nll=0

        return  reward,value_loss,pi,action_nll+light_nll+goal_nll,current_Q,proposal_loss,log_prob,entropy

    def get_reward(self,tokenized_agent,log_prob,key,train_mask=None,agent_mask=None):

        # all_features=tokenized_agent["detach_all_features"]
        map_feature=tokenized_agent["detach_map_feature"]

        # logit = self.encoder.discriminator(all_features,map_feature,agent_mask)[0][:, :, 0]
        #pos=tokenized_agent["sampled_pos"]#.clone()
        #heading=tokenized_agent["sampled_heading"]#.clone()

        # if key=="expert":
        #
        #     pos=pos+torch.randn_like(pos)*0.01
        #     heading=heading+torch.randn_like(heading)*0.01

        logit= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        tokenized_agent["goal_idx"],
                                                        tokenized_agent["valid_mask"],
                                                        tokenized_agent["sampled_pos"],
                                                        tokenized_agent["sampled_heading"] ,
                                                        tokenized_agent,
                                                        map_feature,
                                                        tokenized_agent["light_idx"],
                                                        None)[0][:, :, 0]

        # disc_val = torch.sigmoid(logit)

        exp_f=logit.exp()

        disc_val=exp_f/(exp_f + torch.exp(log_prob))


        returns, rewards = self.running_meanstd.get_return(disc_val, self.gamma,key,reward_type=self.reward_type)

        if train_mask is not None:
            disc_val=disc_val[train_mask]
            returns=returns[train_mask]
            rewards=rewards[train_mask]

        if key == "expert":
            bce_loss = self.bce_loss(disc_val, torch.ones_like(disc_val)) #+ (disc_val-0.5).square().mean()
        else:
            bce_loss = self.bce_loss(disc_val, torch.zeros_like(disc_val)) #+ (disc_val-0.5).square().mean()

        self.log("train/"+key+"_dis_loss", bce_loss, on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_return", returns.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_rewards", rewards.mean().item(), on_step=True, batch_size=1)

        return bce_loss,rewards,returns,logit

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]
        state_mask = valid_mask[:, :-1]
        action_mask = valid_mask[:, 1:]
        train_mask = state_mask & action_mask##
        tokenized_agent["vis_mask"] = None
        all_valid=valid_mask.all(-1)

        if self.iq_learn:
            self.encoder.agent_encoder.a_t_roformer.attn.caching = True
            if self.encoder.agent_encoder.pred_light and not self.encoder.agent_encoder.light_encoder.share:
                self.encoder.agent_encoder.light_encoder.lg_t_roformer.attn.caching = True

        expert_reward,expert_value_loss,expert_pi,expert_nll,expert_Q,expert_proposal_loss,expert_log_prob,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        if self.iq_learn:
            # self.encoder.agent_encoder.pred_light=False

            #train_mask = valid_mask.all(-1)

            #tokenized_agent["train_mask"]= train_mask

            if self.use_gail:
                expert_dis_loss, expert_rewards, expert_returns,expert_logit=self.get_reward(tokenized_agent,expert_log_prob,"expert",all_valid,None)

            expert_light_idx=tokenized_agent["light_idx"].clone()

            # torch.cuda.synchronize()
            # tokenized_agent.update(rollout_result)
            # tokenized_agent_rollout=tokenized_agent

            if self.global_step%self.rollout_freq==0:
                tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent)

                if self.rollout_freq>1:
                    self.tokenized_map={}
                    for key in tokenized_map.keys():
                        if key!="map_feature":
                            self.tokenized_map[key]=tokenized_map[key]

                    self.tokenized_agent_rollout={}

                    for key in ["sampled_idx","sampled_pos", "sampled_heading", "valid_mask","batch", "type", "shape","sampled_log_prob","light_idx","num_graphs","train_mask","detach_all_features"]:
                        self.tokenized_agent_rollout[key] = tokenized_agent_rollout[key]
            else:
                tokenized_map=self.tokenized_map
                tokenized_agent_rollout=self.tokenized_agent_rollout

                # tokenized_agent_rollout["train_mask"]=None

            val_train_mask=None

            if self.encoder.agent_encoder.pred_light:
                eval_light(expert_light_idx, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            agent_reward, agent_value_loss, agent_pi, agent_nll,agent_Q,agent_proposal_loss,agent_log_prob,agent_entropy = self.get_QV(
                tokenized_map, tokenized_agent_rollout, all_valid,key='agent')

            if self.use_gail:
                agent_dis_loss, agent_rewards, agent_returns, agent_logit = self.get_reward(tokenized_agent_rollout, agent_log_prob, "agent",all_valid)

                # if self.buffer_len>1:
                #     with torch.no_grad():
                #         agent_dis_loss, agent_rewards, agent_returns, agent_logit = self.get_reward(
                #             tokenized_agent_rollout["detach_all_features"], "agent",
                #             tokenized_agent_rollout["train_mask"])
                #
                #     all_feats=[]
                #     for feat in tokenized_agent_rollout["detach_all_features"]:
                #         all_feats.append(feat.clone())
                #     self.replay_buffer.append((all_feats,tokenized_agent_rollout["train_mask"]))
                #
                #     detach_all_features,agent_train_mask=random.sample(self.replay_buffer,1)[0]
                #     logit = self.encoder.discriminator(detach_all_features, agent_train_mask)[0][:, :, 0]
                #     agent_dis_loss = self.bce_loss( torch.sigmoid(logit), torch.zeros_like(logit))
                #
                # else:

                if self.automatic_optimization == False:
                    policy_optimizer, discriminator_optimizer = self.optimizers ()

                if self.reward_type=="raw":
                    alpha=0.5
                    critic_loss =-expert_logit.mean()+agent_logit.mean()+expert_logit.square().mean() / (4 * alpha)
                else:
                    critic_loss=expert_dis_loss + agent_dis_loss

                if self.automatic_optimization==False:
                    discriminator_optimizer.zero_grad()
                    self.manual_backward(critic_loss)
                    torch.nn.utils.clip_grad_norm_(self.encoder.discriminator.parameters(), max_norm=0.5)
                    discriminator_optimizer.step()

                if self.encoder.use_value:
                    value_pred = self.encoder.value_network.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                                     tokenized_agent_rollout["goal_idx"],
                                                                     tokenized_agent_rollout["valid_mask"],
                                                                     tokenized_agent_rollout["sampled_pos"],
                                                                     tokenized_agent_rollout["sampled_heading"],
                                                                     tokenized_agent_rollout,
                                                                     tokenized_agent_rollout["detach_map_feature"],
                                                                     tokenized_agent_rollout["light_idx"],
                                                                     None)[0][:, :, 0]

                    advantages,returns=compute_advantages(agent_rewards,value_pred.detach(),None,gamma=self.gamma)

                    value_loss = 0.5 * (returns - value_pred).pow(2).mean()
                    self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

                elif self.encoder.use_critic:
                    action=tokenized_agent_rollout["sampled_idx"][:,2:].unsqueeze(-1)

                    with torch.no_grad():
                        target_q=self.target_net.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                                     tokenized_agent_rollout["goal_idx"],
                                                                     tokenized_agent_rollout["valid_mask"],
                                                                     tokenized_agent_rollout["sampled_pos"],
                                                                     tokenized_agent_rollout["sampled_heading"],
                                                                     tokenized_agent_rollout,
                                                                     tokenized_agent_rollout["detach_map_feature"],
                                                                     tokenized_agent_rollout["light_idx"],
                                                                     None)[0]

                        next_Q = torch.sum(agent_pi[:, 1:] * target_q[:, 1:], dim=-1)

                        next_Q = torch.cat([next_Q, torch.zeros_like(next_Q[:, :1])], dim=1)

                        target_Q = agent_rewards + self.gamma * next_Q

                    q = self.encoder.critic.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                             tokenized_agent_rollout["goal_idx"],
                                                             tokenized_agent_rollout["valid_mask"],
                                                             tokenized_agent_rollout["sampled_pos"],
                                                             tokenized_agent_rollout["sampled_heading"],
                                                             tokenized_agent_rollout,
                                                             tokenized_agent_rollout["detach_map_feature"],
                                                             tokenized_agent_rollout["light_idx"],
                                                             None)[0]

                    current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

                    value_loss= 0.5 * (current_Q - target_Q).pow(2).mean()

                    current_value = torch.sum(agent_pi * q, dim=-1)

                    advantages=(current_Q-current_value).detach()
                    self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

                else:
                    #advantages=agent_returns
                    self.running_meanstd.update(agent_returns.reshape(-1))

                    advantages=self.running_meanstd.normalize(agent_returns.reshape(-1)).reshape(agent_returns.shape)
                    #
                    self.log("train/running_mean", self.running_meanstd.mean, on_step=True, batch_size=1)
                    self.log("train/running_var", self.running_meanstd.var, on_step=True, batch_size=1)
                    #
                    # advantages= (agent_returns - agent_returns.mean()) / (agent_returns.std() + 1e-5)#F.normalize(agent_returns,dim=0)#
                    #
                    # advantages=torch.clamp_(advantages, min=-1.0, max=1.0)
                    value_loss=0


                if self.rollout_freq>1:
                    prev_log_prob=tokenized_agent_rollout["sampled_log_prob"][tokenized_agent_rollout["train_mask"]]
                    ratio = torch.exp(agent_log_prob - prev_log_prob)
                    clip_param = 0.2

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio,
                                        1.0 - clip_param,
                                        1.0 + clip_param) * advantages
                    agent_wNLL = -torch.min(surr1, surr2).mean()
                else:
                    agent_wNLL=-(agent_log_prob*advantages).mean()

                self.log("train/agent_wNLL", agent_wNLL.item(), on_step=True, batch_size=1)
                self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)

                gail_weight=1 #min(self.global_step/1000,1)

                agent_density=torch.cumsum(agent_log_prob,dim=1).mean() #agent_log_prob.mean() #

                self.log("train/agent_density", agent_density.item(), on_step=True, batch_size=1)

                expert_nll = expert_nll + gail_weight*agent_wNLL + value_loss #-0.1*agent_density.mean()  # - 0.01 * agent_entropy.mean()

            else:
                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)
                # constraint_loss=expert_V_diff.square().mean()*5
                #
                # self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_proposal_loss+expert_nll

            if self.automatic_optimization == False:
                policy_optimizer.zero_grad()
                loss=expert_proposal_loss+expert_nll
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters())
                                               +list(self.encoder.value_network.parameters()), 0.5)
                policy_optimizer.step()

        else:
            loss = expert_nll + expert_proposal_loss

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q :
            soft_update(self.encoder.critic, self.target_net, tau = 2e-4)

        return loss

