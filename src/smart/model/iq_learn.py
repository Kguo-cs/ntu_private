from lightning import LightningModule
import numpy as np
import torch

from TrafficManager.LimSim.utils.data_copy import deepcopy
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.metrics.utils import get_euclidean_targets
from src.smart.loss.gmm_dist import  GMM_Dist,get_entropy
from src.smart.loss.iq_loss import get_iqloss,soft_update,eval_light,get_proposal_loss,get_gaussian_loss
from src.smart.loss.rollout_buffer import rollout,get_return_diff,get_return,compute_advantages
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import torch.nn.functional as F
import torch.nn as nn

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

        # if self.use_gail:
        #     self.automatic_optimization = False

           # self.dis_freq=2

        if  self.use_target_q:
            self.target_net = SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent,
                token_processor=self.token_processor
            )
            self.target_net.load_state_dict(self.encoder.state_dict())


    def get_network_QV(self,q_value,tokenized_map, tokenized_agent,action,key):

        action = action.unsqueeze(-1)  # .reshape(-1)

        q = q_value[:, :-1]

        current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

        V =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

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

        return log_prob,logpi,actor_loss,entropy,current_Q,V,value_loss,reward

    def get_QV(self, tokenized_map, tokenized_agent,train_mask, key='expert'):

        pred = self.encoder(tokenized_map, tokenized_agent)#,post_sampling=(key=='expert')

        if pred["proposal"] is not None:
            if key=="expert":
                if self.encoder.agent_encoder.pred_gaussian:
                    proposal_loss,pos_dist, head_diff=get_gaussian_loss(pred["proposal"],tokenized_agent )
                    action = tokenized_agent["sampled_idx"][:, 2:]
                else:
                    proposal_loss, pos_dist, head_diff,min_idx=get_proposal_loss(pred["proposal"],tokenized_agent,self.start_step )

                    self.log("train/" + key + "_head_diff", head_diff.item(), on_step=True, batch_size=1)

                self.log("train/" + key + "_pos_dist", pos_dist.item(), on_step=True, batch_size=1)
                self.log("train/" + key + "_proposal_loss", proposal_loss.item(), on_step=True, batch_size=1)
            else:
                proposal_loss=0
        else:
            proposal_loss=0

        action = tokenized_agent["sampled_idx"][:, self.start_step+1:]

        if pred["agent_q"] is None:
            return 0,0,0,0,0,proposal_loss

        valid_mask = tokenized_agent["valid_mask"][:, self.start_step:]

        if "train_mask" in tokenized_agent.keys():
            valid_mask=valid_mask[train_mask]
            action=action[train_mask]
            train_mask=train_mask[train_mask]

        all_valid_mask=valid_mask.all(-1)

        log_prob,logpi,actor_loss,entropy, current_Q, V,  value_loss, reward=self.get_network_QV(pred["agent_q"], tokenized_map, tokenized_agent,action,key)

        #current_Q_diff, V_diff = get_return_diff(reward,log_prob,current_Q,V,self.alpha,self.gamma)

        if key == "expert":
            log_prob=log_prob[train_mask]

        action_nll = -log_prob.mean()

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

        if self.use_target_q and key=="expert":
            with torch.no_grad():
                pred = self.target_net(tokenized_map, tokenized_agent)

                target_V = self.get_network_QV( pred["agent_q"], tokenized_map, tokenized_agent, action, key)[5]
            self.log("train/" + key + "_target_V", target_V.mean().item(), on_step=True, batch_size=1)
        else:
            target_V=0

        init_V = V[:, 0]
        last_V= V[:,-1]

        actor_loss = actor_loss[all_valid_mask]

        reward = reward[train_mask]

        value_loss=value_loss[train_mask]

        V=V[all_valid_mask]

        current_Q=current_Q[all_valid_mask]

        entropy =entropy[all_valid_mask]

        init_V=init_V[all_valid_mask]

        last_V=last_V[all_valid_mask]

        #current_Q_diff=current_Q_diff[all_valid_mask]
        #V_diff=V_diff[all_valid_mask]

        self.log("train/"+key+"_V", V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_initV", init_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_actor_loss", actor_loss.mean().item(), on_step=True, batch_size=1)
        #self.log("train/"+key+"_Q_diff", current_Q_diff.mean().item(), on_step=True, batch_size=1)
       # self.log("train/"+key+"_V_diff", V_diff.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_nll", action_nll.item(), on_step=True, batch_size=1)

        # off_ratio=(action==self.token_processor.agent_token_all_veh.shape[0])[train_mask].float().mean()
        # self.log("train/"+key+"_off_ratio", off_ratio.item(), on_step=True, batch_size=1)
        
        if self.iq_learn and not self.use_gail:
            action_nll=0

        if self.use_target_q:
            V=(V-target_V)[all_valid_mask]

        return  reward,value_loss,V,action_nll+light_nll,current_Q,proposal_loss,log_prob,entropy

    def get_reward(self,all_features,key,train_mask=None):
        score = self.encoder.discriminator(all_features,train_mask)[0][:, 1:, 0]

        disc_val = torch.sigmoid(score)

        returns, rewards = get_return(disc_val, self.gamma)

        if key == "expert":
            target=torch.ones_like(disc_val)
        else:
            target=torch.zeros_like(disc_val)

        bce_loss = self.bce_loss(disc_val, target)

        self.log("train/"+key+"_dis_loss", bce_loss, on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_return", returns.mean().item(), on_step=True, batch_size=1)

        # rewards=score

        self.log("train/"+key+"_rewards", rewards.mean().item(), on_step=True, batch_size=1)

        return bce_loss,rewards,returns


    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]

        # if self.iq_learn:
        #     train_mask = valid_mask.all(-1)
        #     tokenized_agent["train_mask"]=train_mask
        # else:
        state_mask = valid_mask[:, :-1]
        action_mask = valid_mask[:, 1:]
        train_mask = state_mask & action_mask

        expert_reward,expert_value_loss,expert_V_diff,expert_nll,expert_Q,expert_proposal_loss,_,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        if self.iq_learn:
            self.encoder.agent_encoder.pred_light=False

            train_mask = valid_mask.all(-1)

            tokenized_agent["train_mask"]= valid_mask.all(-1)

            if self.use_gail:
                expert_dis_loss, expert_rewards, expert_returns=self.get_reward(tokenized_agent["all_features"],"expert",train_mask)

            tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent)

            if self.encoder.agent_encoder.pred_light:
                eval_light(tokenized_agent, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            if self.use_gail:
                agent_reward, agent_value_loss, agent_V_diff, agent_nll,agent_Q,agent_proposal_loss,agent_log_prob,agent_entropy = self.get_QV(
                    tokenized_map, tokenized_agent_rollout, train_mask,key='agent')

                agent_dis_loss,agent_rewards,agent_returns=self.get_reward(tokenized_agent_rollout["all_features"],"agent",train_mask)

                if self.automatic_optimization == False:
                    policy_optimizer, discriminator_optimizer = self.optimizers ()

                #alpha=10
                # critic_loss =-expert_rewards.mean()+expert_reward.square().mean() / (4 * alpha)+agent_rewards.mean()
                critic_loss=expert_dis_loss + agent_dis_loss
                self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

                if self.automatic_optimization==False:
                    discriminator_optimizer.zero_grad()
                    self.manual_backward(critic_loss)
                    torch.nn.utils.clip_grad_norm_(self.encoder.discriminator.parameters(), max_norm=0.5)
                    discriminator_optimizer.step()

                if self.encoder.use_value:

                    value_pred=self.encoder.value_network(tokenized_agent_rollout["all_features"],train_mask)[0][:,:-1,0]
                    # value_pred = self.encoder.value_network(tokenized_agent_rollout, tokenized_map["detach_map_feature"])["agent_q"][:,:-1,0]

                    advantages,returns=compute_advantages(agent_rewards,value_pred,None,gamma=self.gamma)

                    value_loss = 0.5 * (returns - value_pred).pow(2).mean()

                    self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)
                    self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)
                else:

                    #advantages,returns=compute_advantages(agent_rewards,expert_return,gamma=self.gamma)

                    # advantages=agent_return-expert_return
                    # value_loss=0
                    # beta=1
                    # weights = torch.exp(advantages / beta).clamp(max=1.0)  # avoid large weights
                    #
                    # self.log("train/weights", weights.mean().item(), on_step=True, batch_size=1)
                    #
                    # agent_wNLL=-(agent_log_prob*weights).mean()

                    advantages= F.normalize(agent_returns, p=2, dim=0)

                    value_loss=0


                agent_wNLL=-(agent_log_prob*advantages).mean()

                self.log("train/agent_wNLL", agent_wNLL.item(), on_step=True, batch_size=1)
                self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)

                expert_nll = expert_nll + agent_wNLL + value_loss  # - 0.01 * agent_entropy.mean()

            else:
                agent_reward, agent_value_loss, agent_V_diff, agent_nll,agent_Q,agent_proposal_loss = self.get_QV(
                    tokenized_map, tokenized_agent_rollout, train_mask,key='agent')

                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)

                self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

                constraint_loss=expert_V_diff.square().mean()*5

                self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_proposal_loss+expert_nll #+constraint_loss#+constraint_loss#critic_loss+constraint_loss #expert_nll #-0.01*agent_entropy.mean() #expert_nll+expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1

            if self.automatic_optimization == False:
                policy_optimizer.zero_grad()
                loss=expert_proposal_loss+expert_nll
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters())
                                               +list(self.encoder.value_network.parameters()), 0.5)
                policy_optimizer.step()

            #print(self.global_step)

            if self.encoder.agent_encoder.use_light:
                self.encoder.agent_encoder.pred_light=True
        else:
            loss = expert_nll + expert_proposal_loss

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q :
            soft_update(self.encoder.agent_encoder, self.target_net.agent_encoder, tau = 2e-4)

        return loss

