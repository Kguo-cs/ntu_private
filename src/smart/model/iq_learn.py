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
        self.output_gmm=self.encoder.output_gmm
        self.alpha = self.encoder.alpha
        self.n_token_agent=self.encoder.agent_encoder.n_token_agent

        if self.iq_learn and self.output_gmm:
            self.automatic_optimization = False

        self.use_target_q=False

        self.start_step=10//self.token_processor.shift-1

        self.use_gail=self.encoder.use_gail

        if  self.use_target_q:
            self.target_net = SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent,
                token_processor=self.token_processor
            )
            self.target_net.load_state_dict(self.encoder.state_dict())


    def get_network_QV(self,q_value,tokenized_map, tokenized_agent,action,key):

        if self.output_gmm:
            dist =  GMM_Dist(q_value)

            cov = q_value[...,q_value.shape[-1]//2+1:].reshape(-1,3)

            self.log("train/" + key + "_xCov", cov[...,0].mean().item(), on_step=True, batch_size=1)
            self.log("train/" + key + "_yCov", cov[...,1].mean().item(), on_step=True, batch_size=1)
            self.log("train/" + key + "_headingCov", cov[...,2].mean().item(), on_step=True, batch_size=1)

            entropy=get_entropy(q_value)

            action, action_valid = get_euclidean_targets(
                pred_pos=tokenized_agent["sampled_pos"],
                pred_head=tokenized_agent["sampled_heading"],
                pred_valid=tokenized_agent["valid_mask"],
                gt_pos=tokenized_agent["sampled_pos"],
                gt_head=tokenized_agent["sampled_heading"],
                gt_valid=tokenized_agent["valid_mask"]
            )
            if "target" in tokenized_agent.keys():
                expert_action=tokenized_agent["target"]
            else:
                expert_action=action

            log_prob = dist.log_prob(expert_action)[:,:-1].clamp_min(min=np.log(1e-5))

            if self.iq_learn:

                sample_num=32

                sampled_action=dist.sample([sample_num*2])

                sampled_log_prob=dist.log_prob(sampled_action)

                sampled_action=torch.cat([sampled_action,action[None]],dim=0)

                pred=network(tokenized_map, tokenized_agent,True)

                feat_a=pred["feat_a"][None].repeat_interleave(sample_num*2+1,dim=0)

                Q=network.get_Q(feat_a,sampled_action)

                current_Q=Q[-1,:,:-1]

                V = Q[:sample_num*2]-self.alpha*sampled_log_prob.detach()

                v_value=V.mean(0)

                current_V= V[:sample_num,:,:-1].mean(0)

                next_V =V[sample_num:,:,1:].mean(0)

                rsampled_action=dist.rsample([sample_num])
                rsampled_log_prob=dist.log_prob(rsampled_action)
                feat_a=pred["feat_a"][None,:,:-1].detach().repeat_interleave(sample_num,dim=0)

                actor_loss=self.alpha * rsampled_log_prob[:,:,:-1] - network.get_Q(feat_a, rsampled_action[:,:,:-1])

                actor_loss=actor_loss.mean(0)

            else:
                next_V =current_Q=current_V=actor_loss= torch.zeros_like(log_prob)
                v_value = torch.zeros_like(torch.cat([log_prob, torch.zeros_like(log_prob[:,:1])],dim=1))
        else:
            action = action.unsqueeze(-1)  # .reshape(-1)

            q = q_value[:, :-1]

            current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

            V =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

            current_V = V[:, :-1]

            next_V = V[:, 1:]

            pi = torch.softmax( q / self.alpha, dim=-1)

            logpi= torch.log(pi+ 1e-10)#.clamp_min(min=1e-10)

            #next_Q=torch.sum(q_value[:,1:]*torch.softmax( q_value[:,1:] / self.alpha, dim=-1),dim=-1)

            # log_pi_stack=torch.log_softmax(all_q_value[:, :-1]/ self.alpha, dim=-1)
            #
            # rolling_action = torch.stack([
            #             torch.roll(action, shifts=-i, dims=1)
            #             for i in range(log_pi_stack.shape[2])
            #         ], dim=-2)  # [B, Tm1, T_a]
            #
            # log_prob1=torch.gather(log_pi_stack, dim=-1, index=rolling_action).squeeze(-1)
            #
            # valid_mask=torch.ones_like(log_prob1)
            # for i in range(log_pi_stack.shape[2]):
            #     log_prob1[:,rolling_action.shape[1]-i:,i]=0
            #     valid_mask[:,rolling_action.shape[1]-i:,i]=0
            #
            # log_prob=log_prob1.sum(-1)/valid_mask.sum(-1)
            # act=action.reshape(-1)
            # log_prob=logpi.reshape(len(act), -1)[torch.arange(len(act)), act].reshape(q.shape[0], q.shape[1])

            log_prob=torch.gather(logpi, dim=-1, index=action).squeeze(-1)
            entropy = -torch.sum(pi * logpi, dim=-1)

            actor_loss = self.alpha * log_prob - current_Q

        dones = torch.zeros_like(next_V)
        dones[:, -1] = 1
        y=self.gamma *(1 - dones) * next_V
        reward = current_Q - y
        value_loss = current_V - y

        # if key=="agent":
        #     current_Q=torch.sum(q*pi,dim=-1)

        return log_prob,logpi,actor_loss,entropy,current_Q,V,value_loss,reward

    def get_QV(self, tokenized_map, tokenized_agent,train_mask, key='expert'):

        pred = self.encoder(tokenized_map, tokenized_agent)#,post_sampling=(key=='expert')

        if pred["proposal"] is not None:
            if key=="expert":
                if self.encoder.agent_encoder.pred_gaussian:
                    proposal_loss,pos_dist, head_diff=get_gaussian_loss(pred["proposal"],tokenized_agent )
                    action = tokenized_agent["sampled_idx"][:, 2:]
                else:
                    proposal_loss, pos_dist, head_diff,min_idx=get_proposal_loss(pred["proposal"],tokenized_agent,self.start_step,train_mask )

                    self.log("train/" + key + "_head_diff", head_diff.item(), on_step=True, batch_size=1)

                self.log("train/" + key + "_pos_dist", pos_dist.item(), on_step=True, batch_size=1)
                self.log("train/" + key + "_proposal_loss", proposal_loss.item(), on_step=True, batch_size=1)
            else:
                # proposal=pred["proposal"][:,1:-1,:,4].flatten(0,1)

                # global_pos, global_head = transform_to_global(
                #     pos_local=proposal[...,:2],
                #     head_local=proposal[...,2],
                #     pos_now=tokenized_agent["sampled_pos"][:,1:-1].flatten(0,1),
                #     head_now=tokenized_agent["sampled_heading"][:,1:-1].flatten(0,1),
                # )

                # global_pos=global_pos.reshape(-1,16,global_pos.shape[-2],2)

                # dist=torch.norm(global_pos - tokenized_agent["sampled_pos"][:, 2:,None], dim=-1)

                # action = torch.argmin(dist, dim=-1)
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

        current_Q_diff, V_diff = get_return_diff(reward,log_prob,current_Q,V,self.alpha,self.gamma)

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

        current_Q_diff=current_Q_diff[all_valid_mask]
        V_diff=V_diff[all_valid_mask]

        self.log("train/"+key+"_V", V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_initV", init_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_actor_loss", actor_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q_diff", current_Q_diff.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_V_diff", V_diff.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_nll", action_nll.item(), on_step=True, batch_size=1)

        off_ratio=(action==self.token_processor.agent_token_all_veh.shape[0])[train_mask].float().mean()
        self.log("train/"+key+"_off_ratio", off_ratio.item(), on_step=True, batch_size=1)
        
        if self.iq_learn and not self.use_gail:
            action_nll=0

        if self.use_target_q:
            V=(V-target_V)[all_valid_mask]

        return  reward,value_loss,V,action_nll+light_nll,current_Q,proposal_loss,log_prob

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]

        if "col_mask" in tokenized_agent.keys():
            col_mask = tokenized_agent["col_mask"][:, 1+self.start_step:]
            state_mask=valid_mask[:,:-1]
            action_mask=valid_mask[:,1:]

            action_mask[:col_mask.shape[0]]=action_mask[:col_mask.shape[0]] & (~col_mask)

            train_mask= state_mask & action_mask
        else:
            train_mask = valid_mask.all(-1)
            tokenized_agent["train_mask"] = train_mask

            # if self.iq_learn:
            #     train_mask = valid_mask.all(-1)
            #     tokenized_agent["train_mask"]=train_mask
            # else:
            #     state_mask = valid_mask[:, :-1]
            #     action_mask = valid_mask[:, 1:]
            #     train_mask = state_mask & action_mask

        # if self.iq_learn:
        #     self.encoder.agent_encoder.a_t_roformer.attn.caching = True

        expert_reward,expert_value_loss,expert_V_diff,expert_nll,expert_Q,expert_proposal_loss,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        if self.iq_learn:
            self.encoder.agent_encoder.pred_light=False
            criterion = nn.BCELoss()

            if self.use_gail:
                # for key in ["sampled_pos", "sampled_heading"]:
                #     tokenized_agent[key] = tokenized_agent[key] + 1e-4 * torch.randn_like(tokenized_agent[key])

                expert_logit=self.encoder.discriminator(tokenized_agent["all_features"])[0]

                expert_d = torch.sigmoid(expert_logit[:,1:,0])

                gamma=0

                expert_return,_=get_return(expert_d,gamma)
                expert_loss = criterion(expert_d, torch.ones_like(expert_d))

                self.log("train/expert_dis_loss", expert_loss, on_step=True, batch_size=1)
                self.log("train/expert_disc_val", expert_d.mean().item(), on_step=True, batch_size=1)
                self.log("train/expert_return", expert_return.mean().item(), on_step=True, batch_size=1)

            tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent)

            if self.encoder.agent_encoder.pred_light:
                eval_light(tokenized_agent, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            if self.use_gail:
                # for key in ["sampled_pos", "sampled_heading"]:
                #     tokenized_agent_rollout[key] = tokenized_agent_rollout[key] + 1e-4 * torch.randn_like(tokenized_agent[key])
                #value_pred=self.encoder.value_network(pred["feat_a"][:,:-1])[:,:,0]

                agent_reward, agent_value_loss, agent_V_diff, agent_nll,agent_Q,agent_proposal_loss,agent_log_prob = self.get_QV(
                    tokenized_map, tokenized_agent_rollout, train_mask,key='agent')
                
                agent_logit=self.encoder.discriminator(tokenized_agent_rollout["all_features"])[0]

                agent_d = torch.sigmoid(agent_logit[:,1:,0])
                agent_loss = criterion(agent_d, torch.zeros_like(agent_d))
                agent_return,agent_rewards=get_return(agent_d,gamma)
                critic_loss = expert_loss + agent_loss

                self.log("train/agent_dis_loss", agent_loss, on_step=True, batch_size=1)
                self.log("train/agent_disc_val", agent_d.mean().item(), on_step=True, batch_size=1)
                self.log("train/agent_return", agent_return.mean().item(), on_step=True, batch_size=1)
                self.log("train/agent_rewards", agent_rewards.mean().item(), on_step=True, batch_size=1)


                #advantages,returns=compute_advantages(agent_rewards,value_pred)

                #weights = torch.exp(advantages / 1).clamp(max=1.0)  # avoid large weights

                # value_loss = 0.5 * (returns - value_pred).pow(2).mean()

                # self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)
                # self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)

                baseline_return,_=get_return(torch.ones_like(agent_d),gamma)
                # adv=agent_return-baseline_return
                #weight=(advantages>0).float()
                
                beta=1
                advantages=agent_return-baseline_return
                weights = torch.exp(advantages / beta)#.clamp(max=1.0)  # avoid large weights
                self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)
                self.log("train/weights", weights.mean().item(), on_step=True, batch_size=1)

                agent_wNLL=-(agent_log_prob*weights).mean()
                self.log("train/agent_wNLL", agent_wNLL.item(), on_step=True, batch_size=1)

                expert_nll=expert_nll+agent_wNLL #+value_loss

            else:
                agent_reward, agent_value_loss, agent_V_diff, agent_nll,agent_Q,agent_proposal_loss = self.get_QV(
                    tokenized_map, tokenized_agent_rollout, train_mask,key='agent')

                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            constraint_loss=expert_V_diff.square().mean()*5

            self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_proposal_loss+expert_nll #+constraint_loss#+constraint_loss#critic_loss+constraint_loss #expert_nll #-0.01*agent_entropy.mean() #expert_nll+expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1
            self.encoder.agent_encoder.pred_light=True

            if self.automatic_optimization==False:
                actor_optimizer,critic_optimizer=self.optimizers()

                actor_loss=expert_actor_loss.mean()/2+agent_actor_loss.mean()/2#expert_nll #
                self.log("train/actor_loss", actor_loss.item(), on_step=True, batch_size=1)

                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters()), max_norm=0.5)
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.critic.parameters(), max_norm=0.5)
                critic_optimizer.step()

                loss=loss+actor_loss
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

