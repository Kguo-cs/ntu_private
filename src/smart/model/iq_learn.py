from jax.example_libraries.stax import logsoftmax
from lightning import LightningModule
import numpy as np
import torch

from src.smart.layers import MLPLayer
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
from src.smart.loss.rollout_buffer import RunningMeanStdTorch,get_reward,get_nei_returns,get_return,get_near_returns
from torch_scatter import scatter_mean

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

        if  self.use_target_q:
            self.target_net = copy.deepcopy(self.encoder.critic)
            self.target_net.load_state_dict(self.encoder.critic.state_dict())
            for param in self.target_net.parameters():
                param.requires_grad = False

        self.use_kl_penalty=self.encoder.agent_encoder.use_kl_penalty

        if self.use_kl_penalty:
            self.target_net = copy.deepcopy(self.encoder.agent_encoder)
            for param in self.target_net.parameters():
                param.requires_grad = False

        self.reward_type='airl'

        if self.iq_learn and self.use_gail:
            #self.running_meanstd=RunningMeanStdTorch(shape=(1))

            self.return_meanstd=RunningMeanStdTorch(shape=(1))
            self.ego_return_meanstd=RunningMeanStdTorch(shape=(1))
            self.global_return_meanstd=RunningMeanStdTorch(shape=(1))

        self.use_lcf=self.encoder.agent_encoder.use_lcf

        self.dis_loss="gail"

        self.learn_lcf=self.encoder.learn_lcf

        if self.use_lcf and self.iq_learn:

            if self.learn_lcf:

                self.lcf_parameters = MLPLayer(128,128,1)#[0.0, np.log(0.1)]

                self.automatic_optimization=False

        self.use_distance =False

            # self.lcf_parameters = torch.nn.Parameter(torch.as_tensor(lcf_parameters), requires_grad=True)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Unused parameter: {name}")

    def get_network_QV(self, q_value, tokenized_map, tokenized_agent, action, key):

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
            return 0,0,0,0,0,proposal_loss,0,0

        if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
            train_mask=tokenized_agent["train_mask"]
            valid_mask=valid_mask[train_mask]
            action=action[train_mask]
            train_mask=train_mask[train_mask]

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
            reward=reward[train_mask]

            value_loss=value_loss[train_mask]

            V=V[all_valid_mask]

            current_Q=current_Q[all_valid_mask]

            entropy =entropy[all_valid_mask]

            init_V=init_V[all_valid_mask]

            last_V=last_V[all_valid_mask]

            #off_ratio=(action==self.token_processor.agent_token_all_veh.shape[0])[train_mask].float().mean()
           # self.log("train/"+key+"_off_ratio", off_ratio.item(), on_step=True, batch_size=1)

        action_nll = -log_prob[train_mask].mean()

        self.log("train/"+key+"_V", V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_Q", current_Q.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_entropy", entropy.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_reward", reward.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_lastV", last_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_initV", init_V.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_nll", action_nll.item(), on_step=True, batch_size=1)

        if self.encoder.agent_encoder.interative_decoder.filter_ratio>0:
            self.log("train/"+key+"_a_ratio", self.encoder.agent_encoder.interative_decoder.a_ratio, on_step=True, batch_size=1)
            self.log("train/"+key+"_pt_ratio", self.encoder.agent_encoder.interative_decoder.pt_ratio, on_step=True, batch_size=1)

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

        # if len(pred["goal_q"]):
        #     goal_idx=tokenized_agent["goal_idx"][:, 2:]
        #
        #     log_prob_goal,goal_logpi=self.get_network_QV(pred["goal_q"], tokenized_map, tokenized_agent,goal_idx,key)[:2]
        #
        #     if key == "expert":
        #         goal_nll=-log_prob_goal[train_mask].mean()
        #         self.log("train/" + key + "_goal_nll", goal_nll.item(), on_step=True, batch_size=1)
        #     else:
        #         log_prob=log_prob_goal+log_prob
        #         goal_nll=0
        #
        # else:
        #     goal_nll=0

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

        return  reward,value_loss,pi,action_nll+light_nll,current_Q,proposal_loss,log_prob,entropy

    def get_d(self,f,log_prob):
        return torch.sigmoid(f)#-log_prob.detach()

    def get_reward(self,tokenized_agent,log_prob,key,train_mask=None,expert_disc_val=0):

        # all_features=tokenized_agent["detach_all_features"]
        map_feature=tokenized_agent["detach_map_feature"]

        # logit = self.encoder.discriminator(all_features,map_feature,agent_mask)[0][:, :, 0]
        #pos=tokenized_agent["sampled_pos"]#.clone()
        #heading=tokenized_agent["sampled_heading"]#.clone()

        # if key=="expert":
        #
        #     pos=pos+torch.randn_like(pos)*0.01
        #     heading=heading+torch.randn_like(heading)*0.01

        # logit=self.encoder.discriminator(tokenized_agent["feat_a"])#.detach()

        #distance_to_expert=1

        # state=tokenized_agent["feat_a"][train_mask].reshape(-1,128)
        #
        # state=(state-state.mean(0,keepdim=True))/(state.std(0,keepdim=True) + 1e-5)
        #
        # disc_val=self.encoder.discriminator._compute_disc_val(state, tokenized_agent["agent_token_emb"][:,2:][train_mask].reshape(-1,128)).reshape(-1,16)




        # sa=torch.cat([tokenized_agent["feat_a"],tokenized_agent["agent_token_emb"][:,2:]],dim=-1)[train_mask]
        #
        # logit =self.encoder.discriminator(sa)

        logit= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        tokenized_agent["goal_idx"],
                                                        tokenized_agent["valid_mask"],
                                                        tokenized_agent["sampled_pos"],
                                                        tokenized_agent["sampled_heading"] ,
                                                        tokenized_agent,
                                                        map_feature,
                                                        tokenized_agent["light_idx"],
                                                        None)[0]

        #logit=torch.tanh(logit)*10


        disc_val = self.get_d(logit[:, :, 0],log_prob)
        # svo=torch.sigmoid(logit[:,:,1])
        #
        # nei_disval=get_near_returns(tokenized_agent,disc_val,train_mask=train_mask)
        #
        # disc_val = svo*disc_val +(1-svo)* nei_disval

        if key == "agent" and self.use_kl_penalty:
            with torch.no_grad():
                target_q = self.target_net(tokenized_agent, map_feature)[
                    "agent_q"]

                ref_logprobs = (torch.softmax(target_q / self.alpha, dim=-1)+1e-10).log()

                # KL per token: sum_a p(a) * (log p(a) - log q(a))
                kl_coef=1

                kl_per_token = kl_coef * torch.sum(agent_pi *( (agent_pi+1e-10).log() - ref_logprobs), dim=-1)  # (B,T)

            self.log("train/" + key + "_kl_penalty", kl_per_token.mean().item(), on_step=True, batch_size=1)

        else:
            kl_per_token=0

            with torch.no_grad():
                # if key=="agent":
                #     self.encoder.discriminator.eval()
                #     logit = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                #                                                  tokenized_agent["goal_idx"],
                #                                                  tokenized_agent["valid_mask"],
                #                                                  tokenized_agent["sampled_pos"],
                #                                                  tokenized_agent["sampled_heading"],
                #                                                  tokenized_agent,
                #                                                  map_feature,
                #                                                  tokenized_agent["light_idx"],
                #                                                  None)[0]
                #     disc_val_eval = self.get_d(logit[:, :, 0],log_prob)
                #
                #     # svo = torch.sigmoid(logit[:, :, 1])
                #     #
                #     # nei_disc_val_eval = get_near_returns(tokenized_agent, disc_val_eval)
                #     #
                #     # disc_val_eval = svo * disc_val_eval +  (1 - svo)*nei_disc_val_eval
                #
                #     self.encoder.discriminator.train()
                # else:
                disc_val_eval=disc_val
                if  self.dis_loss == "wgan":
                    rewards=logit[:, :, 0].detach()
                else:
                    rewards=get_reward(disc_val_eval,kl_per_token)

                #rewards=(rewards-rewards.mean())/(rewards.std()+1e-4)

                returns = get_return(rewards, self.gamma)

            if  self.use_lcf and not self.encoder.use_value:
                with torch.no_grad():

                    batch = tokenized_agent["batch"]
                    global_rewards=scatter_mean(rewards,batch,dim=0)
                    # global_rewards_per_agent = global_rewards[batch]  # [N]

                    #global_returns=self.running_meanstd.get_return(global_rewards_per_agent, self.gamma)

                    self.log("train/" + key + "_global_rewards", global_rewards.mean().item(), on_step=True, batch_size=1)

                    #global_returns=scatter_mean(returns,batch,dim=0)[batch]


                    #self.log("train/" + key + "_global_returns", global_returns.mean().item(), on_step=True, batch_size=1)

                    nei_returns=get_nei_returns(tokenized_agent,returns)
                    self.log("train/" + key + "_nei_returns", nei_returns.mean().item(), on_step=True, batch_size=1)

                    #
                    # lcf =current_lcf_mean+ torch.randn_like(returns)*current_lcf_std
                    # step_lcf = torch.clamp(lcf, -1, 1)
                    # # Note: step_lcf is in [-1, 1]

                    #step_lcf=0.5
                    #used_lcf = step_lcf * np.pi / 2

                    # used_lcf=tokenized_agent["lcf"][:,:,0]
                    #
                    # self.log("train/" + key + "_lcf_mean", torch.cos(used_lcf).mean(), on_step=True, batch_size=1)
                    # self.log("train/" + key + "_lcf_std", torch.cos(used_lcf).std(), on_step=True, batch_size=1)

                    # returns=(returns-returns.mean())/(returns.std()+1e-4)
                    # nei_returns=(nei_returns-nei_returns.mean())/(nei_returns.std()+1e-4)

                    # self.ego_return_meanstd.update(returns)
                    # ego_returns = self.ego_return_meanstd.normalize(returns)
                    #
                    # self.global_return_meanstd.update(nei_returns)
                    # nei_returns = self.global_return_meanstd.normalize(nei_returns)
                    ego_returns=(returns-returns.mean())/(returns.std()+1e-4)
                    nei_returns=(nei_returns-nei_returns.mean())/(nei_returns.std()+1e-4)

                    returns=0.5*ego_returns+0.5*nei_returns

                # self._raw_lcf_adv_mean = returns.mean()
                # self._raw_lcf_adv_std = max(1e-4, returns.std())

            bottleneck_loss=0

        #entropy = -(disc_val * torch.log(disc_val + 1e-8) + (1 - disc_val) * torch.log(1 - disc_val + 1e-8)).mean()

        #entropy1= (1. - disc_val) * logit[:,:,0][train_mask] - torch.log(disc_val)
        if  self.dis_loss=="pugail":
            positive_class_prior = 0.7
            pugail_beta=None

            if key == "expert":
                # positive loss: prior * -ln(D(expert)) = prior * -logsigmoid(logits)
                bce_loss = positive_class_prior * -disc_val.log()
            else:
                bce_loss = -(1 - disc_val).log() - positive_class_prior * -(1 - expert_disc_val).log()

                # negative loss: -ln(1 - D(policy)) - prior * -ln(1 - D(expert))
                if pugail_beta is not None:
                    bce_loss = torch.clamp(bce_loss, min=-1.0 * pugail_beta)

            bce_loss = bce_loss.mean()
        elif self.dis_loss=="wgan":
            if key == "expert":
                bce_loss = -logit[:, :, 0].mean()#self.bce_loss(disc_val, torch.ones_like(disc_val)) #-disc_val.log()
            else:
                bce_loss = logit[:, :, 0].mean()#self.bce_loss(disc_val, torch.zeros_like(disc_val)) # -(1 - disc_val).log()
        else:
            if key == "expert":
                bce_loss = self.bce_loss(disc_val, torch.ones_like(disc_val)) #-disc_val.log()
            else:
                bce_loss = self.bce_loss(disc_val, torch.zeros_like(disc_val)) # -(1 - disc_val).log()

        self.log("train/"+key+"_dis_loss", bce_loss, on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_return", returns.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_rewards", rewards.mean().item(), on_step=True, batch_size=1)

        if self.dis_loss == "wgan" and key == "agent":
            expert_pos=tokenized_agent["expert_sampled_pos"]
            expert_sampled_heading=tokenized_agent["expert_sampled_heading"]
            expert_valid_mask=tokenized_agent["expert_valid_mask"]
            pos=tokenized_agent["sampled_pos"]
            heading=tokenized_agent["sampled_heading"]

            alpha= torch.rand((expert_pos.size(0), expert_pos.size(1)), device=expert_pos.device)
            interpolate_pos = alpha[:,:,None] * expert_pos + (1 - alpha[:,:,None]) * pos
            interpolate_heading = alpha * expert_sampled_heading + (1 - alpha) * heading

            interpolates_pos=torch.cat((interpolate_pos, interpolate_heading[:,:,None]), dim=-1)

            interpolates=interpolates_pos[train_mask,2:]

            interpolates.requires_grad_(True)  # IMPORTANT

            interpolates_pos[train_mask,2:]=interpolates

            # input_pos=torch.cat([interpolates_pos[:,:2],interpolates],dim=1)
            #scores=self.encoder.discriminator1(interpolates_pos)
            scores= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                            tokenized_agent["goal_idx"],
                                                            expert_valid_mask,
                                                            interpolates_pos[:,:,:2],
                                                            interpolates_pos[:,:,2] ,
                                                            tokenized_agent,
                                                            map_feature,
                                                            tokenized_agent["light_idx"],
                                                            None)[0]
            score_sum = scores.sum()

            # gradient wrt interpolates
            gradients = torch.autograd.grad(
                outputs=score_sum,
                inputs=interpolates,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]  # shape: [B, T, 3]

            grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
            gp = ((grad_norm - 1) ** 2).mean() * 0.01

            print(gp)

            self.log("train/gp", gp, on_step=True, batch_size=1)

            bce_loss=gp+bce_loss

        return bce_loss+bottleneck_loss,rewards,returns,disc_val #-0.03*entropy

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]
        train_mask = valid_mask[:, 1:] &  valid_mask[:, :-1]
        tokenized_agent["vis_mask"] = None
        all_valid=valid_mask.all(-1)

        if self.iq_learn:
            self.encoder.agent_encoder.a_t_roformer.attn.caching = True
            if self.encoder.agent_encoder.pred_light and not self.encoder.agent_encoder.light_encoder.share:
                self.encoder.agent_encoder.light_encoder.lg_t_roformer.attn.caching = True

        expert_reward,expert_value_loss,expert_pi,expert_nll,expert_Q,expert_proposal_loss,expert_log_prob,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        tokenized_agent["train_mask"]=all_valid

        if self.iq_learn:
            if self.use_gail and not self.use_distance:
                expert_dis_loss, expert_rewards, expert_returns,expert_disc_val=self.get_reward(tokenized_agent,expert_log_prob,"expert",all_valid)

            expert_light_idx=tokenized_agent["light_idx"].clone()

            if self.dis_loss=="wgan":
                tokenized_agent["expert_sampled_pos"]=tokenized_agent["sampled_pos"].clone()
                tokenized_agent["expert_sampled_heading"]=tokenized_agent["sampled_heading"].clone()
                tokenized_agent["expert_valid_mask"]=tokenized_agent["valid_mask"].clone()

            if self.use_distance:
                #gt_contour = cal_polygon_contour(tokenized_agent["sampled_pos"][all_valid][:,2:], tokenized_agent["sampled_heading"][all_valid][:,2:], tokenized_agent["token_agent_shape"][all_valid][:,None])

                pos=tokenized_agent["gt_pos_raw"].clone()#use original pos
                heading=tokenized_agent["gt_head_raw"].clone()#use original pos
                token_agent_shape=tokenized_agent["token_agent_shape"][:,None][all_valid]

                # pos_noise=torch.randn_like(pos)*0.05#*torch.rand_like(pos)
                #
                # heading_noise=torch.randn_like(heading)*0.05#*torch.rand_like(heading)

                # noised_pos=pos+pos_noise
                # noised_heading=wrap_angle(heading+heading_noise)

                noised_pos= tokenized_agent["sampled_pos"]
                noised_heading=tokenized_agent["sampled_heading"]

                pos_local, heading_local=transform_to_local(pos.reshape(-1,1,2),heading.reshape(-1,1),noised_pos.reshape(-1,2),noised_heading.reshape(-1))

                pos_noise=pos_local.reshape(pos.shape)
                heading_noise=heading_local.reshape(heading.shape)
                # pos_noise=noised_pos-pos
                # heading_noise=wrap_angle(noised_heading-heading)

                noise_pred = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                                 tokenized_agent["goal_idx"],
                                                                 tokenized_agent["valid_mask"],
                                                                 noised_pos,
                                                                 noised_heading,
                                                                 tokenized_agent,
                                                                 tokenized_agent["detach_map_feature"],
                                                                 tokenized_agent["light_idx"],
                                                                 None)[0]

                pos_global, head_global=transform_to_global(noise_pred[:,:,:2].reshape(-1,1,2),noise_pred[:,:,2].reshape(-1,1),
                                                            noised_pos[all_valid][:,2:].reshape(-1,2),noised_heading[all_valid][:,2:].reshape(-1))

                pred_pos = pos_global.reshape(noise_pred.shape[0],noise_pred.shape[1],2)

                pred_heading = head_global.reshape(noise_pred.shape[0],noise_pred.shape[1])

                # pred_pos=noised_pos[all_valid][:,2:]-noise_pred[:,:,:2]
                # pred_heading=noised_heading[all_valid][:,2:]-noise_pred[:,:,2]

                pred_contour=cal_polygon_contour(pred_pos, pred_heading, token_agent_shape)

                real_contour=cal_polygon_contour(pos[all_valid][:,2:], heading[all_valid][:,2:], token_agent_shape)

                noise_error=torch.linalg.norm(pred_contour-real_contour,dim=-1).mean()

                real_noise=torch.cat([pos_noise,heading_noise[:,:,None]],dim=-1)[all_valid][:,2:]

                #noise_error=torch.linalg.norm(noise_pred-real_noise,ord=1,dim=-1).mean()

                pos_error=torch.linalg.norm(noise_pred[:,:,:2]-real_noise[:,:,:2],dim=-1).mean()
                heading_error=wrap_angle(noise_pred[:,:,2]-real_noise[:,:,2]).abs().mean()

                self.log("train/expert_pos_loss", pos_error.item(), on_step=True, batch_size=1)
                self.log("train/expert_heading_loss", heading_error.item(), on_step=True, batch_size=1)

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

            if self.encoder.agent_encoder.pred_light:
                eval_light(expert_light_idx, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            # tokenized_agent_rollout["train_mask"]=None

            agent_reward, agent_value_loss, agent_pi, agent_nll,agent_Q,agent_proposal_loss,agent_log_prob,agent_entropy = self.get_QV(
                tokenized_map, tokenized_agent_rollout, None,key='agent')

            # tokenized_agent_rollout["train_mask"]=all_valid

            if self.use_gail:
                if not self.use_distance:

                    agent_dis_loss, agent_rewards, agent_returns, agent_disc_val = self.get_reward(tokenized_agent_rollout, agent_log_prob, "agent",all_valid,expert_disc_val)

                else:
                    # agent_contour = cal_polygon_contour(tokenized_agent_rollout["sampled_pos"][all_valid][:, 2:],
                    #                                  tokenized_agent_rollout["sampled_heading"][all_valid][:, 2:],
                    #                                  tokenized_agent_rollout["token_agent_shape"][all_valid][:, None])
                    #
                    # agent_reward= -torch.linalg.norm(agent_contour-gt_contour,dim=-1).mean(-1)
                    #
                    # agent_rewards= (agent_reward-agent_reward.mean())/(agent_reward.std()+1e-4)
                    with torch.no_grad():
                        error_pred = self.encoder.discriminator.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                                              tokenized_agent_rollout["goal_idx"],
                                                                              tokenized_agent_rollout["valid_mask"],
                                                                              tokenized_agent_rollout["sampled_pos"],
                                                                              tokenized_agent_rollout["sampled_heading"],
                                                                              tokenized_agent_rollout,
                                                                              tokenized_agent_rollout["detach_map_feature"],
                                                                              tokenized_agent_rollout["light_idx"],
                                                                              None)[0]

                        pos_error = torch.linalg.norm(error_pred[:, :, :2], dim=-1).mean()
                        heading_error = (error_pred[:, :, 2]).abs().mean()

                        self.log("train/agent_pos_error", pos_error.item(), on_step=True, batch_size=1)
                        self.log("train/agent_heading_error", heading_error.item(), on_step=True, batch_size=1)

                        agent_contour = cal_polygon_contour(tokenized_agent_rollout["sampled_pos"][all_valid][:,2:],
                                                            tokenized_agent_rollout["sampled_heading"][all_valid][:,2:], token_agent_shape)

                        # pred_pos=tokenized_agent_rollout["sampled_pos"][all_valid][:,2:]+error_pred[:,:,:2]
                        # pred_heading=tokenized_agent_rollout["sampled_heading"][all_valid][:,2:]+error_pred[:,:,2]

                        pos_global, head_global = transform_to_global(error_pred[:, :, :2].reshape(-1, 1, 2),
                                                                      error_pred[:, :, 2].reshape(-1, 1),
                                                                      tokenized_agent_rollout["sampled_pos"][all_valid][:, 2:].reshape(-1, 2),
                                                                      tokenized_agent_rollout["sampled_heading"][all_valid][:, 2:].reshape(-1))

                        pred_pos = pos_global.reshape(error_pred.shape[0], error_pred.shape[1], 2)

                        pred_heading = head_global.reshape(error_pred.shape[0], error_pred.shape[1])

                        pred_contour = cal_polygon_contour(pred_pos, pred_heading, token_agent_shape)

                        agent_rewards =-torch.linalg.norm(agent_contour-pred_contour,dim=-1).mean(-1) #torch.linalg.norm(error_pred,ord=1,dim=-1)

                        #agent_rewards = (agent_rewards - agent_rewards.mean()) / (agent_rewards.std() + 1e-4)

                    expert_dis_loss=noise_error
                    agent_dis_loss=torch.tensor(0.0)


                if self.encoder.agent_encoder.use_latent:

                    logits = self.encoder.RecognitionQ.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                         tokenized_agent_rollout["goal_idx"],
                                                         tokenized_agent_rollout["valid_mask"],
                                                         tokenized_agent_rollout["sampled_pos"],
                                                         tokenized_agent_rollout["sampled_heading"],
                                                         tokenized_agent_rollout,
                                                         tokenized_agent_rollout["detach_map_feature"],
                                                         tokenized_agent_rollout["light_idx"],
                                                         None)[0]#[all_valid]

                    log_q = F.log_softmax(logits, dim=-1)
                    z_idx=tokenized_agent_rollout["latent_z"][all_valid]
                    action=z_idx[:,None].repeat(1,log_q.shape[1],1)

                    #s=one_hot(action[:,:,0],K=2)

                    # 1) Cross-entropy for Q (supervised)
                    # logits_flat = logits.reshape(-1, 2)  # [B*T, K]
                    # targets_flat = action.reshape(-1)  # [B*T]
                    # loss_q = F.cross_entropy(logits_flat, targets_flat)  # scalar

                    bonus = torch.gather(log_q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a] #larger z likelihood

                    loss_q=-bonus.mean() # increase the z likelihood

                    self.log("train/loss_q", loss_q.item(), on_step=True, batch_size=1)

                    expert_nll=expert_nll+loss_q

                    mi_beta=0.1
                    r_mi = mi_beta * bonus

                    agent_rewards=agent_rewards+r_mi.detach()

                    # logits_p=tokenized_agent["logits_p"][all_valid]
                    #
                    # q_probs =log_q.detach()
                    #
                    # # KL(Q || P) averaged
                    # log_p = F.log_softmax(logits_p, dim=-1)[:,None].repeat(1,q_probs.shape[1],1)
                    #
                    # loss_prior = (q_probs * (q_probs.clamp_min(1e-8).log() - log_p)).sum(-1).mean()
                    #
                    # expert_nll=loss_prior+expert_nll

                    # # Optional entropy regularizer on P to avoid overconfidence
                    # p_probs = log_p.exp()
                    # H_p = -(p_probs * log_p).sum(-1).mean()
                    #
                    # tau=0.01
                    #
                    # loss_prior = kl_qp - tau * H_p  # tau ~ 0.01–0.1

                    #@self.log("train/loss_prior", loss_prior.item(), on_step=True, batch_size=1)


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


                if self.reward_type=="raw":
                    alpha=0.5
                    critic_loss =-expert_logit.mean()+agent_logit.mean()+expert_logit.square().mean() / (4 * alpha)
                else:
                    critic_loss=expert_dis_loss + agent_dis_loss

                if self.encoder.use_value:
                    # logit = self.encoder.value_network.predict_agent(tokenized_agent_rollout["sampled_idx"],
                    #                                                  tokenized_agent_rollout["goal_idx"],
                    #                                                  tokenized_agent_rollout["valid_mask"],
                    #                                                  tokenized_agent_rollout["sampled_pos"],
                    #                                                  tokenized_agent_rollout["sampled_heading"],
                    #                                                  tokenized_agent_rollout,
                    #                                                  tokenized_agent_rollout["detach_map_feature"],
                    #                                                  tokenized_agent_rollout["light_idx"],
                    #                                                  None)[0]#[all_valid]
                    # logit=self.encoder.value_network(tokenized_agent_rollout["all_features"],tokenized_agent_rollout["detach_map_feature"],all_valid)[0]
                    value_pred=self.encoder.value_network(tokenized_agent_rollout["feat_a"][all_valid])[:,:,0]

                    #value_pred=logit[:,:,0]

                    ego_advantages,returns=compute_advantages(agent_rewards,value_pred.detach(),None,gamma=self.gamma)#[all_valid]

                    value_loss = torch.pow(returns - value_pred, 2.0).clamp(min=0,max=100).mean()

                    if self.use_lcf:
                        nei_rewards = get_near_returns(tokenized_agent, agent_rewards,train_mask=all_valid)

                        # nei_value_pred=logit[:,:,1]

                        nei_value_pred=self.encoder.nei_value_network(tokenized_agent_rollout["feat_a"][all_valid])[:,:,0]

                        nei_advantages,nei_returns=compute_advantages(nei_rewards,nei_value_pred.detach(),None,gamma=self.gamma)

                        nei_value_loss = torch.pow(nei_returns - nei_value_pred, 2.0).clamp(min=0,max=100).mean()

                        value_loss = nei_value_loss + value_loss

                        if self.learn_lcf:

                            batch = tokenized_agent["batch"]

                            global_rewards=scatter_mean(agent_rewards,batch,dim=0)[batch]

                            global_value_pred=self.encoder.global_value_network(tokenized_agent_rollout["feat_a"][all_valid])[:,:,0]

                            global_advantages,global_returns=compute_advantages(global_rewards[all_valid],nei_value_pred.detach(),None,gamma=1.0)

                            self.global_return_meanstd.update(global_advantages)

                            global_advantages = self.global_return_meanstd.normalize(global_advantages)

                            global_value_loss = torch.pow(global_returns - global_value_pred, 2.0).mean()

                            value_loss= value_loss+global_value_loss

                            lcf_parameters=self.lcf_parameters(tokenized_agent_rollout["feat_a"][all_valid])

                            # current_lcf_mean = torch.clamp(torch.tanh(lcf_parameters[...,0]), -1 + 1e-6, 1 - 1e-6)
                            # current_lcf_std = torch.exp(torch.clamp(lcf_parameters[...,1], -20, 2))

                            # self.log("train/lcf_mean", current_lcf_mean.item(), on_step=True, batch_size=1)
                            # self.log("train/lcf_std", current_lcf_std.item(), on_step=True, batch_size=1)
                            #
                            # step_lcf=torch.randn_like(ego_advantages[:,:1])*current_lcf_std+current_lcf_mean

                            step_lcf=torch.clamp(torch.tanh(lcf_parameters[...,0]), -1 + 1e-6, 1 - 1e-6)

                            self.log("train/lcf_mean", step_lcf.mean().item(), on_step=True, batch_size=1)
                            self.log("train/lcf_std", step_lcf.std().item(), on_step=True, batch_size=1)
                        else:
                            step_lcf=torch.tensor(0.5)

                        used_lcf = step_lcf.detach() * np.pi / 2

                        advantages=  torch.cos(used_lcf) * ego_advantages + torch.sin(used_lcf) *nei_advantages

                        # _raw_lcf_adv_mean = advantages.mean()
                        # _raw_lcf_adv_std = max(1e-4, advantages.std())

                        # advantages=0.5*(advantages+nei_advantages)
                    else:
                        advantages=ego_advantages

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
                    advantages=agent_returns[all_valid]
                    value_loss=0
                    # expert_returns=expert_returns[all_valid]
                    # expert_advantages=self.return_meanstd.normalize(expert_returns)
                    # expert_wNLL=-(expert_log_prob[all_valid]*expert_advantages).mean()
                    #
                    # self.log("train/expert_wNLL", expert_wNLL.item(), on_step=True, batch_size=1)
                    # if self.use_lcf:
                    #     global_returns=self.global_returns[all_valid]
                    #     global_advantages=(global_returns - global_returns.mean()) / (global_returns.std() + 1e-8)
                    #
                    #     new_policy_loss=-(agent_log_prob[all_valid]*global_advantages).mean()
                    #
                    #     self.encoder.agent_encoder.
                    #
                    #     new_policy_grad=torch.autograd.grad(new_policy_loss,self.encoder.agent_encoder.parameters())
                    #     new_policy_grad = [g for g in new_policy_grad if g is not None]

                self.return_meanstd.update(advantages.detach().reshape(-1))
                advantages=self.return_meanstd.normalize(advantages)
                self.log("train/running_mean", self.return_meanstd.mean.mean(), on_step=True, batch_size=1)
                self.log("train/running_var", self.return_meanstd.var.mean(), on_step=True, batch_size=1)

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
                    agent_wNLL=-(agent_log_prob*advantages).mean()#[all_valid]

                self.log("train/agent_wNLL", agent_wNLL.item(), on_step=True, batch_size=1)
                self.log("train/advantages", advantages.mean().item(), on_step=True, batch_size=1)

                agent_density=torch.cumsum(agent_log_prob,dim=1).mean() #agent_log_prob.mean() #

                self.log("train/agent_density", agent_density.item(), on_step=True, batch_size=1)

                expert_nll = expert_nll + agent_wNLL + value_loss #-0.1*agent_density.mean()  # - 0.01 * agent_entropy.mean()

                if self.use_kl_penalty:
                    with torch.no_grad():
                        target_q = self.target_net(tokenized_agent_rollout, tokenized_agent_rollout["detach_map_feature"])[
                            "agent_q"]

                        ref_logprobs = (torch.softmax(target_q / self.alpha, dim=-1)+1e-10).log()

                    kl_coef=0.1

                    kl_per_token = kl_coef * torch.sum(agent_pi *( (agent_pi+1e-10).log() - ref_logprobs), dim=-1).mean() /torch.sqrt(self.running_meanstd.var.float()) # (B,T)

                    self.log("train/kl_penalty", kl_per_token.item(), on_step=True, batch_size=1)

                    expert_nll=expert_nll+kl_per_token
            else:
                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)
                # constraint_loss=expert_V_diff.square().mean()*5
                #
                # self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_proposal_loss+expert_nll

            if self.automatic_optimization == False:
                old_policy_loss = agent_log_prob.mean()
                params = [p for p in self.encoder.parameters() if p.requires_grad]

                self.encoder.zero_grad()
                old_policy_grad = torch.autograd.grad(old_policy_loss, params,retain_graph=True,allow_unused=True)
                old_policy_grad = [g for g in old_policy_grad if g is not None]

                policy_optimizer,lcf_optimizer=self.optimizers()

                policy_optimizer.zero_grad()
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(params, 0.5)
                policy_optimizer.step()


                agent_reward, agent_value_loss, agent_pi, agent_nll, agent_Q, agent_proposal_loss, agent_log_prob, agent_entropy = self.get_QV(
                    tokenized_map, tokenized_agent_rollout, None, key='agent')

                new_policy_loss=-(agent_log_prob[all_valid]*global_advantages).mean()

                self.encoder.zero_grad()
                new_policy_grad = torch.autograd.grad(new_policy_loss, params, allow_unused=True)
                new_policy_grad = [g for g in new_policy_grad if g is not None]

                grad_value = 0

                for a, b in zip(new_policy_grad, old_policy_grad):
                    assert a.shape == b.shape
                    grad_value += (a * b).sum()

                #step_lcf = torch.randn_like(ego_advantages[:, :1]) * current_lcf_std + current_lcf_mean
                used_lcf = step_lcf * np.pi / 2

                advantages = torch.cos(used_lcf) * ego_advantages + torch.sin(used_lcf) * nei_advantages

                lcf_advantages = self.return_meanstd.normalize(advantages)
                lcf_lcf_adv_loss = lcf_advantages.mean()
                lcf_final_loss = grad_value * lcf_lcf_adv_loss
                lcf_optimizer.zero_grad()
                lcf_final_loss.backward()
                lcf_optimizer.step()

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

