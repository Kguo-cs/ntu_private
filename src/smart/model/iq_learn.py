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
from src.smart.loss.rollout_buffer import RunningMeanStdTorch,get_reward,get_nei_returns,get_return,get_near_returns,per_scene_zscore_clip
from torch_scatter import scatter_mean
from torch.distributions import Categorical, Normal, Independent
from src.smart.loss.kl_loss import BalancedKL
from src.smart.loss.collision_check import oriented_box_collision,signed_distance_boxes_sat_fast,value_to_hist_class
from src.smart.loss.offroad_check import corners_offroad_signed_distance_per_batch
from torch_scatter import scatter_max

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
            self.bc_net = copy.deepcopy(self.encoder.agent_encoder)

            self.bc_net.target_net=True
            for param in self.bc_net.parameters():
                param.requires_grad = False
            self.bc_net.eval()

            if self.encoder.map_encoder.type_pt_emb.weight.requires_grad:
                self.bc_map_net = copy.deepcopy(self.encoder.map_encoder)

                for param in self.bc_map_net.parameters():
                    param.requires_grad = False

                self.bc_map_net.eval()
            else:
                self.bc_map_net = None

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

        if self.encoder.agent_encoder.use_vae:

            self.l_vae_kl = BalancedKL(kl_balance_scale=0.2, kl_free_nats=1.0)

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

    def get_reward(self,tokenized_agent,agent_log_prob,agent_pi,key,train_mask=None,expert_disc_val=0,tokenized_map=None):

        disc_out= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        tokenized_agent["goal_idx"],
                                                        tokenized_agent["valid_mask"],#expert_
                                                        tokenized_agent["sampled_pos"],
                                                        tokenized_agent["sampled_heading"] ,
                                                        tokenized_agent,
                                                        tokenized_agent["detach_map_feature"],
                                                        tokenized_agent["light_idx"],
                                                        None,
                                                       # latent_z=tokenized_agent["latent_z"]
                                                        )#Metrics-Guided Adversarial Training

        logit=disc_out[0]

        disc_val = torch.sigmoid(logit[:, :, 0])

        if key == "agent" and self.use_kl_penalty:
            with torch.no_grad():
                if self.bc_map_net is not None:
                    map_feature=self.bc_map_net(tokenized_map)

                target_q = self.bc_net(tokenized_agent, map_feature)["agent_q"]

                logp_ref = (torch.softmax(target_q / self.alpha, dim=-1)+1e-10).log()

                actions=tokenized_agent["sampled_idx"][:,2:][train_mask]

                logp_a_ref=torch.gather(logp_ref, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

                kl_penalty =  torch.sum(agent_pi *( (agent_pi+1e-10).log() - logp_ref), dim=-1).mean()  # (B,T)

                self.log("train/kl_penalty", kl_penalty.item(), on_step=True, batch_size=1)

                kl_coef=1#np.power(0.9999,self.global_step)
                kl_taken = (agent_log_prob - logp_a_ref)

                kl_per_token=kl_coef *kl_taken

        else:
            kl_per_token=0

        with torch.no_grad():
            disc_val_eval=disc_val
            if  self.dis_loss == "wgan":
                rewards=logit[:, :, 0].detach()
            else:
                rewards=get_reward(disc_val_eval,kl_per_token=kl_per_token)

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
            if train_mask is not None and not self.encoder.discriminator.interative_decoder.centric:
                disc_val=disc_val[train_mask]

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

        return bce_loss,rewards,returns,disc_out[-1] #-0.03*entropy

    def get_collision_loss(self,tokenized_agent,tokenized_map,dis_feature,train_mask,all_valid ,key):

        col_pred = self.encoder.col_head(tokenized_agent["feat_a_nodetach"][all_valid])

        if train_mask is not None:
            col_pred=col_pred[train_mask]
        else:
            col_pred = col_pred.reshape(-1,col_pred.shape[-1])

        if self.encoder.pred_dis_aux:
            dis_col_pred = self.encoder.dis_col_head(dis_feature)

        if self.encoder.agent_encoder.use_sign_dist:
            sign_dist = signed_distance_boxes_sat_fast(tokenized_agent["sampled_pos"][:, 2:],
                                                       tokenized_agent["sampled_heading"][:, 2:],
                                                       tokenized_agent["shape"][:, :2],
                                                       tokenized_agent["batch"])

            col_flag = sign_dist < 0
            hist = {"min_val": -5.0, "max_val": 10.0, "num_bins": 3}

            target = value_to_hist_class(sign_dist, **hist)
            col_loss = F.cross_entropy(col_pred, target[train_mask])
            dis_loss=F.cross_entropy(dis_col_pred.reshape(-1, col_pred.shape[-1]), target[all_valid].reshape(-1))
        else:
            col_flag = oriented_box_collision(tokenized_agent["sampled_pos"][:, 2:],
                                             tokenized_agent["sampled_heading"][:, 2:],
                                             tokenized_agent["shape"][:, :2],
                                             tokenized_agent["batch"])[0].float()[all_valid]

            if train_mask is not None:
                col_flag = col_flag[train_mask]
            else:
                col_flag = col_flag.reshape(-1)

            col_loss = self.bce_loss(col_pred[:,0], col_flag)
            if self.encoder.pred_dis_aux:
                dis_loss = self.bce_loss(dis_col_pred[:,:,0], col_flag)
                self.log('train/'+key+'_dis_col_loss', dis_loss.item(), on_step=True, batch_size=1)
            else:
                dis_loss = 0

        self.log('train/'+key+'_col_loss', col_loss.item(), on_step=True, batch_size=1)
        self.log('train/'+key+'_col_rate', col_flag.float().mean().item(), on_step=True, batch_size=1)

        if self.encoder.map_encoder.pred_offroad:
            # for i in range(len(tokenized_agent["sampled_pos"][:, 2:])):
            #     print(i)
            # train_mask=torch.zeros_like(tokenized_agent["sampled_pos"][:, 2:]).bool()
            # train_mask[i]=True

            near_dist=corners_offroad_signed_distance_per_batch(tokenized_agent["sampled_pos"][:, 2:][all_valid],
                                                     tokenized_agent["sampled_heading"][:, 2:][all_valid],
                                                     tokenized_agent["shape"][:, :2][all_valid],
                                                     tokenized_agent["batch"][all_valid],
                                                      tokenized_map["global_edge"],
                                                      tokenized_map["batch_edge"],
                                                      )[1]
            offroad_flag= (near_dist < 0).float()

            if train_mask is not None:
                valid_off_flag=offroad_flag[train_mask]
            else:
                valid_off_flag=offroad_flag.reshape(-1)

            off_road_loss = self.bce_loss(col_pred[:,1], valid_off_flag)

            if self.encoder.pred_dis_aux:
                dis_off_road_loss=self.bce_loss(dis_col_pred[:,:,1], offroad_flag[all_valid])
                self.log('train/'+key+'_dis_off_loss', dis_off_road_loss.item(), on_step=True, batch_size=1)
            else:
                dis_off_road_loss=0

            self.log('train/'+key+'_off_road_loss', off_road_loss.item(), on_step=True, batch_size=1)
            self.log('train/'+key+'_offroad_rate', valid_off_flag.mean().item(), on_step=True, batch_size=1)

            dis_loss=dis_loss+dis_off_road_loss
            col_loss=col_loss+off_road_loss

        return 0.1*col_loss+dis_loss

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]
        train_mask = valid_mask[:, 1:] &  valid_mask[:, :-1]
        tokenized_agent["vis_mask"] = None

        if "pred_mask" in tokenized_agent.keys():
            all_valid=tokenized_agent["pred_mask"] & valid_mask.all(-1)
        else:
            all_valid=valid_mask.all(-1)
        #tokenized_agent["expert_valid_mask"] = tokenized_agent["valid_mask"].clone()

        if self.use_kl_penalty:
            expert_nll=0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            if self.iq_learn:
                self.encoder.agent_encoder.a_t_roformer.attn.caching = True
                if self.encoder.agent_encoder.pred_light and not self.encoder.agent_encoder.light_encoder.share:
                    self.encoder.agent_encoder.light_encoder.lg_t_roformer.attn.caching = True

            expert_reward,expert_value_loss,expert_pi,expert_nll,expert_Q,expert_proposal_loss,expert_log_prob,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        if self.encoder.agent_encoder.use_vae:
            latent_post=tokenized_agent["latent_post"]
            latent_prior=tokenized_agent["latent_prior"]

            error_vae = self.l_vae_kl.compute(latent_post.distribution, latent_prior.distribution).mean()

            self.log("train/error_vae", error_vae.item(), on_step=True, batch_size=1)

            expert_nll=expert_nll+error_vae

        tokenized_agent["train_mask"] = all_valid

        train_mask=train_mask[all_valid]

        if self.iq_learn:
            if self.use_gail and not self.use_distance:
                expert_dis_loss, expert_rewards, expert_returns,expert_dis_feat=self.get_reward(tokenized_agent,None,None,"expert",None)
                if self.encoder.agent_encoder.pred_col:
                    col_loss=self.get_collision_loss(tokenized_agent,tokenized_map,expert_dis_feat,train_mask,all_valid,'expert')

                    expert_nll=expert_nll+col_loss

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

                noised_pos= tokenized_agent["sampled_pos"]
                noised_heading=tokenized_agent["sampled_heading"]

                pos_local, heading_local=transform_to_local(pos.reshape(-1,1,2),heading.reshape(-1,1),noised_pos.reshape(-1,2),noised_heading.reshape(-1))

                pos_noise=pos_local.reshape(pos.shape)
                heading_noise=heading_local.reshape(heading.shape)

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

                pred_contour=cal_polygon_contour(pred_pos, pred_heading, token_agent_shape)

                real_contour=cal_polygon_contour(pos[all_valid][:,2:], heading[all_valid][:,2:], token_agent_shape)

                noise_error=torch.linalg.norm(pred_contour-real_contour,dim=-1).mean()

                real_noise=torch.cat([pos_noise,heading_noise[:,:,None]],dim=-1)[all_valid][:,2:]

                pos_error=torch.linalg.norm(noise_pred[:,:,:2]-real_noise[:,:,:2],dim=-1).mean()
                heading_error=wrap_angle(noise_pred[:,:,2]-real_noise[:,:,2]).abs().mean()

                self.log("train/expert_pos_loss", pos_error.item(), on_step=True, batch_size=1)
                self.log("train/expert_heading_loss", heading_error.item(), on_step=True, batch_size=1)

            tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent)

            if self.encoder.agent_encoder.pred_light:
                eval_light(expert_light_idx, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            #tokenized_agent_rollout["train_mask"]=None

            agent_reward, agent_value_loss, agent_pi, agent_nll,agent_Q,agent_proposal_loss,agent_log_prob,agent_entropy = self.get_QV(
                tokenized_map, tokenized_agent_rollout, None,key='agent')

            #tokenized_agent_rollout["train_mask"]=all_valid

            if self.use_gail:
                if not self.use_distance:

                    agent_dis_loss, agent_rewards, agent_returns, agent_disc_feat = self.get_reward(tokenized_agent_rollout, agent_log_prob,agent_pi, "agent",None,tokenized_map=tokenized_map)
                    critic_loss=expert_dis_loss + agent_dis_loss

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

                if self.encoder.agent_encoder.pred_col:
                    col_loss=self.get_collision_loss(tokenized_agent_rollout,tokenized_map,agent_disc_feat,None,all_valid,'agent')

                    expert_nll=expert_nll+col_loss

                if self.encoder.agent_encoder.use_infogail:

                    logits = self.encoder.RecognitionQ.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                                     tokenized_agent_rollout["goal_idx"],
                                                                     tokenized_agent_rollout["valid_mask"],
                                                                     tokenized_agent_rollout["sampled_pos"],
                                                                     tokenized_agent_rollout["sampled_heading"],
                                                                     tokenized_agent_rollout,
                                                                     tokenized_agent_rollout["detach_map_feature"],
                                                                     tokenized_agent_rollout["light_idx"],
                                                                     None)[0]#[all_valid]

                    latent_z = tokenized_agent_rollout["latent_z"][all_valid]

                    if logits.shape[-1]==self.encoder.agent_encoder.k_dim:
                        log_q = F.log_softmax(logits, dim=-1)
                        action=latent_z[:,:,None].repeat(1,log_q.shape[1],1)
                        z_logp = torch.gather(log_q, dim=-1, index=action).squeeze(-1)  #larger z likelihood # [B, Tm1, T_a]
                        kl_prior=0
                    else:
                        mu = logits[:,:, :self.encoder.agent_encoder.k_dim]
                        logvar = logits[:,:, self.encoder.agent_encoder.k_dim:]

                        z = latent_z.expand_as(mu)  # [B, T, k_dim]
                        std = torch.exp(0.5 * logvar)

                        base = Normal(loc=mu, scale=std)
                        dist = Independent(base, reinterpreted_batch_ndims=1)  # event dim = last

                        z_logp = dist.log_prob(z)  # shape: [...]

                        mu_p = torch.zeros_like(mu)
                        logvar_p = torch.zeros_like(logvar)

                        var_q = logvar.exp()
                        var_p = logvar_p.exp()

                        kl_prior = 0.5 * (logvar_p - logvar + (var_q + (mu - mu_p).pow(2)) / var_p - 1 ).sum(-1) .mean()
                        self.log("train/mu", mu.mean().item(), on_step=True, batch_size=1)
                        self.log("train/std", std.mean().item(), on_step=True, batch_size=1)

                    loss_q=-z_logp.mean() # increase the z likelihood
                    expert_nll=expert_nll+loss_q+kl_prior

                    self.log("train/loss_q", loss_q.item(), on_step=True, batch_size=1)
                    mi_beta=0.1
                    agent_rewards=agent_rewards+mi_beta * z_logp.detach()

                    # # Optional entropy regularizer on P to avoid overconfidence
                    # p_probs = log_p.exp()
                    # H_p = -(p_probs * log_p).sum(-1).mean()
                    #
                    # tau=0.01
                    #
                    # loss_prior = kl_qp - tau * H_p  # tau ~ 0.01–0.1

                    #@self.log("train/loss_prior", loss_prior.item(), on_step=True, batch_size=1)

                if self.encoder.use_value:
                    feat_a=tokenized_agent_rollout["feat_a_nodetach"][all_valid]

                    if self.encoder.discriminator.interative_decoder.centric:
                        index=tokenized_agent_rollout["batch"][all_valid][:,None].repeat(1,feat_a.shape[1])
                        feat_a, argmax = scatter_max(feat_a, index, dim=0)  # out: [B,T,C]

                    value_pred=self.encoder.value_network(feat_a)[:,:,0]#

                    agent_rewards=per_scene_zscore_clip(agent_rewards,tokenized_agent_rollout["batch"][all_valid],torch.ones_like(agent_rewards).to(bool))

                    ego_advantages,returns=compute_advantages(agent_rewards,value_pred.detach(),None,gamma=self.gamma)#[all_valid]

                    value_loss = torch.pow(returns - value_pred, 2.0).mean()#.clamp(min=0,max=100)

                    if self.use_lcf:
                        nei_rewards = get_near_returns(tokenized_agent, agent_rewards,train_mask=all_valid,neighbor_dist=60.0)

                        nei_value_pred=self.encoder.nei_value_network(tokenized_agent_rollout["feat_a"][~all_valid])[:,:,0]

                        nei_advantages,nei_returns=compute_advantages(nei_rewards,nei_value_pred.detach(),None,gamma=self.gamma)

                        nei_value_loss = torch.pow(nei_returns - nei_value_pred, 2.0).clamp(min=0,max=100).mean()#

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

                            step_lcf=torch.clamp(torch.tanh(lcf_parameters[...,0]), -1 + 1e-6, 1 - 1e-6)

                            self.log("train/lcf_mean", step_lcf.mean().item(), on_step=True, batch_size=1)
                            self.log("train/lcf_std", step_lcf.std().item(), on_step=True, batch_size=1)
                        else:
                            step_lcf=torch.tensor(0.5)

                        advantages=torch.zeros_like(agent_log_prob)

                        advantages[all_valid]=ego_advantages
                        advantages[~all_valid]=nei_advantages
                        # used_lcf = step_lcf.detach() * np.pi / 2
                        #
                        # advantages=  torch.cos(used_lcf) * ego_advantages + torch.sin(used_lcf) *nei_advantages

                    else:
                        if self.encoder.discriminator.interative_decoder.centric:
                            advantages=ego_advantages[index[:,0]]
                        else:
                            advantages=ego_advantages

                    self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

                else:
                    advantages=agent_returns[all_valid]
                    value_loss=0

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

                expert_nll = expert_nll + agent_wNLL +0.001* value_loss #-0.1*agent_density.mean()  # - 0.01 * agent_entropy.mean()

                # if self.use_kl_penalty:
                #     with torch.no_grad():
                #         #map_feature=self.bc_map_net(tokenized_map)
                #         target_q = self.bc_net(tokenized_agent_rollout, tokenized_agent_rollout["detach_map_feature"])[ "agent_q"]
                #         ref_logprobs = (torch.softmax(target_q / self.alpha, dim=-1)+1e-10).log()
                #
                #     kl_coef=1
                #
                #     kl_per_token =  torch.sum(agent_pi *( (agent_pi+1e-10).log() - ref_logprobs), dim=-1).mean()
                #
                #     self.log("train/kl_penalty", kl_per_token.item(), on_step=True, batch_size=1)
                #
                #     expert_nll=expert_nll+kl_coef *kl_per_token
            else:
                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)
                # constraint_loss=expert_V_diff.square().mean()*5
                #
                # self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_nll

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
            loss = expert_nll

        return loss

    def training_step(self, data, batch_idx):

        tokenized_map, tokenized_agent = self.token_processor(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q :
            soft_update(self.encoder.critic, self.target_net, tau = 2e-4)

        return loss

