from jax.example_libraries.stax import logsoftmax
from lightning import LightningModule
import numpy as np
import torch

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

        #self.automatic_optimization=False

        if self.encoder.agent_encoder.use_vae:

            self.l_vae_kl = BalancedKL(kl_balance_scale=0.2, kl_free_nats=1.0)

        # self.lcf_parameters = torch.nn.Parameter(torch.as_tensor(lcf_parameters), requires_grad=True)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Unused parameter: {name}")
    #

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

            proposal_loss,proposal_log_prob,pos_dist, head_diff = get_proposal_loss(pred["proposal"], tokenized_agent,
                                                                                self.start_step)

            if key=="expert":
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
            proposal_log_prob=0

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

        return  reward,value_loss,pi,action_nll+light_nll+proposal_loss,current_Q,proposal_loss,log_prob+proposal_log_prob,entropy

    def get_reward(self,tokenized_agent,agent_log_prob,agent_pi,key,train_mask=None,expert_disc_val=0,tokenized_map=None):

        sampled_pos=tokenized_agent["sampled_pos"]#torch.round(tokenized_agent["sampled_pos"]*10)/10##
        sampled_heading=tokenized_agent["sampled_heading"]#torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#
        # sampled_heading=torch.round(wrap_angle(tokenized_agent["sampled_heading"])/10)*10#tokenized_agent["sampled_heading"]#

        disc_out= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        tokenized_agent["goal_idx"],
                                                        tokenized_agent["valid_mask"],#expert_
                                                        sampled_pos,
                                                        sampled_heading ,
                                                        tokenized_agent,
                                                        tokenized_agent["detach_map_feature"],
                                                        tokenized_agent["light_idx"],
                                                        None,
                                                       # latent_z=tokenized_agent["latent_z"]
                                                        )#[0]#Metrics-Guided Adversarial Training

        logit=disc_out[0]
        if not self.encoder.discriminator.interative_decoder.diff_dicriminator:
            disc_val = torch.sigmoid(logit[:, :, 0])

        if key == "agent" and self.use_kl_penalty:
            with torch.no_grad():
                if self.bc_map_net is not None:
                    map_feature=self.bc_map_net(tokenized_map)
                else:
                    map_feature = tokenized_agent["detach_map_feature"]

                target_q = self.bc_net(tokenized_agent, map_feature)["agent_q"]

                logp_ref = (torch.softmax(target_q / self.alpha, dim=-1)+1e-10).log()

                actions=tokenized_agent["sampled_idx"][:,2:][train_mask]

                logp_a_ref=torch.gather(logp_ref, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

                kl_penalty =  torch.sum(agent_pi *( (agent_pi+1e-10).log() - logp_ref), dim=-1).mean()  # (B,T)

                self.log("train/kl_penalty", kl_penalty.item(), on_step=True, batch_size=1)

                kl_coef=0.1#np.power(0.9999,self.global_step)
                kl_taken = (agent_log_prob - logp_a_ref)

                kl_per_token=kl_coef *kl_taken

        else:
            kl_per_token=0

        rewards,nei_sum_rewards = disc_out[2]#.detach()

        weight= disc_out[3]

        rewards=rewards+kl_per_token

        # nei_rewards=get_nei_returns(tokenized_agent,rewards,train_mask=train_mask)
        #
        # rewards = 0.5 * rewards + 0.5 * nei_rewards

        returns = get_return(rewards, self.gamma)

        # with torch.no_grad():
        #     if self.encoder.discriminator.interative_decoder.use_edge_feature.:
        #         rewards=disc_out[2]
        #     else:
        #         disc_val_eval=disc_val
        #         if  self.dis_loss == "wgan":
        #             rewards=logit[:, :, 0].detach()
        #         else:
        #             rewards=get_reward(disc_val_eval,kl_per_token=kl_per_token)

        if  self.use_lcf and not self.encoder.use_value:
            with torch.no_grad():
                batch = tokenized_agent["batch"]
                global_rewards=scatter_mean(rewards,batch,dim=0)
                self.log("train/" + key + "_global_rewards", global_rewards.mean().item(), on_step=True, batch_size=1)
                nei_returns=get_nei_returns(tokenized_agent,returns)
                self.log("train/" + key + "_nei_returns", nei_returns.mean().item(), on_step=True, batch_size=1)
                ego_returns=(returns-returns.mean())/(returns.std()+1e-4)
                nei_returns=(nei_returns-nei_returns.mean())/(nei_returns.std()+1e-4)

                returns=0.5*ego_returns+0.5*nei_returns

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
            # if train_mask is not None and not self.encoder.discriminator.interative_decoder.centric:
            #     disc_val=disc_val[train_mask]
            agent_num=rewards.shape[0]*rewards.shape[1]
            ego_dis_eval=disc_val[:agent_num]
            other_disc_val=disc_val[agent_num:]
            if key == "expert":

                bce_loss =F.binary_cross_entropy(ego_dis_eval, torch.ones_like(ego_dis_eval), weight=None, reduction='mean')
                if len(other_disc_val)>0:
                    bce_loss =bce_loss+F.binary_cross_entropy(other_disc_val, torch.ones_like(other_disc_val), weight=weight, reduction='mean')
            else:
                bce_loss =F.binary_cross_entropy(ego_dis_eval, torch.zeros_like(ego_dis_eval), weight=None, reduction='mean')
                if len(other_disc_val)>0:
                    bce_loss =bce_loss+F.binary_cross_entropy(other_disc_val, torch.zeros_like(other_disc_val), weight=weight, reduction='mean')

        self.log("train/"+key+"_dis_loss", bce_loss, on_step=True, batch_size=1)
        self.log("train/"+key+"_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_return", returns.mean().item(), on_step=True, batch_size=1)
        self.log("train/"+key+"_rewards", rewards.mean().item(), on_step=True, batch_size=1)

        # if "a2a_entropy" in tokenized_agent.keys():
        #     a2a_entropy=disc_out[2].mean()#tokenized_agent["a2a_entropy"].mean()
        #     self.log("train/" + key + "_a2a_entropy", a2a_entropy.item(), on_step=True, batch_size=1)

            # bce_loss=bce_loss+0.01*a2a_entropy
        #
        # if  key == "expert":
        #     expert_pos=tokenized_agent["sampled_pos"]#tokenized_agent["expert_sampled_pos"]#
        #     expert_sampled_heading=tokenized_agent["sampled_heading"]#tokenized_agent["expert_sampled_heading"]#
        #     expert_valid_mask=tokenized_agent["valid_mask"]#tokenized_agent["expert_valid_mask"]#
        #     pos=tokenized_agent["sampled_pos"]
        #     heading=tokenized_agent["sampled_heading"]
        #
        #     batch_idx = tokenized_agent['batch']
        #     alpha = torch.rand(size=(max(batch_idx) + 1, 1), device=batch_idx.device)
        #
        #     alpha=alpha[batch_idx]
        #
        #     #alpha= torch.rand((pos.size(0), pos.size(1)), device=pos.device)
        #     interpolate_pos = alpha[:,:,None] * expert_pos + (1 - alpha[:,:,None]) * pos
        #     interpolate_heading =alpha * expert_sampled_heading + (1 - alpha) * heading
        #
        #     interpolates_pos=torch.cat((interpolate_pos, interpolate_heading[:,:,None]), dim=-1)
        #
        #     interpolates=interpolates_pos[expert_valid_mask]#[train_mask,2:]
        #
        #     interpolates.requires_grad_(True)  # IMPORTANT
        #
        #     interpolates_pos[expert_valid_mask]=interpolates
        #
        #     scores= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
        #                                                     tokenized_agent["goal_idx"],
        #                                                     expert_valid_mask,
        #                                                     interpolates_pos[:,:,:2],
        #                                                     interpolates_pos[:,:,2] ,
        #                                                     tokenized_agent,
        #                                                     tokenized_agent["detach_map_feature"],
        #                                                     tokenized_agent["light_idx"],
        #                                                     None)[0]
        #     score_sum = scores.view(-1).sum()
        #
        #     gradients = torch.autograd.grad(
        #         outputs=score_sum,
        #         inputs=interpolates,
        #         create_graph=True,
        #         retain_graph=True,
        #         only_inputs=True,
        #     )[0]  # shape: [B, T, 3]
        #
        #     # gp = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        #     gp=gradients.pow(2).sum(dim=-1).mean()
        #     # print(gp)
        #
        #     self.log("train/gp", gp, on_step=True, batch_size=1)
        #
        #     bce_loss=gp* 10+bce_loss

        return bce_loss,rewards,nei_sum_rewards, disc_val#torch.sigmoid(logit[:,:,-1]) #-0.03*entropy

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, self.start_step:]
        train_mask = valid_mask[:, 1:] &  valid_mask[:, :-1]
        tokenized_agent["vis_mask"] = None

        if "pred_mask" in tokenized_agent.keys():
            all_valid=tokenized_agent["pred_mask"] & valid_mask.all(-1)
        else:
            all_valid=valid_mask.all(-1)

        if self.use_kl_penalty:
            expert_nll=0
            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["detach_map_feature"] = {k: v.detach() for k, v in map_feature.items()}
        else:
            if self.iq_learn and self.encoder.agent_encoder.use_roformer:
                self.encoder.agent_encoder.a_t_roformer.attn.caching = True
                if self.encoder.agent_encoder.pred_light and not self.encoder.agent_encoder.light_encoder.share:
                    self.encoder.agent_encoder.light_encoder.lg_t_roformer.attn.caching = True

            expert_reward,expert_value_loss,expert_pi,expert_nll,expert_Q,expert_proposal_loss,expert_log_prob,_ = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        # if "a2a_entropy" in tokenized_agent.keys():
        #     a2a_entropy=tokenized_agent["a2a_entropy"].mean()
        #     self.log("train/expert_a2a_ent", a2a_entropy.item(), on_step=True, batch_size=1)
        #
        #     expert_nll=expert_nll+0.1*a2a_entropy


        if self.encoder.agent_encoder.use_vae:
            latent_post=tokenized_agent["latent_post"]
            latent_prior=tokenized_agent["latent_prior"]

            log_q = F.log_softmax(latent_post, dim=-1)
            log_p = F.log_softmax(latent_prior, dim=-1)
            q = log_q.exp()
            error_vae = (q * (log_q - log_p)).sum(dim=-1).mean()

            # kl_reduced, kl_per_item = self.l_vae_kl.kl_diag_gaussians(latent_post, latent_prior)#.mean()
            # free_nats = 0.0
            # error_vae =  torch.clamp(kl_per_item, min=free_nats).mean()  # or simply: beta * kl_mean
            self.log("train/error_vae", error_vae.item(), on_step=True, batch_size=1)

            # post_std = torch.exp(0.5 * latent_post[1])
            # prior_std = torch.exp(0.5 * latent_prior[1])
            #
            # self.log("train/post_std", post_std.mean().item(), on_step=True, batch_size=1)
            # self.log("train/prior_std", prior_std.mean().item(), on_step=True, batch_size=1)

            expert_nll=expert_nll+error_vae

        tokenized_agent["train_mask"] = all_valid

        #train_mask=train_mask[all_valid]

        if self.iq_learn:
            if self.use_gail and not self.use_distance:

                # with torch.no_grad():
                #     expert_Value=self.encoder.value_network(tokenized_agent["feat_a_nodetach"][all_valid])[:,:,0]
                #
                #     self.log("train/expert_value", expert_Value.mean().item(), on_step=True, batch_size=1)


                expert_dis_loss, expert_rewards, expert_returns,expert_dis_feat=self.get_reward(tokenized_agent,None,None,"expert",all_valid)
                if self.encoder.agent_encoder.pred_col:
                    col_loss=self.get_collision_loss(tokenized_agent,tokenized_map,expert_dis_feat,None,all_valid,'expert')

                    expert_nll=expert_nll+col_loss

            expert_light_idx=tokenized_agent["light_idx"].clone()

            #if self.dis_loss=="wgan":
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

            # if "a2a_entropy" in tokenized_agent_rollout.keys():
            #     a2a_entropy = tokenized_agent_rollout["a2a_entropy"].mean()
            #     self.log("train/agent_a2a_ent", a2a_entropy.item(), on_step=True, batch_size=1)
            #
            #     expert_nll = expert_nll + 0.1 * a2a_entropy

            #expert_nll=expert_nll-0.001*agent_entropy

            #tokenized_agent_rollout["train_mask"]=all_valid

            if self.use_gail:
                if self.buffer_len>1:
                    with torch.no_grad():
                        agent_dis_loss, agent_rewards, agent_returns, agent_disc_feat = self.get_reward(
                            tokenized_agent_rollout, agent_log_prob, agent_pi, "agent", None)

                    if self.global_step%2==0:
                        current_rollout={}

                        for key in {"sampled_idx","goal_idx","valid_mask","sampled_heading","sampled_pos","detach_map_feature","light_idx","type","shape","batch","num_graphs","train_mask"}:
                            current_rollout[key]=tokenized_agent_rollout[key]

                        self.replay_buffer.append(current_rollout)

                    old_rollout = random.sample(self.replay_buffer, 1)[0]
                    agent_dis_loss, _, _, _ = self.get_reward(
                        old_rollout, None, None, "agent", None)
                else:
                    agent_dis_loss, agent_rewards, nei_rewards, agent_disc_feat = self.get_reward(tokenized_agent_rollout, agent_log_prob,agent_pi, "agent",all_valid,tokenized_map=tokenized_map)

                critic_loss=expert_dis_loss + agent_dis_loss


                if self.encoder.agent_encoder.pred_col:
                    col_loss=self.get_collision_loss(tokenized_agent_rollout,tokenized_map,agent_disc_feat,None,all_valid,'agent')

                    expert_nll=expert_nll+col_loss

                if self.encoder.use_value:
                    feat_a=tokenized_agent_rollout["feat_a_nodetach"][all_valid]

                    if self.encoder.discriminator.interative_decoder.centric:
                        index=tokenized_agent_rollout["batch"][all_valid][:,None].repeat(1,feat_a.shape[1])
                        feat_a, argmax = scatter_max(feat_a, index, dim=0)  # out: [B,T,C]

                    # agent_rewards=per_scene_zscore_clip(agent_rewards,tokenized_agent_rollout["batch"][all_valid],torch.ones_like(agent_rewards).to(bool))
                    #agent_rewards = torch.round(agent_rewards / 0.02).clip(-10.0,10.0)/10.0 #.floor().long()

                    #agent_rewards[:,1:]= agent_rewards[:,1:]-agent_rewards[:,:-1]

                    v_denorm=self.encoder.value_network(feat_a)[:,:,0]

                    self.log("train/agent_value", v_denorm.mean().item(), on_step=True, batch_size=1)

                    # agent_rewards = (agent_rewards-torch.mean(agent_rewards,dim=1,keepdim=True))/(torch.std(agent_rewards,dim=1,keepdim=True))
                    #agent_rewards = torch.clamp(agent_rewards, -2, 2)
                    ego_advantages,gae_returns=compute_advantages(agent_rewards,v_denorm.detach(),None,gamma=self.gamma)#[all_valid]

                    # with torch.no_grad():
                    #     v_denorm, _=self.encoder.value_network(feat_a)
                    #     ego_advantages,gae_returns=compute_advantages(agent_rewards,v_denorm.detach(),None,gamma=self.gamma)#[all_valid]
                    #
                    #     self.encoder.value_network.update_stats_and_rescale(agent_returns.reshape(-1))
                    #     y_norm = (gae_returns - self.encoder.value_network.mu) / (self.encoder.value_network.sigma + self.encoder.value_network.eps)

                    # normalized target

                    # _, v_norm=self.encoder.value_network(feat_a)
                    # value_loss = F.mse_loss(v_norm, y_norm)  # train on normalized target

                    value_loss = torch.pow(gae_returns - v_denorm, 2.0).clamp(min=0,max=100).mean()#

                    advantages=ego_advantages

                    if self.use_lcf:
                        #nei_rewards = get_nei_returns(tokenized_agent, agent_rewards,train_mask=all_valid)

                        nei_value_pred=self.encoder.nei_value_network(tokenized_agent_rollout["feat_a_nodetach"][all_valid])[:,:,0]

                        nei_advantages,nei_returns=compute_advantages(nei_rewards,nei_value_pred.detach(),None,gamma=self.gamma)

                        nei_value_loss = torch.pow(nei_returns - nei_value_pred, 2.0).clamp(min=0,max=100).mean()

                        value_loss = nei_value_loss + value_loss

                        advantages=  0.5 * ego_advantages + 0.5 *nei_advantages


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

                gail_weight=1#-np.power(0.9999,self.global_step)expert_nll +

                expert_nll = gail_weight*agent_wNLL +1e-3* value_loss #- 0.01 * agent_entropy.mean()
            else:
                critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+expert_nll
            if self.automatic_optimization == False:
                policy_optimizer, discriminator_optimizer = self.optimizers()

                # print(agent_rewards.mean())

                discriminator_optimizer.zero_grad()
                self.manual_backward(critic_loss)
                discriminator_optimizer.step()

            if self.automatic_optimization == False:
                policy_optimizer.zero_grad()
                self.manual_backward(expert_nll)
                policy_optimizer.step()
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

