from lightning import LightningModule
import random
from collections import deque
import numpy as np

from src.smart.modules.smart_decoder import SMARTDecoder

import torch

from src.smart.metrics.utils import get_euclidean_targets
from src.smart.loss.gmm_dist import  GMM_Dist,get_entropy
from src.smart.loss.iq_loss import get_iqloss,soft_update,get_return,eval_light
from src.smart.loss.rollout_buffer import rollout

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.rollout_freq=1
        self.gamma = 0.99
        self.iq_learn=self.encoder.iq_learn
        self.output_gmm=self.encoder.output_gmm
        self.alpha = self.encoder.alpha
        self.n_token_agent=self.encoder.agent_encoder.n_token_agent
        self.batch_replay=False

        # if self.batch_replay:
        #     self.replay_buffer = deque(maxlen=4000)
        # else:
        #     self.replay_buffer = deque(maxlen=1)

        if self.iq_learn and self.output_gmm:
            self.automatic_optimization = False
            
        self.use_target_q=False

        if  self.use_target_q:
            self.target_net = SMARTDecoder(
                **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
            )
            self.target_net.load_state_dict(self.encoder.state_dict())


    def get_network_QV(self,all_q_value,tokenized_map, tokenized_agent,action,key):

        # pred = network(tokenized_map, tokenized_agent)

        # q_value = pred["q_value"][:,:,0]
        q_value=all_q_value[:,:,0]

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
            action = action.unsqueeze(-1).long()  # .reshape(-1)

            q = q_value[:, :-1]

            if self.encoder.agent_encoder.pred_res:
                traj=q[:,:,self.n_token_agent:]
                q=q[:,:,:self.n_token_agent]

            current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

            v_value =  self.alpha * torch.logsumexp(q_value / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

            current_V = v_value[:, :-1]

            next_V = v_value[:, 1:]

            pi = torch.softmax( q / self.alpha, dim=-1)

            logpi= torch.log(pi+ 1e-10)#.clamp_min(min=1e-10)

            log_pi_stack=torch.log_softmax(all_q_value[:, :-1]/ self.alpha, dim=-1)

            rolling_action = torch.stack([
                        torch.roll(action, shifts=-i, dims=1)
                        for i in range(log_pi_stack.shape[2])
                    ], dim=-2)  # [B, Tm1, T_a]

            log_prob1=torch.gather(log_pi_stack, dim=-1, index=rolling_action).squeeze(-1)

            valid_mask=torch.ones_like(log_prob1)
            for i in range(log_pi_stack.shape[2]):
                log_prob1[:,rolling_action.shape[1]-i:,i]=0
                valid_mask[:,rolling_action.shape[1]-i:,i]=0

            log_prob=log_prob1.sum(-1)/valid_mask.sum(-1)
            # act=action.reshape(-1)
            # log_prob=logpi.reshape(len(act), -1)[torch.arange(len(act)), act].reshape(q.shape[0], q.shape[1])

            #log_prob=torch.gather(logpi, dim=-1, index=action).squeeze(-1)
            entropy = -torch.sum(pi * logpi, dim=-1)

            if self.encoder.agent_encoder.pred_res and key=="expert":
                actor_loss = torch.abs(traj-tokenized_agent["target"][:,2:]).mean(-1)
            else:
                actor_loss=self.alpha * log_prob - current_Q

        dones = torch.zeros_like(next_V)

        dones[:, -1] = 1

        y=self.gamma *(1 - dones) * next_V

        reward = current_Q - y
        value_loss = current_V - y

        return log_prob,logpi,actor_loss,entropy,current_Q,v_value,value_loss,reward

    def get_QV(self, tokenized_map, tokenized_agent,train_mask, key='expert'):
        action = tokenized_agent["sampled_idx"][:, 2:]
        valid_mask = tokenized_agent["valid_mask"][:, 1:]
        agent_num = len(action)

        all_valid_mask=valid_mask[:agent_num].all(-1)#train_mask #

        pred = self.encoder(tokenized_map, tokenized_agent)

        log_prob,logpi,actor_loss,entropy, current_Q, V,  value_loss, reward=self.get_network_QV(pred["agent_q"], tokenized_map, tokenized_agent,action,key)

        current_Q_diff, V_diff = get_return(reward,log_prob,current_Q,V,all_valid_mask,self.alpha,self.gamma)

        if self.encoder.agent_encoder.pred_light:
            light_action=torch.clamp_max(tokenized_agent["light_idx"][:, 2:],max=self.token_processor.light_type-1)

            log_prob_light,light_logpi=self.get_network_QV(pred["light_q"], tokenized_map, tokenized_agent,light_action,key)[:2]

            light_pred = torch.argmax(light_logpi, dim=-1)
            real_light = tokenized_agent["light_idx"][:, 2:]
            light_acc = (light_pred == real_light)[train_mask[agent_num:]]
            self.log("train/" + key + "_light_acc", light_acc.float().mean().item(), on_step=True, batch_size=1)
            log_prob=torch.cat([log_prob,log_prob_light],dim=0)

        action_nll = -log_prob[train_mask].mean()

        if self.use_target_q:
            with torch.no_grad():
                target_V = self.get_network_QV(self.target_net, tokenized_map, tokenized_agent,action,key)[4]

        init_V = V[:, 0]
        last_V= V[:,-1]

        actor_loss = actor_loss[all_valid_mask]

        reward = reward[all_valid_mask]

        value_loss=value_loss[all_valid_mask]

        V=V[all_valid_mask]

        current_Q=current_Q[all_valid_mask]

        entropy =entropy[all_valid_mask]

        init_V=init_V[all_valid_mask]

        last_V=last_V[all_valid_mask]

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


        return  reward,value_loss,init_V,action_nll,actor_loss

    def iq_update(self, tokenized_map, tokenized_agent):
        valid_mask= tokenized_agent["valid_mask"][:, 1:]
        col_mask = tokenized_agent["col_mask"][:, 1:]
        state_mask=valid_mask[:,:-1]
        action_mask=valid_mask[:,1:] & (~col_mask[:,1:])

        train_mask=state_mask & action_mask

       # train_mask=valid_mask.all(-1)

        expert_reward,expert_value_loss,expert_V_diff,expert_nll,expert_actor_loss = self.get_QV(tokenized_map, tokenized_agent,train_mask)

        self.log("train/expert_nll", expert_nll.item(), on_step=True, batch_size=1)

        if not self.iq_learn:
            if self.encoder.agent_encoder.pred_res:
                loss=expert_actor_loss.mean()+expert_nll
            else:
                loss =expert_nll
        else:
            if self.global_step % self.rollout_freq== 0:
                tokenized_map_rollout, tokenized_agent_rollout = rollout(self.encoder, tokenized_map, tokenized_agent)
            else:
                tokenized_map_rollout, tokenized_agent_rollout =random.sample(self.replay_buffer,1)[0]

            if self.encoder.agent_encoder.pred_light:
                eval_light(tokenized_agent, tokenized_agent_rollout, self.log, self.encoder.agent_encoder.light_type)

            agent_reward, agent_value_loss, agent_V_diff, _,agent_actor_loss = self.get_QV(
                tokenized_map_rollout, tokenized_agent_rollout, train_mask,key='agent')

            critic_loss=get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss)

            self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

            constraint_loss=expert_V_diff.square().mean() #*5

            self.log("train/constraint_loss", constraint_loss.item(), on_step=True, batch_size=1)

            loss = critic_loss+constraint_loss#critic_loss+constraint_loss #expert_nll #-0.01*agent_entropy.mean() #expert_nll+expert_nll+expert_nll+.square().square()expert_nll++(expert_target_loss+agent_target_loss) # #*0.1

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

        #print(loss)

        return loss

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

        tokenized_map, tokenized_agent = self.token_processor(data)

        loss = self.iq_update(tokenized_map, tokenized_agent)

        self.log("train/loss", loss, on_step=True, batch_size=1)

        if self.use_target_q :
            soft_update(self.encoder.agent_encoder, self.target_net.agent_encoder, tau = 1e-4)

        return loss
