from lightning import LightningModule

import torch.optim as optim
import random
from collections import deque
import torch.nn as nn
from tensorflow_probability.substrates.jax.distributions.student_t import entropy
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
from torch_geometric.profile.benchmark import require_grad

from src.smart.model.rollout_buffer import ReplayBuffer
from torch.autograd import Variable, grad
import numpy as np

class IQ_SoftQ(LightningModule):

    def __init__(self, model_config) -> None:
        super(IQ_SoftQ, self).__init__(model_config)

        self.replay_buffer = deque(maxlen=100)
        self.critic_tau=0.001
        self.critic_target_update_frequency=4
        self.gamma = 0.99
        self.alpha = 1

        self.reg_mult=0.5

        self.Q_max = 1.0 / (self.reg_mult * (1 - self.gamma))
        self.Q_min = - 1.0 / (self.reg_mult * (1 - self.gamma))

    def rollout(self,tokenized_map,tokenized_agent):
        pred = self.encoder.inference(
            tokenized_map,
            tokenized_agent,
            sampling_scheme=self.training_rollout_sampling,
        )

        tokenized_agent_rollout={}

        tokenized_agent_rollout['sampled_pos'] = pred["pred_pos"]
        tokenized_agent_rollout['sampled_heading'] = pred['pred_head']
        tokenized_agent_rollout['sampled_idx'] = pred["pred_idx"]
        tokenized_agent_rollout['valid_mask'] = tokenized_agent["valid_mask"]
        tokenized_agent_rollout['trajectory_token_veh'] = tokenized_agent['trajectory_token_veh']
        tokenized_agent_rollout['trajectory_token_ped'] = tokenized_agent['trajectory_token_ped']
        tokenized_agent_rollout['trajectory_token_cyc'] = tokenized_agent['trajectory_token_cyc']
        tokenized_agent_rollout['type'] = tokenized_agent['type']
        tokenized_agent_rollout['shape'] = tokenized_agent['shape']
        tokenized_agent_rollout['batch'] = tokenized_agent['batch']
        tokenized_agent_rollout['num_graphs'] = tokenized_agent['num_graphs']
        # tokenized_agent_rollout['next_route'] = tokenized_agent['next_route']
        # tokenized_agent_rollout['light'] = tokenized_agent['light']

        self.replay_buffer.append((tokenized_map,tokenized_agent_rollout))

    def get_QV(self,tokenized_map, tokenized_agent,agent=False):

        q_value=self.encoder(tokenized_map, tokenized_agent)["q_value"]

        q =q_value[:,:-1]
        current_V = self.alpha * torch.logsumexp(q / self.alpha, dim=-1, keepdim=False)#V=Q-alpha*H

        pi = torch.softmax(q / self.alpha, dim=-1)  # Compute policy
        entropy=-torch.sum(pi * torch.log(pi + 1e-10), dim=-1)

        action= tokenized_agent["sampled_idx"][:,2:].reshape(-1)

        # if agent:
        #     current_Q= torch.sum(pi * q, dim=-1)
        # else:
        current_Q =  q.reshape(len(action),-1)[torch.arange(len(action)), action].reshape(q.shape[0],q.shape[1])

        with torch.no_grad():
            next_q = self.target_net(tokenized_map, tokenized_agent)["q_value"][:,1:]
            target_v = self.alpha * torch.logsumexp(next_q/self.alpha, dim=-1, keepdim=False)

        self._target_clipping=True

        if self._target_clipping:
             target_v = torch.clip(target_v, self.Q_min, self.Q_max)

        return current_Q,current_V,target_v,entropy

    def iq_update(self,tokenized_map, tokenized_agent,alpha=0.5):

        expert_Q,expert_V,expert_target_v,expert_entropy=self.get_QV(tokenized_map,tokenized_agent)

        done=torch.zeros_like(expert_target_v)

        done[:,-1]=1

        expert_reward = expert_Q - (1 - done) *self.gamma * expert_target_v

        valid_mask=tokenized_agent["valid_mask"]

        expert_valid=valid_mask[:,2:] & valid_mask[:,1:-1] &valid_mask[:,:-2]

        expert_reward_loss =expert_reward[expert_valid].mean()

        self.log("train/expert_reward", expert_reward[expert_valid].mean().item(), on_step=True, batch_size=1)

        self.log("train/expert_reward_loss", expert_reward_loss.item(), on_step=True, batch_size=1)

        expert_entropy_loss=expert_entropy[expert_valid].mean()

        self.log("train/expert_entropy", expert_entropy_loss.item(), on_step=True, batch_size=1)

        expert_Q_loss=F.mse_loss(expert_Q[expert_valid],torch.ones_like(expert_Q[expert_valid])*self.Q_max)

        self.log("train/expert_Q_loss", expert_Q_loss.item(), on_step=True, batch_size=1)

        agent_tokenized_map, agent_tokenized_agent = random.sample(self.replay_buffer,1)[0]

        agent_Q,agent_V,agent_target_v,agent_entropy=self.get_QV(agent_tokenized_map, agent_tokenized_agent)

        done=torch.zeros_like(agent_target_v)

        done[:,-1]=1

        agent_reward=agent_Q - (1 - done) *self.gamma * agent_target_v

        valid_mask=agent_tokenized_agent["valid_mask"]

        agent_valid=valid_mask[:,2:] & valid_mask[:,1:-1] &valid_mask[:,:-2]

        self.log("train/agent_reward", agent_reward[agent_valid].mean().item(), on_step=True, batch_size=1)

        entropy_loss=agent_entropy[agent_valid].mean()

        self.log("train/agent_entropy", entropy_loss.item(), on_step=True, batch_size=1)

        expert_value_loss=(expert_V-expert_target_v)[expert_valid].mean()

        agent_value_loss=(agent_V-agent_target_v)[agent_valid].mean()

        value_loss=(expert_value_loss*expert_valid.sum()+agent_value_loss*agent_valid.sum())/(expert_valid.sum()+agent_valid.sum())

        self.log("train/expert_value_loss", expert_value_loss.item(), on_step=True, batch_size=1)
        self.log("train/agent_value_loss", agent_value_loss.item(), on_step=True, batch_size=1)
        self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)

        # expert_r_min=-(1 / self.reg_mult)

        # expert_value_mse_loss=torch.square(expert_V-(expert_r_min+expert_target_v))[expert_valid].mean()

        # policy_r_min=(1 - done) * (-(1 / self.reg_mult))\
        #             + done * (1 / (1 - self.gamma)) * (-(1 / self.reg_mult))

        # agent_value_mse_loss=torch.square(agent_V-(policy_r_min+agent_target_v))[agent_valid].mean()

        # value_mse_loss=(expert_value_mse_loss*expert_valid.sum()+agent_value_mse_loss*agent_valid.sum())/(expert_valid.sum()+agent_valid.sum())

        # critic_loss=expert_Q_loss+value_mse_loss

        #min_reward= torch.clamp_min(expert_reward[expert_valid],min=0)

        # expert_js_loss=-(-expert_reward[expert_valid]-1).exp().mean()

        expert_reward=expert_reward[expert_valid]

        valid_expert_reward=expert_reward[expert_reward>-np.log(2)]

        expert_js_loss=(2-(-valid_expert_reward).exp()).log().mean()

        self.log("train/expert_js_loss", expert_js_loss.item(), on_step=True, batch_size=1)

        #expert_constrianed_loss=torch.max(torch.tensor(0.0),-expert_reward-np.log(2)).exp().mean()*10
        expert_constrianed_loss=torch.max(torch.tensor(0.0),expert_reward-2).exp().mean()*10

        self.log("train/expert_constrianed_loss", expert_constrianed_loss.item(), on_step=True, batch_size=1)

        agent_reward=agent_reward[agent_valid]

        agent_constrianed_loss=torch.max(torch.tensor(0.0),agent_reward-2).exp().mean()*10

        self.log("train/agent_constrianed_loss", agent_constrianed_loss.item(), on_step=True, batch_size=1)

        constrianed_loss=expert_constrianed_loss #+agent_constrianed_loss

        self.log("train/constrianed_loss", constrianed_loss.item(), on_step=True, batch_size=1)

        #critic_loss = -expert_js_loss+value_loss+constrianed_loss


        expert_chi2_loss=(expert_reward-expert_reward.square()/4).mean()

        self.log("train/expert_chi2_loss", expert_chi2_loss.item(), on_step=True, batch_size=1)

        critic_loss = -expert_chi2_loss+value_loss+constrianed_loss


        # chi2_expert_loss = 1 / (4 * alpha) * (expert_reward[expert_valid] ** 2).mean()
        #
        # chi2_agent_loss = 1 / (4 * alpha) * (agent_reward[agent_valid]  ** 2).mean()
        #
        # chi2_loss=(chi2_expert_loss*expert_valid.sum()+chi2_agent_loss*agent_valid.sum())/(expert_valid.sum()+agent_valid.sum())
        #
        # self.log("train/chi2_expert_loss", chi2_expert_loss.item(), on_step=True, batch_size=1)
        # self.log("train/chi2_agent_loss", chi2_agent_loss.item(), on_step=True, batch_size=1)
        # self.log("train/chi2_loss", chi2_loss.item(), on_step=True, batch_size=1)
        #
        # critic_loss=-expert_reward_loss+value_loss+chi2_loss

        return critic_loss

    def soft_update(self,net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def training_step(self, data, batch_idx):
        #print(self.global_step)
        #time1=time.time()
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)

            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
                train_mask=data["agent"]["train_mask"],  # [n_agent]
                current_epoch=self.current_epoch,
            )
        else:
            if len(self.replay_buffer)<self.replay_buffer.maxlen or self.global_step%10==0:
                with torch.no_grad():
                    self.rollout(tokenized_map, tokenized_agent)
            #print(time.time() - time1)

            loss=self.iq_update(tokenized_map, tokenized_agent)
            #print(time.time() - time1)

            if self.global_step % self.critic_target_update_frequency == 0:
                self.soft_update(self.encoder, self.target_net, self.critic_tau)

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
                #self.logger.log_metrics(epoch_wosac_metrics)
                for key, value in epoch_wosac_metrics.items():
                    self.log(key, value, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

            self.wosac_metrics.reset()
            self.minADE.reset()


