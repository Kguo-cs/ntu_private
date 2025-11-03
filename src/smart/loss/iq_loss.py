import torch

from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal,MultivariateNormal

def soft_update( net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss,expert_Q,agent_Q   ):
    div = 'rkl'
    alpha = 0.5
    eps = 1e-3
    if div == "lsif":
        critic_loss = -expert_reward.exp().mean() + 1 / 2 * (2 * agent_reward).exp().mean()
    elif div == 'bce':
        critic_loss = ((-expert_reward / 1).exp() + 1).log().mean() + ((agent_reward / 1).exp() + 1).log().mean()
    elif div == 'ukl':
        critic_loss = -expert_reward.mean() + agent_reward.exp().mean()
    elif div == 'exp':
        critic_loss = (-expert_reward).exp().mean() + agent_reward.exp().mean()
    elif div == 'rkl':
        value_loss = (agent_value_loss.mean() + expert_value_loss.mean()) / 2
        alpha=1
        critic_loss = alpha * ((-expert_reward-1) / alpha).exp().mean() + value_loss
    elif div == 'recoil':
        chi_loss=(expert_reward.square().mean()+agent_reward.square().mean())/2
        critic_loss = -expert_Q.mean() + agent_Q.mean()+chi_loss/(4*0.02)

    elif div == 'tv':
        critic_loss = (-expert_reward).mean() + agent_reward.mean()
    elif div == 'x2':
        value_loss = (agent_value_loss.mean() + expert_value_loss.mean()) / 2

        # value_loss=agent_value_loss.mean()

        # chi2_loss=(expert_reward.square().mean()/ (4 * alpha)+agent_reward.square().mean()/ (4 * alpha)) /2

        chi2_loss = expert_reward.square().mean() / (4 * alpha)
        critic_loss = -expert_reward.mean() + value_loss + chi2_loss

        # (agent_value_loss.mean()/2+expert_value_loss.mean()/2)#agent_value_loss.mean()
    elif div == 'kl':
        expert_reward = torch.clamp_min(expert_reward, min=alpha * eps)
        critic_loss = -alpha * ((expert_reward / alpha).log().mean() + 1) + agent_reward.mean()
    elif div == 'sh':
        critic_loss = - (expert_reward / (1 + expert_reward)).mean() + agent_reward.mean()
    elif div == 'js':
        # phi_grad = torch.exp(-expert_reward) / (2 - torch.exp(-expert_reward))
        #
        value_loss = (agent_value_loss.mean() + expert_value_loss.mean()) / 2
        # critic_loss =  -(phi_grad.detach()*expert_reward).mean()+value_loss

        expert_reward = torch.clamp_min(expert_reward,
                                        min=alpha * (np.log(1 / 2 + eps)))  # ,max=alpha*(np.log(1/2+1/eps))
        critic_loss = -(2 - (-expert_reward / alpha).exp()).log().mean() + value_loss.mean()
    else:
        critic_loss = (expert_reward - 1).square().mean() + (agent_reward + 1).square().mean()

    return critic_loss

def eval_light(expert_light_idx,tokenized_agent_rollout,logger,light_type):
    real_light = expert_light_idx[:, 2:]

    batch_lg = tokenized_agent_rollout["batch_lg"]

    batch_mask = batch_lg[:, None] == batch_lg[None]

    real_light_mask = (real_light < light_type).all(
        -1)

    repeat_pred = expert_light_idx[:, 1:2].repeat(1, real_light.shape[1])

    repeat_light_acc = (repeat_pred == real_light)[real_light_mask].float().mean()

    logger("train/repeat_light_acc", repeat_light_acc.item(), on_step=True, batch_size=1)

    real_relation = (real_light[:, None] == real_light[None])[batch_mask]

    real_relation_mask = (real_light_mask[:, None] & real_light_mask[None])[batch_mask]

    repeat_relation = (repeat_pred[:, None] == repeat_pred[None])[batch_mask]

    repeat_relation_acc = (real_relation == repeat_relation)[real_relation_mask].float().mean()

    logger("train/repeat_relation_acc", repeat_relation_acc.item(), on_step=True, batch_size=1)

    light_rollout = tokenized_agent_rollout["light_idx"][:, 2:]

    light_acc = (light_rollout == real_light)[real_light_mask].float().mean()

    logger("train/agent_light_acc", light_acc.item(), on_step=True, batch_size=1)

    agent_relation = (light_rollout[:, None] == light_rollout[None])[batch_mask]

    agent_relation_acc = (real_relation == agent_relation)[real_relation_mask].float().mean()

    logger("train/agent_relation_acc", agent_relation_acc.item(), on_step=True, batch_size=1)



def padding(tensor,lengths,padding_value=0.0 ):
    padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=padding_value)

    return padded_tensor


def get_proposal_loss(proposal,tokenized_agent,start_step):

    #token_agent_shape = tokenized_agent["token_agent_shape"][:, None, None, None]#[train_mask]
    sampled_pos = tokenized_agent["sampled_pos"][:, start_step:-1]#[train_mask]
    sampled_heading = tokenized_agent["sampled_heading"][:, start_step:-1]#[train_mask]
    # target_global_traj = tokenized_agent["target_global_traj"][:, start_step:-1,:proposal.shape[3]]#[train_mask]
    # target_mask = tokenized_agent["target_mask"][:, start_step:-1, None,:proposal.shape[3]]#[train_mask]
    #

    target_global_traj=torch.cat([tokenized_agent["sampled_pos"],tokenized_agent["sampled_heading"][:,:,None]],dim=-1)[:, start_step+1:,None]
    target_mask=tokenized_agent["valid_mask"][:, start_step+1:,None,None]

    if "train_mask" in tokenized_agent.keys():
        train_mask = tokenized_agent["train_mask"]
       # token_agent_shape=token_agent_shape[train_mask]
        sampled_pos=sampled_pos[train_mask]
        sampled_heading=sampled_heading[train_mask]
        target_global_traj=target_global_traj[train_mask]
        target_mask=target_mask[train_mask]

    #target_mask=target_mask & train_mask[:,None,None,None]

    target_global_pos = target_global_traj[..., :2].flatten(0, 1)
    target_global_head = target_global_traj[..., 2].flatten(0, 1)

    target_pos, target_head = transform_to_local(
        pos_global=target_global_pos,  # [n_agent*18, 1, 2]
        head_global=target_global_head,  # [n_agent*18, 1]
        pos_now=sampled_pos.flatten(0, 1),  # [n_agent*18, 2]
        head_now=sampled_heading.flatten(0, 1),  # [n_agent*18]
    )

    target_pos = target_pos.reshape(-1, target_global_traj.shape[1], 1, target_global_traj.shape[2], 2)
    target_head = target_head.reshape(-1, target_global_traj.shape[1], 1, target_global_traj.shape[2])


    pos_loss = (torch.linalg.norm(proposal[...,:1,-1:, :2] - target_pos, dim=-1) * target_mask)#[..., -1]
    head_loss = (wrap_angle(proposal[...,:1,-1:, 2] - target_head).abs() * target_mask)#[..., -1]


    if proposal.shape[2]==2:

        proposal_mean=proposal[:,:,0,-1]

        proposal_std=proposal[:,:,1,-1]

        distribution= Independent(Normal(proposal_mean, proposal_std),1)

        target_state=torch.cat((target_pos, target_head[...,None]), dim=-1)[:,:,0,-1]

        proposal_logprob=distribution.log_prob(target_state)

        proposal_loss = -proposal_logprob[target_mask[:,:,0,-1]].mean()

        print(proposal_logprob[target_mask[:,:,0,-1]].min(), proposal_logprob[target_mask[:,:,0,-1]].max(),proposal_logprob[target_mask[:,:,0,-1]].mean(), proposal_loss.mean())

        #proposal_logprob=proposal_logprob.sum(-1)

    # else:
    #     target_contour = cal_polygon_contour(target_pos, target_head, token_agent_shape)
    #
    #     proposal_contour = cal_polygon_contour(proposal[...,,:1,:, :2], proposal[...,,:1,:, 2], token_agent_shape)
    #
    #     counter_dist = torch.linalg.norm(proposal_contour - target_contour, dim=-1).mean(-1) * target_mask
    #
    #     target_sum = target_mask.sum() + 1e-8
    #
    #     proposal_loss = counter_dist.sum(-1).amin(-1).sum()/target_sum
    #
    #     proposal5_loss = counter_dist[:, :, :, 4]
    #
    #     action = torch.argmin(proposal5_loss, dim=-1)

    return proposal_loss,proposal_logprob, pos_loss, head_loss


def get_gaussian_loss(proposal,tokenized_agent):

    proposal=proposal.reshape(proposal.shape[0], proposal.shape[1],-1,4)

    proposal_mean=proposal[...,:2]
    proposal_cov=proposal[...,2:].exp()+1

    dist=Independent(Normal(proposal_mean, proposal_cov),1)

    def get_future_30_every_5th_step_with_padding(tensor, pad_value=0.0):
        B, T, D = tensor.shape
        max_future = 6

        # Pad extra steps to the right to safely index up to t+30
        padded_tensor = F.pad(tensor, (0, 0, 0, max_future), value=pad_value)  # shape: (B, T+30, D)

        # Start indices every 5 steps
        starts = torch.arange(0, T, 1, device=tensor.device)  # (T//5,)

        # For each start t, get future steps t+1 to t+30 (exclude t)
        offsets = torch.tensor([1, 2, 3, 4, 5,6],
                               device=tensor.device)  # torch.arange(1, max_future + 1, device=tensor.device)  # (30,)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (T//5, 30)
        gathered = padded_tensor[:, indices]  # (B, T//5, 30, D)

        return gathered

    sampled_pos = tokenized_agent["sampled_pos"]
    sampled_heading = tokenized_agent["sampled_heading"]
    valid_mask=tokenized_agent["valid_mask"][:,1:]

    gt_traj = torch.cat([sampled_pos, sampled_heading[:, :, None]], dim=-1)[:,1:]

    gt_traj[~valid_mask]=0

    target_global_traj = get_future_30_every_5th_step_with_padding(gt_traj)[:,:-1]  # shape: (B, T//5, 30, 2)

    target_mask = target_global_traj.any(-1) != 0

    target_pos = target_global_traj[..., :2].flatten(0, 1)
    target_head = target_global_traj[..., 2].flatten(0, 1)

    target_pos, target_head = transform_to_local(
        pos_global=target_pos,  # [n_agent*18, 1, 2]
        head_global=target_head,  # [n_agent*18, 1]
        pos_now=sampled_pos[:,1:-1] .flatten(0, 1),  # [n_agent*18, 2]
        head_now=sampled_heading[:,1:-1].flatten(0, 1),  # [n_agent*18]
    )

    target_pos = target_pos.reshape(-1, target_global_traj.shape[1],  target_global_traj.shape[2], 2)
    #target_head = target_head.reshape(-1, target_global_traj.shape[1],  target_global_traj.shape[2])

    #target_head=wrap_angle(target_head)

    target_local=target_pos #torch.cat([target_pos, target_head[:,:,:,None]], dim=-1)

    proposal_loss = -(dist.log_prob(target_pos)*target_mask).mean(-1)#.clamp_min(min=np.log(1e-10))

    pos_dist = (torch.linalg.norm(target_local[..., :2]-proposal_mean[..., :2],dim=-1)*target_mask).mean(-1)
    head_diff =0 #(wrap_angle(target_local[..., 2]-proposal_mean[..., 2]).abs() *target_mask).mean(-1)

    return proposal_loss, pos_dist, head_diff


def get_network_QV(self, q, tokenized_map, tokenized_agent, action, key):

    action = action.unsqueeze(-1)  # .reshape(-1)

    current_Q = torch.gather(q, dim=-1, index=action).squeeze(-1)  # [B, Tm1, T_a]

    current_V = self.alpha * torch.logsumexp(q / self.alpha, dim=-1, keepdim=False)  # V=Q+alpha*H

    V = torch.cat([current_V, torch.zeros_like(current_V[:, :1])], dim=-1)

    current_V = V[:, :-1]
    next_V = V[:, 1:]

    pi = torch.softmax(q / self.alpha, dim=-1)

    logpi = torch.log(pi + 1e-10)  # .clamp_min(min=1e-10)

    log_prob = torch.gather(logpi, dim=-1, index=action).squeeze(-1)
    entropy = -torch.sum(pi * logpi, dim=-1)

    actor_loss = self.alpha * log_prob - current_Q

    dones = torch.zeros_like(next_V)
    dones[:, -1] = 1
    y = self.gamma * (1 - dones) * next_V
    reward = current_Q - y
    value_loss = current_V - y

    return log_prob, pi, actor_loss, entropy, current_Q, V, value_loss, reward


def log_iq():
    all_valid_mask = valid_mask.all(-1)

    # log_prob, pi, actor_loss, entropy, current_Q, V, value_loss, reward = self.get_network_QV(pred["agent_q"],
    #                                                                                           tokenized_map,
    #                                                                                           tokenized_agent,
    #                                                                                           action, key)

    # current_Q_diff, V_diff = get_return_diff(reward,log_prob,current_Q,V,self.alpha,self.gamma)

    # if self.use_target_q and key=="expert":
    #     with torch.no_grad():
    #         pred = self.target_net(tokenized_map, tokenized_agent)
    #
    #         target_V = self.get_network_QV( pred["agent_q"], tokenized_map, tokenized_agent, action, key)[5]
    #     self.log("train/" + key + "_target_V", target_V.mean().item(), on_step=True, batch_size=1)
    # else:
    #     target_V=0

    init_V = V[:, 0]
    last_V = V[:, -1]

    if train_mask is not None:
        reward = reward[train_mask]

        value_loss = value_loss[train_mask]

        V = V[all_valid_mask]

        current_Q = current_Q[all_valid_mask]

        entropy = entropy[all_valid_mask]

        init_V = init_V[all_valid_mask]

        last_V = last_V[all_valid_mask]

    if self.use_ce and key == "expert":
        pred = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": pred["agent_q"] / 0.1,  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        }
        action_nll = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=tokenized_agent["train_mask_ce"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
    else:
        action_nll = -log_prob[train_mask].mean()

    self.log("train/" + key + "_V", V.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_Q", current_Q.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_entropy", entropy.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_reward", reward.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_lastV", last_V.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_initV", init_V.mean().item(), on_step=True, batch_size=1)
    self.log("train/" + key + "_value_loss", value_loss.mean().item(), on_step=True, batch_size=1)

