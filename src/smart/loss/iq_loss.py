import torch
from sympy.physics.units import action

from src.smart.metrics.utils import get_euclidean_targets

from src.smart.loss.gmm_dist import  GMM_Dist,get_entropy
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
    div = 'x2'
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

        critic_loss = alpha * (-expert_reward / alpha).exp().mean() + value_loss
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

def get_return(reward,log_prob,current_Q,V,all_valid_mask,alpha,gamma):
    rewards=reward - alpha * log_prob
    returns = torch.zeros_like(V)
    running_return=returns[:,-1]

    for i in range(rewards.size(1)-1,-1,-1):
        running_return = rewards[:, i] + gamma *running_return
        returns[:, i] = running_return

    current_Q_diff = (current_Q - returns[:,:-1])[all_valid_mask]
    V_diff=(V[:,:-1]-returns[:,:-1])[all_valid_mask]

    return current_Q_diff, V_diff

def eval_light(tokenized_agent,tokenized_agent_rollout,logger,light_type):
    real_light = tokenized_agent["light_idx"][:, 2:]

    batch_lg = tokenized_agent["batch_lg"]

    batch_mask = batch_lg[:, None] == batch_lg[None]

    real_light_mask = (real_light < light_type).all(
        -1)

    repeat_pred = tokenized_agent["light_idx"][:, 1:2].repeat(1, real_light.shape[1])

    repeat_light_acc = (repeat_pred == real_light)[real_light_mask].float().mean()

    logger("train/repeat_light_acc", repeat_light_acc.item(), on_step=True, batch_size=1)

    real_relation = (real_light[:, None] == real_light[None])[batch_mask]

    real_relation_mask = (real_light_mask[:, None] & real_light_mask[None])[batch_mask]

    repeat_relation = (repeat_pred[:, None] == repeat_pred[None])[batch_mask]

    repeat_relation_acc = (real_relation == repeat_relation)[real_relation_mask].float().mean()

    logger("train/repeat_relation_acc", repeat_relation_acc.item(), on_step=True, batch_size=1)

    light_rollout = tokenized_agent_rollout["light_idx"][:, 2:]

    light_acc = (light_rollout == real_light)[real_light_mask].float().mean()

    logger("train/agent_light_acc", (light_acc - repeat_light_acc).item(), on_step=True, batch_size=1)

    agent_relation = (light_rollout[:, None] == light_rollout[None])[batch_mask]

    agent_relation_acc = (real_relation == agent_relation)[real_relation_mask].float().mean()

    logger("train/agent_relation_acc", (agent_relation_acc - repeat_relation_acc).item(), on_step=True, batch_size=1)



def padding(tensor,lengths,padding_value=0.0 ):
    padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=padding_value)

    return padded_tensor


def get_proposal_loss(proposal,tokenized_agent,start_step,train_mask):

    token_agent_shape = tokenized_agent["token_agent_shape"][:, None, None, None]
    sampled_pos = tokenized_agent["sampled_pos"][:, start_step:-1]
    sampled_heading = tokenized_agent["sampled_heading"][:, start_step:-1]
    target_global_traj = tokenized_agent["target_global_traj"][:, start_step:-1,:proposal.shape[3]]
    target_mask = tokenized_agent["target_mask"][:, start_step:-1, None,:proposal.shape[3]]

    target_pos = target_global_traj[..., :2].flatten(0, 1)
    target_head = target_global_traj[..., 2].flatten(0, 1)

    target_pos, target_head = transform_to_local(
        pos_global=target_pos,  # [n_agent*18, 1, 2]
        head_global=target_head,  # [n_agent*18, 1]
        pos_now=sampled_pos.flatten(0, 1),  # [n_agent*18, 2]
        head_now=sampled_heading.flatten(0, 1),  # [n_agent*18]
    )

    target_pos = target_pos.reshape(-1, target_global_traj.shape[1], 1, target_global_traj.shape[2], 2)
    target_head = target_head.reshape(-1, target_global_traj.shape[1], 1, target_global_traj.shape[2])

    target_contour = cal_polygon_contour(target_pos, target_head, token_agent_shape)

    proposal_contour = cal_polygon_contour(proposal[..., :2], proposal[..., 2], token_agent_shape)#B,T,N,F,4,2

    pos_loss = (torch.linalg.norm(proposal[..., :2] - target_pos, dim=-1) * target_mask)
    head_loss = (wrap_angle(proposal[..., 2] - target_head).abs() * target_mask)

    counter_dist =torch.linalg.norm(proposal_contour - target_contour, dim=-1).mean(-1) * target_mask

    target_sum=target_mask.sum()

    proposal_loss = counter_dist.sum(-1).amin(-1).sum()/target_sum

    proposal5_loss = counter_dist[:, :, :, 4]

    action = torch.argmin(proposal5_loss, dim=-1)

    pos_dist = torch.gather(pos_loss[..., 4], index=action[:, :, None], dim=-1).sum()/target_sum
    head_diff = torch.gather(head_loss[..., 4], index=action[:, :, None], dim=-1).sum()/target_sum

    return proposal_loss, pos_dist, head_diff,action


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