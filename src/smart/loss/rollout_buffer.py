import numpy as np
import torch
from collections import deque

import random

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.distributed as dist


def _is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_reduce_loss(loss, loss2=None):
    local_count = torch.tensor(
        [loss.numel()], device=loss.device, dtype=torch.float32
    )
    if _is_dist_available_and_initialized():
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        normalizer = local_count.item() / dist.get_world_size()
    else:
        normalizer = local_count.item()

    normalizer = max(normalizer, 1.0)
    if loss2 is not None:
        return (loss.sum() + loss2.sum()) / normalizer
    return loss.sum() / normalizer


class RunningMeanStdTorch(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4):
        super().__init__()

        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float64))
        self.initialized = False

    def _is_dist_available_and_initialized(self):
        return dist.is_available() and dist.is_initialized()

    def update(self, x):
        if x.numel() == 0 or x.size(0) == 0:
            return
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)
        if self._is_dist_available_and_initialized():
            self._update_distributed_from_moments(batch_mean, batch_var, batch_count)
        else:
            self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean#1.648
        self.var = new_var#39.758
        self.count = tot_count#18265

    def _update_distributed_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Distributed-safe update using all-reduce sum of sums and sum of squared-sums.
        Works for arbitrary shape in `mean`/`var`.
        """
        # device for tensors — use the same device as buffers (could be cpu or gpu)
        device = self.mean.device

        # local sums: sum(x) = mean * count ; sum(x^2) = (var + mean^2) * count
        local_sum = (batch_mean * float(batch_count)).to(device=device, dtype=torch.float64)
        local_sq_sum = ((batch_var + batch_mean.pow(2)) * float(batch_count)).to(device=device, dtype=torch.float64)
        local_count = torch.tensor(float(batch_count), dtype=torch.float64, device=device)

        # Prepare tensors for reduction (they must be same shape as buffers)
        # local_sum and local_sq_sum are same shape as mean/var, local_count is scalar.
        # We'll all_reduce them across processes.
        # Note: all_reduce operates in-place.
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        existing_sum = (self.mean * self.count).to(device=device, dtype=torch.float64)
        existing_sq_sum = ((self.var + self.mean.pow(2)) * self.count).to(device=device, dtype=torch.float64)

        # Sum existing + newly reduced local sums (local_sum already contains sums from all processes)
        total_sum = existing_sum + local_sum
        total_sq_sum = existing_sq_sum + local_sq_sum
        total_count = self.count + local_count  # scalar

        # Avoid division by zero
        eps = 1e-12
        new_mean = total_sum / (total_count + eps)
        new_var = total_sq_sum / (total_count + eps) - new_mean.pow(2)

        # ensure var is non-negative (numerical)
        new_var = torch.clamp(new_var, min=0.0)

        # write back into buffers
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


    def normalize(self, x):
        res=(x - self.mean.float()) / (torch.sqrt(self.var.float()) + 1e-8)
        return res

    def denormalize(self, x):
        res=x * torch.sqrt(self.var.float())  + self.mean.float()
        return res

def get_reward(s,kl_per_token, eps=1e-20, reward_type="airl"):
    s = s.detach()

    if reward_type == 'airl':
        rewards = (s + eps).log() - (1 - s + eps).log()

     #   rewards=torch.tanh(rewards)
    elif reward_type == 'gail':
        rewards = (s + eps).log()
    elif reward_type == 'raw':
        rewards = s
    elif reward_type == "positive":
        rewards = - (1 - s + eps).log()
    elif reward_type == 'airl-positive':
        rewards = (s + eps).log() - (1 - s + eps).log() + 20
    elif reward_type == 'symmetric_kl':
        rkl = (s + eps).log() - (1 - s + eps).log()
        kl = rkl.exp() * (-rkl)
        rewards = rkl + kl
    elif reward_type == 'revise':
        d_x = (s + eps).log()
        rewards = d_x + (-1 - (-d_x).log())

    rewards=rewards-kl_per_token
    # if key=='agent':
    #
    #     self.update(rewards.reshape(-1))
    #
    #     rewards=self.normalize(rewards)

        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    # rewards=F.normalize(rewards,dim=0)

    return rewards


def get_return(rewards, gamma):

    returns = torch.zeros_like(rewards)
    running_return = returns[:, -1]

    for i in reversed(range(rewards.shape[1])):
        running_return = rewards[:, i] + gamma * running_return
        returns[:, i] = running_return

    return returns

def get_nei_returns(tokenized_agent,reward,neighbor_dist=10,train_mask=None):
    pos = tokenized_agent["sampled_pos"][:,1:-1]
    batch = tokenized_agent["batch"]

    if train_mask is not None:
        pos = pos[train_mask]
        batch = batch[train_mask]

    M = pos.size(0)

    # Pairwise distances [M, M]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [M, M, 2]
    dist = torch.norm(diff, dim=-1)  # [M, M]

    # Mask: same batch & within distance & not self
    same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)  # [M, M]
    within_dist = dist < neighbor_dist
    not_self = ~torch.eye(M, dtype=torch.bool,device=pos.device)
    mask = same_batch[:,:,None] & within_dist & not_self[:,:,None]

    # Gather neighbor rewards
    neighbor_rewards = (mask * reward.unsqueeze(0)).sum(dim=1)  # [M]
    neighbor_counts = mask.sum(dim=1)
    neighbor_mean_rewards = neighbor_rewards / (neighbor_counts+1e-6)

    return neighbor_mean_rewards

import torch

def per_scene_zscore_clip(r, batch, mask, clip_val=5.0, eps=1e-6):
    """
    r:     [N, T]  raw per-agent rewards
    batch: [N]     scene id for each agent (0..B-1)
    mask:  [N, T]  bool; False where (agent,t) is invalid
    """
    r = r.clone()
    mask_f = mask.float()
    r = r * mask_f

    # unique scene ids
    uniq = torch.unique(batch)
    r_out = torch.zeros_like(r)

    for b in uniq.tolist():
        idx = (batch == b)                     # agents in scene b  [N]
        if idx.any():
            r_b = r[idx]                       # [Nb, T]
            m_b = mask_f[idx]                  # [Nb, T]
            count = m_b.sum()
            if count < 1:
                continue
            mean = (r_b).sum() / count
            var = ((r_b - mean) * m_b).pow(2).sum() / count
            std = var.sqrt().clamp_min(eps)

            r_norm = ((r_b - mean) / std) * m_b
            r_norm = torch.clamp(r_norm, -clip_val, clip_val)
            r_out[idx] = r_norm
    return r_out

import torch

@torch.no_grad()
def get_near_returns(
    tokenized_agent: dict,
    agent_rewards: torch.Tensor,   # [N_valid, T]  (only valid agents)
    train_mask: torch.Tensor,      # [N] bool, True = valid
    neighbor_dist: float = 60.0,
    pos_keys=("sampled_pos", "pos_a", "pos"),
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    For each INVALID agent and each timestep, compute the mean reward of VALID neighbors
    within `neighbor_dist` (same scene). Output is [N, T]; values for valid agents are 0.

    Assumes `agent_rewards` rows align with `valid_idx = train_mask.nonzero(as_tuple=True)[0]`.
    """
    # ---- fetch positions [N, T, 2] ----
    pos = None
    for k in pos_keys:
        if k in tokenized_agent:
            pos = tokenized_agent[k]
            break
    if pos is None:
        raise KeyError(f"None of {pos_keys} found in tokenized_agent")
    if pos.ndim != 3 or pos.size(-1) < 2:
        raise ValueError(f"pos must be [N, T, 2+], got {tuple(pos.shape)}")
    pos = pos[...,2:, :2]

    batch = tokenized_agent.get("batch", None)
    if batch is None:
        raise KeyError("tokenized_agent['batch'] is required (scene id per agent)")
    if batch.ndim != 1:
        batch = batch.view(-1)

    N, T, _ = pos.shape
    assert train_mask.shape == (N,) and train_mask.dtype == torch.bool

    device = pos.device
    dtype = agent_rewards.dtype

    valid_idx = torch.nonzero(train_mask, as_tuple=True)[0]        # [N_valid]
    invalid_idx = torch.nonzero(~train_mask, as_tuple=True)[0]     # [N_invalid]

    if agent_rewards.shape != (valid_idx.numel(), T):
        raise ValueError(f"agent_rewards must be [N_valid, T]; got {tuple(agent_rewards.shape)}")

    out = torch.zeros((N, T), dtype=dtype, device=device)
    if invalid_idx.numel() == 0:
        return out

    # process per scene for correctness & efficiency
    if batch.numel() == 0:
        return out[~train_mask]
    B = int(batch.max().item()) + 1
    for b in range(B):
        inv_b = invalid_idx[(batch[invalid_idx] == b)]
        if inv_b.numel() == 0:
            continue

        val_mask_b = (batch[valid_idx] == b)
        if not val_mask_b.any():
            # no valid neighbors in this scene
            continue

        val_b = valid_idx[val_mask_b]                             # [n_val_b]
        pos_inv_b = pos[inv_b]                                    # [n_inv_b, T, 2]
        pos_val_b = pos[val_b]                                    # [n_val_b, T, 2]
        rew_val_b = agent_rewards[val_mask_b]                     # [n_val_b, T]

        # batch cdist over time: [T, n_inv_b, n_val_b]
        pos_inv_bt = pos_inv_b.transpose(0, 1)                    # [T, n_inv_b, 2]
        pos_val_bt = pos_val_b.transpose(0, 1)                    # [T, n_val_b, 2]
        D = torch.cdist(pos_inv_bt, pos_val_bt, p=2)              # [T, n_inv_b, n_val_b]

        # neighbors per step
        nb = (D <= neighbor_dist)                                 # [T, n_inv_b, n_val_b]
        cnt = nb.sum(dim=2)                                       # [T, n_inv_b]

        # weighted sum of valid rewards per step
        # rew_val_bt: [T, 1, n_val_b]
        rew_val_bt = rew_val_b.transpose(0, 1).unsqueeze(1)       # [T, 1, n_val_b]
        sum_rew = (nb * rew_val_bt).sum(dim=2)                    # [T, n_inv_b]

        avg = torch.where(
            cnt > 0,
            sum_rew / cnt.clamp(min=1),
            torch.as_tensor(fill_value, dtype=dtype, device=device),
        )  # [T, n_inv_b]

        out[inv_b] = avg.transpose(0, 1)                          # -> [n_inv_b, T]

    return out[~train_mask]

#
#
# def get_near_returns(tokenized_agent, reward, neighbor_dist=60.0, k=1,train_mask=None):
#     """
#     Average reward of the k nearest *valid* neighbors (same batch, not self)
#     for each agent at each timestep. If fewer than k are within neighbor_dist,
#     average over the available ones; if none, return 0.
#
#     Args:
#         tokenized_agent: dict with keys:
#             - "sampled_pos": [M, T, 2]
#             - "batch":       [M]
#         reward: [M, T] rewards per agent per timestep
#         neighbor_dist: scalar distance threshold
#         k: number of nearest neighbors to consider (upper bound)
#
#     Returns:
#         avg_nn_reward: [M, T] averaged neighbor reward
#         valid_counts:  [M, T] how many neighbors actually contributed
#     """
#     pos   = tokenized_agent["sampled_pos"][:, 1:-1]   # [M, T, 2]
#     batch = tokenized_agent["batch"]                  # [M]
#
#     if train_mask is not None:
#         pos = pos[train_mask]
#         batch = batch[train_mask]
#
#     M, T, _ = pos.shape
#     device = pos.device
#
#     # Pairwise distances per timestep: [M, M, T]
#     diff = pos.unsqueeze(1) - pos.unsqueeze(0)        # [M, M, T, 2]
#     dist = torch.norm(diff, dim=-1)                   # [M, M, T]
#
#     # Valid neighbor mask: same batch & not self
#     same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)  # [M, M]
#     not_self   = ~torch.eye(M, dtype=torch.bool, device=device)
#     valid_pair = same_batch & not_self
#
#     # Mask invalid pairs with +inf so they won't be chosen
#     dist = dist.masked_fill(~valid_pair[:, :, None], float("inf"))  # [M, M, T]
#
#     # Limit k to available neighbors (<= M-1)
#     K = min(k, max(1, M-1))
#
#     # k nearest neighbors along neighbor dim (M): shapes [M, K, T]
#     nn_dist, nn_idx = dist.topk(K, dim=1, largest=False)
#
#     # Gather rewards of those neighbors at the same timestep
#     t_idx = torch.arange(T, device=device).expand(M, K, T)          # [M, K, T]
#     nn_rewards = reward[nn_idx, t_idx]                              # [M, K, T]
#
#     # Keep only neighbors within neighbor_dist
#     valid = nn_dist < neighbor_dist                                 # [M, K, T]
#
#     # Sum and count over valid neighbors
#     sum_rewards = (nn_rewards * valid).sum(dim=1)                   # [M, T]
#     counts      = valid.sum(dim=1)                                  # [M, T]
#
#     # Average over available neighbors; if none, return 0
#     avg = torch.where(counts > 0, sum_rewards / counts.clamp(min=1), torch.zeros_like(sum_rewards))
#
#     return avg

# def get_near_returns(tokenized_agent, reward, neighbor_dist=60.0):
#     pos = tokenized_agent["sampled_pos"][:, 1:-1]  # [M, T, 2]
#     batch = tokenized_agent["batch"]               # [M]
#     M = pos.size(0)
#
#     # Pairwise distances across trajectory
#     diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [M, M, T, 2]
#     dist = torch.norm(diff, dim=-1)             # [M, M, T]
#
#
#     # Mask out different batches and self
#     same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)  # [M, M]
#     not_self = ~torch.eye(M, dtype=torch.bool, device=pos.device)
#     mask = same_batch & not_self
#
#     dist_masked = dist.masked_fill(~mask[:,:,None], float("inf"))
#
#     # Reduce across time (choose your criterion: min/mean/last)
#     nn_dist,nn_idx = dist_masked.min(dim=1) #.values  # [M, T]
#
#     # Gather rewards
#     M, T = nn_idx.shape
#
#     # make timestep indices [M, T]
#     t_idx = torch.arange(T, device=nn_idx.device).expand(M, T)
#
#     # gather neighbor rewards
#     nn_reward = reward[nn_idx, t_idx]  # [M, T]
#
#     # Optionally zero if too far
#     nn_reward = torch.where(nn_dist < neighbor_dist,
#                             nn_reward,
#                             torch.zeros_like(nn_reward))
#
#     return nn_reward
#



class ReplayBuffer:
    def __init__(self, max_len=1):

        self.state_action_list=deque(maxlen=max_len)

        self.value_list=deque(maxlen=max_len)
        self.action_log_probs_list=deque(maxlen=max_len)

    def __len__(self):
        return len(self.state_action_list)

    def initialize(self,map, agent_num,device):

        self.rewards = torch.zeros(self.num_steps, agent_num).to(device)
        self.value_preds = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.returns = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.action_log_probs = torch.zeros(self.num_steps, agent_num).to(device)
        self.actions = torch.zeros(self.num_steps, agent_num).to(device).to(torch.int)
        self.masks = torch.ones(self.num_steps + 1, agent_num).to(device)
        self.masks[-1]=0
        self.map=map

        self.smaple_idx=list(range(self.num_steps))

        random.shuffle(self.smaple_idx)

    def insert(self, sample,step):
        self.state.append(sample["state"])
        self.value_preds[step]=sample["value"]

        if step<self.num_steps:
            self.action_log_probs[step]=sample["prev_log_prob"]
            self.actions[step]=sample["action"]

    def sample(self):
        return {"state_action": self.state_action_list[-1],
                "prev_log_prob":self.action_log_probs_list[-1],
                "adv":self.advantages,
                "value":self.value_preds[:,:-1],
                "return":self.returns
                }


    def compute_advantages(self,gamma=0.99,gae_lambda=0.95):
        self.value_preds=torch.cat([self.value_list[-1],torch.zeros_like(self.value_list[-1][:,:1])],dim=1)

        self.masks = torch.ones_like(self.value_preds)
        self.masks[:,-1]=0

        self.returns = []

        gae = 0
        for step in reversed(range(self.rewards.size(1))):
            delta = (
                self.rewards[:,step]
                + gamma * self.value_preds[:,step + 1] * self.masks[:,step + 1]
                - self.value_preds[:,step]
            )
            gae = (
                delta
                + gamma * gae_lambda * self.masks[:,step + 1] * gae
            )
            #self.returns[step] = gae + self.value_preds[step]
            self.returns.insert(0, gae + self.value_preds[:,step])

        self.returns=torch.stack(self.returns,dim=1)

        advantages = self.returns - self.value_preds[:,:-1]
        # Normalize the advantages
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

def rollout(encoder, tokenized_map, tokenized_agent):
    train_data_len=tokenized_agent["sampled_idx"].shape[1]
    max_step = (train_data_len - (encoder.agent_encoder.num_historical_steps)//encoder.agent_encoder.shift)*encoder.agent_encoder.shift

    encoder.eval()
    with torch.no_grad():
        pred = encoder.inference(
            tokenized_map,
            tokenized_agent,
            n_step_future_10hz=max_step
        )
    encoder.train()

    tokenized_agent.update(pred)
    return tokenized_agent


def get_return_diff(reward,log_prob,current_Q,V,alpha,gamma):
    rewards=reward - alpha * log_prob
    returns = torch.zeros_like(V)
    running_return=returns[:,-1]

    for i in range(rewards.size(1)-1,-1,-1):
        running_return = rewards[:, i] + gamma *running_return
        returns[:, i] = running_return

    current_Q_diff = (current_Q - returns[:,:-1])
    V_diff=(V[:,:-1]-returns[:,:-1])

    return current_Q_diff, V_diff

def get_train_mask(tokenized_agent,start_step):
    valid_mask = tokenized_agent["valid_mask"][:, start_step:]
    # valid_mask1 = valid_mask.clone()
    train_mask = valid_mask[:, 1:] & valid_mask[:, :-1]

    if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
        train_mask = train_mask[tokenized_agent["train_mask"]]

    return train_mask.transpose(0, 1)


def compute_advantages(rewards, values,gamma=0.99,lam=0.95,infinite_horizon=False):#0.95

    values=values.reshape(rewards.shape[0],rewards.shape[1])

    v_detach=values.detach()#t,a

    dones = torch.zeros_like(rewards)

    # dones[-1]=1

    if infinite_horizon:
        rewards_len = rewards.shape[0]
    else:
        dones[-1] = 1
        rewards_len = rewards.shape[0]

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards_len)):
        if t == rewards.shape[0] - 1:
            next_value = v_detach[t] if infinite_horizon else 0
        else:
            next_value = v_detach[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - v_detach[t]
        advantages[t] = last_adv = delta + gamma * lam * next_non_terminal * last_adv

    returns = advantages + v_detach

    value_loss = (returns - values).square().clamp(min=0, max=100) #.mean()

    return advantages,value_loss

# def compute_advantages(rewards, values, mask, gamma=0.99, lam=0.95):
#     """
#     Infinite-horizon / continuing-task GAE.
#     - No artificial done at last step.
#     - Bootstraps from the value function at the end of the rollout.
#     """
#     # Ensure shape [T, ...] matches rewards
#     values = values.reshape_as(rewards)
#
#     v_detach = values.detach()        # [T, ...]
#     T = rewards.shape[0]
#
#     advantages = torch.zeros_like(rewards)
#     last_adv = 0
#
#     # GAE for a continuing MDP (no terminal at T-1)
#     for t in reversed(range(T)):
#         if t == T - 1:
#             # Bootstrap from the value at the last state
#             next_value = v_detach[t]
#         else:
#             next_value = v_detach[t + 1]
#
#         # No dones: treat everything as non-terminal (infinite horizon)
#         delta = rewards[t] + gamma * next_value - v_detach[t]
#         last_adv = delta + gamma * lam * last_adv
#         advantages[t] = last_adv
#
#     returns = advantages + v_detach
#
#     # Same value loss with mask
#     value_loss = (returns - values)[mask].square().clamp(min=0, max=100).mean()
#
#     return advantages, value_loss
#
