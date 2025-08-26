"""
Multi-agent GAIL with an explicit shared latent z (InfoGAIL-style)
-----------------------------------------------------------------

What this gives you
- Parameter-shared decentralized actors that *all* receive a public latent z.
- Centralized critic (CTDE) that also conditions on z.
- Joint discriminator D(s, a_1..a_N) (does not get z) for imitation reward.
- Recognition network Q(z | context) to maximize MI(z; context) (InfoGAIL).
- PPO updates with imitation reward + MI bonus.

You provide
- A vectorized multi-agent env (N homogeneous agents), with step(joint_action) → (s', o_i, r_env, done, info).
- An expert dataset that can yield minibatches of (s, a_joint) for the discriminator.

Notes
- This is a clean, minimal reference. Wire into your runner / buffers.
- Supports continuous actions (diagonal Gaussian). Easy to extend to discrete.
- z is categorical with K modes; sampled once per episode by default (easy switch to step-wise).

Author: ChatGPT (B2: Explicit shared latent z broadcast to actors)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def one_hot(idx: torch.LongTensor, K: int) -> torch.Tensor:
    y = torch.zeros(idx.shape + (K,), device=idx.device)
    y.scatter_(-1, idx.unsqueeze(-1), 1.0)
    return y


def mlp(sizes: List[int], act=nn.ReLU, last_act: Optional[nn.Module] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    if last_act is not None:
        layers += [last_act]
    return nn.Sequential(*layers)

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class SharedActor(nn.Module):
    """Parameter-shared actor π(a_i | o_i, z, id_i) with diagonal-Gaussian head.

    Args:
        obs_dim: per-agent observation dim
        z_dim: dimensionality of one-hot z  (K)
        id_dim: number of agent IDs for one-hot embedding (or 0 to disable)
        act_dim: per-agent action dim (continuous)
    """
    def __init__(self, obs_dim: int, z_dim: int, id_dim: int, act_dim: int,
                 hidden: int = 128):
        super().__init__()
        in_dim = obs_dim + z_dim + (id_dim if id_dim > 0 else 0)
        self.core = mlp([in_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.id_dim = id_dim

    def forward(self, obs: torch.Tensor, z_oh: torch.Tensor, agent_id_oh: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.id_dim > 0:
            assert agent_id_oh is not None, "agent_id_oh is required when id_dim>0"
            x = torch.cat([obs, z_oh, agent_id_oh], dim=-1)
        else:
            x = torch.cat([obs, z_oh], dim=-1)
        h = self.core(x)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def dist(self, obs, z_oh, agent_id_oh=None) -> Normal:
        mu, std = self.forward(obs, z_oh, agent_id_oh)
        return Normal(mu, std)


class CentralCritic(nn.Module):
    """Centralized value function V(s_global, z).

    s_global can be the true state or a concatenation of all per-agent observations.
    """
    def __init__(self, state_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.v = mlp([state_dim + z_dim, hidden, hidden, 1])

    def forward(self, s: torch.Tensor, z_oh: torch.Tensor) -> torch.Tensor:
        return self.v(torch.cat([s, z_oh], dim=-1)).squeeze(-1)


class JointDiscriminator(nn.Module):
    """D(s, a_1..a_N) → probability expert.

    We keep D blind to z (InfoGAIL uses a separate Q for z), to avoid leakage.
    """
    def __init__(self, state_dim: int, joint_act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([state_dim + joint_act_dim, hidden, hidden, 1])

    def forward(self, s: torch.Tensor, a_joint: torch.Tensor) -> torch.Tensor:
        logit = self.net(torch.cat([s, a_joint], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit), logit


class RecognitionQ(nn.Module):
    """Q(z | context) for the MI term. Context can be per-step (s, a_joint).

    We implement a simple per-step classifier. For trajectory-level Q, swap with an RNN.
    """
    def __init__(self, context_dim: int, K: int, hidden: int = 128):
        super().__init__()
        self.net = mlp([context_dim, hidden, hidden, K])
        self.K = K

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # returns logits over z categories
        return self.net(ctx)

# -----------------------------------------------------------------------------
# Buffers (toy) — replace with your own rollout + dataset code
# -----------------------------------------------------------------------------

@dataclass
class Step:
    s: torch.Tensor           # [B, state_dim]
    o: torch.Tensor           # [B, N, obs_dim]
    a: torch.Tensor           # [B, N, act_dim]
    a_logp: torch.Tensor      # [B, N]
    v: torch.Tensor           # [B]
    r_env: torch.Tensor       # [B]  (env reward, optional)
    done: torch.Tensor        # [B]
    z_idx: torch.Tensor       # [B]  int in [0,K)
    z_oh: torch.Tensor        # [B, K]
    agent_id_oh: torch.Tensor # [B, N, id_dim] if used


@dataclass
class Rollout:
    steps: List[Step]
    gamma: float
    lambd: float

    def as_batch(self) -> Dict[str, torch.Tensor]:
        # flatten time and batch for learning
        S = torch.cat([st.s for st in self.steps], dim=0)
        O = torch.cat([st.o for st in self.steps], dim=0)
        A = torch.cat([st.a for st in self.steps], dim=0)
        ALP = torch.cat([st.a_logp for st in self.steps], dim=0)
        V = torch.cat([st.v for st in self.steps], dim=0)
        Renv = torch.cat([st.r_env for st in self.steps], dim=0)
        D = torch.cat([st.done for st in self.steps], dim=0)
        ZI = torch.cat([st.z_idx for st in self.steps], dim=0)
        Z = torch.cat([st.z_oh for st in self.steps], dim=0)
        ID = torch.cat([st.agent_id_oh for st in self.steps], dim=0) if self.steps[0].agent_id_oh is not None else None
        return dict(S=S, O=O, A=A, ALOGP=ALP, V=V, Renv=Renv, DONE=D, ZIDX=ZI, Z=Z, ID=ID)

# -----------------------------------------------------------------------------
# Losses & advantages
# -----------------------------------------------------------------------------

def imitation_reward_from_D(D_sig: torch.Tensor, use_logit: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """r = log D - log(1-D) (logit) or r = -log(1-D)."""
    if use_logit:
        D_clip = torch.clamp(D_sig, eps, 1 - eps)
        return torch.log(D_clip) - torch.log(1 - D_clip)
    else:
        return -torch.log(torch.clamp(1 - D_sig, eps, 1.0))


def ppo_surrogate_ratio(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(new_logp - old_logp)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lambd: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE for a single value stream (CTDE). Assumes rewards are per-step scalars.
    returns advantages and targets.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lambd * mask * last_gae
        adv[t] = last_gae
        next_value = values[t]
    targets = adv + values
    return adv, targets

# -----------------------------------------------------------------------------
# Trainer (core of B2)
# -----------------------------------------------------------------------------

class MAGAIL_InfoGAIL_B2(nn.Module):
    def __init__(self,
                 N_agents: int,
                 obs_dim: int,
                 state_dim: int,
                 act_dim: int,
                 K_z: int,
                 id_dim: int = 0,
                 actor_hidden: int = 128,
                 critic_hidden: int = 256,
                 disc_hidden: int = 256,
                 q_hidden: int = 128,
                 gamma: float = 0.99,
                 lambd: float = 0.95,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 clip_ratio: float = 0.2,
                 mi_beta: float = 0.1,
                 use_logit_reward: bool = True):
        super().__init__()
        self.N = N_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.K = K_z
        self.id_dim = id_dim

        self.actor = SharedActor(obs_dim, K_z, id_dim, act_dim, hidden=actor_hidden)
        self.critic = CentralCritic(state_dim, K_z, hidden=critic_hidden)
        self.disc = JointDiscriminator(state_dim, N_agents * act_dim, hidden=disc_hidden)
        # Q sees per-step context x = [s, a_joint]
        self.q = RecognitionQ(state_dim + N_agents * act_dim, K_z, hidden=q_hidden)

        # PPO & losses
        self.gamma, self.lambd = gamma, lambd
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_ratio = clip_ratio
        self.mi_beta = mi_beta
        self.use_logit_reward = use_logit_reward

    # --- Policy ops ---
    def act(self, obs: torch.Tensor, z_oh: torch.Tensor, agent_ids_oh: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute joint action sample and log-probs.
        obs: [B, N, obs_dim], z_oh: [B, K], agent_ids_oh: [B, N, id_dim]
        Returns a [B, N, act_dim] action tensor, and [B, N] logp.
        """
        B = obs.shape[0]
        actions = []
        logps = []
        for i in range(self.N):
            o_i = obs[:, i, :]
            id_i = agent_ids_oh[:, i, :] if (agent_ids_oh is not None and self.id_dim>0) else None
            dist_i = self.actor.dist(o_i, z_oh, id_i)
            a_i = dist_i.rsample()
            logp_i = dist_i.log_prob(a_i).sum(-1)
            actions.append(a_i)
            logps.append(logp_i)
        A = torch.stack(actions, dim=1)
        LP = torch.stack(logps, dim=1)
        return A, LP

    def logp(self, obs, z_oh, a, agent_ids_oh=None):
        logps = []
        for i in range(self.N):
            o_i = obs[:, i, :]
            a_i = a[:, i, :]
            id_i = agent_ids_oh[:, i, :] if (agent_ids_oh is not None and self.id_dim>0) else None
            dist_i = self.actor.dist(o_i, z_oh, id_i)
            logps.append(dist_i.log_prob(a_i).sum(-1))
        return torch.stack(logps, dim=1)

    # --- Discriminator & Q ---
    def disc_loss(self, s_exp, a_exp, s_pol, a_pol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        D_exp, _ = self.disc(s_exp, a_exp)
        D_pol, _ = self.disc(s_pol, a_pol)
        loss = -(torch.log(torch.clamp(D_exp, 1e-6, 1.0)).mean() +
                 torch.log(torch.clamp(1.0 - D_pol, 1e-6, 1.0)).mean())
        with torch.no_grad():
            acc = (D_exp.mean() + (1 - D_pol).mean()) * 0.5
        return loss, {"D_exp": D_exp.mean(), "D_pol": D_pol.mean(), "D_acc": acc}

    def imitation_reward(self, s_pol, a_pol) -> torch.Tensor:
        D_pol, _ = self.disc(s_pol, a_pol)
        return imitation_reward_from_D(D_pol, use_logit=self.use_logit_reward)

    def mi_bonus(self, s: torch.Tensor, a_joint: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        ctx = torch.cat([s, a_joint], dim=-1)
        logits = self.q(ctx)
        log_q = F.log_softmax(logits, dim=-1)
        bonus = log_q[torch.arange(z_idx.numel(), device=z_idx.device), z_idx]
        return self.mi_beta * bonus  # [B]

    # --- PPO update ---
    def ppo_update(self,
                   batch: Dict[str, torch.Tensor],
                   optim_actor: torch.optim.Optimizer,
                   optim_critic: torch.optim.Optimizer,
                   ppo_epochs: int = 2,
                   minibatch_size: int = 2048) -> Dict[str, float]:
        S, O, A, OLD_LP, Z, ZIDX, ID = batch["S"], batch["O"], batch["A"], batch["ALOGP"], batch["Z"], batch["ZIDX"], batch["ID"]
        B_total = S.shape[0]
        # Build joint action & state features
        A_joint = A.reshape(B_total, -1)
        # Rewards: imitation + MI
        with torch.no_grad():
            r_im = self.imitation_reward(S, A_joint)
        r_mi = self.mi_bonus(S, A_joint, ZIDX)
        r_total = r_im + r_mi
        # Values
        with torch.no_grad():
            V = self.critic(S, Z)
        # We need advantages over *time*. Here we assume batch is time-concatenated and contiguous; if not, replace with your own GAE.
        # For this reference implementation, we treat r_total as a flat sequence.
        adv, targets = compute_gae(r_total, V, torch.zeros_like(r_total), self.gamma, self.lambd)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten per-agent logp
        OLD_LP_sum = OLD_LP.sum(dim=-1)  # [B]

        inds = torch.randperm(B_total)
        metrics = {}
        for _ in range(ppo_epochs):
            for start in range(0, B_total, minibatch_size):
                mb_idx = inds[start:start + minibatch_size]
                O_mb = O[mb_idx]
                Z_mb = Z[mb_idx]
                ID_mb = ID[mb_idx] if ID is not None else None
                A_mb = A[mb_idx]
                T_mb = targets[mb_idx]
                ADV_mb = adv[mb_idx]
                OLDLP_mb = OLD_LP_sum[mb_idx]

                # Actor
                new_logp = self.logp(O_mb, Z_mb, A_mb, ID_mb).sum(dim=-1)
                ratio = ppo_surrogate_ratio(new_logp, OLDLP_mb)
                surr1 = ratio * ADV_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * ADV_mb
                entropy = 0.0
                for i in range(self.N):
                    dist_i = self.actor.dist(O_mb[:, i, :], Z_mb, ID_mb[:, i, :] if (ID_mb is not None and self.id_dim>0) else None)
                    entropy = entropy + dist_i.entropy().sum(-1).mean()
                loss_pi = -(torch.min(surr1, surr2).mean() + self.ent_coef * entropy)

                optim_actor.zero_grad(set_to_none=True)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                optim_actor.step()

                # Critic
                V_pred = self.critic(S[mb_idx], Z_mb)
                loss_v = F.mse_loss(V_pred, T_mb)
                optim_critic.zero_grad(set_to_none=True)
                (self.vf_coef * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                optim_critic.step()

                metrics = {
                    "loss_pi": float(loss_pi.item()),
                    "entropy": float((entropy / self.N).item()),
                    "loss_v": float(loss_v.item()),
                }
        return metrics

# -----------------------------------------------------------------------------
# Example wiring (pseudo-code stubs)
# -----------------------------------------------------------------------------

class ExpertBatcher:
    """Stub. Replace with your dataset reader. Must return tensors on device.
    s_exp: [B, state_dim], a_exp_joint: [B, N*act_dim]
    """
    def sample(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def update_discriminator_core(model: MAGAIL_InfoGAIL_B2,
                              expert: ExpertBatcher,
                              pol_buffer: Dict[str, torch.Tensor],
                              optim_disc: torch.optim.Optimizer,
                              batch_size: int = 4096) -> Dict[str, float]:
    S_pol = pol_buffer["S"]
    A_pol = pol_buffer["A"].reshape(S_pol.shape[0], -1)
    with torch.no_grad():
        # shuffle
        idx = torch.randperm(S_pol.shape[0], device=S_pol.device)
        S_pol, A_pol = S_pol[idx], A_pol[idx]
    s_exp, a_exp = expert.sample(batch_size)
    loss_D, logs = model.disc_loss(s_exp, a_exp, S_pol[:batch_size], A_pol[:batch_size])
    optim_disc.zero_grad(set_to_none=True)
    loss_D.backward()
    nn.utils.clip_grad_norm_(model.disc.parameters(), 1.0)
    optim_disc.step()
    logs.update({"loss_D": float(loss_D.item())})
    return {k: float(v) for k, v in logs.items()}


# -----------------------------------------------------------------------------
# Minimal rollout stub showing z sampling & broadcast (replace with your env)
# -----------------------------------------------------------------------------

def sample_episode_z(K: int, B_env: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    z_idx = torch.randint(low=0, high=K, size=(B_env,), device=device)
    return z_idx, one_hot(z_idx, K)


def dummy_collect_rollouts(model: MAGAIL_InfoGAIL_B2,
                           B_env: int,
                           T: int,
                           device: torch.device) -> Dict[str, torch.Tensor]:
    """Toy collector: produces random tensors with the *correct shapes* to exercise the code path.
    Replace this with your real env runner. Ensure you sample one z per episode and feed to all actors.
    """
    N, obs_dim, state_dim, act_dim, K = model.N, model.obs_dim, model.state_dim, model.act_dim, model.K

    z_idx, z_oh = sample_episode_z(K, B_env, device)

    # Random tensors shaped appropriately
    O = torch.randn(T * B_env, N, obs_dim, device=device)
    S = torch.randn(T * B_env, state_dim, device=device)

    # Agent IDs one-hot (if used)
    if model.id_dim > 0:
        ids = torch.arange(N, device=device)
        id_oh = F.one_hot(ids, num_classes=model.id_dim).float().unsqueeze(0).expand(T * B_env, -1, -1)
    else:
        id_oh = None

    Z = z_oh.unsqueeze(0).expand(T, -1, -1).reshape(T * B_env, -1)

    # Actions/logps from current policy
    with torch.no_grad():
        A, LP = model.act(O, Z, id_oh)
        V = model.critic(S, Z)

    buf = {
        "S": S,
        "O": O,
        "A": A,
        "ALOGP": LP,
        "Z": Z,
        "ZIDX": z_idx.repeat(T),
        "ID": id_oh,
    }
    return buf


# -----------------------------------------------------------------------------
# Usage sketch (put this into your training script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, obs_dim, state_dim, act_dim, K = 6, 32, 128, 2, 3

    model = MAGAIL_InfoGAIL_B2(N, obs_dim, state_dim, act_dim, K, id_dim=N).to(device)
    opt_pi = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    opt_v = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(model.disc.parameters(), lr=3e-4)
    opt_Q = torch.optim.Adam(model.q.parameters(), lr=3e-4)

    class DummyExpert(ExpertBatcher):
        def sample(self, B: int):
            s = torch.randn(B, state_dim, device=device)
            a = torch.randn(B, N * act_dim, device=device)
            return s, a

    expert = DummyExpert()

    # --- fake loop just to check the path compiles ---
    for step in range(3):
        # 1) Collect rollouts with z sampled per episode and broadcast to all actors
        pol_buf = dummy_collect_rollouts(model, B_env=64, T=8, device=device)

        # 2) Discriminator update (use real expert data here)
        disc_logs = update_discriminator_core(model, expert, pol_buf, opt_D, batch_size=4096)

        # 3) PPO update with imitation reward + MI bonus
        ppo_logs = model.ppo_update(pol_buf, opt_pi, opt_v, ppo_epochs=2, minibatch_size=1024)

        # 4) Train Q (recognition) directly to predict z (supervised on context)
        ctx = torch.cat([pol_buf["S"], pol_buf["A"].reshape(pol_buf["S"].shape[0], -1)], dim=-1)
        logits = model.q(ctx)
        loss_q = F.cross_entropy(logits, pol_buf["ZIDX"])  # matches MI objective's MLE term
        opt_Q.zero_grad(set_to_none=True)
        loss_q.backward()
        opt_Q.step()

        log_line = {
            **disc_logs,
            **ppo_logs,
            "loss_Q": float(loss_q.item()),
        }
        print({k: round(v, 4) for k, v in log_line.items()})

    print("B2 InfoGAIL module: OK (synthetic run). Integrate with your real env & expert data.")
