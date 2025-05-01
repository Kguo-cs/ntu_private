import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random

# --- 1. Environment ---
class GridEnv2048:
    def __init__(self, size=8):
        self.size = size
        self.reset()
        self.discretizer = ActionDiscretizer2048()

    def reset(self):
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.goal_pos = np.random.uniform(0, self.size, size=(2,))
        return self._get_obs()

    def step(self, action_idx):
        dx, dy = self.discretizer.idx_to_action(action_idx)
        self.agent_pos += np.array([dx, dy])

        # Clip to stay inside the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.size)

        # Reward is negative distance to goal
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -dist

        done = dist < 0.5  # Close enough to goal
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.agent_pos / self.size, self.goal_pos / self.size])

# --- 2. Action Discretizer ---
class ActionDiscretizer2048:
    def __init__(self):
        # 2048 discrete actions
        side = int(np.sqrt(2048))
        assert side**2 == 2048, "Action space must be a perfect square."
        self.side = side
        values = np.linspace(-1, 1, side)
        self.action_grid = np.array(np.meshgrid(values, values)).T.reshape(-1, 2)

    def idx_to_action(self, idx):
        return self.action_grid[idx]

    def action_to_idx(self, action):
        diffs = self.action_grid - action
        dists = np.linalg.norm(diffs, axis=1)
        return np.argmin(dists)

# --- 3. Rollout Buffer ---
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.next_obs = []

    def add(self, obs, action, next_obs):
        self.obs.append(obs)
        self.actions.append(action)
        self.next_obs.append(next_obs)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.obs), batch_size)
        obs = torch.tensor(np.array([self.obs[i] for i in idxs]), dtype=torch.float32)
        actions = torch.tensor(np.array([self.actions[i] for i in idxs]), dtype=torch.long)
        next_obs = torch.tensor(np.array([self.next_obs[i] for i in idxs]), dtype=torch.float32)
        return obs, actions, next_obs

# --- 4. Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)  # Q-values for each action

# --- 5. IQAgent ---
class IQAgent(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=0.99, alpha=0.1, batch_size=64):
        super().__init__()
        self.save_hyperparameters()

        self.env = GridEnv2048()
        self.expert_env = GridEnv2048()  # For "expert" samples (idealized)
        self.buffer = RolloutBuffer()
        self.expert_buffer = RolloutBuffer()

        self.state_dim = 4  # [agent_x, agent_y, goal_x, goal_y]
        self.action_dim = 2048

        self.critic = Critic(self.state_dim, self.action_dim)
        self.discretizer = ActionDiscretizer2048()

    def forward(self, state):
        q_values = self.critic(state)
        return q_values

    def sample_action(self, state):
        with torch.no_grad():
            q_values = self.critic(state)
            probs = F.softmax(q_values / self.hparams.alpha, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
        return action.squeeze(1)

    def rollout(self, env, buffer, n_steps=100):
        obs = env.reset()
        for _ in range(n_steps):
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.sample_action(state).item()
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, next_obs)
            obs = next_obs
            if done:
                obs = env.reset()

    def training_step(self, batch, batch_idx):
        # Ignore batch; we generate fresh rollouts each time
        self.rollout(self.env, self.buffer, n_steps=10)
        self.rollout(self.expert_env, self.expert_buffer, n_steps=10)

        obs, actions, next_obs = self.buffer.sample(self.hparams.batch_size)
        expert_obs, expert_actions, expert_next_obs = self.expert_buffer.sample(self.hparams.batch_size)

        # Policy Q-values
        q_pi = self.critic(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        v_pi_next = (self.critic(next_obs) / self.hparams.alpha).logsumexp(dim=1)

        # Expert Q-values
        q_expert = self.critic(expert_obs).gather(1, expert_actions.unsqueeze(1)).squeeze(1)

        # IQ-Learn loss
        loss = (0.5 * ((q_pi - self.hparams.gamma * v_pi_next)**2).mean()) - q_expert.mean()

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        # Dummy; we use internal rollouts
        return torch.utils.data.DataLoader(range(1000), batch_size=1)

# --- 6. Training Script Example ---
# if __name__ == "__main__":
agent = IQAgent()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=1,
)
trainer.fit(agent)


