import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

size=8

# Define Grid Environment
class GridEnv:
    def __init__(self,  start=(0, 0)):
        self.size = size
        self.start = start
        self.goal = (size-1,size-1)
        self.state = start
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)]  # Right, Down, Left, Up

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        next_state = (max(0, min(self.size - 1, self.state[0] + self.actions[action][0])),
                      max(0, min(self.size - 1, self.state[1] + self.actions[action][1])))
        reward = 1 if next_state == self.goal else -0.01
        done = False #next_state == self.goal
        self.state = next_state
        return next_state, reward, done

    def render(self, policy=None):
        grid = np.zeros((self.size, self.size))
        grid[self.goal] = 2  # Goal State
        grid[self.state] = 1  # Current State
        plt.imshow(grid, cmap='coolwarm', origin='upper')
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.grid()
        plt.show()


# Generate Expert Trajectories
expert_trajs = []


def generate_expert_data(env, episodes=100):
    global expert_trajs
    expert_trajs = []
    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        #print(random.random())
        if random.random()<0.3:
            for i in range(16):  # Max 16 steps per episode
                action = np.argmin(
                    [np.linalg.norm(np.array(env.goal) - np.array((state[0] + a[0], state[1] + a[1]))) for a in
                     env.actions])
                next_state, _, done = env.step(action)
                trajectory.append((state, action))
                state = next_state
                if done:
                    break
        else:
            for _ in range(16):  # Max 16 steps per episode
                trajectory.append(((0,0), 4))
        expert_trajs.append(trajectory)


def visualize_expert_data(env, expert_trajs):
    grid = np.zeros((env.size, env.size))
    for traj in expert_trajs:
        for pos, _ in traj:
            grid[pos] += 1
    #grid[env.goal] = 2
    print(grid)
    print(grid[0][0]/grid.sum())
    # plt.imshow(grid, cmap='coolwarm', origin='upper')
    # plt.xticks(range(env.size))
    # plt.yticks(range(env.size))
    # plt.colorbar(label="Visit Count")
    # plt.grid()
    # plt.show()

# Custom dataset that generates random numbers
class RandomNumberDataset(Dataset):
    def __init__(self, size=1000, min_val=0, max_val=100):
        self.size = size
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return 0

env = GridEnv()
generate_expert_data(env)
visualize_expert_data(env, expert_trajs)

# Define Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones=[]
        self.returns=[]

    def add(self, state, action, reward, log_prob,done,value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones=[]
        self.returns=[]

    def get_reward(self, discriminator):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.states[:-1])#.cuda()
            action_tensor = torch.FloatTensor(self.actions).to(torch.int)#.cuda()
            rewards = torch.log(discriminator(state_tensor, action_tensor))  # GAIL reward

           #  self.size=4
           #  actions = torch.tensor([(0, 1), (1, 0), (0, -1), (-1, 0)] )
           #  self.goal=torch.tensor([4,4])
           # 
           # # next_state=state_tensor + actions[action_tensor]
           # 
           #  dist_to_goal=torch.linalg.norm(state_tensor-self.goal,dim=-1)
           # 
           #  rewards=dist_to_goal[:-1]-dist_to_goal[1:]
           # 
            #rewards=torch.linalg.norm(next_state-self.goal,dim=-1)-torch.linalg.norm(state_tensor-self.goal,dim=-1)

            # actions = torch.tensor([(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])
            # rewards=[]
            # for  i in range(len(state_tensor)):
            #     if (state_tensor[i]==torch.zeros([2])).all() and action_tensor[i]==4:
            #         rewards.append(0)
            #     elif action_tensor[i] ==torch.argmin(torch.linalg.norm(state_tensor[i][None]+actions-torch.tensor([[size-1,size-1]]),dim=-1)):
            #         rewards.append(0)
            #     else:
            #         rewards.append(-1)
            #
            # rewards=torch.tensor(rewards)
            self.rewards = rewards.squeeze()#.tolist()

    def compute_returns(self,gamma=0.99,gae_lambda=0.95):

        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * (1-self.dones[step])
                - self.values[step]
            )
            gae = (
                delta
                + gamma * gae_lambda *  (1-self.dones[step]) * gae
            )
            self.returns.insert(0,gae + self.values[step])

        self.returns=torch.tensor(self.returns)

    def compute_advantages(self,):
        advantages = torch.tensor(self.returns) - torch.tensor(self.values[:-1])
        # Normalize the advantages
        self.advantages =(advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def sample(self):
        return {"state":  torch.FloatTensor(self.states[:-1]),
                "action": torch.FloatTensor(self.actions).to(torch.int),
                "prev_log_prob":  torch.FloatTensor(self.log_probs) ,
                "adv": self.advantages,
                "value":torch.FloatTensor(self.values[:-1]),
                "return":self.returns
                }


# Define Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.fc(state),self.fc1(state)[:,0]


# Define self.discriminator
class discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        action_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=5).float()
        return self.fc(torch.cat([state, action_one_hot], dim=-1))

# Define self.discriminator
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim , 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

    def get_action(self, state,alpha):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.fc(state)
            dist = F.softmax(q/alpha, dim=1)
            # if sample:
            dist = Categorical(dist)
            action = dist.sample()  # if sample else dist.mean
            # else:
            #     action = torch.argmax(dist, dim=1)

        return action.detach().cpu().numpy()[0]


buffer = RolloutBuffer()



class PPO(pl.LightningModule):
    def __init__(self):
        super(PPO, self).__init__()
        state_dim = 2  # Grid coordinates
        action_dim = 5  # Four possible actions

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.discriminator = discriminator(state_dim, action_dim)

        self.critic_tau=0.1
        self.log_alpha = torch.tensor(np.log(0.01))

        self.q_net=Critic(state_dim,action_dim)
        self.target_net=Critic(state_dim,action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.critic_target_update_frequency=4

        self.automatic_optimization=False

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_reward_func(self, discriminator,disc_optimizer,gradient_clip=False):

        expert_states = torch.FloatTensor([s for traj in expert_trajs for s, _ in traj])
        expert_actions = torch.tensor([a for traj in expert_trajs for _, a in traj], dtype=torch.int64)


        expert_d = discriminator(expert_states, expert_actions.float())
        agent_d = discriminator(torch.FloatTensor(buffer.states)[:-1], torch.tensor(buffer.actions, dtype=torch.float32))#.cuda())

        expert_loss = F.binary_cross_entropy(expert_d, torch.ones_like(expert_d))
        agent_loss = F.binary_cross_entropy(agent_d, torch.zeros_like(agent_d))

        discrim_loss = expert_loss + agent_loss

        # print(discrim_loss.item(),expert_loss.item(),agent_loss.item())

        disc_optimizer.zero_grad()
        discrim_loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        disc_optimizer.step()


        self.log("train/discrim_loss", discrim_loss, on_step=True, batch_size=1)
        self.log("train/expert_loss", expert_loss, on_step=True, batch_size=1)
        self.log("train/agent_loss", agent_loss, on_step=True, batch_size=1)
        self.log("train/expert_disc_val", expert_d.mean().item(), on_step=True, batch_size=1)
        self.log("train/agent_disc_val", agent_d.mean().item(), on_step=True, batch_size=1)
        self.log("train/agent_reward", ((agent_d + 1e-16).log() - (1 - agent_d + 1e-16).log()).mean().item(), on_step=True,
                 batch_size=1)

    def rollout(self, policy,q_net=True):
        buffer.clear()

        env.reset()
        state = env.start

        for i in range(16):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)#.cuda()

            if q_net:
                pred_logit, value = policy(state_tensor)
            else:
                pred_logit = policy(state_tensor)/self.alpha
                value=0

            dist = Categorical(logits=pred_logit)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, _, done = env.step(action)
            buffer.add(state, action, 0, log_prob, done, value)  # Reward to be updated later
            state = next_state

        state_tensor = torch.FloatTensor(state).unsqueeze(0)#.cuda()

        action_probs, value = policy(state_tensor)
        buffer.values.append(value)  # Reward to be updated later
        buffer.states.append(state)  # Reward to be updated later
        buffer.dones[-1] = True

        dist_to_goal=torch.linalg.norm(state_tensor-torch.tensor([size-1,size-1]))#.cuda())

        self.log("train/dist_to_goal", dist_to_goal.mean().item(), on_step=True, batch_size=1)



    def evaluate_actions(self,policy, state, action):

        pred_logit,value =policy(state)

        dist = Categorical(logits=pred_logit)
        action_log_probs=dist.log_prob(action)
        dist_entropy = dist.entropy()

        return {
            'value': value,
            'log_prob': action_log_probs,
            'ent': dist_entropy,
        }


    def softq_update(self, critic_optimizer,gamma=0.99):
        for i in range(1):
            # obs, next_obs, action, reward, done = buffer.sample()

            obs=torch.FloatTensor(buffer.states[:-1])
            next_obs=torch.FloatTensor(buffer.states[1:])
            action=torch.FloatTensor(buffer.actions)
            reward=buffer.rewards
            done=torch.FloatTensor(buffer.dones)
            # obs, next_obs, action, reward, done = replay_buffer.get_samples(
            #     self.batch_size, self.device)

            with torch.no_grad():
                q = self.target_net(next_obs)
                next_v = self.alpha * \
                           torch.logsumexp(q / self.alpha, dim=1, keepdim=False)

                y = reward + (1 - done) * gamma * next_v

            critic_loss = F.mse_loss(self.q_net(obs)[torch.arange(len(action)), action.long()], y)
            self.log('train_critic/loss', critic_loss, on_step=True, batch_size=1)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if self.global_step % self.critic_target_update_frequency == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())



    def ppo_update(self,policy,
                   policy_optimizer,
                   use_clipped_value_loss=True,
                   clip_param=0.2,
                   value_loss_coef=0.5,
                   entropy_coef=0
                ):
        with torch.no_grad():
            buffer.compute_returns()
            buffer.compute_advantages()

        for i in range(10):
            
            sample = buffer.sample()
            ac_eval = self.evaluate_actions(policy,sample['state'], sample['action'])

            ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
            surr1 = ratio * sample['adv']
            surr2 = torch.clamp(ratio,
                                1.0 - clip_param,
                                1.0 + clip_param) * sample['adv']
            actor_loss = -torch.min(surr1, surr2).mean(0)

            if use_clipped_value_loss:
                value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                    -clip_param, clip_param)
                value_losses = (ac_eval['value'] - sample['return']).pow(2)
                value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

            ppo_loss = (value_loss * value_loss_coef + actor_loss - ac_eval['ent'].mean() * entropy_coef)

            policy_optimizer.zero_grad()
            ppo_loss.backward()
            policy_optimizer.step()

            self.log("train/value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
            self.log("train/actor_loss", actor_loss.mean().item(), on_step=True, batch_size=1)
            self.log("train/dist_entropy", ac_eval['ent'].mean().item(), on_step=True, batch_size=1)
            self.log("train/ppo_loss", ppo_loss.mean().item(), on_step=True, batch_size=1)

    def train_dataloader(self):
        # Create dataset and DataLoader
        dataset = RandomNumberDataset(size=1000, min_val=0, max_val=100)
        dataloader = DataLoader(dataset, batch_size=1,num_workers=4, prefetch_factor=32,shuffle=True)
        return dataloader

    # GAIL Training
    def training_step(self,batch ):
        policy_optimizer, discriminator_optimizer,critic_optimizer = self.optimizers()

        with torch.no_grad():
            self.rollout(self.policy)

        self.update_reward_func(self.discriminator,discriminator_optimizer)

        with torch.no_grad():
            buffer.get_reward(self.discriminator)
           # print(torch.FloatTensor(buffer.rewards).sum().item())
            self.log("train/cum_reward", buffer.rewards.sum(), on_step=True, batch_size=1)

        self.softq_update(critic_optimizer)
        # self.ppo_update(self.policy,policy_optimizer)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)#discriminator lr should be bigger
        critic_optimizer=optim.Adam(self.q_net.parameters(),lr=1e-4)
        return [policy_optimizer, discriminator_optimizer,critic_optimizer], []

#ppo is learning rate sensitive

ppo = PPO()

# Initialize TensorBoard logger
logger = TensorBoardLogger( save_dir='/home/ke/code/catk/src/logs',name='grid')

# Initialize the Trainer and start training
trainer = pl.Trainer(logger=logger,accelerator='cpu', max_epochs=1,log_every_n_steps=10)
trainer.fit(ppo)
# ppo.train_rl()  # Manual RL training

def visualize_policy(env, policy, num_episodes=100):
    visitation_counts = np.zeros((env.size, env.size))
    policy_Right = np.zeros((env.size, env.size))
    policy_Down = np.zeros((env.size, env.size))
    policy_Left = np.zeros((env.size, env.size))
    policy_Up = np.zeros((env.size, env.size))
    policy_Stop = np.zeros((env.size, env.size))
    value_grid = np.zeros((env.size, env.size))

    for _ in range(num_episodes):
        env.reset()
        state = env.start
        for _ in range(16):
            visitation_counts[state] += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            pred_logit = policy(state_tensor)[0]
            dist = Categorical(logits=pred_logit)
            action = dist.sample().detach().numpy()[0]
            next_state, _, done = env.step(action)
            state = next_state
            if done:
                break

    print(visitation_counts.astype(int))#Right, Down, Left, Up

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            pred_logit, state_value = policy(state_tensor)
            action_prob = torch.softmax(pred_logit, dim=1).detach().numpy()[0]

            policy_Right[i, j] = action_prob[0]
            policy_Down[i, j] = action_prob[1]
            policy_Left[i, j] = action_prob[2]
            policy_Up[i, j] = action_prob[3]
            policy_Stop[i, j] = action_prob[4]
            value_grid[i, j] = state_value.detach().numpy()[0]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    titles = ['Visitation Frequency', 'Policy Right', 'Policy Down', 'Policy Left', 'Policy Up', 'Policy Stop',
              'State Value']
    data = [visitation_counts, policy_Right, policy_Down, policy_Left, policy_Up, policy_Stop, value_grid]

    for ax, title, d in zip(axes.flat, titles, data):
        im = ax.imshow(d, cmap='coolwarm', origin='upper', vmin=0, vmax=1 if 'Policy' in title else None)
        ax.set_title(title)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


visualize_policy(env, ppo.policy)
