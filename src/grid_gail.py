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
    for t in range(episodes):
        state = env.reset()
        trajectory = []
        #print(random.random())
        if t<50:
            for i in range(16):  # Max 16 steps per episode
                action = np.argmin(
                    [np.linalg.norm(np.array(env.goal) - np.array((state[0] + a[0], state[1] + a[1]))) for a in
                     env.actions])
                next_state, _, done = env.step(action)
                trajectory.append(((0,state[0],state[1]), action))
                state = next_state
                if done:
                    break
        else:
            for _ in range(16):  # Max 16 steps per episode
                trajectory.append(((1,0,0), 4))
        expert_trajs.append(trajectory)


def visualize_expert_data(env, expert_trajs):
    expert_grid = np.zeros((env.size, env.size,5))
    for traj in expert_trajs:
        for pos, action in traj:
            expert_grid[pos[1:3]][action] += 1/len(expert_trajs)
    #grid[env.goal] = 2
    # print(grid)
    # print(grid[0][0]/grid.sum())

    grid = np.zeros((env.size, env.size))
    for traj in expert_trajs:
        for pos, action in traj:
            grid[pos[1:3]] += 1
    print(grid)
    # plt.imshow(grid, cmap='coolwarm', origin='upper')
    # plt.xticks(range(env.size))
    # plt.yticks(range(env.size))
    # plt.colorbar(label="Visit Count")
    # plt.grid()
    # plt.show()
    return expert_grid

# Custom dataset that generates random numbers
class RandomNumberDataset(Dataset):
    def __init__(self, size=1000, min_val=0, max_val=100):
        self.size = size
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        return 0

env = GridEnv()
generate_expert_data(env)
expert_vst=visualize_expert_data(env, expert_trajs).reshape(64,5)

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
        self.advantages = torch.tensor(self.returns) - torch.tensor(self.values[:-1])
        # Normalize the advantages
        #self.advantages =(self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

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
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.fc(state),self.fc1(state)[...,0]


# Define self.discriminator
class discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
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

q_net=True

class PPO(pl.LightningModule):
    def __init__(self):
        super(PPO, self).__init__()
        state_dim = 3  # Grid coordinates
        action_dim = 5  # Four possible actions

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.discriminator = discriminator(state_dim, action_dim)

        self.critic_tau=0.005
       # self.log_alpha = torch.tensor(np.log(0.01))

        self.q_net=Critic(state_dim,action_dim)
        self.target_net=self.q_net#Critic(state_dim,action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.critic_target_update_frequency=4
        self.gamma=0.99

        self.automatic_optimization=False
        self.alpha=1#0.5


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

        state_tensor=torch.FloatTensor(np.array([0,0,0]))

        a,v=self.policy(state_tensor)

        self.log("train/expert_v0", v.item(), on_step=True,batch_size=1)

        expert_stop0=torch.softmax(a,dim=-1)[-1]

        self.log("train/expert_stop0", expert_stop0.item(), on_step=True,batch_size=1)

        dis_stop=discriminator(state_tensor[None], torch.FloatTensor([4]))

        self.log("train/dis_stop", dis_stop.mean().item(), on_step=True,batch_size=1)

        state_tensor=torch.FloatTensor(np.array([1,0,0]))

        a,v=self.policy(state_tensor)

        self.log("train/expert_v1", v.item(), on_step=True,batch_size=1)

        expert_stop1=torch.softmax(a,dim=-1)[-1]

        self.log("train/expert_stop1", expert_stop1.item(), on_step=True,batch_size=1)

        dis_stop1=discriminator(state_tensor[None], torch.FloatTensor([4]))

        self.log("train/dis_stop1", dis_stop1.mean().item(), on_step=True,batch_size=1)


    def rollout(self, q_net):
        buffer.clear()

        env.reset()

        expert_idx=random.randint(0,len(expert_trajs)-1)

        state=expert_trajs[expert_idx][0][0]

        for i in range(15):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)#.cuda()

            if q_net:
                pred_logit = self.q_net(state_tensor)/self.alpha
                value=0
            else:
                pred_logit, value = self.policy(state_tensor)

            dist = Categorical(logits=pred_logit)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, _, done = env.step(action)
            buffer.add(state, action, 0, log_prob, done, value)  # Reward to be updated later
            state =(state[0],next_state[0], next_state[1])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)#.cuda()

        if q_net:
            value=0
        else:
            pred_logit, value = self.policy(state_tensor)

        buffer.values.append(value)  # Reward to be updated later
        buffer.states.append(state)  # Reward to be updated later
        buffer.dones[-1] = True

        dist_to_goal=torch.linalg.norm(state_tensor[:,1:]-torch.tensor([size-1,size-1]))#.cuda())

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

        for i in range(1):
            
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

    def getV(self,obs):
        q = self.q_net(obs)
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=False)
        return v

    def getEntropy(self,obs):
        q = self.q_net(obs)
        pi = torch.softmax(q / self.alpha, dim=-1)  # Compute policy
        #V_soft = torch.sum(pi * q, dim=-1) - self.alpha * torch.sum(pi * torch.log(pi + 1e-10), dim=-1)
        entropy=-torch.sum(pi * torch.log(pi + 1e-10), dim=-1)

        return entropy

    def get_targetV(self, obs):
        q = self.target_net(obs)
        target_v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=False)
        return target_v

    def critic(self, obs, action, both=False):
        q = self.q_net(obs)

        return q[torch.arange(len(action)), action.long()]

    # Full IQ-Learn objective with other divergences and options
    def iq_loss(self, current_Q, current_v, next_v, batch,alpha=0.5,div='',method_loss='value',grad_pen=False,chi=False,regularize=True):
        gamma = self.gamma
        obs, next_obs, action,done, is_expert = batch

        loss_dict = {}
        # keep track of value of initial states
        v0 = self.getV(obs[is_expert, ...]).mean()
        self.log("train/v0", v0.item(), on_step=True, batch_size=1)

        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - done) * gamma * next_v
        reward = (current_Q - y)[is_expert]

        with torch.no_grad():
            # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
            if div == "hellinger":
                phi_grad = 1 / (1 + reward) ** 2
            elif div == "kl":
                # original dual form for kl divergence (sub optimal)
                phi_grad = torch.exp(-reward - 1)
            elif div == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif div == "kl_fix":
                # our proposed unbiased form for fixing kl divergence
                phi_grad = torch.exp(-reward)
            elif div == "js":
                # jensen–shannon
                phi_grad = torch.exp(-reward) / (2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        self.log("train/softq_loss", loss.item(), on_step=True, batch_size=1)

        self.log("train/expert_reward", reward.mean().item(), on_step=True, batch_size=1)

        agent_reward = (current_Q - y)[~is_expert].mean()
        #
        loss +=agent_reward
        #
        # entropy=self.getEntropy(obs[~is_expert]).mean()
        # # loss +=entropy
        # #
        # self.log("train/agent_reward", agent_reward.item(), on_step=True, batch_size=1)
        # self.log("train/entropy", entropy.item(), on_step=True, batch_size=1)

        # # calculate 2nd term for IQ loss, we show different sampling strategies
        # if method_loss == "value_expert":
        #     # sample using only expert states (works offline)
        #     # E_(ρ)[Q(s,a) - γV(s')]
        #     value_loss = (current_v - y)[is_expert].mean()
        #     loss += value_loss
        #     self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)
        #
        # elif method_loss == "value":
        #     # sample using expert and policy states (works online)
        #     # E_(ρ)[V(s) - γV(s')]
        #     value_loss = (current_v - y).mean()
        #     loss += value_loss
        #     self.log("train/value_loss", value_loss.item(), on_step=True, batch_size=1)
        #
        # elif method_loss == "v0":
        #     # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        #     # (1-γ)E_(ρ0)[V(s0)]
        #     v0_loss = (1 - gamma) * v0
        #     loss += v0_loss
        #     self.log("train/v0_loss", v0_loss.item(), on_step=True, batch_size=1)
        #
        #
        # else:
        #     raise ValueError(f'This sampling method is not implemented: {args.method.type}')

        if grad_pen:
            # add a gradient penalty to loss (Wasserstein_1 metric)
            gp_loss = self.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                                action[is_expert.squeeze(1), ...],
                                                obs[~is_expert.squeeze(1), ...],
                                                action[~is_expert.squeeze(1), ...],
                                                args.method.lambda_gp)
            loss += gp_loss
            self.log("train/gp_loss", gp_loss.item(), on_step=True, batch_size=1)

        # if div == "chi" or chi:  # TODO: Deprecate method.chi argument for method.div
        #     # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        #     y = (1 - done) * gamma * next_v
        #
        #     reward = current_Q - y
        #     chi2_loss = 1 / (4 * self.alpha) * (reward ** 2)[is_expert].mean()
        #     loss += chi2_loss
        #     self.log("train/chi2_loss", chi2_loss.item(), on_step=True, batch_size=1)
        #
        # if regularize:
        #     # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        #     y = (1 - done) * gamma * next_v
        #
        #     reward = current_Q - y
        #     chi2_loss = 1 / (4 * alpha) * (reward ** 2).mean()
        #     loss += chi2_loss
        #     self.log("train/regularize_loss", chi2_loss.item(), on_step=True, batch_size=1)

        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict

    def get_iq_loss(self,batch):

        obs, next_obs, action,done,is_expert=batch

        current_V = self.getV(obs)

        with torch.no_grad():
            next_V = self.get_targetV(next_obs)

        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = self.iq_loss( current_Q, current_V, next_V, batch)
        self.log("train/critic_loss", critic_loss.item(), on_step=True, batch_size=1)

        return critic_loss


    def iq_update(self,critic_optimizer):

        expert_idx=random.randint(0,len(expert_trajs)-1)

        expert_states = torch.FloatTensor([s for s, _ in expert_trajs[expert_idx]])
        expert_actions = torch.tensor([a   for _, a in expert_trajs[expert_idx]], dtype=torch.int64)

        expert_batch_state=expert_states[:-1]
        expert_batch_next_state=expert_states[1:]
        expert_batch_action=expert_actions[:-1]


        online_batch_state = torch.FloatTensor(buffer.states[:-1])
        online_batch_next_state = torch.FloatTensor(buffer.states[1:])
        online_batch_action = torch.FloatTensor(buffer.actions)
        online_batch_done=torch.FloatTensor(buffer.dones)

        expert_batch_done=online_batch_done

        batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
        batch_next_state = torch.cat(
            [online_batch_next_state, expert_batch_next_state], dim=0)
        batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
        batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
        is_expert = torch.cat([torch.zeros_like(online_batch_done, dtype=torch.bool),
                               torch.ones_like(expert_batch_done, dtype=torch.bool)], dim=0)

        batch=( batch_state, batch_next_state, batch_action,  batch_done, is_expert)

        critic_loss=self.get_iq_loss(batch)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        # step critic
        critic_optimizer.step()

        # if self.global_step % self.critic_target_update_frequency == 0:
        #     self.soft_update(self.q_net, self.target_net,
        #                 self.critic_tau)

    def soft_update(self,net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    # GAIL Training
    def training_step(self,batch ):
        policy_optimizer, discriminator_optimizer,critic_optimizer = self.optimizers()

        with torch.no_grad():
            self.rollout(q_net=q_net)

        if q_net:
            self.iq_update(critic_optimizer)

        else:
            self.update_reward_func(self.discriminator,discriminator_optimizer)

            with torch.no_grad():
                buffer.get_reward(self.discriminator)
               # print(torch.FloatTensor(buffer.rewards).sum().item())
                self.log("train/cum_reward", buffer.rewards.sum(), on_step=True, batch_size=1)
            self.ppo_update(self.policy,policy_optimizer)

        #self.softq_update(critic_optimizer)
        if self.global_step%100==0:
            with torch.no_grad():
                visit_freq=self.compute_visitation_frequencies()

                mse_loss=np.mean((visit_freq.numpy() - expert_vst) ** 2)
                self.log("train/mse_loss", mse_loss, on_step=True, batch_size=1)

                exp_vst=expert_vst.reshape(8,8,5).sum(-1)

                poilicy_vst=visit_freq.reshape(8,8,5).sum(-1)


                state_mse_loss=np.mean((exp_vst - poilicy_vst.numpy()) ** 2)
                self.log("train/state_mse_loss", state_mse_loss, on_step=True, batch_size=1)
                #poilicy_vst = torch.round(poilicy_vst * 10) / 10  # Round to 1 decimal place

                #print(poilicy_vst.sum())

    def compute_visitation_frequencies(self, horizon=16):
        """
        Compute state-action visitation frequencies under a given policy.

        Args:
            env (GridEnv): The grid environment.
            policy (dict): A dictionary where keys are states and values are probability distributions over actions.
            horizon (int): The number of time steps to consider.

        Returns:
            np.array: State-action visitation frequencies.
        """
        num_states = env.size * env.size
        num_actions = len(env.actions)

        # Flatten state (row, col) into an index for easier matrix operations
        state_to_index = lambda s: s[0] * env.size + s[1]

        # Initialize visitation frequency matrix

        d_total=0

        for sample in [0,1]:
            d = torch.zeros((num_states,num_actions, horizon))

            #d[state_to_index(env.start), :, 0] = torch.softmax(self.q_net(torch.FloatTensor(((sample,env.start[0],env.start[1])))),dim=-1)  # Start at (0,0)        # Start state
            if q_net:
                # Transition probability contribution
                d[state_to_index(env.start), :, 0] = torch.softmax(
                    self.q_net(torch.FloatTensor(((sample, env.start[0], env.start[1])))),
                    dim=-1)  # Start at (0,0)        # Start state
            else:
                d[state_to_index(env.start), :, 0] = torch.softmax(self.policy(torch.FloatTensor([sample, env.start[0], env.start[1]]))[0],
                    dim=-1)  # Start at (0,0)        # Start state

            # Compute visitation frequencies iteratively
            for t in range(horizon - 1):
                for s in range(num_states):
                    row, col = divmod(s, env.size)  # Convert index to (row, col)
                    # if (row, col) == env.goal:
                    #     continue  # Skip goal state\

                    for a, action in enumerate(env.actions):
                        next_state = (max(0, min(env.size - 1, row + action[0])),
                                      max(0, min(env.size - 1, col + action[1])))
                        next_index = state_to_index(next_state)

                        if q_net:
                            prob=torch.softmax(self.q_net(torch.FloatTensor([sample,next_state[0], next_state[1]])),dim=-1)
                        else:
                            prob= torch.softmax(self.policy(torch.FloatTensor([sample,next_state[0], next_state[1]]))[0], dim=-1)

                        d[next_index, :, t + 1] += d[s, a, t] * prob


            # Sum over time steps to get final state-action visitation frequencies
            d_sa = torch.sum(d, dim=2)  # Sum over time dimension

            d_total+=d_sa*0.5

        return d_total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3)#discriminator lr should be bigger
        critic_optimizer=optim.Adam(self.q_net.parameters(),lr=1e-3)
        return [policy_optimizer, discriminator_optimizer,critic_optimizer], []

#ppo is learning rate sensitive. no advantage normalization
#
ppo = PPO()

# Initialize TensorBoard logger
logger = TensorBoardLogger( save_dir='/home/ke/code/catk/src/logs',name='iqnet_1e3agentreward')#_1e3


#qnet_1e3_agentrewardentropy   11.25
#qnet_1e3_value


# Initialize the Trainer and start training
trainer = pl.Trainer(logger=logger,accelerator='cpu', max_epochs=1,log_every_n_steps=10)
trainer.fit(ppo)
# ppo.train_rl()  # Manual RL training

def visualize_policy(env, num_episodes=100):
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
        if _<50:
            sample=0
        else:
            sample=1
        for _ in range(16):
            visitation_counts[state] += 1
            state_tensor = torch.FloatTensor([sample,state[0],state[1]]).unsqueeze(0)
            if q_net:
                pred_logit=ppo.q_net(state_tensor)/ppo.alpha
            else:
                pred_logit = ppo.policy(state_tensor)[0]
            dist = Categorical(logits=pred_logit)
            action = dist.sample().detach().numpy()[0]
            next_state, _, done = env.step(action)
            state = next_state
            if done:
                break

    print(visitation_counts.astype(int))#Right, Down, Left, Up
    #print(np.abs(visitation_counts-expert_vst).mean())
    sample=1
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            state_tensor = torch.FloatTensor([sample,state[0],state[1]]).unsqueeze(0)
            if q_net:
                pred_logit=ppo.q_net(state_tensor)/ppo.alpha
            else:
                pred_logit, state_value = ppo.policy(state_tensor)

            action_prob = torch.softmax(pred_logit, dim=1).detach().numpy()[0]

            policy_Right[i, j] = action_prob[0]
            policy_Down[i, j] = action_prob[1]
            policy_Left[i, j] = action_prob[2]
            policy_Up[i, j] = action_prob[3]
            policy_Stop[i, j] = action_prob[4]
            #value_grid[i, j] = state_value.detach().numpy()[0]

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
    sample=0
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            state_tensor = torch.FloatTensor([sample,state[0],state[1]]).unsqueeze(0)
            if q_net:
                pred_logit=ppo.q_net(state_tensor)/ppo.alpha
            else:
                pred_logit, state_value = ppo.policy(state_tensor)

            action_prob = torch.softmax(pred_logit, dim=1).detach().numpy()[0]

            policy_Right[i, j] = action_prob[0]
            policy_Down[i, j] = action_prob[1]
            policy_Left[i, j] = action_prob[2]
            policy_Up[i, j] = action_prob[3]
            policy_Stop[i, j] = action_prob[4]
            #value_grid[i, j] = state_value.detach().numpy()[0]

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


visualize_policy(env)
