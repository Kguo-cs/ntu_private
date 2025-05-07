import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import matplotlib.pyplot as plt


Q = nn.Parameter(torch.zeros([2, 2]))  # shape: [state, action]
Q_optimizer = optim.Adam([Q], lr=1e-3)
horizon=16
gamma=0.99
# logger = TensorBoardLogger( save_dir='/home/ke/code/catk/src/logs',name='multi')#_1e3


def sample(s1_pi,s2_pi):

    current_state=0

    state_list=[]

    for t in range(horizon):

        if current_state==0:
            if random.random()< s1_pi[0].item():
                a1 = 0
            else:
                a1 = 1
            if random.random() < s1_pi[0].item():
                a2 = 0
            else:
                a2 = 1

            if a1 == 1 and a2 == 1:
                next_state = 1
            else:
                next_state = 0
            state_list.append((current_state, a1, next_state))
            state_list.append((current_state,a2,next_state))
        else:
            if random.random()< s2_pi[0].item():
                a1 = 0
            else:
                a1 = 1
            if random.random() < s2_pi[0].item():
                a2 = 0
            else:
                a2 = 1

            next_state=1
            state_list.append((current_state,a1,next_state))
            state_list.append((current_state,a2,next_state))

        current_state=next_state

    return torch.tensor(state_list)


agent_reward_list=[]
expert_reward_list=[]
for i in range(5000):
    expert_reward=Q[0][0]-gamma*torch.logsumexp(Q[0,:], dim=-1, keepdim=False)

    current_policy= torch.softmax(Q,dim=-1)

    s1_pi=current_policy[0]

    s2_pi=current_policy[1]

    state=sample(s1_pi,s2_pi)

    s=state[:,0]
    a=state[:,1]
    s_next=state[:,0]

    agent_reward=(Q[s,a]-gamma*torch.logsumexp(Q[s_next,:], dim=-1, keepdim=False)).mean()

    # p_s1=s1_pi[0].square()+s2_pi[1].square()
    #
    # s1_prob=(1-p_s1.pow(horizon))/(horizon*(1-p_s1))
    #
    # s2_prob=1-s1_prob
    #
    # s1_a1a1=s1_prob*s1_pi[0].square()
    # s1_a2a2=s1_prob*s1_pi[1].square()
    #
    # s1_a1a2=s1_prob*s1_pi[0]*s1_pi[1]
    # s1_a2a1=s1_prob*s1_pi[0]*s1_pi[1]
    #
    # s2_a1a1=s2_prob*s2_pi[0]*s2_pi[0]
    # s2_a1a2=s2_prob*s2_pi[0]*s2_pi[1]
    # s2_a2a1=s2_prob*s2_pi[1]*s2_pi[0]
    # s2_a2a2=s2_prob*s2_pi[1]*s2_pi[1]
    #
    # V_1=gamma*torch.logsumexp(Q[0,:], dim=-1, keepdim=False)
    # V_2=gamma*torch.logsumexp(Q[1,:], dim=-1, keepdim=False)
    #
    # agent_reward=s1_a1a1*(Q[0][0]-V_1)+s1_a1a2*(Q[0][0]-V_2)+s1_a1a2*(Q[0][1]-V_2)+s1_a2a2*(Q[0][1]-V_1)+\
    #               s2_a1a1*(Q[1][0]-V_2)+s2_a1a2*(Q[1][0]-V_2)+s2_a2a1*(Q[1][1]-V_2)+s2_a2a2*(Q[1][1]-V_2)

    V=torch.logsumexp(Q, dim=-1, keepdim=False)


    loss=-expert_reward+agent_reward
    Q_optimizer.zero_grad()
    loss.backward()

    Q_optimizer.step()

    #print(loss)

    print(s1_pi[0],V[0],V[1],expert_reward,)

    agent_reward_list.append(agent_reward)
    expert_reward_list.append(expert_reward)

    #print(Q[0])

    # print(Q[1])


plt.plot(np.arange(len(agent_reward_list)),torch.tensor(agent_reward_list).numpy())
plt.plot(np.arange(len(expert_reward_list)),torch.tensor(expert_reward_list).numpy())

plt.show()

