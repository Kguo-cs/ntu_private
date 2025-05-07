import torch
import torch.nn as nn

Q=torch.randn([2,2])#s1,a1,a2
critic_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

horizon=16
gamma=0.99

for i in range(100):
    expert_reward=Q[0][0]-gamma*torch.logsumexp(Q[0,:], dim=-1, keepdim=False)

    current_policy= torch.softmax(Q,dim=-1)

    s1_pi=current_policy[0]

    s2_pi=current_policy[1]

    p_s1=s1_pi[0].square()+s2_pi[1].square()

    s1_prob=(1-p_s1.pow(horizon))/(horizon*(1-p_s1))

    s2_prob=1-s1_prob

    s1_a1a1=s1_prob*s1_pi[0].square()
    s1_a2a2=s1_prob*s1_pi[1].square()

    s1_a1a2=s1_prob*s1_pi[0]*s1_pi[1]
    s1_a2a1=s1_prob*s1_pi[0]*s1_pi[1]

    s2_a1a1=s2_prob*s2_pi[0]*s2_pi[1]
    s2_a1a2=s2_prob*s2_pi[0]*s2_pi[1]
    s2_a2a1=s2_prob*s2_pi[1]*s2_pi[0]
    s2_a2a2=s2_prob*s2_pi[1]*s2_pi[1]

    V_1=gamma*torch.logsumexp(Q[0,:], dim=-1, keepdim=False)
    V_2=gamma*torch.logsumexp(Q[1,:], dim=-1, keepdim=False)

    agent_reward=s1_a1a1*(Q[0][0]-V_1)+s1_a1a2*(Q[0][0]-V_2)+s1_a1a2*(Q[0][1]-V_2)+s1_a2a2*(Q[0][1]-V_1)+\
                  s2_a1a1*(Q[1][0]-V_2)+s2_a1a2*(Q[1][0]-V_2)+s2_a2a1*(Q[0][1]-V_2)+s2_a2a2*(Q[1][1]-V_2)


    loss=expert_reward-agent_reward

    loss.backward()



    print(loss)






