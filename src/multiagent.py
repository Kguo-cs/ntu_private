import torch


Q=torch.randn([2,2])#s1,a1,a2


gamma=0.99

for i in range(100):
    expert_reward=Q[0][0]-gamma*torch.logsumexp(Q[0,:], dim=-1, keepdim=False)

    current_policy= torch.softmax(Q[0],dim=-1)

    s1_prob=current_policy[0]

