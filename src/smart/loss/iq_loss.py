import torch



def soft_update( net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss):
    div = 'x2'
    alpha = 1
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