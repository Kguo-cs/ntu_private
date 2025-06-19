import torch
from src.smart.metrics.utils import get_euclidean_targets

from src.smart.loss.gmm_dist import  GMM_Dist,get_entropy
from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle


def soft_update( net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def get_iqloss(expert_reward,agent_reward,agent_value_loss,expert_value_loss):
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
