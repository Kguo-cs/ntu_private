import torch
from src.smart.metrics.utils import get_euclidean_targets

from src.smart.loss.gmm_dist import  GMM_Dist,get_entropy


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


def process_data( data,token_processor,encoder,pred_agent=True,pred_light=False):
    tokenized_agent = {}
    tokenized_map = {}
    tokenized_agent['num_graphs'] = data.num_graphs

    if pred_agent:
        map = data["tokenized_map"]
        agent = data["tokenized_agent"]

        for key in ["position", "orientation", "batch", "token_idx", "type", "pl_type", "light_type"]:
            tokenized_map[key] = map[key]

        agent_shape, token_traj_all, token_traj = token_processor._get_agent_shape_and_token_traj(
            agent['type']
        )
        tokenized_agent['token_traj_all'] = token_traj_all
        tokenized_agent["token_agent_shape"] = agent_shape  # [n_token, 2]

        if "col_mask" in agent.keys():
            tokenized_agent["col_mask"] = agent["col_mask"]

        if "gt_pos_raw" in agent.keys():
            for key in ["gt_pos_raw", "gt_head_raw", "gt_valid_raw"]:
                tokenized_agent[key] = agent[key][:, 5::5]
            for key in ["type", "batch", "shape"]:
                tokenized_agent[key] = agent[key]

            token_dict = token_processor.my_match_agent_token(agent["gt_valid_raw"], agent["gt_pos_raw"],
                                                                   agent["gt_head_raw"],
                                                                   agent_shape, token_traj
                                                                   )
            tokenized_agent.update(token_dict)

            map_feature = encoder.map_encoder(tokenized_map)

            detach_map_feature={}
            for key in map_feature.keys():
                detach_map_feature[key] = map_feature[key].detach()

            tokenized_map["detach_map_feature"] = detach_map_feature
            tokenized_map["map_feature"] = map_feature

            with torch.no_grad():
                pred_dict=encoder.agent_encoder(tokenized_agent, map_feature)
                dist =  GMM_Dist(pred_dict["q_value"])

                noised_pos=dist.sample()

                token_dict = token_processor.my_match_agent_token(agent["gt_valid_raw"], agent["gt_pos_raw"],
                                                                       agent["gt_head_raw"],
                                                                       agent_shape, token_traj, noised_pos
                                                                       )
            tokenized_agent.update(token_dict)


            target, target_valid = get_euclidean_targets(
                pred_pos=tokenized_agent["sampled_pos"],
                pred_head=tokenized_agent["sampled_heading"],
                pred_valid=tokenized_agent["valid_mask"],
                gt_pos=tokenized_agent["gt_pos_raw"],
                gt_head=tokenized_agent["gt_head_raw"],
                gt_valid=tokenized_agent["gt_valid_raw"]
            )

            tokenized_agent["target"] = target

        else:
            for key in ["sampled_pos", "sampled_heading", "type", "batch", "shape", "sampled_idx", "valid_mask"]:
                tokenized_agent[key] = agent[key]

    if pred_light:
        tokenized_light = data["tokenized_light"]

        light_idx = tokenized_light["light_idx"]

        # light_mask=light_idx<self.encoder.agent_encoder.light_type

        # light_pred_mask=light_mask.all(-1)#torch.ones_like(light_idx[:,0]).to(torch.bool)
        # light_idx[light_idx>2]=0

        # light_pred_mask=torch.ones_like(light_pred_mask)
        # pos_lg, orient_lg=self.rotate(pos_lg, orient_lg, batch_lg)

        tokenized_agent["light_idx"] = light_idx.long()  # [light_pred_mask]
        pos_lg = tokenized_light["pos_lg"]  # [light_pred_mask]
        orient_lg = tokenized_light["orient_lg"]  # [light_pred_mask]
        batch_lg = tokenized_light["batch"]  # [light_pred_mask]

        lengths_lg = torch.bincount(batch_lg, minlength=data.num_graphs).tolist()

        sinusoidal_lg = general_rope(pos_lg, self.encoder.agent_encoder.head_dim, orient_lg)
        sinusoidal_lg = padding(sinusoidal_lg, lengths_lg)
        tokenized_agent["lengths_lg"] = lengths_lg
        tokenized_agent["batch_lg"] = batch_lg
        tokenized_agent["sinusoidal_lg"] = sinusoidal_lg

    # if self.encoder.agent_encoder.pred_route:
    #     route_idx = agent["route_idx"] // (120 // self.encoder.agent_encoder.route_type)
    #     route_idx[:, :2] = -1
    #
    #     route_idx[route_idx == -1] = self.encoder.agent_encoder.route_type
    #
    #     tokenized_agent["route_idx"] = route_idx.long()
    #     tokenized_agent["route_valid_mask"] = route_idx != self.encoder.agent_encoder.route_type
    #
    return tokenized_map, tokenized_agent
