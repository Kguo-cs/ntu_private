import torch


def compute_gp(key,tokenized_agent,dis_mask,discriminator):
    if key == "expert":
        tokenized_agent["expert_sampled_pos"] = tokenized_agent['sampled_pos'].clone()
        tokenized_agent["expert_sampled_heading"] = tokenized_agent['sampled_heading'].clone()
        tokenized_agent["expert_valid_mask"] = tokenized_agent['valid_mask'].clone()
        tokenized_agent["expert_token_mask"] = tokenized_agent['token_mask'].clone()
        gp = 0
    else:
        expert_pos = tokenized_agent["expert_sampled_pos"]  # [B, N, 2]
        expert_head = tokenized_agent["expert_sampled_heading"]  # [B, N, 1]
        policy_pos = tokenized_agent["sampled_pos"]  # [B, N, 2]
        policy_head = tokenized_agent["sampled_heading"]  # [B, N, 1]

        dis_loss = 'r2'

        if dis_loss == 'r1':
            valid_mask = tokenized_agent['expert_valid_mask']
            token_mask = tokenized_agent['expert_token_mask']
            alpha = torch.ones_like(expert_pos[..., 0])  # [B, N, 1]
        elif dis_loss == 'r2':
            valid_mask = tokenized_agent['valid_mask']
            token_mask = tokenized_agent['token_mask']
            alpha = torch.zeros_like(expert_pos[..., 0])  # [B, N, 1]
        else:
            valid_mask = tokenized_agent['valid_mask'] & tokenized_agent['expert_valid_mask']
            token_mask = tokenized_agent['token_mask'] & tokenized_agent['expert_token_mask']
            # alpha = torch.rand_like(expert_pos[..., 0])
            batch_idx = tokenized_agent['batch']

            alpha = torch.rand(size=(max(batch_idx) + 1, 1), device=batch_idx.device)

            alpha = alpha[batch_idx]

        interp_pos = alpha[..., None] * expert_pos + (1.0 - alpha[..., None]) * policy_pos  # [B, N, 2]
        interp_head = alpha * expert_head + (1.0 - alpha) * policy_head  # [B, N, 1]

        train_valid_mask = valid_mask & tokenized_agent["train_mask"][:, None]

        interpolates_pose = torch.cat((interp_pos, interp_head[:, :, None]), dim=-1)

        interpolates = interpolates_pose[train_valid_mask]  # [train_mask,2:]

        interpolates.requires_grad_(True)  # IMPORTANT

        interpolates_pose[train_valid_mask] = interpolates

        disc_out_interp = discriminator.predict_agent(None,
                                                                   token_mask,
                                                                   valid_mask,
                                                                   interpolates_pose[..., :2],
                                                                   interpolates_pose[..., 2],
                                                                   tokenized_agent,
                                                                   tokenized_agent["detach_map_feature"],
                                                                   abs_time=tokenized_agent["abs_time"])

        ego_logits, interact_logits = disc_out_interp[0]
        ego_logits = ego_logits[dis_mask]
        logit = torch.cat([ego_logits, interact_logits], dim=0)

        disc_flat = logit.reshape(-1, 1)
        grad_outputs = torch.ones_like(disc_flat)

        # Compute gradients wrt interpolated inputs
        grad_all = torch.autograd.grad(
            outputs=disc_flat,  # whatever you use
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = grad_all.norm(2, dim=1)  # [B]
        gp_lambda = 1

        if dis_loss == 'r1' or dis_loss == 'r2':
            gp = (grad_norm ** 2).mean() * gp_lambda / 2
        else:
            gp = ((grad_norm - 1.0) ** 2).mean() * gp_lambda

    return gp


