import torch


def _select_ego_logits(
    ego_logits: torch.Tensor,
    dis_mask: torch.Tensor | None,
    mask_t: torch.Tensor | None,
) -> torch.Tensor:
    """
    Support two discriminator-output conventions:

    1. ego_logits is already compressed to entries where mask_t is True.
    2. ego_logits is dense over the flattened time-agent grid.
    """
    if dis_mask is None or mask_t is None:
        return ego_logits

    base_mask = mask_t.flatten()

    if dis_mask.dtype != torch.bool:
        raise TypeError("dis_mask must be a Boolean tensor.")

    if dis_mask.numel() != base_mask.numel():
        raise ValueError(
            f"dis_mask has {dis_mask.numel()} elements, but mask_t has "
            f"{base_mask.numel()} flattened elements."
        )

    compressed_size = int(base_mask.sum().item())

    if ego_logits.numel() == compressed_size:
        # predict_agent() already removed invalid entries.
        return ego_logits[dis_mask[base_mask]]

    if ego_logits.numel() == base_mask.numel():
        # predict_agent() returned a dense grid.
        return ego_logits[base_mask & dis_mask]

    raise ValueError(
        "Cannot align ego_logits with mask_t. "
        f"ego_logits.numel()={ego_logits.numel()}, "
        f"mask_t.numel()={base_mask.numel()}, "
        f"mask_t.sum()={compressed_size}."
    )


def compute_gp(
    key: str,
    tokenized_agent: dict,
    dis_mask: torch.Tensor | None,
    mask_t: torch.Tensor | None,
    discriminator,
    dis_loss: str = "r2",
    gp_lambda: float = 1,
    regularize_shape: bool = True,
) -> torch.Tensor:
    """
    Calculate scene-level R1, R2, or interpolation-based gradient penalty.

    dis_loss:
        "r1": regularize discriminator gradient at expert samples.
        "r2": regularize discriminator gradient at policy samples.
        "wgan-gp": regularize interpolated expert-policy samples.
    """
    policy_pos = tokenized_agent["sampled_pos"]
    device = policy_pos.device

    if key == "expert":
        # Cache expert data without retaining any upstream graph.
        tokenized_agent["expert_sampled_pos"] = (
            tokenized_agent["sampled_pos"].detach().clone()
        )
        tokenized_agent["expert_sampled_heading"] = (
            tokenized_agent["sampled_heading"].detach().clone()
        )
        tokenized_agent["expert_valid_mask"] = (
            tokenized_agent["valid_mask"].detach().clone()
        )
        tokenized_agent["expert_token_mask"] = (
            tokenized_agent["token_mask"].detach().clone()
        )
        tokenized_agent["expert_shape"] = (
            tokenized_agent["shape"].detach().clone()
        )

        return policy_pos.new_zeros(())

    required_keys = (
        "expert_sampled_pos",
        "expert_sampled_heading",
        "expert_valid_mask",
        "expert_token_mask",
        "expert_shape",
    )
    missing_keys = [k for k in required_keys if k not in tokenized_agent]
    if missing_keys:
        raise RuntimeError(
            "Expert tensors must be cached before computing the policy GP. "
            f"Missing keys: {missing_keys}"
        )

    expert_pos = tokenized_agent["expert_sampled_pos"].detach()
    expert_head = tokenized_agent["expert_sampled_heading"].detach()
    expert_shape = tokenized_agent["expert_shape"].detach()

    policy_pos = tokenized_agent["sampled_pos"].detach()
    policy_head = tokenized_agent["sampled_heading"].detach()
    policy_shape = tokenized_agent["shape"].detach()

    batch_idx = tokenized_agent["batch"]
    num_agents = batch_idx.numel()
    if num_agents == 0:
        return policy_pos.new_zeros(())
    num_graphs = int(batch_idx.max().item()) + 1

    if expert_pos.shape != policy_pos.shape:
        raise ValueError("Expert and policy positions must have identical shapes.")

    if expert_head.shape != policy_head.shape:
        raise ValueError("Expert and policy headings must have identical shapes.")

    if expert_shape.shape != policy_shape.shape:
        raise ValueError("Expert and policy shapes must have identical shapes.")

    if expert_shape.shape[-1] < 2:
        raise ValueError("shape must contain at least two continuous geometry dimensions.")

    if dis_loss == "r1":
        valid_mask = tokenized_agent["expert_valid_mask"]
        token_mask = tokenized_agent["expert_token_mask"]
        alpha_agent = torch.ones(
            (num_agents, 1),
            device=device,
            dtype=policy_pos.dtype,
        )

    elif dis_loss == "r2":
        valid_mask = tokenized_agent["valid_mask"]
        token_mask = tokenized_agent["token_mask"]
        alpha_agent = torch.zeros(
            (num_agents, 1),
            device=device,
            dtype=policy_pos.dtype,
        )

    elif dis_loss == "wgan-gp":
        valid_mask = (
            tokenized_agent["valid_mask"]
            & tokenized_agent["expert_valid_mask"]
        )
        token_mask = (
            tokenized_agent["token_mask"]
            & tokenized_agent["expert_token_mask"]
        )

        alpha_graph = torch.rand(
            (num_graphs, 1),
            device=device,
            dtype=policy_pos.dtype,
        )
        alpha_agent = alpha_graph[batch_idx]

    else:
        raise ValueError(f"Unsupported dis_loss: {dis_loss}")

    train_mask = tokenized_agent.get("train_mask")
    if train_mask is None:
        train_agent_mask = torch.ones(
            num_agents,
            dtype=torch.bool,
            device=device,
        )
    else:
        train_agent_mask = train_mask.bool()

    train_valid_mask = valid_mask & train_agent_mask[:, None]

    # Position interpolation: [A, 1, 1] * [A, T, 2]
    alpha_pos = alpha_agent[..., None]
    interp_pos = alpha_pos * expert_pos + (1.0 - alpha_pos) * policy_pos

    # Circular interpolation for heading angles in radians.
    heading_delta = torch.atan2(
        torch.sin(expert_head - policy_head),
        torch.cos(expert_head - policy_head),
    )
    interp_head = policy_head + alpha_agent * heading_delta

    pose_base = torch.cat(
        [interp_pos, interp_head.unsqueeze(-1)],
        dim=-1,
    )

    # Introduce an explicit leaf tensor only for active poses.
    pose_leaf = (
        pose_base[train_valid_mask]
        .detach()
        .requires_grad_(True)
    )

    pose_input = pose_base.detach().clone()
    pose_input[train_valid_mask] = pose_leaf

    # Only regularize continuous shape geometry: length and width.
    shape_xy_base = (
        alpha_agent * expert_shape[..., :2]
        + (1.0 - alpha_agent) * policy_shape[..., :2]
    )

    shape_agent_mask = train_agent_mask & valid_mask.any(dim=1)

    if regularize_shape:
        shape_xy_leaf = (
            shape_xy_base[shape_agent_mask]
            .detach()
            .requires_grad_(True)
        )

        shape_xy_input = shape_xy_base.detach().clone()
        shape_xy_input[shape_agent_mask] = shape_xy_leaf
    else:
        shape_xy_leaf = None
        shape_xy_input = shape_xy_base.detach()

    # Preserve any additional non-geometric shape attributes.
    if policy_shape.shape[-1] > 2:
        if dis_loss == "r1":
            shape_rest = expert_shape[..., 2:]
        else:
            # Do not interpolate categorical or static attributes.
            shape_rest = policy_shape[..., 2:]

        interp_shape_input = torch.cat(
            [shape_xy_input, shape_rest.detach()],
            dim=-1,
        )
    else:
        interp_shape_input = shape_xy_input

    disc_out_interp = discriminator.predict_agent(
        None,
        token_mask,
        valid_mask,
        pose_input[..., :2],
        pose_input[..., 2],
        tokenized_agent,
        tokenized_agent["map_feature"],
        interp_shape_input,
    )

    ego_logits, interact_logits = disc_out_interp[0]

    ego_logits = _select_ego_logits(
        ego_logits=ego_logits,
        dis_mask=dis_mask,
        mask_t=mask_t,
    )

    if torch.is_tensor(interact_logits) and interact_logits.numel() > 0:
        all_logits = torch.cat([ego_logits, interact_logits], dim=0)
    else:
        all_logits = ego_logits

    if all_logits.numel() == 0 or not all_logits.requires_grad:
        return policy_pos.new_zeros(())

    grad_inputs = [pose_leaf]
    if regularize_shape and shape_xy_leaf is not None:
        grad_inputs.append(shape_xy_leaf)

    gradients = torch.autograd.grad(
        outputs=all_logits.sum(),
        inputs=tuple(grad_inputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )

    grad_pose = gradients[0]
    grad_shape = gradients[1] if len(gradients) > 1 else None

    # Calculate one accumulated squared-gradient norm per scene.
    grad_sq_per_graph = torch.zeros(
        num_graphs,
        device=device,
        dtype=policy_pos.dtype,
    )
    zero = policy_pos.new_zeros(())
    pose_penalty = zero
    shape_penalty = zero

    if grad_pose is not None and grad_pose.numel() > 0:
        pose_grad_sq = grad_pose.square().sum(dim=-1)
        pose_agent_idx = train_valid_mask.nonzero(as_tuple=False)[:, 0]
        pose_graph_idx = batch_idx[pose_agent_idx]
        grad_sq_per_graph.index_add_(0, pose_graph_idx, pose_grad_sq)
        pose_penalty = pose_grad_sq.mean()

    if grad_shape is not None and grad_shape.numel() > 0:
        shape_grad_sq = grad_shape.square().sum(dim=-1)
        shape_graph_idx = batch_idx[shape_agent_mask]
        grad_sq_per_graph.index_add_(0, shape_graph_idx, shape_grad_sq)
        shape_penalty = shape_grad_sq.mean()

    if dis_loss in {"r1", "r2"}:
        gp = 0.5 * gp_lambda * (pose_penalty + shape_penalty)
    else:
        grad_norm_per_graph = torch.sqrt(grad_sq_per_graph + 1e-12)
        gp = gp_lambda * (grad_norm_per_graph - 1.0).square().mean()

    return gp

import torch
import torch.nn.functional as F

def _masked_gradient_cap_penalty(
    gradient,
    mask,
    cap,
    reference,
):
    if gradient is None or mask is None or not torch.any(mask):
        return reference.new_zeros(())

    mask = mask.bool()

    grad = gradient[mask]
    grad_norm = grad.reshape(grad.shape[0], -1).square().sum(dim=-1).sqrt()

    cap = cap[mask].to(dtype=grad_norm.dtype)
    cap = cap.clamp_min(1e-6)

    penalty = torch.relu(grad_norm - cap).square()

    return penalty.mean()

def ConfidenceAdaptiveGradientCapGP(
        sampled_pos,
        sampled_heading,
        shape,
        critic_score,
        valid_mask,
        gamma=0.01,
        tau=1.0,
        cap_far=0.5,
        cap_boundary=5.0,
        w_pos=1.0,
        w_heading=0.2,
        w_shape=0.01,
):
    """
    Rule-free GP for traffic discriminator.

    Does not force gradient to zero.
    It only prevents gradient explosion.

    Near discriminator decision boundary:
        large cap -> allows sharp reward.

    Far from decision boundary:
        small cap -> enforces smoothness.
    """

    valid_mask = valid_mask.bool()
    valid_agent_mask = valid_mask.any(dim=-1)

    if critic_score.ndim > 0:
        grad_outputs = torch.ones_like(critic_score)
    else:
        grad_outputs = None

    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=(sampled_pos, sampled_heading, shape),
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    grad_pos, grad_heading, grad_shape = gradients

    with torch.no_grad():
        score = critic_score.detach()

        if score.shape == valid_mask.shape:
            score_t = score

        elif score.numel() == int(valid_mask.sum().item()):
            score_t = sampled_pos.new_zeros(valid_mask.shape)
            score_t[valid_mask] = score.to(dtype=sampled_pos.dtype)

        else:
            # If critic_score is scalar, cannot infer local uncertainty.
            # Use conservative middle cap.
            score_t = sampled_pos.new_zeros(valid_mask.shape)

        uncertainty = torch.exp(-score_t.abs() / tau)

        cap_t = cap_far + (cap_boundary - cap_far) * uncertainty
        cap_t = cap_t * valid_mask.float()

        cap_agent = (
            cap_t.sum(dim=-1)
            / valid_mask.float().sum(dim=-1).clamp_min(1.0)
        )

    pos_penalty = _masked_gradient_cap_penalty(
        grad_pos,
        valid_mask,
        cap_t,
        critic_score,
    )

    heading_penalty = _masked_gradient_cap_penalty(
        grad_heading,
        valid_mask,
        cap_t,
        critic_score,
    )

    shape_penalty = _masked_gradient_cap_penalty(
        grad_shape,
        valid_agent_mask,
        cap_agent,
        critic_score,
    )

    total = (
        w_pos * pos_penalty
        + w_heading * heading_penalty
        + w_shape * shape_penalty
    )

    scale = gamma / 2.0

    return (
        scale * total,
        scale * pos_penalty,
        scale * heading_penalty,
        scale * shape_penalty,
    )

def _masked_square_norm_mean(gradient, mask, reference):
    """Mean squared gradient norm over valid entries only."""
    if gradient is None or mask is None or not torch.any(mask):
        return reference.new_zeros(())

    valid_gradient = gradient[mask]
    return valid_gradient.reshape(valid_gradient.shape[0], -1).abs().sum(dim=-1).mean()#.square()


def ZeroCenteredGradientPenalty(
        sampled_pos,
        sampled_heading,
        shape,
        critic_score,
        valid_mask,
        gamma=0.01,
):
    """Zero-centered R2 penalty on generated trajectories.

    The discriminator locally differentiates through the fixed graph topology.
    Padding entries are excluded from the normalization.
    """
    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=(sampled_pos, sampled_heading, shape),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    grad_pos, grad_heading, grad_shape = gradients
    valid_mask = valid_mask.bool()
    valid_agent_mask = valid_mask.any(dim=-1)

    pos_penalty = _masked_square_norm_mean(grad_pos, valid_mask, critic_score)
    heading_penalty = _masked_square_norm_mean(grad_heading, valid_mask, critic_score)
    shape_penalty = _masked_square_norm_mean(grad_shape, valid_agent_mask, critic_score)

    scale = gamma / 2.0
    return (
        scale * (pos_penalty + heading_penalty + shape_penalty),
        scale * pos_penalty,
        scale * heading_penalty,
        scale * shape_penalty,
    )


def _reshape_valid_rewards(rewards, mask_t, name):
    """Restore compressed valid-node rewards to a dense [T, A] tensor."""
    if not torch.is_tensor(rewards):
        raise TypeError(f"{name} must be a tensor, got {type(rewards)}")

    if rewards.numel() == mask_t.numel():
        return rewards.reshape(mask_t.shape)

    valid_count = int(mask_t.sum().item())
    if rewards.numel() != valid_count:
        raise ValueError(
            f"Cannot align {name} with mask_t: rewards.numel()={rewards.numel()}, "
            f"mask_t.numel()={mask_t.numel()}, mask_t.sum()={valid_count}."
        )

    dense_rewards = rewards.new_zeros(mask_t.shape)
    dense_rewards[mask_t] = rewards
    return dense_rewards


def _has_elements(x) -> bool:
    """Return True only for a non-empty tensor."""
    return torch.is_tensor(x) and x.numel() > 0


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    target: float,
    weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    BCE loss with stable normalization.

    When weights are provided, normalize by sum(weight) rather than by the
    number of elements. This prevents the loss scale from changing when the
    average interaction weight changes across batches.
    """
    targets = torch.full_like(logits, fill_value=target)

    if logits.numel() == 0:
        return logits.new_zeros(())

    if weight is None:
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
        )

    weight = weight.to(dtype=logits.dtype)
    elementwise_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weight,
        reduction="mean",
    )
    return elementwise_loss #.sum() / weight.sum().clamp_min(eps)

def get_reward(self, tokenized_agent, key, dis_mask=None):

    mask_t = tokenized_agent["valid_mask"].transpose(0, 1)[self.dis_start_step:]

    if "train_mask" in tokenized_agent.keys() and tokenized_agent["train_mask"] is not None:
        mask_t = mask_t[:, tokenized_agent["train_mask"]]

    if dis_mask is None and not self.pred_init:
        dis_mask = mask_t.flatten(0, 1)

        tokenized_agent["dis_mask"] = dis_mask

    disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        tokenized_agent["token_mask"],
                                                        tokenized_agent["valid_mask"],
                                                        tokenized_agent["sampled_pos"],
                                                        tokenized_agent["sampled_heading"],
                                                        tokenized_agent,
                                                        tokenized_agent["map_feature"],
                                                        tokenized_agent["shape"]
                                                        )

    ego_logits, interact_logits = disc_out[0]

    ego_rewards, nei_rewards, valid_ego_reward, valid_interact_reward = disc_out[2]

    if not self.encoder.discriminator.training:
        return ego_rewards.reshape(mask_t.shape)

    if len(nei_rewards) > 0:
        all_rewards = ego_rewards + nei_rewards
        self.log("train/" + key + "_all_rewards", all_rewards.mean().detach(), on_step=True, batch_size=1)
        self.log("train/" + key + "_nei_rewards", nei_rewards.mean().detach(), on_step=True, batch_size=1)

    self.log("train/" + key + "_rewards", ego_rewards.mean().detach(), on_step=True, batch_size=1)
    self.log("train/" + key + "_valid_ego_reward", valid_ego_reward.mean().detach(), on_step=True, batch_size=1)
    self.log("train/" + key + "_valid_interact_reward", valid_interact_reward.mean().detach(), on_step=True,
             batch_size=1)

    if key == "expert":
        target = 1
    else:
        target = 0
        ego_rewards = ego_rewards.reshape(mask_t.shape)  # [self.gail_start_step-self.dis_start_step:] #t,a
        if len(nei_rewards):
            nei_rewards = nei_rewards.reshape(mask_t.shape)  # t,a

        if dis_mask is not None:
            ego_logits = ego_logits[dis_mask[mask_t.flatten(0, 1)]]  # valid ego logit

    self.log("train/" + key + "_ego_score", torch.sigmoid(ego_logits).mean().detach(), on_step=True, batch_size=1)

    bce_loss = F.binary_cross_entropy_with_logits(ego_logits, torch.zeros_like(ego_logits) + target,
                                                  reduction='mean')

    if len(interact_logits) > 0:
        weight = disc_out[3]

        self.log("train/" + key + "_inter_score", torch.sigmoid(interact_logits).mean().detach(), on_step=True,
                 batch_size=1)

        # interact_bce_loss=F.binary_cross_entropy_with_logits(interact_logits, torch.zeros_like(interact_logits) + target,
        #                                              weight=weight, reduction='sum')/len(ego_logits)#/len(interact_logits)#
        #
        interact_bce_loss = F.binary_cross_entropy_with_logits(interact_logits,
                                                               torch.zeros_like(interact_logits) + target,
                                                               weight=weight, reduction='mean')  # /dis_mask.sum()

        ego_logits = torch.cat([ego_logits, interact_logits], dim=0)
        self.log("train/" + key + "_interact_logits", interact_logits.mean().detach(), on_step=True, batch_size=1)
    else:
        interact_bce_loss = 0

    disc_val = torch.sigmoid(ego_logits)

    self.log("train/" + key + "_disc_val", disc_val.mean().detach(), on_step=True, batch_size=1)
    self.log("train/" + key + "_disc_val_std", disc_val.std().detach(), on_step=True, batch_size=1)

    if self.use_gradient_penalty:
        gp = compute_gp(key, tokenized_agent, dis_mask, mask_t, self.encoder.discriminator)
        self.log("train/" + key + "_gp", gp, on_step=True, batch_size=1)
    else:
        gp = 0

    return bce_loss + interact_bce_loss, ego_rewards, nei_rewards, gp, dis_mask  # ,mask_s.flatten(0,1)
