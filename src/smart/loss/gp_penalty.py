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
    gp_lambda: float = 1.0,
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
    num_graphs = int(batch_idx.max().item()) + 1
    num_agents = batch_idx.numel()

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

    # Calculate one gradient norm per scene.
    # grad_sq_per_graph = torch.zeros(
    #     num_graphs,
    #     device=device,
    #     dtype=policy_pos.dtype,
    # )
    #
    if grad_pose is not None and grad_pose.numel() > 0:
       # pose_agent_idx = train_valid_mask.nonzero(as_tuple=False)[:, 0]
       # pose_graph_idx = batch_idx[pose_agent_idx]

        pose_grad_sq = grad_pose.square().sum(dim=-1)
       # grad_sq_per_graph.index_add_(0, pose_graph_idx, pose_grad_sq)

    if grad_shape is not None and grad_shape.numel() > 0:
        #shape_graph_idx = batch_idx[shape_agent_mask]

        shape_grad_sq = grad_shape.square().sum(dim=-1)
    #     grad_sq_per_graph.index_add_(0, shape_graph_idx, shape_grad_sq)

    if dis_loss in {"r1", "r2"}:
        gp = 0.5 * gp_lambda * (pose_grad_sq.mean()+shape_grad_sq.mean())#grad_sq_per_graph.mean()
    else:
        grad_norm_per_graph = torch.sqrt(grad_sq_per_graph + 1e-12)
        gp = gp_lambda * (grad_norm_per_graph - 1.0).square().mean()

    return gp

import torch
import torch.nn.functional as F


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

    if weight is None:
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
        )

    weight = weight.to(dtype=logits.dtype)

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weight,
        reduction="sum",
    ) / weight.sum().clamp_min(eps)


def get_reward(
    self,
    tokenized_agent: dict,
    key: str,
    dis_mask: torch.Tensor | None = None,
):
    """
    Calculate discriminator loss and rewards for expert or policy trajectories.

    Args:
        tokenized_agent:
            Dictionary containing tokenized trajectories, masks, map features,
            shape features, and optional train_mask.

        key:
            Either "expert" or "agent".

        dis_mask:
            Optional Boolean mask over the flattened time-agent grid. It is
            created from mask_t when omitted.

    Returns:
        disc_loss:
            Ego BCE loss + interaction BCE loss.

        ego_rewards:
            For policy samples, reshaped to [T, A].
            For expert samples, returned in the original discriminator format.

        nei_rewards:
            For policy samples, reshaped to [T, A] when available.
            For expert samples, returned in the original discriminator format.

        regularization_loss:
            R1, R2, or interpolation gradient penalty.

        dis_mask:
            Boolean discriminator mask.
    """

    discriminator = self.encoder.discriminator

    # valid_mask: [A, T]
    # mask_t:     [T_selected, A_selected]
    mask_t = tokenized_agent["valid_mask"].transpose(0, 1)[
        self.dis_start_step:
    ]

    train_mask = tokenized_agent.get("train_mask")

    if train_mask is not None:
        train_mask = train_mask.bool()
        mask_t = mask_t[:, train_mask]

    # Create a dense mask over the selected [time, agent] grid.
    # This mask is reused to keep expert and policy discriminator losses aligned.
    if dis_mask is None or (self.pred_init and key == "agent"):
        dis_mask = mask_t.flatten()

    if dis_mask is not None:
        dis_mask = dis_mask.bool()
        tokenized_agent["dis_mask"] = dis_mask

    disc_out = discriminator.predict_agent(
        tokenized_agent["sampled_idx"],
        tokenized_agent["token_mask"],
        tokenized_agent["valid_mask"],
        tokenized_agent["sampled_pos"],
        tokenized_agent["sampled_heading"],
        tokenized_agent,
        tokenized_agent["map_feature"],
        tokenized_agent["shape"],
    )

    ego_logits, interact_logits = disc_out[0]

    (
        ego_rewards,
        nei_rewards,
        valid_ego_reward,
        valid_interact_reward,
    ) = disc_out[2]

    # During validation or rollout, only return the policy reward grid.
    if not discriminator.training:
        return ego_rewards.reshape(mask_t.shape)

    # Apply the same ego-logit mask to both expert and policy samples.
    # This avoids training expert and generated samples on different subsets.
    ego_logits = _select_ego_logits(
        ego_logits=ego_logits,
        dis_mask=dis_mask,
        mask_t=mask_t,
    )

    target = 1.0 if key == "expert" else 0.0

    ego_bce_loss = _weighted_bce_with_logits(
        logits=ego_logits,
        target=target,
    )

    has_nei_rewards = _has_elements(nei_rewards)
    has_interact_logits = _has_elements(interact_logits)

    if has_interact_logits:
        interaction_weight = disc_out[3]

        interact_bce_loss = _weighted_bce_with_logits(
            logits=interact_logits,
            target=target,
            weight=interaction_weight,
        )

        combined_logits = torch.cat(
            [ego_logits.reshape(-1), interact_logits.reshape(-1)],
            dim=0,
        )
    else:
        interact_bce_loss = ego_logits.new_zeros(())
        combined_logits = ego_logits.reshape(-1)

    disc_loss = ego_bce_loss + interact_bce_loss

    # Reshape only policy rewards because these are consumed by the policy
    # update as a [time, agent] reward matrix.
    if key == "agent":
        ego_rewards = ego_rewards.reshape(mask_t.shape)

        if has_nei_rewards:
            nei_rewards = nei_rewards.reshape(mask_t.shape)

    # ----------------------------
    # Logging
    # ----------------------------
    self.log(
        f"train/{key}_rewards",
        ego_rewards.mean(),
        on_step=True,
        batch_size=1,
    )

    if _has_elements(valid_ego_reward):
        self.log(
            f"train/{key}_valid_ego_reward",
            valid_ego_reward.mean(),
            on_step=True,
            batch_size=1,
        )

    if _has_elements(valid_interact_reward):
        self.log(
            f"train/{key}_valid_interact_reward",
            valid_interact_reward.mean(),
            on_step=True,
            batch_size=1,
        )

    if has_nei_rewards:
        all_rewards = ego_rewards + nei_rewards

        self.log(
            f"train/{key}_all_rewards",
            all_rewards.mean(),
            on_step=True,
            batch_size=1,
        )

        self.log(
            f"train/{key}_nei_rewards",
            nei_rewards.mean(),
            on_step=True,
            batch_size=1,
        )

    ego_score = torch.sigmoid(ego_logits)

    self.log(
        f"train/{key}_ego_score",
        ego_score.mean(),
        on_step=True,
        batch_size=1,
    )

    if has_interact_logits:
        interact_score = torch.sigmoid(interact_logits)

        self.log(
            f"train/{key}_inter_score",
            interact_score.mean(),
            on_step=True,
            batch_size=1,
        )

        self.log(
            f"train/{key}_interact_logits",
            interact_logits.mean(),
            on_step=True,
            batch_size=1,
        )

    disc_val = torch.sigmoid(combined_logits)

    self.log(
        f"train/{key}_disc_val",
        disc_val.mean(),
        on_step=True,
        batch_size=1,
    )

    self.log(
        f"train/{key}_disc_val_std",
        disc_val.std(unbiased=False),
        on_step=True,
        batch_size=1,
    )

    # ----------------------------
    # Gradient regularization
    # ----------------------------
    if self.use_gradient_penalty:
        regularization_loss = compute_gp(
            key=key,
            tokenized_agent=tokenized_agent,
            dis_mask=dis_mask,
            mask_t=mask_t,
            discriminator=discriminator,
            dis_loss="r2",
            gp_lambda=1.0,
            regularize_shape=True,
        )

        self.log(
            f"train/{key}_gp",
            regularization_loss,
            on_step=True,
            batch_size=1,
        )
    else:
        regularization_loss = ego_logits.new_zeros(())

    return (
        disc_loss,
        ego_rewards,
        nei_rewards,
        regularization_loss,
        dis_mask,
    )