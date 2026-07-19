import torch
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum


def destination_normalized_interaction_bce(
    logits: Tensor,
    target: float,
    distance_weight: Tensor,
    destination_index: Tensor,
    num_nodes: int,
    eps: float = 1e-8,
) -> Tensor:
    """Compute interaction BCE without density-dependent loss scaling.

    Each destination agent contributes approximately one loss term, regardless
    of how many incoming interaction edges it has.
    """
    if logits.numel() == 0:
        return logits.new_zeros(())

    logits = logits.reshape(-1)
    distance_weight = distance_weight.reshape(-1).detach()
    destination_index = destination_index.reshape(-1)

    if logits.shape != distance_weight.shape:
        raise ValueError(
            "logits and distance_weight must have the same shape, "
            f"got {tuple(logits.shape)} and "
            f"{tuple(distance_weight.shape)}."
        )

    targets = torch.full_like(logits, fill_value=target)

    edge_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    weighted_loss_sum = scatter_sum(
        edge_loss * distance_weight,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )

    weight_mass = scatter_sum(
        distance_weight,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )

    # Equivalent to:
    # confidence_i * weighted_average_loss_i,
    # where confidence_i = min(weight_mass_i, 1).
    node_loss = weighted_loss_sum / weight_mass.clamp_min(1e-5)

    valid_node_mask = weight_mass > eps
    if not valid_node_mask.any():
        return logits.new_zeros(())

    return node_loss[valid_node_mask].mean()

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
        train_agent_mask = train_mask

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

def _masked_square_norm_mean(gradient, mask, reference):
    """Mean squared gradient norm over valid entries only."""
    if gradient is None or mask is None or not torch.any(mask):
        return reference.new_zeros(())

    valid_gradient = gradient[mask]
    return valid_gradient.reshape(valid_gradient.shape[0], -1).square().sum(dim=-1).mean()#.square()

def aggregate_interaction_logits_for_gp(
    interaction_logits: Tensor,
    destination_index: Tensor,
    num_nodes: int,
    edge_weight: float,
) -> tuple[Tensor, Tensor]:

    weighted_logit_sum = scatter_sum(
        edge_weight * interaction_logits,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )

    weight_mass = scatter_sum(
        edge_weight,
        destination_index,
        dim=0,
        dim_size=num_nodes,
    )

    node_logits = (
        weighted_logit_sum
        / weight_mass.clamp_min(1.0)
    )

    return node_logits, weight_mass

def mean_squared_gradient_norm(
    gradients: tuple[Tensor | None, ...],
    reference: Tensor,
) -> Tensor:
    penalties = []

    for gradient in gradients:
        if gradient is not None and gradient.numel() > 0:
            penalties.append(
                gradient.reshape(gradient.shape[0], -1)
                .square()
                .sum(dim=-1)
                .mean()
            )

    if not penalties:
        return reference.new_zeros(())

    return torch.stack(penalties).sum()

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
    valid_mask = valid_mask
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
    elementwise_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weight,
        reduction="mean",
    )

    #elementwise_loss=(F.relu(1+(1-2*targets)*logits)*weight).mean()
   # elementwise_loss=(((2*targets-1)-logits).square()*weight).mean()

    return elementwise_loss #.sum() / weight.sum().clamp_min(eps)
