from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
from torch_scatter import scatter_sum


def _zero(reference: Tensor) -> Tensor:
    """Return a scalar zero with the same device and dtype as reference."""
    return reference.new_zeros(())


def _get_valid_agent_mask(valid_mask: Tensor) -> Tensor:
    """Convert a temporal valid mask into an agent-level valid mask.

    Args:
        valid_mask:
            [num_agents, num_steps] or [num_agents].

    Returns:
        [num_agents] boolean mask.
    """
    valid_mask = valid_mask

    if valid_mask.ndim == 1:
        return valid_mask

    return valid_mask.any(dim=-1)


def _mask_matches_gradient(
    gradient: Tensor,
    mask: Tensor,
) -> bool:
    """Check whether mask matches the leading dimensions of gradient."""
    if gradient.ndim < mask.ndim:
        return False

    return tuple(gradient.shape[: mask.ndim]) == tuple(mask.shape)


def _resolve_gradient_mask(
    gradient: Optional[Tensor],
    temporal_valid_mask: Tensor,
    agent_valid_mask: Tensor,
    prefer_temporal: bool,
) -> Optional[Tensor]:
    """Find a valid mask compatible with a gradient tensor.

    Typical cases:
        sampled_pos gradient:     [A, T, 2] -> temporal mask [A, T]
        sampled_heading gradient: [A, T]    -> temporal mask [A, T]
        shape gradient:           [A, 2]    -> agent mask [A]

    Compact tensors are also supported:
        [N_valid, D] -> all entries are treated as valid.
    """
    if gradient is None:
        return None

    candidate_masks = (
        (temporal_valid_mask, agent_valid_mask)
        if prefer_temporal
        else (agent_valid_mask, temporal_valid_mask)
    )

    for candidate in candidate_masks:
        if _mask_matches_gradient(gradient, candidate):
            return candidate.to(device=gradient.device, dtype=torch.bool)

    # Tensor may already contain only compact valid entries.
    num_temporal_valid = int(temporal_valid_mask.sum().item())
    num_agent_valid = int(agent_valid_mask.sum().item())

    if gradient.ndim > 0:
        if gradient.shape[0] in {
            num_temporal_valid,
            num_agent_valid,
        }:
            return torch.ones(
                gradient.shape[0],
                device=gradient.device,
                dtype=torch.bool,
            )

    raise ValueError(
        "Cannot align gradient and valid mask. "
        f"gradient shape={tuple(gradient.shape)}, "
        f"temporal mask shape={tuple(temporal_valid_mask.shape)}, "
        f"agent mask shape={tuple(agent_valid_mask.shape)}."
    )


def _masked_square_norm_mean(
    gradient: Optional[Tensor],
    mask: Optional[Tensor],
    reference: Tensor,
) -> Tensor:
    """Mean squared gradient norm over valid entries only.

    For example:

        gradient: [A, T, 2]
        mask:     [A, T]

    produces:

        mean_{valid a,t} ||gradient[a,t]||_2^2
    """
    if gradient is None or mask is None:
        return _zero(reference)

    mask = mask.to(device=gradient.device, dtype=torch.bool)

    if not torch.any(mask):
        return _zero(reference)

    if not _mask_matches_gradient(gradient, mask):
        raise ValueError(
            "Mask must match the leading gradient dimensions, "
            f"got gradient={tuple(gradient.shape)}, "
            f"mask={tuple(mask.shape)}."
        )

    valid_gradient = gradient[mask]

    if valid_gradient.numel() == 0:
        return _zero(reference)

    valid_gradient = valid_gradient.reshape(
        valid_gradient.shape[0],
        -1,
    )

    return (
        valid_gradient.square()
        .sum(dim=-1)
        .mean()
    )


def _select_valid_ego_logits(
    ego_logits: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Select valid ego logits when their layout can be inferred.

    Supports:
        [A, T]
        [A, T, 1]
        [A]
        [A, 1]
        [N_valid]
        [N_valid, 1]

    If ego_logits are already compact and do not match the original mask
    layout, all logits are retained.
    """
    if ego_logits.numel() == 0:
        return ego_logits.reshape(-1)

    valid_mask = valid_mask
    valid_agent_mask = _get_valid_agent_mask(valid_mask)

    logits = ego_logits

    if logits.ndim > 0 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    # Full temporal layout: [A, T].
    if (
        logits.ndim >= valid_mask.ndim
        and tuple(logits.shape[: valid_mask.ndim])
        == tuple(valid_mask.shape)
    ):
        selected = logits[valid_mask]
        return selected.reshape(-1)

    logits_flat = logits.reshape(-1)

    # Flattened full temporal layout.
    if logits_flat.numel() == valid_mask.numel():
        return logits_flat[valid_mask.reshape(-1)]

    # Already compact temporal logits.
    if logits_flat.numel() == int(valid_mask.sum().item()):
        return logits_flat

    # Full agent-level layout.
    if logits_flat.numel() == valid_agent_mask.numel():
        return logits_flat[valid_agent_mask]

    # Already compact agent-level logits.
    if logits_flat.numel() == int(valid_agent_mask.sum().item()):
        return logits_flat

    # Assume that the decoder already removed invalid entries.
    return logits_flat


def _infer_interaction_node_valid_mask(
    valid_mask: Tensor,
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    """Infer node validity for common graph indexing conventions.

    Supported node layouts:
        num_nodes == A * T:
            nodes correspond to flattened agent-time entries.

        num_nodes == A:
            nodes correspond to agents.

        num_nodes == valid_mask.sum():
            graph nodes are already compact valid entries.

    Otherwise, graph construction is assumed to have already removed
    invalid nodes.
    """
    valid_mask = valid_mask
    valid_agent_mask = _get_valid_agent_mask(valid_mask)

    if num_nodes == valid_mask.numel():
        return valid_mask.reshape(-1).to(device=device)

    if num_nodes == valid_agent_mask.numel():
        return valid_agent_mask.to(device=device)

    if num_nodes == int(valid_mask.sum().item()):
        return torch.ones(
            num_nodes,
            dtype=torch.bool,
            device=device,
        )

    if num_nodes == int(valid_agent_mask.sum().item()):
        return torch.ones(
            num_nodes,
            dtype=torch.bool,
            device=device,
        )

    # For a compact interaction graph, all represented nodes are valid.
    return torch.ones(
        num_nodes,
        dtype=torch.bool,
        device=device,
    )


def aggregate_interaction_logits_for_gp(
    interaction_logits: Tensor,
    destination_index: Tensor,
    edge_weight: Tensor | float,
    num_nodes: Optional[int] = None,
    node_valid_mask: Optional[Tensor] = None,
    min_mass: float = 1.0,
    detach_edge_weight: bool = True,
) -> tuple[Tensor, Tensor]:
    """Aggregate edge-level interaction logits into node-level scores.

    For destination node i:

        node_logit_i =
            sum_j weight_ji * interaction_logit_ji
            ------------------------------------------------
            max(sum_j weight_ji, min_mass)

    The use of min_mass=1 preserves attenuation when all edges are far,
    while preventing dense nodes from receiving a linearly larger score.

    Important:
        edge_weight should contain only the relevance weight, for example

            exp(-distance / distance_decay)

        It should not include interaction_reward_weight or dis_weight.

    Args:
        interaction_logits:
            [E] or [E, 1].

        destination_index:
            [E], destination node of every edge.

        edge_weight:
            Scalar, [E], or [E, 1].

        num_nodes:
            Total number of nodes. If None, inferred as max index + 1.

        node_valid_mask:
            Optional [num_nodes] mask.

        min_mass:
            Denominator lower bound. Recommended value: 1.0.

        detach_edge_weight:
            Prevent the GP from regularizing the distance weighting function.

    Returns:
        node_logits:
            [num_nodes].

        weight_mass:
            [num_nodes].
    """
    interaction_logits = interaction_logits.reshape(-1)
    destination_index = destination_index.reshape(-1).long()

    num_edges = interaction_logits.numel()

    if destination_index.numel() != num_edges:
        raise ValueError(
            "interaction_logits and destination_index must have the same "
            f"number of elements, got {num_edges} and "
            f"{destination_index.numel()}."
        )

    if num_edges == 0:
        resolved_num_nodes = 0 if num_nodes is None else int(num_nodes)

        return (
            interaction_logits.new_zeros(resolved_num_nodes),
            interaction_logits.new_zeros(resolved_num_nodes),
        )

    if num_nodes is None:
        num_nodes = int(destination_index.max().item()) + 1
    else:
        num_nodes = int(num_nodes)

    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")

    if destination_index.min() < 0:
        raise ValueError("destination_index contains a negative index.")

    if destination_index.max() >= num_nodes:
        raise ValueError(
            "destination_index exceeds num_nodes: "
            f"max index={int(destination_index.max().item())}, "
            f"num_nodes={num_nodes}."
        )

    edge_weight = torch.as_tensor(
        edge_weight,
        device=interaction_logits.device,
        dtype=interaction_logits.dtype,
    )

    if edge_weight.numel() == 1:
        edge_weight = edge_weight.expand(num_edges)
    else:
        edge_weight = edge_weight.reshape(-1)

    if edge_weight.numel() != num_edges:
        raise ValueError(
            "edge_weight must be scalar or have one value per edge, "
            f"got edge_weight={edge_weight.numel()}, "
            f"num_edges={num_edges}."
        )

    if detach_edge_weight:
        edge_weight = edge_weight.detach()

    # Relevance weights must not be negative.
    edge_weight = edge_weight.clamp_min(0.0)

    if node_valid_mask is not None:
        node_valid_mask = node_valid_mask.to(
            device=interaction_logits.device,
            dtype=torch.bool,
        ).reshape(-1)

        if node_valid_mask.numel() != num_nodes:
            raise ValueError(
                "node_valid_mask must have num_nodes entries, "
                f"got {node_valid_mask.numel()} and {num_nodes}."
            )

        valid_edge_mask = node_valid_mask[destination_index]
        edge_weight = edge_weight * valid_edge_mask.to(edge_weight.dtype)

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
        / weight_mass.clamp_min(min_mass)
    )

    return node_logits, weight_mass


def _safe_autograd_grad(
    score: Tensor,
    inputs: Sequence[Tensor],
) -> tuple[Optional[Tensor], ...]:
    """Run autograd.grad while supporting unused or disabled inputs."""
    if score.numel() != 1:
        raise ValueError(
            f"score must be scalar, got shape={tuple(score.shape)}."
        )

    if not score.requires_grad:
        return tuple(None for _ in inputs)

    active_indices = [
        index
        for index, value in enumerate(inputs)
        if value is not None and value.requires_grad
    ]

    if not active_indices:
        return tuple(None for _ in inputs)

    active_inputs = tuple(inputs[index] for index in active_indices)

    active_gradients = torch.autograd.grad(
        outputs=score,
        inputs=active_inputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    gradients: list[Optional[Tensor]] = [None] * len(inputs)

    for index, gradient in zip(active_indices, active_gradients):
        gradients[index] = gradient

    return tuple(gradients)


def _compute_branch_raw_penalty(
    gradients: tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ],
    sampled_pos: Tensor,
    sampled_heading: Tensor,
    shape: Tensor,
    valid_mask: Tensor,
    position_scale: float,
    heading_scale: float,
    shape_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute valid-mask-aware penalties for one discriminator branch."""
    valid_mask = valid_mask.to(
        device=sampled_pos.device,
        dtype=torch.bool,
    )

    valid_agent_mask = _get_valid_agent_mask(valid_mask)

    grad_pos, grad_heading, grad_shape = gradients

    pos_mask = _resolve_gradient_mask(
        gradient=grad_pos,
        temporal_valid_mask=valid_mask,
        agent_valid_mask=valid_agent_mask,
        prefer_temporal=True,
    )

    heading_mask = _resolve_gradient_mask(
        gradient=grad_heading,
        temporal_valid_mask=valid_mask,
        agent_valid_mask=valid_agent_mask,
        prefer_temporal=True,
    )

    shape_mask = _resolve_gradient_mask(
        gradient=grad_shape,
        temporal_valid_mask=valid_mask,
        agent_valid_mask=valid_agent_mask,
        prefer_temporal=False,
    )

    pos_penalty = _masked_square_norm_mean(
        gradient=grad_pos,
        mask=pos_mask,
        reference=sampled_pos,
    )

    heading_penalty = _masked_square_norm_mean(
        gradient=grad_heading,
        mask=heading_mask,
        reference=sampled_pos,
    )

    shape_penalty = _masked_square_norm_mean(
        gradient=grad_shape,
        mask=shape_mask,
        reference=sampled_pos,
    )

    raw_penalty = (
        position_scale * pos_penalty
        + heading_scale * heading_penalty
        + shape_scale * shape_penalty
    )

    return (
        raw_penalty,
        pos_penalty,
        heading_penalty,
        shape_penalty,
    )


def ZeroCenteredGradientPenalty_edge(
    sampled_pos: Tensor,
    sampled_heading: Tensor,
    shape: Tensor,
    critic_score: tuple,
    valid_mask: Tensor,
    gamma: float = 0.01,
    interaction_gamma: Optional[float] = None,
    num_nodes: Optional[int] = None,
    position_scale: float = 1.0,
    heading_scale: float = 1.0,
    shape_scale: float = 1.0,
    interaction_min_mass: float = 1.0,
    detach_edge_weight: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Zero-centered gradient penalty for decomposed discriminators.

    Expected critic_score formats:

        4-element form:
            (
                ego_logits,
                interaction_logits,
                edge_weight,
                destination_index,
            )

        5-element form:
            (
                ego_logits,
                interaction_logits,
                edge_weight,
                destination_index,
                num_nodes,
            )

    valid_mask must correspond to the trajectory entries actually consumed
    by the discriminator. Padding and ignored history steps should be False.

    Returns:
        total_gp:
            scene_gp + interaction_gp.

        scene_gp:
            GP from the scene discriminator branch.

        interaction_gp:
            GP from the interaction discriminator branch.

        raw_total_penalty:
            Unscaled scene + interaction gradient norm penalty.
    """

    ego_logits = critic_score[0]
    interaction_logits = critic_score[1]
    edge_weight = critic_score[2]
    destination_index = critic_score[3]

    valid_mask = valid_mask.to(
        device=sampled_pos.device,
        dtype=torch.bool,
    )

    if interaction_gamma is None:
        interaction_gamma = gamma

    # ---------------------------------------------------------
    # Scene discriminator score
    # ---------------------------------------------------------
    valid_ego_logits = _select_valid_ego_logits(
        ego_logits=ego_logits,
        valid_mask=valid_mask,
    )

    if valid_ego_logits.numel() > 0:
        # Use sum rather than mean.
        #
        # The gradient penalty helper already averages over valid input
        # entries. Using mean here would introduce another 1 / N factor,
        # causing the squared GP to scale approximately as 1 / N^2.
        scene_score = valid_ego_logits.sum()
    else:
        scene_score = _zero(sampled_pos)

    scene_gradients = _safe_autograd_grad(
        score=scene_score,
        inputs=(
            sampled_pos,
            sampled_heading,
            shape,
        ),
    )

    (
        scene_raw_penalty,
        scene_pos_penalty,
        scene_heading_penalty,
        scene_shape_penalty,
    ) = _compute_branch_raw_penalty(
        gradients=scene_gradients,
        sampled_pos=sampled_pos,
        sampled_heading=sampled_heading,
        shape=shape,
        valid_mask=valid_mask,
        position_scale=position_scale,
        heading_scale=heading_scale,
        shape_scale=shape_scale,
    )

    scene_gp = 0.5 * gamma * scene_raw_penalty

    # ---------------------------------------------------------
    # Interaction discriminator score
    # ---------------------------------------------------------
    interaction_logits_flat = interaction_logits.reshape(-1)
    destination_index_flat = destination_index.reshape(-1).long()

    if interaction_logits_flat.numel() == 0:
        interaction_raw_penalty = _zero(sampled_pos)
        interaction_gp = _zero(sampled_pos)

    else:
        if num_nodes is None:
            num_nodes = int(destination_index_flat.max().item()) + 1

        node_valid_mask = _infer_interaction_node_valid_mask(
            valid_mask=valid_mask,
            num_nodes=num_nodes,
            device=interaction_logits.device,
        )

        (
            interaction_node_logits,
            interaction_mass,
        ) = aggregate_interaction_logits_for_gp(
            interaction_logits=interaction_logits_flat,
            destination_index=destination_index_flat,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            node_valid_mask=node_valid_mask,
            min_mass=interaction_min_mass,
            detach_edge_weight=detach_edge_weight,
        )

        valid_interaction_nodes = (
            node_valid_mask
            & (interaction_mass > 1e-6)
        )
        interaction_score = (
            interaction_node_logits[valid_interaction_nodes]
            .sum()
        )

        #interaction_score=(interaction_logits*edge_weight).sum()

            # Same reasoning as scene_score: use sum, then average the
            # resulting input gradients over valid entries.

        interaction_gradients = _safe_autograd_grad(
            score=interaction_score,
            inputs=(
                sampled_pos,
                sampled_heading,
                shape,
            ),
        )

        (
            interaction_raw_penalty,
            interaction_pos_penalty,
            interaction_heading_penalty,
            interaction_shape_penalty,
        ) = _compute_branch_raw_penalty(
            gradients=interaction_gradients,
            sampled_pos=sampled_pos,
            sampled_heading=sampled_heading,
            shape=shape,
            valid_mask=valid_mask,
            position_scale=position_scale,
            heading_scale=heading_scale,
            shape_scale=shape_scale,
        )

        interaction_gp = (
            0.5
            * interaction_gamma
            * interaction_raw_penalty
        )

        # else:
        #     interaction_raw_penalty = _zero(sampled_pos)
        #     interaction_gp = _zero(sampled_pos)

    total_gp = scene_gp + interaction_gp

    raw_total_penalty = (
        scene_raw_penalty
        + interaction_raw_penalty
    )

    return (
        total_gp,
        scene_gp,
        interaction_gp,
        raw_total_penalty,
    )