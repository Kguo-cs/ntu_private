from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import math
import torch
import numpy as np
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    wrap_angle,
)
from torch_scatter import scatter_sum, scatter_mean,scatter_max
from torch import Tensor


def sort_agents_by_xy_keep_last(
    pos: Tensor,
    batch: Tensor,
    agent_type: Tensor,
    num_types: int = 3,
):
    """
    Sort agents by x + y inside each (batch, type) group.

    The last agent in each batch is not sorted and remains at its original
    absolute position. Sorting never moves an agent across batches or types.

    Args:
        pos:
            Agent positions with shape [N, 2] or [N, D].
            The first two columns are interpreted as x and y.

        batch:
            Batch index of each agent, shape [N].

        agent_type:
            Agent type of each agent, shape [N].
            Valid types are expected to be in [0, num_types - 1].

        num_types:
            Number of agent types.

    Returns:
        perm:
            Mapping from each output slot to its original input index:
                sorted_pos = pos[perm]

    """
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError("pos must have shape [N, D] with D >= 2.")

    if batch.ndim != 1 or agent_type.ndim != 1:
        raise ValueError("batch and agent_type must be 1D tensors.")

    if not (pos.shape[0] == batch.shape[0] == agent_type.shape[0]):
        raise ValueError("pos, batch, and agent_type must contain the same number of agents.")

    num_agents = batch.numel()

    if num_agents == 0:
        return torch.empty_like(batch)

    if torch.any(agent_type < 0) or torch.any(agent_type >= num_types):
        raise ValueError(f"agent_type must be in [0, {num_types - 1}].")

    device = batch.device
    original_idx = torch.arange(num_agents, device=device)

    # ---------------------------------------------------------------
    # 1. Detect the last occurrence of each batch in the original input.
    # ---------------------------------------------------------------
    num_batches = int(batch.max().item()) + 1

    last_idx_per_batch = torch.full(
        (num_batches,),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )

    # scatter_reduce_ is available in recent PyTorch versions.
    last_idx_per_batch.scatter_reduce_(
        dim=0,
        index=batch,
        src=original_idx,
        reduce="amax",
        include_self=True,
    )

    keep_fixed = original_idx == last_idx_per_batch[batch]
    sortable = ~keep_fixed

    # ---------------------------------------------------------------
    # 2. Sort movable agents lexicographically by:
    #       primary key:   (batch, type)
    #       secondary key: x + y
    #
    # Stable sorting ensures deterministic ordering when x + y ties.
    # ---------------------------------------------------------------
    sortable_idx = original_idx[sortable]

    sortable_batch = batch[sortable_idx]
    sortable_type = agent_type[sortable_idx]
    score = pos[sortable_idx, 0] + pos[sortable_idx, 1]

    group_id = sortable_batch * num_types + sortable_type

    # Stable lexicographic sort:
    # first sort by secondary key, then by primary key.
    order = torch.argsort(score, stable=True)
    order = order[torch.argsort(group_id[order], stable=True)]

    sorted_source_idx = sortable_idx[order]

    # ---------------------------------------------------------------
    # 3. Write sorted agents back into their original (batch, type)
    #    slots. Fixed last agents remain untouched.
    #
    # The sortable slots must also be arranged by (batch, type), while
    # preserving their original positions within each group.
    # ---------------------------------------------------------------
    slot_order = torch.argsort(group_id, stable=True)
    target_slots = sortable_idx[slot_order]

    perm = original_idx.clone()
    perm[target_slots] = sorted_source_idx

    # ---------------------------------------------------------------
    # 4. Compute rank within each sorted (batch, type) group.
    #    Fixed agents receive -1.
    # ---------------------------------------------------------------
    sorted_group_id = group_id[order]

    group_change = torch.ones_like(sorted_group_id, dtype=torch.bool)
    group_change[1:] = sorted_group_id[1:] != sorted_group_id[:-1]

    sorted_position = torch.arange(
        sorted_group_id.numel(),
        device=device,
        dtype=torch.long,
    )

    group_start = torch.where(group_change, sorted_position, 0)
    group_start = torch.cummax(group_start, dim=0).values

    sorted_rank = sorted_position - group_start

    pos_idx = torch.full(
        (num_agents,),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    pos_idx[target_slots] = sorted_rank

    return perm

def sinusoidal_embedding(position, D, device=None):
    """Create sinusoidal embeddings for explicit positions or ``range(N)``."""
    if D % 2 != 0:
        raise ValueError(f"D must be even, got {D}.")

    if isinstance(position, int):
        if position < 0:
            raise ValueError("position count must be non-negative.")
        position = torch.arange(position, device=device, dtype=torch.float32)[:, None]
    elif torch.is_tensor(position):
        if position.ndim == 1:
            position = position[:, None]
        elif position.ndim != 2 or position.shape[1] != 1:
            raise ValueError("position must be an int, [N], or [N, 1] tensor.")
        position = position.to(dtype=torch.float32)
    else:
        raise TypeError("position must be an int or a tensor.")

    div_term = torch.exp(
        torch.arange(0, D, 2, device=position.device, dtype=position.dtype)
        * (-math.log(10000.0) / D)
    )
    pe = torch.zeros(position.shape[0], D, device=position.device, dtype=position.dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def get_type_position_index(
    batch: Tensor,
    agent_type: Tensor,
    num_types: int = 3,
    invalid_value: int = -1,
) -> Tensor:
    """
    Assign a zero-based position index to each valid agent within its
    corresponding (batch, agent_type) group.

    The input does not need to be sorted.

    Args:
        batch:
            Tensor of shape [N]. batch[i] is the batch index of agent i.

        agent_type:
            Tensor of shape [N]. agent_type[i] is the type of agent i.
            Valid types are 0, ..., num_types - 1.
            Invalid entries, such as -1, receive invalid_value.

        num_types:
            Total number of valid agent types.

        invalid_value:
            Output value assigned to invalid entries.

    Returns:
        pos_idx:
            Tensor of shape [N]. For each valid agent, pos_idx[i] is its
            zero-based index among agents with the same batch and type.
    """
    if batch.ndim != 1 or agent_type.ndim != 1:
        raise ValueError("batch and agent_type must be 1D tensors.")

    if batch.shape != agent_type.shape:
        raise ValueError("batch and agent_type must have the same shape.")

    pos_idx = torch.full_like(batch, invalid_value)

    # Only assign indices to valid agent types.
    valid_mask = (agent_type >= 0) & (agent_type < num_types)

    if not valid_mask.any():
        return pos_idx

    valid_batch = batch[valid_mask]
    valid_type = agent_type[valid_mask]

    # Unique group ID for each (batch, type) pair.
    group_id = valid_batch * num_types + valid_type

    # Stable sorting preserves the original order within each group.
    sorted_group_id, perm = torch.sort(group_id, stable=True)

    # Identify the first element of every group.
    group_change = torch.ones_like(sorted_group_id, dtype=torch.bool)
    group_change[1:] = sorted_group_id[1:] != sorted_group_id[:-1]

    sorted_pos = torch.arange(
        sorted_group_id.numel(),
        device=sorted_group_id.device,
        dtype=sorted_group_id.dtype,
    )

    group_start = torch.where(group_change, sorted_pos, 0)
    group_start = torch.cummax(group_start, dim=0).values

    # Rank inside each (batch, type) group.
    sorted_rank = sorted_pos - group_start

    # Restore the original order of valid agents.
    valid_pos_idx = torch.empty_like(sorted_rank)
    valid_pos_idx[perm] = sorted_rank

    # Insert valid results into the complete output tensor.
    pos_idx[valid_mask] = valid_pos_idx

    return pos_idx

def gaussian_nll_2d(mu, sigma, target,min=1e-3):
    dx = target - mu
    sigma = torch.clamp(sigma.exp(), min=1e-4)

    n = target.shape[-1]

    loss = 0.5 * ((dx / sigma) ** 2).sum(dim=-1)
    loss += torch.log(sigma).sum(dim=-1)
    loss += 0.5 * n * math.log(2 * math.pi)

    return loss

def gm_kl_loss(means,logweights,logstds, sample, eps=1e-4):
    """
    Gaussian mixture KL divergence loss (without constant terms), a.k.a. GM NLL loss.

    Args:
        gm (dict):
            means (torch.Tensor): (bs, num_gaussians, D)
            logstds (torch.Tensor): (bs, 1, 1)
            logweights (torch.Tensor): (bs, num_gaussians, 1)
        sample (torch.Tensor): (bs, D)

    Returns:
        torch.Tensor: (bs, )
    """

   # means=means.reshape(means.shape[0],-1,2)
    logweights=logweights[:,:,None]
    logstds=logstds[:,None]

    inverse_stds = torch.exp(-logstds).clamp(max=1 / eps)
    diff_weighted = (sample.unsqueeze(-2) - means) * inverse_stds  # (bs, num_gaussians, D)
    gaussian_ll = (-0.5 * diff_weighted.square() - logstds).sum(dim=-1)  # (bs, num_gaussians)
    gm_nll = -torch.logsumexp(gaussian_ll + logweights.squeeze(-1), dim=-1)  # (bs, )
    return gm_nll


def get_scale(x0_prediction,x0):
    weight_factor=torch.abs(x0_prediction.double() - x0.double()) .mean(dim=tuple(range(1, x0.ndim)), keepdim=True) .clip(min=0.00001)

    return  ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

def _robust_component_loss(fake, real, beta=0.2, use_huber=True):
    if use_huber:
        return 2.0 * beta*F.smooth_l1_loss(fake, real, reduction="none", beta=beta).mean(-1)
    return F.mse_loss(fake, real, reduction="none").mean(-1)

def matching_loss(
    real_state,
    fake_state,
    w_pos=0.1, w_heading=0.5, w_shape=0.2, w_vel=0.2,
    use_huber=False,
    huber_beta=0.1,
    t=None
):

    fake_pos, fake_heading, fake_shape,fake_vel = fake_state[:, :2], fake_state[:, 2:4], fake_state[:, 4:6],fake_state[:, 6:]
    real_pos, real_heading, real_shape,real_vel = real_state[:, :2], real_state[:, 2:4], real_state[:, 4:6],real_state[:, 6:8]



    if fake_state.shape[-1]<16:
        pos_loss = _robust_component_loss(fake_pos, real_pos, beta=1, use_huber=True)
        heading_loss = _robust_component_loss(fake_heading, real_heading, beta=1, use_huber=True)
        shape_loss = _robust_component_loss(fake_shape, real_shape, beta=1, use_huber=True)
        vel_loss = _robust_component_loss(fake_vel, real_vel, beta=1, use_huber=True)

    elif fake_state.shape[-1]==16:
        fake_vel = fake_state[:, 6:8]

        pos_std,heading_std, shape_std,vel_std=fake_state[:, 8:10], fake_state[:, 10:12], fake_state[:, 12:14], fake_state[:, 14:]

        pos_loss=gaussian_nll_2d(fake_pos,pos_std, real_pos,min=1e-3).mean()
        heading_loss = gaussian_nll_2d(fake_heading,heading_std, real_heading, min=1e-2).mean()
        shape_loss = gaussian_nll_2d(fake_shape, shape_std,real_shape, min=1e-2).mean()
        vel_loss = gaussian_nll_2d(fake_vel, vel_std, real_vel, min=1e-2).mean()
    else:
        K=8
        mean = fake_state[:, :8 * K].reshape(-1,K,8)

        fake_pos, fake_heading, fake_shape, fake_vel = mean[:,:, :2], mean[:, :, 2:4], mean[:,:,  4:6],mean[:,:,  6:8]

        w=fake_state[:, 8*K:9*K].log_softmax(dim=1)

        pos_std,heading_std, shape_std,vel_std=fake_state[:, 9*K:9*K+2], fake_state[:, 9*K+2:9*K+4], fake_state[:, 9*K+4:9*K+6], fake_state[:, 9*K+6:]

        pos_loss=gm_kl_loss(fake_pos,w,pos_std, real_pos)#.mean()
        heading_loss = gm_kl_loss(fake_heading,w,heading_std, real_heading)#.mean()
        shape_loss = gm_kl_loss(fake_shape, w,shape_std,real_shape)#.mean()
        vel_loss = gm_kl_loss(fake_vel, w,vel_std, real_vel)#.mean()


    total_loss = (
        w_pos  *pos_loss +
        w_heading  * heading_loss +
        w_shape  *shape_loss+
        w_vel*vel_loss
    )

    return total_loss,pos_loss,heading_loss,shape_loss,vel_loss

def compute_vehicle_circles_torch(
    pos,        # (N, 2)
    heading,    # (N,)
    length,     # (N,)
    width,      # (N,)
    num_circles=5
):
    """
    Returns:
        centers: (N, C, 2)
        radii:   (N, C)
    """
    device = pos.device
    C = num_circles

    radius = width / 2.0                         # (N,)
    rel_x = torch.linspace(
        -0.5, 0.5, C, device=device
    )[None] * (length[:, None] - 2 * radius[:, None])

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    dx = cos_h[:, None] * rel_x
    dy = sin_h[:, None] * rel_x

    centers = torch.stack([
        pos[:, 0][:, None] + dx,
        pos[:, 1][:, None] + dy
    ], dim=-1)

    radii = width[:, None].expand(-1, C)/ torch.sqrt(torch.tensor(3.8, device=device))

    return centers, radii

def compute_penetration(state, start_idx, end_idx, num_circles=5, eps=1e-8):
    pos = state[:, :2]
    heading = torch.atan2(state[:, 3], state[:, 2])

    # detach if you do not want gradients w.r.t. shape
    length = state[:, 4].detach()
    width = state[:, 5].detach()

    centers, radii = compute_vehicle_circles_torch(
        pos, heading, length, width, num_circles
    )  # centers: (N, C, 2), radii: (N, C)

    ci = centers[start_idx]  # (E, C, 2)
    cj = centers[end_idx]    # (E, C, 2)

    ri = radii[start_idx]    # (E, C)
    rj = radii[end_idx]      # (E, C)

    diff = ci[:, :, None, :] - cj[:, None, :, :]  # (E, C, C, 2)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (E, C, C)

    pair_penetration = ri[:, :, None] + rj[:, None, :] - dist  # (E, C, C)

    # Positive means collision/overlap, negative means no collision
    penetration = pair_penetration.amax(dim=(1, 2))  # (E,)

    return penetration

def multi_circle_collision_loss_mem_efficient( fake_state,real_state, batch,w):

    same_batch = batch[:, None] == batch[None, :]
    not_self = ~torch.eye(len(fake_state), dtype=torch.bool, device=fake_state.device)
    edge_mask = same_batch & not_self

    start_idx, end_idx = edge_mask.nonzero(as_tuple=True)

    mask = start_idx < end_idx
    start_idx = start_idx[mask]
    end_idx = end_idx[mask]


    penetration_fake=compute_penetration(fake_state, start_idx, end_idx)#.clamp_max(max=2)*2

    #penetration_real=compute_penetration(real_state, start_idx, end_idx)

    fake_col = torch.relu(penetration_fake) #fake>0
    real_col = 0#torch.relu(penetration_real) #real>0

    loss = torch.relu(fake_col-real_col).expm1()

    return loss,end_idx,start_idx#.mean() if reduction == "mean" else loss.sum()

def get_col_rate(tokenized_agent,pred_init):

    col_reward, end_idx, start_idx = multi_circle_collision_loss_mem_efficient(pred_init[:, :2],
                                                                               torch.atan2(pred_init[:, 3],
                                                                                           pred_init[:, 2]),
                                                                               pred_init[:, 4], pred_init[:, 5],
                                                                               tokenized_agent["batch"])

    N = len(pred_init)

    col_reward_end = scatter_sum(
        col_reward,
        end_idx,
        dim=0,
        dim_size=N
    )

    col_reward_start = scatter_sum(
        col_reward,
        start_idx,
        dim=0,
        dim_size=N
    )

    col_reward_agent = -col_reward_end - col_reward_start

    noncol_rate = (col_reward_agent == 0).float()  # -0.5#col_reward <0 collision 0

    return noncol_rate

def get_closest_sum_idx(
    fake_state,
    real_state,
    tokenized_agent,
    all_state=False,
    use_all_type=False,
):
    fake_feat = fake_state if all_state else fake_state[:, :2]
    real_feat = real_state if all_state else real_state[:, :2]

    if use_all_type:
        batch = tokenized_agent
        groups = [(batch == b) for b in batch.unique()]
    else:
        batch = tokenized_agent["batch"][-len(fake_state):]
        agent_type = tokenized_agent["type"][-len(fake_state):]
        groups = [
            (batch == b) & (agent_type == t)
            for b in batch.unique()
            for t in agent_type[batch == b].unique()
        ]

    fake_idx_all, real_idx_all = [], []

    for mask in groups:
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue

        dist = torch.cdist(real_feat[idx], fake_feat[idx]).square()

        row, col = linear_sum_assignment(dist.detach().cpu().numpy())

        # row=torch.arange(len(real_feat[idx]))
        # col=torch.randperm(len(real_feat[idx]))
        #
        real_idx_all.append(idx[row])
        fake_idx_all.append(idx[col])

    real_idx=torch.cat(real_idx_all)
    inv_real_idx = torch.empty_like(real_idx)
    inv_real_idx[real_idx] = torch.arange(real_idx.numel(), device=real_idx.device)

    fake_idx=torch.cat(fake_idx_all)[inv_real_idx]

    return fake_idx


@torch.no_grad()
def get_closest_sum_idx_fast(
    fake_state: torch.Tensor,
    real_state: torch.Tensor,
    tokenized_agent,
    all_state: bool = False,
    use_all_type: bool = False,
) -> torch.Tensor:
    """
    Exact one-to-one Hungarian matching inside each group.

    Returns:
        fake_idx: [N]
            fake_state[fake_idx[i]] is assigned to real_state[i].
    """
    if fake_state.shape[0] != real_state.shape[0]:
        raise ValueError(
            "fake_state and real_state must have the same number of states: "
            f"{fake_state.shape[0]} vs {real_state.shape[0]}"
        )

    num_states = fake_state.shape[0]
    output_device = fake_state.device

    if num_states == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=output_device,
        )

    fake_feat = fake_state if all_state else fake_state[:, :2]
    real_feat = real_state if all_state else real_state[:, :2]

    # ------------------------------------------------------------
    # Transfer features to CPU only once.
    # scipy.linear_sum_assignment is CPU-based anyway.
    # ------------------------------------------------------------
    fake_np = (
        fake_feat.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    real_np = (
        real_feat.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )

    # ------------------------------------------------------------
    # Build groups by sorting rather than repeatedly constructing
    # full-length Boolean masks.
    # ------------------------------------------------------------
    if use_all_type:
        batch_tensor = torch.as_tensor(tokenized_agent)[-num_states:]

        batch_np = (
            batch_tensor.detach()
            .to(device="cpu", dtype=torch.long)
            .numpy()
        )

        order = np.argsort(batch_np, kind="stable")
        sorted_batch = batch_np[order]

        group_change = (
            sorted_batch[1:] != sorted_batch[:-1]
        )
    else:
        batch_tensor = tokenized_agent["batch"][-num_states:]
        type_tensor = tokenized_agent["type"][-num_states:]

        batch_np = (
            batch_tensor.detach()
            .to(device="cpu", dtype=torch.long)
            .numpy()
        )
        type_np = (
            type_tensor.detach()
            .to(device="cpu", dtype=torch.long)
            .numpy()
        )

        # Lexicographic ordering by (batch, type).
        order = np.lexsort((type_np, batch_np))

        sorted_batch = batch_np[order]
        sorted_type = type_np[order]

        group_change = (
            (sorted_batch[1:] != sorted_batch[:-1])
            | (sorted_type[1:] != sorted_type[:-1])
        )

    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(group_change).astype(np.int64) + 1,
        )
    )
    group_ends = np.concatenate(
        (
            group_starts[1:],
            np.array([num_states], dtype=np.int64),
        )
    )

    # Default identity assignment also handles singleton groups.
    matched_fake_idx = np.arange(
        num_states,
        dtype=np.int64,
    )

    # ------------------------------------------------------------
    # Exact Hungarian matching inside each group.
    # ------------------------------------------------------------
    for start, end in zip(group_starts, group_ends):
        idx = order[start:end]
        group_size = end - start

        if group_size <= 1:
            continue

        real_group = real_np[idx]  # [M, D]
        fake_group = fake_np[idx]  # [M, D]

        # Squared Euclidean distance without an [M, M, D]
        # intermediate tensor:
        #
        # ||r-f||² = ||r||² + ||f||² - 2 r f^T
        real_sq = np.sum(
            real_group * real_group,
            axis=1,
            keepdims=True,
        )
        fake_sq = np.sum(
            fake_group * fake_group,
            axis=1,
            keepdims=True,
        ).T

        cost = (
            real_sq
            + fake_sq
            - 2.0 * real_group @ fake_group.T
        )

        # Remove tiny negative values caused by floating-point error.
        np.maximum(cost, 0.0, out=cost)

        if not np.isfinite(cost).all():
            raise ValueError(
                "Non-finite values were found in the matching "
                f"cost matrix for a group of size {group_size}."
            )

        # Avoid calling scipy for the common two-agent case.
        if group_size == 2:
            identity_cost = cost[0, 0] + cost[1, 1]
            swap_cost = cost[0, 1] + cost[1, 0]

            if swap_cost < identity_cost:
                matched_fake_idx[idx[0]] = idx[1]
                matched_fake_idx[idx[1]] = idx[0]
            else:
                matched_fake_idx[idx] = idx

            continue

        row, col = linear_sum_assignment(cost)

        # Directly store the assignment in original real-state order.
        matched_fake_idx[idx[row]] = idx[col]

    return torch.from_numpy(
        matched_fake_idx
    ).to(device=output_device)

def get_diff_loss(
    tokenized_agent, fake_state,real_state,t,t_eps,scale=1 ,all_state=False,use_col=False,use_all_type=False,use_match=False,x_pred=False,
    w_pos=0.1/ 5, w_heading=0.5/ 5, w_shape=0.2/ 5, w_vel=0.2/ 5,
    #w_pos=0.1, w_heading=0.1, w_shape=0.1, w_vel=0.1,
    # w_pos = 0.0015,
    # w_heading = 0.5,
    # w_shape = 0.3,
    # w_vel = 0.075,
    max_loss_weight=25.0,
    use_huber=False,
    huber_beta=0.1,
    ):

    batch = tokenized_agent["batch"][-len(fake_state):]  # [t_mask]#

    if use_match:
        fake_idx=get_closest_sum_idx(fake_state/scale, real_state/scale, tokenized_agent,all_state=all_state,use_all_type=use_all_type)

        fake_state=fake_state[fake_idx]

    if use_col and x_pred:
        pos_denom = (1 - t[:,0]).clamp_min(t_eps)
        w=1/pos_denom.square()

        col_loss,end_idx,start_idx=multi_circle_collision_loss_mem_efficient(fake_state,real_state,batch,w)#[0].mean()
        #
        # edge_idx = torch.cat([start_idx, end_idx], dim=0)
        # edge_loss = torch.cat([col_loss, col_loss], dim=0)
        #
        # col_loss = scatter_sum(
        #     edge_loss,
        #     edge_idx,
        #     dim=0,
        #     dim_size=len(fake_state),
        # )/50#[0]

        # col_loss1 = scatter_mean(
        #     edge_loss,
        #     edge_idx,
        #     dim=0,
        #     dim_size=len(fake_state),
        # )#[0]

        # col_loss = scatter_max(
        #     edge_loss,
        #     edge_idx,
        #     dim=0,
        #     dim_size=len(fake_state),
        # )[0]/50


        col_loss=(col_loss*w[start_idx])

       # col_loss=scatter_mean(col_loss,batch)

    else:
        col_loss = torch.zeros_like(fake_state.mean())  #.detach().detach()

    t_mask = (t[:,0] > 0) & (t[:,0] < 1)

    if x_pred:

        denom = (1 - t).clamp_min(t_eps)

        denom_sq = denom.square()

        inv_denom_sq = denom_sq.reciprocal()*t_mask.float()#*t_dt.square()#.clamp(max=max_loss_weight)
    else:
        inv_denom_sq  = t_mask.float()

    match_loss, pos_loss, heading_loss, shape_loss, vel_loss = matching_loss(
        real_state,
        fake_state,
        w_pos=w_pos * inv_denom_sq[:, 0],
        w_heading=w_heading * inv_denom_sq[:, 0],
        w_shape=w_shape * inv_denom_sq[:, 0],
        w_vel=w_vel * inv_denom_sq[:, 0],
        use_huber=use_huber,
        huber_beta=huber_beta,
    )

   # match_loss=scatter_mean(match_loss,batch)

    return match_loss, col_loss.mean() , pos_loss, heading_loss, shape_loss, vel_loss


def sample_linear_t(
    num_samples,
    a,
    device=None,
    dtype=torch.float32,
):

    if abs(a) < 1e-8:
        return torch.rand(
            num_samples,
            device=device,
            dtype=dtype,
        )

    assert 0 <= a <= 2

    b = 1.0 - a / 2.0

    u = torch.rand(
        num_samples,
        device=device,
        dtype=dtype,
    )

    return (
        -b
        + torch.sqrt(b * b + 2 * a * u)
    ) / a


def calculate_shift(
    image_seq_len,
    base_seq_len: int =32, #256,
    max_seq_len: int = 256,#4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def adaptive_x0_loss_per_sample(pred_x0, target_x0):
    """
    pred_x0:   [B, A, T, D]
    target_x0: [B, A, T, D]
    valid_mask:[B, A, T]
    return:    [B]
    """
    err = pred_x0 - target_x0

    mse = err.square()  # [B, A, T, D]
    l1 = err.abs().mean(dim=-1, keepdim=True)
    l1 = l1.clamp_min(1e-5).detach()

    loss = (mse / l1).mean(dim=-1)  # [B, A, T]
    # denom = (1 - t_n_sampled[:, 0]).clamp_min(self.t_eps)
    # denom_sq = denom.square()
    # sampled_match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_diff_loss(
    #     tokenized_agent,
    #     x_pred[:, 0],
    #     x_sampled[:, 0],
    #     z_sampled[:, 0],
    #     e_sampled[:, 0],
    #     t_n_sampled[:, 0],
    #     use_col=False,
    #     x_pred=self.x_pred
    # )

    # inv_denom_sq = denom_sq.reciprocal()
    # mse_Loss = F.mse_loss(x_pred[:, 0] / scale, x_sampled[:, 0] / scale, reduction="none")
    # # l1_Loss=F.l1_loss(x_pred[:, 0] /scale, x_sampled[:, 0]/scale , reduction="none").mean(-1, keepdim=True).clip(min=0.00001).detach()
    # #
    # sampled_match_loss = (mse_Loss * inv_denom_sq).mean(-1).reshape(self.mc_num, -1).mean(0)
    #
    return -loss.mean(dim=1)*0.1
