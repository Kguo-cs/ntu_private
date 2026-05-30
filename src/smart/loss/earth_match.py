
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import math
import torch
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)
from torch_scatter import scatter_sum,scatter_mean

def gaussian_nll_2d(mu, sigma, target):
    dx = target - mu
    sigma = torch.clamp(sigma.exp(), min=1e-5)

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

def matching_loss(
    real_state,
    fake_state,
    w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
):

    fake_pos, fake_heading, fake_shape,fake_vel = fake_state[:, :2], fake_state[:, 2:4], fake_state[:, 4:6],fake_state[:, 6:]
    real_pos, real_heading, real_shape,real_vel = real_state[:, :2], real_state[:, 2:4], real_state[:, 4:6],real_state[:, 6:8]



    # Position: L1 or L2

    # dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)
    #
    # pos_loss = dist.mean()

    if fake_state.shape[-1]<16:
        pos_loss=F.mse_loss(fake_pos, real_pos, reduction="none").mean(-1)
        #pos_loss=torch.tensor(0.0).to(real_state.device)
        #fake_vel=torch.cat([fake_pos,fake_vel],dim=-1)
        #real_vel=torch.cat([real_pos,real_vel],dim=-1)
       # pos_loss=torch.norm(fake_pos- real_pos,dim=-1)#, reduction="none").mean(-1)

        #cluster_valid_mask=~torch.isnan(real_vel)

        heading_loss = F.mse_loss(fake_heading, real_heading, reduction="none").mean(-1)

        # if fake_state.shape[1]==44:
        #     vel_loss = F.l1_loss(fake_state[:, 4:], real_state[:, 4:], reduction="none").mean()
        #     shape_loss =torch.zeros_like(vel_loss)
        #
        # else:
        shape_loss = F.mse_loss(fake_shape, real_shape, reduction="none").mean(-1)

        vel_loss = F.mse_loss(fake_vel, real_vel, reduction="none").mean(-1)

        # pos_loss=get_scale(fake_pos,real_pos)
        # heading_loss=get_scale(fake_heading, real_heading)
        # shape_loss=get_scale(fake_shape, real_shape)
        # vel_loss=get_scale(fake_vel, real_vel)

    elif fake_state.shape[-1]==16:
        fake_vel = fake_state[:, 6:8]

        pos_std,heading_std, shape_std,vel_std=fake_state[:, 8:10], fake_state[:, 10:12], fake_state[:, 12:14], fake_state[:, 14:]

        pos_loss=gaussian_nll_2d(fake_pos,pos_std, real_pos).mean()
        heading_loss = gaussian_nll_2d(fake_heading,heading_std, real_heading).mean()
        shape_loss = gaussian_nll_2d(fake_shape, shape_std,real_shape).mean()
        vel_loss = gaussian_nll_2d(fake_vel, vel_std, real_vel).mean()
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

        #print(1)

        # Shape: L1
        # means(torch.Tensor): (bs, num_gaussians, D)
        # logstds(torch.Tensor): (bs, 1, 1)
        # logweights(torch.Tensor): (bs, num_gaussians, 1)

        # cluster_valid_mask=~torch.isnan(real_shape[:,2:])

        # cluster_valid_mask1=cluster_valid_mask.reshape(-1,90,2)

    # Heading: periodic-safe loss
    # heading_diff = torch.atan2(
    #     torch.sin(fake_heading - real_heading),
    #     torch.cos(fake_heading - real_heading)
    # )
    # heading_diff=wrap_angle(fake_heading - real_heading)
    # heading_loss = heading_diff.abs().mean()



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


    penetration_fake=compute_penetration(fake_state, start_idx, end_idx).clamp_max(max=1)*10

    #penetration_real=compute_penetration(real_state, start_idx, end_idx)

    fake_col = torch.relu(penetration_fake) #fake>0
    real_col = 0#torch.relu(penetration_real) #real>0

    loss = torch.relu(fake_col-real_col).expm1()#*10#*w[start_idx]

    return loss,end_idx,start_idx#.mean() if reduction == "mean" else loss.sum()

def get_col_rate(tokenized_agent,pred_init):

    col_reward, end_idx, start_idx = multi_circle_collision_loss_mem_efficient(pred_init[:, :2],
                                                                               torch.atan2(pred_init[:, 3],
                                                                                           pred_init[:, 2]),
                                                                               pred_init[:, 4], pred_init[:, 5],
                                                                               tokenized_agent["nonego_batch"])

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
        batch = tokenized_agent["nonego_batch"][-len(fake_state):]
        agent_type = tokenized_agent["nonego_type"][-len(fake_state):]
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

    return torch.cat(fake_idx_all), torch.cat(real_idx_all)

def get_matching_loss(
    tokenized_agent, fake_state,real_state,z,e,t,
    scale=1 ,all_state=False,use_col=False,use_all_type=False,use_match=False,x_pred=False,
    t_eps=0.05,w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
    ):

    if use_match:
        fake_idx, real_idx=get_closest_sum_idx(fake_state/scale, real_state/scale, tokenized_agent,all_state=all_state,use_all_type=use_all_type)

        fake_state=fake_state[fake_idx]
        real_state=real_state[real_idx]

    denom = (1 - t[~tokenized_agent["ego_mask"]]).clamp_min(t_eps)  # /t.clamp_min(self.t_eps)torch.ones_like(t) #

    # if x_pred:
    #
    #     v_target = (real_state - z) / denom
    #
    #     v_pred = (fake_state - z) / denom
    # else:
    # v_target = real_state/ denom #- e
    #
    # v_pred = fake_state/ denom

    denom_sq=denom.square()

    match_loss, pos_loss, heading_loss, shape_loss, vel_loss = matching_loss(
        real_state[~tokenized_agent["ego_mask"]], fake_state[~tokenized_agent["ego_mask"]],
        w_pos=w_pos/denom_sq[:,0], w_heading=w_heading/denom_sq[:,2], w_shape=w_shape/denom_sq[:,4], w_vel=w_vel/denom_sq[:,6]
    )

    # match_loss1, pos_loss, heading_loss, shape_loss, vel_loss = matching_loss(
    #     real_state[~tokenized_agent["ego_mask"]]/denom, fake_state[~tokenized_agent["ego_mask"]]/denom,
    #     w_pos=w_pos, w_heading=w_heading, w_shape=w_shape, w_vel=w_vel
    # )
    #
    if use_col and x_pred:
        # t_mask=t[:,0]>0.8
        #
        # fake_state=fake_state[t_mask]
        #
        # real_state = real_state[t_mask]
        #
        batch = tokenized_agent["nonego_batch"]#[t_mask]#[-len(fake_state):]
        denom = (1 - t[:,0]).clamp_min(t_eps)  # /t.clamp_min(self.t_eps)torch.ones_like(t) #
        w=1/denom.square()

        col_loss=multi_circle_collision_loss_mem_efficient(fake_state,real_state,batch,w)[0].mean()

        #
        # col_loss1=multi_circle_collision_loss_mem_efficient(real_state[:,:2], torch.atan2(real_state[:,3],real_state[:,2]), real_state[:,4].detach(),real_state[:,5].detach(),batch)[0].mean()

    else:
        col_loss = torch.zeros_like(match_loss.mean())  #.detach().detach()

    return match_loss,pos_loss,heading_loss,shape_loss,vel_loss,col_loss


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

def time_shift_fn(t, timeshift=1):
    return t/(t+(1-t)*timeshift)

def sample_cfg_scale(
    batch_size,
    cfg_min=0.3,
    cfg_max=3.0,
    device=None,
    dtype=torch.float32,
):
    """
    Sample CFG scale from log-uniform distribution
    in [cfg_min, cfg_max].

    Equivalent to the JAX version.
    """

    u = torch.rand(
        batch_size,
        device=device,
        dtype=dtype,
    )

    a = torch.tensor(
        1.0 + cfg_min,
        device=device,
        dtype=dtype,
    )

    b = torch.tensor(
        1.0 + cfg_max,
        device=device,
        dtype=dtype,
    )

    return a * torch.exp(
        u * torch.log(b / a)
    ) - 1.0
