
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

def gaussian_nll(mu, sigma, target):
    dx = target - mu
    sigma = torch.clamp(sigma, min=1e-5)

    n = target.shape[-1]

    loss = 0.5 * ((dx / sigma) ** 2).sum(dim=-1)
    loss += torch.log(sigma).sum(dim=-1)
    loss += 0.5 * n * math.log(2 * math.pi)

    return loss

def matching_loss(
    fake_state,
    real_state,
    w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
):

    fake_pos, fake_heading, fake_shape,fake_vel = fake_state[:, :2], fake_state[:, 2:4], fake_state[:, 4:6],fake_state[:, 6:]
    real_pos, real_heading, real_shape,real_vel = real_state[:, :2], real_state[:, 2:4], real_state[:, 4:6],real_state[:, 6:]



    # Position: L1 or L2

    # dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)
    #
    # pos_loss = dist.mean()

    if fake_state.shape[1]!=16:
        pos_loss=F.l1_loss(fake_pos, real_pos, reduction="none").mean()
        #pos_loss=torch.tensor(0.0).to(real_state.device)
        #fake_vel=torch.cat([fake_pos,fake_vel],dim=-1)
        #real_vel=torch.cat([real_pos,real_vel],dim=-1)

        #cluster_valid_mask=~torch.isnan(real_vel)

        heading_loss = F.l1_loss(fake_heading, real_heading, reduction="none").mean()

        # if fake_state.shape[1]==44:
        #     vel_loss = F.l1_loss(fake_state[:, 4:], real_state[:, 4:], reduction="none").mean()
        #     shape_loss =torch.zeros_like(vel_loss)
        #
        # else:
        shape_loss = F.l1_loss(fake_shape, real_shape, reduction="none").mean()

        vel_loss = F.l1_loss(fake_vel, real_vel, reduction="none").mean()

    else:
        pos_std,heading_std, shape_std,vel_std=fake_state[:, 8:10], fake_state[:, 10:12], fake_state[:, 12:14], fake_state[:, 14:]

        pos_loss=gaussian_nll_2d(fake_pos,pos_std, real_pos).mean()
        heading_loss = gaussian_nll_2d(fake_heading,heading_std, real_heading).mean()
        shape_loss = gaussian_nll_2d(fake_shape, shape_std,real_shape).mean()
        vel_loss = gaussian_nll_2d(fake_vel, vel_std, real_vel).mean()


        # Shape: L1

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

    radii = radius[:, None].expand(-1, C)

    return centers, radii

def multi_circle_collision_loss_mem_efficient(
    pos, heading, length, width, batch,
    num_circles=5,
    reduction="mean",
    use_edge=True
):
    device = pos.device
    N = pos.shape[0]

    centers, _ = compute_vehicle_circles_torch(
        pos, heading, length, width, num_circles
    )  # (N, C, 2)

    same_batch = batch[:, None] == batch[None, :]
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    edge_mask = same_batch & not_self

    if use_edge:

        start_idx, end_idx = edge_mask.nonzero(as_tuple=True)

        mask = start_idx < end_idx
        start_idx = start_idx[mask]
        end_idx = end_idx[mask]

        # Gather centers for edges
        ci = centers[start_idx]  # (E, C, 2)
        cj = centers[end_idx]  # (E, C, 2)

        # Compute pairwise circle distances per edge
        # (E, C, C, 2)
        diff = ci[:, :, None, :] - cj[:, None, :, :]
        dist = torch.norm(diff, dim=-1)  # (E, C, C)

        # min over all circle pairs
        min_dist = dist.amin(dim=(1, 2))  # (E,)

        # collision threshold per edge
        thresh = (width[start_idx] + width[end_idx]) / torch.sqrt(
            torch.tensor(3.8, device=device)
        )  # (E,)
    else:

        # 初始化为 +inf
        min_dist = torch.full((N, N), float("inf"), device=device)

        for i in range(num_circles):
            ci = centers[:, i]           # (N, 2)
            for j in range(num_circles):
                cj = centers[:, j]       # (N, 2)
                d = torch.cdist(ci, cj)  # (N, N)
                min_dist = torch.minimum(min_dist, d)
            # diff = ci[:, None, None, :] - centers[None, :, :, :]
            # dist = torch.norm(diff, dim=-1)   # (N, N, C)
            #
            # # min over j circles
            # min_dist = torch.minimum(min_dist, dist.amin(dim=-1))
        min_dist=min_dist[edge_mask]

        thresh = (width[:, None] + width[None, :]) / torch.sqrt(
            torch.tensor(3.8, device=device)
        )[edge_mask]

    penetration = thresh - min_dist

    loss = torch.relu(penetration).expm1()*100

    if loss.numel() == 0:
        return torch.tensor(0.0, device=device)

    return loss.mean() if reduction == "mean" else loss.sum()



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
    tokenized_agent, fake_state,real_state,
    denom,scale=1 ,all_state=False,use_col=False,use_all_type=False,use_match=True,
    w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
    ):

    if use_match:
        fake_idx, real_idx=get_closest_sum_idx(fake_state/scale, real_state/scale, tokenized_agent,all_state=all_state,use_all_type=use_all_type)

        fake_state=fake_state[fake_idx]
        real_state=real_state[real_idx]


    match_loss, pos_loss, heading_loss, shape_loss, vel_loss = matching_loss(
        fake_state/denom, real_state/denom,
        w_pos=w_pos, w_heading=w_heading, w_shape=w_shape, w_vel=w_vel
    )

   # if latent or use_all_type:

    # match_loss = torch.norm(fake_norm_state[row] - real_norm_state[col],p=1,dim=-1).mean()#.square()

    # match_loss=((fake_state[row] - real_state[col]) ** 2)#.mean()

    # radius = 0.5 * torch.norm(fake_shape, dim=-1)  # circumscribed circle
    #
    # dist = torch.cdist(fake_pos, fake_pos)
    # penetration = radius[:, None] + radius[None, :] - dist
    # same_batch = batch[:, None] == batch[None, :]
    #
    # # remove self-collision
    # not_self = ~torch.eye(len(batch), dtype=torch.bool, device=batch.device)
    #
    # mask = same_batch & not_self
    #
    # col_loss = torch.relu(penetration)[mask].mean()# [02:20<09:00,  5.59it/s, v_num=05jo]

    #col_loss=collision_loss(fake_pos, fake_heading, fake_shape,batch )#[00:28<19:45,  3.13it/s, v_num=oc9q]
    #col_loss=torch.zeros_like(match_loss)#
    if use_col:
        col_loss=multi_circle_collision_loss_mem_efficient(fake_state[:,:2], torch.atan2(fake_state[:,3],fake_state[:,2]), fake_state[:,4],fake_state[:,5],tokenized_agent["nonego_batch"] )
    else:
        col_loss = torch.zeros_like(match_loss.mean())  #

    # [03:08<13:55,  3.71it/s, v_num=4e1b]

    return match_loss,pos_loss,heading_loss,shape_loss,vel_loss,col_loss

