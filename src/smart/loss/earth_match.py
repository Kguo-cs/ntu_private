
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

def gaussian_nll_2d(mu, sigma, target):
    # mu: (..., 2)
    # sigma: (..., 2)  (std, must be >0)
    # target: (..., 2)

    dx = target - mu
    sigma = torch.clamp(sigma, min=1e-5)
    var = sigma ** 2

    loss = 0.5 * (dx**2 / var).sum(dim=-1)
    loss += torch.log(sigma).sum(dim=-1)
    loss += math.log(2 * math.pi)

    return loss#.mean()

def matching_loss(
    fake_state,
    real_state,
    w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
):

    fake_pos, fake_heading, fake_shape,fake_vel = fake_state[:, :2], fake_state[:, 2:4], fake_state[:, 4:6],fake_state[:, 6:8]
    real_pos, real_heading, real_shape,real_vel = real_state[:, :2], real_state[:, 2:4], real_state[:, 4:6],real_state[:, 6:8]



    # Position: L1 or L2

    # dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)
    #
    # pos_loss = dist.mean()

    if fake_state.shape[1]!=16:
        pos_loss=F.l1_loss(fake_pos, real_pos, reduction="none").mean(-1)
        #pos_loss=torch.tensor(0.0).to(real_state.device)

        heading_loss = F.l1_loss(fake_heading, real_heading, reduction="none").mean(-1)
        shape_loss = F.l1_loss(fake_shape, real_shape, reduction="none").mean(-1)

        #fake_vel=torch.cat([fake_pos,fake_vel],dim=-1)
        #real_vel=torch.cat([real_pos,real_vel],dim=-1)

        #cluster_valid_mask=~torch.isnan(real_vel)

        vel_loss = F.l1_loss(fake_vel, real_vel, reduction="none").mean(-1)

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

def box_projection_radius(length, width, axis_x, axis_y, cos_h, sin_h):
    """
    Projection radius of a rotated rectangle onto an axis
    """
    # box local axes
    ux_x, ux_y = cos_h, sin_h
    uy_x, uy_y = -sin_h, cos_h

    r = (
        0.5 * length * torch.abs(axis_x * ux_x + axis_y * ux_y) +
        0.5 * width  * torch.abs(axis_x * uy_x + axis_y * uy_y)
    )
    return r

def collision_loss(
    fake_pos,        # (N, 2)
    fake_heading,    # (N, 2) -> (cos, sin)
    fake_shape,      # (N, 2) -> (length, width)
    batch,
    margin=0.0,
    reduction="mean"
):
    device = fake_pos.device
    total_loss = 0.0
    total_pairs = 0

    for b in batch.unique():
        idx = (batch == b).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue

        pos = fake_pos[idx]          # (M, 2)
        heading = fake_heading[idx]  # (M, 2)
        shape = fake_shape[idx]      # (M, 2)

        M = pos.shape[0]

        # pairwise differences
        delta = pos[:, None, :] - pos[None, :, :]      # (M, M, 2)
        dist = torch.norm(delta + 1e-6, dim=-1)        # (M, M)

        # unit direction vectors
        axis = delta / (dist[..., None] + 1e-6)
        axis_x, axis_y = axis[..., 0], axis[..., 1]

        # projection radii
        r_i = box_projection_radius(
            shape[:, 0][:, None], shape[:, 1][:, None],
            axis_x, axis_y,
            heading[:, 0][:, None], heading[:, 1][:, None]
        )

        r_j = box_projection_radius(
            shape[:, 0][None], shape[:, 1][None],
            axis_x, axis_y,
            heading[:, 0][None], heading[:, 1][None]
        )

        # penetration depth
        penetration = r_i + r_j - dist + margin

        # mask self-collision
        mask = ~torch.eye(M, dtype=torch.bool, device=device)

        collision = torch.relu(penetration)[mask]

        total_loss += collision.sum()
        total_pairs += collision.numel()

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)

    if reduction == "mean":
        return total_loss / total_pairs
    else:
        return total_loss


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
    reduction="mean"
):
    device = pos.device
    N = pos.shape[0]

    centers, _ = compute_vehicle_circles_torch(
        pos, heading, length, width, num_circles
    )  # (N, C, 2)

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

    thresh = (width[:, None] + width[None, :]) / torch.sqrt(
        torch.tensor(3.8, device=device)
    )

    penetration = thresh - min_dist

    same_batch = batch[:, None] == batch[None, :]
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)

    loss = torch.relu(penetration)[same_batch & not_self]

    if loss.numel() == 0:
        return torch.tensor(0.0, device=device)

    return loss.mean() if reduction == "mean" else loss.sum()



def get_closest_sum_idx(fake_state,real_state,batch,initial_type,all_state=False,use_all_type=False):

    fake_pos = fake_state[:, :2]
    real_pos = real_state[:, :2]


    if use_all_type:
        rows, cols = [], []

        for b in batch.unique():
            f_idx = ((batch == b) ).nonzero(as_tuple=True)[0]

            if all_state:
                dist = torch.norm(fake_state[f_idx][:, None] - real_state[f_idx][None], p=1, dim=-1)  # .square()
            else:
                dist = torch.cdist(fake_pos[f_idx], real_pos[f_idx])

            cost = dist.cpu().detach().numpy()

            row, col = linear_sum_assignment(cost)

            rows.append(f_idx[row])
            cols.append(f_idx[col])
    else:
        rows, cols = [], []

        for b in batch.unique():
            for type in initial_type[batch == b].unique():
                f_idx = ((batch == b) & (initial_type == type)).nonzero(as_tuple=True)[0]

                if all_state:
                    dist = torch.cdist(fake_state[f_idx],real_state[f_idx]).square()
                else:
                    dist = torch.cdist( real_pos[f_idx],fake_pos[f_idx])

                cost = dist.cpu().detach().numpy()

                row, col = linear_sum_assignment(cost)

                rows.append(f_idx[row])
                cols.append(f_idx[col])

    real_idx = torch.cat(rows)
    fake_idx = torch.cat(cols)

    # print(torch.all(torch.all(real_idx==torch.arange(len(real_idx)).cuda())))

    return fake_idx, real_idx


def get_matching_loss(
    tokenized_agent, fake_state,real_state,
    denom ,all_state=False,use_col=False,use_all_type=False
    ):
    initial_type, batch=tokenized_agent['nonego_type'],tokenized_agent["nonego_batch"]

    fake_idx, real_idx=get_closest_sum_idx(fake_state, real_state, batch, initial_type,all_state=all_state,use_all_type=use_all_type)


    match_loss, pos_loss, heading_loss, shape_loss, vel_loss = matching_loss(
        fake_state[fake_idx]/denom, real_state[real_idx]/denom
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
        col_loss=multi_circle_collision_loss_mem_efficient(fake_pos, torch.atan2(fake_heading[:,1],fake_heading[:,0]), fake_shape[:,0],fake_shape[:,1],batch )
    else:
        col_loss = torch.zeros_like(match_loss.mean())  #
    # [03:08<13:55,  3.71it/s, v_num=4e1b]

    return match_loss,pos_loss,heading_loss,shape_loss,vel_loss,(col_loss).expm1()*10

