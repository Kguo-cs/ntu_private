
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

def matching_loss(
    fake_pos, fake_heading, fake_shape,
    real_pos, real_heading, real_shape,
    w_pos=0.1, w_heading=0.5, w_shape=0.2,w_vel=0.2
):
    # Position: L1 or L2

    # dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)
    #
    # pos_loss = dist.mean()

    pos_loss=F.l1_loss(fake_pos, real_pos)

    # Heading: periodic-safe loss
    # heading_diff = torch.atan2(
    #     torch.sin(fake_heading - real_heading),
    #     torch.cos(fake_heading - real_heading)
    # )
    # heading_diff=wrap_angle(fake_heading - real_heading)
    # heading_loss = heading_diff.abs().mean()

    heading_loss= F.l1_loss(fake_heading, real_heading)

    # Shape: L1
    shape_loss = F.l1_loss(fake_shape[:,:2], real_shape[:,:2])

    vel_loss = F.l1_loss(fake_shape[:,2:], real_shape[:,2:])

    total_loss = (
        w_pos * pos_loss +
        w_heading * heading_loss +
        w_shape * shape_loss+
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


def get_matching_loss(
    initial_type, batch, fake_state,real_state,latent=False
    ):
    fake_pos, fake_heading, fake_shape = fake_state[:, :2], fake_state[:, 2:4], fake_state[   :, 4:]
    real_pos, real_heading, real_shape = real_state[:, :2], real_state[:, 2:4], real_state[ :, 4:]

    rows, cols = [], []

    for b in batch.unique():
        for type in initial_type[batch == b].unique():
            f_idx = ((batch == b) & (initial_type == type)).nonzero(as_tuple=True)[0]

            if latent:
                dist = torch.norm(fake_state[f_idx][:,None]-real_state[f_idx][None],p=1,dim=-1)#.square()
            else:
                dist = torch.cdist(fake_pos[f_idx], real_pos[f_idx])

            cost = dist.cpu().detach().numpy()

            row, col = linear_sum_assignment(cost)

            rows.append(f_idx[row])
            cols.append(f_idx[col])

    row = torch.cat(rows)
    col = torch.cat(cols)

    if latent:
        match_loss = pos_loss = heading_loss = shape_loss = vel_loss = torch.tensor(0.0,
                                                                                    device=fake_state.device)

        match_loss = torch.norm(fake_state[row] - real_state[col],p=1,dim=-1).mean()#.square()
    else:
        match_loss, pos_loss, heading_loss, shape_loss,vel_loss = matching_loss(
            fake_pos[row], fake_heading[row], fake_shape[row],
            real_pos[col], real_heading[col], real_shape[col]
        )

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
    # col_loss = torch.relu(penetration)[mask].mean()

    col_loss=collision_loss(fake_pos, fake_heading, fake_shape,batch )

    return match_loss,pos_loss,heading_loss,shape_loss,vel_loss,col_loss

