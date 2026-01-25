
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


def get_matching_loss(
    initial_type, batch, fake_state,real_state,latent=False
    ):
    # real_state = m_init * normal_scale + normal_mean
    # fake_state = x_pred * normal_scale + normal_mean

    fake_pos, fake_heading, fake_shape = fake_state[:, :2], fake_state[:, 2:4], fake_state[   :, 4:]
    real_pos, real_heading, real_shape = real_state[:, :2], real_state[:, 2:4], real_state[ :, 4:]

    rows, cols = [], []

    for b in batch.unique():
        for type in initial_type[batch == b].unique():
            f_idx = ((batch == b) & (initial_type == type)).nonzero(as_tuple=True)[0]

            if latent:
                dist = torch.norm(fake_state[f_idx][:,None]-real_state[f_idx][None],p=2,dim=-1).square()
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

        match_loss = torch.norm(fake_state[row] - real_state[col],p=2,dim=-1).square().mean()
    else:
        match_loss, pos_loss, heading_loss, shape_loss,vel_loss = matching_loss(
            fake_pos[row], fake_heading[row], fake_shape[row],
            real_pos[col], real_heading[col], real_shape[col]
        )

    # match_loss=((fake_state[row] - real_state[col]) ** 2)#.mean()

    return match_loss,pos_loss,heading_loss,shape_loss,vel_loss

