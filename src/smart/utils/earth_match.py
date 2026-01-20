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
    w_pos=0.1, w_heading=0.5, w_shape=0.2
):
    # Position: L1 or L2

    dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)

    pos_loss = dist.mean()

    # Heading: periodic-safe loss
    # heading_diff = torch.atan2(
    #     torch.sin(fake_heading - real_heading),
    #     torch.cos(fake_heading - real_heading)
    # )
    heading_diff=wrap_angle(fake_heading - real_heading)
    heading_loss = heading_diff.abs().mean()

    # Shape: L1
    shape_loss = F.l1_loss(fake_shape, real_shape)

    total_loss = (
        w_pos * pos_loss +
        w_heading * heading_loss +
        w_shape * shape_loss
    )

    return total_loss,pos_loss,heading_loss,shape_loss


def get_matching_loss(
    initial_type, batch, fake_pos,fake_heading,fake_shape,
    real_pos,real_heading,real_shape
    ):
    rows, cols = [], []

    for b in batch.unique():
        for type in initial_type[batch == b].unique():
            f_idx = ((batch == b) & (initial_type == type)).nonzero(as_tuple=True)[0]

            dist = torch.cdist(fake_pos[f_idx], real_pos[f_idx])

            cost = dist.cpu().detach().numpy()

            row, col = linear_sum_assignment(cost)

            rows.append(f_idx[row])
            cols.append(f_idx[col])

    row = torch.cat(rows)
    col = torch.cat(cols)

    match_loss, pos_loss, heading_loss, shape_loss = matching_loss(
        fake_pos[row], fake_heading[row], fake_shape[row],
        real_pos[col], real_heading[col], real_shape[col]
    )

    return match_loss,pos_loss,heading_loss,shape_loss

