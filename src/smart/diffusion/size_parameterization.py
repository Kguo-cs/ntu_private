"""Optional clean-size coordinates; inputs remain in their model representation."""
import torch


def size_time_residual_x0(prediction, latent, time):
    """Replace only length/width by s_t + t*r, preserving the x0 loss target.

    Linear state uses meters; log state uses log meters. Physical decoding stays
    in the existing model conversion, after this representation-space formula.
    """
    size = latent[:, 4:6] + time * prediction[:, 4:6]
    return torch.cat((prediction[:, :4], size, prediction[:, 6:]), dim=-1)
