"""Optional clean-size coordinates; inputs remain in their model representation."""
import torch


def size_time_residual_x0(prediction, latent, time):
    """Replace only length/width by s_t + t*r, preserving the x0 loss target.

    Linear state uses meters; log state uses log meters. Physical decoding stays
    in the existing model conversion, after this representation-space formula.
    """
    size = latent[:, 4:6] + time * prediction[:, 4:6]
    return torch.cat((prediction[:, :4], size, prediction[:, 6:]), dim=-1)


def size_train_affine_x0(prediction, mean, scale):
    """Decode standardized clean sizes using saved TRAIN representation stats.

    Only the output coordinates change: the existing x0 target, source,
    interpolation and physical/log conversion remain unchanged. Statistics are
    buffers established by the existing training normalizer, not current-batch
    or evaluation attributes. This option introduces no trainable parameters.
    """
    size = (mean[:, 4:6].detach().to(prediction)
            + scale[:, 4:6].detach().to(prediction) * prediction[:, 4:6])
    return torch.cat((prediction[:, :4], size, prediction[:, 6:]), dim=-1)
