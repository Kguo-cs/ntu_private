import torch
import math

def central_diff(t: torch.Tensor, pad_value: float) -> torch.Tensor:
    """Computes the central difference along the last axis with padding."""
    # Compute central difference: [f(x+h) - f(x-h)] / 2h
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    # Pad with pad_value on both ends to retain shape
    pad_shape = list(t.shape[:-1]) + [1]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)

def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi] range."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def kinematic_likelihood(pos: torch.Tensor, heading: torch.Tensor) -> dict:
    seconds_per_step = 10

    # Linear velocity
    dpos = central_diff(pos, pad_value=float('nan'))  # shape [3, steps]
    linear_speed = torch.linalg.norm(dpos, ord=2, dim=0) / seconds_per_step  # shape [steps]

    # Linear acceleration
    linear_accel = central_diff(linear_speed, pad_value=float('nan')) / seconds_per_step  # shape [steps]

    # Angular velocity and acceleration
    dh_step = wrap_angle(central_diff(heading, pad_value=float('nan')) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = wrap_angle(central_diff(dh_step, pad_value=float('nan')) * 2) / 2
    d2h = d2h_step / (seconds_per_step ** 2)

    return {
        "linear_speed": linear_speed,
        "linear_accel": linear_accel,
        "angular_speed": dh,
        "angular_accel": d2h
    }

# linear_speed: {
#   histogram: {
#     min_val: 0.0
#     max_val: 25.0
#     num_bins: 10
#     additive_smoothing_pseudocount: 0.1
#   }
#   independent_timesteps: true
#   metametric_weight: 0.05
# }
#
# linear_acceleration: {
#   histogram: {
#     min_val: -12.0
#     max_val: 12.0
#     num_bins: 11
#     additive_smoothing_pseudocount: 0.1
#   }
#   independent_timesteps: true
#   metametric_weight: 0.05
# }
#
# angular_speed: {
#   histogram: {
#     min_val: -0.628
#     max_val: 0.628
#     num_bins: 11
#     additive_smoothing_pseudocount: 0.1
#   }
#   independent_timesteps: true
#   metametric_weight: 0.05
# }
#
# angular_acceleration: {
#   histogram: {
#     min_val: -3.14
#     max_val: 3.14
#     num_bins: 11
#     additive_smoothing_pseudocount: 0.1
#   }
#   independent_timesteps: true
#   metametric_weight: 0.05
# }
