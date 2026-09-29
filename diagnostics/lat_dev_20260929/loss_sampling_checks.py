"""Small CPU-only checks of current lateral-loss and sampling mechanics.

Run: /home/ke/miniconda3/envs/sim/bin/python /tmp/sim_lat_audit/loss_sampling_checks.py
The analytic mixture example is explanatory; it is not a measured model result.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, ndtri
import torch

sys.path.insert(0, '/home/ke/code/sim')
from src.smart.diffusion.diffusion_utils import get_diff_loss
from src.smart.utils.rollout import transform_to_global, transform_to_local


def main():
    pred = torch.ones(1, 8, requires_grad=True)
    target = torch.zeros_like(pred)
    meta = {'batch': torch.zeros(1, dtype=torch.long)}
    times = torch.full((1, 1), .5)
    loss = get_diff_loss(meta, pred, target, times, .05,
                         scale=torch.ones(1, 8), x_pred=True)[0].mean()
    loss.backward()
    different_scale_loss = get_diff_loss(meta, pred.detach(), target, times, .05,
                                         scale=torch.full((1, 8), 100.),
                                         x_pred=True)[0].mean()
    gradients = pred.grad[0].tolist()

    eps = .05
    total_weight = 1 + math.log(1 / eps)
    weighting = {str(cut): (1 + math.log(cut / eps)) / total_weight
                 for cut in (.05, .1, .2, .5)}
    endpoints = []
    for steps in (10, 20, 40, 64):
        last_t = 1 / steps
        prediction_fraction = last_t / max(last_t, eps)
        endpoints.append({'steps': steps, 'last_model_time': last_t,
                          'final_pred_x0_fraction': prediction_fraction,
                          'final_previous_latent_fraction': 1 - prediction_fraction})

    torch.manual_seed(817)
    local_pos = torch.randn(64, 2)
    local_head = torch.randn(64)
    anchor = torch.randn(64, 2)
    angle = torch.randn(64)
    global_pos, global_head = transform_to_global(local_pos, local_head, anchor, angle)
    recovered_pos, recovered_head = transform_to_local(global_pos, global_head, anchor, angle)
    angle_error = torch.atan2((recovered_head - local_head).sin(),
                             (recovered_head - local_head).cos()).abs().max()

    # Flow matching needs the conditional mean velocity. A deterministic L1
    # x0 loss instead learns component-wise conditional medians. This exact
    # two-mode example demonstrates that the generated mode proportions can
    # differ even with an oracle L1 minimizer and fine integration.
    n = 10000
    source = ndtri((np.arange(n) + .5) / n)
    a, positive_prior = 2., .2
    mixture = {}
    for objective in ('L2_conditional_mean', 'L1_conditional_median'):
        y = source.copy()
        for t in np.linspace(1., .001, 1000):
            posterior = expit(math.log(positive_prior / (1 - positive_prior))
                              + 2 * a * (1 - t) * y / (t * t))
            x0 = a * (2 * posterior - 1) if objective.startswith('L2') else np.where(posterior > .5, a, -a)
            y += -.001 * (y - x0) / t
        mixture[objective] = {'positive_mode_fraction': float(np.mean(y > 0)),
                              'mean_distance_to_nearest_mode': float(np.abs(np.abs(y) - a).mean())}

    result = {
        'output_coordinate_gradients_at_t_0_5': gradients,
        'velocity_to_position_output_gradient_ratio': gradients[6] / gradients[0],
        'loss_scale_1': float(loss),
        'loss_scale_100': float(different_scale_loss),
        'time_output_gradient_weight_fractions_below_cutoff': weighting,
        'time_weight_caveat': 'These are output-gradient weights for equally active L1 residuals, not measured parameter-gradient or loss fractions.',
        'current_euler_endpoint_formula': endpoints,
        'coordinate_roundtrip_max_position_error': float((recovered_pos-local_pos).abs().max()),
        'coordinate_roundtrip_max_angle_error': float(angle_error),
        'analytic_mixture_target_positive_fraction': positive_prior,
        'analytic_mixture_results': mixture,
        'analytic_mixture_caveat': 'Toy distribution only. Demonstrates lack of distributional correctness for L1 flow; does not prove the cause of this model lateral-deviation plateau.',
    }
    destination = Path('/tmp/sim_lat_audit/loss_sampling_checks.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
