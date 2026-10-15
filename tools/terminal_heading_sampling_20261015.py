"""Process-local final angular extrapolation study; native midpoint elsewhere."""
from __future__ import annotations
import math
from types import MethodType
import torch
from src.smart.diffusion.midpoint_sampling import install_midpoint
from src.smart.utils import wrap_angle


def install_terminal_heading_gain(flow, gain=2.0):
    """Use theta_final + (gain-2)*wrap(theta_x0_mid-theta_x0_start).

    The two x0 estimates already exist in the last midpoint interval. This
    changes only its final heading, uses no additional model call or target
    statistics, and keeps conditioned heading fields clean. Gain2 is exact
    native-midpoint behavior. Other gains are inference heuristics.
    """
    if not math.isfinite(gain) or not 0.5 <= gain <= 3.5:
        raise ValueError('Terminal heading gain must be finite in[0.5,3.5]')
    if flow.heading_noise != 'circular':
        raise ValueError('Terminal angular extrapolation requires circular heading')
    if getattr(flow, 'use_refiner', False):
        raise ValueError('Terminal-only study requires the refiner to be disabled')
    install_midpoint(flow)
    flow._terminal_heading_gain = float(gain)
    if gain == 2.0:
        return
    midpoint_step = flow._sample_step

    @torch.no_grad()
    def step(self, latent, start, end, agent, feature, branch_mask=None):
        if float(end) != 0.0:
            return midpoint_step(latent, start, end, agent, feature, branch_mask=branch_mask)
        velocity = self._model_velocity
        predictions = []

        def capture(unused_self, *args, **kwargs):
            result = velocity(*args, **kwargs)
            x0 = result[1]
            predictions.append(torch.atan2(x0[:, 3], x0[:, 2]))
            return result

        self._model_velocity = MethodType(capture, self)
        try:
            result = midpoint_step(latent, start, end, agent, feature, branch_mask=branch_mask)
        finally:
            self._model_velocity = velocity
        if len(predictions) != 2:
            raise RuntimeError('Final midpoint interval must have exactly2 x0 predictions')
        state = result[0].clone()
        angle = torch.atan2(state[:, 3], state[:, 2])
        angle = wrap_angle(angle + (gain - 2.0) * wrap_angle(predictions[1] - predictions[0]))
        heading = torch.stack((angle.cos(), angle.sin()), -1)
        fixed = self._conditioned_state_mask(agent, state)[:, 2:4]
        clean = self._joint_clean(agent['expert_input'], agent)[:, 2:4]
        state[:, 2:4] = torch.where(fixed, clean, heading)
        return (state, *result[1:])

    flow._sample_step = MethodType(step, flow)
