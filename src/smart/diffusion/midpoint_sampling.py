"""Deterministic midpoint integration shared by native and opt-in samplers.

Each interval uses two denoiser calls. A terminal_time of .1 keeps the last
midpoint call at the native velocity floor .05. No training objective changes.
"""
from types import MethodType
import torch
from src.smart.utils import wrap_angle


def validate_midpoint(flow):
    if flow.use_sde or getattr(flow, 'sampling_exact_x0', False):
        raise ValueError('Midpoint requires deterministic native-floor sampling')


@torch.no_grad()
def midpoint_step(self, original, latent, time_scalar, next_time_scalar,
                  tokenized_agent, map_feature, branch_mask=None):
    if branch_mask is not None:
        raise ValueError('Midpoint does not support stochastic branches')
    middle_time = (time_scalar + next_time_scalar) * .5
    if float(middle_time) < self.t_eps - 1e-7:
        raise ValueError('Midpoint below native velocity floor; increase terminal_time')
    type_initial = None
    if self.generate_type and not self.joint_type:
        type_initial = tokenized_agent['_init_diffusion_type_state'].clone()
    first = original(latent, time_scalar, middle_time, tokenized_agent,
                     map_feature, branch_mask=None)
    predictor = first[0]
    type_middle = (tokenized_agent['_init_diffusion_type_state'].clone()
                   if type_initial is not None else None)
    second = original(predictor, middle_time, next_time_scalar,
                      tokenized_agent, map_feature, branch_mask=None)
    result = latent + 2. * (second[0] - predictor)
    if self.heading_noise == 'circular':
        initial_angle = torch.atan2(latent[:, 3], latent[:, 2])
        middle_angle = torch.atan2(predictor[:, 3], predictor[:, 2])
        end_angle = torch.atan2(second[0][:, 3], second[0][:, 2])
        angle = wrap_angle(initial_angle + 2. * wrap_angle(end_angle - middle_angle))
        result = torch.cat((result[:, :2], torch.stack((angle.cos(), angle.sin()), -1),
                            result[:, 4:]), dim=-1)
    fixed = self._conditioned_state_mask(tokenized_agent, result)
    clean = self._joint_clean(tokenized_agent['expert_input'], tokenized_agent)
    result = torch.where(fixed, clean, result)
    if type_initial is not None:
        state = type_initial + 2. * (tokenized_agent['_init_diffusion_type_state'] - type_middle)
        mask = self._conditioned_type_mask(tokenized_agent)
        state[mask] = tokenized_agent['_init_diffusion_type_source'][mask]
        tokenized_agent['_init_diffusion_type_state'] = state
    if self.joint_type:
        tokenized_agent['_init_diffusion_type_state'] = result[:, self.model.m_delta_dim:]
    return (result, second[1], first[2], second[3], second[4], second[5])


def install_midpoint(flow):
    """Opt in process-locally; native Flow can select the helper directly."""
    validate_midpoint(flow)
    if getattr(flow, '_midpoint_installed', False):
        raise ValueError('Midpoint is already installed')
    if getattr(flow, 'sampling_integrator', 'euler') != 'euler':
        raise ValueError('Native midpoint is already selected by configuration')
    original = flow._sample_step
    # Capture the function, not a bound source instance, so deepcopy uses its
    # own state/model instead of accidentally calling the original Flow.
    function = original.__func__

    @torch.no_grad()
    def midpoint(self, latent, start, end, agent, feature, branch_mask=None):
        return midpoint_step(self, function.__get__(self, type(self)), latent,
                             start, end, agent, feature, branch_mask)

    flow._sample_step = MethodType(midpoint, flow)
    flow._midpoint_installed = True
    return original
