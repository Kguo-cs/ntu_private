"""Independent, inference-only N20 power-grid prototype.

This module does not change production Flow, checkpoints, or training. It reuses
Flow._sample_step on an instance-local shallow copy that shares read-only model
weights. A future TEST-only adapter must install it only inside an isolated
native evaluation child, after strict checkpoint restore and EMA selection.
There is currently no registered GPU evaluation or performance claim.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOW_SOURCE = ROOT / 'src/smart/diffusion/scale_flow.py'
REVIEWED_FLOW_SHA256 = 'b0d60e3fecfc57b5a3f6c8234e5770ec9f0d30816401ca8df52187fbe2922fcf'
ALLOWED_POWERS = (0.5, 1.0, 2.0)
REVIEWED_RECIPE = {
    'heading_noise': 'circular', 'heading_objective': 'x0',
    'velocity_representation': 'speed', 'size_representation': 'linear',
    'generate_type': True, 'type_process': 'separate', 'type_noise_stats': 'standard',
    'joint_type': False, 'feature_type': False,
    'pos_source': 'gaussian', 'shape_source': 'gaussian', 'velocity_source': 'gaussian',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_time_grid(power, *, device, dtype, steps=20):
    """Exactly 20 positive evaluation times followed by the data endpoint 0.

    p=1 uses the legacy torch.linspace tensor without redundant arithmetic;
    this preserves its exact floating-point values for the parity control.
    """
    import torch
    require(type(steps) is int and steps == 20, 'This prototype requires exactly N20')
    require(not isinstance(power, bool) and math.isfinite(float(power))
            and float(power) in ALLOWED_POWERS, 'Reviewed powers are 0.5, 1, and 2')
    require(dtype == torch.float32, 'Require native 32-true state dtype')
    linear = torch.linspace(1., 0., steps + 1, device=device, dtype=dtype)
    grid = linear if float(power) == 1. else linear.pow(float(power))
    require(bool(torch.isfinite(grid).all()) and float(grid[0]) == 1.
            and float(grid[-1]) == 0. and bool((grid[:-1] > grid[1:]).all()),
            'Invalid or nonmonotone effective time grid')
    return grid


def validate_reviewed_flow(flow):
    """Reject model modes not covered by the actual native parity audit."""
    for name, expected in REVIEWED_RECIPE.items():
        actual = getattr(flow, name, None)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        require(matches, f'Unreviewed recipe: {name} must be {expected!r}')
    require(flow.model.m_delta_dim == 7, 'Reviewed scalar-speed state has 7 fields')
    require(flow.t_eps == .05, 'Reviewed baseline inference floor is .05')


def sample_with_power_grid(flow, tokenized_agent, map_feature, *, power=1., steps=20,
                           audit=None, denominator_policy):
    """Native initialization/final decoding for a reviewed N20 grid/guard arm.

    legacy_floor retains the exact original inference t_eps. The separate
    p=2 actual_time arm changes its copied scalar guard to float32 epsilon;
    positive-time physical/type velocities then divide by actual t. This is a
    joint integration change, whereas p=2 legacy_floor changes spacing only.
    Both arms preserve the original training Flow's t_eps and objective. The
    production circular angular update and fixed-field restoration are reused.
    SDE, RL/refiner sampling and training are deliberately unsupported.
    """
    import torch
    import torch.nn.functional as F
    validate_reviewed_flow(flow)
    from src.smart.diffusion.scale_flow import _noise_endpoint

    require(source_sha256(FLOW_SOURCE) == REVIEWED_FLOW_SHA256,
            'Production Flow source changed; prototype requires a fresh code review')
    require(not flow.training and not flow.model.training, 'Inference prototype requires eval mode')
    require(not flow.use_sde and not flow.use_refiner,
            'Prototype supports supervised ODE inference without SDE/refiner')
    require(flow.model.normal_scale.dtype == torch.float32, 'Require native 32-true state dtype')
    require(bool(torch.isfinite(flow.model.normal_mean).all())
            and bool(torch.isfinite(flow.model.normal_scale).all())
            and bool((flow.model.normal_scale > 0).all()),
            'Training normalizer must be finite with positive scale')
    require(bool(getattr(flow.model, '_normalizer_from_file', False))
            or bool((flow.model.normal_mean != 0).any()),
            'Training normalizer must be initialized before inference')
    grid = power_time_grid(power, device=tokenized_agent['batch'].device,
                           dtype=flow.model.normal_scale.dtype, steps=steps)
    require(denominator_policy in ('legacy_floor', 'actual_time'), 'Unknown denominator policy')
    require(denominator_policy != 'actual_time' or float(power) == 2.,
            'The separate actual_time guard ablation is reviewed only for p=2')
    proxy = copy.copy(flow)
    original_eps = flow.t_eps
    if denominator_policy == 'actual_time':
        proxy.t_eps = torch.finfo(grid.dtype).eps
    model_call_times = []

    def observe_forward(module, args):
        time = args[1]
        model_call_times.append({'minimum': float(time.min()), 'maximum': float(time.max()),
                                 'input_rows': len(args[0])})

    # Hook is temporary within this isolated inference invocation, not a method
    # replacement. It observes real denoiser forwards rather than loop counters.
    handle = flow.model.register_forward_pre_hook(observe_forward)
    try:
        with torch.no_grad():
            tokenized_agent.pop('_init_diffusion_ego_metrics', None)
            tokenized_agent.pop('_init_diffusion_component_metrics', None)
            agent_batch = tokenized_agent['batch'].long()
            num_agents = agent_batch.numel()
            proxy._require_type_normalizer()
            if getattr(proxy, 'use_ego_embedding', False):
                proxy.model._ego_role_mask(tokenized_agent)
            if proxy.size_representation == 'log':
                expert_input, _ = proxy.model.get_input(tokenized_agent)
                tokenized_agent['expert_input'] = expert_input
                tokenized_agent['_init_diffusion_size_representation'] = 'log'
            if proxy.joint_type:
                physical = torch.empty(num_agents, proxy.model.m_delta_dim,
                                       device=agent_batch.device, dtype=proxy.model.normal_scale.dtype)
                latent = proxy._draw_joint_source(physical)
            else:
                latent = torch.randn(num_agents, proxy.model.m_delta_dim,
                                     device=agent_batch.device, dtype=proxy.model.normal_scale.dtype)
                latent = _noise_endpoint(proxy.model, latent, proxy.sigma_h, proxy.heading_noise,
                                         pos_source=proxy.pos_source, shape_source=proxy.shape_source,
                                         velocity_source=proxy.velocity_source)
            tokenized_agent['gen_noise'] = latent[:, :proxy.model.m_delta_dim].clone()
            if proxy.velocity_representation == 'speed' or 'expert_input' not in tokenized_agent:
                expert_input, _ = proxy.model.get_input(tokenized_agent)
                tokenized_agent['expert_input'] = expert_input
            if getattr(proxy, 'generate_type', False):
                for key in ('_init_diffusion_type_logits', '_init_diffusion_type_prediction',
                            '_init_diffusion_type_metrics', '_init_diffusion_generated_type'):
                    tokenized_agent.pop(key, None)
                type_source = (latent[:, proxy.model.m_delta_dim:].clone() if proxy.joint_type else
                               proxy._draw_type_source(torch.randn(num_agents, 3, device=latent.device,
                                                                    dtype=latent.dtype)))
                fixed_type = proxy._conditioned_type_mask(tokenized_agent)
                if fixed_type.any():
                    type_source[fixed_type] = F.one_hot(tokenized_agent['type'][fixed_type].long(), 3).to(type_source)
                tokenized_agent['_init_diffusion_type_source'] = type_source
                tokenized_agent['_init_diffusion_type_state'] = type_source.clone()
                if proxy.joint_type:
                    latent[:, proxy.model.m_delta_dim:] = type_source

            for step in range(steps):
                latent, _, _, _, _, _ = proxy._sample_step(
                    latent, grid[step], grid[step + 1], tokenized_agent, map_feature,
                    branch_mask=None)
                if step == 0 and 'noise_feat_cur' in tokenized_agent:
                    tokenized_agent['noise_feat'] = tokenized_agent['noise_feat_cur'][:, None]

            fixed = proxy._conditioned_state_mask(tokenized_agent, latent)
            clean = proxy._joint_clean(tokenized_agent['expert_input'], tokenized_agent)
            latent = torch.where(fixed, clean, latent)
            if getattr(proxy, 'generate_type', False):
                types = (latent[:, proxy.model.m_delta_dim:].argmax(-1) if proxy.feature_type else
                         tokenized_agent['_init_diffusion_type_logits'].argmax(-1))
                fixed_type = proxy._conditioned_type_mask(tokenized_agent)
                types[fixed_type] = tokenized_agent['_init_diffusion_type_source'][fixed_type].argmax(-1)
                tokenized_agent['_init_diffusion_generated_type'] = types
                tokenized_agent['type'] = types
            if proxy.joint_type:
                latent = latent[:, :proxy.model.m_delta_dim]
            tokenized_agent['gen_z'] = latent
    finally:
        handle.remove()
    require(flow.t_eps == original_eps, 'Original training Flow epsilon changed')
    require(len(model_call_times) == 20, 'Actual denoiser NFE must be exactly 20')
    if audit is not None:
        audit.update(power=float(power), solver='Explicit Euler; native circular retraction',
                     requested_NFE=20, actual_denoiser_forward_calls=len(model_call_times),
                     effective_grid=grid.cpu().tolist(), effective_dt=(grid[1:] - grid[:-1]).cpu().tolist(),
                     model_call_times=model_call_times, last_positive_time=float(grid[-2]),
                     denominator_policy=denominator_policy,
                     integration_change='spacing plus denominator guard' if denominator_policy == 'actual_time' else 'spacing only',
                     positive_time_denominator='max(t, original .05 floor)' if denominator_policy == 'legacy_floor' else 'actual positive t, guarded by float32 epsilon',
                     actual_guard_change=proxy.t_eps != original_eps,
                     base_floor_active_positive_times=grid[:-1][grid[:-1] < original_eps].cpu().tolist(),
                     heading_denominator='unchanged native actual t, guarded by dtype epsilon',
                     original_flow_t_eps=float(original_eps), proxy_t_eps=float(proxy.t_eps),
                     original_flow_t_eps_unchanged=flow.t_eps == original_eps,
                     extra_trainable_parameters=0, objective_changed=False,
                     reviewed_recipe=dict(REVIEWED_RECIPE),
                     production_source_sha256=REVIEWED_FLOW_SHA256,
                     prototype_source_sha256=source_sha256(__file__),
                     all_output_values_finite=bool(torch.isfinite(latent).all()))
    return latent
