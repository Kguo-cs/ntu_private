"""Inference-only projection onto the known shortest-arc velocity support.

Every training target is wrap(theta_noise-theta_data), hence in[-pi,pi].
Projection cannot increase squared error against any such target. This does
not establish that clipping improves sampled distributions; GPU comparison
against the unchanged model remains required.
"""
from pathlib import Path
from types import MethodType
import copy
import hashlib
import json
import math
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bounded_prediction_velocity(self, latent, time, prediction, type_logits=None):
    if self.training:
        raise RuntimeError('This exploratory projection is inference-only')
    index = self.model.feature_dim
    omega = prediction[:, index:index + 1]
    bounded = omega.clamp(-math.pi, math.pi)
    active = getattr(self, '_heading_bound_active', torch.ones_like(omega[:, 0], dtype=torch.bool))
    assert len(active) == len(omega)
    absolute = omega[:, 0].abs()
    self._heading_bound_audit.append(((absolute.gt(math.pi) & active).sum(), active.sum(),
                                      torch.where(active, absolute, 0.).max().detach()))
    modified = (torch.cat((prediction[:, :index], torch.where(active[:, None], bounded, omega),
                           prediction[:, index + 1:]), -1) if self._heading_bound_apply else prediction)
    return self._heading_bound_original(self, latent, time, modified, type_logits=type_logits)


def audit_denoise(self, latent, time, tokenized_agent, *args, **kwargs):
    self._heading_bound_active = (~self._conditioned_state_mask(tokenized_agent, latent)[:, 2]
                                  & (time[:, 0] > 0))
    return self._heading_bound_original_denoise(self, latent, time, tokenized_agent, *args, **kwargs)


def install_inference_bound(flow, *, apply=True):
    if flow.heading_objective != 'angular_velocity' or flow.heading_noise != 'circular':
        raise ValueError('Only native shortest-arc scalar angular velocity is supported')
    if hasattr(flow, '_heading_bound_original'):
        raise ValueError('Already installed')
    flow._heading_bound_original = flow._prediction_velocity.__func__
    flow._heading_bound_audit = []
    flow._heading_bound_apply = bool(apply)
    if hasattr(flow, '_denoise'):
        flow._heading_bound_original_denoise = flow._denoise.__func__
        flow._denoise = MethodType(audit_denoise, flow)
    flow._prediction_velocity = MethodType(bounded_prediction_velocity, flow)


def main():
    import torch
    from types import SimpleNamespace
    from src.smart.diffusion.scale_flow import Flow
    generator = torch.Generator().manual_seed(4817)
    target = (torch.rand(100000, generator=generator) * 2 - 1) * math.pi
    prediction = torch.randn(100000, generator=generator) * 8
    assert ((prediction.clamp(-math.pi, math.pi) - target).square()
            <= (prediction - target).square() + 1e-5).all()
    flow = SimpleNamespace(model=SimpleNamespace(feature_dim=7, m_delta_dim=7),
        heading_objective='angular_velocity', heading_noise='circular',
        joint_type=False, feature_type=False, training=False, t_eps=.05)
    flow._prediction_velocity = MethodType(Flow._prediction_velocity, flow)
    latent = torch.randn(5, 7, generator=generator)
    latent[:, 2:4] = torch.tensor([1., 0.])
    raw = torch.randn(5, 8, generator=generator)
    raw[:, 7] = torch.tensor([-8., -math.pi, 0., math.pi, 8.])
    time = torch.full((5, 1), .2)
    plain_velocity, plain_x0 = flow._prediction_velocity(latent, time, raw)
    rng = torch.get_rng_state().clone()
    install_inference_bound(flow)
    assert torch.equal(rng, torch.get_rng_state())
    velocity, x0 = flow._prediction_velocity(latent, time, raw)
    physical = [0, 1, 4, 5, 6]
    torch.testing.assert_close(velocity[:, physical], plain_velocity[:, physical], rtol=0, atol=0)
    torch.testing.assert_close(x0[:, physical], plain_x0[:, physical], rtol=0, atol=0)
    assert (velocity[:, 2:4].norm(dim=-1) <= math.pi + 1e-6).all()
    flow._heading_bound_apply = False
    audited_velocity, audited_x0 = flow._prediction_velocity(latent, time, raw)
    torch.testing.assert_close(audited_velocity, plain_velocity, rtol=0, atol=0)
    torch.testing.assert_close(audited_x0, plain_x0, rtol=0, atol=0)
    flow._heading_bound_apply = True
    flow._ego_field_fixed = MethodType(Flow._ego_field_fixed, flow)
    flow._conditioned_state_mask = MethodType(Flow._conditioned_state_mask, flow)
    flow._heading_bound_original_denoise = lambda owner, value, beta, agent: value
    agent = {'ego_mask': torch.tensor([True, False, False, False, False])}
    assert audit_denoise(flow, latent, time, agent) is latent
    assert torch.equal(flow._heading_bound_active, ~agent['ego_mask'])
    del flow._heading_bound_active
    other = copy.deepcopy(flow)
    torch.testing.assert_close(other._prediction_velocity(latent, time, raw)[0], velocity, rtol=0, atol=0)
    assert len(other._heading_bound_audit) == 3 and len(flow._heading_bound_audit) == 2
    flow.training = True
    try:
        flow._prediction_velocity(latent, time, raw)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Must reject accidental training use')
    result = dict(status='CPU_SUPPORT_AND_NATIVE_KINEMATICS_PROOF_PASS',
        projection_MSE_never_worse_for_100000_bounded_targets=True,
        native_single_step_non_heading_outputs_unchanged=True, heading_tangent_norm_bounded=True,
        no_RNG_consumption=True, deepcopy_owns_audit_state=True, accidental_training_rejected=True,
        audit_only_native_outputs_identical=True, native_probe_excludes_conditioned_heading_rows=True,
        quality_unproven=True, GPU_evaluation_not_run=True,
        proof_scope='One native prediction-to-velocity conversion; full rollout distributions can change in every field.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = ROOT / 'output/reports/init_diffusion_continuous_20261014/bounded_heading_velocity_cpu_proof_20261015.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
