"""CPU checks against an analytic ODE, including categorical state and ego."""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.smart.diffusion.midpoint_sampling import install_midpoint
from src.smart.diffusion.scale_flow import Flow


class LinearFlow:
    use_sde = False
    sampling_exact_x0 = False
    t_eps = .05
    generate_type = True
    joint_type = False
    heading_noise = 'circular'
    model = SimpleNamespace(m_delta_dim=7)

    def _conditioned_state_mask(self, agent, state):
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[0] = True
        return mask

    def _conditioned_type_mask(self, agent):
        return torch.tensor([True, False])

    def _joint_clean(self, clean, agent):
        return clean

    def _sample_step(self, state, start, end, agent, feature, branch_mask=None):
        self.calls += 1
        # dx/dt=x; exact solution across1->0 is exp(-1)*x(1).
        dt = end - start
        result = state + dt * state
        angle = torch.atan2(state[:, 3], state[:, 2]) + dt * .4
        result[:, 2:4] = torch.stack((angle.cos(), angle.sin()), -1)
        fixed = self._conditioned_state_mask(agent, result)
        result = torch.where(fixed, agent['expert_input'], result)
        types = agent['_init_diffusion_type_state']
        agent['_init_diffusion_type_state'] = types + dt * types
        return (result, result, state.new_full((2, 1), start),
                state.new_full((2, 1), end), state.new_zeros(2), state.new_zeros(state.shape))


def integrate(intervals):
    flow = LinearFlow()
    flow.calls = 0
    install_midpoint(flow)
    state = torch.ones(2, 7, dtype=torch.float64)
    state[:, 2] = 1.
    state[:, 3] = 0.
    source = torch.ones(2, 3, dtype=torch.float64)
    source[0] = torch.tensor([1., 0., 0.])
    agent = dict(expert_input=state.clone(), _init_diffusion_type_source=source.clone(),
                 _init_diffusion_type_state=source.clone())
    times = torch.linspace(1., 0., intervals + 1, dtype=torch.float64)
    for start, end in zip(times[:-1], times[1:]):
        state = flow._sample_step(state, start, end, agent, {})[0]
    return flow, state, agent


class NativeStepFlow(LinearFlow):
    training = False
    collision_guidance = None

    def _conditioned_agent_mask(self, agent):
        return torch.tensor([True, False])

    def _fix_conditioned_agents(self, clean, latent, time, agent):
        fixed = self._conditioned_state_mask(agent, latent)
        latent[fixed] = clean[fixed]
        time[0] = 0.

    def _model_velocity(self, latent, time, agent, feature):
        self.calls += 1
        velocity = latent.clone()
        angle = torch.atan2(latent[:, 3], latent[:, 2])
        velocity[:, 2:4] = .4 * torch.stack((-angle.sin(), angle.cos()), -1)
        agent['_init_diffusion_type_logits'] = latent.new_tensor([[0., 1., 2.], [0., 1., 2.]])
        return velocity, latent - time * velocity

    _sample_step = Flow._sample_step


def native_step_check():
    flow = NativeStepFlow()
    flow.calls = 0
    install_midpoint(flow)
    state = torch.ones(2, 7, dtype=torch.float64)
    state[:, 2:4] = torch.tensor([1., 0.])
    clean = state.clone()
    source = state.new_tensor([[1., 0., 0.], [0., 0., 0.]])
    agent = dict(expert_input=clean, _init_diffusion_type_source=source,
                 _init_diffusion_type_state=source.clone())
    # Native method mutates fixed input fields. Start ego from an incorrect value
    # to verify the wrapper uses/restores the true condition rather than noise.
    state[0, :2] = 5.
    times = torch.tensor([1., .7, .4, .2, .1, 0.], dtype=state.dtype)
    for start, end in zip(times[:-1], times[1:]):
        state = flow._sample_step(state, start, end, agent, {})[0]
    probability = agent['_init_diffusion_type_logits'][1].softmax(-1)
    return (flow.calls == 10 and torch.equal(state[0], clean[0])
            and torch.allclose(agent['_init_diffusion_type_state'][1], probability)
            and torch.equal(agent['_init_diffusion_type_state'][0], source[0])
            and torch.allclose(state[:, 2:4].norm(dim=-1), state.new_ones(2)))


def main():
    a, first, aa = integrate(5)
    b, second, bb = integrate(10)
    exact = math.exp(-1.)
    errors = [abs(float(x[1, 0]) - exact) for x in (first, second)]
    second_order = errors[0] / errors[1] > 3.8
    type_matches = torch.equal(second[1, 0].expand(3), bb['_init_diffusion_type_state'][1])
    fixed_ego = torch.equal(second[0], bb['expert_input'][0])
    fixed_type = torch.equal(bb['_init_diffusion_type_state'][0], bb['_init_diffusion_type_source'][0])
    circle = bool(torch.allclose(second[:, 2:4].norm(dim=-1), second.new_ones(2)))
    angle_correct = abs(float(torch.atan2(second[1, 3], second[1, 2])) + .4) < 1e-12
    native_step = bool(native_step_check())
    result = dict(analytic_ODE_errors=errors, observed_error_ratio=errors[0] / errors[1],
                  second_order=second_order, type_state_integrated=type_matches,
                  fixed_ego=fixed_ego, fixed_type=fixed_type, unit_circle=circle,
                  analytic_heading=angle_correct, calls=[a.calls, b.calls],
                  real_native_step_conditions_and_type=native_step,
                  status='PASS' if all([second_order, type_matches, fixed_ego,
                                        fixed_type, circle, angle_correct, a.calls == 10, b.calls == 20,
                                        native_step]) else 'FAIL',
                  scope='Analytic CPU ODE only; real model evaluation required.')
    p = ROOT / 'output/reports/init_diffusion_continuous_20261014/midpoint_prototype/proof.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    assert result['status'] == 'PASS', result


if __name__ == '__main__':
    main()
