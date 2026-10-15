"""CPU checks of terminal extrapolation through the actual native Flow step."""
from pathlib import Path
import gc
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sim_mpl_continuous')
gc.disable()
import jinja2
import torch
from tools.prove_midpoint_sampling_20261015 import NativeStepFlow
from tools.terminal_heading_sampling_20261015 import install_terminal_heading_gain
from src.smart.utils import wrap_angle


class QuadraticBiasFlow(NativeStepFlow):
    def _model_velocity(self, latent, time, agent, feature):
        self.calls += 1
        theta = torch.atan2(latent[:, 3], latent[:, 2])
        # Cross the +/-pi branch between t=.1 and t=.05.
        theta0 = wrap_angle(math.radians(179.) + math.radians(2000.) * time[:, 0].square())
        x0 = latent - time * latent
        x0[:, 2:4] = torch.stack((theta0.cos(), theta0.sin()), -1)
        velocity = (latent - x0) / time.clamp_min(self.t_eps)
        omega = wrap_angle(theta - theta0) / time[:, 0].clamp_min(self.t_eps)
        velocity[:, 2:4] = torch.stack((-theta.sin(), theta.cos()), -1) * omega[:, None]
        agent['_init_diffusion_type_logits'] = latent.new_tensor([[0., 1., 2.], [0., 1., 2.]])
        return velocity, x0


def run(gain, start=.1, end=0.):
    flow = QuadraticBiasFlow()
    flow.calls = 0
    install_terminal_heading_gain(flow, gain)
    state = torch.ones(2, 7, dtype=torch.float64)
    state[:, 2:4] = torch.tensor([1., 0.])
    source = state.new_tensor([[1., 0., 0.], [0., 0., 0.]])
    agent = dict(expert_input=state.clone(), _init_diffusion_type_source=source.clone(),
                 _init_diffusion_type_state=source.clone())
    state[0, :2] = 5.  # Native conditions must still overwrite noisy ego.
    result = flow._sample_step(state, state.new_tensor(start), state.new_tensor(end), agent, {})[0]
    return flow, result, agent


def main():
    _, baseline, baseline_agent = run(2.)
    rows = []
    for gain in (1., 1.5, 2., 2.5, 3.):
        flow, state, agent = run(gain)
        expected = wrap_angle(state.new_tensor(math.radians(199. - 15. * gain)))
        error = abs(float(wrap_angle(torch.atan2(state[1, 3], state[1, 2]) - expected)))
        row = dict(gain=gain, calls=flow.calls, analytical_angle_error=error,
                   unit_circle=bool(torch.allclose(state[:, 2:4].norm(dim=-1), state.new_ones(2))),
                   non_heading_exactly_unchanged=torch.equal(state[:, [0, 1, 4, 5, 6]], baseline[:, [0, 1, 4, 5, 6]]),
                   fixed_ego=torch.equal(state[0], agent['expert_input'][0]),
                   type_state_exactly_unchanged=torch.equal(agent['_init_diffusion_type_state'], baseline_agent['_init_diffusion_type_state']))
        assert error < 1e-12 and flow.calls == 2
        assert all(row[k] for k in ('unit_circle', 'non_heading_exactly_unchanged', 'fixed_ego', 'type_state_exactly_unchanged'))
        rows.append(row)
    # Non-final intervals must be unaffected for every gain.
    _, a, _ = run(2., .4, .2)
    _, b, _ = run(1.5, .4, .2)
    assert torch.equal(a, b)
    result = dict(status='PASS', rows=rows, non_final_intervals_exactly_unchanged=True,
                  scope='Native Flow._sample_step with analytic predictions; no real-model quality claim.')
    p = ROOT / 'output/reports/init_diffusion_continuous_20261014/terminal_heading_gain/proof.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
