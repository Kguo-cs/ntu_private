"""Real encoder proof: continuity, periodicity, gradients and unchanged weights."""
from types import SimpleNamespace
from pathlib import Path
import copy
import gc
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    gc.disable()
    import jinja2
    import torch
    from src.smart.modules.edge_encoder import EdgeEncoder
    from tools.periodic_edge_heading_20261015 import install_periodic_heading
    torch.manual_seed(817)
    edge = EdgeEncoder(hidden_dim=32, num_freq_bands=16, use_a2a=True, use_pl2a=True).double()
    control = copy.deepcopy(edge)
    before = {k: v.clone() for k, v in edge.state_dict().items()}
    rng = torch.get_rng_state().clone()
    policy = install_periodic_heading(SimpleNamespace(edge_encoder=edge), 8)
    assert torch.equal(rng, torch.get_rng_state())
    assert set(edge.state_dict()) == set(before)
    assert all(torch.equal(value, edge.state_dict()[key]) for key, value in before.items())
    input = torch.tensor([[2., -1., -.7], [3., 4., torch.pi-1e-8]], dtype=torch.double, requires_grad=True)
    shifted = input.detach().clone(); shifted[:, 2] += 2 * torch.pi
    eps = input.detach().clone(); eps[1, 2] = -torch.pi+1e-8
    rows = []
    for name in ('r_a2a_emb', 'r_pt2a_emb'):
        module = getattr(edge, name)
        a, b, c = module(input), module(shifted), module(eps)
        periodic_error = (a-b).abs().max().item()
        boundary_jump = (a[1]-c[1]).norm().item()
        legacy_jump = (getattr(control, name)(input)[1]-getattr(control, name)(eps)[1]).norm().item()
        assert periodic_error < 1e-10 and boundary_jump < 1e-5
        a.square().sum().backward(retain_graph=True)
        assert torch.isfinite(input.grad).all() and torch.isfinite(module.freqs.weight.grad).all()
        assert module.freqs.weight.grad[2].abs().max().item() == 0
        copied = copy.deepcopy(module)
        assert torch.equal(copied(input), module(input))
        with torch.no_grad():
            module.to_out[-1].bias.add_(2)
        assert not torch.equal(copied(input), module(input))
        rows.append(dict(name=name, periodic_error=periodic_error, boundary_jump=boundary_jump,
                         legacy_boundary_jump=legacy_jump, finite_gradients=True, deepcopy_uses_own_parameters=True))
    out = ROOT / 'output/reports/init_diffusion_continuous_20261014/periodic_edge_heading_proof.json'
    out.write_text(json.dumps(dict(status='PASS', policy=policy, rows=rows, no_added_parameters=True,
        initialization_rng_unchanged=True, parameter_layout_unchanged=True,
        limitation='Structural CPU proof only. No learning or JSD improvement claimed; new training required.'), indent=2)+'\n')
    print('PERIODIC_EDGE_PROOF_PASS ' + json.dumps(rows), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
