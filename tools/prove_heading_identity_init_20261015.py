"""Verify actual denoiser identity heading and compatible checkpoint layout."""
from pathlib import Path
from types import SimpleNamespace
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
    from src.smart.diffusion.scale_flow import Flow
    from tools.heading_identity_initialization_20261015 import initialize_heading_identity
    torch.manual_seed(817)
    args = SimpleNamespace(input_dim=8, hidden_dim=32, num_heads=2, dropout=0.,
                           num_denoiser_layers=1, heading_noise='circular')
    processor = SimpleNamespace(use_refiner=False, learn_init=True, init_map_range=50.)
    flow = Flow(args, processor, False)
    control = copy.deepcopy(flow.model)
    original = {k: v.clone() for k, v in flow.model.state_dict().items() if torch.is_tensor(v)}
    rng = torch.get_rng_state().clone()
    record = initialize_heading_identity(flow.model)
    assert torch.equal(rng, torch.get_rng_state())
    modified = [k for k, v in original.items() if not torch.equal(v, flow.model.state_dict()[k])]
    assert len(modified) == 2 and all(k.startswith('to_out_m_delta.') for k in modified)
    features = torch.randn(3, 32)
    raw = flow.model.to_out_m_delta(features)
    assert torch.equal(raw[:, 2:4], torch.tensor([[1., 0.]]).expand(3, -1))
    pos = torch.tensor([[2., 3.], [-4., 7.], [1., -5.]])
    head = torch.tensor([torch.pi-.01, -torch.pi+.01, 1.3])
    decoded = flow.model.output_transform(raw, pos, head)
    torch.testing.assert_close(decoded[:, 2:4], torch.stack((head.cos(), head.sin()), -1))
    control.load_state_dict(flow.model.state_dict(), strict=True)
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in flow.model.state_dict().items()
               if torch.is_tensor(v))
    output = ROOT / 'output/reports/init_diffusion_continuous_20261014/heading_identity_init_proof.json'
    output.write_text(json.dumps(dict(status='PASS', **record, changed_tensor_keys=modified,
        constructor_rng_unchanged=True, ordinary_checkpoint_strict_load=True,
        initial_heading_equals_noisy_heading=True,
        limitation='Initialization/compatibility proof only; no training or JSD improvement claimed.'), indent=2)+'\n')
    print('HEADING_IDENTITY_INIT_PASS ' + json.dumps(record), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
