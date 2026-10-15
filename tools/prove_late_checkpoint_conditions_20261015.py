"""Verify frozen conditioning before comparing genuine late EMA checkpoints."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_checkpoint_snapshots'


def main():
    gc.disable()
    import jinja2
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    torch.set_num_threads(1)
    rows = []
    for control, candidates in [('roadK8', ['roadK8_e111', 'roadK8_e119']),
                                ('N24match4_final', ['N24match4_e119'])]:
        reference = BASE / 'sampler_snapshots_20261015' / control
        a = torch.load(reference / 'model.ckpt', map_location='cpu', weights_only=False)
        keys = [k for k, v in a['state_dict'].items() if torch.is_tensor(v)
                and ('encoder.map_encoder.' in k or 'normal_' in k or 'statistics_' in k)]
        assert keys
        for name in candidates:
            folder = BASE / 'sampler_late_snapshots_20261015' / name
            assert (folder / 'config.yaml').read_bytes() == (reference / 'config.yaml').read_bytes()
            b = torch.load(folder / 'model.ckpt', map_location='cpu', weights_only=False)
            metadata = json.loads((folder / 'provenance.json').read_text())
            digest = hashlib.sha256((folder / 'model.ckpt').read_bytes()).hexdigest()
            assert digest == metadata['sha256']
            assert (b['epoch'], b['global_step']) == (metadata['epoch'], metadata['global_step'])
            changed = [k for k in keys if not torch.equal(a['state_dict'][k], b['state_dict'][k])]
            assert not changed, changed
            rows.append(dict(snapshot=name, control=control, epoch=b['epoch'], step=b['global_step'],
                             checkpoint_sha256=digest, fixed_tensors_checked=len(keys), changed=changed,
                             source_config_identical=True))
            del b
        del a
    output = BASE / 'sampler_late_snapshots_20261015/paired_conditions_verified.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(status='PASS', rows=rows,
        scope='Same frozen map encoder, normalization/statistics and source configuration within each family; '
              'trained denoiser and EMA intentionally differ by epoch.'), indent=2) + '\n')
    print('LATE_CONDITIONS_PASS', [(r['snapshot'], r['epoch'], r['fixed_tensors_checked']) for r in rows], flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
