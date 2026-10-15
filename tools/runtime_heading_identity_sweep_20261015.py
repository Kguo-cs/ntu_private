"""Initialization-only trial adapter; reuse the proven scratch training chain."""
from pathlib import Path
import functools
import gc
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    gc.disable()
    import jinja2
    from src.smart.diffusion.denoiser import InitDenoiser
    from tools.heading_identity_initialization_20261015 import initialize_heading_identity
    name = sys.argv[sys.argv.index('--variants') + 1]
    if name not in ('control', 'identity'):
        raise ValueError('This adapter requires the registered control/identity arm')
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    audit_path = output.parent / (output.name + '.heading_initialization.json')
    records = []
    if name == 'identity':
        original = InitDenoiser.__init__

        @functools.wraps(original)
        def init(self, *a, **kw):
            original(self, *a, **kw)
            records.append(initialize_heading_identity(self))
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(dict(arm=name, constructor_records=records,
                 scope='Two heading rows initialized to local identity before checkpoint restore; '
                       'other parameters unchanged. No inference or loss change.'), indent=2)+'\n')

        InitDenoiser.__init__ = init
    from tools.runtime_periodic_edge_sweep_20261015 import main as periodic_main
    periodic_main()


if __name__ == '__main__':
    main()
