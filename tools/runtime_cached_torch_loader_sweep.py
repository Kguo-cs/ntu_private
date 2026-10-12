"""Isolated overnight sweep entry point with a public cached pickle_module.

Accepts the original overnight sweep CLI unchanged. Requires explicit cyclic GC
deferral and --young-gc-interval 0; successful jobs flush then hard-exit. Model,
data, tensor contents, training budget and objective remain in the original runner.
"""
from __future__ import annotations

import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_cached_loader(torch_module):
    """Only explicit trusted weights_only=False loads receive the public proxy.

    Caller-supplied pickle modules and safe weights-only loaders retain their
    original behavior. A fresh unpickler instance/persistent_load is still used
    for each file; only installed Torch's verified class definition is reused.
    """
    from tools.torch_pickle_class_cache import pickle_module_proxy
    original_load = torch_module.load
    proxy = pickle_module_proxy()

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        # torch.load's third positional parameter is pickle_module. Preserve it.
        if kwargs.get('weights_only') is False and len(args) < 3 and kwargs.get('pickle_module') is None:
            kwargs['pickle_module'] = proxy
        return original_load(*args, **kwargs)

    torch_module.load = load
    return original_load


def runtime_policy():
    from tools.torch_pickle_class_cache import cache_stats
    files = [Path(__file__).resolve(), ROOT / 'tools/torch_pickle_class_cache.py']
    return {'policy': 'public_pickle_module_class_cache_and_full_cyclic_gc_deferral',
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'automatic_gc_enabled': gc.isenabled(), 'manual_gc_collect': 'disabled',
            'young_gc_interval': 0, 'class_cache': cache_stats(),
            'tensor_and_data_conversion': False, 'objective_or_model_change': False,
            'only_explicit_weights_only_false_loads': True,
            'custom_pickle_module_preserved': True, 'success_exit': 'flush_then_os._exit(0)'}


def main():
    deferred = os.environ.get('SIM_SWEEP_DEFER_CYCLIC_GC') == '1' or '--defer-cyclic-gc' in sys.argv
    try:
        interval = int(sys.argv[sys.argv.index('--young-gc-interval') + 1])
    except (ValueError, IndexError):
        raise SystemExit('Cached-loader entry point requires explicit --young-gc-interval 0')
    if not deferred or interval != 0:
        raise SystemExit('Cached-loader entry point requires GC deferral and --young-gc-interval 0')
    gc.disable()
    gc.collect = lambda *unused_args, **unused_kwargs: 0
    # Set before any Torch/TF/Lightning imports so objects cannot be collected
    # automatically while these native extensions initialize.
    os.environ['SIM_SWEEP_DEFER_CYCLIC_GC'] = '1'
    import torch
    install_cached_loader(torch)
    from tools import overnight_init_diffusion_sweep
    print('RUNTIME_LOADER ' + json.dumps(runtime_policy(), sort_keys=True), flush=True)
    overnight_init_diffusion_sweep.sweep.main()
    # Successful return is not inferred from a partially written summary. The
    # supervising recovery driver separately checks DONE + complete actual cells.
    if '--output' in sys.argv:
        output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
        (output / 'runtime_loader_policy.json').write_text(json.dumps(runtime_policy(), indent=2) + '\n')
    print('RUNTIME_LOADER_SUCCESS ' + json.dumps(runtime_policy(), sort_keys=True), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
