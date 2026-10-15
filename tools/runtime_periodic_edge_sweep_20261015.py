"""Run the existing scratch trainer with a guarded periodic edge prototype."""
from pathlib import Path
import argparse
import functools
import gc
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_training_policy(max_harmonic):
    from src.smart.diffusion.denoiser import InitDenoiser
    from src.smart.diffusion.initial_diffusion import InitDiffusion
    from tools.periodic_edge_heading_20261015 import install_periodic_heading
    denoiser_init, diffusion_init = InitDenoiser.__init__, InitDiffusion.__init__

    @functools.wraps(denoiser_init)
    def init_denoiser(self, *args, **kwargs):
        denoiser_init(self, *args, **kwargs)
        if max_harmonic:
            install_periodic_heading(self, max_harmonic)

    @functools.wraps(diffusion_init)
    def init_diffusion(self, *args, **kwargs):
        diffusion_init(self, *args, **kwargs)
        if max_harmonic:
            assert self.G1.model._experimental_edge_heading['max_harmonic'] == max_harmonic
            # The native loader compares this full dictionary. A checkpoint
            # cannot silently load in ordinary wrapped-angle inference, or
            # with a different harmonic limit, even though tensor shapes match.
            self.denoiser_options.update(edge_heading_encoding='integer_harmonics',
                                         edge_heading_max_harmonic=max_harmonic)

    InitDenoiser.__init__, InitDiffusion.__init__ = init_denoiser, init_diffusion


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--periodic-edge-harmonic', type=int, required=True)
    options, remaining = parser.parse_known_args()
    if not 0 <= options.periodic_edge_harmonic <= 32:
        raise ValueError('Use0 for the ordinary control, or1..32 integer harmonics')
    gc.disable()
    gc.collect = lambda *unused_args, **unused_kwargs: 0
    import jinja2
    install_training_policy(options.periodic_edge_harmonic)
    from tools import overnight_init_diffusion_sweep as sweep_adapter
    sweep_adapter.sweep.SEARCH_CONFIG_KEYS.add('model.model_config.decoder.init_diffusion.sampling_integrator')
    output = Path(remaining[remaining.index('--output') + 1]).resolve()
    policy_path = output.parent / (output.name + '.periodic_runtime.json')
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    assert not policy_path.exists()
    source_files = [Path(__file__), ROOT / 'tools/periodic_edge_heading_20261015.py']
    policy = dict(kind='legacy' if options.periodic_edge_harmonic == 0 else 'integer_harmonics',
                  max_harmonic=options.periodic_edge_harmonic,
                  new_training_required=bool(options.periodic_edge_harmonic), parameter_count_unchanged=True,
                  map_encoder_unmodified=True, native_checkpoint_option_guard=True,
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_files})
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    print('PERIODIC_TRAINING_RUNTIME ' + json.dumps(policy), flush=True)
    sys.argv = [sys.argv[0]] + remaining
    from tools.runtime_cached_torch_loader_sweep import main as cached_main
    cached_main()


if __name__ == '__main__':
    main()
