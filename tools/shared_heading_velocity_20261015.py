"""Research-only angular-velocity readout using the existing state head.

No production monkey patch is installed until install_training_policy is called.
The checkpoint options deliberately distinguish this readout from native x0 and
from the native separate angular-velocity head.
"""
import functools
import inspect
from types import MethodType


def shared_heading_velocity(self, features):
    return self.to_out_m_delta(features)[:, 3:4]


def install_training_policy():
    from src.smart.diffusion.denoiser import InitDenoiser
    from src.smart.diffusion.initial_diffusion import InitDiffusion

    denoiser_init = InitDenoiser.__init__
    diffusion_init = InitDiffusion.__init__
    signature = inspect.signature(denoiser_init)

    @functools.wraps(denoiser_init)
    def init_denoiser(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        requested = bound.arguments.get('heading_velocity', False)
        if requested:
            bound.arguments['heading_velocity'] = False
        denoiser_init(*bound.args, **bound.kwargs)
        if requested:
            self.heading_velocity = True
            self.to_out_heading_velocity = MethodType(shared_heading_velocity, self)

    @functools.wraps(diffusion_init)
    def init_diffusion(self, *args, **kwargs):
        diffusion_init(self, *args, **kwargs)
        if self.heading_objective == 'angular_velocity':
            self.denoiser_options['heading_velocity_readout'] = 'shared_state_row3'

    InitDenoiser.__init__ = init_denoiser
    InitDiffusion.__init__ = init_diffusion


def main():
    import gc
    import json
    from pathlib import Path
    import sys

    gc.disable()
    import jinja2  # Import before TensorFlow initializes native threads.
    install_training_policy()
    from tools import scratch_component_loss_sweep as sweep
    sweep.SEARCH_CONFIG_KEYS.update({
        'model.model_config.decoder.init_diffusion.' + key
        for key in ('heading_objective', 'heading_x0_loss', 'heading_flow_loss_weight')
    })
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    policy = output.parent / (output.name + '.shared_heading_velocity.json')
    policy.parent.mkdir(parents=True, exist_ok=True)
    if policy.exists():
        raise FileExistsError(policy)
    policy.write_text(json.dumps(dict(
        readout='shared_state_row3', parameter_count_unchanged=True,
        heading_objective='angular_velocity', new_training_required=True,
        native_checkpoint_option_guard=True,
        limitation='A new heading objective, not an established JSD improvement.'
    ), indent=2) + '\n')
    from tools.runtime_isolated_cache_recovery_20261015 import main as run
    run()


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
