"""Select the proven shared-omega readout without changing SMART's train loop."""
from src.smart.model.smart_gail import SMART_GAIL


class CircularVelocitySMART_GAIL(SMART_GAIL):
    def __init__(self, *args, **kwargs):
        from src.smart.diffusion.denoiser import InitDenoiser
        from src.smart.diffusion.initial_diffusion import InitDiffusion
        from tools.shared_heading_velocity_20261015 import install_training_policy
        original_denoiser,original_diffusion=InitDenoiser.__init__,InitDiffusion.__init__
        try:
            install_training_policy()
            super().__init__(*args, **kwargs)
        finally:
            InitDenoiser.__init__,InitDiffusion.__init__=original_denoiser,original_diffusion
        decoder=self.encoder.init_decoder
        assert decoder.heading_objective=='angular_velocity'
        assert decoder.denoiser_options['heading_velocity_readout']=='shared_state_row3'
