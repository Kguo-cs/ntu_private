"""Keep the native SMART training loop while selecting a research edge policy."""
from src.smart.model.smart_gail import SMART_GAIL

class PeriodicSMART_GAIL(SMART_GAIL):
    def __init__(self, *args, experimental_edge_heading_max_harmonic=8, **kwargs):
        harmonic = experimental_edge_heading_max_harmonic
        if isinstance(harmonic,bool) or not isinstance(harmonic,int) or not 1<=harmonic<=32:
            raise ValueError('Experimental edge harmonic must be an integer in1..32')
        from src.smart.diffusion.denoiser import InitDenoiser
        from src.smart.diffusion.initial_diffusion import InitDiffusion
        from tools.runtime_periodic_edge_sweep_20261015 import install_training_policy
        original_denoiser, original_diffusion = InitDenoiser.__init__, InitDiffusion.__init__
        try:
            install_training_policy(harmonic)
            super().__init__(*args, **kwargs)
        finally:
            InitDenoiser.__init__, InitDiffusion.__init__ = original_denoiser, original_diffusion
        options = self.encoder.init_decoder.denoiser_options
        assert options['edge_heading_encoding']=='integer_harmonics'
        assert options['edge_heading_max_harmonic']==harmonic
