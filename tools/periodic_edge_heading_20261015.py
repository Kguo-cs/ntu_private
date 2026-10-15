"""Opt-in research prototype for periodic denoiser edge heading features.

The current FourierEmbedding includes a raw wrapped angle and learns arbitrary
angle frequencies, so its representation jumps at +/-pi. This prototype keeps
the spatial encoders and parameter layout, using integer angular harmonics and
sin(theta) for the angle slot. It needs new training; it is not installed in any
production config or current evaluation process.
"""
from types import MethodType
import math
import torch
from src.smart.layers.fourier_embedding import FourierEmbedding


def periodic_forward(self, continuous_inputs=None, categorical_embs=None):
    if continuous_inputs is None or categorical_embs is not None:
        raise ValueError('Periodic edge prototype requires continuous-only relation features')
    if continuous_inputs.ndim != 2 or continuous_inputs.shape[-1] != 3:
        raise ValueError('Expected edge relations [E,local_x/local_y/relative_heading]')
    spatial = continuous_inputs[:, :2, None]
    spatial_phase = spatial * self.freqs.weight[None, :2] * (2 * math.pi)
    spatial_features = torch.cat((spatial_phase.cos(), spatial_phase.sin(), spatial), -1)
    angle = continuous_inputs[:, 2, None]
    angular_phase = angle * self._periodic_heading_harmonics.to(angle)
    angular_features = torch.cat((angular_phase.cos(), angular_phase.sin(), angle.sin()), -1)
    features = torch.cat((spatial_features, angular_features[:, None]), 1)
    outputs = [mlp(value) for mlp, value in zip(self.mlps, features.unbind(1))]
    return self.to_out(torch.stack(outputs).sum(0))


def install_periodic_heading(denoiser, max_harmonic=8):
    if isinstance(max_harmonic, bool) or not isinstance(max_harmonic, int) or not 1 <= max_harmonic <= 32:
        raise ValueError('max_harmonic must be an integer in1..32')
    edge = denoiser.edge_encoder
    names = ('r_a2a_emb', 'r_pt2a_emb')
    modules = [getattr(edge, name) for name in names]
    for module in modules:
        if not isinstance(module, FourierEmbedding) or module.share or module.input_dim != 3:
            raise ValueError('Prototype requires three-channel independent Fourier edge embeddings')
        if hasattr(module, '_periodic_heading_harmonics'):
            raise ValueError('Periodic heading is already installed')
    for module in modules:
        bands = module.freqs.weight.shape[-1]
        harmonics = (torch.arange(bands, device=module.freqs.weight.device) % max_harmonic + 1)
        module.register_buffer('_periodic_heading_harmonics', harmonics.to(module.freqs.weight), persistent=False)
        module.forward = MethodType(periodic_forward, module)
    denoiser._experimental_edge_heading = dict(kind='integer_harmonics', max_harmonic=max_harmonic,
                                               raw_angle_replacement='sin(theta)', training_required=True)
    return denoiser._experimental_edge_heading
