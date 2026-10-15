"""Research-only nonnegative Gaussian speed source, with no learned parameters."""
import torch
from torch import nn
from tools.shared_heading_full.model import CircularVelocitySMART_GAIL

class RectifiedGaussianSpeedSource(nn.Module):
    speed=True
    shape=False
    def get_extra_state(self):
        return dict(kind='physical_rectified_gaussian_speed',speed=True,shape=False)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('Checkpoint speed source differs from rectified Gaussian')
    def forward(self,standard_noise,physical_noise):
        if standard_noise.shape!=physical_noise.shape or physical_noise.ndim!=2 or physical_noise.shape[1]!=7:
            raise ValueError('Rectified speed requires a7D scalar-speed state')
        result=physical_noise.clone()
        result[:,6]=physical_noise[:,6].clamp_min(0.)
        return result

class RectifiedSpeedSMART_GAIL(CircularVelocitySMART_GAIL):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        decoder=self.encoder.init_decoder
        flow=decoder.G1
        if (flow.velocity_representation!='speed' or flow.model.attribute_source is not None
                or flow.velocity_source!='gaussian' or flow.use_sde):
            raise ValueError('Rectified experiment requires deterministic Gaussian scalar speed')
        flow.model.attribute_source=RectifiedGaussianSpeedSource()
        decoder.denoiser_options['attribute_source']=flow.model.attribute_source.get_extra_state()
