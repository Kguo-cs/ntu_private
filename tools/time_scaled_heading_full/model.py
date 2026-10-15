"""Supply analytic1/t scaling while preserving the directomega MSE objective."""
from types import MethodType
import torch
from src.smart.diffusion.denoiser import InitDenoiser
from tools.shared_heading_full.model import CircularVelocitySMART_GAIL

_BASE_FORWARD=InitDenoiser.forward


def time_scaled_forward(self,m_delta,beta,tokenized_agent,map_feature,eval_mask=None,mode=1,use_map_condition=True):
    prediction=_BASE_FORWARD(self,m_delta,beta,tokenized_agent,map_feature,eval_mask=eval_mask,mode=mode,use_map_condition=use_map_condition)
    effective_beta=beta if eval_mask is None else beta[eval_mask]
    time=self._format_beta(effective_beta,len(prediction))[:,2:3]
    omega=prediction[:,-1:]/time.clamp_min(self._heading_velocity_time_floor)
    return torch.cat((prediction[:,:-1],omega),dim=-1)


def install_time_scaled_readout(decoder):
    flow=decoder.G1;model=flow.model
    if decoder.heading_objective!='angular_velocity' or decoder.denoiser_options.get('heading_velocity_readout')!='shared_state_row3':
        raise ValueError('Requires the shared scalaromega readout')
    if 'heading_velocity_parameterization' in decoder.denoiser_options:
        raise ValueError('Heading velocity parameterization already selected')
    model._heading_velocity_time_floor=float(flow.t_eps)
    model.forward=MethodType(time_scaled_forward,model)
    decoder.denoiser_options.update(heading_velocity_parameterization='time_scaled',heading_velocity_time_floor=float(flow.t_eps))


class TimeScaledCircularVelocitySMART_GAIL(CircularVelocitySMART_GAIL):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_time_scaled_readout(self.encoder.init_decoder)
