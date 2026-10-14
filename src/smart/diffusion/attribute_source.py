"""Optional TRAIN-marginal source for scalar speed and linear vehicle sizes.

This changes the starting distribution, not the target, interpolation or loss.
The inverse empirical CDF uses existing Gaussian draws; it consumes no extra RNG
and never conditions on an evaluation agent's ground-truth type or attributes.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from collections.abc import Mapping
import torch
from torch import nn


class AttributeQuantileSource(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, Mapping):
            raise ValueError('attribute_source must be a mapping')
        if set(config)-{'path','sha256','speed','shape','knots'}:
            raise ValueError('Unknown attribute_source options')
        self.speed=config.get('speed',True)
        self.shape=config.get('shape',False)
        if not isinstance(self.speed,bool) or not isinstance(self.shape,bool) or not (self.speed or self.shape):
            raise ValueError('At least one boolean speed/shape source must be enabled')
        self.knots=config.get('knots',2049)
        if isinstance(self.knots,bool) or not isinstance(self.knots,int) or self.knots<3:
            raise ValueError('attribute_source.knots must be an integer >=3')
        self.register_buffer('quantile_values',torch.full((self.knots,3),float('nan')))
        self._ready=False
        self.source_sha256=config.get('sha256')
        path=config.get('path')
        if path:
            if not isinstance(self.source_sha256,str) or len(self.source_sha256)!=64:
                raise ValueError('TRAIN source requires an explicit SHA256')
            path=Path(path)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.source_sha256:
                raise ValueError('Attribute source file hash differs from configuration')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload.get('fields')!=['length','width','speed']:
                raise ValueError('Expected physical TRAIN [length,width,speed] fields')
            is_table=payload.get('kind')=='quantile_table'
            values=payload['quantile_values'] if is_table else payload['attributes']
            if values.ndim!=2 or values.shape[1]!=3 or len(values)<2:
                raise ValueError('Expected at least two TRAIN attribute observations')
            if not torch.isfinite(values).all() or (values[:,:2]<=0).any() or (values[:,2]<0).any():
                raise ValueError('TRAIN size/speed source must be finite and physically valid')
            if is_table:
                if len(values)!=self.knots or (values[1:]<values[:-1]).any():
                    raise ValueError('Saved TRAIN quantile table must be ordered and match configured knots')
                quantiles=values.float()
            else:
                quantiles=torch.quantile(values.float(),torch.linspace(0,1,self.knots),dim=0)
            self.quantile_values.copy_(quantiles)
            self._ready=True

    def get_extra_state(self):
        return dict(source_sha256=self.source_sha256,knots=self.knots,speed=self.speed,shape=self.shape)

    def set_extra_state(self,state):
        if not isinstance(state,Mapping) or any(state.get(k)!=getattr(self,k) for k in ('knots','speed','shape')):
            raise ValueError('Checkpoint attribute source differs from configured source')
        if self.source_sha256 is not None and state.get('source_sha256')!=self.source_sha256:
            raise ValueError('Checkpoint TRAIN attribute source hash mismatch')
        self.source_sha256=state.get('source_sha256')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        value=state_dict.get(prefix+'quantile_values')
        if value is not None:
            if (value.shape!=self.quantile_values.shape or not torch.isfinite(value).all()
                    or (value[1:]<value[:-1]).any() or (value[:,:2]<=0).any() or (value[:,2]<0).any()):
                error_msgs.append('Invalid checkpoint attribute quantiles')
            else:self._ready=True
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,standard_noise,physical_noise):
        if not self._ready:
            raise RuntimeError('Load TRAIN attribute source or its saved checkpoint before sampling')
        if standard_noise.ndim!=2 or standard_noise.shape!=physical_noise.shape or standard_noise.shape[1]!=7:
            raise ValueError('Attribute source requires [N,7] scalar-speed physical states')
        # Float32 avoids coarse bfloat16 CDF ranks even if the denoiser is mixed precision.
        ranks=(.5*(1+torch.erf(standard_noise[:,4:7].float()/math.sqrt(2)))).clamp(0,1)*(self.knots-1)
        lo=ranks.floor().long();hi=(lo+1).clamp_max(self.knots-1)
        columns=torch.arange(3,device=ranks.device)
        table=self.quantile_values.to(device=ranks.device,dtype=torch.float32)
        attrs=torch.lerp(table[lo,columns],table[hi,columns],ranks-lo).to(physical_noise.dtype)
        result=physical_noise.clone()
        if self.shape:result[:,4:6]=attrs[:,:2]
        if self.speed:result[:,6]=attrs[:,2]
        return result
