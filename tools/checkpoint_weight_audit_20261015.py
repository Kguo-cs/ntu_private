"""Verify which genuine checkpoint parameters are used, without modifying them."""
import hashlib

def parameter_digest(named_parameters):
 digest=hashlib.sha256()
 for name,value in named_parameters:
  value=value.detach().cpu().contiguous()
  digest.update(str((name,str(value.dtype),tuple(value.shape))).encode())
  digest.update(value.numpy().tobytes())
 return digest.hexdigest()

def checkpoint_parameter_digest(path,mode):
 import torch
 if mode not in ('raw','ema'):raise ValueError('weights must be raw or ema')
 checkpoint=torch.load(path,map_location='cpu',weights_only=False)
 state=checkpoint['state_dict'];extra=state['encoder.init_decoder._extra_state']
 names=extra['ema_parameter_names']
 if not names or len(set(names))!=len(names):raise ValueError('Missing or duplicate parameter names')
 if mode=='ema':
  values=extra['ema']['shadow_params']
  if len(names)!=len(values):raise ValueError('EMA parameter layout mismatch')
 else:values=[state['encoder.init_decoder.G1.'+name] for name in names]
 return parameter_digest(zip(names,values))
