"""PureCPU Torch-cache reader; only NumPy/plain values cross the process pipe."""
import gc
import os


def initialize():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    gc.enable()


def read(path):
    import numpy as np
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True)
    def pack(x):
        if torch.is_tensor(x):
            if x.requires_grad or x.layout != torch.strided or any(s<0 for s in x.stride()):
                raise ValueError('Unsupported TRAIN tensor layout')
            return {'__tensor_array__':True,'shape':tuple(x.shape),'stride':tuple(x.stride()),
                    'array':np.array(x.numpy(),copy=True,order='C')}
        if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
        if isinstance(x,(str,int,float,bool,type(None))):return x
        raise TypeError('Tensor-only TRAIN cache has unsupported value: '+type(x).__name__)
    return pack(value)
