"""Isolate cache deserialization from the CUDA training interpreter."""
from pathlib import Path
import functools
import hashlib
import json
import multiprocessing as mp
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def install_reader(torch,pool,policy):
    from tools.tensor_cache_numpy_worker_20261015 import read
    original=torch.load
    records=dict(mode='spawnCPU Tensor->NumPy pipe->Tensor',loads=0,
                 main_TRAIN_torch_deserialization=False,worker_cuda_visible_devices='',
                 preserve_tensor_values_shapes_dtypes_strides=True,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def unpack(x):
        if isinstance(x,dict) and x.get('__tensor_array__') is True:
            source=torch.from_numpy(x['array']).reshape(x['shape'])
            if any(s==0 and n>1 for s,n in zip(x['stride'],x['shape'])):
                raise ValueError('Overlapping stride-zero TRAIN tensor needs explicit support')
            value=torch.empty_strided(x['shape'],x['stride'],dtype=source.dtype,device='cpu')
            value.copy_(source)
            return value
        if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
        return x
    @functools.wraps(original)
    def load(source,*args,**kwargs):
        name=source if isinstance(source,(str,os.PathLike)) else getattr(source,'name',None)
        if isinstance(name,(str,os.PathLike)) and 'training_map2_sd' in Path(name).parts:
            value=unpack(pool.apply_async(read,(str(name),)).get(timeout=60))
            records['loads']+=1
            if records['loads']==1 or records['loads']%4096==0:
                policy.parent.mkdir(parents=True,exist_ok=True)
                policy.write_text(json.dumps(records,indent=2)+'\n')
            return value
        return original(source,*args,**kwargs)
    torch.load=load
    return records


def main():
    import gc,faulthandler
    gc.disable()
    faulthandler.enable(all_threads=True)
    import jinja2
    from tools.tensor_cache_numpy_worker_20261015 import initialize
    pool=mp.get_context('spawn').Pool(1,initializer=initialize,maxtasksperchild=8192)
    import torch
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    policy=output.parent/(output.name+'.isolated_cache_io.json')
    install_reader(torch,pool,policy)
    from tools.runtime_tensor_cache_recovery_20261015 import main as run
    try:
        run()
    finally:
        pool.terminate()
        pool.join()


if __name__=='__main__':
    main()
