"""Actual-cache value/order/stride and main-process RNG parity before use."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/sim_mpl_continuous'
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import datetime,json,multiprocessing as mp,pickle,random,hashlib
    import numpy as np
    import torch
    from tools.tensor_cache_numpy_worker_20261015 import initialize
    from tools.prefetch_tensor_cache_20261015 import PrefetchPool
    from tools.runtime_isolated_cache_recovery_20261015 import install_reader
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    base=root/'output/reports/init_diffusion_continuous_20261014'
    manifest=root/'logs/init_diffusion_continuous_20261014/periodic_edge_training_v2/control_T4817_S2048_N8192/control/train_files.pkl'
    names=pickle.loads(manifest.read_bytes())['files']
    ids=list(range(128))+list(range(225280,225344))+list(range(273920,273984))+[len(names)-1,0,1,61,64,1000]
    pool=mp.get_context('spawn').Pool(1,initializer=initialize,maxtasksperchild=128)
    proxy=PrefetchPool(pool,manifest,base/'prefetch_cache_parity_io.json')
    native=torch.load
    install_reader(torch,proxy,base/'prefetch_cache_parity_isolated_io.json')
    torch.manual_seed(517)
    rng=torch.get_rng_state().clone();python_rng=random.getstate();numpy_rng=pickle.dumps(np.random.get_state())
    fields=0
    try:
        for i in ids:
            file=root/'src/waymo_data/full/training_map2_sd'/names[i]
            a=native(file,map_location='cpu',weights_only=True)
            b=torch.load(file,map_location='cpu',weights_only=False)
            assert a.keys()==b.keys()
            for k in a:
                assert a[k].keys()==b[k].keys()
                for key,x in a[k].items():
                    y=b[k][key]
                    if torch.is_tensor(x):
                        assert x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride()
                        torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
                        fields+=1
                    else:assert x==y
        assert torch.equal(rng,torch.get_rng_state())
        assert python_rng==random.getstate() and numpy_rng==pickle.dumps(np.random.get_state())
        assert proxy.records['cache_hits']>200 and proxy.records['batch_rpcs']<len(ids)//8
        result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            files_requested=len(ids),tensor_fields=fields,values_dtypes_shapes_strides_equal=True,
            main_Torch_Python_NumPy_rng_unchanged=True,epoch_boundary_and_random_jump_checked=True,
            io=proxy.records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitation='Input parity proof only. Does not prove GPUtraining stability or performance gains.')
        (base/'prefetch_cache_parity_20261015.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:proxy.terminate();proxy.join()


if __name__=='__main__':main()
