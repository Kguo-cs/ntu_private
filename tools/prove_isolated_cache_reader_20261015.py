"""Check subprocess reader tensor/RNG parity before resuming any training."""
from pathlib import Path
import multiprocessing as mp
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch,pickle,json,hashlib,datetime
    from tools.tensor_cache_numpy_worker_20261015 import initialize
    from tools.runtime_isolated_cache_recovery_20261015 import install_reader
    base=ROOT/'output/reports/init_diffusion_continuous_20261014'
    names=pickle.loads((ROOT/'logs/init_diffusion_continuous_20261014/periodic_edge_training_v2/'
        'control_T4817_S2048_N8192/control/train_files.pkl').read_bytes())['files']
    ids=list(range(16))+list(range(273920,273984))+list(range(225280,225344))
    original=torch.load
    pool=mp.get_context('spawn').Pool(1,initializer=initialize,maxtasksperchild=8192)
    policy=base/'isolated_io_preflight_policy.json'
    install_reader(torch,pool,policy)
    torch.manual_seed(517)
    rng=torch.get_rng_state().clone()
    fields=0
    try:
        for i in ids:
            file=ROOT/'src/waymo_data/full/training_map2_sd'/names[i]
            a=original(file,map_location='cpu',weights_only=True)
            b=torch.load(file,map_location='cpu',weights_only=False)
            assert a.keys()==b.keys()
            for key in a:
                assert a[key].keys()==b[key].keys()
                for k,x in a[key].items():
                    y=b[key][k]
                    if torch.is_tensor(x):
                        assert x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride()
                        torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
                        fields+=1
                    else:assert x==y
        assert torch.equal(rng,torch.get_rng_state())
        result={'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'files':len(ids),'tensor_fields':fields,'values_dtypes_shapes_strides_equal':True,
            'main_torch_rng_unchanged':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitation':'Storage offsets/aliasing not retained; cache inputs are read-only. '
                         'No claim of training stability until actual continuation succeeds.'}
        (base/'isolated_cache_reader_parity_20261015.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result,flush=True)
    finally:
        pool.close();pool.join()


if __name__=='__main__':
    main()
