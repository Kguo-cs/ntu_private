"""Compare cache-read throughput with identical inputs and one CPU worker."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/sim_mpl_continuous'
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import datetime,json,multiprocessing as mp,pickle,time
    import torch
    from tools.tensor_cache_numpy_worker_20261015 import initialize
    from tools.prefetch_tensor_cache_20261015 import PrefetchPool
    from tools.runtime_isolated_cache_recovery_20261015 import install_reader
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];base=root/'output/reports/init_diffusion_continuous_20261014'
    manifest=root/'logs/init_diffusion_continuous_20261014/periodic_edge_training_v2/control_T4817_S2048_N8192/control/train_files.pkl'
    names=pickle.loads(manifest.read_bytes())['files']
    paths=[root/'src/waymo_data/full/training_map2_sd'/n for n in names[:320]]
    pool=mp.get_context('spawn').Pool(1,initializer=initialize,maxtasksperchild=8192)
    try:
        install_reader(torch,pool,base/'prefetch_benchmark_scalar_io.json')
        torch.load(paths[0],map_location='cpu',weights_only=False)
        start=time.monotonic()
        for p in paths[64:]:torch.load(p,map_location='cpu',weights_only=False)
        scalar=time.monotonic()-start
        proxy=PrefetchPool(pool,manifest,base/'prefetch_benchmark_batch_io.json')
        install_reader(torch,proxy,base/'prefetch_benchmark_outer_io.json')
        torch.load(paths[0],map_location='cpu',weights_only=False)
        start=time.monotonic()
        for p in paths[64:]:torch.load(p,map_location='cpu',weights_only=False)
        batched=time.monotonic()-start
        result=dict(status='CACHE_IO_BENCHMARK_COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            files=256,scalar_seconds=scalar,batched_seconds=batched,read_speedup=scalar/batched,
            worker_spawn_warmup_excluded=True,same_single_worker=True,
            limitation='Cache IO only, hotpagecache and shortsample. Not an end-to-end training speedup claim.')
        (base/'prefetch_cache_benchmark_20261015.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:pool.terminate();pool.join()


if __name__=='__main__':main()
