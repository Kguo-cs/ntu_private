"""Compare real batched TRAIN values/RNG for native0 vs8 data workers."""
from pathlib import Path
import argparse,gc,hashlib,json,pickle,random,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[0,8],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    gc.disable();gc.collect=lambda *args,**kwargs:0
    import jinja2,torch,numpy as np
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    from src.smart.datasets.scalable_dataset import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
    from torch_geometric.loader import DataLoader
    torch.set_num_threads(1)
    with (ROOT/'src/waymo_data/full/training_map2_sd_files.pkl').open('rb') as f:names=list(pickle.load(f)['files'])
    random.Random(4817).shuffle(names);names=names[:4096]
    manifest=a.output/'files.pkl';manifest.write_bytes(pickle.dumps({'files':names}))
    dataset=MultiDataset(str(ROOT/'src/waymo_data/full/training_map2_sd'),WaymoTargetBuilderTrain(32),sample_list=str(manifest),sample_list_check_exists=False)
    loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=a.workers,pin_memory=False,persistent_workers=False,drop_last=False)
    random.seed(4817);np.random.seed(4817);torch.manual_seed(4817)
    def rng():
        return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                    numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())
    before=rng();rows=[];begin=time.monotonic();it=iter(loader)
    for i in range(len(loader)):
        t=time.monotonic();data=next(it);wait=time.monotonic()-t;h=hashlib.sha256();tensor_fields=0
        def visit(value):
            nonlocal tensor_fields
            if torch.is_tensor(value):
                tensor_fields+=1;v=value.detach().cpu();h.update(str((str(v.dtype),tuple(v.shape),tuple(v.stride()))).encode());h.update(v.contiguous().numpy().tobytes())
            elif isinstance(value,np.ndarray):h.update(str((value.dtype.str,value.shape)).encode());h.update(value.tobytes())
            elif isinstance(value,dict):
                for k in sorted(value,key=repr):h.update(repr(k).encode());visit(value[k])
            elif isinstance(value,(list,tuple)):
                h.update(type(value).__name__.encode())
                for v in value:visit(v)
            else:h.update((type(value).__name__+repr(value)).encode())
        visit(data.to_dict());visit(data.num_graphs)
        rows.append(dict(batch=i,num_graphs=data.num_graphs,tensor_fields=tensor_fields,sha256=h.hexdigest(),loader_wait_seconds=wait))
    end=rng();elapsed=time.monotonic()-begin
    del it,loader
    r=dict(status='COMPLETE',workers=a.workers,scenes=len(names),batch_size=512,rows=rows,rng_before=before,rng_after=end,elapsed_seconds_including_hashes=elapsed,
           sum_next_batch_wait_seconds=sum(x['loader_wait_seconds'] for x in rows),kind='CPU input/value/order/RNG evidence only; no training-speed or CUDA-stability claim')
    (a.output/'proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['rows','rng_before','rng_after']}))

if __name__=='__main__':main()
