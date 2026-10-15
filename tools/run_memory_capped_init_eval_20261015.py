"""Run the native CLI evaluator with an explicit process-local CUDA memory cap."""
from pathlib import Path
import argparse,json,runpy,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import gc
    gc.disable()
    gc.collect=lambda *args,**kwargs:0
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu-memory-fraction',type=float,required=True)
    parser.add_argument('--resource-report',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    if not 0<args.gpu_memory_fraction<=.5:raise ValueError('Use a bounded secondary evaluation allocation')
    report=args.resource_report.resolve();report.parent.mkdir(parents=True,exist_ok=True)
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,0)
    begin=time.monotonic();success=False
    try:
        sys.argv=[str(ROOT/'src/run.py'),*remaining]
        runpy.run_path(str(ROOT/'src/run.py'),run_name='__main__')
        success=True
    finally:
        record=dict(native_cli_success=success,process_memory_fraction=args.gpu_memory_fraction,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            elapsed_seconds=time.monotonic()-begin,command=remaining,
            caveat='Memory cap applies to this process only. Batch64 changes generation RNG batching versus batch128; record it explicitly.')
        report.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
