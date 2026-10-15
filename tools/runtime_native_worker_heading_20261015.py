"""Fresh-only scratch training with native data workers and a scalar heading option.

Reuses the scratch trainer/tester. No isolated per-file RPC reader is installed.
"""
from pathlib import Path
import argparse,gc,hashlib,json,os,resource,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--heading-readout',choices=['raw','time_scaled'],required=True)
    p.add_argument('--gpu-memory-fraction',type=float,required=True)
    a,remaining=p.parse_known_args()
    if '--resume-map' in remaining or '--resume-cyclic-epochs' in remaining:
        raise ValueError('Native worker runtime is fresh-only until worker resume is separately verified')
    if not 0<a.gpu_memory_fraction<=1:raise ValueError('Invalid process memory fraction')
    gc.disable();gc.collect=lambda *args,**kwargs:0
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,16384),hard),hard))
    import jinja2,torch
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,0)
    from tools.shared_heading_velocity_20261015 import install_training_policy
    install_training_policy()
    if a.heading_readout=='time_scaled':
        import functools
        from src.smart.diffusion.initial_diffusion import InitDiffusion
        from tools.time_scaled_heading_full.model import install_time_scaled_readout
        original=InitDiffusion.__init__
        @functools.wraps(original)
        def init(self,*args,**kwargs):
            original(self,*args,**kwargs)
            if self.heading_objective=='angular_velocity':install_time_scaled_readout(self)
        InitDiffusion.__init__=init
    from tools import scratch_component_loss_sweep as sweep
    sweep.SEARCH_CONFIG_KEYS.update({'model.model_config.decoder.init_diffusion.'+k for k in ('heading_objective','heading_x0_loss','heading_flow_loss_weight')})
    original_main=sweep.main
    output=Path(remaining[remaining.index('--output')+1]).resolve()
    assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    policy=output.parent/(output.name+'.native_workers.json')
    settings=dict(heading_readout=a.heading_readout,process_memory_fraction=a.gpu_memory_fraction,
        worker_count=int(remaining[remaining.index('--workers')+1]),fresh_only=True,isolated_per_file_RPC=False,
        normal_native_train_and_eval_loaders=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    policy.write_text(json.dumps(settings,indent=2)+'\n')
    def observed_main():
        start=time.monotonic();original_main()
        record=dict(**settings,status='COMPLETE',elapsed_seconds=time.monotonic()-start,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        (output/'native_worker_resource_report.json').write_text(json.dumps(record,indent=2)+'\n')
    sweep.main=observed_main
    sys.argv=[sys.argv[0],*remaining]
    from tools.runtime_periodic_edge_sweep_20261015 import main as run
    run()

if __name__=='__main__':main()
