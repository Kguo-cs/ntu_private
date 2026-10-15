"""Isolated recovery I/O: native safe TRAIN loads and save-only512-step guards."""
from pathlib import Path
import functools
import gc
import faulthandler
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    faulthandler.enable(all_threads=True)
    gc.disable()
    gc.collect=lambda *args,**kwargs:0
    import torch
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    policy=output.parent/(output.name+'.tensor_cache_policy.json')
    records=dict(train_cache_loader='native weights_only=True',safe_loads=0,
                 checkpoint_and_validation_loaders='existing trusted loader',
                 save_only_checkpoint_interval=512,faulthandler=True,
                 data_model_and_objective_unchanged=True,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        policy.parent.mkdir(parents=True,exist_ok=True)
        policy.write_text(json.dumps(records,indent=2)+'\n')
    original=torch.load
    @functools.wraps(original)
    def load(source,*args,**kwargs):
        name=source if isinstance(source,(str,os.PathLike)) else getattr(source,'name',None)
        train=(isinstance(name,(str,os.PathLike)) and 'training_map2_sd' in Path(name).parts)
        if train:
            if len(args)>1:
                raise ValueError('Safe TRAIN loader does not accept positional pickle_module')
            kwargs=dict(kwargs,weights_only=True)
            kwargs.pop('pickle_module',None)
        result=original(source,*args,**kwargs)
        if train:
            records['safe_loads']+=1
            if records['safe_loads']==1 or records['safe_loads']%4096==0:
                save()
        return result
    torch.load=load
    from tools import scratch_component_loss_sweep as sweep
    init=sweep.ScratchDiagnostics.__init__
    @functools.wraps(init)
    def diagnostics(self,checkpoint_steps,folder,metadata,*args,**kwargs):
        if not checkpoint_steps:
            return init(self,checkpoint_steps,folder,metadata,*args,**kwargs)
        end=max(checkpoint_steps)
        start=int(metadata.get('resume_global_step',0))
        guard=range((start//512+1)*512,end+1,512)
        # The runner's evaluation list stays unchanged; only the callback gets
        # additional recovery save points, with post-update RNG/optimizer state.
        extended=sorted(set(checkpoint_steps)|set(guard))
        metadata=dict(metadata,train_cache_io_policy='native_weights_only',
                      recovery_save_interval=512)
        return init(self,extended,folder,metadata,*args,**kwargs)
    sweep.ScratchDiagnostics.__init__=diagnostics
    save()
    from tools.runtime_periodic_edge_sweep_20261015 import main as run
    run()


if __name__=='__main__':
    main()
