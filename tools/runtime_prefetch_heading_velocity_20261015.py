"""Native shared-omega scratch trainer with isolated64-file NumPy prefetch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import multiprocessing as mp
    from types import SimpleNamespace
    from tools import runtime_isolated_cache_recovery_20261015 as isolated
    from tools.prefetch_tensor_cache_20261015 import PrefetchPool
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    arm=sys.argv[sys.argv.index('--variants')+1]
    manifest=output/arm/'train_files.pkl'
    policy=output.parent/(output.name+'.prefetch_cache_io.json')
    class Context:
        def __init__(self,context):self.context=context
        def Pool(self,*args,**kwargs):
            assert kwargs['maxtasksperchild']==8192
            kwargs=dict(kwargs,maxtasksperchild=128)
            return PrefetchPool(self.context.Pool(*args,**kwargs),manifest,policy,batch_files=64)
    # Replace only this runtime module's namespace, not multiprocessing globally.
    isolated.mp=SimpleNamespace(get_context=lambda kind:Context(mp.get_context(kind)))
    from tools.shared_heading_velocity_20261015 import main as run
    run()


if __name__=='__main__':main()
