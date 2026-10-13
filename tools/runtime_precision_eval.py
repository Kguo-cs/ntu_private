"""Evaluation-only native runtime with an explicit, recorded matmul override."""
import gc
import os
from pathlib import Path
import sys
gc.disable();gc.collect=lambda *_,**__:0
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['SIM_SWEEP_DEFER_CYCLIC_GC']='1'
if '--evaluation-checkpoints' not in sys.argv:
    raise SystemExit('This diagnostic entry point permits evaluation only')
precision=os.environ.get('SIM_AUDIT_MATMUL_PRECISION','highest')
if precision not in ('highest','high','medium'):raise SystemExit('Invalid explicit matmul precision')
from tools import overnight_init_diffusion_sweep
from tools import scratch_component_loss_sweep as sweep
from tools import runtime_cached_torch_loader_sweep as runtime
from omegaconf import open_dict
original_compose=sweep.compose


def compose(*args,**kwargs):
    cfg=original_compose(*args,**kwargs)
    with open_dict(cfg):cfg.float32_matmul_precision=precision
    return cfg


sweep.compose=compose
if __name__=='__main__':runtime.main()
