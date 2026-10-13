"""Isolated native sweep with periodic recovery saves and lighter CSV logging.

Adds checkpoint/trace writes only. Forward passes, optimizer, EMA, scheduler,
training/evaluation data and randomness remain in the original native runner.
"""
import gc
import json
import os
from pathlib import Path
import sys
gc.disable();gc.collect=lambda *_,**__:0
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['SIM_SWEEP_DEFER_CYCLIC_GC']='1'
import torch
from tools import overnight_init_diffusion_sweep
from tools import scratch_component_loss_sweep as sweep
from tools import runtime_cached_torch_loader_sweep as runtime
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from lightning.fabric.loggers.csv_logs import _ExperimentWriter

# The exact resolved model configuration is already saved once as config.yaml.
# Avoid reserializing its nested OmegaConf at every CSV metrics flush. This is
# a process-local logger policy, not a change to training or installed packages.
ExperimentWriter.save=_ExperimentWriter.save
Original=sweep.ScratchDiagnostics


class RecoveryDiagnostics(Original):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx)
        if trainer.global_step % 512 == 0:
            if trainer.global_step not in self.checkpoint_steps:
                trainer.save_checkpoint(self.folder/f'step_{trainer.global_step}.ckpt')
            path=self.folder/'recovery_trace.json';tmp=path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(self.records)+'\n');os.replace(tmp,path)


sweep.ScratchDiagnostics=RecoveryDiagnostics
if __name__=='__main__':
    runtime.main()
