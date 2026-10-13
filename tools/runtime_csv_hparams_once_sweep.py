"""Inactive fallback entrypoint: cached Torch loading plus once-per-update YAML.

Calls the existing cached-loader entrypoint unchanged. Only this process's
sweep.CSVLogger and runtime_policy hooks are replaced, then restored on return
or exception. Real successful runs retain the underlying flush + os._exit(0).
No import-time Torch/Lightning import, monkeypatch, training or GPU operation.
This reduces repeated telemetry serialization; it is not a proven SIGSEGV fix.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_fresh_output(argv):
    try:output=Path(argv[argv.index('--output')+1]).resolve()
    except (ValueError,IndexError):raise SystemExit('This telemetry entrypoint requires an explicit fresh --output')
    if output.exists():
        if not output.is_dir():raise RuntimeError('Output is not a directory: '+str(output))
        # A newly supervising legacy/edge worker creates only these before launch.
        scaffolding={'edge_contract.json','console.log','driver_console.log'}
        nonfresh=[str(p) for p in output.iterdir() if p.name not in scaffolding or not p.is_file()]
        existing=list(output.rglob('metrics.csv'))
        if nonfresh or existing:
            raise RuntimeError('Telemetry output is not fresh; never reuse previous CSV/protocol/data: '+str(nonfresh+list(map(str,existing))))
    return output


def hparams_sha(hparams):
    """Fingerprint endpoint configuration, never each metrics flush."""
    from omegaconf import OmegaConf
    def plain(value):
        if OmegaConf.is_config(value):return plain(OmegaConf.to_container(value,resolve=True,enum_to_str=True))
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        if isinstance(value,Path):return str(value)
        if value is None or isinstance(value,(str,bool,int,float)):return value
        raise TypeError('Unsupported hparams snapshot type: '+type(value).__name__)
    return hashlib.sha256(json.dumps(plain(hparams),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_logger_factory(registry):
    # main establishes GC deferral before these imports, as in the old runtime.
    from tools.prototype_csv_hparams_once import HparamsOnceCSVLogger,HparamsOnceWriter
    from lightning.fabric.loggers.logger import rank_zero_experiment

    class TrackedWriter(HparamsOnceWriter):
        def __init__(self,log_dir):
            super().__init__(log_dir)
            self.save_calls=0;self.hparams_log_calls=0;self.invalidations=0
            self.declared_hparams_sha256=hparams_sha(self.hparams)
            self.first_written_hparams_sha256=None
            self.first_hparams_file_sha256=None
            registry['writers'].append(self)

        def invalidate_hparams(self):
            super().invalidate_hparams();self.invalidations+=1
            self.declared_hparams_sha256=hparams_sha(self.hparams)

        def log_hparams(self,params):
            self.hparams_log_calls+=1
            return super().log_hparams(params)

        def save(self):
            self.save_calls+=1
            target=Path(self.log_dir)/self.NAME_HPARAMS_FILE
            if self._hparams_dirty or not target.is_file():
                current=hparams_sha(self.hparams)
                if current!=self.declared_hparams_sha256:
                    raise RuntimeError('Hparams changed without log_hyperparams/invalidate_hparams: '+self.log_dir)
                if self.first_written_hparams_sha256 is None:self.first_written_hparams_sha256=current
            result=super().save()
            if self.first_hparams_file_sha256 is None and target.is_file():self.first_hparams_file_sha256=sha(target)
            return result

        def counter_snapshot(self):
            current=hparams_sha(self.hparams)
            if current!=self.declared_hparams_sha256:
                raise RuntimeError('Hparams mutated without explicit invalidation; runtime success refused: '+self.log_dir)
            target=Path(self.log_dir)/self.NAME_HPARAMS_FILE
            return {'log_dir':self.log_dir,'save_calls':self.save_calls,
                'hparams_log_calls':self.hparams_log_calls,'explicit_invalidations':self.invalidations,
                'hparams_YAML_serializations':self.hparams_serializations,
                'first_written_hparams_sha256':self.first_written_hparams_sha256,
                'last_declared_hparams_sha256':self.declared_hparams_sha256,
                'current_hparams_sha256':current,'matches_last_declared_hparams':True,
                'unchanged_since_first_write':self.first_written_hparams_sha256==current,
                'first_hparams_file_sha256':self.first_hparams_file_sha256,
                'current_hparams_file_sha256':sha(target) if target.is_file() else None}

    class FreshHparamsOnceCSVLogger(HparamsOnceCSVLogger):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if Path(self.root_dir).exists():
                raise RuntimeError('CSV directory is not fresh; version increment cannot hide reuse: '+self.root_dir)
            registry['loggers_created']+=1

        @property
        @rank_zero_experiment
        def experiment(self):
            if self._experiment is not None:return self._experiment
            self._fs.makedirs(self.root_dir,exist_ok=True)
            self._experiment=TrackedWriter(log_dir=self.log_dir)
            return self._experiment

    return FreshHparamsOnceCSVLogger


def invoke(runtime,sweep):
    """Install only scoped CSV/policy hooks; keep native loader and main intact."""
    original_csv=sweep.CSVLogger;original_policy=runtime.runtime_policy
    registry={'writers':[],'loggers_created':0}
    factory=make_logger_factory(registry)
    from lightning.pytorch.loggers.csv_logs import CSVLogger as NativeCSVLogger
    if original_csv is not NativeCSVLogger:raise RuntimeError('Unexpected preexisting CSVLogger replacement')
    telemetry_sources={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tools/prototype_csv_hparams_once.py',Path(runtime.__file__).resolve())}

    def policy():
        underlying=original_policy()
        if any(sha(path)!=digest for path,digest in telemetry_sources.items()):
            raise RuntimeError('Telemetry runtime source changed during child execution')
        sources={**underlying.get('source_hashes',{}),**telemetry_sources}
        counters=[writer.counter_snapshot() for writer in registry['writers']]
        return {**underlying,'policy':'public_cached_loader_full_gc_deferral_csv_hparams_once',
            'source_hashes':sources,'underlying_runtime_policy':underlying,
            'telemetry_policy':{'name':'CSV_hparams_first_save_and_explicit_update_only',
                'entrypoint':str(Path(__file__).resolve()),'logger_factory':factory.__name__,
                'source_hashes':telemetry_sources,'loggers_created':registry['loggers_created'],
                'writer_counters':counters,
                'nested_updates':'Direct/nested hparams changes require explicit invalidate_hparams; endpoint fingerprints refuse success if bypassed.',
                'snapshot_scope':'First write, declared updates and runtime endpoint; not per-step snapshots or proof of identical training noise.',
                'CSV_metrics_writer':'Unchanged native FabricWriter.save via prototype',
                'freshness':'Explicit --output; only new supervisor contract/log scaffolding allowed; existing CSV root/version directories rejected.',
                'failure_claim':'Not a proven SIGSEGV fix; first/updated YAML still uses original serializer.',
                'model_data_optimizer_objective_change':False}}

    sweep.CSVLogger=factory;runtime.runtime_policy=policy
    try:return runtime.main()
    finally:
        sweep.CSVLogger=original_csv
        runtime.runtime_policy=original_policy


def main():
    deferred=os.environ.get('SIM_SWEEP_DEFER_CYCLIC_GC')=='1' or '--defer-cyclic-gc' in sys.argv
    try:interval=int(sys.argv[sys.argv.index('--young-gc-interval')+1])
    except (ValueError,IndexError):raise SystemExit('CSV telemetry runtime requires explicit --young-gc-interval 0')
    if not deferred or interval!=0:raise SystemExit('CSV telemetry runtime requires GC deferral and interval0')
    require_fresh_output(sys.argv)
    # Preserve existing runtime policy before any Torch/TF/Lightning imports.
    gc.disable();gc.collect=lambda *unused_args,**unused_kwargs:0
    os.environ['SIM_SWEEP_DEFER_CYCLIC_GC']='1'
    runtime=importlib.import_module('tools.runtime_cached_torch_loader_sweep')
    sweep=importlib.import_module('tools.overnight_init_diffusion_sweep').sweep
    return invoke(runtime,sweep)


if __name__=='__main__':main()
