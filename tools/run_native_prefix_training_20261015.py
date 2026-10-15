"""Run the original run0 trainer with a process-local loader/runtime policy.

The audit records the actual per-epoch filename order, LR and initial generator
parameters. The added dataset field is metadata only; the native shuffle,
optimizer, loss, validation loop and scheduler remain intact.
"""
from pathlib import Path
from types import MethodType
import argparse
import gc
import hashlib
import json
import os
import resource
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest_parameters(module):
    import torch
    digest = hashlib.sha256()
    for name, tensor in module.named_parameters():
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--gpu-memory-fraction', type=float, required=True)
    p.add_argument('--audit-output', type=Path, required=True)
    p.add_argument('--expected-initial-g1-sha256')
    p.add_argument('--expected-initial-map-sha256')
    a, overrides = p.parse_known_args()
    assert 0 < a.gpu_memory_fraction <= .5
    a.audit_output.mkdir(parents=True, exist_ok=True)
    gc.disable()
    gc.collect = lambda *args, **kwargs: 0
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 16384), hard), hard))
    import jinja2, torch, lightning as L
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader, runtime_policy
    install_cached_loader(torch)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction, 0)
    from src.smart.datasets.scalable_dataset import MultiDataset
    original_get = MultiDataset.get

    def get(self, index):
        data = original_get(self, index)
        data['research_cache_file'] = Path(self.raw_paths[index]).name
        return data

    MultiDataset.get = get
    # Linux fork workers inherit the metadata-only get wrapper and cached loader.
    # Explicitly reject another context rather than silently lose the audit.
    import multiprocessing

    class NativePrefixAudit(L.Callback):
        def __init__(self):
            self.record = dict(epochs=[], numerical_training_change=False,
                metadata_only_dataset_field='research_cache_file', actual_shuffle_verified=False)

        def on_train_start(self, trainer, model):
            assert multiprocessing.get_context().get_start_method() == 'fork'
            decoder = model.encoder.init_decoder
            loader = trainer.train_dataloader
            assert type(loader.sampler).__name__ == 'RandomSampler', type(loader.sampler).__name__
            assert all(not v.requires_grad for v in model.encoder.map_encoder.parameters())
            self.record.update(initial_G1_sha256=digest_parameters(decoder.G1),
                initial_map_sha256=digest_parameters(model.encoder.map_encoder),
                generator_parameters=sum(v.numel() for v in decoder.G1.parameters()),
                training_dataset_size=len(trainer.datamodule.train_dataset),
                actual_shuffle_verified=True, sampler=type(loader.sampler).__name__,
                native_scheduler_horizon=model.lr_total_steps,
                initial_EMA_updates=decoder.ema.num_updates,
                runtime_policy=runtime_policy())
            assert model.lr_total_steps == 128
            assert decoder.ema.num_updates == 0, 'Native prefix requires a fresh generator'
            if a.expected_initial_g1_sha256:
                assert self.record['initial_G1_sha256'] == a.expected_initial_g1_sha256
            if a.expected_initial_map_sha256:
                assert self.record['initial_map_sha256'] == a.expected_initial_map_sha256
            self.flush()

        def on_train_epoch_start(self, trainer, model):
            self.epoch_hash = hashlib.sha256()
            self.examples = 0
            self.first_lr = trainer.optimizers[0].param_groups[0]['lr']

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            names = batch['research_cache_file']
            if isinstance(names, str):
                names = [names]
            assert len(names) == batch.num_graphs
            self.epoch_hash.update(('\n'.join(names) + '\n').encode())
            self.examples += len(names)

        def on_train_epoch_end(self, trainer, model):
            self.record['epochs'].append(dict(epoch=trainer.current_epoch, actual_examples=self.examples,
                order_sha256=self.epoch_hash.hexdigest(), first_lr=self.first_lr,
                last_lr=trainer.optimizers[0].param_groups[0]['lr'], global_step=trainer.global_step,
                EMA_updates=model.encoder.init_decoder.ema.num_updates))
            self.flush()

        def on_train_end(self, trainer, model):
            final_map = digest_parameters(model.encoder.map_encoder)
            assert final_map == self.record['initial_map_sha256'], 'Frozen map weights changed'
            self.record.update(status='NATIVE_PREFIX_COMPLETE', actual_step=trainer.global_step,
                EMA_updates=model.encoder.init_decoder.ema.num_updates,
                final_map_sha256=final_map, frozen_map_unchanged=True,
                final_CPU_RNG_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                final_CUDA_RNG_sha256=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                                      for x in torch.cuda.get_rng_state_all()])
            self.flush()

        def flush(self):
            temporary = a.audit_output / 'training_audit.json.tmp'
            temporary.write_text(json.dumps(self.record, indent=2) + '\n')
            temporary.replace(a.audit_output / 'training_audit.json')

    original_trainer = L.Trainer.__init__
    import functools

    @functools.wraps(original_trainer)
    def trainer_init(self, *args, **kwargs):
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), NativePrefixAudit()]
        return original_trainer(self, *args, **kwargs)

    L.Trainer.__init__ = trainer_init
    start = time.monotonic()
    success = False
    try:
        sys.argv = [str(ROOT / 'src/run0.py'), *overrides]
        runpy.run_path(str(ROOT / 'src/run0.py'), run_name='__main__')
        success = True
    finally:
        L.Trainer.__init__ = original_trainer
        MultiDataset.get = original_get
        (a.audit_output / 'resources.json').write_text(json.dumps(dict(native_cli_success=success,
            process_memory_fraction=a.gpu_memory_fraction, elapsed_seconds=time.monotonic() - start,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2), indent=2) + '\n')
    # run0 already finishes W&B. Preserve the successful native final files and
    # avoid the previously observed shutdown-time native GC/destructor path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
