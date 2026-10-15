"""Evaluate genuine full-training checkpoints with the native test pipeline."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--sampler', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--integration', choices=['native', 'midpoint', 'ralston'], default='native')
    parser.add_argument('--seed', type=int, default=2817)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--full-set', action='store_true')
    parser.add_argument('--eval-matmul-precision', choices=['highest', 'high', 'medium'], default='highest')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['WANDB_MODE'] = 'disabled'
    gc.disable()
    gc.collect = lambda *a, **kw: 0
    # Load this pure Python dependency before native extensions. PyG imports it
    # lazily; keep dependency initialization outside the seeded model execution.
    import jinja2
    import torch
    import hydra
    import lightning as L
    from omegaconf import OmegaConf, open_dict
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    import src.smart.datamodules
    import src.smart.model.smart_gail
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    cfg = OmegaConf.load(args.source_config)
    sampler = json.loads(args.sampler.read_text())
    names = json.loads(args.manifest.read_text())
    if isinstance(names, dict):
        names = names['files']
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected a unique filename manifest')
    if args.full_set and len(names) != 50000:
        raise ValueError('Full evaluation requires all50000 scenes')
    selected = args.output / 'eval_files.pkl'
    with selected.open('xb') as f:
        pickle.dump({'files': names}, f)
    if args.integration == 'midpoint' and (sampler['sampling_steps'] != 12
            or sampler['sampling_time_grid'].get('terminal_time') != .1):
        raise ValueError('Midpoint requires12 intervals and final network time=.05')
    if args.integration == 'ralston' and (sampler['sampling_steps'] != 12
            or sampler['sampling_time_grid'].get('terminal_time', 0) < .15):
        raise ValueError('Ralston requires12 intervals and terminal_time>=.15')
    with open_dict(cfg):
        cfg.action = 'test'
        cfg.ckpt_path = str(args.checkpoint.resolve())
        cfg.seed = args.seed
        cfg.float32_matmul_precision = args.eval_matmul_precision
        cfg.paths.root_dir = str(ROOT)
        cfg.paths.output_dir = str(args.output.resolve())
        cfg.paths.log_dir = str(args.output.resolve())
        cfg.paths.work_dir = str(ROOT)
        cfg.callbacks = None
        cfg.logger = False
        mc = cfg.model.model_config
        mc.initial_map_checkpoint = None
        mc.sd_r_reference = None
        mc.sd_require_full_set = args.full_set
        # The cache retains the complete official catalogue. The dataset selects
        # the generated subset, and reference_mode='matched' sums its GT only.
        mc.sd_expected_scenes = 50000
        mc.sd_reference_mode = 'matched'
        mc.decoder.init_diffusion.normalizer_path = None
        for key, value in sampler.items():
            OmegaConf.update(cfg, 'model.model_config.decoder.init_diffusion.' + key, value, force_add=True)
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        cfg.data.test_batch_size = args.batch_size
        cfg.data.scenario_dreamer_eval_set = str(selected.resolve())
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = False
    OmegaConf.save(cfg, args.output / 'config.yaml', resolve=False)
    checkpoint_hash = sha(args.checkpoint)
    checkpoint_metadata = json.loads(args.checkpoint.with_name('provenance.json').read_text())
    assert checkpoint_metadata['sha256'] == checkpoint_hash
    checkpoint_epoch = checkpoint_metadata['epoch']
    checkpoint_step = checkpoint_metadata['global_step']
    provenance = dict(based_on_runner_sha256=sha(ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'),
                      experimental_integrator_source_sha256=sha(ROOT / 'tools/rk2_sampling_20261015.py'),
                      runtime_override='Experimental research wrapper; native config retains Euler base.',
                      checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                      checkpoint_epoch=checkpoint_epoch, checkpoint_step=checkpoint_step,
                      output=str(args.output.resolve()),
                      source_config_sha256=sha(args.source_config), manifest_sha256=sha(args.manifest),
                      sampler=sampler, seed=args.seed, scenes=len(names), integration=args.integration,
                      network_evaluations_per_scene=24 if args.integration in ('midpoint', 'ralston') else sampler['sampling_steps'],
                      runner_sha256=sha(__file__), matmul_precision=args.eval_matmul_precision,
                      smart_source_sha256={str(p.relative_to(ROOT)): sha(p)
                                           for p in sorted((ROOT / 'src/smart').rglob('*.py'))},
                      checkpoint_origin='Full training; no fabricated scratch metadata')
    (args.output / 'execution.json').write_text(json.dumps(provenance, indent=2) + '\n')
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.eval_matmul_precision)
    torch.set_num_threads(4)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.video_dir = args.output.resolve() / 'videos'
    model.video_dir.mkdir(exist_ok=True)
    flow = model.encoder.init_decoder.G1
    if args.integration == 'midpoint':
        from src.smart.diffusion.midpoint_sampling import install_midpoint
        install_midpoint(flow)
    elif args.integration == 'ralston':
        from tools.rk2_sampling_20261015 import install_ralston
        install_ralston(flow)
    original_sample = flow.sample
    denoiser_calls = [0]
    sample_call_counts = []
    source_hashes = []

    def count_call(*unused):
        denoiser_calls[0] += 1

    counter_hook = flow.model.register_forward_pre_hook(count_call)

    def sample(self, *a, **kw):
        steps = kw.get('steps', a[2] if len(a) > 2 else 20)
        if steps != sampler['sampling_steps']:
            raise ValueError('Sampler configuration did not reach the native sample call')
        before = denoiser_calls[0]
        result = original_sample(*a, **kw)
        actual = denoiser_calls[0] - before
        expected = steps * (2 if args.integration in ('midpoint', 'ralston') else 1)
        if actual != expected:
            raise RuntimeError(f'Expected {expected} denoiser calls, observed {actual}')
        sample_call_counts.append(actual)
        agent = kw.get('tokenized_agent', a[0] if a else None)
        digest = hashlib.sha256()
        for key in ('gen_noise', '_init_diffusion_type_source'):
            value = agent[key].detach().cpu().contiguous()
            digest.update(str((key, value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        source_hashes.append(digest.hexdigest())
        return result

    flow.sample = MethodType(sample, flow)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=False,
                                    enable_checkpointing=False)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=str(args.checkpoint.resolve()))
    counter_hook.remove()
    report = json.loads((args.output / 'sd_agent_metrics.json').read_text())
    if args.integration == 'ralston':
        report['decoder'].update(sampling_integrator='experimental_ralston',
            sampling_network_evaluations=24, experimental_integrator_source_sha256=sha(ROOT / 'tools/rk2_sampling_20261015.py'),
            integration_stage_fraction=2/3, integration_weights=[.25,.75])
        (args.output / 'sd_agent_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['num_samples'] == report['num_gt_samples'] == len(names)
    assert report['reference_mode'] == 'matched'
    assert report['decoder']['ema_num_updates'] == checkpoint_step
    assert sha(args.checkpoint) == checkpoint_hash
    from src.smart.metrics.metric_core import SPECS, finalize_metrics
    import numpy as np
    evaluator = model.sd_evaluator
    recomputed = finalize_metrics(evaluator.generated, evaluator.real)
    assert all(abs(report['agent_metrics'][key] - value) < 1e-10
               for key, value in recomputed.items())
    histograms = {}
    for i, (name, lo, hi, step, multiplier) in enumerate(SPECS):
        histograms[name] = dict(edges=np.arange(lo, hi + step, step).tolist(),
                               generated=evaluator.generated.histograms[i].tolist(),
                               matched_gt=evaluator.real.histograms[i].tolist(),
                               multiplier=multiplier)
    (args.output / 'feature_histograms.json').write_text(json.dumps(histograms, indent=2) + '\n')
    assert sample_call_counts
    provenance.update(status='FULL_CHECKPOINT_EVAL_COMPLETE', metrics=report['agent_metrics'],
                      observed_denoiser_calls_per_sample_batch=sample_call_counts,
                      sampling_source_sha256_per_batch=source_hashes,
                      metrics_recomputed_from_integer_histograms=True)
    (args.output / 'verified.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('FULL_CHECKPOINT_EVAL_COMPLETE ' + json.dumps(report['agent_metrics']), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
