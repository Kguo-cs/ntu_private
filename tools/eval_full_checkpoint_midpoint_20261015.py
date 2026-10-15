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
    parser.add_argument('--integration', choices=['native', 'midpoint'], default='native')
    parser.add_argument('--seed', type=int, default=2817)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--full-set', action='store_true')
    parser.add_argument('--r-reference', type=Path)
    parser.add_argument('--eval-matmul-precision', choices=['highest', 'high', 'medium'], default='highest')
    parser.add_argument('--weights', choices=['ema', 'raw'], default=None,
                        help='Override saved EMA selection for a paired diagnostic; default preserves config.')
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
        mc.sd_r_reference = None if args.r_reference is None else str(args.r_reference.resolve())
        mc.sd_require_full_set = args.full_set
        # The cache retains the complete official catalogue. The dataset selects
        # the generated subset, and reference_mode='matched' sums its GT only.
        mc.sd_expected_scenes = 50000
        mc.sd_reference_mode = 'matched'
        mc.decoder.init_diffusion.normalizer_path = None
        for key, value in sampler.items():
            OmegaConf.update(cfg, 'model.model_config.decoder.init_diffusion.' + key, value, force_add=True)
        if args.weights is not None:
            mc.decoder.init_diffusion.use_ema = args.weights == 'ema'
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        cfg.data.test_batch_size = args.batch_size
        cfg.data.scenario_dreamer_eval_set = str(selected.resolve())
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = False
    OmegaConf.save(cfg, args.output / 'config.yaml', resolve=False)
    configured_integrator = mc.decoder.init_diffusion.get('sampling_integrator', 'euler')
    midpoint_selected = args.integration == 'midpoint' or configured_integrator == 'midpoint'
    evaluation_multiplier = 2 if midpoint_selected else 1
    checkpoint_hash = sha(args.checkpoint)
    checkpoint_metadata = json.loads(args.checkpoint.with_name('provenance.json').read_text())
    assert checkpoint_metadata['sha256'] == checkpoint_hash
    checkpoint_epoch = checkpoint_metadata['epoch']
    checkpoint_step = checkpoint_metadata['global_step']
    provenance = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                      checkpoint_epoch=checkpoint_epoch, checkpoint_step=checkpoint_step,
                      output=str(args.output.resolve()),
                      source_config_sha256=sha(args.source_config), manifest_sha256=sha(args.manifest),
                      sampler=sampler, seed=args.seed, scenes=len(names), integration=args.integration,
                      network_evaluations_per_scene=sampler['sampling_steps'] * evaluation_multiplier,
                      effective_integrator='midpoint' if midpoint_selected else 'euler',
                      runner_sha256=sha(__file__), matmul_precision=args.eval_matmul_precision,
                      r_reference_sha256=None if args.r_reference is None else sha(args.r_reference),
                      smart_source_sha256={str(p.relative_to(ROOT)): sha(p)
                                           for p in sorted((ROOT / 'src/smart').rglob('*.py'))},
                      checkpoint_origin='Full training; no fabricated scratch metadata')
    provenance['weights_mode'] = 'ema' if mc.decoder.init_diffusion.get('use_ema', False) else 'raw'
    expected_weight_digest = None
    if args.weights is not None:
        from tools.checkpoint_weight_audit_20261015 import checkpoint_parameter_digest
        expected_weight_digest = checkpoint_parameter_digest(args.checkpoint, args.weights)
        provenance['expected_evaluated_parameter_sha256'] = expected_weight_digest
    (args.output / 'execution.json').write_text(json.dumps(provenance, indent=2) + '\n')
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.eval_matmul_precision)
    torch.set_num_threads(4)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.video_dir = args.output.resolve() / 'videos'
    model.video_dir.mkdir(exist_ok=True)
    flow = model.encoder.init_decoder.G1
    if args.integration == 'midpoint' and getattr(flow, 'sampling_integrator', 'euler') != 'midpoint':
        from src.smart.diffusion.midpoint_sampling import install_midpoint
        install_midpoint(flow)
    original_sample = flow.sample
    denoiser_calls = [0]
    sample_call_counts = []
    source_hashes = []
    observed_weight_digests = []

    def count_call(*unused):
        if expected_weight_digest is not None and not observed_weight_digests:
            from tools.checkpoint_weight_audit_20261015 import parameter_digest
            observed = parameter_digest(flow.named_parameters())
            if observed != expected_weight_digest:
                raise RuntimeError('Actual first denoiser call did not use the requested checkpoint weights')
            observed_weight_digests.append(observed)
        denoiser_calls[0] += 1

    counter_hook = flow.model.register_forward_pre_hook(count_call)

    def sample(self, *a, **kw):
        steps = kw.get('steps', a[2] if len(a) > 2 else 20)
        if steps != sampler['sampling_steps']:
            raise ValueError('Sampler configuration did not reach the native sample call')
        before = denoiser_calls[0]
        result = original_sample(*a, **kw)
        actual = denoiser_calls[0] - before
        expected = steps * evaluation_multiplier
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
    assert report['num_samples'] == report['num_gt_samples'] == len(names)
    assert report['reference_mode'] == 'matched'
    assert report['decoder']['ema_num_updates'] == checkpoint_step
    assert report['decoder']['use_ema'] == (provenance['weights_mode'] == 'ema')
    if expected_weight_digest is not None:
        assert observed_weight_digests == [expected_weight_digest]
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
    provenance['observed_evaluated_parameter_sha256'] = (
        observed_weight_digests[0] if observed_weight_digests else None)
    (args.output / 'verified.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('FULL_CHECKPOINT_EVAL_COMPLETE ' + json.dumps(report['agent_metrics']), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
