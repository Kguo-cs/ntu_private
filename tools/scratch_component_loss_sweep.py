"""Train a fresh InitDiffusion, then evaluate using the repository pipeline.

Uses the repository's Lightning model, datamodule and Scenario Dreamer evaluator.
Training statistics must come from a separate training-only calibration pass.
All arms share a seeded full training order. Continuations restore optimizer,
scheduler, EMA and RNG state and start after previously consumed examples.
By default the map is also random/trainable. --map-checkpoint instead loads
only the strict map state and freezes it; it never resumes the source model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import pickle
import random
import resource
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf, open_dict

from tools.quick_component_loss_sweep import Diagnostics, save_json, summarize_trace

# Preserve the real collector before an opt-in isolated job defers automatic GC.
_ORIGINAL_GC_COLLECT = gc.collect

SEARCH_CONFIG_KEYS = {
    "model.model_config.optimizer", "model.model_config.lr",
    "model.model_config.lr_warmup_steps", "model.model_config.lr_min_ratio",
    *{"model.model_config.decoder.init_diffusion." + key for key in (
        "edge_embedding_type", "time_embedding_type", "time_embedding_scale",
        "count_embedding_type", "count_lane_source", "count_max_num_agents",
        "count_max_num_lanes", "ego_context_heading_encoding",
        "size_representation", "velocity_representation", "invalid_size_policy",
        "pos_source", "shape_source", "velocity_source", "speed_loss_weight",
        "speed_loss_scale", "ema_decay",
    )},
}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_hash(model, prefix):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(prefix) and torch.is_tensor(tensor):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_manifest(path, files):
    with path.open("wb") as handle:
        pickle.dump({"files": files}, handle)
    text_path = path.with_suffix(".txt")
    text_path.write_text("\n".join(files) + "\n")
    return hash_file(text_path)


def apply_variant(loss, changes):
    """Accept component overrides and optional global heading_space settings."""
    if changes is None:
        return None
    for name, value in changes.items():
        if name in ("__lr", "__config", "__clip"):
            continue
        if name in ("heading_space", "std_floor", "normalize_time_weights"):
            loss[name] = value
        elif name == "components":
            for component, options in value.items():
                for key, option in options.items():
                    loss.components[component][key] = option
        else:
            for key, option in value.items():
                loss.components[name][key] = option
    return loss


def calibrate(model, calibration, *, verify=False):
    denoiser = model.encoder.init_decoder.G1.model
    for name in ("normal_mean", "normal_scale"):
        value = torch.as_tensor(calibration[name], dtype=getattr(denoiser, name).dtype).reshape(1, -1)
        current = getattr(denoiser, name)
        if current.shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid training calibration {name}: {tuple(value.shape)}")
        if name == "normal_scale" and not (value > 0).all():
            raise ValueError("Training normal_scale must be strictly positive")
        if verify:
            if not torch.equal(current.cpu(), value):
                raise ValueError(f"Continuation/evaluation checkpoint has different {name}")
        else:
            current.copy_(value.to(current))
    if torch.all(denoiser.normal_mean == 0):
        raise ValueError("All-zero normal_mean would trigger lazy reinitialization")


class ScratchDiagnostics(Diagnostics):
    def __init__(self, checkpoint_steps, folder, metadata, resume_rng=None, runtime_memory=False,
                 young_gc_interval=0):
        super().__init__(checkpoint_steps, folder)
        self.metadata = metadata
        self.resume_rng = resume_rng
        self.runtime_memory = runtime_memory
        self.young_gc_interval = young_gc_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.young_gc_interval and trainer.global_step % self.young_gc_interval == 0:
            torch.cuda.synchronize()
            collected = _ORIGINAL_GC_COLLECT(0)
            print(f"RUNTIME_YOUNG_GC step={trainer.global_step} collected={collected}", flush=True)
        if self.runtime_memory and (trainer.global_step % 256 == 0 or trainer.global_step == 1):
            print("RUNTIME_MEMORY " + json.dumps({
                "step": trainer.global_step,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                "gc_counts": gc.get_count(),
            }), flush=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        for label, module in (("map", pl_module.encoder.map_encoder),
                              ("generator", pl_module.encoder.init_decoder.G1)):
            values = [parameter.grad.detach().float().norm() for parameter in module.parameters()
                      if parameter.grad is not None]
            self.current[f"{label}_grad_norm_before_clip"] = (
                float(torch.stack(values).norm().cpu()) if values else 0.0
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.young_gc_interval and (batch_idx + 1) % self.young_gc_interval == 0:
            torch.cuda.synchronize()
            collected = _ORIGINAL_GC_COLLECT(0)
            print(f"RUNTIME_YOUNG_GC test_batch={batch_idx + 1} collected={collected}", flush=True)

    def on_train_start(self, trainer, pl_module):
        if self.resume_rng is not None:
            # The checkpoint's optimizer step is global, but this rebuilt
            # dataloader starts at the first *unconsumed* scene. Lightning
            # adds current.ready to its fetch counter and would count old
            # batches against the shortened loader's length a second time.
            # Reset only loader-local batch progress; preserve optimizer
            # progress/global_step, epoch, scheduler and logging counters.
            trainer.fit_loop.epoch_loop.batch_progress.current.reset()
            trainer.fit_loop.epoch_loop.batch_progress.is_last_batch = False
            random.setstate(self.resume_rng["python"])
            np.random.set_state(self.resume_rng["numpy"])
            torch.set_rng_state(self.resume_rng["torch_cpu"])
            if self.resume_rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(self.resume_rng["torch_cuda"])
            print("RESTORED training RNG; reset sliced-loader local batch progress", flush=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["scratch_sweep"] = dict(self.metadata, saved_global_step=trainer.global_step)
        checkpoint["scratch_sweep_rng"] = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=512, help="Absolute final optimizer step")
    parser.add_argument("--checkpoint-steps", default="128,256", help="Absolute intermediate steps")
    parser.add_argument("--scenes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--defer-cyclic-gc", action="store_true",
                        default=os.environ.get("SIM_SWEEP_DEFER_CYCLIC_GC") == "1",
                        help="Isolated-job workaround: defer cyclic GC until process exit; reference counting remains active")
    parser.add_argument("--young-gc-interval", type=int,
                        default=int(os.environ.get("SIM_SWEEP_YOUNG_GC_INTERVAL", "0")),
                        help="Opt-in synchronized batch-boundary generation-0 collection; requires deferred automatic GC")
    parser.add_argument("--sharing-strategy", choices=("file_descriptor", "file_system"),
                        help="Optional explicit Torch CPU-tensor worker sharing strategy")
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--eval-seed", type=int, default=2817)
    parser.add_argument("--extra-eval-seed", type=int)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-manifest", type=Path,
                        help="JSON filename list (or {files: [...]}); overrides the sorted slice")
    parser.add_argument("--train-offset", type=int, default=0, help="Initial offset in seeded full order")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--map-checkpoint", type=Path,
                        help="Strict map-only startup; frozen map, fresh generator/optimizer/EMA")
    parser.add_argument("--variants", default="mixed")
    parser.add_argument("--variant-file", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--normalizer-json", "--calibration", dest="normalizer_json", type=Path, required=True)
    parser.add_argument("--evaluation-checkpoints", type=Path, help="JSON name -> saved scratch checkpoint")
    parser.add_argument("--resume-map", type=Path, help="JSON name -> scratch checkpoint to continue")
    parser.add_argument("--allow-multiple-epochs", action="store_true",
                        help="Fresh runs only: repeat the seeded full data order across real Lightning epochs")
    parser.add_argument("--evaluate-ema", action="store_true", help="Also evaluate fresh-history G1 EMA")
    parser.add_argument("--evaluation-weights", choices=("online", "ema", "both"),
                        help="Explicit evaluation weights; overrides the legacy --evaluate-ema switch")
    parser.add_argument("--require-full-set", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/component_loss_scratch" /
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return parser


def main():
    parser = parser_for_cli()
    args = parser.parse_args()
    if args.young_gc_interval < 0 or (args.young_gc_interval and not args.defer_cyclic_gc):
        parser.error("Young-generation GC interval must be nonnegative and requires --defer-cyclic-gc")
    if args.defer_cyclic_gc:
        gc.disable()
        # Some native objects in this environment segfault during cyclic GC.
        # Each sweep job is a separate process; reference counting still frees
        # ordinary tensors, and the operating system reclaims remaining cycles.
        gc.collect = lambda *unused_args, **unused_kwargs: 0
        if args.young_gc_interval:
            # Freeze the already imported static object graph once. Keep later
            # short-lived loader classes in generation 0 for safe-boundary GC.
            gc.freeze()
        print("RUNTIME cyclic GC deferred for this isolated process", flush=True)
    if args.sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    evaluation_modes = ({"online": [False], "ema": [True], "both": [False, True]}[args.evaluation_weights]
                        if args.evaluation_weights else ([False, True] if args.evaluate_ema else [False]))
    variants = json.loads(args.variant_file.read_text()) if args.variant_file else {"mixed": {}}
    eval_map = json.loads(args.evaluation_checkpoints.read_text()) if args.evaluation_checkpoints else {}
    resume_map = json.loads(args.resume_map.read_text()) if args.resume_map else {}
    names = [name for name in args.variants.split(",") if name]
    if set(names) - set(variants) or set(resume_map) - set(names):
        parser.error("Unknown training variant or resume-map name")
    if len(set(names + list(eval_map))) != len(names) + len(eval_map):
        parser.error("Training/evaluation names must be unique")
    if min(args.steps, args.scenes, args.batch_size, args.eval_batch_size) < 1:
        parser.error("Steps, scenes and batch sizes must be positive")
    if min(args.train_offset, args.eval_offset, args.workers) < 0:
        parser.error("Offsets and workers must be nonnegative")
    if not 0 < args.lr < 1:
        parser.error("Learning rate must be in (0,1)")
    if not 0 < args.gradient_clip < 1000:
        parser.error("Gradient clip must be in (0,1000)")
    checkpoint_steps = sorted({int(step) for step in args.checkpoint_steps.split(",") if step})
    if any(step <= 0 or step > args.steps for step in checkpoint_steps):
        parser.error("Checkpoint steps must be positive and at most --steps")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        parser.error("Output already contains protocol.json; choose a fresh directory")
    calibration = json.loads(args.normalizer_json.read_text())
    cache_root = ROOT / "src/waymo_data/full"
    with (cache_root / "training_map2_sd_files.pkl").open("rb") as handle:
        train_order = list(pickle.load(handle)["files"])
    random.Random(args.seed).shuffle(train_order)
    if args.allow_multiple_epochs and (resume_map or args.train_offset):
        raise ValueError("Multi-epoch training requires fresh weights and zero training offset")
    if (not args.allow_multiple_epochs
            and args.train_offset + args.steps * args.batch_size >= len(train_order)):
        raise ValueError("Quick budget must remain within one full-data epoch")
    order_hash = write_manifest(output / "train_order.pkl", train_order)
    if args.eval_manifest:
        if args.eval_offset:
            parser.error("--eval-manifest cannot be combined with --eval-offset")
        selected = json.loads(args.eval_manifest.read_text())
        eval_files = selected.get("files") if isinstance(selected, dict) else selected
        if (not isinstance(eval_files, list)
                or any(not isinstance(name, str) or Path(name).name != name
                       or not (cache_root / "scenario_dreamer_val" / name).is_file()
                       for name in eval_files)
                or len(set(eval_files)) != len(eval_files)):
            raise ValueError("Evaluation manifest must contain unique existing cache filenames")
    else:
        eval_files = sorted(path.name for path in (cache_root / "scenario_dreamer_val").glob("*.pt"))[
            args.eval_offset:args.eval_offset + args.scenes]
    if len(eval_files) != args.scenes:
        raise ValueError("Insufficient evaluation scenes")
    eval_hash = write_manifest(output / "eval_files.pkl", eval_files)

    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="run", overrides=["experiment=init_diffusion_lane_conditioned_mixed_loss"])
    with open_dict(cfg):
        cfg.action = "fit"
        cfg.ckpt_path = None
        cfg.seed = args.seed
        cfg.paths.root_dir = str(ROOT / "src")
        cfg.paths.output_dir = str(output)
        cfg.paths.log_dir = str(output)
        cfg.model.model_config.finetune = args.map_checkpoint is not None
        if args.map_checkpoint:
            cfg.model.model_config.initial_map_checkpoint = str(args.map_checkpoint.resolve())
        cfg.model.model_config.lr = args.lr
        cfg.model.model_config.lr_total_steps = 128
        if args.recipe:
            cfg.model.model_config.decoder.init_diffusion.component_loss = json.loads(args.recipe.read_text())
        # Quick sweeps score against their exposure/cohort-matched controls.
        cfg.model.model_config.sd_r_reference = None
        cfg.model.model_config.sd_reference_mode = "matched"
        cfg.model.model_config.sd_require_full_set = args.require_full_set
        cfg.model.model_config.token_processor.init_map_crop = "square"
        cfg.model.model_config.token_processor.init_map_half_extent = 50.0
        cfg.model.model_config.decoder.sep_map = False
        cfg.model.model_config.decoder.initial_scene_only = True
        cfg.data.train_batch_size = args.batch_size
        cfg.data.test_batch_size = args.eval_batch_size
        cfg.data.val_batch_size = args.eval_batch_size
        cfg.data.num_workers = args.workers
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        cfg.data.shuffle = False
        cfg.data.scenario_dreamer_eval_set = str(output / "eval_files.pkl")
    OmegaConf.resolve(cfg)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    torch.set_num_threads(4)
    initialization_scope = ("frozen_pretrained_map_fresh_G1" if args.map_checkpoint
                            else "map_encoder_and_G1_from_scratch")
    metadata = {"initialization": ("frozen pretrained map + random G1" if args.map_checkpoint
                                   else "random map encoder + G1"), "seed": args.seed,
                "batch_size": args.batch_size, "base_train_offset": args.train_offset,
                "train_order_sha256": order_hash, "calibration_sha256": hash_file(args.normalizer_json)}
    if args.map_checkpoint:
        metadata.update(map_checkpoint=str(args.map_checkpoint.resolve()),
                        map_checkpoint_sha256=hash_file(args.map_checkpoint),
                        map_frozen=True)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(metadata, initialization_scope=initialization_scope,
                    eval_manifest_sha256=eval_hash, map_crop="square", map_half_extent=50.0,
                    reference_mode="matched", evaluation_weight_modes=["ema" if mode else "online" for mode in evaluation_modes],
                    optimizer="AdamW fresh or continued",
                    precision="32-true", float32_matmul_precision=cfg.float32_matmul_precision,
                    scheduler=("Existing epoch-based cosine, 128 full-data epochs; fresh multi-epoch run counts real epochs"
                               if args.allow_multiple_epochs else
                               "Existing epoch-based cosine, 128 full-data epochs; quick stage stays within epoch 0"),
                    epoch_data_order=("The seeded full manifest repeats in the same order; shuffle=False"
                                      if args.allow_multiple_epochs else
                                      "Seeded full order, or its unconsumed suffix for first-epoch continuation; shuffle=False"),
                    resume_data_rule="base_train_offset + saved_global_step * batch_size; local batch progress reset; no replay",
                    resume_rng_rule="Checkpoint RNG restored at on_train_start; workers=0 recommended",
                    torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    shared_map_ema=False, normalizer=calibration)
    protocol["cyclic_gc_policy"] = ("deferred in isolated process; reference counting active"
                                    if args.defer_cyclic_gc else "Python default")
    if args.young_gc_interval:
        protocol["cyclic_gc_policy"] += f"; imports frozen once; synchronized generation-0 collection every {args.young_gc_interval} TRAIN steps"
    save_json(output / "protocol.json", protocol)
    OmegaConf.save(cfg, output / "base_config.yaml")
    shared_evaluator = None
    summary = []

    def flush_summary():
        save_json(output / "summary.json", summary)
        fields = list(dict.fromkeys(key for row in summary for key in row))
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

    def evaluate(model, dm, folder, seed, *, ema=False):
        nonlocal shared_evaluator
        folder.mkdir(parents=True, exist_ok=True)
        model.video_dir = folder / "videos"
        model.video_dir.mkdir(exist_ok=True)
        model.encoder.init_decoder.use_ema = ema
        model.sd_evaluator = shared_evaluator
        L.seed_everything(seed, workers=True)
        trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true", logger=False,
                            enable_checkpointing=False, enable_model_summary=False,
                            enable_progress_bar=False,
                            callbacks=[ScratchDiagnostics([], folder, {}, runtime_memory=args.defer_cyclic_gc,
                                                          young_gc_interval=args.young_gc_interval)],
                            default_root_dir=str(folder))
        started = time.monotonic()
        trainer.test(model, datamodule=dm, verbose=False)
        shared_evaluator = model.sd_evaluator
        report = json.loads((folder / "sd_agent_metrics.json").read_text())
        if report["num_samples"] != args.scenes or report["num_gt_samples"] != args.scenes:
            raise ValueError("Evaluation scope differs from protocol")
        generated_onroad = int(shared_evaluator.generated.totals[1])
        gt_onroad = int(shared_evaluator.real.totals[1])
        report.update(num_generated_onroad_vehicles=generated_onroad, num_gt_onroad_vehicles=gt_onroad,
                      generated_onroad_fraction=generated_onroad / report["num_generated_vehicles"],
                      gt_onroad_fraction=gt_onroad / report["num_gt_vehicles"])
        shared_evaluator.reference_mode = "full"
        report["full_reference_agent_metrics"] = {key: float(value) for key, value in shared_evaluator.compute().items()}
        shared_evaluator.reference_mode = "matched"
        from src.smart.metrics.metric_core import SPECS
        save_json(folder / "feature_histograms.json", {
            name: {"minimum": lo, "maximum": hi, "bin_width": step,
                   "generated": shared_evaluator.generated.histograms[i].tolist(),
                   "matched_gt": shared_evaluator.real.histograms[i].tolist(),
                   "full_gt": shared_evaluator.store.full_real.histograms[i].tolist()}
            for i, (name, lo, hi, step, _) in enumerate(SPECS)
        })
        save_json(folder / "sd_agent_metrics.json", report)
        del trainer
        return report, time.monotonic() - started

    for name in names + list(eval_map):
        print(f"\nSTART {name}", flush=True)
        folder = output / name
        folder.mkdir()
        arm_cfg = copy.deepcopy(cfg)
        source_checkpoint = Path(eval_map.get(name, resume_map.get(name, ""))) if name in eval_map or name in resume_map else None
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False) if source_checkpoint else None
        saved_step = int(checkpoint["global_step"]) if checkpoint else 0
        clip = args.gradient_clip
        with open_dict(arm_cfg):
            arm_cfg.paths.output_dir = str(folder)
            if source_checkpoint and (source_checkpoint.parent / "config.yaml").is_file():
                source_cfg = OmegaConf.load(source_checkpoint.parent / "config.yaml")
                # Restore all model settings, including representation, edge
                # embeddings and optimizer recipe, for exact continuation.
                arm_cfg.model = copy.deepcopy(source_cfg.model)
                # Evaluation membership strictness belongs to the current
                # invocation, not the training checkpoint's monitoring scope.
                arm_cfg.model.model_config.sd_reference_mode = "matched"
                arm_cfg.model.model_config.sd_require_full_set = args.require_full_set
                clip = float(source_cfg.trainer.get("gradient_clip_val", args.gradient_clip))
            elif name not in eval_map:
                arm_cfg.model.model_config.decoder.init_diffusion.component_loss = apply_variant(
                    arm_cfg.model.model_config.decoder.init_diffusion.component_loss, variants[name])
                if variants[name] is not None and "__lr" in variants[name]:
                    arm_cfg.model.model_config.lr = float(variants[name]["__lr"])
                if variants[name] is not None:
                    clip = float(variants[name].get("__clip", clip))
                    for key, value in variants[name].get("__config", {}).items():
                        if key not in SEARCH_CONFIG_KEYS:
                            raise ValueError(f"Variant __config field is not an authorized search parameter: {key}")
                        OmegaConf.update(arm_cfg, key, value, force_add=False)
            if not 0 < clip < 1000:
                raise ValueError("Variant gradient clip must be in (0,1000)")
            arm_cfg.trainer.gradient_clip_val = clip
        offset = (args.train_offset if name in eval_map else
                  args.train_offset + saved_step * args.batch_size)
        arm_train = train_order[offset:]
        train_hash = write_manifest(folder / "train_files.pkl", arm_train)
        with open_dict(arm_cfg):
            arm_cfg.data.scenario_dreamer_train_sample_list = str(folder / "train_files.pkl")
        OmegaConf.save(arm_cfg, folder / "config.yaml")
        L.seed_everything(args.seed, workers=True)
        dm = hydra.utils.instantiate(arm_cfg.data)
        model = hydra.utils.instantiate(arm_cfg.model, _recursive_=False)
        initial_hashes = {"initial_generator_sha256": weights_hash(model, "encoder.init_decoder.G1."),
                          "initial_map_sha256": weights_hash(model, "encoder.map_encoder.")}
        if args.map_checkpoint:
            if model.initial_map_checkpoint_metadata["file_sha256"] != metadata["map_checkpoint_sha256"]:
                raise ValueError("Actual fixed map source differs from the invocation protocol")
            if any(parameter.requires_grad for parameter in model.encoder.map_encoder.parameters()):
                raise ValueError("Pretrained map must remain frozen")
            model.train()
            if model.encoder.map_encoder.training:
                raise ValueError("Frozen pretrained map must stay in evaluation mode")
        elif not all(parameter.requires_grad for parameter in model.encoder.map_encoder.parameters()):
            raise ValueError("Scratch map encoder unexpectedly frozen")
        if not all(parameter.requires_grad for parameter in model.encoder.init_decoder.G1.parameters()):
            raise ValueError("Scratch generator unexpectedly frozen")
        if any(parameter.requires_grad for parameter in model.encoder.agent_encoder.parameters()):
            raise ValueError("Unused agent policy must be frozen in scratch initialization")
        calibrate(model, calibration)
        resume_rng = None
        if checkpoint:
            saved_metadata = checkpoint.get("scratch_sweep")
            if saved_metadata is None:
                raise ValueError("Only checkpoints produced by scratch sweep may be loaded")
            for key in metadata:
                if saved_metadata.get(key) != metadata[key]:
                    raise ValueError(f"Scratch checkpoint protocol mismatch: {key}")
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
            calibrate(model, calibration, verify=True)
            print(f"LOADED scratch checkpoint step={saved_step}, {incompatible}", flush=True)
            resume_rng = checkpoint.get("scratch_sweep_rng")
            if name in resume_map and resume_rng is None:
                raise ValueError("Scratch continuation requires recorded training RNG state")
        frozen_map_hash = weights_hash(model, "encoder.map_encoder.") if args.map_checkpoint else None
        if args.map_checkpoint and initial_hashes["initial_map_sha256"] != frozen_map_hash:
            raise ValueError("Saved training checkpoint changed the fixed source map")
        initialization = dict(metadata, **initial_hashes, train_manifest_sha256=train_hash,
                              eval_manifest_sha256=eval_hash, lr=float(arm_cfg.model.model_config.lr),
                              gradient_clip_val=clip,
                              initialization_scope=initialization_scope,
                              actual_train_offset=None if name in eval_map else offset,
                              resumed_from=str(source_checkpoint) if source_checkpoint else None,
                              resume_global_step=saved_step, optimizer_continued=name in resume_map,
                              resume_local_batch_counter_reset=name in resume_map)
        save_json(folder / "initialization.json", initialization)
        diagnostic = ScratchDiagnostics(sorted(set(checkpoint_steps + [args.steps])), folder,
                                        dict(metadata, **initial_hashes), resume_rng,
                                        runtime_memory=args.defer_cyclic_gc,
                                        young_gc_interval=args.young_gc_interval)
        diagnostic.install(model)
        train_seconds = 0.0
        trained_paths = []
        if name not in eval_map:
            if saved_step >= args.steps:
                raise ValueError("Continuation checkpoint must precede requested final --steps")
            trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true", max_epochs=128,
                                max_steps=args.steps, limit_val_batches=0, num_sanity_val_steps=0,
                                gradient_clip_val=clip, enable_checkpointing=False, enable_model_summary=False,
                                enable_progress_bar=False, log_every_n_steps=1,
                                logger=CSVLogger(str(folder), name="csv"), default_root_dir=str(folder),
                                callbacks=[diagnostic])
            started = time.monotonic()
            trainer.fit(model, datamodule=dm, ckpt_path=str(source_checkpoint) if source_checkpoint else None)
            train_seconds = time.monotonic() - started
            if trainer.global_step != args.steps:
                raise ValueError(f"Training stopped at {trainer.global_step}, expected {args.steps}")
            if args.map_checkpoint:
                if weights_hash(model, "encoder.map_encoder.") != frozen_map_hash:
                    raise ValueError("Frozen map state changed during training")
                if any(parameter.grad is not None for parameter in model.encoder.map_encoder.parameters()):
                    raise ValueError("Frozen map unexpectedly accumulated gradients")
            # Preserve the post-update batch-end checkpoint. Lightning advances
            # epoch schedulers at actual loader boundaries, before this hook.
            # Fit teardown can mark an unfinished epoch as processed; retaining
            # batch-end state preserves that epoch's progress for continuation.
            shutil.copyfile(folder / f"step_{args.steps}.ckpt", folder / "last.ckpt")
            trained_paths = [(step, folder / f"step_{step}.ckpt") for step in checkpoint_steps
                             if saved_step < step <= args.steps and step != args.steps]
            trained_paths.append((args.steps, folder / "last.ckpt"))
            save_json(folder / "train_trace.json", diagnostic.records)
            trace_summary = summarize_trace(diagnostic.records)
            trace_summary.update(gradient_clip_val=clip,
                                 clip_fraction=float(np.mean([row["grad_norm_before_clip"] > clip
                                                             for row in diagnostic.records])))
            save_json(folder / "train_summary.json", trace_summary)
            del trainer
        else:
            trained_paths = [(saved_step, source_checkpoint)]
        del model, checkpoint, diagnostic
        gc.collect()
        torch.cuda.empty_cache()
        for budget, path in trained_paths:
            # Evaluation never mutates saved optimizer state or training RNG.
            L.seed_everything(args.seed, workers=True)
            model = hydra.utils.instantiate(arm_cfg.model, _recursive_=False)
            stage_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(stage_checkpoint["state_dict"], strict=True)
            calibrate(model, calibration, verify=True)
            del stage_checkpoint
            for eval_seed in [args.eval_seed] + ([args.extra_eval_seed] if args.extra_eval_seed is not None else []):
                for ema in evaluation_modes:
                    stage_folder = folder / f"eval_step_{budget}"
                    if eval_seed != args.eval_seed:
                        stage_folder = stage_folder / f"eval_seed_{eval_seed}"
                    if ema:
                        stage_folder = stage_folder / "ema"
                    report, seconds = evaluate(model, dm, stage_folder, eval_seed, ema=ema)
                    row = {
                        **initialization, "variant": name, "budget": budget, "seed": args.seed,
                        # The native loader retains each epoch's final partial
                        # batch. Count those actual scene exposures, not step*B.
                        "seen_training_examples": (
                            (budget // ((len(train_order) + args.batch_size - 1) // args.batch_size)) * len(train_order)
                            + (budget % ((len(train_order) + args.batch_size - 1) // args.batch_size)) * args.batch_size),
                        "eval_seed": eval_seed, "evaluation_only": name in eval_map,
                        "evaluation_weights": "ema" if ema else "online", "checkpoint": str(path.resolve()),
                        "metrics_path": str((stage_folder / "sd_agent_metrics.json").resolve()),
                        "train_seconds": train_seconds, "eval_seconds": seconds,
                        **{key: float(value) for key, value in report["agent_metrics"].items()},
                        **{key: report[key] for key in ("num_samples", "num_generated_vehicles", "num_gt_vehicles",
                            "num_generated_onroad_vehicles", "num_gt_onroad_vehicles", "generated_onroad_fraction",
                            "gt_onroad_fraction", "num_colliding_vehicles", "full_reference_agent_metrics")},
                    }
                    summary.append(row)
                    flush_summary()
                    print(f"RESULT {json.dumps(row)}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del dm
        gc.collect()
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
    if os.environ.get("SIM_SWEEP_DEFER_CYCLIC_GC") == "1" or "--defer-cyclic-gc" in sys.argv:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
