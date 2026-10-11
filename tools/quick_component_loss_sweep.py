"""Controlled short InitDiffusion fine-tuning with the existing Lightning pipeline.

Run with the sim Python. Uses existing data, model, optimizer and SD evaluator;
only manifests, loss settings and experiment logging are supplied here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import pickle
import random
import sys
import time
import types
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

WARM_START = ROOT / "src/logs/sdinit512_tspeed_theta_posscale_egoVFtypeemd50/2026-10-09_19-24-16/bc/scy2l1f4/checkpoints/last.ckpt"
VARIANTS = {
    "mixed": {},
    "heading_mse": {"heading": {"kind": "mse"}},
    "position_mse": {"position": {"kind": "mse"}},
    "scalars_l1": {name: {"kind": "l1"} for name in ("length", "width", "velocity")},
    "scalars_p1": {name: {"time_power": 1} for name in ("length", "width", "velocity")},
    "scalars_p3": {name: {"time_power": 3} for name in ("length", "width", "velocity")},
    "heading_w4": {"heading": {"weight": 4.0}},
    "collision_w05": {"collision": {"weight": 0.5}},
    "legacy": None,
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Diagnostics(L.Callback):
    def __init__(self, checkpoint_steps=(), folder=None):
        self.records = []
        self.current = {}
        self.started = time.monotonic()
        self.checkpoint_steps = set(checkpoint_steps)
        self.folder = folder

    def install(self, model):
        original = model._log_train

        def capture(instance, name, value):
            if torch.is_tensor(value) and value.numel():
                self.current[name] = float(value.detach().float().mean().cpu())
            elif isinstance(value, (int, float)):
                self.current[name] = float(value)
            original(name, value)

        model._log_train = types.MethodType(capture, model)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.current = {"step": trainer.global_step}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = [p.grad.detach().float().norm() for group in optimizer.param_groups
                 for p in group["params"] if p.grad is not None]
        self.current["grad_norm_before_clip"] = float(torch.stack(norms).norm().cpu()) if norms else 0.0
        self.current["learning_rate"] = float(optimizer.param_groups[0]["lr"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self.current["optimized_loss"] = float(outputs["loss"].detach().cpu())
        self.records.append(dict(self.current))
        if trainer.global_step in self.checkpoint_steps:
            trainer.save_checkpoint(self.folder / f"step_{trainer.global_step}.ckpt")
        if trainer.global_step % 16 == 0 or trainer.global_step == 1:
            print(f"STEP {trainer.global_step}: loss={self.current.get('optimized_loss', float('nan')):.4f}, "
                  f"grad={self.current.get('grad_norm_before_clip', 0):.3f}, "
                  f"elapsed={time.monotonic() - self.started:.1f}s", flush=True)

    def on_test_epoch_start(self, trainer, pl_module):
        self.eval_started = time.monotonic()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (batch_idx + 1) % 64 == 0:
            scenes = len(pl_module.sd_evaluator.seen)
            print(f"EVAL {scenes} scenes, elapsed={time.monotonic() - self.eval_started:.1f}s", flush=True)


def summarize_trace(records):
    def means(rows):
        keys = sorted(set().union(*(r.keys() for r in rows)))
        return {key: float(np.mean([r[key] for r in rows if key in r])) for key in keys if key != "step"}
    window = min(32, len(records))
    return {"steps": len(records), "first_window": means(records[:window]),
            "last_window": means(records[-window:]),
            "clip_fraction": float(np.mean([r["grad_norm_before_clip"] > 0.5 for r in records]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--scenes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--eval-seed", type=int, default=2817)
    parser.add_argument("--extra-eval-seed", type=int,
                        help="Optional second generation seed for the same trained weights")
    parser.add_argument("--eval-use-ema", action="store_true", help="Evaluate saved EMA instead of online weights")
    parser.add_argument("--eval-offset", type=int, default=0,
                        help="Official-order offset; use disjoint scenes for final confirmation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=Path, default=WARM_START)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--recipe", type=Path, help="JSON component_loss mapping for adaptive search")
    parser.add_argument("--variant-file", type=Path, help="JSON name -> component overrides (null means legacy)")
    parser.add_argument("--evaluation-checkpoints", type=Path,
                        help="JSON name -> checkpoint path; evaluate these without training")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "logs/component_loss_quick" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--require-full-set", action="store_true")
    parser.add_argument("--checkpoint-steps", default="", help="Comma-separated intermediate optimizer steps to save")
    args = parser.parse_args()
    variants = json.loads(args.variant_file.read_text()) if args.variant_file else VARIANTS
    evaluation_checkpoints = (json.loads(args.evaluation_checkpoints.read_text())
                              if args.evaluation_checkpoints else {})
    names = args.variants.split(",") if args.variants else []
    if set(names) - set(variants):
        parser.error("Unknown variants")
    if min(args.steps, args.scenes, args.batch_size, args.eval_batch_size) < 1:
        parser.error("steps, scenes and batch sizes must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        parser.error("Output already contains a protocol; use a new directory")

    # Fixed random training subset/order; fixed official-order validation subset.
    cache_root = ROOT / "src/waymo_data/full"
    with (cache_root / "training_map2_sd_files.pkl").open("rb") as f:
        all_train = pickle.load(f)["files"]
    train_files = random.Random(args.seed).sample(all_train, args.steps * args.batch_size)
    if args.eval_offset < 0:
        parser.error("eval-offset must be nonnegative")
    eval_files = sorted(p.name for p in (cache_root / "scenario_dreamer_val").glob("*.pt"))[
        args.eval_offset:args.eval_offset + args.scenes]
    if len(eval_files) != args.scenes:
        raise ValueError("Insufficient evaluation files")
    for label, files in (("train", train_files), ("eval", eval_files)):
        with (output / f"{label}_files.pkl").open("wb") as f:
            pickle.dump({"files": files}, f)
        (output / f"{label}_files.txt").write_text("\n".join(files) + "\n")

    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="run", overrides=["experiment=init_diffusion_lane_conditioned_mixed_loss"])
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.ckpt_path = str(args.checkpoint.resolve())
        cfg.paths.root_dir = str(ROOT / "src")
        cfg.paths.output_dir = str(output)
        cfg.paths.log_dir = str(output)
        cfg.model.model_config.lr = args.lr
        cfg.model.model_config.lr_total_steps = 128  # Preserve the epoch LR schedule, not a one-epoch decay.
        if args.recipe:
            cfg.model.model_config.decoder.init_diffusion.component_loss = json.loads(args.recipe.read_text())
        cfg.model.model_config.sd_reference_mode = "matched"
        cfg.model.model_config.sd_require_full_set = args.require_full_set
        # Match this checkpoint's training map condition in every arm.
        cfg.model.model_config.token_processor.init_map_crop = "square"
        cfg.model.model_config.token_processor.init_map_half_extent = 50.0
        cfg.data.train_batch_size = args.batch_size
        cfg.data.test_batch_size = args.eval_batch_size
        cfg.data.val_batch_size = args.eval_batch_size
        cfg.data.num_workers = args.workers
        cfg.data.persistent_workers = False
        cfg.data.shuffle = False
        cfg.data.scenario_dreamer_train_sample_list = str(output / "train_files.pkl")
        cfg.data.scenario_dreamer_eval_set = str(output / "eval_files.pkl")
    OmegaConf.resolve(cfg)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    torch.set_num_threads(4)
    # No optimizer resume: identical online weights, statistics and fresh AdamW.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    weight_hash = hashlib.sha256()
    for key, value in sorted(state.items()):
        if torch.is_tensor(value) and key.startswith("encoder.init_decoder.G1."):
            weight_hash.update(key.encode())
            weight_hash.update(value.contiguous().numpy().tobytes())
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(warm_start_epoch=checkpoint["epoch"], warm_start_step=checkpoint["global_step"],
                    initial_generator_sha256=weight_hash.hexdigest(),
                    train_manifest_sha256=hashlib.sha256((output / "train_files.txt").read_bytes()).hexdigest(),
                    eval_manifest_sha256=hashlib.sha256((output / "eval_files.txt").read_bytes()).hexdigest(),
                    map_crop="square", map_half_extent=50.0, reference_mode="matched",
                    evaluation_weights="ema" if args.eval_use_ema else "online",
                    optimizer="fresh AdamW", precision="32-true",
                    float32_matmul_precision=cfg.float32_matmul_precision,
                    torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    steps_are_per_variant=True)
    protocol["evaluation_checkpoint_paths"] = evaluation_checkpoints
    save_json(output / "protocol.json", protocol)
    OmegaConf.save(cfg, output / "base_config.yaml")
    shared_evaluator = None
    summary = []

    def evaluate(model, trainer, dm, folder, seed=None):
        nonlocal shared_evaluator
        model.encoder.init_decoder.use_ema = args.eval_use_ema
        model.sd_evaluator = shared_evaluator
        L.seed_everything(args.eval_seed if seed is None else seed, workers=True)
        started = time.monotonic()
        trainer.test(model, datamodule=dm, verbose=False)
        shared_evaluator = model.sd_evaluator
        report = json.loads((folder / "sd_agent_metrics.json").read_text())
        if report["num_samples"] != args.scenes or report["num_gt_samples"] != args.scenes:
            raise ValueError("Evaluation scope differs from protocol")
        # Secondary full-50k reference, same generated samples, no resampling.
        shared_evaluator.reference_mode = "full"
        report["full_reference_agent_metrics"] = {k: float(v) for k, v in shared_evaluator.compute().items()}
        shared_evaluator.reference_mode = "matched"
        from src.smart.metrics.metric_core import SPECS
        save_json(folder / "feature_histograms.json", {
            name: {
                "minimum": lo, "maximum": hi, "bin_width": step,
                "generated": shared_evaluator.generated.histograms[i].tolist(),
                "matched_gt": shared_evaluator.real.histograms[i].tolist(),
                "full_gt": shared_evaluator.store.full_real.histograms[i].tolist(),
            }
            for i, (name, lo, hi, step, _) in enumerate(SPECS)
        })
        save_json(folder / "sd_agent_metrics.json", report)
        return report, time.monotonic() - started

    run_names = ([] if args.skip_baseline else ["warm_start"]) + names + list(evaluation_checkpoints)
    if len(set(run_names)) != len(run_names):
        raise ValueError("Run names must be unique")
    for name in run_names:
        started = time.monotonic()
        print(f"\nSTART {name}", flush=True)
        folder = output / name
        folder.mkdir()
        arm_cfg = copy.deepcopy(cfg)
        with open_dict(arm_cfg):
            arm_cfg.paths.output_dir = str(folder)
            if name in evaluation_checkpoints:
                arm_cfg.ckpt_path = str(Path(evaluation_checkpoints[name]).resolve())
                # Preserve the trained objective in evaluation metadata. The
                # loss does not affect sampling, but the common search recipe
                # must not be reported as another checkpoint's objective.
                source_config = Path(evaluation_checkpoints[name]).parent / "config.yaml"
                if source_config.is_file():
                    trained_cfg = OmegaConf.load(source_config)
                    arm_cfg.model.model_config.decoder.init_diffusion.component_loss = (
                        trained_cfg.model.model_config.decoder.init_diffusion.component_loss
                    )
            loss_cfg = arm_cfg.model.model_config.decoder.init_diffusion.component_loss
            if name != "warm_start" and name not in evaluation_checkpoints and variants[name] is None:
                arm_cfg.model.model_config.decoder.init_diffusion.component_loss = None
            elif name != "warm_start" and name not in evaluation_checkpoints:
                for component, changes in variants[name].items():
                    for key, value in changes.items():
                        loss_cfg.components[component][key] = value
        OmegaConf.save(arm_cfg, folder / "config.yaml")
        L.seed_everything(args.seed, workers=True)
        dm = hydra.utils.instantiate(arm_cfg.data)
        model = hydra.utils.instantiate(arm_cfg.model, _recursive_=False)
        arm_state = (torch.load(evaluation_checkpoints[name], map_location="cpu", weights_only=False)["state_dict"]
                     if name in evaluation_checkpoints else state)
        incompatible = model.load_state_dict(arm_state, strict=False)
        trainable_names = {k for k, p in model.named_parameters() if p.requires_grad}
        if trainable_names.intersection(incompatible.missing_keys):
            raise ValueError(f"Missing trainable warm-start weights: {incompatible}")
        print(f"CHECKPOINT missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}", flush=True)
        model.video_dir = folder / "videos"
        model.video_dir.mkdir()
        diagnostic = Diagnostics([int(s) for s in args.checkpoint_steps.split(",") if s], folder)
        diagnostic.install(model)
        trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true", max_epochs=128,
                            max_steps=args.steps, limit_train_batches=args.steps, limit_val_batches=0,
                            check_val_every_n_epoch=1, num_sanity_val_steps=0, gradient_clip_val=0.5,
                            enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
                            log_every_n_steps=1, logger=CSVLogger(str(folder), name="csv"),
                            default_root_dir=str(folder), callbacks=[diagnostic])
        train_seconds = 0.0
        if name != "warm_start" and name not in evaluation_checkpoints:
            train_start = time.monotonic()
            trainer.fit(model, datamodule=dm)
            train_seconds = time.monotonic() - train_start
            trainer.save_checkpoint(folder / "last.ckpt")
            save_json(folder / "train_trace.json", diagnostic.records)
        report, eval_seconds = evaluate(model, trainer, dm, folder)
        row = {"variant": name, "train_seconds": train_seconds, "eval_seconds": eval_seconds,
               "evaluation_only": name == "warm_start" or name in evaluation_checkpoints,
               "checkpoint": str(Path(evaluation_checkpoints.get(name, args.checkpoint)).resolve())
                             if name == "warm_start" or name in evaluation_checkpoints
                             else str(folder / "last.ckpt"),
               **{k: float(v) for k, v in report["agent_metrics"].items()},
               "num_samples": report["num_samples"], "num_generated_vehicles": report["num_generated_vehicles"]}
        if args.extra_eval_seed is not None:
            extra_folder = folder / f"eval_seed_{args.extra_eval_seed}"
            model.video_dir = extra_folder / "videos"
            model.video_dir.mkdir(parents=True)
            extra_report, extra_seconds = evaluate(model, trainer, dm, extra_folder, args.extra_eval_seed)
            row["extra_eval_seed"] = args.extra_eval_seed
            row["extra_eval_seconds"] = extra_seconds
            row["extra_agent_metrics"] = extra_report["agent_metrics"]
        if diagnostic.records:
            trace_summary = summarize_trace(diagnostic.records)
            save_json(folder / "train_summary.json", trace_summary)
            row["clip_fraction"] = trace_summary["clip_fraction"]
        summary.append(row)
        save_json(output / "summary.json", summary)
        keys = list(dict.fromkeys(k for row in summary for k in row))
        with (output / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)
        print(f"RESULT {json.dumps(row)} total={time.monotonic() - started:.1f}s", flush=True)
        del trainer, model, dm, diagnostic
        gc.collect()
        torch.cuda.empty_cache()
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
