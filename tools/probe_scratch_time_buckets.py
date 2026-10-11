"""CPU-only paired training-data denoising diagnostics for scratch checkpoints.

Reports clean-state errors in low/mid/high noise-time buckets. This is a
conditional reconstruction probe, not a generation/JSD evaluation or training.
The actual configured component loss still runs unchanged, without updates.
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
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from src.smart.diffusion.component_loss import heading_angle_residual

BUCKETS = (("low", 0.0, 0.1), ("mid", 0.1, 0.5), ("high", 0.5, 1.0))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BucketProbe:
    def __init__(self, flow):
        self.flow = flow
        self.values = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.input_hash = hashlib.sha256()
        self.forward_calls = 0
        self.agents = 0
        self.scenes = 0
        self.original = flow._component_supervised_loss

    def add(self, component, time, valid, measurements):
        for label, lo, hi in BUCKETS:
            selected = valid & (time > lo) & (time <= hi)
            count = int(selected.sum())
            if not count:
                continue
            self.counts[label][component] += count
            for name, values in measurements.items():
                active = values[selected].detach().double()
                if not torch.isfinite(active).all():
                    raise ValueError(f"Nonfinite {component}/{name} in {label} time bucket")
                self.values[label][component][name] += float(active.sum())

    def regression(self, name, prediction, target, valid, time, scale):
        options = self.flow.component_loss.config["components"][name]
        error = prediction - target
        normalized = error / scale.clamp_min(self.flow.component_loss.config["std_floor"])
        counts = valid.sum(-1)
        mask = counts > 0
        divisor = counts.clamp_min(1)

        def average(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum(-1) / divisor

        l1, mse = average(error.abs()), average(error.square())
        normalized_l1, normalized_mse = average(normalized.abs()), average(normalized.square())
        configured_scale = options["scale"]
        configured_error = normalized if configured_scale == "data" else (
            error if configured_scale == "none" else error / configured_scale)
        configured = average(configured_error.abs() if options["kind"] == "l1" else configured_error.square())
        weighted = configured * self.flow.component_loss.time_weight(name, time, self.flow.t_eps) * options["weight"]
        return mask, {"physical_or_state_l1": l1, "physical_or_state_mse": mse,
                      "normalized_l1": normalized_l1, "normalized_mse": normalized_mse,
                      "configured_weighted_loss": weighted}

    def capture(self, x, x0, noise, time, fixed, agent):
        self.forward_calls += 1
        self.agents += len(x)
        self.scenes += int(agent["num_graphs"])
        for value in (x, noise, time, fixed):
            self.input_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        scalar_time = time[:, 0]
        active = torch.isfinite(scalar_time) & (scalar_time > 0) & (scalar_time < 1)
        valid = (~fixed) & active[:, None]
        size_valid = agent.get("_init_diffusion_size_valid_mask")
        if size_valid is None:
            size_valid = torch.isfinite(x[:, 4:6])
            if self.flow.size_representation == "linear":
                size_valid &= x[:, 4:6] > 0
        valid = valid.clone()
        valid[:, 4:6] &= size_valid
        dim = self.flow.model.m_delta_dim
        for name, fields in (("position", slice(0, 2)), ("length", slice(4, 5)),
                             ("width", slice(5, 6)), ("velocity", slice(6, dim))):
            mask, values = self.regression(name, x0[:, fields], x[:, fields], valid[:, fields],
                                           scalar_time, self.flow.model.normal_scale[:, fields])
            values["mean_time"] = scalar_time
            self.add(name, scalar_time, mask, values)

        heading_valid = valid[:, 2:4].all(-1)
        residual = heading_angle_residual(x0[:, 2:4], x[:, 2:4])
        vector_error = x0[:, 2:4] - x[:, 2:4]
        radius = x0[:, 2:4].norm(dim=-1)
        values = {"angle_l1_rad": residual.abs(), "angle_mse_rad2": residual.square(),
                  "angle_l1_deg": residual.abs() * (180 / torch.pi),
                  "vector_l1": vector_error.abs().mean(-1), "vector_mse": vector_error.square().mean(-1),
                  "prediction_radius_mean": radius, "prediction_radius_below_01_fraction": (radius < 0.1).float(),
                  "mean_time": scalar_time}
        if self.flow.component_loss.config["heading_space"] == "angle":
            _, configured = self.regression("heading", residual[:, None], torch.zeros_like(residual[:, None]),
                                            heading_valid[:, None], scalar_time, torch.ones(1, 1))
        else:
            _, configured = self.regression("heading", x0[:, 2:4], x[:, 2:4], valid[:, 2:4],
                                            scalar_time, self.flow.model.normal_scale[:, 2:4])
        values["configured_weighted_loss"] = configured["configured_weighted_loss"]
        self.add("heading", scalar_time, heading_valid, values)

        if self.flow.generate_type and not self.flow.feature_type:
            labels = agent["type"].long()
            logits = agent["_init_diffusion_type_logits"]
            type_valid = active & ~self.flow._conditioned_type_mask(agent)
            error = F.cross_entropy(logits, labels, reduction="none")
            weight = self.flow.component_loss.config["components"]["type"]["weight"]
            self.add("type", scalar_time, type_valid, {"cross_entropy": error,
                "accuracy": (logits.argmax(-1) == labels).float(), "mean_time": scalar_time,
                "configured_weighted_loss": error * self.flow.component_loss.time_weight("type", scalar_time, self.flow.t_eps) * weight})

    def install(self):
        probe = self

        def wrapped(instance, x, x0, noise, time, fixed, tokenized_agent, prediction=None):
            probe.capture(x, x0, noise, time, fixed, tokenized_agent)
            return probe.original(x, x0, noise, time, fixed, tokenized_agent, prediction)

        self.flow._component_supervised_loss = types.MethodType(wrapped, self.flow)

    def report(self):
        result = {}
        for bucket, lo, hi in BUCKETS:
            components = {}
            for name, count in self.counts[bucket].items():
                components[name] = {"count": count, **{
                    key: total / count for key, total in self.values[bucket][name].items()}}
            result[bucket] = {"time_interval": [lo, hi], "components": components}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True, help="JSON name -> actual scratch checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4717)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-manifest", type=Path,
                        default=ROOT / "logs/component_loss_scratch/calibration/calibration_files.pkl")
    args = parser.parse_args()
    if min(args.scenes, args.batches, args.batch_size) <= 0:
        parser.error("scenes, batches and batch-size must be positive")
    # Disallow accidental GPU use even if the parent currently trains there.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    with args.calibration_manifest.open("rb") as handle:
        data = pickle.load(handle)
    files = data["files"] if isinstance(data, dict) else data
    files = list(files)[:min(args.scenes, args.batches * args.batch_size)]
    if not files:
        raise ValueError("Empty training calibration manifest")
    manifest_path = args.output / "probe_files.pkl"
    with manifest_path.open("wb") as handle:
        pickle.dump({"files": files}, handle)
    manifest_hash = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
    checkpoints = json.loads(args.checkpoints.read_text())
    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    results, rows = {}, []
    paired_hash = None
    for name, checkpoint_path in checkpoints.items():
        started = time.monotonic()
        checkpoint_path = Path(checkpoint_path).resolve()
        cfg_path = checkpoint_path.parent / "config.yaml"
        cfg = copy.deepcopy(OmegaConf.load(cfg_path))
        with open_dict(cfg):
            cfg.ckpt_path = None
            cfg.action = "fit"
            cfg.model.model_config.finetune = False
            cfg.paths.root_dir = str(ROOT / "src")
            cfg.data.scenario_dreamer_train_sample_list = str(manifest_path.resolve())
            cfg.data.train_batch_size = args.batch_size
            cfg.data.num_workers = 0
            cfg.data.persistent_workers = False
            cfg.data.pin_memory = False
            cfg.data.shuffle = False
        seed_all(args.seed)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "scratch_sweep" not in checkpoint:
            raise ValueError("Probe accepts only recorded scratch-training checkpoints")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = model.encoder.init_decoder.G1
        if flow.component_loss is None:
            raise ValueError("Legacy loss cannot be captured through component-loss hook")
        if flow.heading_objective != "x0":
            raise ValueError("This diagnostic compares clean-state x0 heading objectives")
        dm = hydra.utils.instantiate(cfg.data)
        dm.setup("fit")
        loader = dm.train_dataloader()
        model.train()
        probe = BucketProbe(flow)
        probe.install()
        optimized_losses = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                if batch_i >= args.batches:
                    break
                # Pair the same source draws/timesteps across all checkpoints.
                seed_all(args.seed + batch_i)
                tokenized_map, agent = model.token_processor(batch)
                output = model.encoder(tokenized_map, agent)
                optimized_losses.append(float(output["initial_logit"][0]))
        input_hash = probe.input_hash.hexdigest()
        if paired_hash is None:
            paired_hash = input_hash
        elif paired_hash != input_hash:
            raise ValueError("Probe source/target/time pairing differs across models; compare compatible conditions")
        result = {"checkpoint": str(checkpoint_path), "budget": int(checkpoint["global_step"]),
                  "training_seed": checkpoint["scratch_sweep"]["seed"], "probe_seed": args.seed,
                  "probe_manifest_sha256": manifest_hash, "paired_inputs_sha256": input_hash,
                  "training_calibration_sha256": checkpoint["scratch_sweep"]["calibration_sha256"],
                  "component_loss": flow.component_loss.config, "batches": probe.forward_calls,
                  "scenes": probe.scenes, "agents": probe.agents, "cpu_seconds": time.monotonic() - started,
                  "cpu_precision": "float32; no CUDA/TF32", "training_mode": True,
                  "optimizer_updates": 0, "original_component_loss_called": True,
                  "mean_total_regression_loss": float(np.mean(optimized_losses)), "buckets": probe.report()}
        results[name] = result
        write_json(args.output / "time_buckets.json", results)
        for bucket, details in result["buckets"].items():
            for component, measurements in details["components"].items():
                rows.append({"variant": name, "budget": result["budget"], "bucket": bucket,
                             "component": component, **measurements})
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output / "time_buckets.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"PROBED {name}: {probe.scenes} training scenes, {probe.agents} agents, "
              f"{time.monotonic() - started:.1f}s", flush=True)
        del model, flow, dm, loader, probe, checkpoint
        gc.collect()
    text = ["# Scratch checkpoint training-data time buckets", "",
            "CPU, float32, paired training samples/source draws/timesteps; zero optimizer updates.", "",
            "Conditional reconstruction errors are diagnostics and do not establish generated JSD quality.", "",
            "Collision is omitted. Shape columns are state-space units (log units for log-size models).", "",
            "| Model | Step | Bucket | Position normalized MSE | Heading MAE (deg) | Length normalized MSE | Width normalized MSE | Speed normalized MSE |", 
            "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for bucket, details in result["buckets"].items():
            components = details["components"]
            values = [components.get(component, {}).get(key, float("nan")) for component, key in
                      (("position", "normalized_mse"), ("heading", "angle_l1_deg"),
                       ("length", "normalized_mse"), ("width", "normalized_mse"), ("velocity", "normalized_mse"))]
            text.append(f"| {name} | {result['budget']} | {bucket} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    (args.output / "time_buckets.md").write_text("\n".join(text) + "\n")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only probe unexpectedly initialized CUDA")
    print(f"DONE {args.output}", flush=True)


if __name__ == "__main__":
    main()
