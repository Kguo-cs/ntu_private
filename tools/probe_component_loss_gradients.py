"""Probe actual weighted InitDiffusion component gradients without optimization.

Use a fully composed sweep base_config.yaml, its checkpoint and JSON recipes.
The first training batches and forward RNG are shared across every recipe; the
checkpoint is restored before each forward, so this measures local gradients at
one fixed model rather than gradients along different training trajectories.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import types
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf, open_dict


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def parse_recipes(specifications, parser):
    recipes = {}
    for specification in specifications:
        if "=" not in specification:
            parser.error("Each recipe must be NAME=PATH")
        name, filename = specification.split("=", 1)
        if not name or name in recipes:
            parser.error("Recipe names must be nonempty and unique")
        path = Path(filename).resolve()
        recipe = json.loads(path.read_text())
        if not isinstance(recipe, Mapping):
            parser.error(f"{name}: recipe must be a component_loss JSON mapping")
        recipes[name] = (path, recipe)
    return recipes


class CaptureComponents:
    """Wrap the live loss engine and retain its weighted, undetached scalars."""

    def __init__(self, objective):
        self.terms = defaultdict(list)
        self.categorical_depth = 0
        self.objective = objective
        self.original_regression = objective.regression
        self.original_categorical = objective.categorical

    def __enter__(self):
        def regression(instance, name, *args, **kwargs):
            result = self.original_regression(name, *args, **kwargs)
            # A non-CE categorical loss delegates to regression internally.
            if self.categorical_depth == 0:
                self.terms[name].append(result[1])
            return result

        def categorical(instance, *args, **kwargs):
            self.categorical_depth += 1
            try:
                result = self.original_categorical(*args, **kwargs)
            finally:
                self.categorical_depth -= 1
            self.terms["type"].append(result[1])
            return result

        self.objective.regression = types.MethodType(regression, self.objective)
        self.objective.categorical = types.MethodType(categorical, self.objective)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.objective.regression = self.original_regression
        self.objective.categorical = self.original_categorical


def flat_gradient(loss, parameters, *, retain_graph):
    if loss.requires_grad:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                        allow_unused=True, create_graph=False)
    else:
        gradients = (None,) * len(parameters)
    # Store on the CPU: eight full component vectors need not occupy GPU RAM.
    vector = torch.cat([torch.zeros(parameter.numel(), dtype=torch.float32)
                        if gradient is None else gradient.detach().reshape(-1).float().cpu()
                        for parameter, gradient in zip(parameters, gradients)])
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Nonfinite component gradient")
    return vector, sum(gradient is not None for gradient in gradients)


def cosine(left, right, left_norm, right_norm):
    if left_norm == 0 or right_norm == 0:
        return None
    value = float(torch.dot(left.double(), right.double()) / (left_norm * right_norm))
    return max(-1.0, min(1.0, value))


def measure_batch(model, batch, batch_index, forward_seed, clip_threshold,
                  gradient_sum_rtol=1e-3):
    generator = model.encoder.init_decoder.G1
    named_parameters = [(name, parameter) for name, parameter in generator.named_parameters()
                        if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("No trainable G1 parameters")
    parameters = [parameter for _, parameter in named_parameters]
    versions = [parameter._version for parameter in parameters]
    L.seed_everything(forward_seed, workers=True)
    with CaptureComponents(generator.component_loss) as capture:
        actual_loss = model.training_step(batch, batch_index)
    if not torch.is_tensor(actual_loss) or actual_loss.numel() != 1:
        raise ValueError("Expected scalar supervised training_step output")
    if not torch.isfinite(actual_loss):
        raise FloatingPointError("Nonfinite actual training loss")
    terms = {name: sum(values) for name, values in capture.terms.items()}
    if not terms:
        raise ValueError("No component terms were captured")
    captured_loss = sum(terms.values())
    # This catches an extra policy/RL objective or a duplicated categorical term.
    if not torch.allclose(actual_loss.detach(), captured_loss.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError(f"Captured loss {float(captured_loss.detach())} does not match "
                         f"training_step loss {float(actual_loss.detach())}")
    vectors, connected = {}, {}
    for name, term in terms.items():
        vectors[name], connected[name] = flat_gradient(term, parameters, retain_graph=True)
    # A single backward of the captured sum checks completeness separately from
    # sums of independently rounded backwards. The latter need not be bitwise
    # distributive through FP32/TF32 matmuls and parallel scatter reductions.
    captured_vector, _ = flat_gradient(captured_loss, parameters, retain_graph=True)
    actual_vector, actual_connected = flat_gradient(actual_loss, parameters, retain_graph=False)
    actual_double = actual_vector.double()
    summed_vector = torch.zeros_like(actual_double)
    for vector in vectors.values():
        summed_vector.add_(vector)
    gradient_residual = float(torch.linalg.vector_norm(actual_double - summed_vector))
    total_norm = float(torch.linalg.vector_norm(actual_double))
    relative_residual = gradient_residual / max(total_norm, 1e-12)
    captured_gradient_residual = float(torch.linalg.vector_norm(actual_double - captured_vector.double()))
    captured_gradient_relative_error = captured_gradient_residual / max(total_norm, 1e-12)
    # Keep the original stricter guard for two backwards of the same total.
    capture_gradient_rtol = 1e-4
    precision = torch.get_float32_matmul_precision()
    if captured_gradient_relative_error > capture_gradient_rtol:
        raise ValueError("Captured total gradient differs from actual training gradient: "
                         f"relative_error={captured_gradient_relative_error}, "
                         f"rtol={capture_gradient_rtol}, matmul_precision={precision}")
    if relative_residual > gradient_sum_rtol:
        raise ValueError("Separate component gradients do not reconstruct actual gradient: "
                         f"relative_error={relative_residual}, absolute_error={gradient_residual}, "
                         f"total_norm={total_norm}, rtol={gradient_sum_rtol}, "
                         f"captured_total_relative_error={captured_gradient_relative_error}, "
                         f"matmul_precision={precision}")
    if any(parameter._version != version for parameter, version in zip(parameters, versions)):
        raise RuntimeError("Diagnostic forward mutated a G1 trainable parameter")
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("Probe unexpectedly accumulated parameter .grad")
    norms = {name: float(torch.linalg.vector_norm(vector.double())) for name, vector in vectors.items()}
    norm_sum = sum(norms.values())
    components = {
        name: {
            "weighted_loss": float(terms[name].detach()),
            "gradient_norm": norms[name],
            "norm_fraction": norms[name] / norm_sum if norm_sum else 0.0,
            "cosine_with_total": cosine(vector, actual_vector, norms[name], total_norm),
            # Signed shares sum to one; negative means opposition to the total.
            "projection_fraction": (float(torch.dot(vector.double(), actual_vector.double())) / total_norm ** 2
                                    if total_norm else None),
            "connected_parameter_tensors": connected[name],
        }
        for name, vector in vectors.items()
    }
    return {
        "batch_index": batch_index, "forward_seed": forward_seed,
        "actual_loss": float(actual_loss.detach()), "captured_loss": float(captured_loss.detach()),
        "loss_reconstruction_absolute_error": float((actual_loss.detach() - captured_loss.detach()).abs()),
        "actual_gradient_norm": total_norm,
        "would_clip": total_norm > clip_threshold,
        "clip_scale": min(1.0, clip_threshold / (total_norm + 1e-6)),
        "gradient_reconstruction_relative_error": relative_residual,
        "gradient_reconstruction_absolute_error": gradient_residual,
        "gradient_sum_rtol": gradient_sum_rtol,
        "captured_total_gradient_relative_error": captured_gradient_relative_error,
        "captured_total_gradient_absolute_error": captured_gradient_residual,
        "capture_gradient_rtol": capture_gradient_rtol,
        "sum_component_norms_over_total": norm_sum / total_norm if total_norm else None,
        "actual_connected_parameter_tensors": actual_connected,
        "components": components,
        "pairwise_cosines": {
            left: {right: cosine(vectors[left], vectors[right], norms[left], norms[right])
                   for right in vectors} for left in vectors
        },
    }


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(records):
    names = list(records[0]["components"])
    return {
        "batches": len(records),
        "mean_actual_gradient_norm": mean_or_none([record["actual_gradient_norm"] for record in records]),
        "max_gradient_reconstruction_relative_error": max(
            record["gradient_reconstruction_relative_error"] for record in records),
        "max_captured_total_gradient_relative_error": max(
            record["captured_total_gradient_relative_error"] for record in records),
        "clip_fraction": sum(record["would_clip"] for record in records) / len(records),
        "mean_sum_component_norms_over_total": mean_or_none([
            record["sum_component_norms_over_total"] for record in records]),
        "components": {name: {key: mean_or_none([record["components"][name][key] for record in records])
                              for key in records[0]["components"][name]} for name in names},
        "pairwise_cosines": {left: {right: mean_or_none([
            record["pairwise_cosines"][left][right] for record in records]) for right in names}
                             for left in names},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--recipes", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--batches", default=4, type=int)
    parser.add_argument("--seed", type=int, help="Defaults to the composed config's seed")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--clip-threshold", default=0.5, type=float)
    parser.add_argument("--gradient-sum-rtol", default=1e-3, type=float,
                        help="Relative norm tolerance for separate component backwards; "
                             "captured-total backward retains the stricter 1e-4 guard")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.batches < 1 or args.threads < 1 or not math.isfinite(args.clip_threshold) or args.clip_threshold <= 0:
        parser.error("batches, threads and clip-threshold must be positive")
    if not math.isfinite(args.gradient_sum_rtol) or args.gradient_sum_rtol <= 0:
        parser.error("gradient-sum-rtol must be finite and positive")
    if args.output.exists():
        parser.error("Output already exists; use a new filename")
    recipes = parse_recipes(args.recipes, parser)
    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    cfg = OmegaConf.load(args.config)
    with open_dict(cfg):
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
        cfg.data.pin_memory = False
        cfg.data.shuffle = False
    OmegaConf.resolve(cfg)
    seed = int(cfg.get("seed", 817)) if args.seed is None else args.seed
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "highest"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    L.seed_everything(seed, workers=True)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    loader = iter(dm.train_dataloader())
    batches = []
    for _ in range(args.batches):
        try:
            batches.append(next(loader))
        except StopIteration as error:
            raise ValueError("Requested more batches than the fixed training manifest provides") from error
    selected_files = [Path(path).name for path in dm.train_dataset.raw_paths][
        :args.batches * cfg.data.train_batch_size]
    del loader, dm
    manifest = Path(cfg.data.scenario_dreamer_train_sample_list)
    result = {
        "protocol": {
            "config": str(args.config.resolve()), "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint.get("global_step"),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "train_manifest": str(manifest),
            "train_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "selected_train_files": selected_files,
            "batches": args.batches, "batch_size": cfg.data.train_batch_size, "seed": seed,
            "forward_seeds": [seed + index for index in range(args.batches)],
            "device": str(device), "precision": "float32", "torch_version": str(torch.__version__),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "gradient_storage_dtype": "float32", "gradient_diagnostic_sum_dtype": "float64",
            "gradient_sum_rtol": args.gradient_sum_rtol, "capture_gradient_rtol": 1e-4,
            "scalar_loss_rtol": 1e-5, "scalar_loss_atol": 1e-6,
            "clip_threshold": args.clip_threshold,
            "clip_diagnostic_scope": "G1 only; inspect excluded_trainable_parameters for external map branches",
            "parameter_scope": "all G1 parameters with requires_grad=True, including lane_embed",
            "restore_checkpoint_before_each_batch": True, "optimizer_steps": 0,
            "ema_updates": 0, "gradient_vectors": "unclipped weighted component gradients",
            "cosine_zero_norm": None,
        }, "recipes": {},
    }
    print(f"Gradient probe: float32/matmul={torch.get_float32_matmul_precision()}, "
          f"separate-component rtol={args.gradient_sum_rtol}, captured-total rtol=1e-4", flush=True)
    for name, (recipe_path, recipe) in recipes.items():
        started = time.monotonic()
        arm_cfg = copy.deepcopy(cfg)
        with open_dict(arm_cfg):
            arm_cfg.model.model_config.decoder.init_diffusion.component_loss = recipe
        L.seed_everything(seed, workers=True)
        model = hydra.utils.instantiate(arm_cfg.model, _recursive_=False)
        if not model.automatic_optimization or getattr(model, "gail", False):
            raise ValueError("Probe supports automatic supervised InitDiffusion training only")
        model._log_train = lambda *args, **kwargs: None
        model.log = lambda *args, **kwargs: None
        model.log_dict = lambda *args, **kwargs: None
        model.to(device).train()
        generator = model.encoder.init_decoder.G1
        if generator.component_loss is None:
            raise ValueError("Gradient probe requires the component_loss engine")
        generator_parameter_ids = {id(parameter) for parameter in generator.parameters()}
        excluded = [key for key, parameter in model.named_parameters()
                    if parameter.requires_grad and id(parameter) not in generator_parameter_ids]
        records = []
        for index, cpu_batch in enumerate(batches):
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"Checkpoint mismatch: {incompatible}")
            model.train()
            records.append(measure_batch(model, cpu_batch.clone().to(device), index,
                                         seed + index, args.clip_threshold, args.gradient_sum_rtol))
            print(f"{name} batch {index + 1}/{args.batches}: "
                  f"total_grad={records[-1]['actual_gradient_norm']:.5f}, "
                  f"component_sum_error={records[-1]['gradient_reconstruction_relative_error']:.3e}, "
                  f"captured_total_error={records[-1]['captured_total_gradient_relative_error']:.3e}", flush=True)
        result["recipes"][name] = {
            "recipe_path": str(recipe_path), "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "resolved_component_loss": generator.component_loss.config,
            "trainable_parameter_count": sum(parameter.numel() for parameter in generator.parameters()
                                              if parameter.requires_grad),
            "trainable_parameter_names": [key for key, parameter in generator.named_parameters()
                                          if parameter.requires_grad],
            "excluded_trainable_parameters": excluded,
            "summary": summarize(records), "records": records,
            "seconds": time.monotonic() - started,
        }
        save_json(args.output, result)
        del model, generator
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
