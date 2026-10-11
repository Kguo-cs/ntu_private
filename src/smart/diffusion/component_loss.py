"""Independent objectives and timestep weighting for initial-state supervision."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.nn.functional as F


COMPONENTS = ("position", "heading", "length", "width", "velocity", "type", "collision", "speed_magnitude")


class ComponentLoss:
    """Opt-in component means, with coefficients independent of state width.

    Each component averages valid coordinates within an agent, then averages
    over agents with at least one valid coordinate. Constant analytic time
    normalization preserves the requested power without batch-dependent weights.
    """

    def __init__(self, config, *, type_kind="ce", heading_space="vector",
                 type_weight=1.0, heading_weight=1.0, speed_weight=0.0, speed_scale="data"):
        if not isinstance(config, Mapping):
            raise ValueError("component_loss must be a mapping or null")
        unknown = set(config) - {"normalize_time_weights", "std_floor", "heading_space", "components"}
        if unknown:
            raise ValueError(f"Unknown component_loss options: {sorted(unknown)}")
        normalize = config.get("normalize_time_weights", True)
        if not isinstance(normalize, bool):
            raise ValueError("normalize_time_weights must be boolean")
        floor = self._positive(config.get("std_floor", 0.1), "std_floor")
        space = config.get("heading_space", heading_space)
        if space not in ("vector", "angle"):
            raise ValueError("heading_space must be vector or angle")
        entries = config.get("components", {})
        if not isinstance(entries, Mapping):
            raise ValueError("component_loss.components must be a mapping")
        unknown = set(entries) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown loss components: {sorted(unknown)}")
        resolved = {}
        for name in COMPONENTS:
            defaults = dict(kind=type_kind if name == "type" else "l1" if name == "collision" else "mse",
                            time_power=0 if name in ("type", "speed_magnitude") else 1 if name == "collision" else 2,
                            weight=type_weight if name == "type" else heading_weight if name == "heading"
                                   else 0.1 if name == "collision" else speed_weight if name == "speed_magnitude" else 1.0,
                            scale=speed_scale if name == "speed_magnitude" else "none")
            options = entries.get(name, {})
            if not isinstance(options, Mapping):
                raise ValueError(f"component_loss.components.{name} must be a mapping")
            unknown = set(options) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown {name} loss options: {sorted(unknown)}")
            defaults.update(options)
            kind = defaults["kind"]
            if kind not in (("l1", "mse", "ce") if name == "type" else ("l1", "mse")):
                raise ValueError(f"{name}.kind must be l1 or mse" + (" or ce" if name == "type" else ""))
            power = defaults["time_power"]
            if isinstance(power, bool) or not isinstance(power, Integral) or power not in (0, 1, 2, 3):
                raise ValueError(f"{name}.time_power must be an integer in 0, 1, 2, 3")
            weight = defaults["weight"]
            if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name}.weight must be finite and nonnegative")
            scale = defaults["scale"]
            if isinstance(scale, str):
                if scale not in ("none", "data"):
                    raise ValueError(f"{name}.scale must be none, data or a positive number")
            else:
                scale = self._positive(scale, f"{name}.scale")
            if kind == "ce" and scale != "none":
                raise ValueError("CE type loss requires scale=none")
            resolved[name] = dict(kind=kind, time_power=int(power), weight=float(weight), scale=scale)
        self.config = dict(normalize_time_weights=normalize, std_floor=floor,
                           heading_space=space, components=resolved)

    @staticmethod
    def _positive(value, label):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
        return float(value)

    def time_weight(self, name, time, t_eps=0.05):
        eps = self._positive(t_eps, "t_eps")
        if eps > 1:
            raise ValueError("t_eps must be at most 1")
        time = time.reshape(-1)
        if time.dtype in (torch.float16, torch.bfloat16):
            time = time.float()
        active = torch.isfinite(time) & (time > 0) & (time < 1)
        safe_time = torch.where(active, time, torch.ones_like(time))
        power = self.config["components"][name]["time_power"]
        weight = safe_time.clamp_min(eps).reciprocal().pow(power)
        if self.config["normalize_time_weights"]:
            # Integral over Uniform(0, 1), including the clamped interval.
            mean = (1.0 if power == 0 else 1.0 - math.log(eps) if power == 1 else
                    eps ** (1 - power) + (eps ** (1 - power) - 1.0) / (power - 1))
            weight = weight / mean
        return torch.where(active, weight, torch.zeros_like(weight))

    def regression(self, name, prediction, target, time, valid, scale=None, t_eps=0.05):
        options = self.config["components"][name]
        if options["kind"] == "ce":
            raise ValueError("CE requires categorical logits, not a regression state")
        if prediction.ndim == 1:
            prediction, target = prediction[:, None], target[:, None]
            valid = valid[:, None]
        if prediction.ndim != 2 or prediction.shape != target.shape or valid.shape != prediction.shape:
            raise ValueError("Regression prediction, target and mask must have matching [N, D] shapes")
        if time.numel() != len(prediction):
            raise ValueError("Time must contain one value per agent")
        if prediction.dtype in (torch.float16, torch.bfloat16):
            prediction = prediction.float()
        target = target.detach().to(prediction)
        time = time.to(device=prediction.device).reshape(-1)
        active_time = torch.isfinite(time) & (time > 0) & (time < 1)
        valid = valid.to(device=prediction.device, dtype=torch.bool) & active_time[:, None]
        if options["weight"] == 0:
            valid = torch.zeros_like(valid)
        # Mask both operands before subtraction/square to avoid NaN gradients.
        error = (torch.where(valid, prediction, torch.zeros_like(prediction))
                 - torch.where(valid, target, torch.zeros_like(target)))
        configured_scale = options["scale"]
        if configured_scale == "data":
            if scale is None:
                raise ValueError(f"{name}.scale=data requires saved training standard deviations")
            denominator = torch.as_tensor(scale, device=error.device, dtype=error.dtype).detach()
            if not torch.isfinite(denominator).all() or (denominator < 0).any():
                raise ValueError("Saved loss scale must be finite and nonnegative")
            denominator = denominator.clamp_min(self.config["std_floor"])
            error = error / denominator
        elif configured_scale != "none":
            error = error / configured_scale
        errors = error.abs() if options["kind"] == "l1" else error.square()
        count = valid.sum(-1)
        per_agent = errors.sum(-1) / count.clamp_min(1)
        active = count > 0
        raw = per_agent.sum() / active.sum().clamp_min(1)
        weighted = (per_agent * self.time_weight(name, time, t_eps)).sum() / active.sum().clamp_min(1)
        return raw, weighted * options["weight"]

    def categorical(self, logits, labels, time, valid, t_eps=0.05):
        if logits.ndim != 2 or logits.shape[-1] != 3 or labels.shape != logits.shape[:1]:
            raise ValueError("Type loss requires [N, 3] logits and [N] labels")
        if valid.shape != labels.shape or time.numel() != len(labels):
            raise ValueError("Type mask and time must match agent count")
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        time = time.to(device=logits.device).reshape(-1)
        active = valid.to(device=logits.device, dtype=torch.bool) & torch.isfinite(time) & (time > 0) & (time < 1)
        options = self.config["components"]["type"]
        if options["weight"] == 0:
            active = torch.zeros_like(active)
        labels = labels.to(device=logits.device, dtype=torch.long)
        if ((labels[active] < 0) | (labels[active] >= 3)).any():
            raise ValueError("Active type labels must be in [0, 2]")
        safe_logits = torch.where(active[:, None], logits, torch.zeros_like(logits))
        safe_labels = torch.where(active, labels, torch.zeros_like(labels))
        if options["kind"] != "ce":
            probability = safe_logits.softmax(-1)
            target = F.one_hot(safe_labels, 3).to(probability)
            return self.regression("type", probability, target, time, active[:, None].expand_as(probability),
                                   scale=torch.ones(1, 3, device=logits.device), t_eps=t_eps)
        error = F.cross_entropy(safe_logits, safe_labels, reduction="none")
        error = torch.where(active, error, torch.zeros_like(error))
        denominator = active.sum().clamp_min(1)
        return (error.sum() / denominator,
                (error * self.time_weight("type", time, t_eps)).sum() / denominator * options["weight"])


def heading_angle_residual(prediction, target):
    """Shortest angular residual with finite gradients for zero direction pairs."""
    if prediction.dtype in (torch.float16, torch.bfloat16):
        prediction = prediction.float()
    target = target.to(prediction)
    fallback = prediction.new_tensor([1.0, 0.0])
    prediction = torch.where((prediction == 0).all(-1, keepdim=True), fallback, prediction)
    target = torch.where((target == 0).all(-1, keepdim=True), fallback, target)
    cross = target[:, 0] * prediction[:, 1] - target[:, 1] * prediction[:, 0]
    dot = (target * prediction).sum(-1)
    return torch.atan2(cross, dot)
