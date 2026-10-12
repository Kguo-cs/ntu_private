"""Optional training-only marginal transport loss for initial agent attributes."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


class AttributeDistributionLoss:
    """Compare sorted clean predictions/targets in each valid type group.

    This supplements x0 reconstruction; it does not change generation or the
    evaluator. Only training labels and a minibatch's targets are used.
    """

    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise ValueError("distribution_loss must be a mapping or null")
        if set(config) - {"speed", "length", "width", "time_min", "kind", "by_type"}:
            raise ValueError("Unknown distribution_loss options")
        self.weights = {name: float(config.get(name, 0.)) for name in ("speed", "length", "width")}
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("distribution_loss weights must be finite and nonnegative")
        self.time_min = float(config.get("time_min", 0.5))
        if not math.isfinite(self.time_min) or not 0 <= self.time_min < 1:
            raise ValueError("distribution_loss.time_min must be in [0, 1)")
        self.kind = config.get("kind", "mse")
        self.by_type = config.get("by_type", True)
        if self.kind not in ("l1", "mse") or not isinstance(self.by_type, bool):
            raise ValueError("distribution_loss requires l1/mse kind and boolean by_type")

    def __call__(self, prediction, target, time, valid, labels, scale, *, scalar_speed=True):
        zero = prediction.sum() * 0.
        losses, metrics = [], {}
        active = torch.isfinite(time.reshape(-1)) & (time.reshape(-1) >= self.time_min) & (time.reshape(-1) < 1)
        groups = (0, 1, 2) if self.by_type else (None,)
        for name, field in (("length", 4), ("width", 5), ("speed", 6)):
            weight = self.weights[name]
            if weight == 0:
                continue
            if name == "speed" and not scalar_speed:
                values = torch.linalg.vector_norm(prediction[:, 6:8], dim=-1)
                truth = torch.linalg.vector_norm(target[:, 6:8].detach(), dim=-1)
                mask = valid[:, 6:8].all(-1)
                denominator = torch.linalg.vector_norm(scale.reshape(-1)[6:8]).detach().clamp_min(.1)
            else:
                values, truth = prediction[:, field], target[:, field].detach()
                mask = valid[:, field]
                denominator = scale.reshape(-1)[field].detach().clamp_min(.1)
            group_losses = []
            for label in groups:
                selected = mask & active
                if label is not None:
                    selected = selected & (labels == label)
                # Avoid NaN gradients; invalid TRAIN annotations remain masked.
                selected = selected & torch.isfinite(truth)
                if int(selected.sum()) < 2:
                    continue
                error = (values[selected].float().sort().values - truth[selected].float().sort().values) / denominator
                group_losses.append(error.abs().mean() if self.kind == "l1" else error.square().mean())
            loss = torch.stack(group_losses).mean() if group_losses else zero
            metrics[f"distribution_{name}_loss"] = loss.detach()
            metrics[f"distribution_{name}_weighted_loss"] = (weight * loss).detach()
            losses.append(weight * loss)
        return sum(losses, zero), metrics
