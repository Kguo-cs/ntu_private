"""Optional optimizer recipes shared by SMART's supervised training modules."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def build_scenario_dreamer_optimizer(
    module: nn.Module, *, lr: float = 1e-4, warmup_steps: int = 500,
) -> dict:
    """Use the released LDM's AdamW groups and step-based constant schedule.

    Upstream: scenario-dreamer@6754234, models/scenario_dreamer_ldm.py and
    utils/train_helpers.py. Only linear/convolution/recurrent/attention weights
    decay; biases, normalization, embeddings and other parameters do not.
    For tied weights, a non-decaying owner takes precedence.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Scenario Dreamer optimizer lr must be positive and finite")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("Scenario Dreamer warmup_steps must be a non-negative integer")

    named_parameters = sorted(
        (name, parameter) for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise RuntimeError("The model has no trainable parameters.")

    weight_modules = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention,
        nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell,
    )
    decay, no_decay = set(), set()
    for owner in module.modules():
        for name, parameter in owner.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if "bias" not in name and "weight" in name and isinstance(owner, weight_modules):
                decay.add(id(parameter))
            else:
                no_decay.add(id(parameter))

    decay -= no_decay
    groups = [
        {"params": [parameter for _, parameter in named_parameters if id(parameter) in decay],
         "weight_decay": 1e-5},
        {"params": [parameter for _, parameter in named_parameters if id(parameter) not in decay],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=1e-5,
    )

    def schedule(step: int) -> float:
        return step / warmup_steps if step < warmup_steps else 1.0

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": LambdaLR(optimizer, schedule),
            "interval": "step",
            "frequency": 1,
        },
    }
