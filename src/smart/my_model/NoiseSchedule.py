"""
DiT-style Transformer denoiser for trajectory diffusion (PyTorch)
-----------------------------------------------------------------
A compact implementation of a 1D **DiT** (Diffusion Transformer) denoiser for
future 2D trajectory prediction. Drop-in replacement for a U-Net denoiser.

Highlights
* AdaLN-Zero modulation from timestep + context (classifier-free ready)
* Multi-head self-attention over time tokens (trajectory steps)
* Learned positional embeddings
* Works with the same NoiseSchedule + training loop as a DDPM/DDIM wrapper

Usage
-----
python dit_trajectory_diffusion.py --epochs 5 --device cuda

This script includes:
- SyntheticTraj dataset (as in the U-Net example)
- DiT1D model (Transformer blocks with AdaLN-Zero)
- TrajectoryDiffusion wrapper (ε-prediction, cosine schedule, DDIM sampling)

MIT License
"""
from __future__ import annotations
import math
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




# ------------------------------
# Noise schedule & diffusion wrapper (ε-prediction)
# ------------------------------
@dataclass
class NoiseSchedule:
    alphas_cumprod: torch.Tensor

    @classmethod
    def cosine(cls, timesteps: int, s: float = 0.008, device='cpu'):
        t = torch.linspace(0, timesteps, timesteps+1) / timesteps
        a_bar = torch.cos(((t + s) / (1+s)) * math.pi/2) ** 2
        a_bar = a_bar / a_bar[0]
        return cls(alphas_cumprod=a_bar[1:])

    @property
    def timesteps(self):
        return self.alphas_cumprod.shape[0]

    def at(self, idx: torch.Tensor):
        return self.alphas_cumprod[idx]

