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


class SinusoidalTimestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim)
        )

    def forward(self, t: torch.Tensor):
        # t in [0,1], shape [B]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device))
        ang = t[:,:, None] * freqs[None,None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return self.proj(emb)


# ------------------------------
# Noise schedule & diffusion wrapper (ε-prediction)
# ------------------------------
@dataclass
class NoiseSchedule:
    alphas_cumprod: torch.Tensor

    @classmethod
    def cosine(cls, timesteps: int, s: float = 0.008, device='cuda'):
        t = torch.linspace(0, timesteps, timesteps+1,device=device) / timesteps
        a_bar = torch.cos(((t + s) / (1+s)) * math.pi/2) ** 2
        a_bar = a_bar / a_bar[0]
        return cls(alphas_cumprod=a_bar[1:])

    @property
    def timesteps(self):
        return self.alphas_cumprod.shape[0]

    def at(self, idx: torch.Tensor):
        return self.alphas_cumprod[idx]



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
