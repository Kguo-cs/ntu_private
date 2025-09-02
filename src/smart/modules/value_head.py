import torch.nn as nn
import torch.nn.functional as F
import torch
from src.smart.layers import MLPLayer


class PopArtHead(nn.Module):
    def __init__(self, in_dim, beta=0.999, eps=1e-5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.last = nn.Linear(256, 1)  # predicts normalized value
        # running stats of returns
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.beta = beta
        self.eps = eps

    def forward(self, x):
        h = self.body(x)                # any centralized features for each (agent,t)
        v_norm = self.last(h).squeeze(-1)              # [*,]
        v_denorm = v_norm * (self.sigma + self.eps) + self.mu
        return v_denorm, v_norm  # return both for training/convenience

    @torch.no_grad()
    def update_stats_and_rescale(self, targets):
        """
        targets: concatenated returns used to train the critic (masked valid)  [M]
        """
        if targets.numel() == 0:
            return
        mu_old = self.mu.clone()
        sig_old = self.sigma.clone()

        # running mean/std update
        m = targets.mean()
        s = targets.std().clamp_min(self.eps)
        self.mu.mul_(self.beta).add_((1 - self.beta) * m)
        self.sigma.mul_(self.beta).add_((1 - self.beta) * s)

        # rescale last layer so denormalized output stays invariant
        scale = (sig_old + self.eps) / (self.sigma + self.eps)
        self.last.weight.data.mul_(scale)
        self.last.bias.data.mul_(scale)
        self.last.bias.data.add_((mu_old - self.mu) / (self.sigma + self.eps))
