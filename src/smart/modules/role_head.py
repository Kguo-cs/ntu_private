
import torch
import torch.nn as nn
from src.smart.layers import MLPLayer

class RoleHead(nn.Module):
    def __init__(self, emb_dim, K):
        super().__init__()
        # self.proj = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
        #                           nn.Linear(256, K))
        self.proj = MLPLayer(emb_dim, K, emb_dim)
        self.emb  = nn.Embedding(K, emb_dim)  # role embeddings
        self.K = K

    def infer_logits(self, h):        # h: [B, in_dim]
        return self.proj(h)           # [B, K]

    def sample_z(self, logits, tau=1.0, hard=True):
        # Gumbel-Softmax straight-through
        g = -torch.empty_like(logits).exponential_().log()
        y = torch.softmax((logits + g) / tau, dim=-1)  # [B, K]
        if hard:
            y_hard = torch.zeros_like(y).scatter_(1, y.argmax(-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y  # one-hot (st), still has grad

    def embed(self, y_onehot):
        # y_onehot: [B, K]
        idx = y_onehot.argmax(-1)
        return self.emb(idx)          # [B, emb_dim]
