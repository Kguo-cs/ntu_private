# MIT License

# Copyright (c) 2024 D. Carpintero
# Modifications Copyright (c)  Da Saem Lee, 2025

import math
import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(N, D):
    """
    Create sinusoidal positional embeddings for positions 1 to N
    Args:
        N: number of positions (assumes positions 1 to N)
        D: embedding dimension (must be even)
    Returns:
        Tensor of shape [N, D]
    """
    position = torch.arange(1, N + 1).unsqueeze(1)  # shape [N, 1]
    div_term = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))  # shape [D/2]

    pe = torch.zeros(N, D)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            depth,  # current layer index
            num_heads,
            num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
            self,
            x_q,
            x_kv,
            # rel_pos,
            attn_mask=None,
    ):
        q_len = x_q.size(1)
        bsz, tgt_len, embed_dim = x_kv.size()
        src_len = tgt_len

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = q.view(bsz, q_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        attn_mask = (
            torch.zeros_like(attn_mask, dtype=q.dtype)
            .masked_fill_(attn_mask, float("-inf"))
        )
        attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        attn_weights = torch.nan_to_num(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, q_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn, attn_weights

class AttentionHead(nn.Module):
    def __init__(self, n_embd, n_headd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, n_headd * 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, 0.0)

        return torch.bmm(torch.softmax(attn_scores, axis=-1), v)

    def forward(self, x, mask):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.scaled_dot_product_attention(q, k, v, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.output_linear = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask):
        return self.output_linear(torch.cat([head(x, mask) for head in self.heads], dim=-1))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(n_embd, ff_dim),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(ff_dim, n_embd))

    def forward(self, x):
        return self.ff(x)


class TransformerDecoderLayerDiff(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim, layer_id=0, dropout=0.1):
        super().__init__()

        self.a2a_mha = MultiHeadAttention(n_embd, n_head)
        self.map2a_diff = MultiheadDiffAttn(n_embd, layer_id, n_head)

        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.norm_3 = nn.LayerNorm(n_embd)

        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, map_enc, map_mask=None, mask=None):
        attn, attn_weights = self.map2a_diff(x, map_enc, attn_mask=map_mask)
        x = self.norm_1(x + self.dropout(attn))
        x = self.norm_2(x + self.dropout(self.a2a_mha(x, mask)))

        return self.norm_3(x + self.feed_forward(x))

