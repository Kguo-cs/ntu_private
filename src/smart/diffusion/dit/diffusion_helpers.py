import numpy as np
import torch

def extract(a, t, x_shape):
    """ Extracts values from a tensor `a` at indices specified by `t`."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
import numpy as np




def weight_init(m):
    """Initialize weights of PyTorch modules. Inspired by QCNET: https://github.com/ZikangZhou/QCNet"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Taken from https://github.com/facebookresearch/DiT
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AttentionLayerDiT(MessagePassing):
    """Transformer attention layer for DiT, taken from https://github.com/facebookresearch/DiT"""

    def __init__(self,
                 hidden_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(AttentionLayerDiT, self).__init__(aggr='add', node_dim=0, **kwargs)
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)

        # Optional normalization for q and k
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Attention and projection dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def message(self, q_i, k_j, v_j, index, ptr):
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs):
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        return inputs

    def _attn_block(self, x_src, x_dst, edge_index):
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        x_dst = self.propagate(edge_index=edge_index, q=q, k=k, v=v)

        x_dst = self.proj(x_dst)
        x_dst = self.proj_drop(x_dst)
        return x_dst

    def forward(self, x, edge_index):
        x_src = x_dst = x
        return self._attn_block(x_src, x_dst, edge_index)



class AttentionLayerDiT1(MessagePassing):
    """Transformer attention layer for DiT with optional relational bias.

    v3 upgrades (backwards compatible):
    -------------------------------
    1) Optional relational bias (use_rel_bias):
       - Per-edge, per-head additive logit bias.
    2) Optional relational value gate (use_rel_gate):
       - Per-edge, per-head multiplicative gate on the message/value.
       - Gate is parameterized as (1 + tanh(.)) and ZERO-INIT so initial behavior
         is exactly identical to the baseline.

    When use_rel_bias=False, behavior is identical to the original implementation.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        # -------- Relational parameters --------
        use_rel_bias=True,
        rel_dim=64,
        edge_dim=32,
        use_rel_gate: bool = True,
        attn_logit_clip=30,
        **kwargs,
    ):
        super(AttentionLayerDiT1, self).__init__(aggr='add', node_dim=0, **kwargs)
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.hidden_dim = hidden_dim

        # -------- Standard attention components --------
        self.to_q = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)

        # Optional normalization for q and k (improves stability)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Attention and projection dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # -------- Relational bias / gate components --------
        self.use_rel_bias = bool(use_rel_bias)
        self.rel_dim = int(rel_dim)
        self.edge_dim = int(edge_dim)
        self.use_rel_gate = bool(use_rel_gate)
        self.attn_logit_clip = attn_logit_clip

        if self.use_rel_bias:
            # Low-dim projection for pairwise computation (memory efficient)
            self.rel_proj = nn.Linear(hidden_dim, self.rel_dim)
            self.rel_norm = nn.LayerNorm(self.rel_dim)

            # Edge MLP: [u_i, u_j, |u_i - u_j|] -> edge_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * self.rel_dim, self.edge_dim * 2),
                nn.SiLU(),
                nn.Linear(self.edge_dim * 2, self.edge_dim),
            )

            # Edge features -> per-head attention bias (ZERO-INIT for stability)
            self.edge_to_bias = nn.Linear(self.edge_dim, num_heads, bias=False)

            # Optional: Edge features -> per-head value gate (ZERO-INIT; starts as identity)
            if self.use_rel_gate:
                self.edge_to_gate = nn.Linear(self.edge_dim, num_heads, bias=False)

    def _zero_init_rel_bias(self):
        """Zero-initialize relational layers. Called during DiT.initialize_weights()."""
        if not self.use_rel_bias:
            return
        # Ensure edge_mlp outputs ~0 at init
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        # Ensure logit bias starts at 0
        nn.init.zeros_(self.edge_to_bias.weight)
        # Ensure value gate starts at identity: tanh(0)=0 -> 1+0=1
        if getattr(self, "use_rel_gate", False) and hasattr(self, "edge_to_gate"):
            nn.init.zeros_(self.edge_to_gate.weight)

    def message(self, q_i, k_j, v_j, index, ptr, rel_bias=None, rel_gate=None):
        """Compute attention message with optional relational bias/gate.

        Args:
            q_i: (E, num_heads, head_dim) queries of target nodes
            k_j: (E, num_heads, head_dim) keys of source nodes
            v_j: (E, num_heads, head_dim) values of source nodes
            rel_bias: (E, num_heads) additive bias on logits
            rel_gate: (E, num_heads) multiplicative gate on message/value
        """
        sim = (q_i * k_j).sum(dim=-1) * self.scale  # (E, num_heads)

        if rel_bias is not None:
            sim = sim + rel_bias

        if self.attn_logit_clip is not None and float(self.attn_logit_clip) > 0:
            sim = torch.clamp(sim, -float(self.attn_logit_clip), float(self.attn_logit_clip))

        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)

        msg = v_j * attn.unsqueeze(-1)  # (E, num_heads, head_dim)
        if rel_gate is not None:
            msg = msg * rel_gate.unsqueeze(-1)
        return msg

    def update(self, inputs):
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        return inputs

    def _compute_rel_bias_and_gate(self, x, edge_index):
        """Compute per-edge relational bias and optional value gate.

        Returns:
            rel_bias: (E, num_heads)
            rel_gate: (E, num_heads) or None
        """
        u = self.rel_norm(self.rel_proj(x))  # (N, rel_dim)

        # edge_index[0]=src, edge_index[1]=dst (PyG convention)
        u_i = u[edge_index[1]]  # dst (E, rel_dim)
        u_j = u[edge_index[0]]  # src (E, rel_dim)

        edge_feat = torch.cat([u_i, u_j, torch.abs(u_i - u_j)], dim=-1)  # (E, 3*rel_dim)
        edge_feat = self.edge_mlp(edge_feat)  # (E, edge_dim)

        rel_bias = self.edge_to_bias(edge_feat)  # (E, num_heads)

        rel_gate = None
        if self.use_rel_gate and hasattr(self, "edge_to_gate"):
            # Identity at init: 1 + tanh(0) = 1
            rel_gate = 1.0 + torch.tanh(self.edge_to_gate(edge_feat))  # (E, num_heads)

        return rel_bias, rel_gate

    def _attn_block(self, x_src, x_dst, edge_index):
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)

        rel_bias = None
        rel_gate = None
        if self.use_rel_bias:
            rel_bias, rel_gate = self._compute_rel_bias_and_gate(x_src, edge_index)

        x_dst = self.propagate(
            edge_index=edge_index,
            q=q,
            k=k,
            v=v,
            rel_bias=rel_bias,
            rel_gate=rel_gate,
        )

        x_dst = self.proj(x_dst)
        x_dst = self.proj_drop(x_dst)
        return x_dst

    def forward(self, x, edge_index):
        x_src = x_dst = x
        return self._attn_block(x_src, x_dst, edge_index)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    Taken from https://github.com/facebookresearch/DiT
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Taken from https://github.com/facebookresearch/DiT
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels)
        return embeddings


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Taken from https://github.com/facebookresearch/DiT
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Taken from https://github.com/facebookresearch/DiT
    """

    def __init__(self, hidden_size, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionLayerDiT(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout,
                                      proj_drop=dropout, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, edge_index):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), edge_index)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FactorizedDiTBlock(nn.Module):
    """
    Sequence of factorized (a2l, l2l, l2a, a2a) DiT blocks.
    """

    def __init__(
            self,
            hidden_dim,
            hidden_dim_agent,
            num_heads,
            num_heads_agent,
            dropout,
            mlp_ratio=4.0,
            num_l2l_blocks=1):

        super().__init__()
        self.num_l2l_blocks = 0

        if self.num_l2l_blocks >0:

            # l2l
            # we stack several l2l blocks to give more capacity for lane modeling
            self.l2l_blocks = []
            for _ in range(num_l2l_blocks):
                self.l2l_blocks.append(DiTBlock(hidden_dim, num_heads, dropout, mlp_ratio))
            self.l2l_blocks = nn.ModuleList(self.l2l_blocks)

            # a2l
           # self.upsample_x_agent = nn.Linear(hidden_dim_agent, hidden_dim)
            self.a2l_block = DiTBlock(hidden_dim, num_heads, dropout, mlp_ratio)
          #  self.downsample_x_lane = nn.Linear(hidden_dim, hidden_dim_agent)
        # a2a
        self.a2a_block = DiTBlock(hidden_dim_agent, num_heads_agent, dropout, mlp_ratio)

        # l2a
        #
        self.l2a_block = DiTBlock(hidden_dim_agent, num_heads_agent, dropout, mlp_ratio)


    def forward(
            self,
            x_lane,
            x_agent,
            c,
            c_small,
            l2l_edge_index,
            a2a_edge_index,
            l2a_edge_index):

        if self.num_l2l_blocks >0:

            # a2l
            #x_lane_agent = torch.cat([x_lane, self.upsample_x_agent(x_agent)], dim=0)
            x_lane_agent = torch.cat([x_lane, x_agent], dim=0)
            x_lane_agent = self.a2l_block(x_lane_agent, c, l2a_edge_index[[1, 0], :])
            x_lane = x_lane_agent[:x_lane.shape[0]]

            # l2l
            for i in range(self.num_l2l_blocks):
                x_lane = self.l2l_blocks[i](x_lane, c[:x_lane.shape[0]], l2l_edge_index)

            #x_lane=self.downsample_x_lane(x_lane)

        # l2a
        x_lane_agent = torch.cat([x_lane, x_agent], dim=0)
        x_lane_agent = self.l2a_block(x_lane_agent, c_small, l2a_edge_index)
        x_agent = x_lane_agent[x_lane.shape[0]:]

        # a2a
        x_agent = self.a2a_block(x_agent, c_small[x_lane.shape[0]:], a2a_edge_index)

        return x_lane, x_agent


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    Taken from https://github.com/facebookresearch/DiT
    """

    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, latent_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TwoLayerResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerResMLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.transform_linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.transform_norm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        x = self.transform_linear(x)
        x = self.transform_norm(x)

        out = out + x
        out = self.relu(out)
        return out

import torch

def get_edge_index_bipartite(num_src, num_dst):
    """Create a fully connected bipartite `edge_index` tensor from `num_src` to `num_dst` nodes."""
    # Create a meshgrid of all possible combinations of source and destination nodes
    src_indices = torch.arange(num_src)
    dst_indices = torch.arange(num_dst)
    src, dst = torch.meshgrid(src_indices, dst_indices, indexing='ij')

    # Flatten the meshgrid and stack them to create the edge_index
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
    return edge_index


def get_edge_index_complete_graph(graph_size):
    """Create a directed complete-graph `edge_index` tensor of size `graph_size`."""
    edge_index = torch.cartesian_prod(torch.arange(graph_size, dtype=torch.long),
                                      torch.arange(graph_size, dtype=torch.long)).t()

    return edge_index


def get_indices_within_scene(batch):
    """Get indices of nodes for each scene in the batch, where for each batch element the indices start from 0."""
    _, counts = batch.unique(return_counts=True)
    index = torch.arange(batch.size(0), device=batch.device) - torch.cumsum(counts, dim=0).repeat_interleave(counts) + counts.repeat_interleave(counts)
    return index