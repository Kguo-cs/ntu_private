# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch RoFormer model. """


import math
import os
from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class RoFormerSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 dropout,
                 use_bias=True,
                 is_decoder=False,
                 rotary_value=True
                 ):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        self.is_decoder = is_decoder
        self.rotary_value = rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else context_layer
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        outputs=self.to_out(outputs)
        return outputs

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


# import math
# import torch
# import torch.nn as nn
#
#
# def apply_rotary_pos_emb(q, k, sin, cos):
#     # q, k: (batch, heads, seq_len, dim)
#     q1, q2 = q[..., ::2], q[..., 1::2]
#     k1, k2 = k[..., ::2], k[..., 1::2]
#
#     q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
#     k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
#     return q_rotated, k_rotated
#

def get_sin_cos(seq_len, dim, device):
    # Sinusoidal positions: (seq_len, dim // 2)
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    sinusoid = position * freq
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    # Expand to match (1, 1, seq_len, dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)
    return sin, cos

# class RoFormerSelfAttention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0.1):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.to_out = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         b, t, d = x.shape
#         h = self.heads
#         qkv = self.qkv(x).chunk(3, dim=-1)
#
#         qkv = qkv.view(b, t, 3, h, -1).permute(2, 0, 3, 1, 4)  # (3, B, H, T, HD)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, T, HD)
#
#         sin, cos = get_sin_cos(t, q.shape[-1], x.device)
#         q, k = apply_rotary_pos_emb(q, k, sin, cos)
#
#         attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = attn_scores.softmax(dim=-1)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h t d -> b t (h d)')
#         return self.to_out(out)



class RoFormerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RoFormerSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention_head_size=hidden_dim // num_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x,attention_mask,sinusoidal_pos):
        x = x + self.attn(self.norm1(x),attention_mask,sinusoidal_pos)
        x = x + self.mlp(self.norm2(x))
        return x

#
# class RoFormer(nn.Module):
#     def __init__(self, vocab_size, dim, depth, heads, max_len, num_classes, dropout=0.1):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, dim)
#         self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))  # unused in RoPE
#         self.blocks = nn.ModuleList([
#             RoFormerBlock(dim, heads, dropout=dropout)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(dim)
#         self.cls_head = nn.Linear(dim, num_classes)
#
#     def forward(self, x):
#         x = self.token_emb(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.norm(x)
#         return self.cls_head(x[:, 0])
