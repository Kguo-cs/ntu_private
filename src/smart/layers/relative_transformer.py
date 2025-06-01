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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value



class RoFormerSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 dropout,
                 use_bias=True,
                 is_decoder=False,
                 rotary_value=False
                 ):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        self.dropout_p=dropout
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        self.is_decoder = is_decoder
        self.rotary_value = rotary_value

        self.scale =1 / math.sqrt(self.attention_head_size)

        # only used during inference
        self.caching_len, self.cached_k, self.cached_v = 0, None, None

        # self.query_pos = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)
        #
        # self.key_pos = nn.Linear(hidden_dim, self.all_head_size, bias=use_bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def kv_caching(self, caching_len):
        self.caching_len = caching_len
        # self.cached_k=None
        # self.cached_v=None


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        encoder_hidden_states=None,
        encoder_sinusoidal_pos=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        
        # query_layer1 = self.transpose_for_scores(self.query_pos(hidden_states))
        #
        # query_layer1 = self.apply_rotary(query_layer1, sinusoidal_pos)

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            key_layer = self.apply_rotary(key_layer,encoder_sinusoidal_pos)


            # key_layer1 = self.transpose_for_scores(self.key_pos(encoder_hidden_states))
            #
            # key_layer1 = self.apply_rotary(key_layer1, encoder_sinusoidal_pos)

            #value_layer=self.apply_rotary(value_layer,encoder_sinusoidal_pos)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)

            #value_layer=self.apply_rotary(value_layer,sinusoidal_pos)

            # key_layer1 = self.transpose_for_scores(self.key_pos(hidden_states))
            #
            # key_layer1 = self.apply_rotary(key_layer1, sinusoidal_pos)

            # value_layer=self.apply(value_layer,sinusoidal_pos)
        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_layer, value_layer)
        # # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # if attention_mask is not None:
        #     # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
        #     attention_scores = attention_scores + attention_mask
        #
        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        #
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask
        #
        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        #
        # outputs = (
        #     (context_layer, attention_probs) if output_attentions else context_layer
        # )
        # if self.is_decoder:
        #     outputs = outputs + (past_key_value,)

        if self.caching_len:
            # if self.cached_k is None:
            #     self.cached_k = key_layer; self.cached_v = value_layer
            # else:
            key_layer = self.cached_k = torch.cat((self.cached_k, key_layer), dim=2)[:,:,-self.caching_len:]
            value_layer = self.cached_v = torch.cat( (self.cached_v, value_layer), dim=2)[:,:,-self.caching_len:]
            #attention_mask=None
        else:
            self.cached_k = key_layer
            self.cached_v = value_layer



        B, L, C = hidden_states.shape
        attn = query_layer.mul(self.scale) @ key_layer.transpose(-1, -2) # BHLc @ BHcL => BHLL
        if attention_mask is not None:
            attn_bias = torch.where(attention_mask, -1e9, 0.)
            attn.add_(attn_bias)
        attn = attn.softmax(dim=-1)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask, 0)
        attn=F.dropout(attn, p=self.dropout_p, inplace=True) if self.dropout_p > 0 else attn

        #if attention_mask.shape[-1]>20:

        # relative_pos= query_layer1.mul(self.scale) @ key_layer1.transpose(-1, -2) #BHQK

        # relative_pos=relative_pos.masked_fill(attention_mask.bool(), 0)#BHQK

        # #relative_pos=relative_pos.sum(-1) [:,:,:,None] #BHQK     

        # relative_value=value_layer[:,:,None]+relative_pos[:,:,:,:,None]
        
        outputs = attn @ value_layer

        #outputs=torch.einsum('bhqk,bhqkc->bhqc',attn,relative_value)

        # oup = flash_attn_func(query_layer, key_layer, value_layer, dropout_p=self.dropout_p,
        #                       softmax_scale=self.scale).view(B, L, C)

        # outputs = slow_attn(query=query_layer, key=key_layer, value=value_layer, scale=self.scale, attn_mask=attention_mask, dropout_p=self.dropout_p)

        outputs=outputs.transpose(1, 2).reshape(B, L, C)

        outputs=self.to_out(outputs)
        return outputs

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos[:,None].chunk(2, dim=-1)
        #sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


def scene_centric(pos,heading,centering_pos,centering_heading,batch):


    heading = heading - centering_heading[batch]

    pos = pos - centering_pos[batch]

    cos_a = torch.cos(centering_heading)[batch]
    sin_a = torch.sin(centering_heading)[batch]

    x, y = pos[..., 0], pos[..., 1]
    x_rot = cos_a * x + sin_a * y
    y_rot = -sin_a * x + cos_a * y

    pos = torch.stack([x_rot, y_rot], dim=-1)

    return  pos,heading




def general_rope(positions, dim,heading=None,centering_pos=None,centering_heading=None,batch=None):

    if batch is not None:
        positions,heading=scene_centric(positions,heading,centering_pos,centering_heading,batch)

    device = positions.device

    dim=dim//2

    theta_dim=dim%positions.shape[-1]

    if theta_dim==0:
        theta_dim=positions.shape[-1]

    d_k=(dim-theta_dim)//positions.shape[-1]

    div_term = torch.exp(
        torch.arange(d_k, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_k)
    )

    # div_term=torch.pow(10000, 2 * (j // 2) / div_dim)

    sin = torch.sin(positions[...,None] * div_term).flatten(-2,-1)
    cos = torch.cos(positions[...,None] * div_term).flatten(-2,-1)

    if heading is not None:
        theta=heading[...,None].repeat_interleave(theta_dim,dim=-1)

        sin_theta=torch.sin(theta)
        cos_theta=torch.cos(theta)

        sin=torch.cat([sin, sin_theta],dim=-1)
        cos=torch.cat([cos, cos_theta],dim=-1)

    sinusoidal_pos=torch.cat([sin,cos],dim=-1)

    return sinusoidal_pos




# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


def padding(tensor,lengths,padding_value=0 ):
    padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=padding_value)

    return padded_tensor

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

    def forward(self, x,attention_mask,sinusoidal_pos,y=None,y_sinusoidal_pos=None):
        x = x + self.attn(self.norm1(x),attention_mask,sinusoidal_pos,y,y_sinusoidal_pos)
        x = x + self.mlp(self.norm2(x))
        return x
