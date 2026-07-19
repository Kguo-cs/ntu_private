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

from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.layers import MLPLayer

from src.smart.utils.edge_utils import generate_limited_causal_mask

class RoFormerSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 dropout,
                 hist_len,
                 use_bias=True,
                 is_decoder=False,
                 rotary_value=False,
                 pos_emb=False,
                 ):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.headwise_attn_output_gate=False
        self.elementwise_attn_output_gate=False

        if self.headwise_attn_output_gate:
            self.query = nn.Linear(hidden_dim, self.all_head_size + self.num_attention_heads, bias=use_bias)
        elif self.elementwise_attn_output_gate:
            self.query = nn.Linear(hidden_dim, self.all_head_size * 2, bias=use_bias)
        else:
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

        if pos_emb is True:
            # self.mlp = MLPLayer(num_heads,hidden_dim,num_heads)
            num_freq_bands=64
            input_dim_r_a2a=3
            self.r_a2a_emb = FourierEmbedding(
                input_dim=input_dim_r_a2a,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

            self.proj= nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_heads+hidden_dim, num_heads),
            )


            #self.r_a2a_emb=MLPLayer(input_dim_r_a2a+num_heads,hidden_dim,num_heads)

        self.pos_emb = pos_emb

        self.hist_len=hist_len


        self.caching=False

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

    def kv_caching(self,caching_len,current_step=0):
        self.caching_len = caching_len
        self.caching=False

        if self.cached_k is not None and current_step!=0:
            self.cached_k = self.cached_k[:, :, :current_step]
            self.cached_v = self.cached_v[:, :, :current_step]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        encoder_hidden_states=None,
        encoder_sinusoidal_pos=None,
        pos_embeding=None,
        n_agent=1
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.query(hidden_states)

        if self.headwise_attn_output_gate:
            query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1)
            query_states, gate_score = torch.split(query_states, [self.attention_head_size, 1], dim=-1)
            gate_score = gate_score.reshape(bsz, q_len, -1, 1)
            query_layer = query_states.reshape(bsz, q_len, -1, self.attention_head_size).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1)
            query_states, gate_score = torch.split(query_states, [self.attention_head_size, self.attention_head_size], dim=-1)
            gate_score = gate_score.reshape(bsz, q_len, -1, self.attention_head_size)
            query_layer = query_states.reshape(bsz, q_len, -1, self.attention_head_size).transpose(1, 2)
        else:
            query_layer = query_states.view(bsz, q_len, -1, self.attention_head_size).transpose(1, 2)

        if not self.pos_emb and sinusoidal_pos is not None:
           query_layer = self.apply_rotary(query_layer, sinusoidal_pos)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            if not self.pos_emb and sinusoidal_pos is not None:
                key_layer = self.apply_rotary(key_layer, encoder_sinusoidal_pos)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if not self.pos_emb and sinusoidal_pos is not None:
               key_layer = self.apply_rotary(key_layer, sinusoidal_pos)

        if self.caching_len:
            key_layer = self.cached_k = torch.cat((self.cached_k, key_layer), dim=2)[:,:,-self.caching_len*n_agent:]
            value_layer = self.cached_v = torch.cat( (self.cached_v, value_layer), dim=2)[:,:,-self.caching_len*n_agent:]
        elif self.caching:
            self.cached_k = key_layer
            self.cached_v = value_layer

        B, L, C = hidden_states.shape
        attn = query_layer.mul(self.scale) @ key_layer.transpose(-1, -2) # BHLc @ BHcL => BHLL

        if self.pos_emb:
            attn = attn.permute(0,2,3,1)
            mask=~attention_mask[:,0]
            #self.r_a2a_emb(torch.cat([attn[mask],pos_embeding],dim=-1))
            # attn[mask]=self.mlp(attn[mask])
            #attn_rel = attn[mask]+pos_embeding#torch.cat((attn[mask], pos_embeding), dim=-1)
            # attn_rel = self.attention_proj(attn_rel)
            attn[mask]+=self.proj(torch.cat([attn[mask],self.r_a2a_emb(pos_embeding)],dim=-1))
            # attn[mask]=self.r_a2a_emb(torch.cat([attn[mask],pos_embeding],dim=-1))
            attn=attn.permute(0,3,1,2)

        if attention_mask is not None:
            attn_bias = torch.where(attention_mask, -1e9, 0.)
            attn.add_(attn_bias)
        attn = attn.softmax(dim=-1)
        # if attention_mask is not None:
        #     attn = attn.masked_fill(attention_mask, 0)
        attn=F.dropout(attn, p=self.dropout_p, inplace=False) if self.dropout_p > 0 else attn

        #if attention_mask.shape[-1]>20:

        # relative_pos= query_layer1.mul(self.scale) @ key_layer1.transpose(-1, -2) #BHQK

        # relative_pos=relative_pos.masked_fill(attention_mask, 0)#BHQK

        # #relative_pos=relative_pos.sum(-1) [:,:,:,None] #BHQK

        # relative_value=value_layer[:,:,None]+relative_pos[:,:,:,:,None]

        outputs = attn @ value_layer

        #outputs=torch.einsum('bhqk,bhqkc->bhqc',attn,relative_value)

        # oup = flash_attn_func(query_layer, key_layer, value_layer, dropout_p=self.dropout_p,
        #                       softmax_scale=self.scale).view(B, L, C)

        # outputs = slow_attn(query=query_layer, key=key_layer, value=value_layer, scale=self.scale, attn_mask=attention_mask, dropout_p=self.dropout_p)

        #outputs=outputs.transpose(1, 2).reshape(B, L, C)

        attn_output = outputs.transpose(1, 2).contiguous()

        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        outputs = attn_output.reshape(B, L, -1)


        outputs=self.to_out(outputs)
        return outputs

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        if len(sinusoidal_pos.shape) == 3:
            sin, cos = sinusoidal_pos[:,None].chunk(2, dim=-1)
        else:
            sin, cos = sinusoidal_pos.swapaxes(1,2).chunk(2, dim=-1)
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


def general_rope(positions, dim,heading=None,centering_pos=None,centering_heading=None,batch=None,head_dim=8):

    if batch is not None:
        positions,heading=scene_centric(positions,heading,centering_pos,centering_heading,batch)

    device = positions.device

    d_k=dim//2

    div_term = torch.exp(
        torch.arange(d_k, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_k)
    )

    sin = torch.sin(positions[...,None] * div_term)
    cos = torch.cos(positions[...,None] * div_term)

    if heading is not None:
        theta=heading[...,None,None].repeat_interleave(d_k,dim=-1)

        sin_theta=torch.sin(theta)
        cos_theta=torch.cos(theta)

        sin=sin.repeat_interleave(2,dim=-2)
        cos=cos.repeat_interleave(2,dim=-2)

        sin_theta=sin_theta.repeat_interleave(head_dim-sin.shape[-2],dim=-2)
        cos_theta=cos_theta.repeat_interleave(head_dim-sin.shape[-2],dim=-2)

        sin=torch.cat([sin,sin_theta],dim=-2)
        cos=torch.cat([cos,cos_theta],dim=-2)


    sinusoidal_pos=torch.cat([sin,cos],dim=-1)

    return sinusoidal_pos

# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, hidden_dim,num_heads ):
        super().__init__()

        freqs_x,freqs_y,freqs_z,freqs_t = self.init_random_2d_freqs(dim=hidden_dim // num_heads, num_heads=num_heads, theta=1000)

        freqs=torch.stack([freqs_x,freqs_y,freqs_z,freqs_t],dim=0)
        # self.freqs_t = nn.Parameter(freqs_t.clone(), requires_grad=True)
        # self.freqs_x = nn.Parameter(freqs_x.clone(), requires_grad=True)
        # self.freqs_y = nn.Parameter(freqs_y.clone(), requires_grad=True)
        # self.freqs_z = nn.Parameter(freqs_z.clone(), requires_grad=True)
        #self.freqs_t = nn.Parameter(freqs_t.clone(), requires_grad=True)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

        self.num_heads=num_heads

    def init_random_2d_freqs(self,dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
        freqs_x = []
        freqs_y = []
        freqs_z = []
        freqs_t = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

        #mag= 1 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        for i in range(num_heads):
        #     angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        #     fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        #     fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        #     freqs_x.append(fx)
        #     freqs_y.append(fy)
            ft = torch.cat([mag , mag ], dim=-1)
        #
        #     freqs_t.append(ft)
        #
        #     angle_z  = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        #
        #     fz = torch.cat([mag * torch.cos(angle_z), mag * torch.cos(torch.pi / 2 + angle_z)], dim=-1)
        #
        #     freqs_z.append(fz)
            # x channel frequencies
            fx = ft.clone()  # [dim]
            # y channel frequencies
            fy = ft.clone()
            # z channel frequencies
            fz = ft.clone()
            # time frequencies
            ft = ft.clone()

            freqs_x.append(fx)
            freqs_y.append(fy)
            freqs_z.append(fz)
            freqs_t.append(ft)

        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs_z= torch.stack(freqs_z, dim=0)
        freqs_t=torch.stack(freqs_t, dim=0)

        return freqs_x,freqs_y,freqs_z,freqs_t

    def forward(self,positions=None,heading=None,time=None):
        if time is not None:
            freqs_t = time[..., None] * self.freqs[-1]
        else:
            freqs_t= 0

        if positions is not None:
            t_x, t_y = positions[...,0], positions[...,1]

            freqs_x = t_x[...,None,None] * self.freqs[0]
            freqs_y = t_y[...,None,None] * self.freqs[1]
            freqs_xyh = freqs_x + freqs_y

            if positions.shape[-1]==3:
                t_z=positions[...,2]

                freqs_z = t_z[...,None,None] * self.freqs[2]

                freqs_xyh=freqs_xyh+freqs_z
        else:
            freqs_xyh=0

        if heading is not None:
            freqs_xyh=freqs_xyh+heading[...,None,None]

        freqs_t = freqs_xyh +freqs_t

        sinusoidal_pos = torch.cat([torch.sin(freqs_t), torch.cos(freqs_t)], dim=-1)

        return sinusoidal_pos


# sin_embedding=RoFormerSinusoidalPositionalEmbedding(128,8)
#
# position=torch.zeros([10,18,2])
#
# sin_embedding(position)
#
def padding(tensor,lengths,padding_value=0 ):
    padded_tensor = pad_sequence(list(torch.split(tensor, lengths)), batch_first=True, padding_value=padding_value)

    return padded_tensor


class RoFormerDecoder(nn.Module):
    def __init__(self,  hidden_dim, hist_len=0,num_heads=8, mlp_ratio=4.0, dropout=0.1,pos_emb=False):
        super().__init__()

        self.self_attn = RoFormerSelfAttention(hidden_dim, num_heads, dropout,pos_emb=pos_emb,hist_len=hist_len)

        self.multihead_attn = RoFormerSelfAttention(hidden_dim, num_heads, dropout,pos_emb=pos_emb,hist_len=hist_len)

        self.rotary_embedding = RoFormerSinusoidalPositionalEmbedding(hidden_dim=hidden_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)


        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x,x_pos, x_heading,x_mask,y,y_pos, y_heading,y_mask):

        tgt_mask = ~x_mask[:, None, None, :]
        memory_mask = ~y_mask[:, None, None,:]

        sinusoidal_pos = self.rotary_embedding(x_pos, x_heading, None)
        y_sinusoidal_pos = self.rotary_embedding(y_pos, y_heading, None)


        x = x + self.self_attn(self.norm1(x),tgt_mask,sinusoidal_pos,None,None)
        x = x + self.multihead_attn(self.norm2(x),memory_mask,sinusoidal_pos,y,y_sinusoidal_pos)
        x = x + self.mlp(self.norm3(x))

        return x

class RoFormerBlock(nn.Module):
    def __init__(self,  hidden_dim, hist_len=0,num_heads=8, mlp_ratio=4.0, dropout=0.1,pos_emb=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RoFormerSelfAttention(hidden_dim, num_heads, dropout,pos_emb=pos_emb,hist_len=hist_len)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention_head_size=hidden_dim // num_heads

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )

        self.rotary_embedding = RoFormerSinusoidalPositionalEmbedding(hidden_dim=hidden_dim, num_heads=num_heads)

        self.hist_len=hist_len

    def forward(self, x,attention_mask,sinusoidal_pos,y=None,y_sinusoidal_pos=None,pos_embeding=None,n_agent=1):
        x = x + self.attn(self.norm1(x),attention_mask,sinusoidal_pos,y,y_sinusoidal_pos,pos_embeding,n_agent=n_agent)
        x = x + self.mlp(self.norm2(x))

        return x

    def cross_attention(self,src, src_pos, src_heading,src_mask,tgt, tgt_pos, tgt_heading,tgt_mask ):

        causal_mask = ~src_mask[:, None, :, None] | ~tgt_mask[:, None, None,: ]


        src_sinusoidal_pos = self.rotary_embedding(src_pos, src_heading, None)
        tgt_sinusoidal_pos = self.rotary_embedding(tgt_pos, tgt_heading, None)


        feature = self.forward(src, causal_mask, src_sinusoidal_pos,tgt,tgt_sinusoidal_pos)


        return feature

    def temporal_embed(self,feature, pos, heading, n_step, n_current,  mask,n_agent=1,use_time=True,use_causal=True):

        if use_time==False:
            time=None
        else:
            time = torch.arange(n_current, n_step + n_current, device=feature.device)

            if n_agent>1:
                time=time[:,None].repeat(1,n_agent).flatten(0,1)[None, :mask.shape[1], None][:,-pos.shape[1]:]
            else:
                time=time[None,:,None]

        if pos is not None or heading is not None or time is not None:
            sinusoidal_pos = self.rotary_embedding(pos, heading, time)
        else:
            sinusoidal_pos = None

        if n_step>1 and use_causal:
            causal_mask = generate_limited_causal_mask(n_step, self.hist_len,n_agent, device=feature.device)
            if mask is not None:
                causal_mask = causal_mask[None, None] | ~mask[:, None, None, :]
        else:
            if mask is not None:
                causal_mask = ~mask[:, None, None, :]

        feature = self.forward(feature, causal_mask, sinusoidal_pos,n_agent=n_agent)

        return feature
