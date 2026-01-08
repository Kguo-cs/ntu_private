import torch
import torch.nn as nn

from smart.utils import transform_to_global
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer, CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder, topo_rank_among_edges
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)
from typing import Any, Callable, Optional, Union

from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F

class InitDiscriminator(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:

        super(InitDiscriminator, self).__init__()
        self.token_processor = token_processor

        self.hidden_dim = hidden_dim

        self.use_entry_former = True

        if self.use_entry_former:
            self.entry_his_len = 1000000

            self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.2,
                                              hist_len=self.entry_his_len,norm=False)  # replace with gnn

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.2,
                                             hist_len=self.entry_his_len,norm=False)  # drop 01 is important

        else:
            self.edge_encoder = EdgeEncoder(hidden_dim,
                                            num_freq_bands,
                                            a2a=True,
                                            share=False,
                                            hist_drop_prob=0,
                                            time_span=0,
                                            shift=token_processor.shift,
                                            discriminator=False,
                                            use_bird=token_processor.use_bird,
                                            use_cross=True
                                            )

            num_layers = 1

            self.pt2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=hidden_dim // num_heads,
                        dropout=0,
                        bipartite=True,
                        has_pos_emb=True,
                        #  gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )

            self.a2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=hidden_dim // num_heads,
                        dropout=0,
                        bipartite=True,
                        has_pos_emb=True,
                        #  gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.shape_embedding = MLPLayer(3, hidden_dim, hidden_dim,norm=False)
        self.pos_embedding = MLPLayer(2, hidden_dim, hidden_dim,norm=False)
        self.head_embedding = MLPLayer(1, hidden_dim, hidden_dim,norm=False)
        self.type_embedding = nn.Embedding(3, hidden_dim)

        self.score_decoder = MLPLayer(hidden_dim, hidden_dim, 1,norm=False)

    def padding(self, pos, heading, feature, batch, batch_num):
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def embed_input(self, initial_pos, initial_heading, initial_type, initial_shape, batch, batch_num):
        type_embedding = self.type_embedding(initial_type)
        pos_embedding = self.pos_embedding(initial_pos)
        heading_embedding = self.head_embedding(initial_heading[:, None])
        shape_embedding = self.shape_embedding(initial_shape)

        feat_a = type_embedding + heading_embedding + pos_embedding + shape_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch, batch_num)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        return pos_a_b, heading_a_b, feat_a_b, mask_a_b

    def forward(self,inputs, map_feature,  tokenized_agent):

        pos_a=inputs[:,:2]
        head_a=inputs[:,2]
        shape=inputs[:,3:]

        ego_mask = tokenized_agent["ego_mask"]

        non_ego = ~ego_mask

        batch = tokenized_agent["batch"][non_ego]
        type = tokenized_agent["type"][non_ego]
        batch_num = tokenized_agent["num_graphs"]
        head_a = wrap_angle(head_a)

        if self.use_entry_former:
            pos_pl, orient_pl, feat_map, map_mask = map_feature

            pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.embed_input(pos_a, head_a, type, shape, batch, batch_num)

            entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                              heading_a_b, mask_a_b,
                                                              feat_map,
                                                              pos_pl,
                                                              orient_pl, map_mask)

            attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b, 0, 0, mask_a_b,
                                                           use_time=False)

            attr_feature = attr_feature[mask_a_b]
        else:
            mask_a = None

            pos_pl, orient_pl, batch_pl, feat_map = map_feature

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                pos_pl=pos_pl,  # [n_pl, 2]
                orient_pl=orient_pl,  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask_a,  # [n_agent, n_step]
                batch_s=batch,  # [n_agent,n_step]
                batch_pl=batch_pl,  # [n_pl*n_step]
                pl2a_radius=40,
                max_num_neighbors=20,
                agent_train_mask=None,
                layer_num=1
            )

            edge_index_a2a, r_a2a, dist, relative_pos, r_a2a_nei, center_nei_pos, center_nei_heading = self.edge_encoder.build_interaction_edge(
                pos_s=pos_a,  # [n_agent, n_step, 2]
                head_s=head_a,  # [n_agent, n_step]
                head_vector_s=head_vector_a,  # [n_agent, n_step, 2]
                batch_s=batch,  # [n_agent*n_step]
                mask=mask_a,  # [n_agent, n_step]
                max_radius=60,
                max_num_neighbors=20,
                agent_train_mask=None,
                layer_num=1,
                counter_feat_a=None
            )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

            pos_embedding = self.pos_embedding(pos_a)
            heading_embedding = self.head_embedding(head_a[:, None])

            type_embedding = self.type_embedding(type)
            shape_embedding = self.shape_embedding(shape)

            feat_a = type_embedding + shape_embedding + pos_embedding + heading_embedding

            feat_a = self.a2a_attn_layers[0](feat_a, r_a2a, edge_index_a2a)

            attr_feature = self.pt2a_attn_layers[0]((feat_map, feat_a), r_pl2a,
                                                    edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

        score = self.score_decoder(attr_feature)

        return score



class InitGeneator(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:
        super(InitGeneator, self).__init__()
        self.token_processor = token_processor

        self.hidden_dim = hidden_dim

        self.entry_his_len = 1000000

        self.use_entry_former = False

        if self.use_entry_former:

            self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                              hist_len=self.entry_his_len)  # replace with gnn

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                             hist_len=self.entry_his_len)  # drop 01 is important
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim*4,
                dropout=0,
                norm_first=True,
                batch_first=True  # nn.Transformer uses (seq_len, batch, dim)
            )

            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=1
            )
            # d_model = self.hidden_dim
            # nhead=num_heads
            # dropout=0
            # batch_first=True
            # bias=True
            # dim_feedforward=self.hidden_dim*4
            #
            #
            #
            # self.self_attn = MultiheadAttention(
            #     d_model,
            #     nhead,
            #     dropout=dropout,
            #     batch_first=batch_first,
            #     bias=bias
            # )
            # self.multihead_attn = MultiheadAttention(
            #     d_model,
            #     nhead,
            #     dropout=dropout,
            #     batch_first=batch_first,
            #     bias=bias
            # )
            # # Implementation of Feedforward model
            # self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
            # self.dropout = Dropout(dropout)
            # self.linear2 = Linear(dim_feedforward, d_model, bias=bias)
            #
            # self.norm1 = LayerNorm(d_model)
            # self.norm2 = LayerNorm(d_model)
            # self.norm3 = LayerNorm(d_model)
            # self.dropout1 = Dropout(dropout)
            # self.dropout2 = Dropout(dropout)
            # self.dropout3 = Dropout(dropout)
            #
            # self.activation=F.relu


        self.noise_embedding = MLPLayer(6, hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)

        self.pos_embedding = MLPLayer(2, hidden_dim, hidden_dim)
        self.head_embedding = MLPLayer(1, hidden_dim, hidden_dim)

        self.count_embedding = MLPLayer(1, hidden_dim, hidden_dim)

        self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, 2)
        self.head_decoder = MLPLayer(hidden_dim, hidden_dim, 1)
        self.shape_head_decoder = MLPLayer(hidden_dim, hidden_dim, 3)

    def forward(self, map_features, tokenized_agent):
        pos_pl, orient_pl, feat_map, map_mask = map_features

        ego_mask = tokenized_agent["ego_mask"]

        type = tokenized_agent["type"]

        batch = tokenized_agent["batch"]

        batch_num = tokenized_agent["num_graphs"]

        type = type[~ego_mask]

        batch = batch[~ego_mask]

        agent_num = len(type)

        z = torch.rand(agent_num, 6, device=type.device)  #pos,heading and shape

        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_type = padding(type, lengths, padding_value=3)

        mask_a_b = padding_type != 3

        count = torch.arange(padding_type.shape[1], device=type.device)[None].repeat(padding_type.shape[0], 1)

        sort_idx = torch.argsort(padding_type, dim=-1)

        value = torch.zeros_like(count)

        # Scatter count into value at sort_idx positions (row-wise)
        value.scatter_(dim=1, index=sort_idx, src=count)

        value = value[mask_a_b]

        feature = self.noise_embedding(z) + self.type_embedding(type) + self.count_embedding(value[:, None].to(z.dtype))

        feat_a_b = padding(feature, lengths, padding_value=0)  # b, n, d


        feat_map = feat_map + self.pos_embedding(pos_pl) + self.head_embedding(orient_pl[:, :, None])

        if self.use_entry_former:
            pos_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], 2, device=type.device)
            heading_a_b = torch.zeros(feat_a_b.shape[0], feat_a_b.shape[1], device=type.device)
            n_agent = feat_a_b.shape[1]
            entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                              heading_a_b, mask_a_b,
                                                              feat_map,
                                                              pos_pl,
                                                              orient_pl, map_mask)

            entry_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b, n_agent, 0, mask_a_b,
                                                            use_time=False, use_causal=False)  #
        else:
            entry_feature = self.transformer_decoder(
                tgt=feat_a_b,  # self-attention queries
                memory=feat_map,  # cross-attention keys/values
                tgt_key_padding_mask=~mask_a_b,
                memory_key_padding_mask=~map_mask
            )
            # x=feat_a_b
            # tgt_key_padding_mask=~mask_a_b
            # tgt_mask=None
            # tgt_is_causal=None
            # memory=feat_map
            # memory_mask=None
            # memory_key_padding_mask=None
            # memory_is_causal=None
            #
            # x = x + self._mha_block(
            #     self.norm2(x),
            #     memory,
            #     memory_mask,
            #     memory_key_padding_mask,
            #     False,
            # )
            #
            # # x = x + self._sa_block(
            # #     self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            # # )
            # entry_feature = x + self._ff_block(self.norm3(x))

        # entry_feature = self.entry_former1.cross_attention(entry_feature, pos_a_b,
        #                                                   heading_a_b, mask_a_b,
        #                                                   feat_map,
        #                                                   pos_pl,
        #                                                   orient_pl, map_mask)
        #
        #
        # attr_feature = self.attr_former1.temporal_embed(entry_feature, pos_a_b, heading_a_b, n_agent, 0,  mask_a_b,use_time=False,use_causal=False)

        attr_feature = entry_feature[mask_a_b]

        pos = torch.tanh(self.pos_decoder(attr_feature)) * 80

        heading = torch.tanh(self.head_decoder(attr_feature)) * torch.pi

        shape = torch.sigmoid(self.shape_head_decoder(attr_feature))*15

        res=torch.cat([pos, heading, shape], dim=1)

        return res

  # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)