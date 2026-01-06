import torch
import torch.nn as nn

from smart.utils import transform_to_global
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges
from torch_scatter import scatter_max,scatter_mean,scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)


class InitDiscriminator(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:

        super(InitDiscriminator, self).__init__()
        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.entry_his_len = 1000000

        self.use_entry_former = True

        if self.use_entry_former:
            self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.2,
                                              hist_len=self.entry_his_len)  # replace with gnn

        self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.2,
                                         hist_len=self.entry_his_len)        # drop 01 is important

        self.pos_embedding = MLPLayer(2 ,hidden_dim, hidden_dim)
        self.head_embedding = MLPLayer(1,hidden_dim, hidden_dim)
        self.shape_embedding = MLPLayer(3, hidden_dim, hidden_dim)
        self.offset_embedding = MLPLayer(2, hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)

        self.score_decoder = nn.Sequential(MLPLayer(hidden_dim, hidden_dim, 1 ),nn.Sigmoid())

    def padding(self,pos,heading,feature,batch,batch_num):
        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def embed_input(self,initial_pos, initial_heading,initial_type,initial_shape,batch,batch_num ):
        type_embedding = self.type_embedding(initial_type)
        pos_embedding = self.pos_embedding(initial_pos)
        heading_embedding = self.head_embedding(initial_heading[:,None])
        shape_embedding = self.shape_embedding(initial_shape)

        feat_a = type_embedding  + heading_embedding + pos_embedding+ shape_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch,batch_num)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        return pos_a_b, heading_a_b, feat_a_b,mask_a_b


    def forward(self,map_features, pos, heading, shape,tokenized_agent ):

        pos_pl, orient_pl, feat_map, map_mask=map_features

        ego_mask=tokenized_agent["initial_ego_mask"]

        non_ego=~ego_mask

        batch=tokenized_agent["batch"][non_ego]
        type = tokenized_agent["initial_type"][non_ego]
        batch_num=tokenized_agent["num_graphs"]

        heading=wrap_angle(heading)

        pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.embed_input(pos, heading, type, shape, batch, batch_num)

        entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                          heading_a_b, mask_a_b,
                                                          feat_map,
                                                          pos_pl,
                                                          orient_pl, map_mask)

        attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b, 0, 0,  mask_a_b,use_time=False)

        attr_feature = attr_feature[mask_a_b]

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
        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.entry_his_len = 1000000

        self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                          hist_len=self.entry_his_len)  # replace with gnn

        self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                         hist_len=self.entry_his_len)        # drop 01 is important


        self.entry_former1 = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                          hist_len=self.entry_his_len)  # replace with gnn

        self.attr_former1 = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                         hist_len=self.entry_his_len)        # drop 01 is important

        self.noise_embedding = MLPLayer(6 ,hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)

        self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, 2 )
        self.head_decoder = MLPLayer(hidden_dim, hidden_dim,1 )
        self.shape_head_decoder = nn.Sequential( MLPLayer(hidden_dim, hidden_dim, 3),nn.ReLU())

    def forward(self,map_features, tokenized_agent):
        pos_pl, orient_pl, feat_map, map_mask=map_features

        ego_mask=tokenized_agent["initial_ego_mask"]

        type = tokenized_agent["initial_type"]

        batch=tokenized_agent["batch"]

        batch_num=tokenized_agent["num_graphs"]

        type=type[~ego_mask]

        batch=batch[~ego_mask]

        agent_num=len(type)

        z = torch.rand(agent_num, 6, device=type.device)#pos,heading and shape

        feature =self.noise_embedding(z)+self.type_embedding(type)

        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        feat_a_b = padding(feature, lengths, padding_value=0)  # b, n, d

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        pos_a_b=torch.zeros(feat_a_b.shape[0],feat_a_b.shape[1], 2, device=type.device)
        heading_a_b=torch.zeros(feat_a_b.shape[0],feat_a_b.shape[1],  device=type.device)
        n_agent = feat_a_b.shape[1]

        entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                          heading_a_b, mask_a_b,
                                                          feat_map,
                                                          pos_pl,
                                                          orient_pl, map_mask)


        entry_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b, n_agent, 0,  mask_a_b,use_causal=False)


        entry_feature = self.entry_former1.cross_attention(entry_feature, pos_a_b,
                                                          heading_a_b, mask_a_b,
                                                          feat_map,
                                                          pos_pl,
                                                          orient_pl, map_mask)


        attr_feature = self.attr_former1.temporal_embed(entry_feature, pos_a_b, heading_a_b, n_agent, 0,  mask_a_b,use_causal=False)

        attr_feature = attr_feature[mask_a_b]

        pos = self.pos_decoder(attr_feature)

        heading = self.head_decoder(attr_feature)[:,0]

        shape = self.shape_head_decoder(attr_feature)

        return pos,heading,shape
