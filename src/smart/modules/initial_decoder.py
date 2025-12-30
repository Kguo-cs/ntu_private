import torch
import torch.nn as nn

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
    transform_to_global,
    transform_to_local,
    wrap_angle,
)

class InitDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
            start_step
    ) -> None:

        super(InitDecoder, self).__init__()
        self.autoregressive_entry= token_processor.autoregressive_entry
        self.n_token_entry=token_processor.n_token_entry

        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.start_step=start_step

        if self.token_processor.use_bird:
            self.pos_dim=3
        else:
            self.pos_dim=2

        self.entry_his_len = 1000000

        self.use_cross_attention = True

        self.use_entry_former = True

        if  self.use_cross_attention:

            if self.use_entry_former:
                self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=self.entry_his_len)  # replace with gnn

        self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1,
                                         hist_len=self.entry_his_len)        # drop 01 is important

        self.use_refine=False

        if self.use_refine:

            self.refine_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1,
                                             hist_len=self.entry_his_len)        # drop 01 is important



        self.pos_embedding = nn.Embedding(self.n_token_entry , hidden_dim)
        self.head_embedding = nn.Embedding(self.token_processor.n_token_entry_head, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.shape_embedding = MLPLayer(3, hidden_dim, hidden_dim)
        self.offset_embedding = MLPLayer(self.pos_dim + 1, hidden_dim, hidden_dim)

        self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, self.n_token_entry )
        self.head_decoder = MLPLayer(hidden_dim, hidden_dim,self.token_processor.n_token_entry_head )
        self.offset_head_decoder = MLPLayer(hidden_dim, hidden_dim,self.pos_dim + 1)  # offset to offset
        self.shape_head_decoder = MLPLayer(hidden_dim, hidden_dim, 3)

    def padding(self,pos,heading,feature,batch,batch_num):
        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def embed_input(self,initial_pos_token,initial_heading_token,initial_type,initial_shape,initial_offset_xyh,initial_pos, initial_heading,batch,batch_num ):
        type_embedding = self.type_embedding(initial_type)
        shape_embedding = self.shape_embedding(initial_shape)
        heading_embedding = self.head_embedding(initial_heading_token)
        pos_embedding = self.pos_embedding(initial_pos_token)

        feat_a = type_embedding + shape_embedding + heading_embedding + pos_embedding

        if initial_offset_xyh.any():
            offset_embedding = self.offset_embedding(initial_offset_xyh)
            feat_a = feat_a + offset_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch,batch_num)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        return pos_a_b, heading_a_b, feat_a_b,mask_a_b

    def forward(self,map_feature,ego_feature, tokenized_agent):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        batch_num=tokenized_agent["num_graphs"]

        pos_pl_b, orient_pl_b, feat_map_b = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)
        map_mask = torch.any(feat_map_b != 0, dim=-1)

        feat_map_b = feat_map_b + ego_feature[:, None]

        if self.training:
            pred_mask = ~tokenized_agent["ego_mask"]#non-last mask
            iteration_num=1
        else:
            pred_mask = tokenized_agent["initial_ego_mask"]

            lengths = torch.bincount(tokenized_agent["batch"]).tolist()

            all_initial_type = padding(tokenized_agent["initial_type"], lengths, padding_value=-1)  # b, n, d

            agent_mask=all_initial_type!=-1

            all_initial_type[~agent_mask]=0

            iteration_num=all_initial_type.shape[1]-1

            self.attr_former.attn.caching = True

            if self.use_entry_former:
                self.entry_former.attn.caching = True
            ego_position=tokenized_agent["initial_pos"][pred_mask]
            ego_heading=tokenized_agent["initial_heading"][pred_mask]

        initial_type = tokenized_agent["initial_type"][pred_mask]
        initial_pos_token = tokenized_agent["initial_pos_token"][pred_mask]
        initial_offset_xyh = tokenized_agent["initial_offset_xyh"][pred_mask]
        initial_heading_token = tokenized_agent["initial_heading_token"][pred_mask]
        initial_shape = tokenized_agent["initial_shape"][pred_mask]
        batch=tokenized_agent["batch"][pred_mask]

        if self.use_refine:
            initial_offset_xyh=None
            initial_pos = self.token_processor.attr_tokenizer.decode_pos(initial_pos_token, None,
                                                                         ego_position, ego_heading)

            token_heading = self.token_processor.attr_tokenizer.decode_heading(initial_heading_token)

            initial_heading = wrap_angle(token_heading + ego_heading)
        else:
            initial_pos = tokenized_agent["initial_pos"][pred_mask]
            initial_heading = tokenized_agent["initial_heading"][pred_mask]

        global_pos_list=[initial_pos]
        global_heading_list=[initial_heading]
        shape_list=[initial_shape]

        type_list=[initial_type]

        for n_current in range(iteration_num):

            pos_a_b, heading_a_b, feat_a_b, mask_a_b=self.embed_input(initial_pos_token,initial_heading_token,initial_type,initial_shape,initial_offset_xyh,initial_pos, initial_heading,batch,batch_num)


            entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                              heading_a_b, mask_a_b,
                                                              feat_map_b,
                                                              pos_pl_b,
                                                              orient_pl_b, map_mask)

            n_agent=feat_a_b.shape[1]

            attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b,n_agent, n_current, mask_a_b)

            attr_feature=attr_feature[mask_a_b]

            pos_logit = self.pos_decoder(attr_feature)
            head_logit = self.head_decoder(attr_feature)
            initial_shape = self.shape_head_decoder(attr_feature)

            if not self.use_refine:
                initial_offset_xyh = self.offset_head_decoder(attr_feature)

                initial_offset_xyh[...,:2] = torch.tanh(initial_offset_xyh[...,:2]) * self.token_processor.attr_tokenizer.grid_interval/2
                initial_offset_xyh[...,2] = torch.tanh(initial_offset_xyh[...,2]) * (torch.pi/self.token_processor.n_token_entry_head)
            else:
                initial_offset_xyh = torch.zeros_like(initial_shape)

            entry_logit=(pos_logit,head_logit,initial_offset_xyh,initial_shape)

            if not self.training:
                initial_pos_token = Categorical(logits=pos_logit).sample()
                initial_heading_token=Categorical(logits=head_logit).sample()

                token_heading = self.token_processor.attr_tokenizer.decode_heading(initial_heading_token)

                initial_heading=wrap_angle(token_heading+ego_heading+initial_offset_xyh[:,-1])

                initial_pos = self.token_processor.attr_tokenizer.decode_pos(initial_pos_token,initial_offset_xyh[:,:2],ego_position,ego_heading)

                global_pos_list.append(initial_pos)
                global_heading_list.append(initial_heading)
                shape_list.append(initial_shape)

                initial_type=all_initial_type[:,n_current+1]

                type_list.append(initial_type)

                print(initial_type)

        if self.use_refine:
            refine_initial_pos = tokenized_agent["initial_pos"][pred_mask]
            refine_initial_heading = tokenized_agent["initial_heading"][pred_mask]

            if not self.training:
                initial_pos = tokenized_agent["initial_pos"][pred_mask]
                initial_heading = tokenized_agent["initial_heading"][pred_mask]
            else:
                global_token_pos = torch.stack(global_pos_list, dim=1)[agent_mask]  # 32,125
                global_token_heading = torch.stack(global_heading_list, dim=1)[agent_mask]

            offset_embedding = self.offset_embedding(initial_offset_xyh)


            pos_a_b, heading_a_b, feat_offset, mask_a_b=self.embed_input(initial_pos_token,initial_heading_token,initial_type,initial_shape,initial_offset_xyh,initial_pos, initial_heading,batch)

            entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                                                              heading_a_b, mask_a_b,
                                                              entry_feature,
                                                              pos_pl_b,
                                                              orient_pl_b, map_mask)

        if not self.training:
            self.attr_former.attn.kv_caching(0)
            if self.use_entry_former:
                self.entry_former.attn.kv_caching(0)

            global_pos = torch.stack(global_pos_list,dim=1)[agent_mask]#32,125
            global_heading = torch.stack(global_heading_list,dim=1)[agent_mask]
            shape = torch.stack(shape_list,dim=1)[agent_mask]


            # type = torch.stack(type_list,dim=1)[agent_mask]
            #
            # print(torch.all(type==tokenized_agent["initial_type"]))

            tokenized_agent["shape"]=shape
            tokenized_agent["ego_mask"] = tokenized_agent["initial_ego_mask"]

            return global_pos[:,None], global_heading[:,None]

        return entry_logit
        # after interact with agent and map,  predict state and type and shape and tokenized position,
        # then refine predict head token and offset_xy ,
        # #then predict all agent motion
