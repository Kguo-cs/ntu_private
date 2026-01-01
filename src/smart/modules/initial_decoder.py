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
            else:
                self.edge_encoder = EdgeEncoder(hidden_dim,
                                                num_freq_bands,
                                                a2a=False,
                                                share=False,
                                                hist_drop_prob=0,
                                                time_span=0,
                                                shift=token_processor.shift,
                                                discriminator=False,
                                                use_bird=token_processor.use_bird,
                                                use_cross=True
                                                )

                num_layers=1

                self.pt2a_attn_layers = nn.ModuleList(
                    [
                        AttentionLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=hidden_dim//num_heads,
                            dropout=0,
                            bipartite=True,
                            has_pos_emb=True,
                            #  gated_attention=discriminator,
                        )
                        for _ in range(num_layers)
                    ]
                )

        self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1,
                                         hist_len=self.entry_his_len)        # drop 01 is important

        self.use_refine=False

        if self.use_refine:

            self.refine_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1,
                                             hist_len=self.entry_his_len)        # drop 01 is important



        self.pos_embedding = MLPLayer(2 ,hidden_dim, hidden_dim)
        self.head_embedding = MLPLayer(1,hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
       # self.shape_embedding = MLPLayer(2, hidden_dim, hidden_dim)
        #self.offset_embedding = MLPLayer(self.pos_dim + 1, hidden_dim, hidden_dim)

        self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, self.n_token_entry )
        self.head_decoder = MLPLayer(hidden_dim, hidden_dim,self.token_processor.n_token_entry_head )
        self.offset_head_decoder = MLPLayer(hidden_dim, hidden_dim,self.token_processor.offset_tokenizer.grid_size)  # offset to offset
        self.shape_head_decoder = MLPLayer(hidden_dim, hidden_dim, self.token_processor.shape_grid.shape[0])

    def padding(self,pos,heading,feature,batch,batch_num):
        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def embed_input(self,initial_pos, initial_heading,initial_type,initial_shape,batch,batch_num ):
        type_embedding = self.type_embedding(initial_type)
        heading_embedding = self.head_embedding(initial_heading[:,None])
        pos_embedding = self.pos_embedding(initial_pos)
       # shape_embedding = self.shape_embedding(initial_shape)

        feat_a = type_embedding  + heading_embedding + pos_embedding#+ shape_embedding

        # if initial_offset_xyh.any():
        #     offset_embedding = self.offset_embedding(initial_offset_xyh)
        #     feat_a = feat_a + offset_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch,batch_num)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        return pos_a_b, heading_a_b, feat_a_b,mask_a_b

    def graph_embed(self,feat_a_b, pos_a_b, heading_a_b, mask_a_b,batch_s_repeat, feat_map,  pos_pl,    orient_pl,batch_pl):

        res=torch.zeros_like(feat_a_b)

        head_a=heading_a_b[mask_a_b]
        pos_a=pos_a_b[mask_a_b]
        feat_a=feat_a_b[mask_a_b]
        mask_a=None

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
            pos_pl=pos_pl,  # [n_pl, 2]
            orient_pl=orient_pl,  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask_a,  # [n_agent, n_step]
            batch_s=batch_s_repeat,  # [n_agent,n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
            pl2a_radius=100,
            max_num_neighbors=100,
            agent_train_mask=None,
            layer_num=1
        )

        feat_a = self.pt2a_attn_layers[0]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

        res[mask_a_b]=feat_a

        return res

    def forward(self,map_feature, tokenized_agent):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        batch_num=tokenized_agent["num_graphs"]

        if self.use_entry_former:
            pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)
            pred_mask = tokenized_agent["initial_ego_mask"]
            ego_position=tokenized_agent["initial_pos"][pred_mask]
            ego_heading=tokenized_agent["initial_heading"][pred_mask]

            pos_pl,orient_pl=transform_to_local(pos_pl,
                               orient_pl,
                               ego_position,
                               ego_heading,
                               )

        map_mask = torch.any(feat_map != 0, dim=-1)

        if self.training:
            pred_mask = ~tokenized_agent["ego_mask"]#non-last mask
            iteration_num=1
        else:
            pred_mask = tokenized_agent["initial_ego_mask"]
            ego_position=tokenized_agent["initial_pos"][pred_mask]
            ego_heading=tokenized_agent["initial_heading"][pred_mask]

            lengths = torch.bincount(tokenized_agent["batch"]).tolist()

            all_initial_type = padding(tokenized_agent["initial_type"], lengths, padding_value=-1)  # b, n, d

            agent_mask=all_initial_type!=-1

            all_initial_type[~agent_mask]=0

            iteration_num=all_initial_type.shape[1]-1

            self.attr_former.attn.caching = True

            if self.use_entry_former:
                self.entry_former.attn.caching = True

        initial_type = tokenized_agent["initial_type"][pred_mask]
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

        local_pos_list=[initial_pos]
        local_heading_list=[initial_heading]
        shape_list=[initial_shape]

        type_list=[initial_type]

        for n_current in range(iteration_num):

            pos_a_b, heading_a_b, feat_a_b, mask_a_b=self.embed_input(initial_pos, initial_heading,initial_type,initial_shape,batch,batch_num)

            if self.use_entry_former:
                entry_feature = self.entry_former.cross_attention(feat_a_b, torch.zeros_like(pos_a_b),
                                                                  torch.zeros_like(heading_a_b), mask_a_b,
                                                                  feat_map,
                                                                  pos_pl,
                                                                  orient_pl, map_mask)
            else:
                entry_feature = self.graph_embed(feat_a_b, torch.zeros_like(pos_a_b),
                                                  torch.zeros_like(heading_a_b), mask_a_b,
                                                  batch,
                                                  feat_map,
                                                  pos_pl,
                                                  orient_pl, batch_pl)


            n_agent=feat_a_b.shape[1]

            attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b,n_agent, n_current, mask_a_b)

            attr_feature=attr_feature[mask_a_b]

            pos_logit = self.pos_decoder(attr_feature)
            head_logit = self.head_decoder(attr_feature)
            shape_logit = self.shape_head_decoder(attr_feature)

            if not self.use_refine:
                offset_logit = self.offset_head_decoder(attr_feature)

                # initial_offset_xyh=self.token_processor.attr_tokenizer1.decode_pos(initial_offset_idx,0,0,0)

                # initial_offset_xyh[...,:2] = torch.tanh(initial_offset_xyh[...,:2]) * self.token_processor.attr_tokenizer.grid_interval/2
                # initial_offset_xyh[...,2] = torch.tanh(initial_offset_xyh[...,2]) * (torch.pi/self.token_processor.n_token_entry_head)
            else:
                initial_offset_xyh = torch.zeros_like(initial_shape)

            entry_logit=(pos_logit,head_logit,offset_logit,shape_logit)

            if not self.training:
                initial_pos_token = Categorical(logits=pos_logit).sample()
                initial_heading_token=Categorical(logits=head_logit).sample()
                initial_offset_token=Categorical(logits=offset_logit).sample()
                initial_shape_token=Categorical(logits=shape_logit).sample()

                initial_pos= self.token_processor.attr_tokenizer.grid[initial_pos_token]
                token_offset = self.token_processor.offset_tokenizer.grid[initial_offset_token]

                initial_pos=initial_pos+token_offset
                
                initial_heading = self.token_processor.attr_tokenizer.decode_heading(initial_heading_token)

                initial_shape=self.token_processor.shape_grid[initial_shape_token]

                local_pos_list.append(initial_pos)
                local_heading_list.append(initial_heading)
                shape_list.append(initial_shape)

                initial_type=all_initial_type[:,n_current+1]

                type_list.append(initial_type)

        if not self.training:
            self.attr_former.attn.kv_caching(0)
            if self.use_entry_former:
                self.entry_former.attn.kv_caching(0)

            local_pos = torch.stack(local_pos_list,dim=1)[agent_mask]#32,125
            local_heading = torch.stack(local_heading_list,dim=1)[agent_mask]
            shape = torch.stack(shape_list,dim=1)[agent_mask]
            
            global_pos,global_heading=transform_to_global(
                local_pos[:,None],
                local_heading[:,None],
                ego_position[tokenized_agent["batch"]],
                ego_heading[tokenized_agent["batch"]],
            )

            shape = torch.cat([shape, torch.zeros_like(shape[:, :1]) + 1.75], dim=-1)

            tokenized_agent["shape"]=shape
            tokenized_agent["ego_mask"] = tokenized_agent["initial_ego_mask"]
            tokenized_agent["type"] = tokenized_agent['initial_type']
            tokenized_agent['id']=tokenized_agent['initial_id']

            return global_pos, global_heading

        return entry_logit
        # after interact with agent and map,  predict state and type and shape and tokenized position,
        # then refine predict head token and offset_xy ,
        # #then predict all agent motion
