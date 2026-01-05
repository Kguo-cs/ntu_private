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


def box_corners(center, heading, shape):
    """
    center : [K, 2]
    heading: [K]
    shape  : [K, 2]  (length, width)
    return : [K, 5, 2] closed polygons
    """
    l = shape[:, 0] / 2
    w = shape[:, 1] / 2

    # local box corners
    corners = torch.stack([
        torch.stack([ l,  w], dim=1),
        torch.stack([ l, -w], dim=1),
        torch.stack([-l, -w], dim=1),
        torch.stack([-l,  w], dim=1),
        torch.stack([ l,  w], dim=1),
    ], dim=1)  # [K, 5, 2]

    c = torch.cos(heading)
    s = torch.sin(heading)

    R = torch.stack([
        torch.stack([ c, -s], dim=1),
        torch.stack([ s,  c], dim=1),
    ], dim=1)  # [K, 2, 2]

    return corners @ R.transpose(1, 2) + center[:, None, :]

class InitDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:

        super(InitDecoder, self).__init__()
        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

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

        self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.8,
                                         hist_len=self.entry_his_len)        # drop 01 is important

        self.use_refine=False

        if self.use_refine:

            self.refine_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1,
                                             hist_len=self.entry_his_len)        # drop 01 is important

        self.sequential=False

        self.use_offset=False

        self.pos_embedding = MLPLayer(2 ,hidden_dim, hidden_dim)
        self.head_embedding = MLPLayer(1,hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.shape_embedding = MLPLayer(2, hidden_dim, hidden_dim)
        self.offset_embedding = MLPLayer(2, hidden_dim, hidden_dim)

        self.type_count_embedding = MLPLayer(3,hidden_dim, hidden_dim)

        self.pos_decoder = MLPLayer(hidden_dim, hidden_dim, token_processor.n_token_entry )
        self.head_decoder = MLPLayer(hidden_dim, hidden_dim,token_processor.n_token_entry_head )
        self.offset_head_decoder = MLPLayer(hidden_dim, hidden_dim,token_processor.offset_tokenizer.grid_size)  # offset to offset
        self.shape_head_decoder = MLPLayer(hidden_dim, hidden_dim, token_processor.shape_grid.shape[0])

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
        shape_embedding = self.shape_embedding(initial_shape[:,:2])

        feat_a = type_embedding  + heading_embedding + pos_embedding+ shape_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch,batch_num)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        # pos_a_b=torch.zeros_like(pos_a_b)
        # heading_a_b=torch.zeros_like(heading_a_b)

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
        batch=tokenized_agent["batch"]

        batch_num=tokenized_agent["num_graphs"]
        lengths = torch.bincount(batch).tolist()

        all_initial_type = padding(tokenized_agent["initial_type"], lengths, padding_value=3)  # b, n, d

        type_count = torch.nn.functional.one_hot(
            all_initial_type,
            num_classes=4
        ).sum(dim=1).to(torch.float32)

        type_count_feature= self.type_count_embedding(type_count[:,:3])

        if self.use_entry_former:
            pred_mask = tokenized_agent["initial_ego_mask"]
            ego_position=tokenized_agent["global_initial_pos"][pred_mask][:,0]
            ego_heading=tokenized_agent["global_initial_heading"][pred_mask][:,0]

            # ego_p=tokenized_agent["initial_pos"][tokenized_agent["ego_mask"]]
            # ego_h=tokenized_agent["initial_heading"][tokenized_agent["ego_mask"]]


            pos_pl,orient_pl=transform_to_local(pos_pl[:,None],
                               orient_pl[:,None],
                               ego_position[batch_pl],
                               ego_heading[batch_pl],
                               )

            pos_pl=pos_pl[:,0]
            orient_pl=orient_pl[:,0]

            ego_dist=torch.linalg.norm(pos_pl,dim=-1)

            ego_dist_mask=ego_dist<100

            pos_pl=pos_pl[ego_dist_mask]
            orient_pl=orient_pl[ego_dist_mask]
            batch_pl=batch_pl[ego_dist_mask]
            feat_map=feat_map[ego_dist_mask]

            pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)

            map_mask = torch.any(feat_map != 0, dim=-1)

            feat_map=feat_map+type_count_feature[:,None]

        if self.training:
            pred_mask = ~tokenized_agent["ego_mask"]#non-last mask

            if self.sequential:
                pred_mask=torch.ones_like(pred_mask)
            iteration_num=1
        else:
            pred_mask = tokenized_agent["initial_ego_mask"]

            agent_mask=all_initial_type!=3

            all_initial_type[~agent_mask]=0

            iteration_num=all_initial_type.shape[1]-1

            self.attr_former.attn.caching = True

            if self.use_entry_former:
                self.entry_former.attn.caching = True

        initial_type = tokenized_agent["initial_type"][pred_mask]
        initial_shape = tokenized_agent["initial_shape"][pred_mask]
        batch=batch[pred_mask]
        token_initial_pos = tokenized_agent["token_initial_pos"][pred_mask]
        token_initial_heading = tokenized_agent["token_initial_heading"][pred_mask]
        initial_pos_token = tokenized_agent["initial_pos_token"][pred_mask]
        initial_offset_token = tokenized_agent["initial_offset_token"][pred_mask]

        local_pos_list=[token_initial_pos]
        local_heading_list=[token_initial_heading]
        shape_list=[initial_shape]
        type_list=[initial_type]

        #each agent's type, position token, size and heading , trajectory tokens
        #This is achieved by estimating the agent’s recent movement based on its position, velocity, and heading, and then comparing it with the candidate trajectory tokens to select the best match
        #token initialization by speed

        if self.sequential:
            type_embedding = self.type_embedding(initial_type)
            heading_embedding = self.head_embedding(token_initial_heading[:,None])
            shape_embedding = self.shape_embedding(initial_shape[:,:2])
            pos_embedding = self.pos_embedding(self.token_processor.attr_tokenizer.grid[initial_pos_token])
            offset_embedding = self.offset_embedding(self.token_processor.offset_tokenizer.grid[initial_offset_token])

            feat1=type_embedding+pos_embedding

            feat2=feat1+heading_embedding+shape_embedding

            features=torch.stack([feat1,feat2,feat2+offset_embedding],dim=1)

            pos_a_b, heading_a_b, feat_a_b = self.padding(token_initial_pos[:,None], token_initial_heading[:,None], features, batch,batch_num)

            feat_a_b=feat_a_b.flatten(1,2)
            pos_a_b=pos_a_b.repeat(1,1,3,1).flatten(1,2)
            heading_a_b=heading_a_b.repeat(1,1,3).flatten(1,2)

            if self.training:
                feat_a_b=feat_a_b[:,2:-1]
                pos_a_b=pos_a_b[:,2:-1]
                heading_a_b=heading_a_b[:,2:-1]
            else:
                feat_a_b=feat_a_b[:,2:]
                pos_a_b=pos_a_b[:,2:]
                heading_a_b=heading_a_b[:,2:]

            mask_a_b = torch.any(feat_a_b != 0, dim=-1)
            pos_a_b=torch.zeros_like(pos_a_b)
            heading_a_b=torch.zeros_like(heading_a_b)

            if not self.training:
                iteration_num=iteration_num*3



        for n_current in range(iteration_num):

            if not self.sequential:
                pos_a_b, heading_a_b, feat_a_b, mask_a_b=self.embed_input( token_initial_pos, token_initial_heading,initial_type,initial_shape,batch,batch_num)

            # from matplotlib.collections import LineCollection
            # import matplotlib.pyplot as plt
            # for i in range(len(pos_a_b)):
            #     centers=pos_a_b[i][mask_a_b[i]]
            #     headings=heading_a_b[i][mask_a_b[i]]
            #     shapes=initial_shape[batch==i]
            #     polygons = box_corners(centers, headings, shapes)
            #     polygons = polygons.cpu().numpy()  # matplotlib needs numpy
            #
            #     fig, ax = plt.subplots(figsize=(8, 6))
            #
            #     lines = LineCollection(
            #         polygons,
            #         colors="tab:blue",
            #         linewidths=1.0,
            #         alpha=0.7,
            #     )
            #
            #     pos_pl_i=pos_pl[i][map_mask[i]]
            #     orient_pl_i=orient_pl[i][map_mask[i]]
            #
            #     dx = torch.cos(orient_pl_i)
            #     dy = torch.sin(orient_pl_i)
            #
            #     ax.quiver(
            #         pos_pl_i[:, 0].cpu(),
            #         pos_pl_i[:, 1].cpu(),
            #         dx.cpu(),
            #         dy.cpu(),
            #         angles="xy",
            #         scale_units="xy",
            #         scale=5,
            #         color="red",
            #         width=0.01,
            #     )
            #
            #     ax.add_collection(lines)
            #     ax.autoscale()
            #     ax.set_aspect("equal")
            #
            #     ax.set_xlabel("x")
            #     ax.set_ylabel("y")
            #     ax.set_title("All Oriented Bounding Boxes")
            #
            #     plt.show()
            #
            #     print(1)

            if self.use_entry_former:
                entry_feature = self.entry_former.cross_attention(feat_a_b, torch.zeros_like(pos_a_b),
                                                                  torch.zeros_like(heading_a_b), mask_a_b,
                                                                  feat_map,
                                                                  pos_pl,
                                                                  orient_pl, map_mask)
            else:
                entry_feature = self.graph_embed(feat_a_b, pos_a_b,
                                                  heading_a_b, mask_a_b,
                                                  batch,
                                                  feat_map,
                                                  pos_pl,
                                                  orient_pl, batch_pl)

            n_agent=feat_a_b.shape[1]

            attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b,n_agent, n_current, mask_a_b)


            if self.sequential:

                if self.training:
                    agent_mask=mask_a_b[:,1::3]

                    pos_logit = self.pos_decoder(attr_feature[:,::3][agent_mask])      #offset feature predict pos
                    head_logit = self.head_decoder(attr_feature[:,1::3][agent_mask])
                    shape_logit = self.shape_head_decoder(attr_feature[:,1::3][agent_mask])
                    offset_logit = self.offset_head_decoder(attr_feature[:,2::3][agent_mask])#last_feature offset first and remove last

                    entry_logit = (pos_logit, head_logit, offset_logit, shape_logit)
                else:
                    if n_current%3==0:
                        pos_logit = self.pos_decoder(attr_feature)  # offset feature predict pos
                        initial_pos_token = Categorical(logits=pos_logit).sample()
                        token_initial_pos1= self.token_processor.attr_tokenizer.grid[initial_pos_token]
                        initial_type = all_initial_type[:, n_current//3 + 1]

                        type_embedding = self.type_embedding(initial_type[:,None])

                        feat_a_b = self.pos_embedding(token_initial_pos1)+type_embedding
                    elif n_current%3==1:
                        head_logit = self.head_decoder(attr_feature)
                        shape_logit = self.shape_head_decoder(attr_feature)
                        initial_heading_token=Categorical(logits=head_logit/0.1).sample()
                        initial_shape_token=Categorical(logits=shape_logit/0.1).sample()
                        token_initial_heading = self.token_processor.attr_tokenizer.decode_heading(initial_heading_token)
                        initial_shape=self.token_processor.shape_grid[initial_shape_token]

                        local_heading_list.append(token_initial_heading[:,0])
                        shape_list.append(initial_shape[:,0])

                        heading_embedding = self.head_embedding(token_initial_heading[:, None])
                        shape_embedding = self.shape_embedding(initial_shape[:, :2])
                        feat_a_b = feat_a_b+heading_embedding+shape_embedding
                    else:
                        offset_logit = self.offset_head_decoder(attr_feature)
                        initial_offset_token = Categorical(logits=offset_logit/0.1).sample()
                        token_initial_offset = self.token_processor.offset_tokenizer.grid[initial_offset_token]
                        token_initial_pos=token_initial_pos1+token_initial_offset

                        offset_embedding = self.offset_embedding(token_initial_offset)

                        local_pos_list.append(token_initial_pos[:,0])

                        feat_a_b = feat_a_b + offset_embedding
            else:
                attr_feature=attr_feature[mask_a_b]
                pos_logit = self.pos_decoder(attr_feature)
                head_logit = self.head_decoder(attr_feature)
                shape_logit = self.shape_head_decoder(attr_feature)
                offset_logit = self.offset_head_decoder(attr_feature)

                entry_logit=(pos_logit,head_logit,offset_logit,shape_logit)

                if not self.training:
                    initial_pos_token = Categorical(logits=pos_logit).sample()
                    initial_heading_token=Categorical(logits=head_logit/0.1).sample()
                    #initial_offset_token=Categorical(logits=offset_logit/0.1).sample()
                    initial_shape_token=Categorical(logits=shape_logit/0.1).sample()

                    token_initial_pos= self.token_processor.attr_tokenizer.grid[initial_pos_token]
                    #token_offset = self.token_processor.offset_tokenizer.grid[initial_offset_token]

                    #token_initial_pos=token_initial_pos+token_offset
                    #
                    token_initial_heading = self.token_processor.attr_tokenizer.decode_heading(initial_heading_token)

                    initial_shape=self.token_processor.shape_grid[initial_shape_token]

                    local_pos_list.append(token_initial_pos)
                    local_heading_list.append(token_initial_heading)
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
