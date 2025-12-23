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
class EntryDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
            start_step
    ) -> None:

        super(EntryDecoder, self).__init__()
        self.autoregressive_entry= token_processor.autoregressive_entry
        self.n_token_entry=token_processor.n_token_entry

        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.start_step=start_step

        if self.token_processor.use_bird:
            self.pos_dim=3
        else:
            self.pos_dim=2

        if self.autoregressive_entry:
            self.entry_his_len=1000000

            self.start_embedding =nn.Embedding(1, hidden_dim)

            self.use_one_feature= False

            self.use_cross_attention= True

            self.use_entry_former=True

            if self.token_processor.use_bird:
                self.num_levels=3#self.token_processor.tokenizer.num_levels
            else:
                self.num_levels=5


            if self.token_processor.use_bird:
                self.max_entry=140
            else:
                self.max_entry=18

            if self.use_one_feature or self.use_cross_attention:

                if self.use_entry_former:
                    self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)#replace with gnn

                # num_layers=1
                # head_dim=hidden_dim//num_heads
                # self.edge_encoder = EdgeEncoder(hidden_dim,
                #                                 num_freq_bands,
                #                                 share=False,
                #                                 hist_drop_prob=0.1,
                #                                 time_span=0,
                #                                 a2a=False,
                #                                 shift=token_processor.shift,
                #                                 use_route=token_processor.use_route,
                #                                 discriminator=False,
                #                                 use_bird=token_processor.use_bird,
                #                                 use_cross=True
                #                                 )
                # self.a2entry_attn_layers = nn.ModuleList(
                #     [
                #         AttentionLayer(
                #             hidden_dim=hidden_dim,
                #             num_heads=num_heads,
                #             head_dim=head_dim,
                #             dropout=0,
                #             bipartite=True,
                #             has_pos_emb=True,
                #             #  gated_attention=discriminator,
                #         )
                #         for _ in range(num_layers)
                #     ]
                # )

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1, hist_len=self.entry_his_len)
            #drop 01 is important

            self.pos_embedding = nn.Embedding(self.n_token_entry+1, hidden_dim)

            self.head_embedding  = nn.Embedding(self.token_processor.n_token_entry_head, hidden_dim)

            if self.token_processor.token_offset:
                self.offset_embedding =nn.Embedding(self.token_processor.n_token_offset, hidden_dim)
                self.offset_head_decoder = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                    output_dim=self.token_processor.n_token_offset)  # offset to offset

            else:
                self.offset_embedding=MLPLayer(self.pos_dim+1,hidden_dim,hidden_dim)
                self.offset_head_decoder = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                    output_dim=self.pos_dim+1)  # offset to offset


            if not self.token_processor.use_bird:
                self.type_embedding = nn.Embedding(3, hidden_dim)
                self.shape_embedding = MLPLayer(input_dim=3, hidden_dim=hidden_dim,    output_dim=hidden_dim)

                self.type_head_decoder = MLPLayer( input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)

                self.shape_head_decoder = MLPLayer( input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3 )

            self.task_embedding = nn.Embedding(self.num_levels+1, hidden_dim)

            self.number_embedding = MLPLayer(1,hidden_dim, hidden_dim)


            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_entry_head
                    )

        else:
            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim+self.pos_dim+4, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_entry_head
                    )

            self.pos_offset_predict_head = MLPLayer(input_dim=hidden_dim + self.pos_dim + 5, hidden_dim=hidden_dim,
                                                    output_dim=self.pos_dim + 1)

            if not self.token_processor.use_bird:
                # self.type_head=MLPLayer(input_dim=hidden_dim+self.pos_dim+1, hidden_dim=hidden_dim, output_dim=3)
                self.shape_head=MLPLayer(input_dim=hidden_dim+1, hidden_dim=hidden_dim, output_dim=3)
                self.pos_head=MLPLayer(input_dim=hidden_dim+4, hidden_dim=hidden_dim, output_dim=512)

        self.entry_decoder = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.n_token_entry+1
        )

    def pred_entry(self,attr_all_feature,all_pos, all_head,agent_n,n_current=0,tgt_mask=None):

        n_step=attr_all_feature.shape[1]

        entry_num=n_current+n_step-agent_n

        attr_mask = torch.any(attr_all_feature != 0, dim=-1)

        step = torch.arange(n_step,device=all_pos.device)+n_current

        task=(step-agent_n)%self.num_levels

        number=(step-agent_n)//self.num_levels #step-agent_n#

        task[step<agent_n]=self.num_levels

        number_embedding=self.number_embedding(number.float()[:,None])

        attr_all_feature=attr_all_feature+self.task_embedding(task)[None]#+number_embedding[None]

        attr_all_feature[:,-entry_num:]=attr_all_feature[:,-entry_num:]+number_embedding[None,-entry_num:]

        entry_mask = attr_mask[:, -entry_num:]


        if self.use_cross_attention:

            entry_feature = attr_all_feature[:, -entry_num:]
            entry_pos = all_pos[:, -entry_num:]
            entry_head = all_head[:, -entry_num:]

            feat_map,pos_pl,orient_pl,batch_pl,tgt_mask=tgt_mask

            if self.use_entry_former:
                entry_feature = self.entry_former.cross_attention(entry_feature, entry_pos,
                                                             entry_head, entry_mask,
                                                             attr_all_feature[:, :-entry_num],
                                                              all_pos[:, :-entry_num],
                                                             all_head[:, :-entry_num],  tgt_mask)

            # pos_a=entry_pos[entry_mask][:,None]
            # head_a=entry_head[entry_mask][:,None]
            # batch_s=torch.arange(len(entry_feature),device=entry_feature.device)[:,None].repeat(1,entry_feature.shape[1])[entry_mask][:,None]
            #
            # head_vector_a=torch.stack([head_a.cos(), head_a.sin()], dim=-1)
            # mask_a=torch.ones_like(head_a).to(torch.bool)
            #
            # edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
            #     pos_pl=pos_pl,  # [n_pl, 2]
            #     orient_pl=orient_pl,  # [n_pl]
            #     pos_a=pos_a,  # [n_agent, n_step, 2]
            #     head_a=head_a,  # [n_agent, n_step]
            #     head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            #     mask=mask_a,  # [n_agent, n_step]
            #     batch_s=batch_s,  # [n_agent,n_step]
            #     batch_pl=batch_pl,  # [n_pl*n_step]
            #     pl2a_radius=100,
            #     max_num_neighbors=10,
            #     agent_train_mask=None,
            #     layer_num=1
            # )
            #
            # feat_entry=entry_feature[entry_mask]

            # feat_entry = self.a2entry_attn_layers[0]((feat_map, feat_entry), r_pl2a,
            #                                         edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst
            #
            #
            # entry_feature=torch.zeros_like(entry_feature)
            #
            # entry_feature[entry_mask]=feat_entry

            if n_current!=0:
                n_current=n_current-agent_n

            attr_feature = self.attr_former.temporal_embed(entry_feature, entry_pos, entry_head,
                                                           entry_feature.shape[1], n_current, entry_mask)

        else:

            attr_feature = self.attr_former.temporal_embed(attr_all_feature, all_pos, all_head,
                                                            n_step, n_current, attr_mask,use_time=False)

            attr_feature=attr_feature[:,-entry_num:]

        task=task[-entry_num:]

        pos_mask= (task==0)
        if self.token_processor.token_offset:

            head_mask=(task==2)

            offset_mask=(task==1)
        else:
            head_mask = (task == 1)

            offset_mask = (task == 4)

        entry_logit = self.entry_decoder(attr_feature[:,pos_mask])

        entry_head_logit = self.entry_head_decoder(attr_feature[:,head_mask])

        entry_offset = self.offset_head_decoder(attr_feature[:,offset_mask])

        if not self.token_processor.use_bird:
            type_mask = (task == 2)

            shape_mask = (task == 3)

            type_logit = self.type_head_decoder(attr_feature[:,type_mask])
            pred_shape = self.shape_head_decoder(attr_feature[:,shape_mask])
        else:
            type_logit=pred_shape=None

        return entry_logit,entry_head_logit,entry_offset,type_logit,pred_shape


    def auto_pred(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,edge_index_a2a=None,n_current=0):
        if self.training:
            mask_a = mask_a[:, self.start_step:]
            pos_a = pos_a[:, self.start_step:]
            head_a = head_a[:, self.start_step:]
        else:
            mask_a = mask_a[:, -1:]
            pos_a = pos_a[:, -1:]
            head_a = head_a[:, -1:]

        n_agent, n_step = mask_a.shape

        mask_ta = mask_a.transpose(0, 1)

        feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

        feat_a_t[mask_ta] = feat_a#.detach()
        batch = tokenized_agent["batch"]
        batch_num = batch.max() + 1
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        entry_state = torch.zeros([n_step * batch_num, 1, self.pos_dim + 1], device=feat_a.device)
        entry_state[:, :, 0] = -78
        if self.token_processor.use_bird:
            entry_state[:, :, 1] = 36
            entry_state[:, :, 2] = 32

        padding_pos = padding(pos_a, lengths, padding_value=0).permute(2, 0, 1, 3).flatten(0, 1)  # T,b, n, d
        padding_heading = padding(head_a, lengths, padding_value=0).permute(2, 0, 1).flatten(0, 1)
        padding_features = padding(feat_a_t.transpose(0, 1), lengths, padding_value=0).permute(2, 0, 1, 3).flatten(0, 1)

        if not self.token_processor.use_bird:
            ego_mask = tokenized_agent["ego_mask"]
            ego_pos = pos_a[ego_mask].transpose(0, 1).flatten(0, 1)
            ego_heading = head_a[ego_mask].transpose(0, 1).flatten(0, 1)

            padding_pos, padding_heading = transform_to_local(
                padding_pos,  # [n_agent, n_step, 2]
                padding_heading,  # [n_agent, n_step]
                ego_pos,  # [n_agent, 2]
                ego_heading,  # [n_agent]
            )

        agent_n = padding_features.shape[1]

        current_pos = torch.cat([padding_pos, entry_state[..., :-1]], dim=1)

        current_heading = torch.cat([padding_heading, entry_state[..., -1]], dim=1)

        entry_embedding = self.start_embedding.weight[None].repeat(len(padding_features), 1, 1)

        entry_feature = torch.cat([padding_features, entry_embedding], dim=1)

        tgt_mask = torch.any(padding_features != 0, dim=-1)

        batch_a = torch.arange(len(padding_features), device=padding_features.device)[:, None].repeat(1,
                                                                                                      padding_features.shape[
                                                                                                          1])

        tgt_mask = (feat_a, padding_pos[tgt_mask], padding_heading[tgt_mask], batch_a[tgt_mask], tgt_mask)

        if self.training:
            pos_idx = tokenized_agent["pos_idx"]
            head_idx = tokenized_agent["head_idx"]
            offset = tokenized_agent["offset"]

            entry_mask = pos_idx != self.token_processor.n_token_entry

            pos_feature = self.pos_embedding(pos_idx)
            heading_feature = self.head_embedding(head_idx)
            offset_feature = self.offset_embedding(offset)

            pos_feature[entry_mask] = 0
            heading_feature[entry_mask] = 0
            offset_feature[entry_mask] = 0

            if not self.token_processor.use_bird:
                entry_type = tokenized_agent["entry_type"]
                entry_shape = tokenized_agent["entry_shape"]

                type_feature = self.type_embedding(entry_type)
                shape_feature = self.shape_embedding(entry_shape)
                type_feature[entry_mask] = 0
                shape_feature[entry_mask] = 0

                attr_feature = torch.stack([pos_feature, heading_feature, type_feature, shape_feature, offset_feature],
                                           dim=2)
            else:
                attr_feature = torch.stack([pos_feature, heading_feature, offset_feature], dim=2)

            attr_feature = attr_feature.flatten(1, 2)

            attr_all_feature = torch.cat([entry_feature, attr_feature], dim=1)

            entry_pos = torch.zeros([pos_idx.shape[0], attr_feature.shape[1], current_pos.shape[-1]],
                                    device=pos_idx.device)
            entry_head = torch.zeros([pos_idx.shape[0], attr_feature.shape[1]], device=pos_idx.device)

            if self.token_processor.token_offset:
                offset_idx1 = torch.clamp(offset, max=self.token_processor.n_token_entry - 1)

                entry_offset = self.token_processor.offset_token[offset_idx1]
            else:
                entry_offset = offset[:, :, :self.pos_dim]

            pos_idx_clip = torch.clamp(pos_idx, max=self.token_processor.n_token_entry - 1)
            token_pos = self.token_processor.entry_pos_token[pos_idx_clip]

            total_pos = entry_offset + token_pos

            entry_pos[:, ::self.num_levels] = token_pos
            if self.token_processor.token_offset:
                entry_pos[:, 1::self.num_levels] = total_pos
            else:
                entry_pos[:, 1::self.num_levels] = token_pos
                tokenized_heading = self.token_processor.decode_head(head_idx)
                entry_head[:, 1::self.num_levels] = tokenized_heading
                if not self.token_processor.use_bird:
                    entry_pos[:, 2::self.num_levels] = token_pos
                    entry_head[:, 2::self.num_levels] = tokenized_heading
                    entry_pos[:, 3::self.num_levels] = token_pos
                    entry_head[:, 3::self.num_levels] = tokenized_heading

                entry_head[:, self.num_levels - 1::self.num_levels] = wrap_angle(
                    tokenized_heading + offset[:, :, self.pos_dim])

            entry_pos[:, self.num_levels - 1::self.num_levels] = total_pos

            # # entry_idx_all =entry_idx.reshape(entry_idx.shape[0],-1,self.num_levels)
            # entry_pos=[]
            # entry_head=[]

            # for l in range(1,self.num_levels+1):
            # pos_rec, heading_rec = self.token_processor.tokenizer.decode_tokens_to_state(entry_idx_all[:,:,:l])

            #     entry_pos.append(pos_rec)
            #     entry_head.append(heading_rec)
            #
            # entry_pos=torch.stack(entry_pos,dim=2).flatten(1,2)
            # entry_head=torch.stack(entry_head,dim=2).flatten(1,2)

            if self.use_one_feature:
                agent_n = 0
            else:
                entry_pos = torch.cat([current_pos, entry_pos], dim=1)
                entry_head = torch.cat([current_heading, entry_head], dim=1)

            entry_logit = self.pred_entry(attr_all_feature, entry_pos, entry_head, agent_n, tgt_mask=tgt_mask)


        else:
            self.attr_former.attn.caching = True

            if self.use_entry_former:
                self.entry_former.attn.caching = True

            entry_state_list = []
            entry_type_list = []
            entry_shape_list = []

            finish = torch.zeros_like(current_heading[:, 0]).to(torch.bool)

            if self.use_one_feature:
                agent_n = 0
                current_pos = current_pos[:, -1:]
                current_heading = current_heading[:, -1:]

            while True:

                entry_logit, entry_head_logit, entry_offset, type_logit, pred_shape = self.pred_entry(entry_feature,
                                                                                                      current_pos,
                                                                                                      current_heading,
                                                                                                      agent_n,
                                                                                                      n_current,
                                                                                                      tgt_mask)

                if n_current == 0:
                    self.attr_former.attn.kv_caching(self.entry_his_len, n_current)
                    if self.use_entry_former:
                        self.entry_former.attn.kv_caching(self.entry_his_len, n_current)
                    # current_pos = current_pos[:, -1:]
                    current_heading = current_heading[:, -1:]
                    n_current = n_current + agent_n

                if entry_logit.shape[1] != 0:
                    pos_idx = Categorical(logits=entry_logit).sample()

                    finish = finish | (pos_idx[:, 0] == self.n_token_entry)

                    if finish.all() or len(entry_state_list) == self.max_entry:
                        if len(entry_state_list) == 0:
                            entry_logit=None
                            break
                        entry_logit = torch.stack(entry_state_list, dim=1)
                        entry_type = torch.cat(entry_type_list, dim=1)
                        entry_shape = torch.cat(entry_shape_list, dim=1)
                        entry_logit = (entry_logit, entry_type, entry_shape)
                        break

                    pos_idx[pos_idx == self.n_token_entry] = 0

                    token_pos = self.token_processor.entry_pos_token[pos_idx]

                    entry_feature = self.pos_embedding(pos_idx)

                    current_pos = token_pos

                    current_heading = torch.zeros_like(current_heading)

                elif entry_head_logit.shape[1] != 0:
                    entry_head_idx = Categorical(logits=entry_head_logit).sample()

                    tokenized_heading = self.token_processor.decode_head(entry_head_idx)

                    entry_feature = self.head_embedding(entry_head_idx)

                    if self.token_processor.token_offset:

                        heading_rec = tokenized_heading  # +entry_offset[:,0,3:]

                        new_state = torch.cat([pos_rec, heading_rec], dim=-1)

                        new_state[finish] = 0

                        entry_state_list.append(new_state)
                    else:
                        current_heading = tokenized_heading

                elif entry_offset.shape[1] != 0:
                    if self.token_processor.token_offset:
                        offset_idx = Categorical(logits=entry_offset).sample()

                        entry_offset = self.token_processor.offset_token[offset_idx]

                    pos_rec = token_pos[:, 0] + entry_offset[:, 0, :self.pos_dim]

                    current_pos = pos_rec[:, None]

                    if self.token_processor.token_offset:
                        entry_feature = self.offset_embedding(offset_idx)
                    else:
                        entry_feature = self.offset_embedding(entry_offset)

                        heading_rec = wrap_angle(tokenized_heading + entry_offset[:, 0, self.pos_dim:])

                        new_state = torch.cat([pos_rec, heading_rec], dim=-1)

                        new_state[finish] = 0

                        entry_state_list.append(new_state)
                        current_heading = heading_rec
                elif type_logit.shape[1] != 0:
                    type_idx = Categorical(logits=type_logit).sample()
                    entry_feature = self.type_embedding(type_idx)
                    entry_type_list.append(type_idx)
                else:
                    entry_feature = self.shape_embedding(pred_shape)
                    entry_shape_list.append(pred_shape)

                n_current += 1

            self.attr_former.attn.kv_caching(0)
            if self.use_entry_former:
                self.entry_former.attn.kv_caching(0)

        return entry_logit

    def para_pred(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,edge_index_a2a=None,n_current=0):
        #feat_a = feat_a.detach()

        entry_logit = self.entry_decoder(feat_a)
        if self.training:
            entry_idx = tokenized_agent["entry_idx"][:, 1 + self.start_step:].transpose(0, 1).flatten(0, 1)[
                mask_a[:, self.start_step:].transpose(0, 1).flatten(0, 1)]
        else:
            entry_idx = Categorical(logits=entry_logit).sample()
            # entry_idx = tokenized_agent["entry_idx"][:,1]
            # mask=tokenized_agent["valid_mask"][:,0]
            # entry_idx=entry_idx[mask]

        entry_mask = (entry_idx < self.n_token_entry)

        entry_type=entry_idx[entry_mask]
        entry_feature = feat_a[entry_mask]

        feat_type = torch.cat([entry_feature, entry_type[:, None]], dim=-1)

        pred_shape = torch.relu(self.shape_head(feat_type))+0.5

        if self.training:
            entry_shape = tokenized_agent["entry_shape"]
        else:
            entry_shape = pred_shape

        feat_type_shape = torch.cat([feat_type, entry_shape], dim=-1)

        type_logit=self.pos_head(feat_type_shape)

        if self.training:
            entry_pos=tokenized_agent["entry_pos"]
        else:
            if len(type_logit):
                entry_pos= Categorical(logits=type_logit).sample()
            else:
                return None

        entry_local_traj = self.token_processor.entry_pos_token[entry_pos]

        feat_type_shape_pos = torch.cat([feat_type_shape,entry_local_traj], dim=-1)

        head_logit = self.entry_head_decoder(feat_type_shape_pos)

        if self.training:
            entry_head_idx = tokenized_agent["entry_head_idx"]
        else:
            entry_head_idx = Categorical(logits=head_logit).sample()
            # entry_head_idx=tokenized_agent["entry_head_idx"][:len(feat_pos)]

        local_head = self.token_processor.decode_head(entry_head_idx)

        entry_local_traj = torch.cat([entry_local_traj, local_head[:, None]], dim=-1)

        feat_type_shape_pos_head = torch.cat([feat_type_shape_pos, local_head[:, None]], dim=-1)

        # if not self.token_processor.use_bird:
        #     # if self.training:
        #     #     entry_local_all=entry_local_traj+tokenized_agent["entry_pos_offset"]
        #
        #     # feat_offset = torch.cat([entry_feature,entry_local_all], dim=-1)
        #
        #     type_logit = self.type_head(feat_pos_head)
        #
        #     if self.training:
        #         entry_type = tokenized_agent["entry_type"]
        #     else:
        #         entry_type = Categorical(logits=type_logit).sample()
        #
        #     feat_pos_head_type = torch.cat([feat_pos_head, entry_type[:, None]], dim=-1)
        #
        #     pred_shape = torch.relu(self.shape_head(feat_pos_head_type))+0.5
        #
        #     if self.training:
        #         entry_shape = tokenized_agent["entry_shape"]
        #     else:
        #         entry_shape = pred_shape
        #
        #     feat_type_shape_pos_head = torch.cat([feat_pos_head_type, entry_shape], dim=-1)

        pred_offset = self.pos_offset_predict_head(feat_type_shape_pos_head)

        entry_local_traj = entry_local_traj + pred_offset

        if self.training:
            entry_logit = (entry_logit, head_logit, pred_offset, type_logit, pred_shape)
        else:
            entry_logit = (entry_mask, entry_local_traj, entry_type, pred_shape)

        return entry_logit

    def forward(self,feat_a,mask_a,pos_a,head_a,tokenized_agent):

        if self.autoregressive_entry:
            entry_logit=self.auto_pred(feat_a,mask_a,pos_a,head_a,tokenized_agent)
        else:
            entry_logit=self.para_pred(feat_a,mask_a,pos_a,head_a,tokenized_agent)


        return entry_logit
