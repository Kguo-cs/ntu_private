import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention

class EntryDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            token_processor,
            start_step
    ) -> None:

        super(EntryDecoder, self).__init__()
        self.autoregressive_entry= token_processor.autoregressive_entry
        self.n_token_entry=token_processor.n_token_entry

        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.start_step=start_step

        if self.autoregressive_entry:
            self.entry_his_len=1000000

            self.start_embedding =nn.Embedding(1, hidden_dim)

            self.use_one_feature= False

            self.use_cross_attention= True

            self.num_levels=3#self.token_processor.tokenizer.num_levels

            if self.use_one_feature or self.use_cross_attention:
                self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)#replace with gnn

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)

            self.pos_embedding = nn.Embedding(self.n_token_entry+1, hidden_dim)

            self.head_embedding  = nn.Embedding(self.token_processor.n_token_entry_head, hidden_dim)

            self.offset_embedding =nn.Embedding(self.token_processor.n_token_offset, hidden_dim)  #MLPLayer(4,hidden_dim,hidden_dim)

            self.task_embedding = nn.Embedding(self.num_levels+1, hidden_dim)

            self.number_embedding = MLPLayer(1,hidden_dim, hidden_dim)

            self.offset_head_decoder  =MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_offset)#offset to offset

            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_entry_head
                    )

        else:
            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_entry_head
                    )

            # self.entry_head_1 = nn.Linear(hidden_dim+3, hidden_dim)

            # self.entry_head_encoder = AttentionLayer(
            #             hidden_dim=hidden_dim,
            #             num_heads=num_heads,
            #             head_dim=hidden_dim//num_heads,
            #             dropout=0,
            #             bipartite=False,
            #             has_pos_emb=True,
            #         )
            #
            # self.entry_head_2 = nn.Linear(hidden_dim+1, hidden_dim)
            #
            # self.entry_head_encoder2 = AttentionLayer(
            #             hidden_dim=hidden_dim,
            #             num_heads=num_heads,
            #             head_dim=hidden_dim//num_heads,
            #             dropout=0,
            #             bipartite=False,
            #             has_pos_emb=True,
            #         )


            self.use_pos_head_offset=True

            if self.use_pos_head_offset:
                self.pos_offset_predict_head =MLPLayer(input_dim=hidden_dim+4, hidden_dim=hidden_dim, output_dim=4)
            else:
                self.pos_offset_predict_head =MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)

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

            entry_feature = self.entry_former.cross_attention(entry_feature, entry_pos,
                                                             entry_head, entry_mask,
                                                             attr_all_feature[:, :-entry_num],
                                                              all_pos[:, :-entry_num],
                                                             all_head[:, :-entry_num],  tgt_mask)

            if n_current!=0:
                n_current=n_current-agent_n

            attr_feature = self.attr_former.temporal_embed(entry_feature, entry_pos, entry_head,
                                                           entry_feature.shape[1], n_current, entry_mask,use_time=True)

        else:

            attr_feature = self.attr_former.temporal_embed(attr_all_feature, all_pos, all_head,
                                                            n_step, n_current, attr_mask)

            attr_feature=attr_feature[:,-entry_num:]

        task=task[-entry_num:]

        pos_mask= (task==0)

        head_mask=(task==2)

        offset_mask=(task==1)

        entry_logit = self.entry_decoder(attr_feature[:,pos_mask])

        entry_head_logit = self.entry_head_decoder(attr_feature[:,head_mask])

        entry_offset = self.offset_head_decoder(attr_feature[:,offset_mask])

        return entry_logit,entry_head_logit,entry_offset
        # last_mask= (task==0 ) &  entry_mask & (number[None,-entry_num: ] !=0)
        #
        # entry_pos_offset = self.pos_offset_predict_head(attr_feature[last_mask])  # which patch  locate   #t,b, n
        #
        # entry_pos_offset = torch.tanh(entry_pos_offset) * self.token_processor.tokenizer.resolution[None]

    def forward(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,edge_index_a2a,n_current=0):
        edge_index_a2a, r_a2a, relative_pos=edge_index_a2a

        if self.autoregressive_entry:
            n_agent, n_step=mask_a.shape

            mask_ta = mask_a.transpose(0, 1)

            feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

            feat_a_t[mask_ta] = feat_a.detach()
            batch = tokenized_agent["batch"]
            batch_num = batch.max() + 1
            lengths = torch.bincount(batch,minlength=batch_num).tolist()

            entry_state=torch.zeros([n_step*batch_num, 1, 4], device=feat_a.device)
            # entry_state[:,:,0]=(self.token_processor.tokenizer.x_min+self.token_processor.tokenizer.x_max)/2
            # entry_state[:,:,1]=(self.token_processor.tokenizer.y_min+self.token_processor.tokenizer.y_max)/2
            # entry_state[:,:,2]=(self.token_processor.tokenizer.z_min+self.token_processor.tokenizer.z_max)/2

            padding_pos = padding(pos_a, lengths, padding_value=0).permute(2, 0, 1, 3).flatten(0, 1) #T,b, n, d
            padding_heading = padding(head_a, lengths, padding_value=0).permute(2, 0, 1).flatten(0, 1)
            padding_features = padding(feat_a_t.transpose(0, 1), lengths, padding_value=0).permute(2, 0, 1, 3).flatten(
                0, 1)

            agent_n = padding_features.shape[1]

            current_pos = torch.cat([padding_pos, entry_state[..., :-1]], dim=1)

            current_heading = torch.cat([padding_heading, entry_state[..., -1]], dim=1)

            entry_embedding = self.start_embedding.weight[None].repeat(len(padding_features),1,1)

            entry_feature = torch.cat([padding_features, entry_embedding], dim=1)

            tgt_mask = torch.any(entry_feature[:, :agent_n] != 0, dim=-1)

            if self.use_one_feature:

                entry_mask = torch.any(entry_feature != 0, dim=-1)

                entry_feature = self.entry_former.temporal_embed(entry_feature, current_pos[:, :agent_n + 1],
                                                                 current_heading[:, :agent_n + 1], 0, n_current,
                                                                 entry_mask)[:,agent_n:]

            if self.training:
                #entry_idx = tokenized_agent["entry_idx"]#.flatten(1, 2)

                #attr_feature = self.attr_embedding(entry_idx)

                pos_idx=tokenized_agent["pos_idx"]
                head_idx = tokenized_agent["head_idx"]
                offset=tokenized_agent["offset"]

                pos_feature=self.pos_embedding(pos_idx)
                heading_feature=self.head_embedding(head_idx)
                offset_feature=self.offset_embedding(offset)#[:,:,None]

                attr_feature=torch.stack([pos_feature, offset_feature,heading_feature], dim=2)

                #attr_feature[:,pos_idx==self.token_processor.n_token_entry]=0

                attr_feature=attr_feature.flatten(1,2)

                attr_all_feature = torch.cat([entry_feature, attr_feature], dim=1)


                entry_pos=torch.zeros([pos_idx.shape[0],attr_feature.shape[1],current_pos.shape[-1]],device=pos_idx.device)
                entry_head=torch.zeros([pos_idx.shape[0],attr_feature.shape[1]],device=pos_idx.device)
                # entry_idx_all =entry_idx.reshape(entry_idx.shape[0],-1,self.num_levels)
                # entry_pos=[]
                # entry_head=[]
                #
                # for l in range(1,self.num_levels+1):
                #     pos_rec, heading_rec = self.token_processor.tokenizer.decode_tokens_to_state(entry_idx_all[:,:,:l])
                #
                #     entry_pos.append(pos_rec)
                #     entry_head.append(heading_rec)
                #
                # entry_pos=torch.stack(entry_pos,dim=2).flatten(1,2)
                # entry_head=torch.stack(entry_head,dim=2).flatten(1,2)

                if self.use_one_feature:
                    agent_n=0
                else:
                    entry_pos=torch.cat([current_pos, entry_pos], dim=1)
                    entry_head=torch.cat([current_heading, entry_head], dim=1)

                entry_logit = self.pred_entry(attr_all_feature, entry_pos, entry_head,agent_n,tgt_mask=tgt_mask)


            else:
                self.attr_former.attn.caching = True

                if self.use_cross_attention:
                    self.entry_former.attn.caching = True

                entry_list=[]
                entry_state_list = []

                finish=torch.zeros_like(current_heading[:,0]).to(torch.bool)

                if self.use_one_feature:
                    agent_n=0
                    current_pos=current_pos[:, -1:]
                    current_heading=current_heading[:, -1:]

                while True:

                    entry_logit,entry_head_logit,entry_offset = self.pred_entry(entry_feature, current_pos, current_heading, agent_n, n_current,tgt_mask)

                    if n_current==0:
                        self.attr_former.attn.kv_caching(self.entry_his_len,n_current)
                        if self.use_cross_attention:
                            self.entry_former.attn.kv_caching(self.entry_his_len, n_current)
                        current_pos = current_pos[:, -1:]
                        current_heading = current_heading[:, -1:]
                        n_current=n_current+agent_n

                    if entry_logit.shape[1]!=0:
                        pos_idx= Categorical(logits=entry_logit).sample()

                        finish = finish | (pos_idx[:, 0] == self.n_token_entry)

                        if finish.all() or len(entry_state_list)>140:
                            entry_logit=torch.stack(entry_state_list,dim=1)
                            break

                        pos_idx[pos_idx== self.n_token_entry]=0

                        token_pos=self.token_processor.entry_pos_token[pos_idx]

                        entry_feature = self.pos_embedding(pos_idx)

                    elif entry_head_logit.shape[1]!=0:
                        entry_head_idx = Categorical(logits=entry_head_logit).sample()

                        tokenized_heading = self.token_processor.decode_head(entry_head_idx)

                        entry_feature = self.head_embedding(entry_head_idx)

                        heading_rec=tokenized_heading#+entry_offset[:,0,3:]

                        new_state=torch.cat([pos_rec, heading_rec], dim=-1)

                        new_state[finish] = 0

                        entry_state_list.append(new_state)

                    else:
                        offset_idx = Categorical(logits=entry_offset).sample()

                        entry_offset=self.token_processor.offset_token[offset_idx]


                        pos_rec=token_pos[:,0]+entry_offset[:,0,:3]


                        entry_feature = self.offset_embedding(offset_idx)


                    # entry_idx = Categorical(logits=entry_logit).sample()
                    #
                    # entry_list.append(entry_idx)
                    #
                    n_current+=1
                    #
                    # num = len(entry_list) % self.num_levels
                    #
                    # if num==1:
                    #     finish= finish | (entry_idx[:,0]==self.n_token_entry )
                    #
                    #     if len(entry_list) >1:
                    #         entry_idx_all = torch.cat(entry_list, dim=1)[:, -self.num_levels-1:-1]
                    #
                    #         pos_rec, heading_rec = self.token_processor.tokenizer.decode_tokens_to_state(entry_idx_all)
                    #
                    #         pos_rec=pos_rec+entry_offset
                    #
                    #         new_state=torch.cat([pos_rec, heading_rec[:,None]], dim=-1)
                    #
                    #         new_state[finish]=0
                    #
                    #         entry_state_list.append(new_state)
                    #
                    #     if finish.all() or len(entry_list)>140*self.num_levels:
                    #         entry_logit=torch.stack(entry_state_list,dim=1)
                    #         break
                    #
                    # if num==0:
                    #     num=self.num_levels
                    #
                    # entry_token = torch.cat(entry_list, dim=1)[:, -num:]
                    #
                    # current_pos1,current_heading1=self.token_processor.tokenizer.decode_tokens_to_state(entry_token)
                    # current_pos=current_pos1[:,None]
                    # current_heading=current_heading1[:,None]
                    #
                    # entry_feature = self.attr_embedding(entry_idx)

                self.attr_former.attn.kv_caching(0)
                if self.use_cross_attention:
                    self.entry_former.attn.kv_caching(0)

        else:
            #feat_a=feat_a.detach()

            entry_logit = self.entry_decoder(feat_a)
            if self.training:
                entry_idx = tokenized_agent["entry_idx"][:, self.start_step + 1:].transpose(0, 1).flatten(0, 1)[ mask_a.transpose(0, 1).flatten(0, 1)]
            else:
                entry_idx = Categorical(logits=entry_logit).sample()
                # entry_idx = tokenized_agent["entry_idx"][:,1]
                # mask=tokenized_agent["valid_mask"][:,0]
                # entry_idx=entry_idx[mask]

            entry_mask = (entry_idx < self.token_processor.n_token_entry)
            entry_local_traj = self.token_processor.entry_pos_token[entry_idx[entry_mask]]
            entry_feature=feat_a[entry_mask]

            feat_pos = torch.cat([entry_local_traj, entry_feature], dim=-1)

            if self.use_pos_head_offset:
                # Build the mapping from old index -> new compact index
                # old_idx = torch.arange(feat_a.size(0), device=feat_a.device)
                # pos_idx = old_idx[entry_mask]  # e.g., [3, 7, 10, ...]    (indices inside feat_a)
                #
                # # Create the mapping array
                # mapping = pos_idx.new_full((feat_a.size(0),), -1)  # size N_a, fill -1
                # mapping[pos_idx] = torch.arange(pos_idx.size(0), device=feat_a.device)
                #
                # # Now filter edges whose both endpoints survive entry_mask
                # src, dst = edge_index_a2a  # both are indices w.r.t feat_a
                #
                # valid_edges = (mapping[src] != -1) & (mapping[dst] != -1)
                #
                # src_new = mapping[src[valid_edges]]
                # dst_new = mapping[dst[valid_edges]]
                #
                # edge_index_a2a_pos = torch.stack([src_new, dst_new], dim=0)
                #
                # edge_mask = entry_mask[edge_index_a2a[0]] & entry_mask[edge_index_a2a[1]]
                #
                # r_a2a_pos=r_a2a[edge_mask]
                #
                # feat_pos=self.entry_head_1(feat_pos)
                #
                # feat_pos=self.entry_head_encoder(feat_pos, r_a2a_pos, edge_index_a2a_pos)

                head_logit = self.entry_head_decoder(feat_pos)

                if self.training:
                    entry_head_idx = tokenized_agent["entry_head_idx"]
                else:
                    entry_head_idx = Categorical(logits=head_logit).sample()
                    # entry_head_idx=tokenized_agent["entry_head_idx"][:len(feat_pos)]

                local_head = self.token_processor.decode_head(entry_head_idx)

                entry_local_traj = torch.cat([entry_local_traj, local_head[:, None]], dim=-1)

                feat_token = torch.cat([feat_pos, local_head[:, None]], dim=-1)

                # feat_token=self.entry_head_2(feat_token)
                #
                # feat_token=self.entry_head_encoder2(feat_token, r_a2a_pos, edge_index_a2a_pos)


                pred_offset = self.pos_offset_predict_head(feat_token)

                # pred_offset=tokenized_agent["entry_pos_offset"][:len(feat_pos)]

                entry_local_traj = entry_local_traj + pred_offset

            else:
                pred_offset = self.pos_offset_predict_head(feat_pos)

                if  self.training:
                    real_offset = tokenized_agent["entry_pos_offset"][:,:3]
                else:
                    real_offset=pred_offset

                entry_local_traj = entry_local_traj + real_offset

                feat_pos_offset = torch.cat([entry_local_traj, entry_feature], dim=-1)

                head_logit = self.entry_head_decoder(feat_pos_offset)

                if not self.training:
                    entry_head_idx = Categorical(logits=head_logit).sample()

                    local_head = self.token_processor.decode_head(entry_head_idx)

                    entry_local_traj = torch.cat([entry_local_traj, local_head[:, None]], dim=-1)

            if self.training:
                entry_logit = (entry_logit, head_logit,pred_offset)
            else:
                entry_logit=(entry_mask,entry_local_traj)

        return entry_logit
