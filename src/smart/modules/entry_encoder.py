import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical

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

            self.num_levels=self.token_processor.tokenizer.num_levels

            if self.use_one_feature or self.use_cross_attention:
                self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)

            self.attr_embedding = nn.Embedding(self.n_token_entry+1, hidden_dim)

            self.task_embedding = nn.Embedding(self.num_levels+1, hidden_dim)

            self.number_embedding = MLPLayer(1,hidden_dim, hidden_dim)
            self.pos_offset_predict_head =MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)

        else:
            self.entry_head_decoder = MLPLayer(
                        input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=self.token_processor.n_token_entry_head
                    )


            self.use_pos_head_offset=True

            if self.use_pos_head_offset:
                self.pos_offset_predict_head =MLPLayer(input_dim=hidden_dim+4, hidden_dim=hidden_dim, output_dim=4)
            else:
                self.pos_offset_predict_head =MLPLayer(input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=3)

        self.entry_decoder = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.n_token_entry+1
        )

    def pred_entry(self,attr_all_feature,entry_pos, entry_head,agent_n,n_current=0,tgt_mask=None):

        n_step=attr_all_feature.shape[1]

        entry_num=n_current+n_step-agent_n

        attr_mask = torch.any(attr_all_feature != 0, dim=-1)

        step = torch.arange(n_step,device=entry_pos.device)+n_current

        task=(step-agent_n)%self.num_levels

        number=(step-agent_n)//self.num_levels #step-agent_n#

        task[step<agent_n]=self.num_levels

        number_embedding=self.number_embedding(number.float()[:,None])

        attr_all_feature=attr_all_feature+self.task_embedding(task)[None]#+number_embedding[None]

        attr_all_feature[:,-entry_num:]=attr_all_feature[:,-entry_num:]+number_embedding[None,-entry_num:]

        entry_mask = attr_mask[:, -entry_num:]

        if self.use_cross_attention:

            entry_feature = attr_all_feature[:, -entry_num:]
            agent_feature = attr_all_feature[:, :-entry_num]

            entry_feature = self.entry_former.cross_attention(entry_feature, entry_pos[:, -entry_num:],
                                                             entry_head[:, -entry_num:], entry_mask,
                                                             agent_feature, entry_pos[:, :-entry_num],
                                                             entry_head[:, :-entry_num],  tgt_mask)

            if n_current!=0:
                n_current=n_current-agent_n

            attr_feature = self.attr_former.temporal_embed(entry_feature, entry_pos[:, -entry_num:], entry_head[:, -entry_num:],
                                                           entry_feature.shape[1], n_current, entry_mask,use_time=False)

        else:

            attr_feature = self.attr_former.temporal_embed(attr_all_feature, entry_pos, entry_head,
                                                            n_step, n_current, attr_mask)

            attr_feature=attr_feature[:,-entry_num:]

        task=task[-entry_num:]

        entry_logit = self.entry_decoder(attr_feature)  # which patch  locate

        mask= (task!=0)

        entry_logit[:,mask,-1]=-torch.inf

        last_mask= (task==0 ) &  entry_mask & (number[None,-entry_num: ] !=0)

        entry_pos_offset = self.pos_offset_predict_head(attr_feature[last_mask])  # which patch  locate   #t,b, n

        entry_pos_offset = torch.tanh(entry_pos_offset) * self.token_processor.tokenizer.resolution[None]

        return entry_logit,entry_pos_offset

    def forward(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,n_current=0):


        if self.autoregressive_entry:
            n_agent, n_step=mask_a.shape

            mask_ta = mask_a.transpose(0, 1)

            feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

            feat_a_t[mask_ta] = feat_a
            batch = tokenized_agent["batch"]
            batch_num = batch.max() + 1
            lengths = torch.bincount(batch,minlength=batch_num).tolist()

            entry_state=torch.zeros([n_step*batch_num, 1, 4], device=feat_a.device)
            entry_state[:,:,0]=(self.token_processor.tokenizer.x_min+self.token_processor.tokenizer.x_max)/2
            entry_state[:,:,1]=(self.token_processor.tokenizer.y_min+self.token_processor.tokenizer.y_max)/2
            entry_state[:,:,2]=(self.token_processor.tokenizer.z_min+self.token_processor.tokenizer.z_max)/2

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
                entry_idx = tokenized_agent["entry_idx"].flatten(1, 2)

                attr_feature = self.attr_embedding(entry_idx)
                attr_feature[entry_idx==self.token_processor.n_token_entry]=0

                attr_all_feature = torch.cat([entry_feature, attr_feature], dim=1)


                entry_pos=torch.zeros([entry_idx.shape[0],entry_idx.shape[1],current_pos.shape[-1]],device=entry_idx.device)
                entry_head=torch.zeros([entry_idx.shape[0],entry_idx.shape[1]],device=entry_idx.device)
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

                    entry_logit,entry_offset = self.pred_entry(entry_feature, current_pos, current_heading, agent_n, n_current,tgt_mask)

                    if n_current==0:
                        self.attr_former.attn.kv_caching(self.entry_his_len,n_current)
                        if self.use_cross_attention:
                            self.entry_former.attn.kv_caching(self.entry_his_len, n_current)
                       # current_pos = current_pos[:, -1:]
                        #current_heading = current_heading[:, -1:]
                        n_current=n_current+agent_n

                    entry_idx = Categorical(logits=entry_logit).sample()

                    entry_list.append(entry_idx)

                    n_current+=1

                    num = len(entry_list) % self.num_levels

                    if num==1:
                        finish= finish | (entry_idx[:,0]==self.n_token_entry )

                        if len(entry_list) >1:
                            entry_idx_all = torch.cat(entry_list, dim=1)[:, -self.num_levels-1:-1]

                            pos_rec, heading_rec = self.token_processor.tokenizer.decode_tokens_to_state(entry_idx_all)

                            pos_rec=pos_rec+entry_offset

                            new_state=torch.cat([pos_rec, heading_rec[:,None]], dim=-1)

                            new_state[finish]=0

                            entry_state_list.append(new_state)

                        if finish.all() or len(entry_list)>140*self.num_levels:
                            entry_logit=torch.stack(entry_state_list,dim=1)
                            break

                    if num==0:
                        num=self.num_levels

                    entry_token = torch.cat(entry_list, dim=1)[:, -num:]

                    current_pos1,current_heading1=self.token_processor.tokenizer.decode_tokens_to_state(entry_token)
                    current_pos=current_pos1[:,None]
                    current_heading=current_heading1[:,None]

                    entry_feature = self.attr_embedding(entry_idx)

                self.attr_former.attn.kv_caching(0)
                if self.use_cross_attention:
                    self.entry_former.attn.kv_caching(0)

        else:
            entry_logit = self.entry_decoder(feat_a)
            if self.training:
                entry_idx = tokenized_agent["entry_idx"][:, self.start_step + 1:].transpose(0, 1).flatten(0, 1)[ mask_a.transpose(0, 1).flatten(0, 1)]
                entry_mask = (entry_idx < self.token_processor.n_token_entry)
                pos_local = self.token_processor.entry_pos_token[entry_idx[entry_mask]]
                entry_feature=feat_a[entry_mask]

                if self.use_pos_head_offset:
                    feat_head = torch.cat([pos_local, entry_feature], dim=-1)

                    entry_head_idx=tokenized_agent["entry_head_idx"]#[:,self.start_step + 1:].transpose(0, 1).flatten(0, 1)[head_mask]#t,a

                    token_head_local = (entry_head_idx - self.token_processor.n_token_entry_head_half) / (
                        self.token_processor.n_token_entry_head_half) * torch.pi

                    feat_offset = torch.cat([pos_local,token_head_local[:,None], entry_feature], dim=-1)
                else:
                    feat_offset = torch.cat([pos_local, entry_feature], dim=-1)

                    entry_pos_offset=tokenized_agent["entry_pos_offset"]

                    feat_head = torch.cat([pos_local+entry_pos_offset[...,:3], entry_feature], dim=-1)

                pred_offset=self.pos_offset_predict_head(feat_offset)

                head_logit = self.entry_head_decoder(feat_head)               #heading should also be local

                entry_logit = (entry_logit, head_logit,pred_offset)
            else:

                entry_token_idx = Categorical(logits=entry_logit).sample()

                entry_mask = entry_token_idx < self.token_processor.n_token_entry

                entry_local_traj = self.token_processor.entry_pos_token[entry_token_idx[entry_mask]]

                entry_feature = feat_a[entry_mask]

                feat_pos = torch.cat([entry_local_traj, entry_feature], dim=-1)

                if self.use_pos_head_offset:
                    head_logit = self.entry_head_decoder(feat_pos)

                    entry_head_idx = Categorical(logits=head_logit).sample()

                    local_head = (entry_head_idx - self.token_processor.n_token_entry_head_half) / (
                        self.token_processor.n_token_entry_head_half) * torch.pi

                    entry_local_traj=torch.cat([entry_local_traj, local_head[:, None]], dim=-1)

                    feat_token = torch.cat([entry_local_traj, entry_feature], dim=-1)

                    pred_offset = self.pos_offset_predict_head(feat_token)

                    entry_local_traj=entry_local_traj+pred_offset

                else:

                    pred_offset = self.pos_offset_predict_head(feat_pos)

                    entry_local_traj = entry_local_traj + pred_offset

                    feat_pos_offset = torch.cat([entry_local_traj, entry_feature], dim=-1)

                    head_logit = self.entry_head_decoder(feat_pos_offset)

                    entry_head_idx = Categorical(logits=head_logit).sample()

                    local_head = (entry_head_idx - self.token_processor.n_token_entry_head_half) / (
                        self.token_processor.n_token_entry_head_half) * torch.pi

                    entry_local_traj = torch.cat([entry_local_traj, local_head[:, None]], dim=-1)

                entry_logit=(entry_mask,entry_local_traj)

        return entry_logit
