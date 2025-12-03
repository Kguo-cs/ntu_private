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
    ) -> None:

        super(EntryDecoder, self).__init__()
        self.autoregressive_entry= token_processor.autoregressive_entry
        self.n_token_entry=token_processor.n_token_entry

        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        if self.autoregressive_entry:
            self.entry_his_len=1000000

            self.start_embedding =nn.Embedding(1, hidden_dim)

            self.use_one_feature= False

            self.use_cross_attention= True

            if self.use_one_feature or self.use_cross_attention:
                self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=self.entry_his_len)

            self.attr_embedding = nn.Embedding(self.n_token_entry, hidden_dim)

            self.task_embedding = nn.Embedding(5, hidden_dim)

            #self.number_embedding = MLPLayer(1,hidden_dim, hidden_dim)

        self.entry_decoder = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.n_token_entry
        )

    def pred_entry(self,attr_all_feature,entry_pos, entry_head,agent_n,n_current=0,tgt_mask=None):

        n_step=attr_all_feature.shape[1]

        entry_num=n_current+n_step-agent_n

        attr_mask = torch.any(attr_all_feature != 0, dim=-1)

        step = torch.arange(n_step,device=entry_pos.device)+n_current

        task=(step-agent_n)%4

        #number=step-agent_n#(step-agent_n)//4

        task[step<agent_n]=4

        #number_embedding=self.number_embedding(number.float()[:,None])

        attr_all_feature=attr_all_feature+self.task_embedding(task)[None]#+number_embedding[None]

        #attr_all_feature[:,-entry_num:]=attr_all_feature[:,-entry_num:]+number_embedding[None,-entry_num:]

        if self.use_cross_attention:

            entry_feature = attr_all_feature[:, -entry_num:]
            agent_feature = attr_all_feature[:, :-entry_num]
            entry_mask    = attr_mask[:, -entry_num:]

            entry_feature = self.entry_former.cross_attention(entry_feature, entry_pos[:, -entry_num:],
                                                             entry_head[:, -entry_num:], entry_mask,
                                                             agent_feature, entry_pos[:, :-entry_num],
                                                             entry_head[:, :-entry_num],  tgt_mask)

            if n_current!=0:
                n_current=n_current-agent_n

            attr_feature = self.attr_former.temporal_embed(entry_feature, entry_pos[:, -entry_num:], entry_head[:, -entry_num:],
                                                           entry_feature.shape[1], n_current, entry_mask)

        else:

            attr_feature = self.attr_former.temporal_embed(attr_all_feature, entry_pos, entry_head,
                                                            n_step, n_current, attr_mask)

            attr_feature=attr_feature[:,-entry_num:]

        task=task[-entry_num:]

        entry_logit = self.entry_decoder(attr_feature)  # which patch  locate

        mask= (task!=0)

        entry_logit[:,mask,-1]=-torch.inf

        return entry_logit

    def forward(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,n_current=0):


        if self.autoregressive_entry:
            n_agent, n_step=mask_a.shape

            mask_ta = mask_a.transpose(0, 1)

            feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

            feat_a_t[mask_ta] = feat_a.detach()
            batch = tokenized_agent["batch"]
            batch_num = batch.max() + 1
            lengths = torch.bincount(batch,minlength=batch_num).tolist()

            # if self.training:
            #     entry_state = tokenized_agent["entry_state"]
            # else:
            entry_state=torch.zeros([n_step*batch_num, 1, 4], device=feat_a.device)

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
                attr_all_feature = torch.cat([entry_feature, attr_feature], dim=1)
                #entry_pos = current_pos[:, agent_n:, None].repeat(1, 1, 4, 1).flatten(1, 2)[:, :-3]  # 4* entry_agent+1
                #entry_head = current_heading[:, agent_n:, None].repeat(1, 1, 4).flatten(1, 2)[:, :-3]

                entry_pos=torch.zeros([entry_idx.shape[0],entry_idx.shape[1],current_pos.shape[-1]],device=entry_idx.device)
                entry_head=torch.zeros([entry_idx.shape[0],entry_idx.shape[1]],device=entry_idx.device)

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

                    entry_logit = self.pred_entry(entry_feature, current_pos, current_heading, agent_n, n_current,tgt_mask)

                    if n_current==0:
                        self.attr_former.attn.kv_caching(self.entry_his_len,n_current)
                        current_pos = current_pos[:, -1:]
                        current_heading = current_heading[:, -1:]
                        n_current=n_current+agent_n
                        if self.use_cross_attention:
                            self.entry_former.attn.kv_caching(self.entry_his_len, n_current)

                    entry_idx = Categorical(logits=entry_logit).sample()

                    entry_list.append(entry_idx)

                    n_current+=1

                    if len(entry_list)%4==0:
                        entry_idx_x=entry_list[-4]
                        entry_idx_y=entry_list[-3]
                        entry_idx_z=entry_list[-2]
                        entry_idx_head=entry_list[-1]

                        entry_pos_token_x=torch.cat([self.token_processor.entry_pos_token_x,torch.zeros_like(self.token_processor.entry_pos_token_x[:1])])

                        entry_pos_x=entry_pos_token_x[entry_idx_x]
                        entry_pos_y=self.token_processor.entry_pos_token_y[entry_idx_y]
                        entry_pos_z=self.token_processor.entry_pos_token_z[entry_idx_z]
                        entry_pos_head=self.token_processor.entry_head_token[entry_idx_head]

                        finish= finish | (entry_idx_x[:,0]==self.n_token_entry - 1)

                        new_state=torch.stack([entry_pos_x, entry_pos_y, entry_pos_z,entry_pos_head], dim=-1)

                        new_state[finish]=0

                        entry_state_list.append(new_state[:,0])

                        if finish.all() or len(entry_list)==500:
                            entry_logit=torch.stack(entry_state_list,dim=1)
                            break

                    entry_feature = self.attr_embedding(entry_idx)

                self.attr_former.attn.kv_caching(0)
                if self.use_cross_attention:
                    self.entry_former.attn.kv_caching(0)

        else:
            entry_logit = self.entry_decoder(feat_a)
            if self.training:
                entry_idx = tokenized_agent["entry_idx"][:, self.start_step + 1:].transpose(0, 1).flatten(0, 1)[
                    mask_a.transpose(0, 1).flatten(0, 1)]
                entry_mask = (entry_idx < self.token_processor.n_token_entry - 1)
                entry_local = self.token_processor.entry_pos_token[entry_idx[entry_mask]]

                feat_new = torch.cat([entry_local, feat_a[entry_mask]], dim=-1)
                head_logit = self.entry_head_decoder(feat_new)

                entry_logit = (entry_logit, head_logit)

        return entry_logit
