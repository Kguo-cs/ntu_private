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

        if self.autoregressive_entry:
            self.entry_embedding = MLPLayer(4, hidden_dim, hidden_dim)

            self.entry_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=1000000)

            self.attr_former = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0, hist_len=1000000)

            self.attr_embedding = nn.Embedding(self.n_token_entry, hidden_dim)

        self.entry_decoder = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.n_token_entry
        )

    def autoregressive_entry(self,entry_idx,entry_feature,pos,heading,n_current=0):

        attr_feature = self.attr_embedding(entry_idx)
        # attr_all_feature=torch.cat([entry_feature[:,:,None],attr_feature],dim=2).flatten(1,2)    #x,y,z,heading
        attr_all_feature = torch.cat([entry_feature, attr_feature.flatten(1, 2)], dim=1)

        attr_mask = torch.any(attr_all_feature != 0, dim=-1)

        entry_pos = pos[:, :, None].repeat(1, 1, 4, 1).flatten(1, 2)[:, :-3]  # 4* entry_agent+1
        entry_head = heading[:,:, None].repeat(1, 1, 4).flatten(1, 2)[:, :-3]

        attr_feature = self.entry_former.temporal_embed(attr_all_feature, entry_pos, entry_head,
                                                        attr_all_feature.shape[1], n_current, attr_mask)

        entry_logit = self.entry_decoder(attr_feature)  # which patch  locate

        return entry_logit

    def forward(self,feat_a,mask_a,pos_a,head_a,tokenized_agent,n_current=0):
        if self.autoregressive_entry:
            n_agent, n_step=mask_a.shape

            mask_ta = mask_a.transpose(0, 1)

            feat_a_t = torch.zeros([n_step, n_agent, self.hidden_dim], device=feat_a.device)

            feat_a_t[mask_ta] = feat_a

            entry_state = tokenized_agent["entry_state"]
            entry_idx = tokenized_agent["entry_idx"]
            batch = tokenized_agent["batch"]

            lengths = torch.bincount(batch).tolist()

            padding_pos = padding(pos_a, lengths, padding_value=0).permute(2, 0, 1, 3).flatten(0, 1)
            padding_heading = padding(head_a, lengths, padding_value=0).permute(2, 0, 1).flatten(0, 1)
            padding_features = padding(feat_a_t.transpose(0, 1), lengths, padding_value=0).permute(2, 0, 1, 3).flatten(
                0, 1)

            agent_n = padding_features.shape[1]

            pos = torch.cat([padding_pos, entry_state[..., :-1]], dim=1)

            heading = torch.cat([padding_heading, entry_state[..., -1]], dim=1)

            entry_embedding = self.entry_embedding(entry_state[:, :1])

            all_features = torch.cat([padding_features, entry_embedding], dim=1)

            entry_mask = torch.any(all_features != 0, dim=-1)

            entry_feature = self.entry_former.temporal_embed(all_features, pos[:, :agent_n + 1],
                                                             heading[:, :agent_n + 1], all_features.shape[1], n_current,
                                                             entry_mask)[:, agent_n:]

            if self.training:
                entry_logit=self.autoregressive_entry(entry_idx,entry_feature,pos[:,agent_n:],heading[:,agent_n:])
            else:
                self.attr_former.attn.caching = True
                entry_logit=self.autoregressive_entry(entry_idx[:1],entry_feature[:1],pos[:,agent_n:agent_n+1],heading[:,agent_n:agent_n+1])

                entry_idx= Categorical(logits=entry_logit).sample()

                while (entry_idx!=self.n_token_entry-1).any():

                    entry_logit = self.autoregressive_entry(entry_idx[:1], entry_feature[:1],
                                                            pos[:, agent_n:agent_n + 1],
                                                            heading[:, agent_n:agent_n + 1])

            self.attr_former.attn.kv_caching(0)




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
