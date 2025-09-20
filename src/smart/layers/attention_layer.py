from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from src.smart.utils import weight_init

from torch_scatter import scatter_sum

class AttentionLayer(MessagePassing):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        **kwargs
    ) -> None:
        super(AttentionLayer, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5
        self.hidden_dim = hidden_dim

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        self.attention_weight = attn.mean(-1) #.detach()#
        # plogp = attn * (attn.clamp_min(1e-12).log())
        # # Sum within each destination segment
        # seg_entropy = scatter_sum(plogp, index,dim=0)  # shape: [num_dst_nodes]

        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)

from torch_scatter import scatter_add
import torch

import torch

@torch.no_grad()
def feat_list_mask_each_agent_cached(
    layer,                 # patched AttentionLayer (must be in eval mode)
    feat_a_pt,             # [n_step * n_agents_total, hidden_dim]  (same x used in forward)
    r_a2a, edge_index_a2a, # used only for the single cache-filling forward
    train_mask,            # [n_agents_total] (bool; which agents are "valid" to keep)
    batch_s_repeat,        # [n_agents_total, ...], batch id in [:,0]
    n_step: int,
    mask_agents: torch.Tensor = None,  # which agents to mask; default = all with train_mask==True
):
    """
    Returns:
      feat_list: length M; each item is [n_step, kept_in_that_batch, hidden_dim],
                 where row k corresponds to masking agent `mask_agents[k]` ONLY.
    """

    device = feat_a_pt.device
    hidden_dim = feat_a_pt.size(-1)

    # 0) Fill caches with a single forward (dropout OFF for exact renorm)
    layer.eval()
    _ = layer(feat_a_pt, r_a2a, edge_index_a2a)

    # 1) Choose which agents to mask (default: every agent in train_mask)
    train_mask = train_mask.to(torch.bool)
    if mask_agents is None:
        mask_agents = torch.where(train_mask)[0]          # [M]
    else:
        mask_agents = mask_agents.to(torch.long)
    M = mask_agents.numel()
    if M == 0:
        return []

    # 2) Pull caches (PRE-dropout softmax for exact renorm)
    C = layer._cache
    attn = C["attn_soft"]    # [E, H]
    V    = C["v_j"]          # [E, H, Dh]
    dst  = C["dst"]          # [E]
    src  = C["src"]          # [E]
    x_dst = C["x_dst"]       # [N, hidden]

    N, H, Dh = x_dst.size(0), V.size(1), V.size(2)
    A_total  = batch_s_repeat.size(0)
    T = n_step
    assert N == T * A_total, "Expect time-major nodes: node_id = t*A_total + agent_id"

    # 3) Base aggregate y_pre = sum_j a_ij V_j using PRE-dropout attention
    msg = (attn.unsqueeze(-1) * V).reshape(-1, H * Dh)   # [E, H*Dh]
    y_pre = torch.zeros((N, H * Dh), device=device)
    y_pre.index_add_(0, dst, msg)
    y_pre = y_pre.view(N, H, Dh)                         # [N, H, Dh]

    # 4) Vectorized per-agent delta on y_pre with exact softmax renorm (each row = one masked agent)
    # Map every source node to the row that masks its agent id
    map_agent_to_row = torch.full((A_total,), -1, device=device, dtype=torch.long)
    map_agent_to_row[mask_agents] = torch.arange(M, device=device)
    map_src_to_row = map_agent_to_row.repeat(T)          # [N], time-major expansion

    m_row = map_src_to_row[src]                          # [E]
    keep_edges = m_row >= 0                              # edges whose src is "the" masked agent of some row
    if keep_edges.sum() == 0:
        # Nothing changes if none of the masked agents appear as src; just slice per-row below
        masked_outputs = _finish_post_attn_path(layer, feat_a_pt, y_pre, x_dst)
    else:
        attn_e = attn[keep_edges]                        # [E', H]
        V_e    = V[keep_edges]                           # [E', H, Dh]
        dst_e  = dst[keep_edges]                         # [E']
        m_row  = m_row[keep_edges]                       # [E']
        y_dst_pre = y_pre[dst_e]                         # [E', H, Dh]

        a   = attn_e.unsqueeze(-1)                       # [E', H, 1]
        inv = (1.0 - attn_e).clamp_min(1e-8).unsqueeze(-1)
        delta_e = (a/inv) * (y_dst_pre - a * V_e) - a * V_e  # [E', H, Dh]

        # accumulate per (row, dst)
        MN = M * N
        lin = m_row * N + dst_e
        delta_flat = torch.zeros((MN, H * Dh), device=device)
        delta_flat.index_add_(0, lin, delta_e.view(-1, H * Dh))
        delta_all_pre = delta_flat.view(M, N, H, Dh)     # [M, N, H, Dh]
        y_all_pre = y_pre.unsqueeze(0) + delta_all_pre   # [M, N, H, Dh]
        masked_outputs = _finish_post_attn_path(layer, feat_a_pt, y_all_pre, x_dst, batched=True)  # [M, N, hidden]

    # 5) Turn each row into your per-batch slice: [T, kept_in_batch, hidden]
    agent_batch_all = batch_s_repeat[:, 0].to(torch.long)           # [A_total]
    batch_of_row    = agent_batch_all[mask_agents]                  # [M]
    same_batch = (agent_batch_all.unsqueeze(0) == batch_of_row.unsqueeze(1))  # [M, A_total]
    keep_agent_mask = same_batch & train_mask.unsqueeze(0) & \
                      (torch.arange(A_total, device=device).unsqueeze(0) != mask_agents.unsqueeze(1))

    # gather per-row kept agent indices (ragged → turn into list)
    rows, cols = torch.nonzero(keep_agent_mask, as_tuple=True)
    counts = torch.bincount(rows, minlength=M)
    # positions within row groups:
    order = torch.argsort(rows)
    rows_s, cols_s = rows[order], cols[order]
    start = torch.zeros(M, dtype=torch.long, device=device)
    if M > 1:
        start[1:] = counts.cumsum(0)[:-1]
    pos = torch.arange(rows_s.numel(), device=device) - start[rows_s]

    max_kept = int(counts.max().item()) if M > 0 else 0
    keep_idx = torch.zeros((M, max_kept), dtype=torch.long, device=device)
    if rows_s.numel() > 0:
        keep_idx[rows_s, pos] = cols_s

    mo = masked_outputs.view(M, T, A_total, hidden_dim)
    feats_padded = mo.gather(2, keep_idx[:, None, :, None].expand(M, T, max_kept, hidden_dim))
    valid_cols_mask = (torch.arange(max_kept, device=device)[None, :] < counts[:, None])
    feats_padded = feats_padded * valid_cols_mask[:, None, :, None]

    # finally, produce the python list with ragged shapes (one entry per masked agent)
    feat_list = [
        feats_padded[i, :, :counts[i].item(), :].contiguous()   # [T, kept_i, H]
        for i in range(M)
    ]
    return feat_list


def _finish_post_attn_path(layer, feat_a_pt, y_pre_or_batched, x_dst, batched=False):
    """
    Applies the SAME gating/update + to_out + residual + FF path as the layer.
    y_pre_or_batched: [N, H, Dh] if batched=False, else [M, N, H, Dh]
    Returns: [N, hidden] or [M, N, hidden]
    """
    if not batched:
        y = y_pre_or_batched.view(-1, layer.num_heads * layer.head_dim)  # [N, H*Dh]
        # gating/update
        g = torch.sigmoid(layer.to_g(torch.cat([y, x_dst], dim=-1)))
        post = y + g * (layer.to_s(x_dst) - y)                           # [N, H*Dh]
        # to_out + residual + FF
        attn_out = layer.to_out(post)
        x_after  = feat_a_pt + layer.attn_postnorm(attn_out)
        ff_in    = layer.ff_prenorm(x_after)
        ff_out   = layer._ff_block(ff_in)
        return x_after + layer.ff_postnorm(ff_out)                       # [N, hidden]
    else:
        M, N, H, Dh = y_pre_or_batched.shape
        y = y_pre_or_batched.view(M, N, H * Dh)                          # [M, N, H*Dh]
        x_dst_exp = x_dst.unsqueeze(0).expand(M, N, x_dst.size(-1))      # [M, N, hidden]
        g = torch.sigmoid(layer.to_g(torch.cat([y, x_dst_exp], dim=-1))) # [M, N, H*Dh]
        post = y + g * (layer.to_s(x_dst).unsqueeze(0).expand_as(y) - y) # [M, N, H*Dh]
        attn_out = layer.to_out(post)                                    # [M, N, hidden]
        x_after  = feat_a_pt.unsqueeze(0) + layer.attn_postnorm(attn_out)
        ff_in    = layer.ff_prenorm(x_after)
        ff_out   = layer._ff_block(ff_in)
        return x_after + layer.ff_postnorm(ff_out)                       # [M, N, hidden]


class CacheAttention(AttentionLayer):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        **kwargs
    ) -> None:
        super().__init__(hidden_dim, num_heads, head_dim, dropout, bipartite, has_pos_emb, **kwargs)

    def forward(self, x, r, edge_index):
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)

        heads, cache = self._attn_block_with_cache(x_src, x_dst, r, edge_index)
        self._cache = cache  # stash for ablation

        attn_out = self.to_out(heads.view(heads.size(0), -1))     # [N, hidden]
        x = x + self.attn_postnorm(attn_out)
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x, self.attention_weight

    def message(self, q_i, k_j, v_j, r, index, ptr):
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)

        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn_soft = softmax(sim, index, ptr)   # PRE-DROPOUT (sum to 1 over dst segments)
        self.attention_weight = attn_soft.mean(-1)

        attn_used = self.attn_drop(attn_soft)  # DROPOUT APPLIED HERE
        # cache pre-dropout for exact renorm, and values/dst ids
        self._last_attn_soft = attn_soft          # [E, H]
        self._last_v_j       = v_j                # [E, H, Dh]
        self._last_dst_index = index              # [E]

        return v_j * attn_used.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        # `inputs` here is the aggregated pre-projection heads AFTER dropout.
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block_with_cache(self, x_src, x_dst, r, edge_index):
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)

        # cache prenorm destination features used by gating
        self._last_x_dst = x_dst  # [N, hidden]

        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        heads = agg.view(-1, self.num_heads, self.head_dim)  # post-update heads

        cache = {
            "attn_soft": self._last_attn_soft,  # [E, H], PRE-DROPOUT softmax
            "v_j": self._last_v_j,              # [E, H, Dh]
            "dst": self._last_dst_index,        # [E]
            "src": edge_index[0],               # [E]
            "x_dst": self._last_x_dst,          # [N, hidden] (prenorm dst)
        }
        return heads, cache

    def refer(self,
            feat_a_pt,             # [n_step * n_agents_total, hidden_dim]  (input to this layer)
            r_a2a,                 # [E, rdim] or None
            edge_index_a2a,        # [2, E]
            train_mask,            # [n_agents_total] (True=keep)
            batch_s_repeat,        # [n_agents_total, ...], batch id in [:,0]
            n_step: int,
              ):
        device = feat_a_pt.device
        # agents across ALL batches (ragged per-batch counts allowed)
        n_agents_total = batch_s_repeat.size(0)

        # valid agents (candidates to ablate)
        valid_agent = torch.where(train_mask)[0]  # [A]
        A = valid_agent.numel()

        # batch id per agent (global agent indexing 0..n_agents_total-1)
        agent_batch_all = batch_s_repeat[:, 0].to(torch.long)  # [n_agents_total]
        n_batch = int(agent_batch_all.max().item()) + 1

        # valid agents per batch (kept set baseline)
        valid_list = [torch.where((agent_batch_all == b) & train_mask)[0]
                      for b in range(n_batch)]  # list of 1D tensors (ragged)

        # counts of valid agents per batch and their max to drive slot loop
        vb = agent_batch_all[valid_agent]
        counts = torch.bincount(vb, minlength=n_batch)  # [B]
        max_agent_num = int(counts.max().item())

        # map global agent id -> row index in result list (aligned to valid_agent order)
        row_of_agent = torch.full((n_agents_total,), -1, device=device, dtype=torch.long)
        row_of_agent[valid_agent] = torch.arange(A, device=device)

        # compute each valid agent's local index within its batch (0..count[b]-1), in valid_agent order
        start = torch.zeros(n_batch, dtype=torch.long, device=device)
        if n_batch > 1:
            start[1:] = counts.cumsum(0)[:-1]
        local_idx = torch.arange(vb.numel(), device=device) - start[vb]

        # pre-create result list (ragged)
        feat_list = [None] * A

        # helper: expand per-agent mask to per-node mask (time-major layout)
        def agentmask_to_nodemask(agent_mask_bool):
            # flatten order matches your code: repeat over time, transpose, flatten
            return agent_mask_bool[:, None].repeat(1, n_step).transpose(0, 1).reshape(-1)

        # constants for fast time-offset indexing (time-major: t block size = n_agents_total)
        time_offsets = (torch.arange(n_step, device=device) * n_agents_total).view(n_step, 1)  # [T,1]

        # --- slot-parallel ablation: one forward per "position in batch" ---
        for slot in range(max_agent_num):
            # agents to mask at this slot (one per batch that has >= slot+1 valid agents)
            mask_indices = valid_agent[local_idx == slot]  # [<= B]
            if mask_indices.numel() == 0:
                continue

            # build per-agent mask over ALL agents; flip the chosen ones to False
            one_hot_mask = train_mask.clone()
            one_hot_mask[mask_indices] = False

            # prune edges that touch any masked agent across time
            mask_nodes = agentmask_to_nodemask(one_hot_mask)  # [n_step*n_agents_total]
            keep_edges = mask_nodes[edge_index_a2a[0]] & mask_nodes[edge_index_a2a[1]]
            eidx2 = edge_index_a2a[:, keep_edges]
            r2 = r_a2a[keep_edges]

            # one forward for all batches' slot ablation
            feat_masked, _ = self.forward(feat_a_pt, r2, eidx2)  # [n_step*n_agents_total, hidden_dim]

            # For each masked agent a_id, gather its batch slice without reshaping to [B, T, ..]
            for a_id in mask_indices.tolist():
                b = int(agent_batch_all[a_id].item())

                # valid agents in this batch (baseline kept set), then drop the masked agent
                agents_b = valid_list[b]  # 1D tensor of global agent ids
                keep_agents_b = agents_b[agents_b != a_id]  # remove masked agent

                # Gather indices for all (t, agent) pairs in this batch (time-major)
                if keep_agents_b.numel() == 0:
                    # no remaining agents in this batch; create empty (T, 0, H)
                    new_feat = feat_masked.new_empty((n_step, 0, self.hidden_dim))
                else:
                    idx_nodes = (time_offsets + keep_agents_b.view(1, -1)).reshape(-1)  # [T*(|keep|)]
                    new_feat = feat_masked.index_select(0, idx_nodes).view(
                        n_step, keep_agents_b.numel(), self.hidden_dim
                    )

                # place into row aligned to this agent in valid_agent
                row = int(row_of_agent[a_id].item())
                feat_list[row] = new_feat

        return feat_list

    # @torch.no_grad()
    def refer1(
            self,
            feat_a_pt,  # [n_step * n_agents_total, hidden_dim]  (same x used in forward)
            r_a2a,  # [E, rdim] or None
            edge_index_a2a,  # [2, E]
            train_mask,  # [n_agents_total] (bool; True = ablate this agent)
            batch_s_repeat,  # [n_agents_total, ...], batch id in [:,0]
            n_step: int,
    ):
        import torch

        device = feat_a_pt.device
        hidden_dim = feat_a_pt.size(-1)

        # # ---- 0) Fill caches with ONE forward (dropout OFF for exact math) ----
        # self.eval()
        # _ = self.forward(feat_a_pt, r_a2a, edge_index_a2a)

        # ---- 1) Pull caches (PRE-dropout softmax for exact renorm) ----
        C = self._cache
        attn = C["attn_soft"]  # [E, H]
        V = C["v_j"]  # [E, H, Dh]
        dst = C["dst"]  # [E]
        src = C["src"]  # [E]
        x_dst = C["x_dst"]  # [N, hidden]

        N, H, Dh = x_dst.size(0), V.size(1), V.size(2)
        A_total = batch_s_repeat.size(0)
        T = n_step
        assert N == T * A_total, "Expect time-major nodes: node_id = t*A_total + agent_id"

        train_mask = train_mask.to(torch.bool)
        mask_agents = torch.where(train_mask)[0]  # <-- ONLY ablate these agents
        M = mask_agents.numel()
        if M == 0:
            return []

        # ---- 2) Base aggregate y_pre = sum_j a_ij V_j (PRE-dropout) ----
        msg = (attn.unsqueeze(-1) * V).reshape(-1, H * Dh)  # [E, H*Dh]
        y_pre = torch.zeros((N, H * Dh), device=device)
        y_pre.index_add_(0, dst, msg)
        y_pre = y_pre.view(N, H, Dh)  # [N, H, Dh]

        # ---- 3) Vectorized per-agent Δ (each row masks ONE agent in mask_agents) ----
        # Map each source node → row id IF its agent is in mask_agents; else -1
        map_agent_to_row = torch.full((A_total,), -1, device=device, dtype=torch.long)
        map_agent_to_row[mask_agents] = torch.arange(M, device=device)
        map_src_to_row = map_agent_to_row.repeat(T)  # [N], time-major expansion

        m_row = map_src_to_row[src]  # [E]
        keep_edges = m_row >= 0
        if keep_edges.any():
            attn_e = attn[keep_edges]  # [E', H]
            V_e = V[keep_edges]  # [E', H, Dh]
            dst_e = dst[keep_edges]  # [E']
            m_row_e = m_row[keep_edges]  # [E']
            y_dst_pre = y_pre[dst_e]  # [E', H, Dh]

            a = attn_e.unsqueeze(-1)  # [E', H, 1]
            inv = (1.0 - attn_e).clamp_min(1e-8).unsqueeze(-1)
            delta_e = (a / inv) * (y_dst_pre - a * V_e) - a * V_e  # [E', H, Dh]

            MN = M * N
            lin = m_row_e * N + dst_e
            delta_flat = torch.zeros((MN, H * Dh), device=device)
            delta_flat.index_add_(0, lin, delta_e.view(-1, H * Dh))
            delta_all_pre = delta_flat.view(M, N, H, Dh)  # [M, N, H, Dh]
            y_all_pre = y_pre.unsqueeze(0) + delta_all_pre  # [M, N, H, Dh]
        else:
            # None of the masked agents had outgoing edges; nothing changes.
            y_all_pre = y_pre.unsqueeze(0).expand(M, -1, -1, -1)  # [M, N, H, Dh]

        # ---- 4) Apply SAME gating/update + to_out + residual + FF ----
        y_all_pre_flat = y_all_pre.view(M, N, H * Dh)  # [M, N, H*Dh]
        x_dst_exp = x_dst.unsqueeze(0).expand(M, N, hidden_dim)  # [M, N, hidden]
        to_s_x = self.to_s(x_dst).unsqueeze(0).expand(M, N, H * Dh)
        g = torch.sigmoid(self.to_g(torch.cat([y_all_pre_flat, x_dst_exp], dim=-1)))
        post_flat = y_all_pre_flat + g * (to_s_x - y_all_pre_flat)

        attn_out_all = self.to_out(post_flat)  # [M, N, hidden]
        x_after_attn = feat_a_pt.unsqueeze(0) + self.attn_postnorm(attn_out_all)
        ff_in = self.ff_prenorm(x_after_attn)
        ff_out = self._ff_block(ff_in)
        masked_outputs = x_after_attn + self.ff_postnorm(ff_out)  # [M, N, hidden]

        # ---- 5) Build per-row kept agents (same batch, train_mask, and NOT the masked one) ----
        agent_batch_all = batch_s_repeat[:, 0].to(torch.long)  # [A_total]
        batch_of_row = agent_batch_all[mask_agents]  # [M]
        same_batch = (agent_batch_all.unsqueeze(0) == batch_of_row.unsqueeze(1))  # [M, A_total]
        keep_agent_mask = same_batch & train_mask.unsqueeze(0) & \
                          (torch.arange(A_total, device=device).unsqueeze(0) != mask_agents.unsqueeze(1))

        # Gather per-row indices (ragged → padded), then build the python feat_list
        rows, cols = torch.nonzero(keep_agent_mask, as_tuple=True)  # [S],[S]
        counts = torch.bincount(rows, minlength=M)
        order = torch.argsort(rows)
        rows_s, cols_s = rows[order], cols[order]
        start = torch.zeros(M, dtype=torch.long, device=device)
        if M > 1:
            start[1:] = counts.cumsum(0)[:-1]
        pos = torch.arange(rows_s.numel(), device=device) - start[rows_s]
        max_kept = int(counts.max().item()) if M > 0 else 0

        keep_idx = torch.zeros((M, max_kept), dtype=torch.long, device=device)
        if rows_s.numel() > 0:
            keep_idx[rows_s, pos] = cols_s

        mo = masked_outputs.view(M, T, A_total, hidden_dim)  # [M, T, A, H]
        feats_padded = mo.gather(2, keep_idx[:, None, :, None].expand(M, T, max_kept, hidden_dim))
        valid_cols_mask = (torch.arange(max_kept, device=device)[None, :] < counts[:, None])
        feats_padded = feats_padded * valid_cols_mask[:, None, :, None]  # zero padded cols

        feat_list = [
            feats_padded[i, :, :counts[i].item(), :].contiguous()  # [T, kept_i, H]
            for i in range(M)
        ]
        return feat_list

# device = logit_original.device
# A = batch_id.numel()
# Dshape = logit_original.shape[1:]
#
# # --- Baseline 'others' sum per agent: (batch sum) - (self) ---
# # Compress batches
# uniq, inv = torch.unique(batch_id, sorted=True, return_inverse=True)  # inv: [A] in [0..U-1]
# U = uniq.numel()
#
# # Sum originals per batch
# sum_batch = torch.zeros((U,) + Dshape, device=device, dtype=logit_original.dtype)
# sum_batch.index_add_(0, inv, logit_original)  # [U, *D]
#
# # Broadcast batch sum back to each agent, then remove the agent's own logit
# sum_orig_A = sum_batch.index_select(0, inv)   # [A, *D]
#
# counts = torch.tensor([f.shape[1] for f in feat_list], device=device, dtype=torch.long)  # [A]
# owner = torch.arange(A, device=device).repeat_interleave(counts)  # [M_default]
#
# # --- Ablated 'others' sum per agent: sum rows belonging to that agent ---
# sum_abla_A = torch.zeros((A,) + Dshape, device=device, dtype=logit_original.dtype)
# sum_abla_A.index_add_(0, owner, ablated_logit)  # [A, *D]
#
# # --- Reward per agent ---
# rewards1 = sum_orig_A - sum_abla_A  # [A, *D]
#
# print(1)

