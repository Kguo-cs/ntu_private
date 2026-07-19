from torch_geometric.nn.pool import knn_graph,knn
import torch.nn.functional as F
import torch
import  math
import torch

import torch


import torch

def visibility_aware_knn_with_radius_batch(pos, vis_mask, batch, k, max_radius):
    device = pos.device
    edge_src = []
    edge_dst = []

    # For each batch (scene), process independently
    unique_batches = batch.unique(sorted=True)
    for b in unique_batches:
        # Get agents in batch b
        idx = (batch == b)
        pos_b = pos[idx]                # [Nb, 2]
        vis_mask_b = vis_mask[idx]      # [Nb]
        num_agents = pos_b.size(0)

        if num_agents == 0:
            continue

        # Pairwise distance
        dists = torch.cdist(pos_b, pos_b, p=2)  # [Nb, Nb]
        dists.fill_diagonal_(float('inf'))

        # Visibility constraint
        vis_mask_b = vis_mask_b.bool()
        src_visible = vis_mask_b.unsqueeze(1)        # [Nb, 1]
        dst_visible = vis_mask_b.unsqueeze(0)        # [1, Nb]
        visibility_mask = (~src_visible) | dst_visible  # [Nb, Nb]

        # Radius constraint
        radius_mask = dists <= max_radius

        # Combine both
        allow_mask = visibility_mask & radius_mask
        dists[~allow_mask] = float('inf')

        k=min(k,num_agents)

        # Top-k nearest
        knn_dists, knn_indices = torch.topk(dists, k, dim=1, largest=False)

        # Valid edges (skip infs if too few neighbors)
        valid_mask = torch.isfinite(knn_dists)
        src_local = torch.arange(num_agents, device=device).unsqueeze(1).expand(num_agents, k)

        src_valid = src_local[valid_mask]
        dst_valid = knn_indices[valid_mask]

        # Map local indices to global
        global_idx = idx.nonzero(as_tuple=False).squeeze(1)
        src_global = global_idx[src_valid]
        dst_global = global_idx[dst_valid]

        edge_src.append(src_global)
        edge_dst.append(dst_global)

    if len(edge_src) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Final edge index
    edge_index = torch.stack([
        torch.cat(edge_dst, dim=0),
        torch.cat(edge_src, dim=0),
    ], dim=0)  # [2, E]

    return edge_index


# def nearest_mask(padd_pos,nearest_k,max_dist,mask):
#     # padd_pos: [B, N, D]
#     B, N, D = padd_pos.shape
#
#     # Compute squared distances (faster than full norm)
#     diff = padd_pos[:, :, None, :] - padd_pos[:, None, :, :]  # [B, N, N, D]
#     sq_dist = (diff ** 2).sum(-1)  # [B, N, N]
#
#     # Mask self-distance with large value (in-place)
#     inf = float('inf')
#     sq_dist.diagonal(dim1=1, dim2=2).fill_(inf)
#
#     sq_dist[mask]=inf
#
#     # Get indices of 10 nearest (squared) distances
#     top_dist,topk_idx = torch.topk(sq_dist, k=nearest_k, dim=-1, largest=False)  # [B, N, 10]
#
#     # Build nearest-10 mask efficiently (all True except topk)
#     a2a_mask = torch.ones((B, N, N), dtype=torch.bool, device=padd_pos.device)
#
#     dist_mask= top_dist < max_dist ** 2
#
#     batch = torch.arange(B, device=padd_pos.device)[:, None, None]
#     rows = torch.arange(N, device=padd_pos.device)[None, :, None]
#     a2a_mask[batch.expand_as(topk_idx)[dist_mask],
#              rows.expand_as(topk_idx)[dist_mask],
#              topk_idx[dist_mask]] = False
#
#     return a2a_mask
def build_batch( batch, num_graphs, n_step):
    batch = torch.cat(
        [
            batch + num_graphs * t
            for t in range(n_step)
        ],
        dim=0,
    )  # [n_agent*n_step]

    return batch


def nearest_mask(padd_pos, nearest_k, max_dist, mask):
    # padd_pos: [B, N, D]
    B, N, D = padd_pos.shape

    # Compute pairwise squared distances: [B, N, N]
    diff = padd_pos[:, :, None, :] - padd_pos[:, None, :, :]  # [B, N, N, D]
    sq_dist = (diff ** 2).sum(-1)

    # Mask self-distances
    idx = torch.arange(N, device=padd_pos.device)
    sq_dist[:, idx, idx] = float('inf')  # More efficient than diagonal().fill_()

    # Apply custom mask
    sq_dist = sq_dist.masked_fill(mask, float('inf'))

    # Find nearest_k neighbors within max_dist
    k_eff = min(int(nearest_k), int(N))
    if k_eff <= 0:
        return torch.ones((B, N, N), dtype=torch.bool, device=padd_pos.device)
    top_dist, topk_idx = torch.topk(sq_dist, k=k_eff, dim=-1, largest=False)  # [B, N, k_eff]
    dist_mask = top_dist < max_dist ** 2

    # Create full attention mask: default to True
    a2a_mask = torch.ones((B, N, N), dtype=torch.bool, device=padd_pos.device)

    # Set allowed (nearest) entries to False
    b_idx, n_idx = torch.meshgrid(
        torch.arange(B, device=padd_pos.device),
        torch.arange(N, device=padd_pos.device),
        indexing='ij'
    )
    b_idx = b_idx.unsqueeze(-1).expand(-1, -1, k_eff)[dist_mask]
    n_idx = n_idx.unsqueeze(-1).expand(-1, -1, k_eff)[dist_mask]
    k_idx = topk_idx[dist_mask]

    a2a_mask[b_idx, n_idx, k_idx] = False
    return a2a_mask

def nearest_mask2(padd_pos, padd_pos1, nearest_k, max_dist, mask):
    """
    Args:
        padd_pos:   [B, Nq, D]  - Query positions
        padd_pos1:  [B, Nk, D]  - Key positions
        nearest_k:  int         - Number of nearest neighbors to allow
        max_dist:   float       - Maximum distance threshold
        mask:       [B, Nq, Nk] - Predefined mask (True = ignore)

    Returns:
        a2a_mask:   [B, Nq, Nk] - True means masked (not a valid connection)
    """
    B, Nq, D = padd_pos.shape
    Nk = padd_pos1.shape[1]

    # Compute squared distances between padd_pos (query) and padd_pos1 (key)
    diff = padd_pos[:, :, None, :] - padd_pos1[:, None, :, :]  # [B, Nq, Nk, D]
    sq_dist = (diff ** 2).sum(-1)  # [B, Nq, Nk]

    # Apply input mask
    sq_dist = sq_dist.masked_fill(mask, float('inf'))

    # Get top-k smallest distances (nearest neighbors)
    k_eff = min(int(nearest_k), int(Nk))
    if k_eff <= 0:
        return torch.ones((B, Nq, Nk), dtype=torch.bool, device=padd_pos.device)
    top_dist, topk_idx = torch.topk(sq_dist, k=k_eff, dim=-1, largest=False)  # [B, Nq, k_eff]

    # Apply distance threshold
    dist_mask = top_dist < max_dist ** 2  # [B, Nq, k_eff]

    # Initialize mask as fully masked (True)
    a2a_mask = torch.ones((B, Nq, Nk), dtype=torch.bool, device=padd_pos.device)

    # Index updates: set allowed neighbors to False
    b_idx, nq_idx = torch.meshgrid(
        torch.arange(B, device=padd_pos.device),
        torch.arange(Nq, device=padd_pos.device),
        indexing='ij'
    )
    b_idx = b_idx.unsqueeze(-1).expand(-1, -1, k_eff)[dist_mask]
    nq_idx = nq_idx.unsqueeze(-1).expand(-1, -1, k_eff)[dist_mask]
    nk_idx = topk_idx[dist_mask]  # indices in padd_pos1 (key set)

    a2a_mask[b_idx, nq_idx, nk_idx] = False
    return a2a_mask

def radiusGraphNearest(x, batch, r, loop, max_num_neighbors):
    if x.numel() == 0 or max_num_neighbors <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    # Clamp globally. This avoids obvious topk/knn failures when the learnable
    # selector requests a larger candidate pool than the number of available
    # agents in small scenes.
    max_k = x.size(0) if loop else max(x.size(0) - 1, 0)
    if max_k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    k = min(int(max_num_neighbors), int(max_k))

    edge_index = knn_graph(x, k=k, batch=batch, loop=loop)        #source_to_target  edge_index[0] = dst edge_index[1] = src
    src, dst = edge_index
    distances = (x[src] - x[dst]).norm(dim=1)
    mask = distances <= r
    # Step 2: Get relative vectors: y - x (N_edges, 2)

    final_edge_index = edge_index[:, mask]

    return final_edge_index


def get_mask(rel,theta,forward=40,back=20,width=20):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Rotation matrix: inverse of heading
    # [cos, sin; -sin, cos] applied to (dx, dy)
    rel_x = rel[:, 0] * cos_theta + rel[:, 1] * sin_theta
    rel_y = -rel[:, 0] * sin_theta + rel[:, 1] * cos_theta

    # Step 4: Spatial filtering in local x-frame
    in_front = (rel_x >= -back) & (rel_x <= forward)
    in_width = (rel_y >= -width ) & (rel_y <= width)
    mask = in_front & in_width

    return mask

def radiusGraphNearest2(x,y,r, batch_x,batch_y,  max_num_neighbors):
    if x.numel() == 0 or y.numel() == 0 or max_num_neighbors <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    k = min(int(max_num_neighbors), int(y.size(0)))
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    edge_index = knn(y, x, k, batch_x=batch_y, batch_y=batch_x) # for each object in x, the nearest point in y
    src, dst = edge_index
    distances = (x[src] - y[dst]).norm(dim=1)      # x is agent , y is map , src is agent , dst is map,

    mask = (distances < r) & (distances>0)

    # Step 2: Get relative vectors: y - x (N_edges, 2)
    # rel = y[col]-x[row]
    # theta = x_heading[row]
    # mask = get_mask(rel,theta,forward=r,back=r//2,width=r//2)
    final_edge_index = edge_index[:, mask]

    return final_edge_index.flip(0)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def generate_causal_mask(seq_len, device='cpu'):
    # Upper-triangular matrix filled with -inf, including diagonal=1
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask  # [T, T]


def generate_limited_causal_mask(seq_len, history_len,n_agent=1, device='cpu'):
    # for i in range(seq_len):
    #     start = max(0, i - history_len + 1)
    #     mask[i, start:i + 1] = 0.0  # allow self and last `history_len - 1` tokens

    t=torch.arange(seq_len, device=device)[:,None].repeat(1,n_agent).flatten(0,1)

    i = t.unsqueeze(1)
    j = t.unsqueeze(0)
    mask = (j> i) | (j < i - history_len+ 1)  # True means masked

    return mask  # shape: [seq_len, seq_len]


def insert_ego(map_feature,ego_feature,ego_pos,ego_heading):
    # Map features
    batch = map_feature['batch']  # (N,)
    pt_token = map_feature['pt_token']  # (N, F)
    position = map_feature['position']  # (N, D)
    orientation = map_feature['orientation']  # (N, H)

    device = batch.device
    B = ego_feature.size(0)
    E = ego_feature.size(1)  # = 2

    # =========================
    # 1. Count map elements per batch
    # =========================
    counts = torch.bincount(batch, minlength=B)  # (B,)

    # =========================
    # 2. Compute batch offsets (+E ego per batch)
    # =========================
    new_counts = counts + E
    offsets = torch.cumsum(new_counts, dim=0) - new_counts  # (B,)

    # =========================
    # 3. Ego indices
    # =========================
    ego_indices = (
            offsets[:, None] +
            torch.arange(E, device=device)[None, :]
    ).reshape(-1)  # (B*E,)

    # =========================
    # 4. Map indices (order-preserving)
    # =========================
    pos_in_batch = (
            torch.arange(batch.size(0), device=device)
            - torch.cumsum(counts, 0)[batch]
            + counts[batch]
    )

    map_indices = offsets[batch] + E + pos_in_batch

    # =========================
    # 5. Allocate outputs
    # =========================
    N_new = new_counts.sum().item()

    pt_token_out = torch.empty(
        (N_new, pt_token.size(1)),
        device=device,
        dtype=pt_token.dtype,
    )

    position_out = torch.empty(
        (N_new, position.size(1)),
        device=device,
        dtype=position.dtype,
    )

    orientation_out = torch.empty(
        (N_new),
        device=device,
        dtype=orientation.dtype,
    )

    batch_out = torch.empty(
        (N_new,),
        device=device,
        dtype=batch.dtype,
    )

    # =========================
    # 6. Scatter ego (flatten B×E → (B*E))
    # =========================
    pt_token_out[ego_indices] = ego_feature.reshape(-1, pt_token.size(1))
    position_out[ego_indices] = ego_pos.reshape(-1, position.size(1))
    orientation_out[ego_indices] = ego_heading.reshape(-1)
    batch_out[ego_indices] = torch.repeat_interleave(
        torch.arange(B, device=device),
        E
    )

    # =========================
    # 7. Scatter map features
    # =========================
    pt_token_out[map_indices] = pt_token
    position_out[map_indices] = position
    orientation_out[map_indices] = orientation
    batch_out[map_indices] = batch

    # =========================
    # 8. Write back
    # =========================
    map_feature['pt_token'] = pt_token_out
    map_feature['position'] = position_out
    map_feature['orientation'] = orientation_out
    map_feature['batch'] = batch_out

    return map_feature