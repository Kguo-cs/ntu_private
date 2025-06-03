from torch_geometric.nn.pool import knn_graph,knn
import torch.nn.functional as F
import torch
import  math


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
    top_dist, topk_idx = torch.topk(sq_dist, k=nearest_k, dim=-1, largest=False)  # [B, N, k]
    dist_mask = top_dist < max_dist ** 2

    # Create full attention mask: default to True
    a2a_mask = torch.ones((B, N, N), dtype=torch.bool, device=padd_pos.device)

    # Set allowed (nearest) entries to False
    b_idx, n_idx = torch.meshgrid(
        torch.arange(B, device=padd_pos.device),
        torch.arange(N, device=padd_pos.device),
        indexing='ij'
    )
    b_idx = b_idx.unsqueeze(-1).expand(-1, -1, nearest_k)[dist_mask]
    n_idx = n_idx.unsqueeze(-1).expand(-1, -1, nearest_k)[dist_mask]
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
    top_dist, topk_idx = torch.topk(sq_dist, k=nearest_k, dim=-1, largest=False)  # [B, Nq, k]

    # Apply distance threshold
    dist_mask = top_dist < max_dist ** 2  # [B, Nq, k]

    # Initialize mask as fully masked (True)
    a2a_mask = torch.ones((B, Nq, Nk), dtype=torch.bool, device=padd_pos.device)

    # Index updates: set allowed neighbors to False
    b_idx, nq_idx = torch.meshgrid(
        torch.arange(B, device=padd_pos.device),
        torch.arange(Nq, device=padd_pos.device),
        indexing='ij'
    )
    b_idx = b_idx.unsqueeze(-1).expand(-1, -1, nearest_k)[dist_mask]
    nq_idx = nq_idx.unsqueeze(-1).expand(-1, -1, nearest_k)[dist_mask]
    nk_idx = topk_idx[dist_mask]  # indices in padd_pos1 (key set)

    a2a_mask[b_idx, nq_idx, nk_idx] = False
    return a2a_mask
# def nearest_mask2(padd_pos,padd_pos1,nearest_k,max_dist,mask):
#     # padd_pos: [B, N, D]
#     B, N, D = padd_pos.shape
#     B, N1, D = padd_pos1.shape
#
#     # Compute squared distances (faster than full norm)
#     diff = padd_pos[:, :, None, :] - padd_pos1[:, None, :, :]  # [B, N, N1, D]
#     sq_dist = (diff ** 2).sum(-1)  # [B, N, N]
#
#     # Optional: mask out distances greater than max_dist
#     #sq_dist[sq_dist > max_dist ** 2] = float('inf')  # skip far neighbors
#
#     # Get indices of 10 nearest (squared) distances
#     topk_idx = torch.topk(sq_dist, k=nearest_k, dim=-1, largest=False).indices  # [B, N, 10]
#
#     # Build nearest-10 mask efficiently (all True except topk)
#     a2a_mask = torch.ones((B, N, N1), dtype=torch.bool, device=padd_pos.device)
#     batch = torch.arange(B, device=padd_pos.device)[:, None, None]
#     rows = torch.arange(N, device=padd_pos.device)[None, :, None]
#     a2a_mask[batch, rows, topk_idx] = False
#
#     return a2a_mask



def radiusGraphNearest(x, batch, r, loop, max_num_neighbors):
    edge_index = knn_graph(x, k=max_num_neighbors, batch=batch, loop=loop)
    row, col = edge_index
    distances = (x[col] - x[row]).norm(dim=1)
    mask = distances <= r
    final_edge_index = edge_index[:, mask]

    return final_edge_index

def radiusGraphNearest2(x,y,r, batch_x,batch_y,  max_num_neighbors):
    edge_index = knn(y, x, max_num_neighbors, batch_x=batch_y, batch_y=batch_x)
    row, col = edge_index
    distances = (x[row] - y[col]).norm(dim=1)
    mask = distances <= r
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


def generate_limited_causal_mask(seq_len, history_len, device='cpu'):
    # for i in range(seq_len):
    #     start = max(0, i - history_len + 1)
    #     mask[i, start:i + 1] = 0.0  # allow self and last `history_len - 1` tokens

    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (j> i) | (j < i - history_len+ 1)  # True means masked

    return mask  # shape: [seq_len, seq_len]


