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
    # Step 2: Get relative vectors: y - x (N_edges, 2)

    final_edge_index = edge_index[:, mask]

    return final_edge_index

def radiusGraphNearest_head(x,x_heading, batch, r, loop, max_num_neighbors):
    edge_index = knn_graph(x, k=max_num_neighbors, batch=batch, loop=loop)
    row, col = edge_index
    #distances = (x[col] - x[row]).norm(dim=1)
    #mask = distances <= r
    # Step 2: Get relative vectors: y - x (N_edges, 2)
    rel = x[col]-x[row]

    # Step 3: Rotate into x-frame (heading[col])
    theta = x_heading[row]

    mask = get_mask(rel,theta,forward=r,back=r//2,width=r//2)

    final_edge_index = edge_index[:, mask]

    return final_edge_index.flip(0)

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

def radiusGraphNearest2(x,y,x_heading,r, batch_x,batch_y,  max_num_neighbors):
    edge_index = knn(y, x, max_num_neighbors, batch_x=batch_y, batch_y=batch_x)
    row, col = edge_index# row is
    # distances = (x[row] - y[col]).norm(dim=1)
    # mask = (distances <= r)

    # Step 2: Get relative vectors: y - x (N_edges, 2)
    rel = y[col]-x[row]

    # Step 3: Rotate into x-frame (heading[col])
    theta = x_heading[row]

    mask = get_mask(rel,theta,forward=r,back=r//2,width=r//2)

    final_edge_index = edge_index[:, mask]

    return final_edge_index.flip(0)

def radiusGraphNearest_inv(x,y,r, batch_x,batch_y,  max_num_neighbors):
    edge_index = knn(x, y, max_num_neighbors, batch_x=batch_x, batch_y=batch_y)
    row, col = edge_index
    distances = (x[col] - y[row]).norm(dim=1)
    mask = (distances <= r)
    final_edge_index = edge_index[:, mask]

    return final_edge_index

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



def build_map2agent_edge(
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl*n_step]
):
    n_step = pos_a.shape[1]
    mask_pl2a = mask.transpose(0, 1).reshape(-1)
    pos_s = pos_a.transpose(0, 1).flatten(0, 1)
    head_s = head_a.transpose(0, 1).reshape(-1)
    head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

    pos_pl = pos_pl.repeat(n_step, 1)
    orient_pl = orient_pl.repeat(n_step)

    num_pl=len(pos_pl)

    pos_pls=torch.cat((pos_pl, pos_s), dim=0)

    mask_pl2a_all=torch.cat((torch.ones_like(batch_pl).to(torch.bool), torch.zeros_like(batch_s).to(torch.bool)), dim=0)
    mask_a2a_all=torch.cat((torch.zeros_like(batch_pl).to(torch.bool), mask_pl2a), dim=0)
    batch_pls=torch.cat((batch_pl, batch_s), dim=0)

    edge_index_pl2a_a2a = radiusGraphNearest2(x=pos_s[:, :2],
                                          y=pos_pls[:, :2],
                                          r=self.pl2a_radius,
                                          batch_x=batch_s,
                                          batch_y=batch_pls,
                                          max_num_neighbors=self.pt2a_neighbor)# edge 0 :pl , edge 1: [pl, a]

    edge_index_pl2a_a2a = edge_index_pl2a_a2a[:, mask_pl2a[edge_index_pl2a_a2a[1]]]

    edge_index_pl2a = edge_index_pl2a_a2a[:,mask_pl2a_all[edge_index_pl2a_a2a[0]]]

    edge_index_a2a = edge_index_pl2a_a2a[:,mask_a2a_all[edge_index_pl2a_a2a[0]]]

    edge_index_a2a[0]-=num_pl

    rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
    rel_orient_pl2a = wrap_angle(
        orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
    )
    r_pl2a = torch.stack(
        [
            torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_s[edge_index_pl2a[1]],
                nbr_vector=rel_pos_pl2a[:, :2],
            ),
            rel_orient_pl2a,
        ],
        dim=-1,
    )

    r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

    rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
    rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
    r_a2a = torch.stack(
        [
            torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_s[edge_index_a2a[1]],
                nbr_vector=rel_pos_a2a[:, :2],
            ),
            rel_head_a2a,
        ],
        dim=-1,
    )


    r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

    return edge_index_pl2a, r_pl2a,edge_index_a2a, r_a2a