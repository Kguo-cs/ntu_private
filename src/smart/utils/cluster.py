import torch

def kmeans_fast( x, k, iters=10):
    N, D = x.shape
    device = x.device

    # random initialization
    perm = torch.randperm(N, device=device)
    centroids = x[perm[:k]].clone()

    for _ in range(iters):
        # squared distance (faster than cdist)
        dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ centroids.t()
                + centroids.pow(2).sum(1)
        )  # (N, k)

        labels = dist.argmin(dim=1)

        # vectorized centroid update
        counts = torch.bincount(labels, minlength=k).clamp(min=1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(
            0,
            labels.unsqueeze(1).expand(-1, D),
            x,
        )

        centroids = new_centroids / counts.unsqueeze(1)

    return centroids


import torch


def kmeans( padded, mask,k_per_graph,batch,pos, iters=10):

    max_k = k_per_graph.max().item()

    num_graphs,max_points, D=padded.shape

    device=padded.device

    cluster_mask = (
            torch.arange(max_k, device=device)
            .unsqueeze(0)
            < k_per_graph.unsqueeze(1)
    )  # (num_graphs, max_k)
    # --- generate random indices per graph ---
    # first, get random floats per graph & point
    rand_vals = torch.rand((num_graphs, max_points), device=device)

    # mask invalid points so they won't be selected
    rand_vals[~mask] = -1.0  # ensure they are ignored

    # argsort descending so top-k picks random valid points
    sorted_idx = rand_vals.argsort(dim=1, descending=True)  # (num_graphs, max_points)

    selected_idx = sorted_idx[:, :max_k]  # (num_graphs, max_k)
    # zero-out positions where k < max_k
    selected_idx = selected_idx * cluster_mask.long()

    # select the first k_per_graph indices for each graph
    # create batch offsets for advanced indexing
    graph_idx = torch.arange(num_graphs, device=device).unsqueeze(1)  # (num_graphs,1)

    # gather centroids
    centroids = padded[graph_idx, selected_idx]  # (num_graphs, max_k, D)

    centroids_mask=~(cluster_mask[:,None] & mask[:,:,None])[mask]

    # --- K-Means iterations ---
    for _ in range(iters):
        # distances: (num_graphs, max_points, max_k)
        dist = (
            padded.pow(2).sum(-1, keepdim=True)
            - 2 * padded @ centroids.transpose(1, 2)
            + centroids.pow(2).sum(-1).unsqueeze(1)
        )
        dist=dist[mask]

        dist[centroids_mask]=float('inf')

        # assign labels per point (capped at graph's k)
        labels_per_point = dist.argmin(dim=-1)  # (num_graphs, max_points)

        # labels_per_point=labels[mask]

        # total clusters across all graphs
        total_clusters = num_graphs * max_k

        new_centroids = torch.zeros(total_clusters, D, device=device)
        counts_centroids = torch.zeros(total_clusters, device=device)

        # global cluster index per point
        global_idx = batch * max_k + labels_per_point   # (N_total,)

        # scatter sums
        new_centroids.index_add_(0, global_idx, pos)
        counts_centroids.index_add_(0, global_idx, torch.ones_like(global_idx, dtype=torch.float))

        # reshape back
        new_centroids = new_centroids.view(num_graphs, max_k, D)
        counts_centroids = counts_centroids.view(num_graphs, max_k)

        centroids = new_centroids / counts_centroids.clamp(min=1).unsqueeze(-1)
        # # --- vectorized centroid update ---
        # new_centroids = torch.zeros_like(centroids)
        # counts_centroids = torch.zeros(num_graphs, max_k, device=device)
        #
        # # flatten graphs and clusters to 1D indices
        # flat_graph = torch.arange(num_graphs, device=device).unsqueeze(1).expand(-1, max_points).reshape(-1)
        # flat_labels = labels.reshape(-1)
        # flat_points = padded.reshape(-1, D)
        # flat_mask = mask.reshape(-1)
        #
        # # only valid points
        # valid = flat_mask
        # flat_idx = flat_graph[valid] * max_k + flat_labels[valid]  # unique index per graph+cluster
        #
        # new_centroids = new_centroids.reshape(-1, D)
        # counts_centroids = counts_centroids.reshape(-1)
        #
        # # scatter add
        # new_centroids.index_add_(0, flat_idx, flat_points[valid])
        # counts_centroids.index_add_(0, flat_idx, torch.ones_like(flat_labels[valid], dtype=torch.float, device=device))
        #
        # # reshape back
        # new_centroids = new_centroids.reshape(num_graphs, max_k, D)
        # counts_centroids = counts_centroids.reshape(num_graphs, max_k)
        #
        # centroids = new_centroids / counts_centroids.clamp(min=1).unsqueeze(-1)

    return centroids


def batch_increasing_schedule(N, S=128+1, gamma=1):
    """
    N: (B,) tensor of maximum levels per batch
    S: total number of steps (int)
    gamma: curvature (>1 gives more steps to small values)

    Returns:
        schedule: (B, S) integer tensor
    """
    s = torch.arange(S, device=N.device).float()  # (S,)
    t = (s / S).pow(gamma)  # (S,)

    schedule = torch.ceil_(N[:, None] * t[None, :])
    schedule = torch.minimum(schedule, N[:, None])

    schedule =torch.cat([schedule,schedule[:,-1:].repeat(1,64)],dim=-1)

    return schedule.long()

 # [00:48<11:03,  5.34it/s, v_num=vx1p]
def batched_kmeans_variable_k(pos, batch,  num_graphs,iters=10):
    """
    Batched K-Means per graph, allowing variable number of clusters per graph.

    Args:
        pos: (N_total, D) positions
        batch: (N_total,) graph index per point (0..num_graphs-1)
        k_per_graph: (num_graphs,) number of clusters for each graph
        iters: K-Means iterations

    Returns:
        centroids: (num_graphs, max_k, D) padded centroids
        labels: (N_total,) cluster assignment per point
    """
    device = pos.device
    D = pos.shape[1]

    counts = torch.bincount(batch, minlength=num_graphs)
    offsets = torch.cumsum(counts, dim=0)
    offsets = torch.cat([torch.tensor([0], device=batch.device), offsets[:-1]])

    idx_in_graph = torch.arange(len(batch), device=batch.device) - offsets[batch]

    max_points = counts.max()

    padded = torch.zeros(num_graphs, max_points, D, device=pos.device)
    mask = torch.zeros(num_graphs, max_points, dtype=torch.bool, device=pos.device)

    padded[batch, idx_in_graph] = pos
    mask[batch, idx_in_graph] = True

    schedules = batch_increasing_schedule(counts)

    step_number=schedules.shape[1]-1

    step_idx = torch.randint(0, step_number, (num_graphs,), device=counts.device)

    batch_idx = torch.arange(num_graphs, device=counts.device)

    k_per_graph = schedules[batch_idx, step_idx]
    k1_per_graph = schedules[batch_idx, step_idx + 1]
    # # sample uniform float in [0,1)
    # u = torch.rand(num_graphs, device=device)
    # # scale per-graph and floor
    # k_per_graph = torch.floor(u * (512 + 1)).long()
    #
    # k_per_graph = torch.minimum(k_per_graph, counts)

    # k1 = min(k+1, counts)
    # k1_per_graph = torch.minimum(k_per_graph + 1, counts)

    centroids=kmeans(padded, mask,k_per_graph,batch,pos)

    centroids1=kmeans(padded, mask,k1_per_graph,batch,pos)

    return centroids,centroids1, k_per_graph,k1_per_graph,step_idx

def build_less_more_grouped(
    pos,
    type,
    batch,
    num_graphs,
):
    device = pos.device
    D = pos.shape[1]
    G = num_graphs

    veh_mask = (type == 0)

    veh_pos=pos[type==0]
    veh_batch=batch[type==0]

    centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx=batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)

    nonveh_mask = ~veh_mask

    nonveh_pos = pos[nonveh_mask]
    nonveh_batch = batch[nonveh_mask]
    nonveh_type = type[nonveh_mask]

    # Count non-veh per graph
    nonveh_count = torch.bincount(
        nonveh_batch,
        minlength=G
    )

    # --------------------------------------------
    # Compute sizes per graph
    # --------------------------------------------
    less_sizes = k1_per_graph + nonveh_count
    more_sizes = k1_per_graph + nonveh_count

    # Prefix sums (graph offsets)
    less_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        torch.cumsum(less_sizes, dim=0)
    ])[:-1]

    more_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        torch.cumsum(more_sizes, dim=0)
    ])[:-1]

    total_less = less_sizes.sum()
    total_more = more_sizes.sum()

    # Allocate final tensors
    less_centroids = torch.zeros(total_less, D, device=device)

    more_centroids = torch.zeros(total_more, D, device=device)
    more_batch = torch.zeros(total_more, device=device, dtype=torch.long)
    more_type = torch.zeros(total_more, device=device, dtype=torch.long)

    # ----------------------------------------------------
    # 1) Write centroids (LESS and MORE)
    # ----------------------------------------------------
    max_k = centroids_b.shape[1]
    max_k1 = centroids1_b.shape[1]

    arange_k = torch.arange(max_k, device=device)
    arange_k1 = torch.arange(max_k1, device=device)

    valid_k = arange_k.unsqueeze(0) < k_per_graph.unsqueeze(1)
    valid_k1 = arange_k1.unsqueeze(0) < k1_per_graph.unsqueeze(1)

    # Flatten valid entries
    centroids_less_flat = centroids_b[valid_k]
    centroids_more_flat = centroids1_b[valid_k1]

    graph_ids_k = torch.arange(G, device=device).repeat_interleave(k_per_graph)
    graph_ids_k1 = torch.arange(G, device=device).repeat_interleave(k1_per_graph)

    # Compute write indices
    less_write_idx = less_offsets[graph_ids_k] + \
                     torch.arange(centroids_less_flat.shape[0], device=device) \
                     - torch.repeat_interleave(
                         torch.cumsum(k_per_graph, 0) - k_per_graph,
                         k_per_graph
                     )

    more_write_idx = more_offsets[graph_ids_k1] + \
                     torch.arange(centroids_more_flat.shape[0], device=device) \
                     - torch.repeat_interleave(
                         torch.cumsum(k1_per_graph, 0) - k1_per_graph,
                         k1_per_graph
                     )

    less_centroids[less_write_idx] = centroids_less_flat
    more_centroids[more_write_idx] = centroids_more_flat
    more_batch[more_write_idx] = graph_ids_k1
    more_type[more_write_idx] = 0

    # ----------------------------------------------------
    # 2) Write non-vehicle
    # ----------------------------------------------------
    # Compute per-nonveh relative index within its graph
    nonveh_local_idx = (
        torch.arange(nonveh_batch.shape[0], device=device)
        - torch.repeat_interleave(
            torch.cumsum(nonveh_count, 0) - nonveh_count,
            nonveh_count
        )
    )

    # Write positions
    less_nonveh_idx = (
        less_offsets[nonveh_batch]
        + k1_per_graph[nonveh_batch]
        + nonveh_local_idx
    )

    more_nonveh_idx = (
        more_offsets[nonveh_batch]
        + k1_per_graph[nonveh_batch]
        + nonveh_local_idx
    )

    less_centroids[less_nonveh_idx] = nonveh_pos

    more_centroids[more_nonveh_idx] = nonveh_pos
    more_batch[more_nonveh_idx] = nonveh_batch
    more_type[more_nonveh_idx] = nonveh_type

    return (
        less_centroids,
        more_centroids,
        more_batch,
        more_type,centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx
    )

def cluster_points( pos, batch, type, num_graphs):

    # less_centroids,more_centroids,more_batch,more_type,centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx=build_less_more_grouped(
    #     pos,
    #     type,
    #     batch,
    #     num_graphs
    #     )

    veh_pos=pos[type==0]
    veh_batch=batch[type==0]

    centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx=batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)


    device = pos.device
    less_centroids = []
    more_batch = []
    more_centroids = []
    more_type = []

    for i in range(num_graphs):
        mask = (batch == i)
        type_i = type[mask]

        x_non_veh = pos[mask][type_i != 0]

        type_non_veh = type_i[type_i != 0]

        # x = pos[mask][type_i == 0]
        # N = x.shape[0]
        # step = torch.randint(0, N + 1, (1,), device=device).item()
        # k = min(step, N)
        #
        # if k == 0:
        #     centroids = x[:k]
        # else:
        #     centroids = kmeans_fast(x, k)#x[:k]#
        #
        # k1 = min(k + 1, N)  # torch.randint(k+1, N+1, (1,), device=device).item()
        #
        # if k1 == 0:
        #     centroids1 = x[:k1]
        # else:
        #     centroids1 =kmeans_fast(x, k1)# x[:k1] #

        k=k_per_graph[i]
        k1=k1_per_graph[i]
        centroids=centroids_b[i][:k]
        centroids1=centroids1_b[i][:k1]

        # import matplotlib.pylab as plt
        #
        # plt.scatter(centroids[:,0].cpu().numpy(), centroids[:,1].cpu().numpy(),s=30, c='r')
        #
        # plt.scatter(centroids1[:,0].cpu().numpy(), centroids1[:,1].cpu().numpy(),s=20, c='b')
        #
        # plt.scatter(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(),s=10,c='g')
        #
        # plt.show()

        padding_centers = torch.zeros_like(centroids1[k:])

        less_centroids.append(torch.cat([centroids, padding_centers, x_non_veh], dim=0))

        more_batch.append(torch.zeros([k1 + len(x_non_veh)], device=device, dtype=torch.long) + i)

        more_centroids.append(torch.cat([centroids1, x_non_veh], dim=0))

        more_type.append(torch.cat([torch.zeros([k1], device=device, dtype=torch.long), type_non_veh], dim=0))

    less_centroids = torch.cat(less_centroids, dim=0)
    more_batch = torch.cat(more_batch, dim=0)
    more_centroids = torch.cat(more_centroids, dim=0)
    more_type = torch.cat(more_type, dim=0)
   #
   #  print(torch.all(less_centroids == less_centroids1))
   #  print(torch.all(more_centroids == more_centroids1))
   #  print(torch.all(more_batch == more_batch1))
   #  print(torch.all(more_type == more_type1))

    return less_centroids, more_batch, more_centroids, more_type,step_idx