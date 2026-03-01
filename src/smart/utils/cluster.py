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


def batch_increasing_schedule(N, S=256+1, gamma=0.9):
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

    # if torch.max(counts)>256:
    #     print(torch.max(counts))

    schedules = batch_increasing_schedule(counts)

    step_number=schedules.shape[1]-1

    rand_idx = torch.randint(0, step_number, (num_graphs,), device=counts.device)

    batch_idx = torch.arange(num_graphs, device=counts.device)

    k_per_graph = schedules[batch_idx, rand_idx]
    k1_per_graph = schedules[batch_idx, rand_idx + 1]
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

    return centroids,centroids1, k_per_graph,k1_per_graph




def cluster_points( pos, batch, type, num_graphs):
    veh_pos=pos[type==0]
    veh_batch=batch[type==0]

    centroids_b,centroids1_b, k_per_graph,k1_per_graph=batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)

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

    return less_centroids, more_batch, more_centroids, more_type, k_per_graph