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
    N_total = pos.shape[0]

    # --- compute point offsets within each graph ---
    counts = torch.bincount(batch, minlength=num_graphs)
    idx_in_graph = torch.zeros(N_total, device=device, dtype=torch.long)
    temp_counts = torch.zeros(num_graphs, device=device, dtype=torch.long)
    for i in range(N_total):
        g = batch[i]
        idx_in_graph[i] = temp_counts[g]
        temp_counts[g] += 1

    # --- create padded tensor for positions ---
    max_points = counts.max().item()
    padded = torch.zeros(num_graphs, max_points, D, device=device)
    mask = torch.zeros(num_graphs, max_points, dtype=torch.bool, device=device)
    padded[batch, idx_in_graph] = pos
    mask[batch, idx_in_graph] = True

    k_per_graph = torch.zeros_like(temp_counts)

    for i in range(num_graphs):
        k_per_graph[i]=torch.randint(0, temp_counts[i] + 1, (1,), device=device).item()

    max_k = k_per_graph.max().item()

    # --- initialize centroids (first k points per graph) ---
    centroids = torch.zeros(num_graphs, max_k, D, device=device)

    for g in range(num_graphs):
        k = k_per_graph[g].item()
        centroids[g, :k] = padded[g, :k]

    # --- K-Means iterations ---
    for _ in range(iters):
        # distances: (num_graphs, max_points, max_k)
        dist = (
            padded.pow(2).sum(-1, keepdim=True)
            - 2 * padded @ centroids.transpose(1, 2)
            + centroids.pow(2).sum(-1).unsqueeze(1)
        )
        dist[~mask] = float('inf')  # ignore padded points

        # assign labels per point (capped at graph's k)
        labels = dist.argmin(dim=2)  # (num_graphs, max_points)
        # mask labels exceeding k_per_graph
        for g in range(num_graphs):
            labels[g, labels[g] >= k_per_graph[g]] = k_per_graph[g] - 1

        # --- vectorized centroid update ---
        new_centroids = torch.zeros_like(centroids)
        counts_centroids = torch.zeros(num_graphs, max_k, device=device)

        # flatten graphs and clusters to 1D indices
        flat_graph = torch.arange(num_graphs, device=device).unsqueeze(1).expand(-1, max_points).reshape(-1)
        flat_labels = labels.reshape(-1)
        flat_points = padded.reshape(-1, D)
        flat_mask = mask.reshape(-1)

        # only valid points
        valid = flat_mask
        flat_idx = flat_graph[valid] * max_k + flat_labels[valid]  # unique index per graph+cluster

        new_centroids = new_centroids.reshape(-1, D)
        counts_centroids = counts_centroids.reshape(-1)

        # scatter add
        new_centroids.index_add_(0, flat_idx, flat_points[valid])
        counts_centroids.index_add_(0, flat_idx, torch.ones_like(flat_labels[valid], dtype=torch.float, device=device))

        # reshape back
        new_centroids = new_centroids.reshape(num_graphs, max_k, D)
        counts_centroids = counts_centroids.reshape(num_graphs, max_k)

        centroids = new_centroids / counts_centroids.clamp(min=1).unsqueeze(-1)

    # --- flatten labels to original points ---
    flat_labels_out = labels[batch, idx_in_graph]

    return centroids, flat_labels_out




def cluster_points( pos, batch, type, num_graphs):
    # veh_pos=pos[type==0]
    # veh_batch=batch[type==0]
    # batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)


    device = pos.device
    less_centroids = []
    more_batch = []
    more_centroids = []
    more_type = []

    # veh_mask = (type == 0)
    #
    # veh_number_per_batch = torch.bincount(
    #     batch[veh_mask],
    #     minlength=num_graphs
    # )
    #
    # max_number = max(veh_number_per_batch)  # self.G.steps#

    # step = torch.randint(0, max_number+1, (1,), device=device).item()

    for i in range(num_graphs):
        mask = (batch == i)
        type_i = type[mask]

        x_non_veh = pos[mask][type_i != 0]

        x = pos[mask][type_i == 0]

        type_non_veh = type_i[type_i != 0]

        N = x.shape[0]
        step = torch.randint(0, N + 1, (1,), device=device).item()
        k = min(step, N)

        if k == 0:
            centroids = x[:k]
        else:
            centroids = kmeans_fast(x, k)#x[:k]#

        k1 = min(k + 1, N)  # torch.randint(k+1, N+1, (1,), device=device).item()

        if k1 == 0:
            centroids1 = x[:k1]
        else:
            centroids1 =kmeans_fast(x, k1)# x[:k1] #

        # import matplotlib.pylab as plt
        #
        # plt.scatter(centroids[:,0].cpu().numpy(), centroids[:,1].cpu().numpy(),s=30, c='r')
        #
        # plt.scatter(centroids1[:,0].cpu().numpy(), centroids1[:,1].cpu().numpy(),s=20, c='b')
        #
        # plt.scatter(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(),s=10,c='g')
        #
        # plt.show()

        # less_batch.append(torch.zeros([k+len(x_non_veh)],device=device,dtype=torch.long)+i)

        padding_centers = torch.zeros_like(centroids1[k:])

        less_centroids.append(torch.cat([centroids, padding_centers, x_non_veh], dim=0))

        # less_type.append(torch.cat([torch.zeros([k],device=device,dtype=torch.long),type_non_veh],dim=0))

        more_batch.append(torch.zeros([k1 + len(x_non_veh)], device=device, dtype=torch.long) + i)

        more_centroids.append(torch.cat([centroids1, x_non_veh], dim=0))

        more_type.append(torch.cat([torch.zeros([k1], device=device, dtype=torch.long), type_non_veh], dim=0))

    # less_batch = torch.cat(less_batch, dim=0)
    less_centroids = torch.cat(less_centroids, dim=0)
    # less_type=torch.cat(less_type, dim=0)

    more_batch = torch.cat(more_batch, dim=0)
    more_centroids = torch.cat(more_centroids, dim=0)
    more_type = torch.cat(more_type, dim=0)

    return less_centroids, more_batch, more_centroids, more_type, step