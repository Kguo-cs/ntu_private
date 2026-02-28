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


def cluster_points( pos, batch, type, num_graphs):
    device = pos.device
    # less_batch= []
    less_centroids = []
    # less_type =[]
    more_batch = []
    more_centroids = []
    more_type = []

    veh_mask = (type == 0)

    veh_number_per_batch = torch.bincount(
        batch[veh_mask],
        minlength=num_graphs
    )

    max_number = max(veh_number_per_batch)  # self.G.steps#

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
        # k = torch.randint(0, N+1, (1,), device=device).item()
        k1 = min(k + 1, N)  # torch.randint(k+1, N+1, (1,), device=device).item()

        if k == 0:
            centroids = x[:k]
        else:
            centroids = kmeans_fast(x, k)

        if k1 == 0:
            centroids1 = x[:k1]
        else:
            centroids1 = kmeans_fast(x, k1)

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

        padding_centers = torch.zeros_like(centroids1)[k:]

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