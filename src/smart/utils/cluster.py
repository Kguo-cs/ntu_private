import torch

def kmeans( padded, mask,k_per_graph,batch,pos,valid_mask,max_k, initial_centroids=None,iters=10):

    padded=padded[:,:,:2]

    num_graphs,max_points, D=padded.shape

    device=padded.device

    cluster_mask = (
            torch.arange(max_k, device=device)
            .unsqueeze(0)
            < k_per_graph.unsqueeze(1)
    )  # (num_graphs, max_k)

    if initial_centroids is None:
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

    else:
        dist = (
            padded.pow(2).sum(-1, keepdim=True)
            - 2 * padded @ initial_centroids.transpose(1, 2)
            + initial_centroids.pow(2).sum(-1).unsqueeze(1)
        ).swapaxes(1,2)

        initial_cluster_mask=torch.any(initial_centroids,dim=-1)

        dist[~initial_cluster_mask]=float('inf')

        min_dist=dist.amin(dim=1)

        min_dist[~mask]=-1

        max_dist_idx=torch.argsort(min_dist,dim=-1,descending=True)

        fill_mask = cluster_mask & (~initial_cluster_mask)

        # sort points by descending min distance
        sorted_points = padded.gather(
            1,
            max_dist_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        # per-graph rank for fill positions
        fill_rank = (torch.cumsum(fill_mask.float(), dim=1) - 1).long()

        # get replacement points
        replacement_points = sorted_points[
            torch.arange(num_graphs, device=device).unsqueeze(1),
            fill_rank.clamp(min=0)
        ]

        # update centroids
        centroids = initial_centroids.clone()
        centroids[fill_mask] = replacement_points[fill_mask]

    centroids_mask=~(cluster_mask[:,None] & mask[:,:,None])[mask]

    # --- K-Means iterations ---
    for i in range(iters):
        # distances: (num_graphs, max_points, max_k)
        # dist = (
        #     padded.pow(2).sum(-1, keepdim=True)
        #     - 2 * padded @ centroids.transpose(1, 2)
        #     + centroids.pow(2).sum(-1).unsqueeze(1)
        # )
        if i==iters-1:
            D=pos.shape[-1]

        dist=torch.cdist(padded,centroids)
        dist=dist[mask]

        dist[centroids_mask]=float('inf')

        # assign labels per point (capped at graph's k)
        labels_per_point = dist.argmin(dim=-1)  # (valid_point)

        # total clusters across all graphs
        total_clusters = num_graphs * max_k

        new_centroids = torch.zeros(total_clusters, D, device=device)

        if valid_mask is None:
            counts_centroids = torch.zeros(total_clusters, device=device)
        else:
            counts_centroids = torch.zeros(total_clusters, D, device=device)

        # global cluster index per point
        global_idx = batch * max_k + labels_per_point   # (N_total,)

        # scatter sums
        new_centroids.index_add_(0, global_idx, pos[:,:D])

        if valid_mask is None:
            counts_centroids.index_add_(0, global_idx, torch.ones_like(global_idx, dtype=torch.float))
        else:
            counts_centroids.index_add_(0, global_idx, valid_mask[:,:D])

        # reshape back
        new_centroids = new_centroids.view(num_graphs, max_k, D)

        if valid_mask is None:
            counts_centroids = counts_centroids.view(num_graphs, max_k,1)
        else:
            counts_centroids = counts_centroids.view(num_graphs, max_k,D)

        centroids = new_centroids / counts_centroids.clamp(min=1)

    centroids[~cluster_mask]=torch.nan

    if valid_mask is not None:

        centroids[counts_centroids==0]=torch.nan

    return centroids

def batched_kmeans_variable_k(pos, batch,valid_mask, num_graphs,k_per_graph=None,k1_per_graph=None ):
    device = pos.device
    D = pos.shape[1]

    counts = torch.bincount(batch, minlength=num_graphs)
    offsets = torch.cumsum(counts, dim=0)
    offsets = torch.cat([torch.tensor([0], device=batch.device), offsets[:-1]])

    idx_in_graph = torch.arange(len(batch), device=batch.device) - offsets[batch]

    max_points = counts.max()

    padded = torch.zeros(num_graphs, max_points, D, device=pos.device)+torch.nan
    mask = torch.zeros(num_graphs, max_points, dtype=torch.bool, device=pos.device)

    padded[batch, idx_in_graph] = pos
    mask[batch, idx_in_graph] = True
    
    if k_per_graph is None:

        schedules = batch_increasing_schedule(counts)
    
        step_number=schedules.shape[1]-1
    
        step_idx = torch.randint(0, step_number, (num_graphs,), device=counts.device)

        batch_idx = torch.arange(num_graphs, device=counts.device)
    
        k_per_graph = schedules[batch_idx, step_idx]
        k1_per_graph = schedules[batch_idx, step_idx + 1]
    else:
        step_idx=step_number=None
        
    max_k = k1_per_graph.max().item()

    new_k=k_per_graph!=k1_per_graph

    same_batch_mask = new_k[batch]

    selected_batch = batch[same_batch_mask]

    _, consecutive_batch = torch.unique(selected_batch, return_inverse=True)

    padded=torch.cat([padded,padded[new_k]])
    mask=torch.cat([mask,mask[new_k]])
    k_per_graph_all=torch.cat([k_per_graph,k1_per_graph[new_k]])

    batch=torch.cat([batch,consecutive_batch+num_graphs])
    pos=torch.cat([pos,pos[same_batch_mask]])
    if valid_mask is not None:
        valid_mask=torch.cat([valid_mask,valid_mask[same_batch_mask]])

    centroids_all=kmeans(padded, mask,k_per_graph_all,batch,pos,valid_mask,max_k)

    centroids=centroids_all[:num_graphs]

    centroids1 = centroids.clone()

    centroids1[new_k]=centroids_all[num_graphs:]

    #centroids1[new_k]=kmeans(padded[new_k], mask[new_k],k1_per_graph[new_k],consecutive_batch,pos[same_batch_mask],max_k)
#, initial_centroids=centroids[new_k]
    # centroids1 = kmeans(padded, mask, k1_per_graph,batch, pos,max_k)

    return centroids, centroids1, k_per_graph, k1_per_graph, step_idx, step_number

def allocate_k_per_type(k_total, type_counts):
    """
    k_total:     (G,)
    type_counts: (G, T)

    returns:
        k_type:   (G, T)
    """

    G, T = type_counts.shape
    Kmax = k_total.max()

    # divisors: 1..Kmax
    divisors = torch.arange(1, Kmax + 1, device=type_counts.device)

    # compute priority table
    # shape: (G, T, Kmax)
    priorities = type_counts.unsqueeze(-1) / divisors

    # flatten type/divisor axis
    priorities = priorities.reshape(G, T * Kmax)

    # select top-k priorities per graph
    topk_vals, topk_idx = torch.topk(priorities, Kmax, dim=1)

    # mask out positions beyond k_total
    mask = torch.arange(Kmax, device=type_counts.device).unsqueeze(0) < k_total.unsqueeze(1)

    selected = topk_idx * mask

    # recover type index
    type_idx = selected // Kmax

    # count allocations
    k_type = torch.zeros(G, T, device=type_counts.device, dtype=torch.long)

    ones = mask.long()

    k_type.scatter_add_(1, type_idx, ones)

    return k_type

def cluster_point_per_type(
    pos,
    batch,
    tokenized_agent
):
    type = tokenized_agent["nonego_type_sorted"]

    type_counts = tokenized_agent["type_counts"]

    valid_mask=tokenized_agent["nonego_valid"]

    num_graphs,num_types=type_counts.shape

    counts = type_counts.sum(-1)

    schedules = batch_increasing_schedule(counts)

    step_number=schedules.shape[1]-1

    step_idx = torch.randint(0, step_number, (num_graphs,), device=counts.device)

    step_idx[0]=0

    step1_idx = step_idx + 1

    batch_idx = torch.arange(num_graphs, device=counts.device)

    k_per_graph = schedules[batch_idx, step_idx]
    k1_per_graph = schedules[batch_idx, step1_idx]

    centroids_list=[]
    centroids1_list=[]
    type_list=[]

    k_type  = allocate_k_per_type(k_per_graph,  type_counts)
    k1_type = allocate_k_per_type(k1_per_graph, type_counts)

    #print(torch.all(k1_type>=k_type),torch.all(k_type.sum(dim=-1) == k_per_graph))

    for i in range( num_types ):
        k_per_graph_type=k_type[:,i]
        k1_per_graph_type=k1_type[:,i]

        veh_pos=pos[type==i]
        veh_batch=batch[type==i]

        if valid_mask is not None:
            veh_valid_mask=valid_mask[type==i]
        else:
            veh_valid_mask=None

        if k1_per_graph_type.max().item()>0:
            centroids_b,centroids1_b,_,_,_,_=batched_kmeans_variable_k(veh_pos, veh_batch,veh_valid_mask,num_graphs,k_per_graph_type,k1_per_graph_type)

            centroids_list.append(centroids_b)
            centroids1_list.append(centroids1_b)
            type_list.append(torch.zeros_like(centroids1_b[:,:,0])+i)

    centroids_all = torch.cat(centroids_list, dim=1)
    centroids1_all = torch.cat(centroids1_list, dim=1)
    type_list = torch.cat(type_list, dim=1).to(torch.long)

    valid_mask=torch.any(~torch.isnan(centroids1_all),dim=-1)

    less_centroids=centroids_all[valid_mask]
    more_centroids=centroids1_all[valid_mask]
    more_batch=torch.arange(num_graphs, device=counts.device)[:,None].repeat(1,centroids1_all.shape[1])[valid_mask]
    more_type=type_list[valid_mask]

    tokenized_agent['nonego_type_sorted'] = more_type
    tokenized_agent["step_idx"] = step_idx
    tokenized_agent["step_number"] = step_number#7057 ,7056

    tokenized_agent["clustering"]=k_per_graph!=counts

    return less_centroids,more_centroids,more_batch

def build_less_more_grouped1(
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

    centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx,step_number=batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)

    nonveh_mask = ~veh_mask

    nonveh_pos = pos[nonveh_mask]
    nonveh_batch = batch[nonveh_mask]
    nonveh_type = type[nonveh_mask]

    # Count non-veh per graph
    nonveh_count = torch.bincount(
        nonveh_batch,
        minlength=G
    )

    more_sizes = k1_per_graph + nonveh_count

    more_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        torch.cumsum(more_sizes, dim=0)
    ])[:-1]

    total_more = more_sizes.sum()

    # Allocate final tensors
    less_centroids = torch.zeros(total_more, D, device=device)
    more_centroids = torch.zeros(total_more, D, device=device)
    more_batch = torch.zeros(total_more, device=device, dtype=torch.long)
    more_type = torch.zeros(total_more, device=device, dtype=torch.long)

    max_k1 = centroids1_b.shape[1]

    arange_k1 = torch.arange(max_k1, device=device)

    valid_k1 = arange_k1.unsqueeze(0) < k1_per_graph.unsqueeze(1)

    centroids_less_flat = centroids_b[valid_k1]
    centroids_more_flat = centroids1_b[valid_k1]

    graph_ids_k1 = torch.arange(G, device=device).repeat_interleave(k1_per_graph)


    more_write_idx = more_offsets[graph_ids_k1] + \
                     torch.arange(centroids_more_flat.shape[0], device=device) \
                     - torch.repeat_interleave(
                         torch.cumsum(k1_per_graph, 0) - k1_per_graph,
                         k1_per_graph
                     )

    less_centroids[more_write_idx] = centroids_less_flat
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

    more_nonveh_idx = (
        more_offsets[nonveh_batch]
        + k1_per_graph[nonveh_batch]
        + nonveh_local_idx
    )

    less_centroids[more_nonveh_idx] = nonveh_pos
    more_centroids[more_nonveh_idx] = nonveh_pos
    more_batch[more_nonveh_idx] = nonveh_batch
    more_type[more_nonveh_idx] = nonveh_type

    return (
        less_centroids,
        more_centroids,
        more_batch,
        more_type,centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx,step_number
    )


def cluster_points1(pos, batch, type, num_graphs  ):

    # type_vector=torch.zeros([len(type),3],device=pos.device)
    #
    # type_vector[torch.arange(len(type)),type]=1
    #
    # pos=torch.cat([pos,type_vector],dim=-1)

    centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx,step_number=batched_kmeans_variable_k(pos, batch,num_graphs)

    max_k = k1_per_graph.max().item()

    cluster_mask = (
            torch.arange(max_k, device=pos.device)
            .unsqueeze(0)
            < k1_per_graph.unsqueeze(1)
    )  # (num_graphs, max_k)

    less_centroids=centroids_b[cluster_mask]
    more_centroids=centroids1_b[cluster_mask]

    more_batch=torch.where(cluster_mask)[0]

    more_type=centroids1_b[:,8:]

    return less_centroids, more_batch, more_centroids, more_type,step_idx,step_number

def cluster_points(pos, batch, type, type_counts,use_all_type
        ):
    if use_all_type:
        return cluster_points1(pos, batch, type, len(type_counts))

    # veh_pos=pos[type==0]
    # veh_batch=batch[type==0]
    #
    # centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx,step_number=batched_kmeans_variable_k(veh_pos, veh_batch,num_graphs)
    #
    #
    # device = pos.device
    # less_centroids = []
    # more_batch = []
    # more_centroids = []
    # more_type = []
    #
    # for i in range(num_graphs):
    #     mask = (batch == i)
    #     type_i = type[mask]
    #
    #     x_non_veh = pos[mask][type_i != 0]
    #
    #     type_non_veh = type_i[type_i != 0]
    #
    #     x = pos[mask][type_i == 0]
    #     N = x.shape[0]
    #     step = torch.randint(0, N + 1, (1,), device=device).item()
    #     k = min(step, N)
    #
    #     # if k == 0:
    #     #     centroids = x[:k]
    #     # else:
    #     #     centroids = kmeans_fast(x, k)#x[:k]#
    #     #
    #     # k1 = min(k + 1, N)  # torch.randint(k+1, N+1, (1,), device=device).item()
    #     #
    #     # if k1 == 0:
    #     #     centroids1 = x[:k1]
    #     # else:
    #     #     centroids1 =kmeans_fast(x, k1)# x[:k1] #
    #
    #     k=k_per_graph[i]
    #     k1=k1_per_graph[i]
    #     centroids=centroids_b[i][:k]
    #     centroids1=centroids1_b[i][:k1]
    #
    #
    #     padding_centers = torch.zeros_like(centroids1[k:])
    #
    #     less_centroids.append(torch.cat([centroids, padding_centers, x_non_veh], dim=0))
    #
    #     more_batch.append(torch.zeros([k1 + len(x_non_veh)], device=device, dtype=torch.long) + i)
    #
    #     more_centroids.append(torch.cat([centroids1, x_non_veh], dim=0))
    #
    #     more_type.append(torch.cat([torch.zeros([k1], device=device, dtype=torch.long), type_non_veh], dim=0))
    #
    # less_centroids = torch.cat(less_centroids, dim=0)
    # more_batch = torch.cat(more_batch, dim=0)
    # more_centroids = torch.cat(more_centroids, dim=0)
    # more_type = torch.cat(more_type, dim=0)
    #
    # if not (torch.all(less_centroids == less_centroids1) & torch.all(more_centroids == more_centroids1) & torch.all(more_batch == more_batch1) & torch.all(more_type == more_type1)):
    #
    #     print(torch.all(less_centroids == less_centroids1))
    #     print(torch.all(more_centroids == more_centroids1))
    #     print(torch.all(more_batch == more_batch1))
    #     print(torch.all(more_type == more_type1))
    less_centroids,more_centroids,more_batch,more_type,centroids_b,centroids1_b, k_per_graph,k1_per_graph,step_idx,step_number=build_less_more_grouped(
        pos,
        type,
        batch,
        type_counts
        )

    return less_centroids, more_batch, more_centroids, more_type,step_idx,step_number


# import matplotlib.pylab as plt
#
# plt.scatter(centroids[:,0].cpu().numpy(), centroids[:,1].cpu().numpy(),s=30, c='r')
#
# plt.scatter(centroids1[:,0].cpu().numpy(), centroids1[:,1].cpu().numpy(),s=20, c='b')
#
# plt.scatter(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(),s=10,c='g')
#
# plt.show()


def batch_increasing_schedule(count, S=50+1, gamma=1):
    """
    N: (B,) tensor of maximum levels per batch
    S: total number of steps (int)
    gamma: curvature (>1 gives more steps to small values)

    Returns:
        schedule: (B, S) integer tensor
    """
    s = torch.arange(S, device=count.device) # (S,)
    # ratio = (s / S).pow(gamma)  # (S,)
    #
    # schedule = torch.ceil_(count[:, None] * ratio[None, :])
    #schedule = torch.minimum(schedule, count[:, None])

    schedule = (s/S*count[:,None]).to(torch.long)

    schedule1=torch.maximum(s[None],schedule)

    schedule2=torch.minimum(count[:,None],schedule1)


    schedule3 =torch.cat([schedule2,count[:,None].repeat(1,100)],dim=-1)

    return schedule3