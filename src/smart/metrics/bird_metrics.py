import torch
import numpy as np
from torch_geometric.nn.conv.x_conv import knn_graph


def emd_1d(p: torch.Tensor,
           q: torch.Tensor,
           bin_locations: torch.Tensor = None,
           eps: float = 1e-8,
           reduction: str = 'none') -> torch.Tensor:
    """
    Compute 1D Earth Mover's Distance (Wasserstein-1) between two discrete distributions.
    Supports batched inputs and is differentiable (works as a loss).

    Args:
        p: Tensor of shape (..., n) - non-negative counts or probabilities.
        q: Tensor of shape (..., n) - same shape as p.
        bin_locations: optional 1D tensor of length n giving the positions of the bins (must be sorted).
                       If None, unit spacing is assumed and distances between adjacent bins are 1.
        eps: small constant to avoid division by zero when normalizing counts.
        reduction: 'none' (default) returns per-sample distances, 'mean' returns mean, 'sum' returns sum.

    Returns:
        Tensor of shape (...) with the Wasserstein-1 distances (reduced according to `reduction`).
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    # cast to float and keep device
    device = p.device
    p = p.to(dtype=torch.float32)
    q = q.to(dtype=torch.float32)

    # flatten leading dims into batch
    orig_shape = p.shape[:-1]
    n = p.shape[-1]
    p_flat = p.reshape(-1, n)
    q_flat = q.reshape(-1, n)

    # Normalize to probabilities (safe with eps)
    p_sum = p_flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    q_sum = q_flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    p_prob = p_flat / p_sum
    q_prob = q_flat / q_sum

    # CDFs (works on GPU)
    cdf_p = torch.cumsum(p_prob, dim=-1)
    cdf_q = torch.cumsum(q_prob, dim=-1)

    if n == 1:
        distances = torch.zeros(p_flat.shape[0], dtype=torch.float32, device=device)
    else:
        # difference of CDFs up to n-1 (last CDF is 1 if normalized)
        cdf_diff = torch.abs(cdf_p[:, :-1] - cdf_q[:, :-1])  # shape (batch, n-1)

        if bin_locations is None:
            # unit spacing
            deltas = torch.ones(cdf_diff.shape[-1], dtype=torch.float32, device=device)
        else:
            bin_locations = torch.as_tensor(bin_locations, dtype=torch.float32, device=device)
            if bin_locations.numel() != n:
                raise ValueError("bin_locations must have length n (number of bins)")
            # ensure monotonic: we don't enforce sorting but assume caller provides sorted bins
            deltas = torch.diff(bin_locations)  # length n-1

        # multiply CDF differences by distances and sum
        distances = (cdf_diff * deltas).sum(dim=-1)

    distances = distances.reshape(orig_shape)

    return distances.mean()


def _wrap_angle(angle):
  """Wraps angles in the range [-pi, pi]."""
  return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_kinematic_features(traj,fps=29.97):
    velocity = (traj[:,:, 2:] - traj[:, :, :-2])/2 * fps

    speed = torch.linalg.norm(velocity,dim=-1)

    acc = (speed[:, :, 2:] - speed[:, :, :-2])/2 * fps

    return speed, acc

def compute_kinematic_features1(traj,fps=29.97):
    velocity = traj[:,:, 2:] - traj[:, :, :-2]

    heading =torch.arctan2(velocity[:,:,:,1], velocity[:,:,:,0])

    heading_diff=(heading[:,:,2:] - heading[:,:,:-2])/2

    dh_step = _wrap_angle(heading_diff * 2) / 2
    angular_speed = dh_step  * fps

    angular_speed_diff=(dh_step[:,:,2:]-dh_step[:,:,:-2])/2

    d2h_step = _wrap_angle(angular_speed_diff * 2) / 2
    angular_acceleration = d2h_step  * fps*fps

    return angular_speed,angular_acceleration


from torch_scatter import scatter_mean,scatter_sum
import torch

def histogram_estimate_torch(
        batch,
        log_samples: torch.Tensor,        # (B, L)
        sim_samples: torch.Tensor,        # (B, S)
        min_val: float,
        max_val: float,
        num_bins: int = 11,
        additive_smoothing_pseudocount: float = 0.1,
        gt_valid_mask: torch.Tensor = None,   # (B, L) bool/byte/0-1
        sim_valid_mask: torch.Tensor = None   # (B, S) bool/byte/0-1
    ) -> torch.Tensor:
    """
    Vectorized histogram-based likelihood estimator.

    Returns:
      likelihoods: (B,) tensor where likelihoods[b] = exp(sum_{valid i} log p_bin(log_samples[b,i]))
                   If a batch has zero valid gt samples, its likelihood is set to 0.0.
    """
    assert log_samples.dim() == 2 and sim_samples.dim() == 2
    B, L = log_samples.shape
    _, S = sim_samples.shape
    device = log_samples.device
    dtype = log_samples.dtype
    eps = 1e-12

    # Default masks: all True (all samples valid)
    if gt_valid_mask is None:
        gt_valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    else:
        gt_valid_mask = gt_valid_mask.to(torch.bool).to(device)

    if sim_valid_mask is None:
        sim_valid_mask = torch.ones((B, S), dtype=torch.bool, device=device)
    else:
        sim_valid_mask = sim_valid_mask.to(torch.bool).to(device)

    # Clip samples to edges
    log_clamped = torch.clamp(log_samples, min_val, max_val)
    sim_clamped = torch.clamp(sim_samples, min_val, max_val)

    # 2) linear binning (vectorized arithmetic, no bucketize)
    # Map value -> [0, num_bins). We treat last bin as inclusive.
    span = float(max_val) - float(min_val)
    if span <= 0.0:
        raise ValueError("max_val must be > min_val")

    # flatten sim samples and masks
    sim_flat = sim_clamped.reshape(-1)                  # (B*S,)
    sim_valid_flat = sim_valid_mask.reshape(-1)         # (B*S,)
    # normalized fraction in [0,1]
    frac = (sim_flat[sim_valid_flat] - min_val) / span
    # scale to [0, num_bins), use floor, then clamp
    bin_idx_flat = torch.floor(frac * num_bins).to(torch.int32)
    # values exactly == max_val will give index == num_bins, clamp to num_bins-1
    bin_idx_flat = torch.clamp(bin_idx_flat, 0, num_bins - 1)

    # batch indices
    batch_idx = (
        torch.arange(B, device=device, dtype=torch.int32)
        .unsqueeze(1).expand(B, S)
        .reshape(-1)
    ) [sim_valid_flat] # (B*S,)

    # keep only valid sim entries
    valid_mask = sim_valid_flat
    if valid_mask.any():
        dest = batch_idx* num_bins + bin_idx_flat
        # bincount: produce counts for length B * num_bins
        total_bins = B * num_bins
        counts_flat = torch.bincount(dest, minlength=total_bins).to(torch.float32)
        sim_counts = counts_flat.view(B, num_bins)
    else:
        sim_counts = torch.zeros((B, num_bins), device=device, dtype=torch.float32)

    # ---- Add pseudocounts and convert to log-probabilities ----
    # 3) add pseudocount and normalize
    sim_counts_sudo = sim_counts + float(additive_smoothing_pseudocount)
    sim_probs = sim_counts_sudo / (sim_counts_sudo.sum(dim=1, keepdim=True) + eps)  # (B, num_bins)
    log_sim_probs = torch.log(sim_probs.clamp_min(eps))

    # 4) bin log_samples using the same linear mapping
    log_flat = log_clamped.reshape(-1)
    frac_log = (log_flat - min_val) / span
    log_bin_flat = torch.floor(frac_log * num_bins).to(torch.long)
    log_bin_flat = torch.clamp(log_bin_flat, 0, num_bins - 1)
    log_bins = log_bin_flat.view(B, L)  # (B, L)

    # Gather log-prob per sample
    per_sample_logprob = torch.gather(log_sim_probs, 1, log_bins)   # (B, L)


    # Zero-out contributions from invalid gt samples (ignore them in sum)
    mask_float = gt_valid_mask.to(dtype)
    per_sample_logprob_masked = per_sample_logprob * mask_float    # (B, L)

    # Sum log-probs over valid gt samples
    sum_logprob = per_sample_logprob_masked.sum(dim=1) /mask_float.sum(-1)            # (B,)

    # If a batch has zero valid samples, set likelihood to 0.0 (no information)
    # Otherwise, likelihood = exp(sum_logprob)
    likelihoods = torch.exp(sum_logprob).to(dtype)


    # Mask out batches with zero valid gt samples:
    # masked_lihood=likelihoods[gt_valid_mask.any(-1)]
    agent_likelihood=scatter_mean(likelihoods[gt_valid_mask.any(-1)], batch[gt_valid_mask.any(-1)])

    batch_sim_counts=scatter_sum(sim_counts,batch, dim=0)
    batch_sim_probs = batch_sim_counts / (batch_sim_counts.sum(dim=1, keepdim=True) + eps)  # (B, num_bins)
    batch_log_sim_probs = torch.log(batch_sim_probs.clamp_min(eps))

    num_batches = torch.amax(batch).item() + 1
    num_bins = num_bins  # as before

    batch_idx = batch[:, None].repeat(1, gt_valid_mask.shape[1])[gt_valid_mask].to(device)  # (K,)
    bin_idx = log_bins[gt_valid_mask].to(device)  # (K,)

    linear_idx = batch_idx * num_bins + bin_idx  # shape (K,)
    counts_flat = torch.bincount(linear_idx, minlength=(num_batches * num_bins)).to(dtype=torch.long)

    batch_log_bin = counts_flat.reshape(num_batches, num_bins)

    earth_mover_dist=emd_1d(batch_sim_counts,batch_log_bin)

    batch_sum_logprob=(batch_log_sim_probs*batch_log_bin).sum(-1)/batch_log_bin.sum(-1)

    scene_likelihoods = torch.exp(batch_sum_logprob).to(dtype)

    return agent_likelihood,scene_likelihoods.mean(),earth_mover_dist

    # min_val=torch.quantile(valid_gt_speed,0.01)
    # max_val=torch.quantile(valid_gt_speed,0.99)
    # print(min_val,max_val)

def plot_histgram(name, valid_gt_speed, valid_pred_speed,
                  min_val, max_val, num_bins=11, save_dir="/home/ke/code/catk/src/waymo_data/bird_data1/result"):
    import torch
    import matplotlib.pyplot as plt
    import os
    import matplotlib as mpl
    valid_gt_speed=valid_gt_speed.to(torch.float32)


    print(torch.quantile(valid_gt_speed, 0.01),torch.quantile(valid_gt_speed, 0.99))

    mpl.rcParams['toolbar'] = 'None'

    os.makedirs(save_dir, exist_ok=True)

    valid_gt_speed = valid_gt_speed.to(torch.float32)
    valid_pred_speed = valid_pred_speed.to(torch.float32)

    # Clamp to valid range
    valid_gt_speed = torch.clamp(valid_gt_speed, min_val, max_val)
    valid_pred_speed = torch.clamp(valid_pred_speed, min_val, max_val)

    # Compute histograms
    hist_gt = torch.histc(valid_gt_speed, bins=num_bins, min=min_val, max=max_val)
    hist_pred = torch.histc(valid_pred_speed, bins=num_bins, min=min_val, max=max_val)

    hist_gt=hist_gt/hist_gt.sum()
    hist_pred=hist_pred/hist_pred.sum()

    # Bin edges and width
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
    width = (max_val - min_val) / num_bins

    # Plot both histograms together
    plt.figure(figsize=(7, 5))
    plt.bar(bin_edges[:-1].cpu().numpy(), hist_gt.cpu().numpy(),
            width=width, align='edge',
            color='blue', alpha=0.6, label='GT Speed', edgecolor='black')
    plt.bar(bin_edges[:-1].cpu().numpy(), hist_pred.cpu().numpy(),
            width=width, align='edge',
            color='green', alpha=0.5, label='Pred Speed', edgecolor='black')

    plt.title(name+" Distribution Comparison")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # plt.show()
    save_path = os.path.join(save_dir, f"{name}_hist.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram: {save_path}")

def compute_interactive_metric(pred_traj,batch,pred_mask):


    for i in range(torch.amax(batch).item()+1):
        batch_pred_traj=pred_traj[batch==i].flatten(1,2).transpose(0,1)[::10].to(torch.float32)

        dist=torch.cdist(batch_pred_traj,batch_pred_traj)

        dist[dist==0]=10000

        min_dist=dist.amin(-1)

        x=batch_pred_traj#@@.transpose(0,1).flatten(0,1)

        T, N, D = batch_pred_traj.shape

        batch_t = torch.arange(T, device=batch_pred_traj.device).repeat_interleave(N)#.reshape(T, N)#.tranpose(0,1).flatten(0,1)
        #batch_t=batch_t.transpose(0,1).flatten(0,1)

        edge_index = knn_graph(x, k=1, batch=batch_t, loop=False)

        src, dst = edge_index
        min_dist1 = torch.linalg.norm(x[src] - x[dst], dim=-1)#.reshape(min_dist.shape)

        print(1)

    return





def compute_bird_metrics(pred_traj,gt_traj,gt_mask,batch,vis=False,fps=29.97):


    pred_mask=(pred_traj!=10000).any(-1)

    #compute_interactive_metric(pred_traj,batch,pred_mask)


    gt_valid_num=scatter_sum(gt_mask.to(torch.int16),batch,dim=0)

    pred_valid_num=scatter_sum(pred_mask.to(torch.int16),batch,dim=0)

    num_diff=(pred_valid_num-gt_valid_num[:,None]).float()

    num_diff_mean=num_diff.mean()
    num_diff_abs=num_diff.abs().mean()

    entry_mask=~gt_mask[:,:-1] & gt_mask[:,1:]

    exit_mask=gt_mask[:,:-1] & ~gt_mask[:,1:]

    pred_entry_mask = ~pred_mask[:, :, :-1] & pred_mask[:, :, 1:]

    pred_exit_mask = pred_mask[:, :, :-1] & ~pred_mask[:, :, 1:]

    gt_entry_num=scatter_sum(entry_mask.to(torch.int16),batch,dim=0)
    pred_entry_num=scatter_sum(pred_entry_mask.to(torch.int16),batch,dim=0)
    gt_exit_num=scatter_sum(exit_mask.to(torch.int16),batch,dim=0)
    pred_exit_num=scatter_sum(pred_exit_mask.to(torch.int16),batch,dim=0)

    num_entry_diff_mean=(pred_entry_num/(pred_valid_num[:, :, :-1]+1) -(gt_entry_num/(gt_valid_num[:,1:]+1))[:,None]).float().mean()

    num_exit_diff_mean=(pred_exit_num/(pred_valid_num[:, :, :-1]+1) -(gt_exit_num/(gt_valid_num[:,:-1]+1))[:,None]).float().mean()

    speed,acc=compute_kinematic_features(pred_traj,fps=fps)

    gt_speed,gt_acc=compute_kinematic_features(gt_traj[:,None],fps=fps)

    pred_speed_mask=pred_mask[:,:,2:] & pred_mask[:,:,:-2]
    gt_speed_mask=gt_mask[:,2:] & gt_mask[:,:-2]

    pred_acc_mask=pred_speed_mask[:,:,2:] & pred_speed_mask[:,:,:-2]
    gt_acc_mask=gt_speed_mask[:,2:] & gt_speed_mask[:,:-2]

   # if vis:
    # valid_gt_speed=gt_speed[:,0][gt_speed_mask]
    # valid_gt_acc=gt_acc[:,0][gt_acc_mask]
    # valid_gt_ang_speed=gt_ang_speed[:,0][gt_acc_mask]
    # valid_gt_ang_acc=gt_ang_acc[:,0][gt_angular_acc_mask]
    #
    # valid_speed = speed[pred_speed_mask]
    # valid_acc = acc[pred_acc_mask]
    # valid_ang_speed = ang_speed[pred_acc_mask]
    # valid_ang_acc =ang_acc[pred_angular_acc_mask]
    #
    # plot_histgram('Speed',valid_gt_speed,valid_speed,min_val=4,max_val=10)
    # plot_histgram('Acc',valid_gt_acc,valid_acc,min_val=-3,max_val=3)
    # plot_histgram('Angular speed',valid_gt_ang_speed,valid_ang_speed,min_val=-1,max_val=1)
    # plot_histgram('Angular acc',valid_gt_ang_acc,valid_ang_acc,min_val=-2,max_val=2)

    linear_speed_likelihoods= histogram_estimate_torch(batch,gt_speed.flatten(1,2),speed.flatten(1,2),min_val=4,max_val=10,
                                                      gt_valid_mask=gt_speed_mask,sim_valid_mask=pred_speed_mask.flatten(1,2),
                                                      )

    linear_acc_likelihoods= histogram_estimate_torch(batch,gt_acc.flatten(1,2),acc.flatten(1,2),min_val=-3.5,max_val=3.5,
                                                      gt_valid_mask=gt_acc_mask,sim_valid_mask=pred_acc_mask.flatten(1,2),
                                                      )

    ang_speed, ang_acc=compute_kinematic_features1(pred_traj,fps=fps)

    gt_ang_speed, gt_ang_acc=compute_kinematic_features1(gt_traj[:,None],fps=fps)

    pred_angular_acc_mask=pred_acc_mask[:,:,2:] & pred_acc_mask[:,:,:-2]
    gt_angular_acc_mask=gt_acc_mask[:,2:] & gt_acc_mask[:,:-2]

    angular_speed_likelihoods=histogram_estimate_torch(batch,gt_ang_speed.flatten(1,2),ang_speed.flatten(1,2),min_val=-1,max_val=1,
                                                      gt_valid_mask=gt_acc_mask,sim_valid_mask=pred_acc_mask.flatten(1,2),
                                                      )

    angular_acceleration_likelihoods=histogram_estimate_torch(batch,gt_ang_acc.flatten(1,2),ang_acc.flatten(1,2),min_val=-2,max_val=2,
                                                      gt_valid_mask=gt_angular_acc_mask,sim_valid_mask=pred_angular_acc_mask.flatten(1,2),
                                                      )


    # exist_likelihood=histogram_estimate_torch(batch,gt_mask.to(torch.float16),pred_mask.flatten(1,2).to(torch.float16),
    #                                                     min_val=-0.5,max_val=1.5,num_bins=2
    #                                                   )
    # exist_likelihood=(gt_mask[:,None]==pred_mask).float().mean(-1).mean(-1)
    #
    # exist_likelihood=scatter_mean(exist_likelihood, batch)


    return linear_speed_likelihoods, linear_acc_likelihoods, angular_speed_likelihoods, angular_acceleration_likelihoods, num_diff_mean, num_diff_abs,num_entry_diff_mean, num_exit_diff_mean

# tensor(3.864, device='cuda:0') tensor(10.114, device='cuda:0')
# Saved histogram: /home/ke/code/catk/src/waymo_data/bird_data1/result/Speed_hist.png
# tensor(-3.408, device='cuda:0') tensor(3.436, device='cuda:0')
# Saved histogram: /home/ke/code/catk/src/waymo_data/bird_data1/result/Acc_hist.png
# tensor(-1.044, device='cuda:0') tensor(1.062, device='cuda:0')
# Saved histogram: /home/ke/code/catk/src/waymo_data/bird_data1/result/Angular speed_hist.png
# tensor(-2.104, device='cuda:0') tensor(2.118, device='cuda:0')
# Saved histogram: /home/ke/code/catk/src/waymo_data/bird_data1/result/Angular acc_hist.png