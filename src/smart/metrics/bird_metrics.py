import torch
import numpy as np

# "linear_speed_likelihood",
# "linear_acceleration_likelihood",
# "angular_speed_likelihood",
# "angular_acceleration_likelihood",
# "distance_to_nearest_object_likelihood",
# "time_to_collision_likelihood"
# collision_indication_likelihood

def _wrap_angle(angle):
  """Wraps angles in the range [-pi, pi]."""
  return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_kinematic_features(traj,fps=29.97):
    velocity = (traj[:,:, 1:] - traj[:, :, :-1]) * fps

    acceleration = (velocity[:, :, 1:] - velocity[:, :, :-1]) * fps

    speed = torch.linalg.norm(velocity,dim=-1)

    acc=torch.linalg.norm(acceleration,dim=-1)

    heading =torch.arctan2(velocity[:,:,1], velocity[:,:,0])

    heading_diff=heading[:,:,1:] - heading[:,:,:-1]

    dh_step = _wrap_angle(heading_diff * 2) / 2
    angular_speed = dh_step  * fps

    angular_speed=angular_speed[:,:,1:]-angular_speed[:,:,:-1]

    d2h_step = _wrap_angle(angular_speed * 2) / 2
    angular_acceleration = d2h_step  * fps

    return speed, acc,angular_speed,angular_acceleration

import torch

def histogram_estimate_torch( log_samples: torch.Tensor, sim_samples: torch.Tensor,
                              min_val,max_val,
                              num_bins=11,
                              additive_smoothing_pseudocount=0.1
                              ) -> torch.Tensor:
    """
    PyTorch equivalent of histogram_estimate() from TFP.

    Args:
        config: object with attributes min_val, max_val, num_bins, additive_smoothing_pseudocount
        log_samples: (batch_size, log_sample_size)
        sim_samples: (batch_size, sim_sample_size)
    Returns:
        log_likelihoods: (batch_size, log_sample_size)
    """
    assert log_samples.shape[0] == sim_samples.shape[0], "batch_size must match"
    batch_size = log_samples.shape[0]
    _, S = sim_samples.shape

    # 1. Build histogram edges
    edges = torch.linspace(min_val, max_val, num_bins + 1, device=log_samples.device)

    # 2. Clip samples
    log_samples = torch.clamp(log_samples, min_val, max_val)
    sim_samples = torch.clamp(sim_samples, min_val, max_val)

    # 3. Compute histogram counts for each batch
    sim_flat = sim_samples.reshape(-1)                      # (B*S,)
    # bucketize returns indices in [0, len(edges)], subtract 1 to get 0..num_bins-1
    sim_bin_flat = torch.bucketize(sim_flat, edges) - 1
    sim_bin_flat = sim_bin_flat.clamp(0, num_bins - 1).to(torch.long)  # (B*S,)

    # batch indices for each flattened sample
    batch_idx = torch.arange(batch_size, device=log_samples.device).unsqueeze(1).expand(batch_size, S).reshape(-1).to(torch.long)  # (B*S,)

    # create flattened histogram accumulator of length B * num_bins
    total_bins = batch_size * num_bins
    sim_counts_flat = torch.zeros(total_bins, device=log_samples.device, dtype=log_samples.dtype)

    # compute destination indices in the flattened histogram: batch_idx * num_bins + bin_idx
    dest_idx = batch_idx * num_bins + sim_bin_flat   # (B*S,)
    ones = torch.ones_like(sim_bin_flat, dtype=log_samples.dtype)

    # accumulate counts
    sim_counts_flat = sim_counts_flat.scatter_add_(0, dest_idx, ones)

    # reshape to (B, num_bins)
    sim_counts = sim_counts_flat.reshape(batch_size, num_bins)

    # 4. Add pseudocounts and normalize to probabilities
    sim_counts = sim_counts + additive_smoothing_pseudocount
    probs = sim_counts / sim_counts.sum(dim=1, keepdim=True)  # (batch_size, num_bins)
    log_probs = torch.log(probs + 1e-12)

    # 5. For each log sample, find which bin it belongs to
    bin_indices = torch.bucketize(log_samples, edges) - 1  # (batch_size, log_sample_size)
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # 6. Gather log-likelihood for each sample based on its bin
    log_likelihood = torch.gather(log_probs, 1, bin_indices)  # (batch_size, log_sample_size)

    return log_likelihood



def compute_bird_metrics(pred_traj,gt_traj,gt_heading,gt_mask,fps=29.97):

    pred_mask=(pred_traj==0).all(-1)


    speed,acc,ang_speed,ang_acc=compute_kinematic_features(pred_traj,fps=fps)

    gt_speed,gt_acc,gt_ang_speed,gt_ang_acc=compute_kinematic_features(gt_traj[:,None],fps=fps)

    pred_speed_mask=pred_mask[:,:,1:] & pred_mask[:,:,:-1]
    gt_speed_mask=gt_mask[:,1:] & gt_mask[:,:-1]
    gt_speed_mask=gt_speed_mask[:,None]

    speed[~pred_speed_mask]=torch.nan
    gt_speed[~gt_speed_mask]=torch.nan

    linear_speed_log_likelihood= histogram_estimate_torch(speed.flatten(1,2),gt_speed.flatten(1,2),min_val=0,max_val=25)

    cond_sum = linear_speed_log_likelihood*gt_speed_mask
    valid_sum = gt_speed_mask.sum()
    linear_speed_log_likelihood_sum= cond_sum / valid_sum

    return 1

