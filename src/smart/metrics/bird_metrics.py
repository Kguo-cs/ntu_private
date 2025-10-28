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

    speed = torch.linalg.norm(velocity,dim=-1)

    acc = (speed[:, :, 1:] - speed[:, :, :-1]) * fps

    heading =torch.arctan2(velocity[:,:,:,1], velocity[:,:,:,0])

    heading_diff=heading[:,:,1:] - heading[:,:,:-1]

    dh_step = _wrap_angle(heading_diff * 2) / 2
    angular_speed = dh_step  * fps

    angular_speed_diff=angular_speed[:,:,1:]-angular_speed[:,:,:-1]

    d2h_step = _wrap_angle(angular_speed_diff * 2) / 2
    angular_acceleration = d2h_step  * fps

    return speed, acc,angular_speed,angular_acceleration

from torch_scatter import scatter_mean
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
        counts_flat = torch.bincount(dest, minlength=total_bins).to(dtype)
        sim_counts = counts_flat.view(B, num_bins)
    else:
        sim_counts = torch.zeros((B, num_bins), device=device, dtype=dtype)

    # ---- Add pseudocounts and convert to log-probabilities ----
    # 3) add pseudocount and normalize
    sim_counts = sim_counts + float(additive_smoothing_pseudocount)
    sim_probs = sim_counts / (sim_counts.sum(dim=1, keepdim=True) + eps)  # (B, num_bins)
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
    masked_lihood=likelihoods[gt_valid_mask.any(-1)]

    batch=batch[gt_valid_mask.any(-1)]

    scene_likelihood=scatter_mean(masked_lihood, batch)

    return scene_likelihood




def compute_bird_metrics(pred_traj,gt_traj,gt_mask,batch,fps=29.97):

    pred_mask=(pred_traj!=0).any(-1)

    speed,acc,ang_speed,ang_acc=compute_kinematic_features(pred_traj,fps=fps)

    gt_speed,gt_acc,gt_ang_speed,gt_ang_acc=compute_kinematic_features(gt_traj[:,None],fps=fps)

    pred_speed_mask=pred_mask[:,:,1:] & pred_mask[:,:,:-1]
    gt_speed_mask=gt_mask[:,1:] & gt_mask[:,:-1]

    pred_acc_mask=pred_speed_mask[:,:,1:] & pred_speed_mask[:,:,:-1]
    gt_acc_mask=gt_speed_mask[:,1:] & gt_speed_mask[:,:-1]

    pred_angular_acc_mask=pred_acc_mask[:,:,1:] & pred_acc_mask[:,:,:-1]
    gt_angular_acc_mask=gt_acc_mask[:,1:] & gt_acc_mask[:,:-1]


    linear_speed_likelihood= histogram_estimate_torch(batch,gt_speed.flatten(1,2),speed.flatten(1,2),min_val=4,max_val=10,num_bins=10,
                                                      gt_valid_mask=gt_speed_mask,sim_valid_mask=pred_speed_mask.flatten(1,2),
                                                      )

    linear_acc_likelihood= histogram_estimate_torch(batch,gt_acc.flatten(1,2),acc.flatten(1,2),min_val=-4,max_val=4,
                                                      gt_valid_mask=gt_acc_mask,sim_valid_mask=pred_acc_mask.flatten(1,2),
                                                      )

    angular_speed_likelihood=histogram_estimate_torch(batch,gt_ang_speed.flatten(1,2),ang_speed.flatten(1,2),min_val=-1.8,max_val=1.8,
                                                      gt_valid_mask=gt_acc_mask,sim_valid_mask=pred_acc_mask.flatten(1,2),
                                                      )

    angular_acceleration_likelihood=histogram_estimate_torch(batch,gt_ang_acc.flatten(1,2),ang_acc.flatten(1,2),min_val=-40,max_val=40,
                                                      gt_valid_mask=gt_angular_acc_mask,sim_valid_mask=pred_angular_acc_mask.flatten(1,2),
                                                      )


    # exist_likelihood=histogram_estimate_torch(batch,gt_mask.to(torch.float16),pred_mask.flatten(1,2).to(torch.float16),
    #                                                     min_val=-0.5,max_val=1.5,num_bins=2
    #                                                   )
    exist_likelihood=(gt_mask[:,None]==pred_mask).float().mean(-1).mean(-1)

    exist_likelihood=scatter_mean(exist_likelihood, batch)


    return linear_speed_likelihood, linear_acc_likelihood, angular_speed_likelihood, angular_acceleration_likelihood,exist_likelihood

