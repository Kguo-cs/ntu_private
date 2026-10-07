# Extracted unchanged numerical helpers from VectorWorld, revision 4e19a43686e00d0b4e02d83e3230b6a1251592be.
# See ../SOURCES.json for provenance; dataset/trainer dependencies omitted.
from typing import Tuple
import torch

def _to_tensor_stats(mean, std, ref_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace (mean, std) with`ref_latents`Tensor with device / dtype and do the necessary shape check.

    Type of input supported:
    - float / int
    - list / tuple / numpy.ndarray
    - torch.Tensor

    Design elements
    --------
    1. Ifmean/stdis the metric broadcast directly to all dimensions;
    2. Ifmean/stdYes 1D vector Required length=latet_dim(ref_latents.shape[-1I'm sorry, but I don't know.
       Otherwise, it would be wrong to avoid silent broadcasting leading to potential numerical problems;
    3. No change in the original broadcasting semantics: all work on (N,D) or (N,1,D).
    """
    # Convert to Tensor, put on the same device / dtype as ref_latents
    mean_t = torch.as_tensor(mean, dtype=ref_latents.dtype, device=ref_latents.device)
    std_t = torch.as_tensor(std, dtype=ref_latents.dtype, device=ref_latents.device)

    # If the user accidentally transmits a high-dimensional array (e.g. (1, D)), this is flat 1D, semantic equivalent.
    if mean_t.ndim > 1:
        mean_t = mean_t.view(-1)
    if std_t.ndim > 1:
        std_t = std_t.view(-1)

    # Check whether the length corresponds to the flatt_dim if the vector by dimension
    if mean_t.ndim == 1:
        latent_dim = ref_latents.shape[-1]
        if mean_t.shape[0] != latent_dim or std_t.shape[0] != latent_dim:
            raise ValueError(
                f"[normalize_latents] Latent stats length mismatch: "
                f"got mean/std of length ({mean_t.shape[0]}, {std_t.shape[0]}) "
                f"but latent_dim = {latent_dim}. "
                "Please verify that latent_stats.pkl matches the "
                "`agent_latent_dim` / `lane_latent_dim` in the current model configuration, "
                "and re-run `cache_latent_stats` if necessary."
            )

    # Numerical protection: std at least 1e-6, avoid eliminating zeros
    std_t = torch.clamp(std_t, min=1e-6)
    return mean_t, std_t

def normalize_latents(
        agent_latents,
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """Normalize agent & lane latents using cached mean/std(Supporting metrics or vector-by-dimensional vectors)

    Parameters`*_latents_mean/std`Could be float/ list/np.ndarray / torch.TensorI'm not sure what you're doing.
    This function automatically converts to the same level as latentdevice/dtypeTensor.
    """
    if agent_latents.numel() > 0:
        a_mean, a_std = _to_tensor_stats(agent_latents_mean, agent_latents_std, agent_latents)
        agent_latents = (agent_latents - a_mean) / a_std

    if lane_latents.numel() > 0:
        l_mean, l_std = _to_tensor_stats(lane_latents_mean, lane_latents_std, lane_latents)
        lane_latents = (lane_latents - l_mean) / l_std

    return agent_latents, lane_latents

def unnormalize_latents(
        agent_latents,
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """Unnormalize agent & lane latents using cached mean/std(Supporting metrics or vector-by-dimensional vectors)"""
    if agent_latents.numel() > 0:
        a_mean, a_std = _to_tensor_stats(agent_latents_mean, agent_latents_std, agent_latents)
        agent_latents = agent_latents * a_std + a_mean

    if lane_latents.numel() > 0:
        l_mean, l_std = _to_tensor_stats(lane_latents_mean, lane_latents_std, lane_latents)
        lane_latents = lane_latents * l_std + l_mean

    return agent_latents, lane_latents

def reparameterize(mu, log_var):
    """ Reparameterization trick to sample from a Gaussian distribution
    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        log_var (torch.Tensor): Log variance of the Gaussian distribution.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    assert mu.shape == log_var.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

def sample_latents(
        data,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std,
        normalize=True):
    """ Sample latents from the agent and lane data, and (optionally) normalize them."""
    agent_mu = data['agent'].x
    agent_log_var = data['agent'].log_var
    agent_latents = reparameterize(agent_mu, agent_log_var)

    lane_mu = data['lane'].x
    lane_log_var = data['lane'].log_var
    lane_latents = reparameterize(lane_mu, lane_log_var)

    if normalize:
        agent_latents, lane_latents = normalize_latents(
            agent_latents,
            lane_latents,
            agent_latents_mean,
            agent_latents_std,
            lane_latents_mean,
            lane_latents_std)

    return agent_latents, lane_latents
