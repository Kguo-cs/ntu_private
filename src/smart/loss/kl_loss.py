# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import kl_divergence, Independent, Normal, OneHotCategoricalStraightThrough
from typing import Optional, Union
from torch import Tensor
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough, Categorical
from torch.nn import functional as F



class MyDist:
    def __init__(self, *args, **kwargs) -> None:
        self.distribution = None

    def log_prob(self, sample: Tensor) -> Tensor:
        pass

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        pass

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        pass


class DiagGaussian(MyDist):
    def __init__(self, mean: Tensor, log_std: Tensor, valid: Optional[Tensor] = None) -> None:
        """
        mean: [n_sc, n_ag, (k_pred), out_dim]
        """
        super().__init__()
        self.mean = mean
        self.valid = valid
        self.distribution = Independent(Normal(self.mean, log_std.exp()), 1)
        self.stddev = self.distribution.stddev

    def log_prob(self, sample: Tensor) -> Tensor:
        """
        log_prob: [n_sc, n_ag]
        """
        return self.distribution.log_prob(sample)

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_sc, n_ag, out_dim]
        """
        if type(deterministic) is Tensor:
            det_sample = self.distribution.mean
            rnd_sample = self.distribution.rsample()
            sample = det_sample.masked_fill(~deterministic.unsqueeze(-1), 0) + rnd_sample.masked_fill(
                deterministic.unsqueeze(-1), 0
            )
        else:
            if deterministic:
                sample = self.distribution.mean
            else:
                sample = self.distribution.rsample()
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.mean = self.mean.repeat_interleave(repeats, dim)
        self.stddev = self.stddev.repeat_interleave(repeats, dim)
        self.distribution = Independent(Normal(self.mean, self.stddev), 1)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)



class BalancedKL:
    """
    Mastering atari with discrete world models, Algorithm 2: KL Balancing with Automatic Differentiation
    """

    def __init__(self, kl_balance_scale: float, kl_free_nats: float) -> None:
        self.alpha = kl_balance_scale
        self.free_nats = kl_free_nats

    def kl_diag_gaussians(self,post, prior=None, mask=None, reduce="mean"):
        """
        KL(q || p) for diagonal Gaussians where:
          q ~ N(mu_q, diag(sigma_q^2)),  p ~ N(mu_p, diag(sigma_p^2))
        Inputs can be either your DiagGaussian objects or (mu, logvar) tuples.

        Args:
            post: DiagGaussian or (mu_q, logvar_q)
            prior: DiagGaussian or (mu_p, logvar_p) or None for N(0,I)
            mask: optional boolean or float mask broadcastable to mu shape sans last dim,
                  e.g. [B,T] or [B,T,1] or [B,T,K]
            reduce: "mean" | "sum" | "none"  (over masked elements)

        Returns:
            kl_reduced: scalar (if reduce != "none") or per-item tensor [*,] (if "none"),
            kl_per_item: KL summed over latent dim, shape matching post batch/time dims
        """
        # Unpack posterior
        mu_q, logvar_q = post#post.mu, post.logvar

        # Unpack prior (or standard Normal)
        if prior is None:
            mu_p = torch.zeros_like(mu_q)
            logvar_p = torch.zeros_like(logvar_q)
            valid_p = None
        mu_p, logvar_p = prior #prior.mu, prior.logvar

        # Build final mask if not provided
        if mask is None:
            mask = valid_q
            if valid_p is not None:
                mask = mask & valid_p if mask is not None else valid_p

        # Numerical safety on log-variances
        logvar_q = torch.clamp(logvar_q, -20.0, 5.0)
        logvar_p = torch.clamp(logvar_p, -20.0, 5.0)

        # KL for diagonal Gaussians: 0.5 * [ log|Σ_p| - log|Σ_q| + tr(Σ_p^{-1} Σ_q)
        #                              + (μ_p-μ_q)^T Σ_p^{-1} (μ_p-μ_q) - k ]
        # Implemented elementwise and sum over latent dim
        var_q = torch.exp(logvar_q)
        inv_var_p = torch.exp(-logvar_p)
        diff = mu_q - mu_p

        kl_elwise = 0.5 * ((logvar_p - logvar_q) + (var_q + diff ** 2) * inv_var_p - 1.0)  # [..., K]
        kl_per_item = kl_elwise.sum(dim=-1)  # sum over latent dim -> shape [...,]

        if mask is not None:
            mask = mask.to(dtype=kl_per_item.dtype)
            # If mask lacks latent dim, it will broadcast; ok.
            kl_masked = kl_per_item * mask
            if reduce == "mean":
                denom = mask.sum().clamp_min(1.0)
                kl_reduced = kl_masked.sum() / denom
            elif reduce == "sum":
                kl_reduced = kl_masked.sum()
            elif reduce == "none":
                kl_reduced = kl_masked
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
        else:
            if reduce == "mean":
                kl_reduced = kl_per_item.mean()
            elif reduce == "sum":
                kl_reduced = kl_per_item.sum()
            elif reduce == "none":
                kl_reduced = kl_per_item
            else:
                raise ValueError(f"Unknown reduce: {reduce}")

        return kl_reduced, kl_per_item

    def compute(self, posterior: Normal, prior: Normal) -> Tensor:  # type: ignore
        """
        Args:
            posterior: [n_batch, n_agent, E], Normal
            prior: [n_batch, n_agent, E], Normal
        Return:
            error: [n_batch, n_agent, E]
        """
        if self.alpha > 0:
            # latent dist is either DiagGaussian or MultiCategorical, both are wrapped by Independent
            # assert type(posterior) == Independent
            if type(posterior.base_dist) == OneHotCategoricalStraightThrough:
                detach_post = Independent(OneHotCategoricalStraightThrough(probs=posterior.base_dist.probs.detach()), 1)
                detach_prior = Independent(OneHotCategoricalStraightThrough(probs=prior.base_dist.probs.detach()), 1)
            elif type(posterior.base_dist) == Normal:
                detach_post = Independent(
                    Normal(posterior.base_dist.loc.detach(), posterior.base_dist.scale.detach()), 1
                )
                detach_prior = Independent(Normal(prior.base_dist.loc.detach(), prior.base_dist.scale.detach()), 1)
            error_0 = kl_divergence(detach_post, prior)
            error_1 = kl_divergence(posterior, detach_prior)
            if self.free_nats > 0:
                error_0 = torch.max(error_0, error_0.new_full(error_0.size(), self.free_nats))
                error_1 = torch.max(error_1, error_1.new_full(error_1.size(), self.free_nats))
            error =  error_0 + self.alpha * error_1
        else:
            error = kl_divergence(posterior, prior)
            if self.free_nats > 0:
                error = torch.max(error, error.new_full(error.size(), self.free_nats))
        return error
