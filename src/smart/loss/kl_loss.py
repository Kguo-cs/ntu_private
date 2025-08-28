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
