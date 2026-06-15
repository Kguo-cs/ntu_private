import torch
from torch import Tensor


class InfoNoiseSampler:
    def __init__(
        self,
        num_bins: int = 64,
        ema_decay: float = 0.99,
        beta: float = 1.0,
        min_prob: float = 1e-3,
        warmup_steps: int = 1000,
        device: str = "cuda",
    ) -> None:
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        self.beta = beta
        self.min_prob = min_prob
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)

        self.loss_ema = torch.ones(num_bins, device=self.device)
        self.count = torch.zeros(num_bins, device=self.device)
        self.step = 0

    def sample(self, batch_size: int) -> Tensor:
        self.step += 1

        if self.step <= self.warmup_steps:
            bin_idx = torch.randint(
                low=0,
                high=self.num_bins,
                size=(batch_size,),
                device=self.device,
            )
        else:
            score = self.loss_ema.clamp_min(1e-8) ** self.beta
            prob = score / score.sum()

            # Prevent bin collapse.
            uniform = torch.ones_like(prob) / self.num_bins
            prob = (1.0 - self.min_prob) * prob + self.min_prob * uniform
            prob = prob / prob.sum()

            bin_idx = torch.multinomial(
                prob,
                num_samples=batch_size,
                replacement=True,
            )

        # Uniformly sample inside each bin.
        u = torch.rand(batch_size, device=self.device)
        t = (bin_idx.to(torch.float32) + u) / self.num_bins
        return t[:, None], bin_idx

    @torch.no_grad()
    def update(self, bin_idx: Tensor, loss_per_sample: Tensor) -> None:
        """
        Args:
            bin_idx: [B]
            loss_per_sample: [B], detached unweighted loss.
        """
        loss_per_sample = loss_per_sample.detach()

        for k in bin_idx.unique():
            mask = bin_idx == k
            value = loss_per_sample[mask].mean()

            self.loss_ema[k] = (
                self.ema_decay * self.loss_ema[k]
                + (1.0 - self.ema_decay) * value
            )
            self.count[k] += mask.sum()