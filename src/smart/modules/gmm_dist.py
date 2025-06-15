
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal,MultivariateNormal


def GMM_Dist(pred):
    next_logits, next_poses,next_cov=pred[...,0],pred[...,1:pred.shape[-1]//2+1],pred[...,pred.shape[-1]//2+1:]


    if next_cov.shape[-1]==4:
        next_poses = torch.cat(
            [
                next_poses[..., :2],
                next_poses[..., [-1]].cos(),
                next_poses[..., [-1]].sin(),
            ],
            dim=-1,
        )
    gmm=Independent(Normal(next_poses[:,:,0], next_cov[:,:,0]),1)
    # gmm = MixtureSameFamily(
    #     Categorical(logits=next_logits),MultivariateNormal(loc=next_poses, covariance_matrix=cov) )

    # gmm = MixtureSameFamily(
    #     Categorical(logits=next_logits), Independent(Normal(next_poses, next_cov), 1)
    # )
    # cov = (
    #     (gmm_cov * temp_cov)
    #     .repeat_interleave(2)[None, None, :]
    #     .expand(*next_poses.shape)
    # )  # [n_batch, k, 4]

    return gmm

def get_entropy(pred):
    next_logits, next_poses,next_cov=pred[...,0],pred[...,1:pred.shape[-1]//2+1],pred[...,pred.shape[-1]//2+1:]

    # Get categorical probabilities
    logits = next_logits  # shape [B, N]
    probs = torch.softmax(logits, dim=-1)

    # Component entropies: Normal entropy = 0.5 * log(2πe * σ²)
    component_entropy = 0.5 * (1.0 + torch.log(2 * torch.pi * next_cov ** 2)).sum(dim=-1)  # shape [B, N]

    # Weighted entropy
    weighted_entropy = (probs * component_entropy).sum(dim=-1)  # shape [B]

    # Mixture entropy ≈ weighted_entropy - ∑ p * log(p)
    mixture_entropy = weighted_entropy - (probs * torch.log(probs + 1e-8)).sum(dim=-1)

    return mixture_entropy
