
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal,MultivariateNormal


def GMM_Dist(next_logits, next_poses,cov,temp_mode=1,temp_cov=1):
    next_logits = next_logits / temp_mode
    next_poses = torch.cat(
        [
            next_poses[..., :2],
            next_poses[..., [-1]].cos(),
            next_poses[..., [-1]].sin(),
        ],
        dim=-1,
    )
    gmm = MixtureSameFamily(
        Categorical(logits=next_logits),MultivariateNormal(loc=next_poses, covariance_matrix=cov) )

    # gmm = MixtureSameFamily(
    #     Categorical(logits=next_logits), Independent(Normal(next_poses, cov), 1)
    # )
    # cov = (
    #     (gmm_cov * temp_cov)
    #     .repeat_interleave(2)[None, None, :]
    #     .expand(*next_poses.shape)
    # )  # [n_batch, k, 4]

    return gmm