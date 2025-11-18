import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_penalty(critic, expert_s, expert_a, policy_s, policy_a, gp_lambda=10.0):
    """
    WGAN-GP gradient penalty:
    gp_lambda * E[(||∇_{x̂} D(x̂)||_2 - 1)^2]
    where x̂ is an interpolation of expert and policy (s,a).

    expert_s, expert_a, policy_s, policy_a: [B, ...]
    """
    device = expert_s.device

    # Interpolate between expert and policy samples
    alpha = torch.rand(expert_s.size(0), 1, device=device)
    alpha_s = alpha
    alpha_a = alpha

    # Flatten/interpolate
    interpolated_s = alpha_s * expert_s + (1 - alpha_s) * policy_s
    interpolated_a = alpha_a * expert_a + (1 - alpha_a) * policy_a

    interpolated_s = interpolated_s.detach().requires_grad_(True)
    interpolated_a = interpolated_a.detach().requires_grad_(True)

    critic_interpolates = critic(interpolated_s, interpolated_a)  # [B, 1]

    # Aggregate outputs into scalar for autograd
    grad_outputs = torch.ones_like(critic_interpolates)

    # Compute gradients wrt interpolated inputs
    grads = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=[interpolated_s, interpolated_a],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    grad_s, grad_a = grads
    # Concatenate grads over s and a, then take L2 norm per sample
    grad = torch.cat([grad_s, grad_a], dim=-1)  # [B, state_dim + action_dim]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)  # [B]

    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp_lambda * gp


