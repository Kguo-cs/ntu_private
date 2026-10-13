"""Scene context derived entirely from the current noisy agent state."""
import torch
from torch import nn
from src.smart.layers import MLPLayer


class SceneStateContext(nn.Module):
    """Pool normalized current states; no reference labels or future states.

    Zero output initialization lets the optional branch begin with the same
    predictions as its backbone. Its final projection learns first.
    """
    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.embedding = MLPLayer(state_dim, hidden_dim, output_dim)
        nn.init.zeros_(self.embedding.mlp[-1].weight)
        nn.init.zeros_(self.embedding.mlp[-1].bias)

    def forward(self, state, batch, num_scenes, mean, scale):
        if state.ndim != 2 or state.shape[1] < self.state_dim or batch.shape != state.shape[:1]:
            raise ValueError("Scene context requires current states [N,D] and scene indices [N]")
        normalized = (state[:, :self.state_dim] - mean) / scale.clamp_min(1e-6)
        values = normalized.new_zeros((num_scenes, self.state_dim))
        values.index_add_(0, batch.long(), normalized)
        count = torch.bincount(batch.long(), minlength=num_scenes).to(values).clamp_min(1)
        return self.embedding(values / count[:, None])
