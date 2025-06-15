import torch
from torch.distributions import Categorical, MixtureSameFamily, Independent, Normal,MixtureOfDiagNormals

# Example shapes
B = 32          # Batch size
M = 5           # Number of mixture components
D = 2           # Dimensionality

# Inputs
logits = torch.randn(B, M)                     # Mixture logits
loc = torch.randn(B, M, D)                     # Mean of each component
scale = torch.randn(B, M, D).abs() + 1e-3      # Stddev of each component (must be positive)

# Construct GMM
mix = Categorical(logits=logits)
comp = Independent(Normal(loc, scale), 1)  # Each component is a D-dimensional Gaussian
gmm = MixtureOfDiagNormals(mix, comp)

# Reparameterized sample
z = gmm.rsample()  # Shape: [B, D]
