import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,     # codebook size
    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight = 1.   # the weight on the commitment loss
)

x = torch.randn(1, 1024, 256) # encoded

x.requires_grad_()

quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)

print(quantized, indices, commit_loss)