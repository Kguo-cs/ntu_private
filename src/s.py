import torch
torch.backends.cuda.matmul.allow_tf32 = True  # 默认值

A = torch.randn(4096, 4096, device='cuda', dtype=torch.float32) * 1e5
B = A.T

torch.backends.cuda.matmul.allow_tf32 = True
res1 = A @ B

torch.backends.cuda.matmul.allow_tf32 = False
res2 = A @ B

print("max diff:", (res1 - res2).abs().max().item())

