import torch
torch.backends.cuda.matmul.allow_tf32 = True  # 默认值

A = torch.randn(512, 512, dtype=torch.float32, device='cuda')
B = torch.randn(512, 512, dtype=torch.float32, device='cuda')

# baseline
with torch.no_grad():
    ref = torch.matmul(A, B)

# disable TF32
torch.backends.cuda.matmul.allow_tf32 = False
with torch.no_grad():
    no_tf32 = torch.matmul(A, B)

print("diff:", torch.abs(ref - no_tf32).max())  # 差异一般可达 1e-3 ~ 1e-2
