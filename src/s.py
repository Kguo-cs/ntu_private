import torch

device = torch.device('cuda')

# 构造动态范围极大的矩阵，更容易暴露 TF32 的精度问题
A = torch.randn(8192, 8192, device=device, dtype=torch.float32)
scale = torch.linspace(1e-3, 1e6, steps=8192, device=device).unsqueeze(0)
A = A * scale

B = A.T

def matmul_diff():
    torch.backends.cuda.matmul.allow_tf32 = True
    res_tf32 = torch.matmul(A, B)

    torch.backends.cuda.matmul.allow_tf32 = False
    res_fp32 = torch.matmul(A, B)

    diff = (res_tf32 - res_fp32).abs()
    print("Max diff:", diff.max().item())
    print("Mean diff:", diff.mean().item())

if __name__ == '__main__':
    matmul_diff()

