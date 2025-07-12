import torch
import torch.nn as nn

device = torch.device('cuda')

class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def inference_diff():
    torch.manual_seed(42)
    model = SimpleMLP(4096).to(device).eval()
    input = torch.randn(32, 4096, device=device, dtype=torch.float32)

    torch.backends.cuda.matmul.allow_tf32 = True
    out_tf32 = model(input)

    torch.backends.cuda.matmul.allow_tf32 = False
    out_fp32 = model(input)

    diff = (out_tf32 - out_fp32).abs()
    print("Max diff:", diff.max().item())
    print("Mean diff:", diff.mean().item())

if __name__ == '__main__':
    inference_diff()
