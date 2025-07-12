import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# === 简单 GCN 模型 ===
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# === 随机图构造（可替换为真实数据） ===
def generate_graph(num_nodes=1000, num_edges=10000, in_dim=64):
    x = torch.randn((num_nodes, in_dim), device='cuda')
    row = torch.randint(0, num_nodes, (num_edges,), device='cuda')
    col = torch.randint(0, num_nodes, (num_edges,), device='cuda')
    edge_index = torch.stack([row, col], dim=0)
    return Data(x=x, edge_index=edge_index)

# === 统一推理函数 ===
def run_inference(model, data, tf32=True, amp_dtype=None):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    model.eval()
    with torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else torch.no_grad():
        return model(data.x, data.edge_index)

# === 主函数 ===
@torch.no_grad()
def compare_all_modes():
    model = GCN(64, 128, 16).cuda()
    data = generate_graph()

    print("[A] FP32 baseline")
    out_fp32 = run_inference(model, data, tf32=False)

    print("[B] TF32 enabled")
    out_tf32 = run_inference(model, data, tf32=True)
    print("TF32 diff (max):", (out_tf32 - out_fp32).abs().max().item())

    print("[C] AMP with FP16")
    out_fp16 = run_inference(model.half(), data, tf32=False, amp_dtype=torch.float16)
    print("FP16 diff (max):", (out_fp16 - out_fp32).abs().max().item())

    print("[D] AMP with BF16")
    model = model.float()  # 还原模型
    out_bf16 = run_inference(model, data, tf32=False, amp_dtype=torch.bfloat16)
    print("BF16 diff (max):", (out_bf16 - out_fp32).abs().max().item())

if __name__ == '__main__':
    compare_all_modes()
