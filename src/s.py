# run_a6000_compare_with_4090.py
import torch
import torch.nn as nn

def compare_with_4090(
    baseline_file="baseline_4090.pt",
    device="cuda",
):
    # 同样关闭 TF32 等，尽量一致
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

    data = torch.load(baseline_file, map_location="cpu")
    dtype = eval(data["dtype"])
    print(f"Loaded baseline from 4090, dtype = {dtype}")

    A_cpu = data["A"]
    B_cpu = data["B"]
    x_cpu = data["x"]
    conv_state = data["conv_state"]
    matmul_4090 = data["matmul_4090"]
    conv_4090 = data["conv_4090"]
    rng_cpu = data["rng_cpu"]

    # ---- 在 A6000 上执行同样的算子 ----
    A_6000 = A_cpu.to(device)
    B_6000 = B_cpu.to(device)
    x_6000 = x_cpu.to(device)

    conv_6000 = nn.Conv2d(64, 64, 3, padding=1, dtype=dtype).to(device)
    conv_6000.load_state_dict(conv_state)

    matmul_6000 = (A_6000 @ B_6000).cpu()
    conv_out_6000 = conv_6000(x_6000).cpu()

    def report(name, base, other):
        diff = (base - other).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel = max_diff / (base.abs().mean().item() + 1e-12)
        print(f"\n[{name}]")
        print("  Max abs diff:", max_diff)
        print("  Mean abs diff:", mean_diff)
        print("  Rel diff:", rel)

    report("Matrix Multiply", matmul_4090, matmul_6000)
    report("Conv2D", conv_4090, conv_out_6000)

    # RNG：我们只用 CPU 的 baseline，看看 GPU 生成的和 CPU baseline 差多少（可选）
    torch.manual_seed(123)
    rng_6000 = torch.randn(1_000_000, dtype=dtype, device=device).cpu()
    report("RNG (CPU baseline vs A6000 GPU)", rng_cpu, rng_6000)

    print("\n(For pure FP32, max diff ~1e-6 或更小是正常的；如果还很大再一起排查。)")


if __name__ == "__main__":
    compare_with_4090()
