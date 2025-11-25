# run_a6000_compare.py
import torch
import torch.nn as nn

def compare_with_4090(baseline_file="/home/guoke/sim/src/waymo_data/full/baseline_4090.pt"):
    baseline = torch.load(baseline_file)
    dtype = eval(baseline["dtype"])

    print(f"Loaded baseline dtype = {dtype}")

    # ===== Recreate input =====
    torch.manual_seed(42)

    A = torch.randn(2048, 2048, dtype=dtype, device="cuda")
    B = torch.randn(2048, 2048, dtype=dtype, device="cuda")
    x = torch.randn(32, 64, 64, 64, dtype=dtype, device="cuda")

    conv = nn.Conv2d(64, 64, 3, padding=1, dtype=dtype).cuda()

    # ===== Compute A6000 results =====
    out_matmul_6000 = (A @ B).cpu()
    out_conv_6000 = conv(x).cpu()

    torch.manual_seed(123)
    out_rng_6000 = torch.randn(1000000, dtype=dtype, device="cuda").cpu()

    # ===== Compare =====
    def report(name, t0, t1):
        diff = (t0 - t1).abs()
        print(f"\n[{name}]")
        print("  Max abs diff:", diff.max().item())
        print("  Mean abs diff:", diff.mean().item())
        print("  Rel diff:", diff.max().item() / (t0.abs().mean().item() + 1e-12))

    report("Matrix Multiply", baseline["matmul"], out_matmul_6000)
    report("Conv2D", baseline["conv"], out_conv_6000)
    report("RNG", baseline["rng"], out_rng_6000)

    print("\nIf FP32: differences < 1e-6 → 两块卡数值一致")
    print("If TF32/BF16: 1e-3 ~ 1e-2 也正常")

if __name__ == "__main__":
    compare_with_4090()
