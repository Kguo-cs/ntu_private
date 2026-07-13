import torch

def gaussian_kernel_torch(x, y, kernel_mul=1.0, kernel_num=1, fix_sigma=None, eps=1e-8):
    """Gaussian kernel used for set-level MMD.

    Args:
        x: Tensor [N, D].
        y: Tensor [M, D].
    """
    assert x.ndim == 2, x.shape
    assert y.ndim == 2, y.shape

    total = torch.cat([x, y], dim=0)
    n_samples = total.size(0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    l2_distance = ((total0 - total1) ** 2).sum(dim=-1)

    if fix_sigma is not None:
        bandwidth = x.new_tensor(float(fix_sigma))
    else:
        denom = max(n_samples * n_samples - n_samples, 1)
        bandwidth = torch.sum(l2_distance.detach()) / denom

    bandwidth = bandwidth.clamp_min(eps)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = [
        torch.exp(-l2_distance / (bandwidth * (kernel_mul ** i)).clamp_min(eps))
        for i in range(kernel_num)
    ]
    return sum(kernels)


def compute_mmd_different_sizes_torch(x, y, kernel_mul=1.0, kernel_num=1, fix_sigma=None):
    """Compute MMD between two sets with possibly different numbers of samples."""
    assert x.ndim == 2, x.shape
    assert y.ndim == 2, y.shape

    if x.shape[0] == 0 or y.shape[0] == 0:
        return x.new_tensor(float("nan"))

    kernels = gaussian_kernel_torch(
        x=x,
        y=y,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    n_x = x.size(0)
    n_y = y.size(0)

    xx = kernels[:n_x, :n_x]
    yy = kernels[n_x:, n_x:]
    xy = kernels[:n_x, n_x:]
    yx = kernels[n_x:, :n_x]

    mmd = xx.mean() + yy.mean() - xy.mean() - yx.mean()
    return mmd.clamp_min(0.0)


def normalize_angle_torch(angle):
    """Normalize angle to [0, 2*pi)."""
    return torch.remainder(angle, 2.0 * torch.pi)


def angle_to_vector_torch(angle):
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _vehicles_to_torch(vehicles, device="cpu"):
    """Convert vehicles to torch.

    Expected vehicle format:
        [pos_x, pos_y, speed, cos_heading, sin_heading, length, width, vx, vy]
    The last two velocity columns are optional.
    """
    if isinstance(vehicles, torch.Tensor):
        return vehicles.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(vehicles, device=device, dtype=torch.float32)


def _extract_mmd_features(vehicles):
    """Extract TrafficGen-style MMD features from unified vehicle state."""
    assert vehicles.ndim == 2, vehicles.shape
    assert vehicles.shape[-1] >= 7, vehicles.shape

    pos = vehicles[:, 0:2]
    speed = vehicles[:, 2:3]

    heading = torch.atan2(vehicles[:, 4], vehicles[:, 3])
    heading = normalize_angle_torch(heading)

    heading_raw = heading[:, None]
    heading_vec = angle_to_vector_torch(heading)

    size = vehicles[:, 5:7]

    # If vx/vy are available, use vector velocity like UniGen/UMGen/your second file.
    # Otherwise fall back to speed only.
    vel = vehicles[:, 7:9] if vehicles.shape[-1] >= 9 else speed

    return {
        "pos": pos,
        "speed": speed,
        "vel": vel,
        "heading": heading,
        "heading_raw": heading_raw,
        "heading_vec": heading_vec,
        "size": size,
    }


def compute_mmd_metrics(samples, gt_samples, key='',device="cpu"):
    """Compute per-scene, per-sample TrafficGen-style MMD metrics.

    samples[i]["vehicles"] can be:
        [N, S, D] where S is number of generated samples, or
        [N, D] for a single generated sample.

    gt_samples[i]["vehicles"] should be:
        [N_gt, D].

    Vehicle format:
        [pos_x, pos_y, speed, cos_heading, sin_heading, length, width, vx, vy]
    """
    print("Computing agent MMD metrics")

    mmd_pos_all = []
    mmd_speed_all = []
    mmd_vel_all = []
    mmd_head_all = []
    mmd_head_center_all = []
    mmd_head_transformed_all = []
    mmd_size_all = []

    kernel_mul = 1.0
    kernel_num = 1

    for i in range(len(samples)):
        vehicles_gen_all = samples[i]["vehicles"]
        vehicles_real = gt_samples[i][key+"select_agents"]

        vehicles_real = _vehicles_to_torch(vehicles_real, device=device)

        if vehicles_real.numel() == 0:
            continue

        # generated vehicles: support [N, D] or [N, S, D]
        vehicles_gen_all = _vehicles_to_torch(vehicles_gen_all, device=device)
        if vehicles_gen_all.ndim == 2:
            vehicles_gen_all = vehicles_gen_all[:, None, :]
        assert vehicles_gen_all.ndim == 3, vehicles_gen_all.shape

        real_feat = _extract_mmd_features(vehicles_real)

        center_head = real_feat["heading"][-1]

        num_samples = vehicles_gen_all.shape[1]
        for j in range(num_samples):
            vehicles_gen = vehicles_gen_all[:, j]

            if vehicles_gen.numel() == 0:
                continue

            gen_feat = _extract_mmd_features(vehicles_gen)

            mmd_pos_all.append(
                compute_mmd_different_sizes_torch(
                    gen_feat["pos"],
                    real_feat["pos"],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
            )

            # Speed MMD: TrafficGen/LCTGen-style.
            mmd_speed_all.append(
                compute_mmd_different_sizes_torch(
                    gen_feat["speed"],
                    real_feat["speed"],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
            )

            # Velocity-vector MMD: same spirit as your second file's mmd_vel.
            mmd_vel_all.append(
                compute_mmd_different_sizes_torch(
                    gen_feat["vel"],
                    real_feat["vel"],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
            )

            mmd_head_all.append(
                compute_mmd_different_sizes_torch(
                    gen_feat["heading_raw"],
                    real_feat["heading_raw"],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
            )

            # mmd_head_center_all.append(
            #     compute_mmd_different_sizes_torch(
            #         normalize_angle_torch(gen_feat["heading"] - center_head)[:, None],
            #         normalize_angle_torch(real_feat["heading"] - center_head)[:, None],
            #         kernel_mul=kernel_mul,
            #         kernel_num=kernel_num,
            #     )
            # )
            #
            # mmd_head_transformed_all.append(
            #     compute_mmd_different_sizes_torch(
            #         gen_feat["heading_vec"],
            #         real_feat["heading_vec"],
            #         kernel_mul=kernel_mul,
            #         kernel_num=kernel_num,
            #     )
            # )

            mmd_size_all.append(
                compute_mmd_different_sizes_torch(
                    gen_feat["size"],
                    real_feat["size"],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
            )

    def _mean_or_nan(values):
        if len(values) == 0:
            return float("nan")
        values = torch.stack(values)
        return values[torch.isfinite(values)].mean().item()

    return {
        key+"mmd_pos": _mean_or_nan(mmd_pos_all),
        key+"mmd_speed": _mean_or_nan(mmd_speed_all),
        key+"mmd_vel": _mean_or_nan(mmd_vel_all),
        key+"mmd_head": _mean_or_nan(mmd_head_all),
      #  key + "mmd_head_center": _mean_or_nan(mmd_head_center_all),
      #  key + "mmd_head_transformed": _mean_or_nan(mmd_head_transformed_all),
        key + "mmd_size": _mean_or_nan(mmd_size_all),
    }