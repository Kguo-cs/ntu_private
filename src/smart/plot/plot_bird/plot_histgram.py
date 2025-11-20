import torch
import os


def plot_histgram(name, valid_gt_speed, valid_pred_speed,
                  min_val, max_val, num_bins=11, save_dir="/home/ke/code/catk/src/waymo_data/bird_data1/result"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    print(torch.quantile(valid_gt_speed, 0.01),torch.quantile(valid_gt_speed, 0.99))

    mpl.rcParams['toolbar'] = 'None'

    os.makedirs(save_dir, exist_ok=True)

    valid_gt_speed = valid_gt_speed.to(torch.float32)
    valid_pred_speed = valid_pred_speed.to(torch.float32)

    # Clamp to valid range
    valid_gt_speed = torch.clamp(valid_gt_speed, min_val, max_val)
    valid_pred_speed = torch.clamp(valid_pred_speed, min_val, max_val)

    # Compute histograms
    hist_gt = torch.histc(valid_gt_speed, bins=num_bins, min=min_val, max=max_val)
    hist_pred = torch.histc(valid_pred_speed, bins=num_bins, min=min_val, max=max_val)

    hist_gt=hist_gt/hist_gt.sum()
    hist_pred=hist_pred/hist_pred.sum()

    # Bin edges and width
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
    width = (max_val - min_val) / num_bins

    # Plot both histograms together
    plt.figure(figsize=(7, 5))
    plt.bar(bin_edges[:-1].cpu().numpy(), hist_gt.cpu().numpy(),
            width=width, align='edge',
            color='blue', alpha=0.6, label='GT', edgecolor='black')
    plt.bar(bin_edges[:-1].cpu().numpy(), hist_pred.cpu().numpy(),
            width=width, align='edge',
            color='green', alpha=0.5, label='Pred', edgecolor='black')

    plt.title(name+" Distribution Comparison")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # plt.show()
    save_path = os.path.join(save_dir, f"{name}_hist.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
