import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Global font size settings ----
plt.rcParams.update({
    "font.size": 24,       # base font size
    "axes.titlesize": 20,  # subplot titles
    "axes.labelsize": 20,  # x and y labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# Load CSV file (replace with your filename if saved to disk)
df = pd.read_csv("/home/ke/code/catk/src/waymo_data/plot/mean_map.csv")
std= pd.read_csv("/home/ke/code/catk/src/waymo_data/plot/std_map.csv")

# Define runs and colors
runs = {
    "AIRL152_lcf336_val40_learmmap": "tab:red",
    "AIRL80_lcf312_val40_learnmap": "tab:green",
}

lables = {
    "AIRL152_lcf336_val40_learmmap": "DecompGAIL (unfreezing map encoder)",
    "AIRL80_lcf312_val40_learnmap": "DecompGAIL (freezing map encoder)",
}

# Moving average smoothing function
def smooth(series, window=5):
    return series.rolling(window, min_periods=1, center=True).mean()

fig, axes = plt.subplots(1, 2, figsize=(20,8), sharex=True)

# ---------------- Left: Policy Discriminator ----------------
ax = axes[0]
x = df["trainer/global_step"][:300]

for run, color in runs.items():
    y_mean = df[f"{run} - train/agent_disc_val"][:300]
    y_std = std[f"{run} - train/agent_disc_val_std"][:300]

    y_mean_smooth = smooth(y_mean, window=5)
    y_std_smooth = smooth(y_std, window=5)

    ax.plot(x, y_mean_smooth, label=lables[run], color=color)
    ax.fill_between(x, y_mean_smooth - y_std_smooth, y_mean_smooth + y_std_smooth,
                    alpha=0.2, color=color)

ax.set_xlabel("Global Step")
ax.set_ylabel("Policy Discriminator Score")
ax.grid(True)
ax.set_ylim(0.45,0.55)
ax.set_xlim(0,20000)

# ---------------- Right: Realism Meta Metric ----------------
df = pd.read_csv("/home/ke/code/catk/src/waymo_data/plot/result_map.csv")
x = df["trainer/global_step"][:5]
x = np.concatenate([np.array([0]), x.values])

ax = axes[1]
runs = {
    "AIRL80_lcf312_policygnn_val40": "tab:red",
    "AIRL80_lcf336_policygnn_type_val40": "tab:green",
}
lables = {
    "AIRL80_lcf312_policygnn_val40": "DecompGAIL (unfreezing map encoder)",
    "AIRL80_lcf336_policygnn_type_val40": "DecompGAIL (freezing map encoder)",
}

for run, color in runs.items():
    y_mean = df[f"{run} - val_closed/wosac/realism_meta_metric"][:5]
    y_mean = y_mean.values
    y_mean = np.concatenate([np.array([0.7836]), y_mean])
    ax.plot(x, y_mean, label=lables[run], color=color)

ax.set_xlabel("Global Step")
ax.set_ylabel("Realism MetaMetric")
ax.grid(True)
ax.set_xlim(0,15000)
ax.set_ylim(0.78,0.795)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("comparison_plots_largefont.pdf", format="pdf")
plt.show()
