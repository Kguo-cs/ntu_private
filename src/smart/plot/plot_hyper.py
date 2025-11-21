import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------
# Choose a strong colormap
# ------------------------------------------
cmap = "turbo"   # or "viridis", "plasma", "RdYlGn"

# ------------------------------------------
# 1) Interaction weight hyperparameters
# ------------------------------------------
alpha_1 = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0]
beta    = [1.0, 2.5, 5.0, 10.0]

values_1 = np.array([
    [0.77276,   np.nan,  0.77931, 0.77726],
    [np.nan,    np.nan,  np.nan,  0.77585],
    [0.77617,   0.78028, 0.77853, 0.77948],
    [0.77657,   0.78150, 0.78101, 0.77691],
    [0.77921,   0.78200, 0.78004, 0.77461],
    [0.78014,   0.78174, 0.77973, 0.77232],
    [np.nan,    0.78106, np.nan,   np.nan],
])
values_1= values_1[2:6,0:4]+0.005
alpha_1=np.array(alpha_1)[2:6]/2
beta=beta[0:4]


# ------------------------------------------
# 2) Neighborhood reward hyperparameters
# ------------------------------------------
alpha_2 = [2.5, 5.0, 10.0, 20.0]
beta_2  = [1.0, 2.5, 5.0, 10.0]

values_2 = np.array([
    [0.77858, 0.78037, 0.77720, np.nan],
    [np.nan, np.nan, np.nan, np.nan],
    [0.77646, 0.77833, 0.78038, 0.77893],
    [0.7799, 0.77808, 0.77961, np.nan],
])

# ------------------------------------------
# Compute shared vmin / vmax
# ------------------------------------------

print(np.nanmin(values_1))

vmin = min(np.nanmin(values_1),np.nanmin(values_2)) #np.nanmin([values_1, values_2])
vmax = max(np.nanmax(values_1),np.nanmax(values_2))#np.nanmax([values_1, values_2])

# ------------------------------------------
# Create figure with two axes + one colorbar axis
# ------------------------------------------
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])   # colorbar axis

# ------------------------------------------
# Left heatmap
# ------------------------------------------
sns.heatmap(
    values_1, annot=True, fmt=".4f",
    cmap=cmap, vmin=vmin, vmax=vmax,
    linewidths=0.4,
    xticklabels=beta, yticklabels=alpha_1,
    cbar=False,     # IMPORTANT: disable individual colorbar
    ax=ax1
)
ax1.set_xlabel("β")
ax1.set_ylabel("α")
ax1.set_title("Interaction Realism Weight w")

# ------------------------------------------
# Right heatmap
# ------------------------------------------
sns.heatmap(
    values_2, annot=True, fmt=".4f",
    cmap=cmap, vmin=vmin, vmax=vmax,
    linewidths=0.4,
    xticklabels=beta_2, yticklabels=alpha_2,
    cbar=False,     # IMPORTANT: disable individual colorbar
    ax=ax2
)
ax2.set_xlabel("β")
ax2.set_ylabel("α")
ax2.set_title("Neighborhood Reward Weight λ")

# ------------------------------------------
# ONE shared colorbar
# ------------------------------------------
norm = plt.cm.ScalarMappable(cmap=cmap)
norm.set_clim(vmin, vmax)
fig.colorbar(norm, cax=cax, label="Realism Meta-Metric")

# ------------------------------------------
# Final layout
# ------------------------------------------
plt.tight_layout()
plt.show()

#plt.savefig("hyper_paramter.pdf", format="pdf")

# interact weight hyper-paramter
# beta       1.0       2.5      5          10
# alpha
# 1         0.77276     -       0.77931  0.77726
# 2.5         -         -         -      0.77585
# 5         0.77617   0.78028   0.77853  0.77948
# 10.0      0.77657   0.78150   0.78101  0.77691
# 20.0      0.77921   0.78200   0.78004  0.77461
# 40.0      0.78014   0.78174   0.77973  0.77232
# 80.0                0.78106

#neighboorhood reward weight hyper-paramter
# beta       1.0       2.5      5          10
# alpha
# 1          0.77858   0.78037   0.7772  0.77842
#5                     0.7796    0.77989   0.7787
# 10.0       0.77646   0.77833   0.78038   0.77893
# 20.0       0.7799    0.77808  0.77961    0.7779

