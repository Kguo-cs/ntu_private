import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------
# 1) Interaction weight hyperparameters
# ------------------------------------------

alpha_1 = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0]
beta    = [1.0, 2.5, 5.0, 10.0]

values_1 = np.array([
    [0.77276,   np.nan,  0.77931, 0.77726],   # α = 1.0
    [np.nan,    np.nan,  np.nan,  0.77585],   # α = 2.5
    [0.77617,   0.78028, 0.77853, 0.77948],   # α = 5.0
    [0.77657,   0.78150, 0.78101, 0.77691],   # α = 10.0
    [0.77921,   0.78200, 0.78004, 0.77461],   # α = 20.0
    [0.78014,   0.78174, 0.77973, 0.77232],   # α = 40.0
    [np.nan,    0.78106, np.nan,   np.nan],   # α = 80.0
])

# ------------------------------------------
# 2) Neighborhood reward hyperparameters
# ------------------------------------------

alpha_2 = [1.0, 10.0, 20.0]
beta_2  = [1.0, 2.5, 5.0, 10.0]

values_2 = np.array([
    [np.nan, 0.78037, 0.77720, np.nan],     # α = 1.0
    [0.77646, np.nan, 0.78038, 0.77893],    # α = 10.0
    [np.nan, 0.77808, 0.77961, np.nan],     # α = 20.0
])

# ------------------------------------------
# Plot two heatmaps side-by-side
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: interaction weights
sns.heatmap(
    values_1,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    linewidths=0.4,
    xticklabels=beta,
    yticklabels=alpha_1,
    cbar_kws={'label': 'Realism Meta-Metric'},
    ax=axes[0]
)
axes[0].set_xlabel("β")
axes[0].set_ylabel("α")
axes[0].set_title("Interaction Weight Hyperparameters")

# Right: neighborhood reward weights
sns.heatmap(
    values_2,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    linewidths=0.4,
    xticklabels=beta_2,
    yticklabels=alpha_2,
    cbar_kws={'label': 'Realism Meta-Metric'},
    ax=axes[1]
)
axes[1].set_xlabel("β")
axes[1].set_ylabel("α")
axes[1].set_title("Neighborhood Reward Hyperparameters")

plt.tight_layout()
plt.show()


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
# 1                    0.78037   0.7772
# 10.0       0.77646             0.78038   0.77893
# 20.0                 0.77808  0.77961

