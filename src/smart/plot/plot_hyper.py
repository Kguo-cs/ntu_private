import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------
# Fill the matrix according to your table
# Use np.nan for missing entries
# ------------------------------------------

# α values (rows)
alpha = [1.0, 2.5, 5.0, 10.0, 20.0]

# β values (columns)
beta  = [1.0, 2.5, 5.0, 10.0]

# Heatmap matrix
values = np.array([
    [0.77276,   np.nan,  0.77931, 0.77726],   # α = 1.0
    [np.nan,    np.nan,  np.nan,  0.77585],   # α = 2.5
    [np.nan,    0.78028, 0.77853, 0.77948],   # α = 5.0
    [0.77657,   0.78150, 0.78101, 0.77691],   # α = 10.0
    [np.nan,    0.78200, 0.78004, np.nan]     # α = 20.0
])

# ------------------------------------------
# Plot the heatmap
# ------------------------------------------
plt.figure(figsize=(8, 6))

sns.heatmap(
    values,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    linewidths=0.5,
    xticklabels=beta,
    yticklabels=alpha,
    cbar_kws={'label': 'Realism Meta-Metric'}
)

plt.xlabel("β")
plt.ylabel("α")
plt.title("α–β Sweep Performance Heatmap")

plt.tight_layout()
plt.show()


# interact weight
# beta       1.0       2.5      5          10
# alpha
# 1         0.77276     -       0.77931  0.77726
# 2.5         -         -         -      0.77585
# 5         0.77617   0.78028   0.77853  0.77948
# 10.0      0.77657   0.78150   0.78101  0.77691
# 20.0      0.77921   0.78200   0.78004  0.77461
# 40.0                0.78174   0.77973  0.77232
# 80.0                0.78106

#neighboorhood reward weight
# beta       1.0       2.5      5          10
# alpha
# 1                      -       -
# 2.5
# 5
# 10.0                                   -
# 20.0                   -