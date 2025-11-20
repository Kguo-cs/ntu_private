import itertools
import pandas as pd

# -------------------------------
# Hyperparameter grid
# -------------------------------
alphas = [0.1, 1.0, 10.0]
betas = [5.0, 10.0, 20.0]

# -------------------------------
# Placeholder: your training + eval loop
# -------------------------------
def run_experiment(alpha, beta):
    """
    Run DecompGAIL training with:
      - interaction weighting decay scaling `alpha`
      - interaction weighting decay rate `beta`
    and WITHOUT social rewards.

    This is a stub. Replace the body with your actual training call.

    Should return a dict, e.g.:
      {
        "meta_metric": ...,
        "kinematic": ...,
        "interactive": ...,
        "map_based": ...
      }
    """
    # TODO: plug in your real experiment code here
    # Example (fake) values just for demonstration:
    import random
    return {
        "meta_metric": 0.782 + random.uniform(-0.002, 0.002),
        "kinematic": 0.49 + random.uniform(-0.003, 0.003),
        "interactive": 0.81 + random.uniform(-0.004, 0.004),
        "map_based": 0.91 + random.uniform(-0.003, 0.003),
    }

# -------------------------------
# Run sweep
# -------------------------------
results = []

for alpha, beta in itertools.product(alphas, betas):
    metrics = run_experiment(alpha, beta)

    results.append({
        "alpha": alpha,
        "beta": beta,
        "meta_metric": metrics["meta_metric"],
        "kinematic": metrics["kinematic"],
        "interactive": metrics["interactive"],
        "map_based": metrics["map_based"],
    })

# -------------------------------
# Summarize in a table
# -------------------------------
df = pd.DataFrame(results)
df = df.sort_values(by=["alpha", "beta"]).reset_index(drop=True)

print(df.to_string(index=False))

# Optionally: pivot for a heatmap-friendly view of meta-metric
pivot_meta = df.pivot(index="alpha", columns="beta", values="meta_metric")
print("\nMeta-metric (rows = alpha, cols = beta):")
print(pivot_meta.to_string())


# beta       1.0       2.5      5  10
# alpha
# 5    -  -  -
# 10.0    -  -  -
# 20.0   -  -  -
# 40.0   -  -  -