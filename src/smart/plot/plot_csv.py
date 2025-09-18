import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file (replace with your filename if saved to disk)
df = pd.read_csv("/home/ke/code/catk/src/waymo_data/wandb_export_2025-09-18T16_20_51.248+08_00.csv")

# Plot
plt.figure(figsize=(8,5))

# --- Run 1: AIRL80_lcf356_noedge ---
x = df["trainer/global_step"]
y = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val"]
ymin = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val__MIN"]
ymax = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val__MAX"]

plt.plot(x, y, label="AIRL80_lcf356_noedge", color="tab:blue")
plt.fill_between(x, ymin, ymax, alpha=0.2, color="tab:blue")

# --- Run 2: AIRL80_lcf336 ---
y2 = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val"]
y2min = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val__MIN"]
y2max = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val__MAX"]

plt.plot(x, y2, label="AIRL80_lcf336", color="tab:orange")
plt.fill_between(x, y2min, y2max, alpha=0.2, color="tab:orange")

# Labels and formatting
plt.xlabel("Global Step")
plt.ylabel("Agent Disc Val")
plt.title("Agent Discriminator Value Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
