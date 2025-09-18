import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file (replace with your filename if saved to disk)
df = pd.read_csv("/home/ke/code/catk/src/waymo_data/wandb_export_2025-09-18T16_20_51.248+08_00.csv")

# Plot
plt.figure(figsize=(8,5))

# --- Run 1: AIRL80_lcf356_noedge ---
x = df["trainer/global_step"][:300]
y = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val"][:300]
# ymin = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val__MIN"]
# ymax = df["AIRL80_lcf356_policygnn_type_val40_noedge - train/agent_disc_val__MAX"]

plt.plot(x, y, label="PS-GAIL", color="tab:blue")
# plt.fill_between(x, ymin, ymax, alpha=0.2, color="tab:blue")

# --- Run 2: AIRL80_lcf336 ---
y2 = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val"][:300]
#y2min = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val__MIN"][:len(x)]
#y2max = df["AIRL80_lcf336_policygnn_type_val40 - train/agent_disc_val__MAX"][:len(x)]

plt.plot(x, y2, label="DecompGAIL", color="tab:orange")
# plt.fill_between(x, y2min, y2max, alpha=0.2, color="tab:orange")

# Labels and formatting
plt.xlabel("Global Step")
plt.ylabel("Policy Discriminator Value")
# plt.title("Policy Discriminator Value Comparison")
plt.legend(loc="upper right")
plt.grid(True)

plt.ylim(0.18,0.6)
plt.tight_layout()
plt.savefig('1.pdf')
