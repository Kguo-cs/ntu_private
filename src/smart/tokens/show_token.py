import matplotlib.pyplot as plt
import pickle
import torch

agent_token_path="/home/ke/code/catk/src/smart/tokens/agent_vocab_555_s2.pkl"

agent_token_data = pickle.load(open(agent_token_path, "rb"))


# Assuming self.trajectory_token_* are [n_token, 8] after flatten(1, 2)
# We reshape them back to [n_token, 4, 2] for plotting
veh_traj = agent_token_data["token_all"]["veh"][:,-1].reshape(-1, 4, 2)
ped_traj = agent_token_data["token_all"]["ped"][:,-1].reshape(-1, 4, 2)
cyc_traj = agent_token_data["token_all"]["cyc"][:,-1].reshape(-1, 4, 2)

plt.figure(figsize=(8, 8))

# Plot each trajectory type with different color
# for traj in veh_traj:
#     plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.5, label='Vehicle')
# for traj in ped_traj:
#     plt.plot(traj[:, 0], traj[:, 1], color='green', alpha=0.5, label='Pedestrian')
for traj in cyc_traj:
    plt.plot(traj[:, 0], traj[:, 1], color='red', alpha=0.5, label='Cyclist')
#
# Avoid repeated labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("Agent Trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()
