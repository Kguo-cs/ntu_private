import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load
global_pose = torch.load("/home/ke/code/catk/src/waymo_data/token/global_pose.pt")   # [N, 2]
print("global_pose shape:", global_pose.shape)

# If you also saved heading or want to recompute it, let me know.
# For now, use dummy z=0

x = global_pose[:, 0]
y = global_pose[:, 1]
#z = global_pose[:, 2]

print("X min:", torch.quantile(x,0.01), "   X max:", torch.quantile(x,0.99))
print("X min:", torch.quantile(y,0.01), "   X max:", torch.quantile(y,0.99))
#print("X min:", torch.quantile(z,0.01), "   X max:", torch.quantile(z,0.99))

print("X min:", torch.quantile(x,0.001), "   X max:", torch.quantile(x,0.999))
print("X min:", torch.quantile(y,0.001), "   X max:", torch.quantile(y,0.999))
# print("X min:", torch.quantile(z,0.001), "   X max:", torch.quantile(z,0.999))

# 3D scatter plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.scatter(x[::100], y[::100], s=1, alpha=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel("Z (placeholder)")

plt.title("Entry Global Positions (XYZ)")
plt.show()

# X min: -80.70899963378906    X max: 53.74100112915039        140
# Y min: -28.947999954223633    Y max: 102.33000183105469      140
# Z min: -4.548699855804443    Z max: 66.8219985961914         70


# X min: tensor(-66.0600)    X max: tensor(39.4220)   -60, 40
# X min: tensor(-14.6170)    X max: tensor(79.9970)   -10 80
# X min: tensor(1.1736)    X max: tensor(46.2070)       0  40

# X min: tensor(-69.3470)    X max: tensor(42.4150)
# X min: tensor(-17.0291)    X max: tensor(83.6450)
# X min: tensor(0.6075)    X max: tensor(49.4480)

# X min: tensor(-71.7890)    X max: tensor(44.6420)
# X min: tensor(-19.0720)    X max: tensor(86.9870)
# X min: tensor(0.0319)    X max: tensor(52.0450)

# X min: tensor(-74.2390)    X max: tensor(47.1660)
# X min: tensor(-20.9680)    X max: tensor(90.8410)
# X min: tensor(-0.7520)    X max: tensor(55.0210)