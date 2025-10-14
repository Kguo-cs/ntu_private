import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np

#
# data_directory = "/home/ke/code/catk/src/waymo_data/full/nuplan_cross2_03_route/"
#
# files = os.listdir(data_directory)
#
# shape=[]
# type=[]
#
# for filename in tqdm(files):
#     input_path = os.path.join(data_directory, filename)
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#
#     shape.append(data['tokenized_agent']['shape'])
#     type.append(data['tokenized_agent']['type'])
#
# shape =torch.cat(shape)
# type=torch.cat(type)
# #
# torch.save(shape,'shape.pt')
# torch.save(type,'type.pt')
import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'

# Load saved tensors
shape = torch.load("shape.pt")   # [N, 3] = length, width, height
atype = torch.load("type.pt")    # [N]

# Convert to numpy
shape = shape.cpu().numpy()
atype = atype.cpu().numpy()

lengths = shape[:, 0]
widths  = shape[:, 1]
heights = shape[:, 2]

# Define bins automatically
max_val = np.max(shape)
bins = np.linspace(0, 5, 10)

# Unique types
unique_types = np.unique(atype)

fig, axes = plt.subplots(len(unique_types), 3, figsize=(15, 4 * len(unique_types)))

if len(unique_types) == 1:  # single row case
    axes = np.expand_dims(axes, axis=0)

for i, t in enumerate(unique_types):
    mask = atype == t

    # Length
    axes[i, 0].hist(lengths[mask], bins=bins, alpha=0.7, color="tab:blue", log=True)
    axes[i, 0].set_title(f"Type {t} - Length")
    axes[i, 0].set_xlabel("Length")
    axes[i, 0].set_ylabel("Log Count")

    # Width
    axes[i, 1].hist(widths[mask], bins=bins, alpha=0.7, color="tab:orange", log=True)
    axes[i, 1].set_title(f"Type {t} - Width")
    axes[i, 1].set_xlabel("Width")
    axes[i, 1].set_ylabel("Log Count")

    # Height
    axes[i, 2].hist(heights[mask], bins=bins, alpha=0.7, color="tab:green", log=True)
    axes[i, 2].set_title(f"Type {t} - Height")
    axes[i, 2].set_xlabel("Height")
    axes[i, 2].set_ylabel("Log Count")

    print(np.mean(lengths[mask]),np.mean(widths[mask]),np.mean(heights[mask]))

plt.tight_layout()
#plt.savefig("hist.png")
plt.show()
