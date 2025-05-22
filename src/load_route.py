import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData





data_directory = "/home/ke/code/catk/src/waymo_data/full/trainingroute/"
output_path = "/home/ke/code/catk/src/waymo_data/full/trainingroute.pkl"

files = os.listdir(data_directory)

data_dict={}

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    diff_dir=data["diff_dir"] % (2 * torch.pi)

    diff_dir=diff_dir/(2 * torch.pi/100)

    diff_mask=torch.isnan(diff_dir)

    diff_dir=diff_dir.to(torch.int8)

    diff_dir[diff_mask]=-1

    data_dict[filename[:-4]] = diff_dir

with open(output_path, "wb") as f:
    pickle.dump(data_dict, f)

