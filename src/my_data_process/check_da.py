import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "../waymo_data/full/training_map2_a/"
raw_data= "../waymo_data/full/training_map2_all/"

output_path = "../waymo_data/full/training_map2_a/"


# data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/validation_map2/"
# raw_data= "/home/ke/code/catk/src/waymo_data/map/validation/"


files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    current_valid = data1['agent']["current_valid"]

    for key in data['agent'].keys():
        if key !='num_nodes':
            value1=data1['agent'][key][current_valid]
            value=data['agent'][key]
            if not torch.all(value1==value):
                print(key,filename)

