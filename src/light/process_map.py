import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


# data_directory = "/home/ke/code/catk/src/waymo_data/full/training_light_inter10/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/training_inter10_map/"
# raw_data= "/home/ke/code/catk/src/waymo_data/new/training/"


data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
output_path = "/home/ke/code/catk/src/waymo_data/full/validation_map/"
raw_data= "/home/ke/code/catk/src/waymo_data/new/validation/"


files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    del data["tokenized_map"]

    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    data["tokenized_map"]=data1["tokenized_map"]

    output_file = output_path + filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)



