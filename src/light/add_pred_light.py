import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


# data_directory = "/home/ke/code/catk/src/waymo_data/full/training_light_inter10/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/training_inter10_light_col/"
# raw_data= "/home/ke/code/catk/src/waymo_data/full/training_inter10_col/"


data_directory = "/home/ke/code/catk/src/waymo_data/full/training_map2_raw/"
raw_data= "/home/ke/code/catk/src/waymo_data/full/training_a/"

output_path = "/home/ke/code/catk/src/waymo_data/full/training_map2_raw_pred/"

files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # del data['tokenized_map']['light_type']
    # del data['tokenized_map']['pl_type']

    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    # data['tokenized_agent']["col_mask"]=data1['tokenized_agent']["col_mask"]

    role_mask=data1["agent"]["role"]

    pred_mask=role_mask[:,0] | role_mask[:,2]

    data['tokenized_agent']["pred_mask"]=pred_mask

    output_file = output_path + filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)



