import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "../waymo_data/full/training_map2_03_light"
raw_data= "../waymo_data/full/training_map2_init0_light"
output_path = "../waymo_data/full/training_map2_init0_idx/"

raw_data1= "../waymo_data/full/training_map2_a_light"

# data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/validation_edge1_light/"
# raw_data= "/home/ke/code/catk/src/waymo_data/map1_10/validation/"


files = os.listdir(data_directory)[344770:]

data_dict = {}

os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)

    data=torch.load(input_path)

    # del data['tokenized_map']['light_type']
    # del data['tokenized_map']['pl_type']

    input_path1 = os.path.join(raw_data, filename[:-3]+'pt')

    # with open(input_path1, "rb") as f:
    #     data1 = pickle.load(f)
    input_path2 = os.path.join(raw_data1, filename[:-3]+'.pt')

    data2 = torch.load(input_path2)

    valid_mask0=data2["agent"]['valid_mask'][:,0]

    data1 = torch.load(input_path1)

    data1['tokenized_agent']["sampled_pos"]=data['tokenized_agent']["sampled_pos"][:,0][valid_mask0]
    data1['tokenized_agent']["sampled_heading"]=data['tokenized_agent']["sampled_heading"][:,0][valid_mask0]
    data1['tokenized_agent']["sampled_idx"]=data['tokenized_agent']["sampled_idx"][:,:2][valid_mask0]

    output_file = output_path + filename

    torch.save(data1, output_file)

    # with open(output_file, "wb") as f:
    #     pickle.dump(data, f)



