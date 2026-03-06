import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "/home/ke/code/sim/src/waymo_data/full/validation_map2light"
raw_data= "/home/ke/code/sim/src/waymo_data/map_lane/validation/"
output_path = "/home/ke/code/sim/src/waymo_data/full/validation_lane/"


# data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/validation_edge1_light/"
# raw_data= "/home/ke/code/catk/src/waymo_data/map1_10/validation/"


files = os.listdir(data_directory)

data_dict = {}

os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    # with open(input_path, "rb") as f:
    #     data = pickle.load(f)

    data=torch.load(input_path)

    # data=torch.load(input_path)

    # del data['tokenized_map']['light_type']
    # del data['tokenized_map']['pl_type']

    input_path1 = os.path.join(raw_data, filename)

    # with open(input_path1, "rb") as f:
    #     data1 = pickle.load(f)
    data1 = torch.load(input_path1)


    #del data["light"]
    # data['tokenized_agent']["col_mask"]=data1['tokenized_agent']["col_mask"]

    data["map_save"]=data1["map_save"]
    data["pt_token"]=data1["pt_token"]

    output_file = output_path + filename

    torch.save(data, output_file)

    # with open(output_file, "wb") as f:
    #     pickle.dump(data, f)



