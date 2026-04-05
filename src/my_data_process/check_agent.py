import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "../waymo_data/full/validation/"
raw_data= "../waymo_data/full/validation_map2light/"

output_path = "../waymo_data/full/validation_pt/"


# data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/validation_map2/"
# raw_data= "/home/ke/code/catk/src/waymo_data/map/validation/"


files = os.listdir(data_directory)

data_dict = {}

# os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):

    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    #data=torch.load(input_path,weights_only=False)


    input_path1 = os.path.join(raw_data, filename[:-3]+'pt')

    data1=torch.load(input_path1,weights_only=False)


    for key in ['map_save','pt_token','agent']:
        for key1 in data1[key].keys():
            if key1 in data[key]:
                if type(data[key][key1]) == torch.Tensor:
                    # print(key1)
                    if not torch.all(data1[key][key1]==data[key][key1]) :
                        print(filename,key1)
#

