import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


light_token_all=torch.IntTensor(np.load("/home/ke/code/catk/src/initial_tokenizer/light_cluster.npy"))#261

light_token_last=light_token_all[:,-1].long()

map_tensor=torch.tensor([3,4,0,1,2])

light_token_last=map_tensor[light_token_last].to(torch.int8)

data_directory = "/home/ke/code/catk/src/waymo_data/full/training_route1/"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_light/"

files = os.listdir(data_directory)

data_dict={}

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    del data["tokenized_map"]
    del data["tokenized_agent"]

    tokenized_light=data["tokenized_light"]
    light_idx=tokenized_light["light_idx"].long()

    light_idx=light_token_last[light_idx]

    
    data["tokenized_light"]["light_idx"] = light_idx

    output_file=output_path+filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)



