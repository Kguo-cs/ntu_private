import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "/home/ke/code/catk/src/waymo_data/full/training/"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_light/"

files = os.listdir(data_directory)

data_dict={}

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    tokenized_light=data["light"]
    
    map_tensor=torch.tensor([3,4,0,1,2]).to(tokenized_light["light_idx"].device)

    light_idx=map_tensor[tokenized_light["light_idx"].long()]

    # light_idx=self.token_processor.light_token_last[light_idx]

    light_mask=light_idx<3

    light_pred_mask=light_mask.all(-1)#torch.ones_like(light_idx[:,0]).to(torch.bool)

    tokenized_agent={}

    tokenized_agent["light_idx"]=light_idx[light_pred_mask].to(torch.int8)
    #tokenized_agent["light_valid_mask"]=light_mask[light_pred_mask]
    tokenized_agent["pos_lg"]=tokenized_light["light_pos"][light_pred_mask]

    light_polyline=tokenized_light["light_orient"]
    light_orient=torch.atan2(light_polyline[:,-1],light_polyline[:,-2])
    tokenized_agent["orient_lg"]=light_orient[light_pred_mask]

    data["tokenized_light"]=tokenized_agent
    data["tokenized_light"]["num_nodes"]=len(tokenized_agent["orient_lg"])

    del data["light"]

    output_file=output_path+filename

    with open(output_file, "wb") as f:
        pickle.dump(data, f)



