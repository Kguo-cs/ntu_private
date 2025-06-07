import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "/home/ke/code/catk/src/waymo_data/full/training/"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_light_inter10/"

raw_data="/home/ke/code/catk/src/waymo_data/full/training_inter10/"

files = os.listdir(data_directory)

data_dict={}

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    tokenized_light=data["light"]
    
    map_tensor=torch.tensor([3,4,0,1,2]).to(tokenized_light["light_idx"].device)#

    light_idx=map_tensor[tokenized_light["light_idx"].long()]#
    # "LANE_STATE_STOP",
    # "LANE_STATE_GO",
    # "LANE_STATE_CAUTION",
    # "NO_LANE_STATE",
    # "LANE_STATE_UNKNOWN",

    # light_idx=self.token_processor.light_token_last[light_idx]

    #light_mask=light_idx<3

    #light_pred_mask=light_mask.all(-1)#torch.ones_like(light_idx[:,0]).to(torch.bool)

    tokenized_agent={}

    tokenized_agent["light_idx"]=light_idx.to(torch.int8)#[light_pred_mask]
    #tokenized_agent["light_valid_mask"]=light_mask[light_pred_mask]
    tokenized_agent["pos_lg"]=tokenized_light["light_pos"]#[light_pred_mask]

    light_polyline=tokenized_light["light_orient"]
    light_orient=torch.atan2(light_polyline[:,-1],light_polyline[:,-2])
    tokenized_agent["orient_lg"]=light_orient#[light_pred_mask]

    data1["tokenized_light"]=tokenized_agent
    data1["tokenized_light"]["num_nodes"]=len(tokenized_agent["orient_lg"])

    #del data["light"]
    #
    # position=data1["tokenized_map"]["position"]
    #
    # if data1['tokenized_map']['num_nodes']>1000:
    #
    #     centering=torch.mean(position,dim=0)
    #
    #     dist=torch.linalg.norm(position-centering,dim=-1)
    #
    #     sort_idx=torch.argsort(dist)
    #
    #     mask=sort_idx<1000
    #
    #     data1["tokenized_map"]["position"]=data1["tokenized_map"]["position"][mask]
    #     data1["tokenized_map"]["orientation"]=data1["tokenized_map"]["orientation"][mask]
    #     data1["tokenized_map"]['token_idx']=data1["tokenized_map"]['token_idx'][mask]
    #     data1["tokenized_map"]['type']=data1["tokenized_map"]['type'][mask]
    #     data1["tokenized_map"]['pl_type']=data1["tokenized_map"]['pl_type'][mask]
    #     data1["tokenized_map"]['light_type']=data1["tokenized_map"]['light_type'][mask]
    #     data1['tokenized_map']['num_nodes']=len(data1["tokenized_map"]["position"])

    output_file=output_path+filename

    with open(output_file, "wb") as f:
        pickle.dump(data1, f)



