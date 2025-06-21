import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np

import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
torch.set_float32_matmul_precision("high")
import sys


sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')


from src.smart.tokens.token_processor import TokenProcessor

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_444_s2_4096.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
).cuda()
token_processor.eval()

data_directory = "/home/ke/code/catk/src/waymo_data/full/validation/"
raw_data="/home/ke/code/catk/src/waymo_data/full/validation_light/"

files = os.listdir(data_directory)

data_dict={}

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)
        
    
    data["agent"]["batch"]=torch.zeros(data["agent"]["num_nodes"]).long()
    data["pt_token"]["batch"]=torch.zeros(data["pt_token"]["num_nodes"]).long()

    data["light"]["batch"]=torch.zeros(data["light"]["num_nodes"]).long()

    data= HeteroData(data).cuda()
    data.num_graphs=1

    tokenized_map, tokenized_agent = token_processor(data)


    input_path1 = os.path.join(raw_data, filename)

    with open(input_path1, "rb") as f:
        data1 = pickle.load(f)

    data1["agent"]["batch"]=torch.zeros(data1["agent"]["num_nodes"]).long()
    data1["pt_token"]["batch"]=torch.zeros(data1["pt_token"]["num_nodes"]).long()

    data1["light"]["batch"]=torch.zeros(data1["light"]["num_nodes"]).long()

    data1= HeteroData(data1).cuda()
    data1.num_graphs=1

    tokenized_map1, tokenized_agent1 = token_processor(data1)

    for key in tokenized_agent.keys():
        if key != "num_graphs" and key != "lengths_lg":
            if not torch.allclose(tokenized_agent[key],tokenized_agent1[key]):
                print("Mismatch found in key:", key)
            # print(key)
            # print(torch.all(tokenized_agent[key]==tokenized_agent1[key]), key)
    
    # for key in tokenized_map.keys():
    #     if key != "num_graphs":
    #         print(key)
    #         print(torch.all(tokenized_map[key]==tokenized_map1[key]), key)
