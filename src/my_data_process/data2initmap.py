import multiprocessing
import os
import pickle

from sympy.physics.units import current
from tqdm import tqdm
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
import sys

torch.set_float32_matmul_precision("highest")


sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')
sys.path.append('/home/users/ntu/shanhelo/scratch/keguo_projects/sim')
sys.path.append('/mnt/d/code/sim')
sys.path.append('/home/ke/keguo/sim')
sys.path.append('/home/guoke/sim')

from src.smart.tokens.token_processor import TokenProcessor

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0},
    pred_init=True
).cuda()
token_processor.eval()

# Set paths

agent_data_directory = "./waymo_data/full/training_sd/training"
ouput_data_directory = "./waymo_data/full/training_map2_sd"

os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)
    data=torch.load(input_path)

    data= HeteroData(data).cuda()

    data.num_graphs=1
    data["agent"]["batch"]=torch.zeros_like(data["agent"]["type"]).long()
    data["scene_timestep"]=torch.IntTensor([data["scene_timestep"]]).cuda()

    tokenized_map, tokenized_agent = token_processor.process_data(data)

    for key in tokenized_map.keys():
        tokenized_map[key] = tokenized_map[key].cpu()

    tokenized_map["num_nodes"]=len(tokenized_map["position"])

    tokenized_agent1={}

    for key in ["initial_heading", "initial_pos","local_vel","shape","type"]:
        tokenized_agent1[key] = tokenized_agent[key].cpu()

    tokenized_agent1["num_nodes"]=len(data["agent"]["type"])


    data2={}
    data2["tokenized_agent"]=tokenized_agent1
    data2["tokenized_map"]=tokenized_map

    output_path = os.path.join(ouput_data_directory, filename)

    torch.save(data2, output_path)



# if __name__ == "__main__":
files = os.listdir(agent_data_directory)[192058:]

for file in tqdm(files):
    print(file)
    process_file(file)