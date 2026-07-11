import multiprocessing
import os
import pickle

from markdown_it.rules_inline import newline
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
    agent_token_sampling={"num_k": 1, "temp": 1.0}
).cuda()
token_processor.eval()

# Set paths

agent_data_directory = "./waymo_data/full/validation_map2light"
ouput_data_directory = "./waymo_data/full/validation_token"

pred_init=True

os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)

    data=torch.load(input_path)

    data= HeteroData(data).cuda()

    data.num_graphs=1
    data["agent"]["batch"]=torch.zeros(len(data["agent"]["type"])).long().cuda()

    tokenized_map = token_processor.tokenize_map(data)

    for key in tokenized_map.keys():
        tokenized_map[key] = tokenized_map[key].cpu()

    tokenized_agent = token_processor.tokenize_agent(data, tokenized_map)

    if pred_init:
        token_processor.get_init(tokenized_agent)

    del tokenized_map["traj_pos_local"]
    del tokenized_agent["num_graphs"]
    del tokenized_agent["traj_pos_local"]

    new_data={}

    new_data["tokenized_map"]=tokenized_map
    new_data["tokenized_map"]['num_nodes']=len(tokenized_map["type"])
    new_data["tokenized_agent"]['num_nodes']=len(tokenized_agent["type"])

    output_path = os.path.join(ouput_data_directory, filename[:-3]+'pt')

    torch.save(new_data, output_path)

files = os.listdir(agent_data_directory)

for file in tqdm(files):
    process_file(file)
