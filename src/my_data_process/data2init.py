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
    agent_token_sampling={"num_k": 1, "temp": 1.0}
).cuda()
token_processor.eval()

# Set paths

agent_data_directory = "./waymo_data/scenario_dreamer_data"
map_data_directory  = "./waymo_data/full/training_map2_init5"
ouput_data_directory = "./waymo_data/full/training_scene_init5"

pred_init=True

os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)
    data=torch.load(input_path)

    data= HeteroData(data).cuda()

    data.num_graphs=1
    data["agent"]["batch"]=torch.zeros_like(data["agent"]["type"])

    tokenized_agent = token_processor.tokenize_agent(data)

    tokenized_agent["sampled_idx"]=tokenized_agent["sampled_idx"].long()

    shapes, all_tokens, final_tokens = token_processor._get_agent_tokens(tokenized_agent["type"])
    tokenized_agent["token_agent_shape"] = shapes
    tokenized_agent["token_traj_all"] = all_tokens
    tokenized_agent["token_traj"] = final_tokens

    token_processor.get_init(tokenized_agent)

    map_path = os.path.join(map_data_directory, filename)

    data2=torch.load(map_path)


    for key in data2["tokenized_agent"].keys():
        if key != "num_nodes":
           # print(torch.all(data2["tokenized_agent"][key]==tokenized_agent[key].cpu()),key)
            data2["tokenized_agent"][key]=tokenized_agent[key].cpu()

    output_path = os.path.join(ouput_data_directory, filename)

    torch.save(data2, output_path)



# if __name__ == "__main__":
files = os.listdir(agent_data_directory)

for file in tqdm(files):
    process_file(file)