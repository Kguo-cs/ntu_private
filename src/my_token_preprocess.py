import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
torch.set_float32_matmul_precision("high")
import sys


sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')


from src.smart.tokens.my_token_processor import TokenProcessor

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2_4096.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
).cuda()
token_processor.eval()

# Set paths
token_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10_4096/"
data_directory = "/home/ke/code/catk/src/waymo_data/full/training_a/"

agent_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10/"#_light

# Worker function
def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    output_path = os.path.join(token_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)



    data= HeteroData(data).cuda()

    tokenized_map, tokenized_agent,tokenized_light = token_processor(data)

    # Remove unnecessary keys
    tokenized_agent.pop('gt_pos_raw', None)
    tokenized_agent.pop("gt_head_raw", None)
    tokenized_agent.pop("gt_valid_raw", None)
    tokenized_agent.pop('gt_z_raw', None)
    tokenized_agent.pop('gt_idx', None)
    tokenized_agent.pop('gt_heading', None)
    tokenized_agent.pop('gt_pos', None)

    tokenized_agent["sampled_idx"]=  tokenized_agent["sampled_idx"].to(torch.int16)

    for key in tokenized_agent.keys():
        tokenized_agent[key]=tokenized_agent[key].cpu()

    tokenized_agent["num_nodes"] = len(tokenized_agent["sampled_pos"])

    # tokenized_map["num_nodes"] = len(tokenized_map["position"])
    #
    # tokenized_map["token_idx"]=  tokenized_map["token_idx"].to(torch.int16)
    # for key in tokenized_map.keys():
    #     tokenized_map[key]=tokenized_map[key].cpu()

    # for key in tokenized_light.keys():
    #     tokenized_light[key]=tokenized_light[key].cpu()
    #
    # tokenized_light["num_nodes"] = len(tokenized_light["light_idx"])
    agent_path=os.path.join(agent_directory, filename)

    with open(agent_path, "rb") as f:
        data1 = pickle.load(f)

    data_dict = {"tokenized_map": data1["tokenized_map"], "tokenized_agent": tokenized_agent}#,"tokenized_light": data1["tokenized_light"]

    # Save the tokenized data
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    files = os.listdir(data_directory)

    for file in tqdm(files):
        process_file(file)

    # # Use tqdm inside multiprocessing with a wrapper
    # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(process_file, files), total=len(files)))

    #pos = data["agent"]["position"]
    #av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
    #distance = torch.norm(pos - pos[av_index], dim=-1)

    # we do not believe the perception out of range of 150 meters
    #data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)
    # data["num_graphs"]=1
    #
    # data["pt_token"]["batch"]=torch.zeros(len(pos))
    # data["agent"]["batch"]=torch.zeros(len(pos))