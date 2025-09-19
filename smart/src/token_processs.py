import multiprocessing
import os
import pickle

from sympy.physics.units import current
from tqdm import tqdm
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
import sys

torch.set_float32_matmul_precision("high")


sys.path.append('/home/ke/code/catk')


from src.smart.tokens.token_processor import TokenProcessor

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
).cuda()
token_processor.eval()

agent_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_a/"
ouput_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_smart/"

os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    pos = data["agent"]["position"]
    av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
    distance = torch.norm(pos - pos[av_index], dim=-1)

    # we do not believe the perception out of range of 150 meters
    data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

    # we do not predict vehicle too far away from ego car
    role_train_mask = data["agent"]["role"].any(-1)
    extra_train_mask = (distance[:, 10] < 100) & (
            data["agent"]["valid_mask"][:, 10 + 1:].sum(-1) >= 5
    )

    train_mask = extra_train_mask | role_train_mask
    if train_mask.sum() > 32:  # too many vehicle
        _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
        selected_indices = _indices[
            torch.randperm(_indices.size(0))[: 32 - role_train_mask.sum()]
        ]
        data["agent"]["train_mask"] = role_train_mask
        data["agent"]["train_mask"][selected_indices] = True
    else:
        data["agent"]["train_mask"] = train_mask  # [n_agent]

    data1= HeteroData(data).cuda()

    tokenized_map = token_processor.tokenize_map(data1)

    for key in ['token_idx']:
        tokenized_map[key] = tokenized_map[key].to(torch.int16)

    for key in ["type","pl_type","light_type"]:
        tokenized_map[key] = tokenized_map[key].to(torch.int8)

    for key in tokenized_map.keys():
        tokenized_map[key] = tokenized_map[key].cpu()

    output_data={}

    output_data["tokenized_map"]=tokenized_map
    output_data["tokenized_map"]['num_nodes']=len(tokenized_map["type"])

    tokenized_agent = token_processor.tokenize_agent(data1)


    #del tokenized_agent["num_graphs"]
    del tokenized_agent['gt_idx']
    del tokenized_agent['gt_pos']
    del tokenized_agent['gt_heading']
    del tokenized_agent['gt_z_raw']

    tokenized_agent["train_mask"]=data["agent"]["train_mask"]

    for key in tokenized_agent.keys():
        tokenized_agent[key] = tokenized_agent[key].cpu()

    tokenized_agent["sampled_idx"]= tokenized_agent["sampled_idx"].to(torch.int16)

    tokenized_agent["num_nodes"]=len(tokenized_agent["sampled_idx"])

    output_data["tokenized_agent"]=tokenized_agent

    output_path = os.path.join(ouput_data_directory, filename)

    # Save the tokenized data
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)


if __name__ == "__main__":
    files = os.listdir(agent_data_directory)

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