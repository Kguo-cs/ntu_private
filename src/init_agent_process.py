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

agent_data_directory = "./waymo_data/full/training_map2_a_light"
# map_data_directory  = "./waymo_data/map2_light/training"
ouput_data_directory = "./waymo_data/full/training_map2_init10_light"

pred_init=True

os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    #filename='22c647e7272e850a.pkl'
    input_path = os.path.join(agent_data_directory, filename)

    # with open(input_path, "rb") as f:
    #     data = pickle.load(f)

    # output_path = os.path.join(ouput_data_directory, filename[:-3]+'pt')
    #
    # torch.save(data, output_path)
    #
    # return
    data=torch.load(input_path)

    # pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
    #
    av_index = torch.where(data["agent"]["role"][:, 0])[0].item()

    if av_index != len(data["agent"]["role"]) - 1:
        print(av_index,len(data["agent"]["role"]) - 1)


    # distance = torch.norm(pos - pos[av_index], dim=-1)
    #
    # # we do not believe the perception out of range of 150 meters
    # data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)
    #
    # # we do not predict vehicle too far away from ego car
    # role_train_mask = data["agent"]["role"].any(-1)
    # extra_train_mask = (distance[:, 10] < 100) & (
    #         data["agent"]["valid_mask"][:, 10 + 1:].sum(-1) >= 5
    # )
    #
    # train_mask = extra_train_mask | role_train_mask
    # if train_mask.sum() > 32:  # too many vehicle
    #     _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
    #     selected_indices = _indices[
    #         torch.randperm(_indices.size(0))[: 32 - role_train_mask.sum()]
    #     ]
    #     data["agent"]["train_mask"] = role_train_mask
    #     data["agent"]["train_mask"][selected_indices] = True
    # else:
    #     data["agent"]["train_mask"] = train_mask  # [n_agent]

    data1= HeteroData(data).cuda()

    # tokenized_map = token_processor.tokenize_map(data1)
    #
    # agent_path = os.path.join(agent_directory, filename)
    #
    # with open(agent_path, "rb") as f:
    #     data = pickle.load(f)
    #
    # for key in tokenized_map.keys():
    #     tokenized_map[key] = tokenized_map[key].cpu()
    #
    # data["tokenized_map"]=tokenized_map
    # data["tokenized_map"]['num_nodes']=len(tokenized_map["type"])

    start_idx=10

    agent = data1["agent"]

    valid = agent["valid_mask"][:,start_idx]  # [n_agent, n_step]
    heading = agent["heading"] [:,start_idx]  ## [n_agent, n_step]
    pos = agent["position"][..., :2] [:,start_idx]# # [n_agent, n_step, 2]
    vel = agent["velocity"]  [:,start_idx] ## [n_agent, n_step, 2]
    shape = agent["shape"]
    type = agent["type"]

    ego_traj=agent["position"][av_index,start_idx+1:start_idx+11, :2].contiguous()

    tokenized_agent={}

    tokenized_agent["initial_heading"] = heading[valid]  # [n_agent, n_step]
    tokenized_agent["initial_pos"] = pos[valid]  # [n_agent, n_step, 2]
    tokenized_agent["initial_vel"] = vel[valid]  # [n_agent, n_step, 2]
    tokenized_agent["initial_shape"]= shape[valid]
    tokenized_agent["initial_type"] = type[valid]
    tokenized_agent["ego_traj"] = ego_traj

    for key in tokenized_agent.keys():
        tokenized_agent[key] = tokenized_agent[key].cpu()

    tokenized_agent["num_nodes"]=len(tokenized_agent["initial_heading"])

    data["tokenized_agent"]=tokenized_agent#data["tokenized_agent"]

    del data['agent']

    output_path = os.path.join(ouput_data_directory, filename[:-3]+'pt')

    torch.save(data, output_path)

    # output_path = os.path.join(ouput_data_directory, filename)

    # Save the tokenized data
    # with open(output_path, "wb") as f:
    #     pickle.dump(data, f)


# if __name__ == "__main__":
files = os.listdir(agent_data_directory)#[ 357675:]

for file in tqdm(files):
    process_file(file)

   # break

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