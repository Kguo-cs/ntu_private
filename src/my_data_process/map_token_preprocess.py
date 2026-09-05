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

agent_data_directory = "./waymo_data/full/training_map2_init5"
map_data_directory  = "./waymo_data/map_all/training"
ouput_data_directory = "./waymo_data/full/training_mapall_init5"


# agent_data_directory = "/home/ke/code/sim/src/waymo_data/full/validation_map2light"
# map_data_directory  = "./waymo_data/map2_light/validation"


os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)
    # with open(input_path, "rb") as f:
    #     data = pickle.load(f)

    data=torch.load(input_path)

    #data1= HeteroData(data).cuda()

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

    # agent = data1["agent"]
    #
    # agent_shape, token_traj_all, token_traj = token_processor._get_agent_shape_and_token_traj(
    #     agent['type']
    # )
    # valid = agent["valid_mask"]  # [n_agent, n_step]
    # heading = agent["heading"]   ## [n_agent, n_step]
    # pos = agent["position"][..., :2].contiguous()  # # [n_agent, n_step, 2]
    # vel = agent["velocity"]   ## [n_agent, n_step, 2]
    #
    # heading = token_processor._clean_heading(valid, heading)
    # # ! extrapolate to previous 5th step.
    # valid, pos, heading, vel = token_processor._extrapolate_agent_to_prev_token_step(
    #     valid, pos, heading, vel
    # )
    #
    # tokenized_agent = token_processor._match_agent_token(valid, pos,
    #                                     heading,
    #                                     agent_shape, token_traj  )
    # #
    # for key in ["type", "shape"]:#
    #     tokenized_agent[key] = agent[key]
    #
    # for key in tokenized_agent.keys():
    #     tokenized_agent[key] = tokenized_agent[key].cpu()
    #
    # tokenized_agent["sampled_idx"]= tokenized_agent["sampled_idx"].to(torch.int16)
    #
    # tokenized_agent["num_nodes"]=len(tokenized_agent["sampled_idx"])

    # for key in ["valid_mask","sampled_idx","sampled_pos","sampled_heading","target_global_traj","target_mask"]:
    #     data["tokenized_agent"][key] = token_dict[key].cpu()
    # data["tokenized_agent"]["sampled_idx"]= data["tokenized_agent"]["sampled_idx"].to(torch.int16)
    #
    # del data["tokenized_agent"]['gt_pos_raw']
    # del data["tokenized_agent"]['gt_head_raw']
    # del data["tokenized_agent"]['gt_valid_raw']

    map_path = os.path.join(map_data_directory, filename)

    # with open(map_path, "rb") as f:
    #     data2 = pickle.load(f)
    data2=torch.load(map_path)

    data1=HeteroData(data2).cuda()

    tokenized_map = token_processor.tokenize_map(data1)

    for key in tokenized_map.keys():
        tokenized_map[key] = tokenized_map[key].cpu()

    data["tokenized_map"]=tokenized_map
    data["tokenized_map"]['num_nodes']=len(tokenized_map["type"])
    data["tokenized_map"]["token_idx"]=data["tokenized_map"]["token_idx"].to(torch.int16)
    # map=data2["tokenized_map"]
    #
    # # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]
    # dist = torch.sum(
    #     (token_processor.map_token_sample_pt[:, :, 1:] - map["traj_pos_local"].cuda().unsqueeze(1)) ** 2,
    #     dim=(-2, -1),
    # )  # [n_pl, n_token]
    #
    # data2["tokenized_map"]["token_idx"] = torch.argmin(dist, dim=-1).to(torch.int16).cpu()
    #
    # del data2["tokenized_map"]["traj_pos_local"]

    #data2["tokenized_agent"]=data["tokenized_agent"]
    #
    #del data2["tokenized_light"]
    # print(tokenized_map['light_type'].max())
    output_path = os.path.join(ouput_data_directory, filename)

    torch.save(data, output_path)
    # Save the tokenized data
    # with open(output_path, "wb") as f:
    #     pickle.dump(data, f)


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