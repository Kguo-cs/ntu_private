import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
import sys

torch.set_float32_matmul_precision("highest")

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')


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
data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10_raw_light/"
token_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10_2049medium/"

# data_directory = "/home/ke/code/catk/src/waymo_data/new/training/"
# agent_directory  = "/home/ke/code/catk/src/waymo_data/full/training_inter10_2049high/"
#
# token_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_map10_2049/"

os.makedirs(token_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    output_path = os.path.join(token_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

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

    agent = data1["tokenized_agent"]

    agent_shape, token_traj_all, token_traj = token_processor._get_agent_shape_and_token_traj(
        agent['type']
    )
    token_dict = token_processor._match_agent_token(agent["gt_valid_raw"], agent["gt_pos_raw"],
                                        agent["gt_head_raw"],
                                        agent_shape, token_traj  )


    for key in ["valid_mask","sampled_idx","sampled_pos","sampled_heading","target_global_traj","target_mask"]:
        data["tokenized_agent"][key] = token_dict[key].cpu()
    data["tokenized_agent"]["sampled_idx"]= data["tokenized_agent"]["sampled_idx"].to(torch.int16)

    del data["tokenized_agent"]['gt_pos_raw']
    del data["tokenized_agent"]['gt_head_raw']
    del data["tokenized_agent"]['gt_valid_raw']

    # Save the tokenized data
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


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