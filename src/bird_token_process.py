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


sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/ke/code/catk')
sys.path.append('/home/users/ntu/zhangshu/scratch/sim')
sys.path.append('/home/users/ntu/shanhelo/scratch/keguo_projects/sim')
sys.path.append('/mnt/d/code/sim')


from src.smart.tokens.token_bird_processor import TokenProcessor

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="first2048.pkl",
    agent_token_file="bird1024.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0},
    pred_entry=False
).cuda()
token_processor.eval()

# Set paths

agent_data_directory = "/home/ke/code/catk/src/waymo_data/full/bird_train107/"
ouput_data_directory = "/home/ke/code/catk/src/waymo_data/full/bird_train107_all1/"



os.makedirs(ouput_data_directory, exist_ok=True)

# Worker function
def process_file(filename):
    input_path = os.path.join(agent_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)


    data1= HeteroData(data).cuda()

    data1["agent"]["batch"]=torch.zeros_like( data["agent"]["valid_mask"] [:,0] ).to(torch.long).cuda()
    data1.num_graphs=1
    data1["agent"]["time"]=[data1["agent"]["time"]]

    tokenized_agent = token_processor.tokenize_agent(data1)

    del tokenized_agent['num_graphs']
    del tokenized_agent['batch']
    del tokenized_agent['shape']
    del tokenized_agent['token_agent_shape']
    del tokenized_agent['type']

    #del tokenized_agent['gt_idx']
    #del tokenized_agent['max_dist']
    #del tokenized_agent['entry_idx']
    #del tokenized_agent['entry_head_idx']
   # del tokenized_agent['reset_mask']
    del tokenized_agent['token_traj_all']

    for key in tokenized_agent.keys():
        tokenized_agent[key] = tokenized_agent[key].cpu()

    tokenized_agent["sampled_idx"]= tokenized_agent["sampled_idx"].to(torch.int16)
    tokenized_agent['abs_time']= tokenized_agent['abs_time'][:,0]
    #tokenized_agent["entry_idx"]= tokenized_agent["entry_idx"].to(torch.int16)
    #tokenized_agent["entry_head_idx"]= tokenized_agent["entry_head_idx"].to(torch.int16)

    tokenized_agent["num_nodes"]=len(tokenized_agent["sampled_idx"])

    # for key in ["valid_mask","sampled_idx","sampled_pos","sampled_heading","target_global_traj","target_mask"]:
    #     data["tokenized_agent"][key] = token_dict[key].cpu()
    # data["tokenized_agent"]["sampled_idx"]= data["tokenized_agent"]["sampled_idx"].to(torch.int16)
    #
    # del data["tokenized_agent"]['gt_pos_raw']
    # del data["tokenized_agent"]['gt_head_raw']
    # del data["tokenized_agent"]['gt_valid_raw']

    data2={}

    data2["tokenized_agent"]=tokenized_agent#data["tokenized_agent"]
    #
    #del data2["tokenized_light"]

    output_path = os.path.join(ouput_data_directory, filename)

    # Save the tokenized data
    with open(output_path, "wb") as f:
        pickle.dump(data2, f)


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