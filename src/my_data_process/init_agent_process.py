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

agent_data_directory = "./waymo_data/full/training_map2_03_light"
# map_data_directory  = "./waymo_data/map2_light/training"
ouput_data_directory = "./waymo_data/full/training_map2_init5"

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

    # if av_index != len(data["agent"]["role"]) - 1:
    #     print(av_index,len(data["agent"]["role"]) - 1)

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
    if agent_data_directory=="training_map2_a_light":
        agent = data1["agent"]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()

        valid = agent["valid_mask"]  # [n_agent, n_step]
        heading = agent["heading"]   ## [n_agent, n_step]
        pos = agent["position"][..., :2].contiguous()  # # [n_agent, n_step, 2]
        vel = agent["velocity"]   ## [n_agent, n_step, 2]



        start_idx=10

        heading = token_processor._clean_heading(valid[:,:start_idx+11], heading[:,:start_idx+11])
        # ! extrapolate to previous 5th step.
        valid, pos, heading, vel = token_processor._extrapolate_agent_to_prev_token_step(
            valid[:,:start_idx+11], pos[:,:start_idx+11], heading[:,:start_idx+11], vel[:,:start_idx+11]
        )


        ego_traj=pos[av_index,start_idx+1:start_idx+11].contiguous()
        #valid = valid[:,start_idx]  # [n_agent, n_step]
        # heading = heading [:,start_idx]  ## [n_agent, n_step]
        # pos = pos [:,start_idx]# # [n_agent, n_step, 2]
        #vel = vel  [:,start_idx] ## [n_agent, n_step, 2]
        shape = agent["shape"]
        type = agent["type"]

        tokenized_agent={}

        tokenized_agent["initial_heading"] = heading [:,start_idx]#[valid]  # [n_agent, n_step]
        tokenized_agent["initial_pos"] = pos [:,start_idx]#[valid]  # [n_agent, n_step, 2]
        tokenized_agent["prev_pos"] = pos[:,start_idx-5] # [valid]  # [n_agent, n_step, 2]
        tokenized_agent["prev_heading"] = heading[:,start_idx-5] # [valid]  # [n_agent, n_step, 2]

        tokenized_agent["initial_shape"]= shape#[valid]
        tokenized_agent["initial_type"] = type#[valid]
        tokenized_agent["ego_traj"] = ego_traj

        for key in tokenized_agent.keys():
            tokenized_agent[key] = tokenized_agent[key].cpu()

        tokenized_agent["num_nodes"]=len(tokenized_agent["initial_heading"])

        data["tokenized_agent"]=tokenized_agent#data["tokenized_agent"]

        del data['agent']

    else:
        tokenized_agent=data1["tokenized_agent"]

        ego_mask = torch.zeros_like(tokenized_agent['pred_mask']).to(torch.bool)
        ego_mask[-1]=True

        tokenized_agent["ego_mask"]=ego_mask
        agent_shape, token_traj_all, token_traj = token_processor._get_agent_shape_and_token_traj(
            tokenized_agent["type"]
        )

        tokenized_agent['token_traj_all']=token_traj_all

        tokenized_agent["sampled_idx"]=tokenized_agent["sampled_idx"].long()


        token_processor.get_init(tokenized_agent)

        init_tokenized_agent={}

        for key in ["initial_shape", "initial_pos", "initial_heading", "ego_pos2","ego_heading2","local_vel","type"]:
            init_tokenized_agent[key]=tokenized_agent[key]


        init_tokenized_agent["ego_pos2"] =init_tokenized_agent["ego_pos2"].repeat(len(init_tokenized_agent["initial_heading"]),1,1)
        init_tokenized_agent["ego_heading2"] =init_tokenized_agent["ego_heading2"].repeat(len(init_tokenized_agent["initial_heading"]),1)

        for key in init_tokenized_agent.keys():
            init_tokenized_agent[key] = init_tokenized_agent[key].cpu()
        init_tokenized_agent["num_nodes"]=len(init_tokenized_agent["initial_heading"])

        data["tokenized_agent"]=init_tokenized_agent

    output_path = os.path.join(ouput_data_directory, filename[:-3]+'.pt')

    torch.save(data, output_path)



# if __name__ == "__main__":
files = os.listdir(agent_data_directory)#[220000:]#[ 357675:]

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