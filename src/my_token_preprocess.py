import multiprocessing
import os
import pickle
from tqdm import tqdm
from src.smart.tokens.my_token_processor import TokenProcessor
import torch
import datetime

# Initialize the token processor once globally
token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
)
token_processor.eval()

# Set paths
token_data_directory = "/home/ke/code/catk/src/waymo_data/full/training_token/"
data_directory = "/home/ke/code/catk/src/waymo_data/full/training/"

# Worker function
def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    output_path = os.path.join(token_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    pos = data["agent"]["position"]
    av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
    distance = torch.norm(pos - pos[av_index], dim=-1)

    # we do not believe the perception out of range of 150 meters
    data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

    # if 'gt_pos_raw' in data:
    tokenized_map, tokenized_agent = token_processor(data)

    # Remove unnecessary keys
    tokenized_agent.pop('gt_pos_raw', None)
    tokenized_agent.pop("gt_head_raw", None)
    tokenized_agent.pop("gt_valid_raw", None)
    tokenized_agent.pop('gt_z_raw', None)
    tokenized_agent.pop('gt_idx', None)
    tokenized_agent.pop('gt_heading', None)
    tokenized_agent.pop('gt_pos', None)

    data_dict = {"tokenized_map": tokenized_map, "tokenized_agent": tokenized_agent}

    # Save the tokenized data
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    files = os.listdir(data_directory)#[150000:]#:]#

    for file in tqdm(files):
        process_file(file)
        # file = os.path.join(token_data_directory, filename)
        #
        # timestamp = os.path.getmtime(file)
        # modified_time = datetime.datetime.fromtimestamp(timestamp)
        #
        # print("Last modified time:", modified_time)

    # # Use tqdm inside multiprocessing with a wrapper
    # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(process_file, files), total=len(files)))

