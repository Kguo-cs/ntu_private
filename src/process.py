import multiprocessing
import os
import pickle
from tqdm import tqdm
from src.smart.tokens.my_token_processor import TokenProcessor
import torch

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
data_directory = "/home/ke/code/catk/src/waymo_data/full/training_token/"

# Worker function
def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    output_path = os.path.join(token_data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    tokenized_map, tokenized_agent= data["tokenized_map"],data["tokenized_agent"]
    if 'gt_pos_raw'  in tokenized_agent.keys():

        # Remove unnecessary keys
        tokenized_agent.pop('gt_pos_raw', None)
        tokenized_agent.pop("gt_head_raw", None)
        tokenized_agent.pop("gt_valid_raw", None)
        tokenized_agent.pop('gt_z_raw', None)

        tokenized_map["type"]=tokenized_map["type"].to(torch.uint8)
        tokenized_map["pl_type"]=tokenized_map["pl_type"].to(torch.uint8)
        tokenized_map["light_type"]=tokenized_map["light_type"].to(torch.uint8)
        tokenized_map["token_idx"]=tokenized_map["token_idx"].to(torch.int16)

        data_dict = {"tokenized_map": tokenized_map, "tokenized_agent": tokenized_agent}

        # Save the tokenized data
        with open(output_path, "wb") as f:
            pickle.dump(data_dict, f)


if __name__ == "__main__":
    files = os.listdir(data_directory)#[110000:]# [:200]

    for file in tqdm(files):
        process_file(file)

    # # Use tqdm inside multiprocessing with a wrapper
    # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(process_file, files), total=len(files)))

