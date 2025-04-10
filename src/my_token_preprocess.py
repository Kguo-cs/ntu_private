import multiprocessing
import os
import pickle
from src.smart.tokens.my_token_processor import TokenProcessor
from tqdm import tqdm


data_directory="/home/ke/code/catk/src/waymo_data/full/training/"
token_processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        map_token_sampling={"num_k":1,"temp":1.0},
        agent_token_sampling={"num_k":1,"temp":1.0}
)

token_data_directory="/home/ke/code/catk/src/waymo_data/full/training1/"

token_processor.eval()

for file in tqdm(os.listdir(data_directory)[:200]):
    with open(data_directory+file, "rb") as f:
        data = pickle.load(f)

        #data["pt_token"]["batch"]=torch.zeros_like(data["pt_token"]["type"])

        #data["agent"]["batch"]=torch.zeros_like(data["agent"]["id"])

        if 'tokenized_map' not in data.keys():
            tokenized_map, tokenized_agent =token_processor(data)

            del tokenized_agent["gt_idx"]  # Removes key "b"
            del tokenized_agent["gt_pos"]  # Removes key "b"
            del tokenized_agent["gt_heading"]  # Removes key "b"
            del tokenized_agent['token_agent_shape']

            data_dict={"tokenized_map":tokenized_map,"tokenized_agent":tokenized_agent}

            # Save the dictionary to a file
            with open(token_data_directory+file, "wb") as f:
                pickle.dump(data_dict, f)
