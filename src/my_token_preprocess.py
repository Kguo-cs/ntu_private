import multiprocessing
import os
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from src.data_preprocess import decode_tracks_from_proto,decode_map_features_from_proto,decode_dynamic_map_states_from_proto,process_dynamic_map,get_map_features,get_agent_features,_polygon_types,_polygon_light_type
import matplotlib.pyplot as plt
from src.smart.tokens.my_token_processor import TokenProcessor



data_directory="/home/ke/code/catk/src/waymo_data/full/all_training/"
token_processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        map_token_sampling={"num_k":1,"temp":1.0},
        agent_token_sampling={"num_k":1,"temp":1.0}
)

token_data_directory="/home/ke/code/catk/src/waymo_data/full/training/"

token_processor.eval()

for file in os.listdir(data_directory):
    with open(data_directory+file, "rb") as f:
        data = pickle.load(f)

        #data["pt_token"]["batch"]=torch.zeros_like(data["pt_token"]["type"])

        #data["agent"]["batch"]=torch.zeros_like(data["agent"]["id"])

        tokenized_map, tokenized_agent =token_processor(data)

        del tokenized_agent["gt_idx"]  # Removes key "b"
        del tokenized_agent["gt_pos"]  # Removes key "b"
        del tokenized_agent["gt_heading"]  # Removes key "b"

        data_dict={"tokenized_map":tokenized_map,"tokenized_agent":tokenized_agent}

        # Save the dictionary to a file
        with open(token_data_directory+file, "wb") as f:
            pickle.dump(data_dict, f)
