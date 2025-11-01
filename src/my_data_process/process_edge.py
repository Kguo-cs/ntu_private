import multiprocessing
import os
import pickle
import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
from networkx.classes import edges
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np


data_directory = "/home/ke/code/catk/src/waymo_data/edge/training/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/training_inter10_map/"
# raw_data= "/home/ke/code/catk/src/waymo_data/new/training/"


# data_directory = "/home/ke/code/catk/src/waymo_data/full/validation_light/"
# output_path = "/home/ke/code/catk/src/waymo_data/full/validation_map/"
# raw_data= "/home/ke/code/catk/src/waymo_data/new/validation/"


files = os.listdir(data_directory)

data_dict = {}

# os.makedirs(output_path, exist_ok=True)

for filename in tqdm(files):

    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    edges=data['edge']

    for edge in edges:
        plt.plot(edge[:,0], edge[:,1],'r')

    plt.show()