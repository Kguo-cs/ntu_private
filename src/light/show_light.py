import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

light_token_all=torch.IntTensor(np.load("/home/ke/code/catk/src/initial_tokenizer/light_cluster.npy"))#261

light_token_last=light_token_all[:,-1].long()

map_tensor=torch.tensor([3,4,0,1,2])

light_token_last=map_tensor[light_token_last].to(torch.int8)

data_directory = "/home/ke/code/catk/src/waymo_data/full/training_light/"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_light/"

files = os.listdir(data_directory)

data_dict={}

color=['red','green','yellow','blue','black']

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    tokenized_light=data["tokenized_light"]


    light_idx=tokenized_light["light_idx"].long()
    light_pos=tokenized_light["light_pos"]
    light_orient=tokenized_light["light_orient"]

    if len(light_idx):

        dx=np.cos(light_orient)
        dy=np.sin(light_orient)

        for t in range(18):

            for i in range(len(light_pos)):
                light_color=color[light_idx[i][t]]

                plt.arrow(light_pos[i,0], light_pos[i,1], dx[i], dy[i],color=light_color)
        
            plt.show(block=True)

            print(1)

