import pickle
import os
from  tqdm import tqdm


data_directory = "./waymo_data/full/training_token/"


files = os.listdir(data_directory)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    for key in data["tokenized_map"].keys():
        data["tokenized_map"][key]=data["tokenized_map"][key].cpu()

    for key in data["tokenized_agent"].keys():
        data["tokenized_agent"][key]=data["tokenized_agent"][key].cpu()

    # Save the tokenized data
    with open("./waymo_data/full/training/"+filename, "wb") as f:
        pickle.dump(data, f)
