import pickle
from  tqdm import tqdm
import os

with open("./waymo_data/full/trainingroute.pkl", "rb") as f:
    training_route = pickle.load(f)



data_directory = "./waymo_data/full/training_light/"

files = os.listdir(data_directory)


for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    route_idx=training_route[filename[:-4]]
    data['tokenized_agent']['route_idx']=route_idx

    with open(input_path, "wb") as f:
        pickle.dump(data, f)
