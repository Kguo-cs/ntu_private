import pickle
from  tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor

with open("./waymo_data/full/trainingroute.pkl", "rb") as f:
    training_route = pickle.load(f)



data_directory = "./waymo_data/full/training_light/"

files = os.listdir(data_directory)

output_directory="./waymo_data/full/training_route/"
# for filename in tqdm(files):
#     input_path = os.path.join(data_directory, filename)
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#
#     route_idx=training_route[filename[:-4]]
#     data['tokenized_agent']['route_idx']=route_idx
#
#     output_path="./waymo_data/full/training_route/"+filename
#     with open(input_path, "wb") as f:
#         pickle.dump(data, f)
def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    route_idx = training_route[filename[:-4]]
    data['tokenized_agent']['route_idx'] = route_idx

    output_path = os.path.join(output_directory, filename)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

# Parallel execution
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, files), total=len(files)))