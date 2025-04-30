# import pickle
# import os
# from  tqdm import tqdm
#
#
# data_directory = "./waymo_data/full/training_token/"
#
#
# files = os.listdir(data_directory)
#
# for filename in tqdm(files):
#     input_path = os.path.join(data_directory, filename)
#
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#
#     for key in data["tokenized_map"].keys():
#         data["tokenized_map"][key]=data["tokenized_map"][key].cpu()
#
#     for key in data["tokenized_agent"].keys():
#         data["tokenized_agent"][key]=data["tokenized_agent"][key].cpu()
#
#     # Save the tokenized data
#     with open("./waymo_data/full/training/"+filename, "wb") as f:
#         pickle.dump(data, f)
import pickle
import os
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count

data_directory = "./waymo_data/full/training_token/"
output_directory = "./waymo_data/full/training/"
os.makedirs(output_directory, exist_ok=True)

files = os.listdir(data_directory)[300000:]

def process_file(filename):
    input_path = os.path.join(data_directory, filename)
    output_path = os.path.join(output_directory, filename)

    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        for key in data["tokenized_map"]:
            if isinstance(data["tokenized_map"][key], torch.Tensor):
                data["tokenized_map"][key] = data["tokenized_map"][key].cpu()

        for key in data["tokenized_agent"]:
            if isinstance(data["tokenized_agent"][key], torch.Tensor):
                data["tokenized_agent"][key] = data["tokenized_agent"][key].cpu()

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        return filename
    except Exception as e:
        return f"Error with {filename}: {str(e)}"

if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, files), total=len(files)))

    # Optional: print any errors
    errors = [r for r in results if isinstance(r, str) and r.startswith("Error")]
    if errors:
        print("\nErrors encountered:")
        for e in errors:
            print(e)
