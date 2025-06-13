import numpy as np

arr=np.load("/home/ke/code/catk/src/waymo_data/light_all.npy")[:,1:].reshape(-1,5)
# Count unique rows
unique_rows = np.unique(arr, axis=0)
num_unique_rows = unique_rows.shape[0]

print("Number of different rows:", num_unique_rows)

np.save("../../initial_tokenizer/light_cluster.npy", unique_rows)