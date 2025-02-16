import numpy as np

# Load the .npz file
data = np.load("HDFS_v1/preprocessed/HDFS.npz", allow_pickle=True)

# Print the names of the arrays stored in the file
print(data.files)

# To access a specific array, you can do:
for key in data.files:
    print(f"{key}: {data[key]}")