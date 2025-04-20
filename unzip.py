import zipfile
import os



# zip_path = "./src/waymo_data/full/validation_tfrecords_splitted.zip"
# extract_to = "validation_tfrecords_splitted"  # Folder to extract contents
#
# # Create the folder if it doesn't exist
# os.makedirs(extract_to, exist_ok=True)
#
# # Extract the zip
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_to)

# print(f"✅ Extracted {zip_path} to {extract_to}")

zip_path = "./src/waymo_data/full/training.zip"
extract_to = "./src/waymo_data/full"  # Folder to extract contents

# Create the folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
