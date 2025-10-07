import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

# List of URLs
file_links = [
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_boston.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_pittsburgh.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_singapore.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_1.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_2.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_3.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_4.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_5.zip",
    "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.0/nuplan-v1.0_train_vegas_6.zip",
    "https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-maps-v1.0.zip"
]

output_dir = "nuplan_data"
os.makedirs(output_dir, exist_ok=True)


def download_and_unzip(url, output_path):
    local_filename = os.path.join(output_path, url.split("/")[-1])
    # Stream download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    # Unzip the file
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    print(f"Extracted: {local_filename}")
    # os.remove(local_filename)  # Clean up zip file


# Download and unzip all files
for link in file_links:
    download_and_unzip(link, output_dir)