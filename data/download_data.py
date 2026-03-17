import os
import requests
from tqdm import tqdm

def download_dataset(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"📁 Created folder: {dest_folder}")

    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)

    print(f"⏳ Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(file_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("❌ ERROR: Something went wrong with the download.")
    else:
        print(f"✅ Successfully downloaded to: {file_path}")

if __name__ == "__main__":
    # This is the direct mirror for the 10% sampled CSV (xxsmall) 
    # useful for initial server testing.
    DATA_URL = "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CICIoT2023_xxsmall.csv"
    
    # Get the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    download_dataset(DATA_URL, DATA_DIR)