import os
import kagglehub

def download_full_dataset():
    # 1. Define the project root and the path-storage file
    # This helps your .sh scripts find the data later
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH_FILE = os.path.join(BASE_DIR, ".data_path")

    print("📡 Initializing full CIC-IoT-2023 download via Kagglehub...")
    
    try:
        # 2. Download the full dataset (all 169 parts)
        # This will stay in the hidden ~/.cache folder to save project space
        dataset_path = kagglehub.dataset_download("akashdogra/cic-iot-2023")
        
        # 3. Save the path to a hidden file for your Shell Scripts
        with open(PATH_FILE, "w") as f:
            f.write(dataset_path)
            
        print("\n✅ Download Complete!")
        print(f"📂 Full dataset located at: {dataset_path}")
        print(f"📍 Path saved for scripts in: {PATH_FILE}")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_full_dataset()