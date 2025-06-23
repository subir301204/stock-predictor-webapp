import os
import shutil
import subprocess
import zipfile
import platform

def setup_kaggle():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    src_path = "kaggle.json"
    if not os.path.exists(src_path):
        raise FileNotFoundError("Please place your kaggle.json in the root folder")
    
    dest_path = os.path.join(kaggle_dir, "kaggle.json")
    shutil.copy(src_path, dest_path)
    
    # Set permissions: 600 (owner read/write), skip on Windows
    if platform.system() != "Windows":
        os.chmod(dest_path, 0o600)

def download_dataset():
    print("Downloading dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "rohanrao/nifty50-stock-market-data"], check=True)
    
    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile("nifty50-stock-market-data.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    
    os.remove("nifty50-stock-market-data.zip")
    print("Download complete! Please rename your desired CSV to data/historical.csv")

if __name__ == "__main__":
    setup_kaggle()
    download_dataset()
