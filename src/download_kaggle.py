import os

def setup_kaggle():
          os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
          if not os.path.exists("kaggle.json"):
                    raise FileNotFoundError("Please place your kaggle.json in the root folder")
          os.system("cp kaggle.json ~/.kaggle/")
          os.system("chmod 600 ~/.kaggle/kaggle.json")

def download_dataset():
          print("Downloading dataset from Kaggle...")
          os.system("kaggle datasets download -d rohanrao/nifty50-stock-market-data")
          os.makedirs("data", exist_ok=True)
          os.system("unzip -o nifty50-stock-market-data.zip -d data/")
          print("Download ccomplete! Please rename your desired CSV to data/historical.csv")

if __name__ == "__main__":
          setup_kaggle()
          download_dataset()
