from pathlib import Path
import shutil
import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "/workspace/.cache/kagglehub"


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "nih_kaggle"
HANDLE = "nih-chest-xrays/data"

def main():
    cache_path = kagglehub.dataset_download(HANDLE)
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, DEST, dirs_exist_ok=True)
    print(f"Dataset copied to: {DEST}")

if __name__ == "__main__":
    main()