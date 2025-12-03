from pathlib import Path
import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import shutil
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nih_kaggle"
ENRTY_CSV = RAW / "Data_Entry_2017.csv"
ALL_IMG_DIR = ROOT / "data" / "processed" / "images_all"

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]


TRAIN_RATIO = 0.8
VAL_RATIO   = 0.2
SEED = 42


def patient_table(df) -> pd.DataFrame:
    return df.groupby("patient_id", as_index=False)[LABELS].max()


def split_patients(pat_df: pd.DataFrame, val_ratio: float, seed: int):
    X = pat_df[["patient_id"]].to_numpy()
    Y = pat_df[LABELS].to_numpy(dtype=int)

    # Split into train vs temp using iterative stratification
    # We emulate a single split with a 2-fold KFold where one fold is ~val_ratio.
    # Compute fold sizes by shuffling and cutting, but keep iterative balance via KFold trick:
    n_splits = round(1 / val_ratio)  # e.g., 1/(1-0.2)=1.25 -> 1; fallback to 2
    n_splits = max(n_splits, 2)                 # ensure at least 2 folds
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(mskf.split(X, Y))
    # Take the first split: one fold as validation, the rest as train
    train_idx, val_idx = splits[0]
    train_pat = set(pat_df.iloc[train_idx]["patient_id"])
    val_pat   = set(pat_df.iloc[val_idx]["patient_id"])

    actual_val_ratio = len(val_idx) / len(Y)
    print(f"Requested val_ratio: {val_ratio:.3f}, Actual val_ratio: {actual_val_ratio:.3f}")

    assert train_pat.isdisjoint(val_pat)
    return train_pat, val_pat

def combine_all_img():
    ALL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for p in RAW.glob("images_*/images/*"):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        d = ALL_IMG_DIR / p.name
        if d.exists(): continue
        try:
            d.symlink_to(p.resolve())
        except OSError:
            os.link(p.resolve(), d)


def main():
    # print("preprocess_label")
    combine_all_img()
    df = pd.read_csv(ENRTY_CSV)
    df = df.iloc[:, :-1]
    df["labels"] = df["Finding Labels"].fillna("").str.split("|")

    rows = []
    for _, row in df.iterrows():
        y = {k:0 for k in LABELS}
        for t in row["labels"]:
            if t == "No Finding" or t == "": continue
            if t in y: y[t] = 1
        rows.append({
            "image_path": str((ALL_IMG_DIR / row["Image Index"])),
            "image_idx": row["Image Index"],
            "patient_id": row["Patient ID"],
            **y
        })
    df = pd.DataFrame(rows)


    try:
        with open(RAW / "train_val_list.txt", "r") as f:
            entries = f.readlines()
    except Exception as e:
        print(f"An error occurred: {e}")   

    train_val = [line.strip() for line in entries]


    train_val_data = []
    test_data = []

    for _, img in df.iterrows():
        row_data = img.copy()

        if img["image_idx"] in train_val :
            train_val_data.append(row_data)
        else:
            test_data.append(row_data)

    train_val_df = pd.DataFrame(train_val_data)
    test_df = pd.DataFrame(test_data)


    train_val_df.to_csv("data/train_val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)



    required = {"patient_id", *LABELS}
    missing = required - set(df.columns)


    pat_df = patient_table(df)
    train_pat, val_pat = split_patients(pat_df, VAL_RATIO, 42)
    
    train_val_df["split"] = np.where(train_val_df["patient_id"].isin(train_pat), "train", "val")
    train_df = train_val_df[train_val_df.split == "train"].reset_index(drop=True).drop(["split"], axis=1)
    val_df   = train_val_df[train_val_df.split == "val"].reset_index(drop=True).drop(["split"], axis=1)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)

    print("Counts:", {"train": len(train_df), "val": len(val_df)})
    print("Train prevalence:", train_df[LABELS].mean().round(4).to_dict())
    print("Val prevalence:",   val_df[LABELS].mean().round(4).to_dict())

    # if input(f"Are you sure you want to delete raw data at {RAW}? (y/N) ").lower() == 'y':
    #     try:
    #         shutil.rmtree(RAW)
    #         print(f"Directory '{RAW}' and its contents removed successfully.")
    #     except OSError as e:
    #         print(f"Error: {e.filename} - {e.strerror}")
    # else:
    #     print("Deletion cancelled.")

if __name__ == "__main__":
    main()