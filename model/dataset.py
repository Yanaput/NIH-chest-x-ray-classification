from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS: Sequence[str] = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

class NIHDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform = None,
        use_cached_dir: Optional[str | Path] = None,
        image_col: str = "image_path"
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_col = image_col
        self.cached_dir = Path(use_cached_dir) if use_cached_dir else None

        missing_cols = [c for c in LABELS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns in {csv_path}: {missing_cols}")
        if image_col not in self.df.columns:
            raise ValueError(f"Missing '{image_col}' column in {csv_path}")
        

    def __len__(self) -> int:
        return len(self.df)
    

    def __getitem__(self, index):
        row = self.df.iloc[index]
        p = Path(row[self.image_col])
        if self.cached_dir:
            p = self.cached_dir / p.name

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {p}")

        img = img[..., None]  # H x W x 1
        if self.transform is not None:
            img = self.transform(image=img)["image"]  # -> Tensor [1,H,W] float32

        y = row[LABELS].astype("float32").to_numpy()
        y = torch.from_numpy(y)
            
        return {"image": img, "target": y, "path": str(p)}
    

    @staticmethod
    def compute_pos_weight(csv_path: str | Path) -> torch.Tensor:
        """
        pos_weight = (N - P) / P per class, computed on this CSV.
        """
        df = pd.read_csv(csv_path)
        N = len(df)
        P = df[LABELS].sum().to_numpy(dtype="float64")
        P = np.clip(P, 1.0, None)
        pos_weight = (N - P) / P
        return torch.tensor(pos_weight, dtype=torch.float32)