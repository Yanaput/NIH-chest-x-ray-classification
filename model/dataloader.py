from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import lightning.pytorch as pl
from .dataset import NIHDataset


def build_transforms(img_size: int, trans_crop: int, mean: float, std: float):
    train_transforms_list = [
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(trans_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]
    train_transforms = transforms.Compose(train_transforms_list)

    val_transforms_list = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]
    val_transforms = transforms.Compose(val_transforms_list)

    return train_transforms, val_transforms

class NIHDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        test_csv: Optional[str | Path] = None,
        img_size: int = 256,
        trans_crop: int = 224,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cached_dir: Optional[str | Path] = None,
        mean: float = 0.5,
        std: float = 0.25,
    ) -> None:
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv   = Path(val_csv)
        self.test_csv  = Path(test_csv) if test_csv else None
        self.img_size  = img_size
        self.trans_crop  = trans_crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cached_dir = Path(cached_dir) if cached_dir else None
        self.mean = mean
        self.std = std

        self.train_ds: Optional[NIHDataset] = None
        self.val_ds: Optional[NIHDataset] = None
        self.test_ds: Optional[NIHDataset] = None

    def prepare_data(self) -> None:
        if not self.train_csv.exists():
            raise FileNotFoundError(f"Missing train CSV: {self.train_csv}")
        if not self.val_csv.exists():
            raise FileNotFoundError(f"Missing val CSV: {self.val_csv}")
        if self.test_csv is not None and not self.test_csv.exists():
             raise FileNotFoundError(f"Missing test CSV: {self.test_csv}")

    def setup(self, stage: Optional[str] = None) -> None:
        train_tf, val_tf = build_transforms(self.img_size, self.trans_crop, self.mean, self.std)

        if stage in (None, "fit"):
            self.train_ds = NIHDataset(self.train_csv, transform=train_tf, use_cached_dir=self.cached_dir)
            self.val_ds   = NIHDataset(self.val_csv,   transform=val_tf,   use_cached_dir=self.cached_dir)

        if stage in (None, "test") and self.test_csv is not None:
            self.test_ds  = NIHDataset(self.test_csv,  transform=val_tf,   use_cached_dir=self.cached_dir)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
             raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
             raise RuntimeError("Val dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
