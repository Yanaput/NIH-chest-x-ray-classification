import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import model.resnet as Resnet
from model.densenet import DenseNet
from model.dataset import NIHDataset
from model.dataloader import NIHDataModule
from model.classifier import NIHClassifier

def main():
    parser = argparse.ArgumentParser(description="Train NIH Chest X-ray Classifier")
    parser.add_argument("--train_csv", type=str, help="Path to train CSV", default="data/train.csv")
    parser.add_argument("--val_csv", type=str, help="Path to validation CSV", default="data/val.csv")
    parser.add_argument("--test_csv", type=str, help="Path to test CSV", default="data/test.csv")
    parser.add_argument("--image_dir", type=str, help="Path to image directory (optional cache)", default="data/processed/images_all") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--trans_crop", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a quick development run")
    
    args = parser.parse_args()

    # DataModule
    dm = NIHDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        img_size=args.img_size,
        trans_crop=args.trans_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cached_dir=args.image_dir
    )
    backbone = DenseNet(num_classes=14, in_channels=1)

    try:
        pos_weight = NIHDataset.compute_pos_weight(args.train_csv)
        print("Computed positive weights for class imbalance handling.")
        print(pos_weight)
    except Exception as e:
        print(f"Warning: Could not compute positive weights: {e}")
        pos_weight = None

    model = NIHClassifier(model=backbone, pos_weight=pos_weight, lr=args.lr)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f"{type(backbone)}"+"--{epoch}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=3,
        # save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(save_dir=args.log_dir, name="nih_classifier")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=100
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()