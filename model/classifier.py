import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os


LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

class NIHClassifier(pl.LightningModule):
    def __init__(self, model, pos_weight=None, test_img_out_dir=".",thesh=0.5 ,lr=1e-4):
        super().__init__()

        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.thesh = thesh
        self.test_img_out_dir = test_img_out_dir
        
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("train_loss_step", loss, prog_bar=True)
        
        self.training_step_outputs.append({
            "logits": logits.detach().cpu(),
            "targets": y.detach().cpu(),
            "loss": loss.detach().cpu()
        })
        
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        if not outputs:
            return
            
        all_logits = torch.cat([x["logits"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        metrics = self._calculate_metrics(all_logits, all_targets)
        metrics["train_loss_epoch"] = avg_loss
        
        self.log_dict(metrics, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.validation_step_outputs.append({
            "logits": logits.detach().cpu(),
            "targets": y.detach().cpu(),
            "loss": loss.detach().cpu()
        })
        
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return

        all_logits = torch.cat([x["logits"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        metrics = self._calculate_metrics(all_logits, all_targets, prefix="val_")
        metrics["val_loss_epoch"] = avg_loss
        
        self.log_dict(metrics, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.test_step_outputs.append({
            "logits": logits.detach().cpu(),
            "targets": y.detach().cpu(),
            "loss": loss.detach().cpu()
        })
        return loss

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        if not outputs:
            return

        all_logits = torch.cat([x["logits"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        metrics = self._calculate_metrics(all_logits, all_targets, prefix="test_")
        metrics["test_loss_epoch"] = avg_loss
        
        self.log_dict(metrics, prog_bar=True)
        
        # Plot ROC curves for each class
        self._plot_roc_curves(all_logits, all_targets)
        
        self.test_step_outputs.clear()

    def _calculate_metrics(self, logits, targets, prefix="train_"):
        probs = torch.sigmoid(logits).numpy()
        targets = targets.numpy()
        
        preds = (probs > self.thesh).astype(int)
    
        try:
            auc_macro = roc_auc_score(targets, probs, average="macro")
            auc_per_class = roc_auc_score(targets, probs, average=None)
        except ValueError:
            auc_macro = 0.0
            auc_per_class = np.zeros(targets.shape[1])
            
        precision = precision_score(targets, preds, average="macro", zero_division=0)
        recall = recall_score(targets, preds, average="macro", zero_division=0)
        
        metrics = {
            f"{prefix}auc": auc_macro,
            f"{prefix}precision": precision,
            f"{prefix}recall": recall
        }

        for i, class_auc in enumerate(auc_per_class):
            metrics[f"{prefix}auc_{LABELS[i]}"] = class_auc

        return metrics

    def _plot_roc_curves(self, logits, targets):
        """Plot ROC curves for each class"""
        probs = torch.sigmoid(logits).numpy()
        targets = targets.numpy()
        
        n_classes = len(LABELS)
        n_cols = 4
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for i, label in enumerate(LABELS):
            try:
                fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
                axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Random')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{label}')
                axes[i].legend(loc="lower right")
                axes[i].grid(alpha=0.3)
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           ha='center', va='center')
                axes[i].set_title(f'{label} (Error)')
        
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        os.makedirs(self.test_img_out_dir, exist_ok=True)
        plt.savefig(f'{self.test_img_out_dir}/roc_curves_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self._plot_combined_roc(probs, targets)
    
    def _plot_combined_roc(self, probs, targets):
        """Plot all ROC curves on a single plot"""

        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(LABELS)))
        
        for i, (label, color) in enumerate(zip(LABELS, colors)):
            try:
                fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{label} (AUC = {roc_auc:.3f})')
            except Exception:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=14)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(self.test_img_out_dir, exist_ok=True)
        plt.savefig(f'{self.test_img_out_dir}/roc_curves_combined.png', dpi=300, bbox_inches='tight')
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch"
            }
        }