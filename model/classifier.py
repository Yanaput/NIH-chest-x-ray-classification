import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score, precision_score, recall_score


LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

class NIHClassifier(pl.LightningModule):
    def __init__(self, model, pos_weight=None, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []

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
        
        # Calculate metrics
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

    def _calculate_metrics(self, logits, targets, prefix="train_"):
        probs = torch.sigmoid(logits).numpy()
        targets = targets.numpy()
        # print(probs)
        
        preds = (probs > 0.5).astype(int)
    
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
