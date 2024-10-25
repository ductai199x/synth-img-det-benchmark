import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from .srec import SrecModel


class SrecTrainWrapper(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model = SrecModel()
        self.training_config = training_config

        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        if x.max() <= 1:
            x = x * 255
        bits = self.forward(x)
        loss = bits.get_total_bpsp(x.numel())
        self.log("train_bpsp", loss.item(), prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        for key in bits.get_keys():
            if "rounding" in key or "codes" in key:
                continue
            self.log(f"train_{key}_self_bpsp", bits.get_self_bpsp(key).item(), prog_bar=False, sync_dist=True)
            self.log(f"train_{key}_scaled_bpsp", bits.get_scaled_bpsp(key, x.numel()).item(), prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if x.max() <= 1:
            x = x * 255
        bits = self.forward(x)
        loss = bits.get_total_bpsp(x.numel())
        self.log("val_bpsp", loss.item(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        for key in bits.get_keys():
            if "rounding" in key or "codes" in key:
                continue
            self.log(f"val_{key}_self_bpsp", bits.get_self_bpsp(key).item(), prog_bar=False, sync_dist=True)
            self.log(f"val_{key}_scaled_bpsp", bits.get_scaled_bpsp(key, x.numel()).item(), prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config["lr"],
            weight_decay=self.training_config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.training_config["decay_steps"], self.training_config["decay_rate"])
        return [optimizer], [scheduler]
