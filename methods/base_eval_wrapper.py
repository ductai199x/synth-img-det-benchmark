import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class BaseEvalWrapper(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        output_dir=None,
        is_save_logits=False,
        is_save_embeds=False,
    ):
        super().__init__()
        self.model = model.eval()

        self.output_dir = output_dir
        self.is_saving = output_dir is not None
        self.is_save_logits = is_save_logits
        self.is_save_embeds = is_save_embeds

        self.transforms = None
        self.save_logits_target = None
        self.save_embeds_target = None

        self.metrics()

    def metrics(self):
        raise NotImplementedError

    def save_logits_fn(self, logits, targets):
        raise NotImplementedError

    def save_embeds_fn(self, embeds, targets):
        raise NotImplementedError

    def save_results_fn(self):
        raise NotImplementedError

    def transforms_fn(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.model(self.transforms_fn(x))

    def forward_intermediate(self, x):
        return self.model.forward_intermediate(self.transforms_fn(x))

    def test_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.unsqueeze(1)
        logits, embeds = self.forward_intermediate(x)
        if logits.shape[1] == 2:
            y = torch.cat([1 - y, y], dim=1)
        self.accuracy(logits, y)
        self.auroc(logits, y)

        self.log("accuracy", self.accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("auroc", self.auroc, on_step=False, on_epoch=True, sync_dist=True)

        if self.is_saving:
            if self.is_save_logits:
                self.save_logits_fn(logits, y)
            if self.is_save_embeds:
                self.save_embeds_fn(embeds, y)

    def on_test_epoch_end(self) -> None:
        if self.save_logits_target is not None:
            try:
                self.save_logits_target.close()
            except Exception as e:
                pass
        if self.save_embeds_target is not None:
            try:
                self.save_embeds_target.close()
            except Exception as e:
                pass
        self.save_results_fn()
        return super().on_test_epoch_end()
