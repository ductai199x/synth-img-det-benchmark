import torch.nn as nn


class PatchForModel(nn.Module):
    def __init__(self, patch_clf: nn.Module):
        super().__init__()
        self.patch_clf = patch_clf

    def forward(self, x):
        patch_logits = self.patch_clf(x)
        logits = patch_logits.mean((-1, -2)).softmax(1)[:, 0].unsqueeze(1)
        return logits

    def forward_intermediate(self, x):
        patch_logits = self.patch_clf(x)
        embeds = patch_logits.softmax(-1)
        logits = patch_logits.mean((-1, -2)).softmax(1)[:, 0].unsqueeze(1)
        return logits, embeds
