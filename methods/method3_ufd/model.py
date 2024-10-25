import torch.nn as nn
from transformers import CLIPModel


class UFDModel(nn.Module):
    def __init__(self, clip_model: CLIPModel, fc: nn.Linear):
        super().__init__()
        self.clip_model = clip_model
        self.clip_vision = clip_model.vision_model
        self.clip_vision_proj = clip_model.visual_projection
        self.fc = fc

    def forward(self, x):
        clip_out = self.clip_vision(x).pooler_output
        clip_out = self.clip_vision_proj(clip_out)
        return self.fc(clip_out)

    def forward_intermediate(self, x):
        clip_out = self.clip_vision(x).pooler_output
        clip_out = self.clip_vision_proj(clip_out)
        pred = self.fc(clip_out)
        return pred, clip_out
