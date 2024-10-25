import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration as BlipModel, BlipProcessor


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super().__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DeFakeModel(nn.Module):
    def __init__(
        self, 
        clf: nn.Module, 
        blip_model: BlipModel,
        blip_processor: BlipProcessor,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
    ):
        super().__init__()
        self.clf = clf
        self.blip_model = blip_model
        self.clip_model = clip_model
        self.blip_processor = blip_processor
        self.clip_processor = clip_processor

        self.clip_vision = clip_model.vision_model
        self.clip_vision_proj = clip_model.visual_projection
        self.clip_text = clip_model.text_model
        self.clip_text_proj = clip_model.text_projection

    @torch.inference_mode()
    def generate_caption(self, img):
        device = img.device
        inputs = self.blip_processor(img, return_tensors="pt", max_length=60, do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption, device
    
    @torch.inference_mode()
    def get_visual_embed(self, img):
        device = img.device
        inputs = self.clip_processor(images=img, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        visual_embed = self.clip_vision(inputs["pixel_values"]).pooler_output
        visual_embed = self.clip_vision_proj(visual_embed)
        return visual_embed
    
    @torch.inference_mode()
    def get_text_embed(self, text, device):
        inputs = self.clip_processor(text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        text_embed = self.clip_text(inputs["input_ids"]).pooler_output
        text_embed = self.clip_text_proj(text_embed)
        return text_embed

    def forward(self, x: torch.Tensor):
        assert x.shape[0] == 1
        x = x.clamp(0, 1)
        caption = self.generate_caption(x)
        visual_embed = self.get_visual_embed(x)
        text_embed = self.get_text_embed(caption)
        embed = torch.cat([visual_embed, text_embed], dim=1).float()
        logits = self.clf(embed)
        return logits
    
    def forward_intermediate(self, x: torch.Tensor):
        assert x.shape[0] == 1
        x = x.clamp(0, 1)
        caption, device = self.generate_caption(x)
        visual_embed = self.get_visual_embed(x)
        text_embed = self.get_text_embed(caption, device)
        embed = torch.cat([visual_embed, text_embed], dim=1).float()
        logits = self.clf(embed)
        return logits, embed


