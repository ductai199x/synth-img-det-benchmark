import torch
import torch.nn as nn
from .guided_diffusion.respace import SpacedDiffusion
from .guided_diffusion.unet import UNetModel
from ..common_transforms_utils import Compose, CenterCrop, Normalize


GUIDED_DIFFUSION_CONFIGS = {
    "image_size": 256,
    "num_channels": 256,
    "num_res_blocks": 2,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_head_channels": 64,
    "attention_resolutions": "32,16,8",
    "channel_mult": "",
    "dropout": 0.1,
    "class_cond": False,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_fp16": True,
    "use_new_attention_order": False,
    "learn_sigma": True,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "timestep_respacing": "ddim20",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
}
GUIDED_DIFFUSION_WEIGHTS_NAME = "256x256_diffusion_uncond"
# GUIDED_DIFFUSION_WEIGHTS_NAME = "256x256_diffusion_lsun_bedroom"


class DireModel(nn.Module):
    def __init__(self, unet: UNetModel, diffusion: SpacedDiffusion, classifier: nn.Module, diff_steps=1000):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.classifier = classifier
        self.diff_steps = diff_steps

        self.reverse_fn = self.diffusion.ddim_reverse_sample_loop
        self.sample_fn = self.diffusion.ddim_sample_loop
        self.unet_dtype = self.unet.dtype

        self.clf_transforms = Compose(
            [
                CenterCrop((224, 224)), 
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def get_reconstruction(self, x: torch.Tensor):
        noise = x.clone().div(127.5).sub(1).type(self.unet_dtype)
        latent = self.reverse_fn(
            self.unet,
            (1, 3, 256, 256),
            noise=noise,
            clip_denoised=True,
            model_kwargs=dict(),
            real_step=self.diff_steps,
            device=None,  # automatically use the same device as the model
            progress=False,
        )
        recons = self.sample_fn(
            self.unet,
            (1, 3, 256, 256),
            noise=latent,
            clip_denoised=True,
            model_kwargs=dict(),
            real_step=self.diff_steps,
            device=None,  # automatically use the same device as the model
            progress=False,
        ).float()
        recons = recons.add(1).mul(127.5).clamp(0, 255).int().float()
        return recons

    def get_dire(self, x: torch.Tensor, recons: torch.Tensor):
        dire = (x - recons).abs().div(255)
        return dire

    def forward(self, x: torch.Tensor):
        x = x * 255
        recons = self.get_reconstruction(x)
        dire = self.get_dire(x, recons)
        dire = self.clf_transforms(dire)
        return self.classifier(dire)

    def forward_intermediate(self, x: torch.Tensor):
        x = x * 255
        recons = self.get_reconstruction(x)
        dire = self.get_dire(x, recons)
        dire = self.clf_transforms(dire)
        return self.classifier.forward_intermediate(dire)
