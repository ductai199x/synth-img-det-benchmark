import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, VQModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput, DiagonalGaussianDistribution
from .lpips.lpips_custom import CustomLPIPS
from typing import Optional, List, Union, Tuple


def retrieve_latents(
    encoder_output: Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]],
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class AerobladeModel(nn.Module):
    def __init__(
        self,
        autoenc_list: List[Union[AutoencoderKL, VQModel]],
        decode_dtype_list: List[torch.dtype],
    ):
        super().__init__()
        self.autoenc_list = autoenc_list
        self.decode_dtype_list = decode_dtype_list
        self.loss_fn = CustomLPIPS(net="vgg")

    def forward(self, x):
        B, C, H, W = x.shape
        recon_errors = []
        for autoenc, decode_dtype in zip(self.autoenc_list, self.decode_dtype_list):
            x_norm = x.type(decode_dtype) * 2 - 1
            x_enc = autoenc.encode(x_norm)
            latents = retrieve_latents(x_enc)
            if isinstance(autoenc, VQModel):
                recon = autoenc.decode(
                    latents.type(decode_dtype), force_not_quantize=True, return_dict=False
                )[0]
            else:
                recon = autoenc.decode(latents.type(decode_dtype), return_dict=False)[0]

            recon = ((recon / 2) + 0.5).clamp(0, 1)
            recon_errors.append(self.loss_fn(recon, x).squeeze())
        min_error = torch.stack(recon_errors).unsqueeze(0).min(dim=1, keepdim=True).values
        return -min_error

    def forward_intermediate(self, x):
        recon_error = self.forward(x)
        return recon_error, recon_error
