import torch
from diffusers import AutoPipelineForImage2Image


def get_autoenc(repo_id, cache_dir):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16" if "kandinsky-2" not in repo_id else None,
        cache_dir=cache_dir,
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae"):
        autoenc = pipe.vae
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
    elif hasattr(pipe, "movq"):
        autoenc = pipe.movq
    autoenc = torch.compile(autoenc)
    decode_dtype = next(iter(autoenc.post_quant_conv.parameters())).dtype
    return autoenc, decode_dtype