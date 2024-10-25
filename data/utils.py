import torch
from torchvision.utils import _log_api_usage_once


class ComposeReturnAdditionalInfo:
    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img):
        info = {}
        for t in self.transforms:
            output = t(img)
            if isinstance(output, dict):
                img = output.pop("image")
                info.update(output)
            else:
                img = output
        if len(info) > 0:
            return {
                "image": img,
                **info,
            }
        else:
            return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
