import random
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from kornia.augmentation import AugmentationBase2D
from typing import Tuple, Union


__all__ = ["RandomScaleResize"]


class RandomScaleResize(AugmentationBase2D):
    def __init__(
        self,
        scale: Union[torch.Tensor, Tuple[float, float]],
        min_short_size: int = 256,
        resample: Union[str, int] = "bicubic",
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        return_scale: bool = False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.scale = list(_pair(scale))
        assert self.scale[0] < self.scale[1], "The minimum scale must be less than the maximum scale"
        self.min_short_size = min_short_size
        self.resample = resample
        self.align_corners = align_corners
        self.return_scale = return_scale

    def _return_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.keepdim:
            x = x.squeeze(0)
        return x

    def return_result(self, x, scale=None):
        if self.return_scale:
            return {
                "image": self._return_x(x),
                "scale": scale if scale is not None else 1.0,
            }
        else:
            return x

    def get_scale(self, short_size: int):
        if short_size * self.scale[0] < self.min_short_size:
            self.scale[0] = self.min_short_size / short_size + 1e-3
            if self.scale[0] >= self.scale[1]:
                scale = self.scale[0]
            else:
                scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = random.uniform(self.scale[0], self.scale[1])
        return round(scale, 2)

    def forward(self, x: torch.Tensor, params=None, **kwargs) -> torch.Tensor:
        if random.random() > self.p:
            return self.return_result(x)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        scale = self.get_scale(min(x.shape[-2:]))
        x = F.interpolate(
            x, scale_factor=scale, mode=self.resample, align_corners=self.align_corners, antialias=True
        )

        return self.return_result(x, scale)
