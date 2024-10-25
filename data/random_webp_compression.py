import random
import torch
from torch.nn.modules.utils import _pair
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from kornia.augmentation import AugmentationBase2D
from PIL import Image
from io import BytesIO


__all__ = ["RandomWebpCompression"]


class RandomWebpCompression(AugmentationBase2D):
    def __init__(
        self,
        qf: tuple = (50, 100),
        p: float = 1.0,
        same_on_batch: bool = False,
        debug: bool = False,
        return_qf: bool = False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=p)
        qf = _pair(qf)
        assert qf[0] < qf[1], "The minimum quality factor must be less than the maximum quality factor"
        self.quality_min = qf[0]
        self.quality_max = qf[1]
        self.debug = debug
        self.return_qf = return_qf

    def return_result(self, x, qf=None):
        if self.return_qf:
            return {
                "image": x,
                "qf": (qf if qf is not None else 100) / 100,
            }
        else:
            return x

    def forward(self, x: torch.Tensor, params=None, **kwargs) -> torch.Tensor:
        if random.random() > self.p:
            return self.return_result(x)

        x: Image.Image = to_pil_image(x)
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = BytesIO()
        x.save(buffer, format="webp", quality=quality)
        buffer.seek(0)
        x = Image.open(buffer).convert("RGB")
        x = pil_to_tensor(x).float().div(255)

        return self.return_result(x, quality)
