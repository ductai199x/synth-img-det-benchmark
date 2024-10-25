from torch.nn.modules.utils import _pair
from .dataset_cam_id import CamIdWebdataset, BytesIO, Image, pil_to_tensor, RandomCrop
from .random_scale_resize import RandomScaleResize
from .utils import ComposeReturnAdditionalInfo


__all__ = ["ResamplingWebdataset"]


class ResamplingWebdataset(CamIdWebdataset):
    def __init__(self, resize_scale, p=1.0, **kwargs):
        self.resize_scale = _pair(resize_scale)
        self.p = p
        super().__init__(**kwargs)

    def get_transforms(self, other_transforms):
        resize_transform = RandomScaleResize(
            scale=self.resize_scale,
            min_short_size=self.model_patch_size,
            p=self.p,
            keepdim=True,
            return_scale=True,
        )
        transforms = [resize_transform] + [] if other_transforms is None else [other_transforms]
        transforms.append(RandomCrop(self.model_patch_size))
        return ComposeReturnAdditionalInfo(transforms)

    def get_data(self, example):
        image = Image.open(BytesIO(example["png.png"])).convert("RGB")
        image = pil_to_tensor(image).float().div(255)
        image_dict = self.transforms(image)
        image = image_dict["image"]
        label = image_dict["scale"]
        return image, label
