from torch.nn.modules.utils import _pair
from .dataset_cam_id import CamIdWebdataset, BytesIO, Image, pil_to_tensor, RandomCrop
from .random_jpeg_compression import RandomJpegCompression
from .utils import ComposeReturnAdditionalInfo


__all__ = ["JpegCompressionWebdataset"]


class JpegCompressionWebdataset(CamIdWebdataset):
    def __init__(self, jpeg_qf, p=1.0, **kwargs):
        self.jpeg_qf = _pair(jpeg_qf)
        self.p = p
        super().__init__(**kwargs)

    def get_transforms(self, other_transforms):
        jpeg_transform = RandomJpegCompression(
            qf=self.jpeg_qf,
            p=self.p,
            return_qf=True,
        )
        transforms = [jpeg_transform] + [] if other_transforms is None else [other_transforms]
        transforms.append(RandomCrop(self.model_patch_size))
        return ComposeReturnAdditionalInfo(transforms)

    def get_data(self, example):
        image = Image.open(BytesIO(example["png.png"])).convert("RGB")
        image = pil_to_tensor(image).float().div(255)
        image_dict = self.transforms(image)
        image = image_dict["image"]
        label = image_dict["qf"]
        return image, label
