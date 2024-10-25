from functools import partial
from torch.nn.functional import interpolate
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, crop, resize
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, Lambda, Grayscale


def center_crop_no_pad(image, size):
    B, C, H, W = image.shape
    crop_height, crop_width = size, size
    crop_height = min(H, crop_height)
    crop_width = min(W, crop_width)
    crop_top = int(round((H - crop_height) / 2.0))
    crop_left = int(round((W - crop_width) / 2.0))
    return crop(image, crop_top, crop_left, crop_height, crop_width)
