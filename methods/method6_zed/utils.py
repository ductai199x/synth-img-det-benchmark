import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple


def conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, rate: int = 1, stride: int = 1
) -> nn.Conv2d:
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate, padding=padding, bias=bias
    )


def get_act(act: str, n_feats: int = 0) -> nn.Module:
    """param act: Name of activation used.
    n_feats: channel size.
    returns the respective activation module, or raise
        NotImplementedError if act is not implememted.
    """
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "prelu":
        return nn.PReLU(n_feats)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act == "none":
        return nn.Identity()  # type: ignore
    raise NotImplementedError(f"{act} is not implemented")


def tensor_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x - 0.001)


def average_downsamples(x: torch.Tensor, num_scales: int) -> List[torch.Tensor]:
    downsampled = []
    for _ in range(num_scales):
        downsampled.append(x.detach())
        x = F.avg_pool2d(pad_to_even(tensor_round(x)), 2)
    downsampled.append(x.detach())
    return downsampled


def pad_to_even(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.size()
    pad_right = w % 2 == 1
    pad_bottom = h % 2 == 1
    padding = [0, 1 if pad_right else 0, 0, 1 if pad_bottom else 0]
    x = F.pad(x, padding, mode="replicate")
    return x


def pad(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    _, _, xH, xW = x.size()
    padding = [0, W - xW, 0, H - xH]
    return F.pad(x, padding, mode="replicate")


def join_2x2(padded_slices: List[torch.Tensor], shape: Tuple[int, int]) -> torch.Tensor:
    assert len(padded_slices) == 4, len(padded_slices)
    # 4 N 3 H W
    x = torch.stack(padded_slices)
    # N 3 4 H W
    x = x.permute(1, 2, 0, 3, 4)
    N, _, _, H, W = x.size()
    # N 12 H W
    x = x.contiguous().view(N, -1, H, W)
    # N 3 2H 2W
    x = F.pixel_shuffle(x, upscale_factor=2)
    # return x[..., :unpad_h, :unpad_w]
    return x[..., : shape[-2], : shape[-1]]


def get_shapes(H: int, W: int) -> List[Tuple[int, int]]:
    shapes = [(H, W)]
    h = H
    w = W
    for _ in range(3):
        h = (h + 1) // 2
        w = (w + 1) // 2
        shapes.append((h, w))
    return shapes


def get_2x2_shapes(
    H: int, W: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    top_H = (H + 1) // 2
    left_W = (W + 1) // 2
    bottom_H = H - top_H
    right_W = W - left_W
    return ((top_H, left_W), (top_H, right_W), (bottom_H, left_W), (bottom_H, right_W))
