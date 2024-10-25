import math
import torch
import torch.nn as nn

from .utils import conv, get_act
from typing import List


class ResBlock(nn.Module):
    """Implementation for ResNet block."""

    def __init__(
        self, n_feats: int, kernel_size: int, act: str = "leaky_relu", atrous: int = 1, bn: bool = False
    ) -> None:
        """param n_feats: Channel size.
        param kernel_size: kernel size.
        param act: string of activation to use.
        param atrous: controls amount of dilation to use in final conv.
        param bn: Turns on batch norm.
        """
        super().__init__()

        m: List[nn.Module] = []
        _repr = []
        for i in range(2):
            atrous_rate = 1 if i == 0 else atrous
            conv_filter = conv(n_feats, n_feats, kernel_size, rate=atrous_rate, bias=True)
            m.append(conv_filter)
            _repr.append(
                f"Conv({n_feats}x{kernel_size}" + (f";A*{atrous_rate})" if atrous_rate != 1 else "") + ")"
            )

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f"BN({n_feats})")

            if i == 0:
                m.append(get_act(act))
                _repr.append("Act")
        self.body = nn.Sequential(*m)

        self._repr = "/".join(_repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        res = self.body(x)
        res += x
        return res

    def __repr__(self) -> str:
        return f"ResBlock({self._repr})"


class Upsampler(nn.Sequential):
    def __init__(
        self, scale: int, n_feats: int, bn: bool = False, act: str = "none", bias: bool = True
    ) -> None:
        m: List[nn.Module] = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                m.append(get_act(act))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            m.append(get_act(act))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSRDec(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        resblocks: int = 8,
        kernel_size: int = 3,
        tail: str = "none",
        channel_attention: bool = False,
    ) -> None:
        super().__init__()
        self.head = conv(in_ch, out_ch, 1)
        m_body: List[nn.Module] = [ResBlock(out_ch, kernel_size) for _ in range(resblocks)]

        m_body.append(conv(out_ch, out_ch, kernel_size))
        self.body = nn.Sequential(*m_body)

        self.tail: nn.Module
        if tail == "conv":
            self.tail = conv(out_ch, out_ch, 1)
        elif tail == "none":
            self.tail = nn.Identity()  # type: ignore
        elif tail == "upsample":
            self.tail = Upsampler(scale=2, n_feats=out_ch)
        else:
            raise NotImplementedError(f"{tail} is not implemented.")

    def forward(
        self,  # type: ignore
        x: torch.Tensor,
        features_to_fuse: torch.Tensor = 0.0,  # type: ignore
    ) -> torch.Tensor:
        """
        :param x: N C H W
        :return: N C" H W
        """
        x = self.head(x)
        x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return x
