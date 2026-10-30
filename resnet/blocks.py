"""Residual building blocks (He et al., 2016).

The paper reformulates a stack of layers to learn a residual function
F(x) := H(x) - x instead of the desired mapping H(x) directly. A block computes

    y = F(x, {W_i}) + x          (Eq. 1)   identity shortcut
    y = F(x, {W_i}) + W_s x      (Eq. 2)   projection shortcut

and applies a ReLU after the addition (Fig. 2 / Fig. 5). Batch norm is placed
right after every convolution and before the activation (Sec. 3.4).

Shortcut types when the shape of x and F(x) differ (Sec. 4.1, "Identity vs.
Projection Shortcuts"):

    A: identity (zero-padded for extra channels, stride 2 for subsampling),
       parameter-free
    B: projection (1x1 conv) only where the dimensions change, identity
       elsewhere
    C: projection on every shortcut
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class PaddedIdentity(nn.Module):
    """Option A shortcut: parameter-free.

    Subsamples spatially with stride `stride` and appends zero channels so that
    the output has `out_planes` channels.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        if out_planes < in_planes:
            raise ValueError("PaddedIdentity can only increase the channel count")
        self.stride = stride
        self.extra = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, 0, self.extra))


class Projection(nn.Sequential):
    """W_s x in Eq. (2): a strided 1x1 convolution followed by batch norm."""

    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
        )


def make_shortcut(option: str, in_planes: int, out_planes: int,
                  stride: int) -> nn.Module:
    if option not in ("A", "B", "C"):
        raise ValueError(f"unknown shortcut option {option!r}, expected A, B or C")
    shape_changes = stride != 1 or in_planes != out_planes
    if option == "C":
        return Projection(in_planes, out_planes, stride)
    if not shape_changes:
        return nn.Identity()
    if option == "A":
        return PaddedIdentity(in_planes, out_planes, stride)
    return Projection(in_planes, out_planes, stride)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions (Fig. 2, used by ResNet-18/34 and the CIFAR nets)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B", residual: bool = True):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        out_planes = planes * self.expansion
        self.shortcut: Optional[nn.Module] = (
            make_shortcut(shortcut, in_planes, out_planes, stride) if residual else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 "bottleneck" design (Fig. 5 right, ResNet-50/101/152).

    The 1x1 layers reduce and then restore the dimensions, leaving the 3x3 layer
    with smaller input/output dimensions. The paper does not state where the
    stride-2 of a downsampling block goes. The reference Caffe model puts it on
    the first 1x1 convolution, which is what matches the 3.8 / 7.6 / 11.3 GFLOPs
    reported in Table 1; `stride_on_3x3=True` selects the placement that later
    became popular instead.
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B", residual: bool = True,
                 stride_on_3x3: bool = False):
        super().__init__()
        stride1, stride2 = (1, stride) if stride_on_3x3 else (stride, 1)
        out_planes = planes * self.expansion
        self.conv1 = conv1x1(in_planes, planes, stride1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: Optional[nn.Module] = (
            make_shortcut(shortcut, in_planes, out_planes, stride) if residual else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.relu(out)
