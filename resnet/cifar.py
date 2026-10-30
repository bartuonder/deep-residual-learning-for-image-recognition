"""CIFAR-10 networks (Sec. 4.2).

The inputs are 32x32 images. The first layer is a 3x3 convolution. Then a stack
of 6n layers with 3x3 convolutions on feature maps of sizes {32, 16, 8}
respectively, with 2n layers for each feature map size. The numbers of filters
are {16, 32, 64}. The subsampling is performed by convolutions with a stride of
2. The network ends with a global average pooling, a 10-way fully-connected
layer, and softmax. There are totally 6n+2 stacked weighted layers.

The paper uses identity shortcuts everywhere (option A, so the residual nets
have exactly the same depth, width and number of parameters as the plain ones).
"""
import torch
import torch.nn as nn

from .blocks import BasicBlock, conv3x3
from .init import init_weights

FILTERS = (16, 32, 64)


class CifarResNet(nn.Module):
    def __init__(self, n: int, num_classes: int = 10, shortcut: str = "A",
                 residual: bool = True):
        super().__init__()
        self.conv1 = conv3x3(3, FILTERS[0])
        self.bn1 = nn.BatchNorm2d(FILTERS[0])
        self.relu = nn.ReLU(inplace=True)

        layers = []
        in_planes = FILTERS[0]
        for i, planes in enumerate(FILTERS):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(in_planes, planes, stride,
                                         shortcut=shortcut, residual=residual))
                in_planes = planes
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(FILTERS[-1], num_classes)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def cifar_resnet(depth: int, **kwargs) -> CifarResNet:
    """Build the (6n+2)-layer network, e.g. depth=20, 32, 44, 56, 110, 1202."""
    if (depth - 2) % 6 != 0 or depth < 8:
        raise ValueError(f"depth must be 6n+2 (n >= 1), got {depth}")
    return CifarResNet((depth - 2) // 6, **kwargs)
