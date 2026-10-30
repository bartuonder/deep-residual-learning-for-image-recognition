"""ImageNet networks (Sec. 3.3, Table 1).

                 | 18-layer | 34-layer | 50-layer | 101-layer | 152-layer
    conv1        | 7x7, 64, stride 2 (112x112)
    conv2_x      | 3x3 max pool, stride 2, then the first stage (56x56)
    conv3_x      | 28x28
    conv4_x      | 14x14
    conv5_x      | 7x7
                 | average pool, 1000-d fc, softmax

Downsampling is performed by conv3_1, conv4_1 and conv5_1 with a stride of 2.
The 18/34-layer nets use BasicBlock, the deeper ones use Bottleneck.
"""
from typing import Sequence, Type, Union

import torch
import torch.nn as nn

from .blocks import BasicBlock, Bottleneck
from .init import init_weights

# depth -> (block, blocks per stage); Table 1
CONFIGS = {
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3)),
}
STAGE_PLANES = (64, 128, 256, 512)


class ImageNetResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],
                 layers: Sequence[int], num_classes: int = 1000,
                 shortcut: str = "B", residual: bool = True,
                 stride_on_3x3: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_kwargs = dict(shortcut=shortcut, residual=residual)
        if block is Bottleneck:
            block_kwargs["stride_on_3x3"] = stride_on_3x3

        stages = []
        in_planes = 64
        for i, (planes, num_blocks) in enumerate(zip(STAGE_PLANES, layers)):
            blocks = []
            for j in range(num_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(block(in_planes, planes, stride, **block_kwargs))
                in_planes = planes * block.expansion
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(STAGE_PLANES[-1] * block.expansion, num_classes)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def imagenet_resnet(depth: int, **kwargs) -> ImageNetResNet:
    if depth not in CONFIGS:
        raise ValueError(f"depth must be one of {sorted(CONFIGS)}, got {depth}")
    block, layers = CONFIGS[depth]
    return ImageNetResNet(block, layers, **kwargs)
