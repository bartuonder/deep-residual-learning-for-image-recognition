"""Weight initialization.

Sec. 3.4: "We initialize the weights as in [13] and train all plain/residual
nets from scratch", where [13] is He et al., "Delving Deep into Rectifiers"
(zero-mean Gaussian with std = sqrt(2 / n_l), n_l = k^2 * c the fan-in of the
layer). Batch norm scale/shift start at 1/0 as usual.

The paper does not say how the final fully-connected layer is initialized, so
it keeps the PyTorch default.
"""
import torch.nn as nn


def init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
