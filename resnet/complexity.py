"""Parameter and FLOP counting, used to check the models against the paper.

Following the paper (Table 1 footnote / Fig. 3) "FLOPs" means multiply-adds of
the convolution and fully-connected layers.
"""
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_flops(model: nn.Module, input_size=(1, 3, 224, 224)) -> int:
    total = 0

    def conv_hook(m, inp, out):
        nonlocal total
        kh, kw = m.kernel_size
        total += out.numel() * (m.in_channels // m.groups) * kh * kw

    def fc_hook(m, inp, out):
        nonlocal total
        total += out.numel() * m.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(fc_hook))

    was_training = model.training
    model.eval()
    model(torch.zeros(*input_size))
    model.train(was_training)
    for h in handles:
        h.remove()
    return total // input_size[0]
