"""CIFAR-10 data pipeline (Sec. 4.2, 3.4).

Preprocessing and augmentation as in the paper:
  * per-pixel mean subtracted (the mean image is computed on the training set)
  * "4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip"
  * "For testing, we only evaluate the single view of the original 32x32 image"

The whole dataset is small enough to live on the GPU, so augmentation is done
there as well, which avoids data-loader overhead.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10


def load_cifar10(root: str, device: torch.device, val_split: bool = False):
    """Returns (train_x, train_y, eval_x, eval_y, mean_image).

    By default trains on the 50k training images and evaluates on the 10k test
    images. With `val_split=True` it uses the 45k/5k train/val split the paper
    used to pick the schedule.
    """
    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)

    def to_tensor(ds) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).float() / 255.0
        y = torch.tensor(ds.targets, dtype=torch.long)
        return x, y

    train_x, train_y = to_tensor(train)
    test_x, test_y = to_tensor(test)

    if val_split:
        # fixed split so every run sees the same 5k validation images
        perm = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(0))
        val_idx, train_idx = perm[:5_000], perm[5_000:]
        test_x, test_y = train_x[val_idx], train_y[val_idx]
        train_x, train_y = train_x[train_idx], train_y[train_idx]

    mean = train_x.mean(dim=0, keepdim=True)  # (1, 3, 32, 32) per-pixel mean
    train_x, test_x = train_x - mean, test_x - mean
    return (train_x.to(device), train_y.to(device),
            test_x.to(device), test_y.to(device), mean.to(device))


def augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip + random crop from the zero-padded image."""
    b, c, h, w = x.shape
    dev = x.device
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (oy[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    batch = torch.arange(b, device=dev)[:, None, None, None]
    chan = torch.arange(c, device=dev)[None, :, None, None]
    return x[batch, chan, rows, cols]


class BatchSampler:
    """Iterates over shuffled epochs forever, yielding fixed-size batches."""

    def __init__(self, n: int, batch_size: int, device: torch.device):
        self.n, self.batch_size, self.device = n, batch_size, device
        self._perm, self._pos = None, 0

    def next(self) -> torch.Tensor:
        if self._perm is None or self._pos + self.batch_size > self.n:
            self._perm = torch.randperm(self.n, device=self.device)
            self._pos = 0
        idx = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx
