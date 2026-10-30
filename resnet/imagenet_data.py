"""ImageNet data pipeline (Sec. 3.4 and Sec. 4.1).

Training: "The image is resized with its shorter side randomly sampled in
[256, 480] for scale augmentation [41]. A 224x224 crop is randomly sampled from
an image or its horizontal flip, with the per-pixel mean subtracted [21]. The
standard color augmentation in [21] is used."

Testing: "In testing, for comparison studies we adopt the standard 10-crop
testing [21]. For best results, we adopt the fully-convolutional form as in
[41, 13], and average the scores at multiple scales (images are resized such
that the shorter side is in {224, 256, 384, 480, 640})."

Notes on details the paper leaves open:
  * the mean is a per-channel mean (a per-pixel mean image only makes sense
    for fixed-size inputs, which scale augmentation does not give);
  * pixels are scaled to [0, 1] and only the mean is subtracted, no std division;
  * the color augmentation is the PCA lighting noise of Krizhevsky et al. with
    sigma 0.1.
"""
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)

# PCA of the ImageNet RGB pixel covariance, as used in Krizhevsky et al. (2012)
PCA_EIGVAL = torch.tensor([0.2175, 0.0188, 0.0045])
PCA_EIGVEC = torch.tensor([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])


class RandomShorterSideResize:
    """Resize so the shorter side is a random integer in [min_size, max_size]."""

    def __init__(self, min_size: int = 256, max_size: int = 480):
        self.min_size, self.max_size = min_size, max_size

    def __call__(self, img):
        return TF.resize(img, random.randint(self.min_size, self.max_size),
                         interpolation=T.InterpolationMode.BILINEAR, antialias=True)


class PCALighting:
    """AlexNet-style color augmentation: add a random multiple of the principal
    components of the RGB distribution to every pixel."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return x
        alpha = torch.randn(3) * self.std
        rgb = PCA_EIGVEC @ (alpha * PCA_EIGVAL)
        return x + rgb.view(3, 1, 1)


class SubtractMean:
    def __init__(self, mean=IMAGENET_MEAN):
        self.mean = torch.tensor(mean).view(3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean


def train_transform(crop: int = 224, min_size: int = 256, max_size: int = 480,
                    color_std: float = 0.1):
    return T.Compose([
        RandomShorterSideResize(min_size, max_size),
        T.RandomCrop(crop),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        PCALighting(color_std),
        SubtractMean(),
    ])


def ten_crop_transform(resize: int = 256, crop: int = 224):
    """Standard 10-crop: four corners + center, and their horizontal flips."""
    return T.Compose([
        T.Resize(resize),
        T.TenCrop(crop),
        T.Lambda(lambda crops: torch.stack(
            [SubtractMean()(TF.to_tensor(c)) for c in crops])),
    ])


def center_crop_transform(resize: int = 256, crop: int = 224):
    return T.Compose([T.Resize(resize), T.CenterCrop(crop),
                      T.ToTensor(), SubtractMean()])


def full_image_transform(scale: int):
    """Resize the shorter side to `scale` and keep the whole image, for the
    fully-convolutional multi-scale test."""
    return T.Compose([T.Resize(scale), T.ToTensor(), SubtractMean()])


def imagenet_folder(root: str, split: str, transform) -> ImageFolder:
    return ImageFolder(f"{root}/{split}", transform=transform)
