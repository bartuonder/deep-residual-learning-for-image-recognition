import torch
from PIL import Image

from resnet.imagenet_data import (PCALighting, RandomShorterSideResize, SubtractMean,
                                  full_image_transform, ten_crop_transform, train_transform)


def _img(w=500, h=375):
    return Image.new("RGB", (w, h), (120, 100, 80))


def test_scale_augmentation_range():
    t = RandomShorterSideResize(256, 480)
    for _ in range(50):
        out = t(_img())
        assert 256 <= min(out.size) <= 480


def test_train_transform_shape():
    out = train_transform()(_img())
    assert out.shape == (3, 224, 224)


def test_ten_crop_shape():
    assert ten_crop_transform()(_img()).shape == (10, 3, 224, 224)


def test_full_image_keeps_aspect_ratio():
    out = full_image_transform(384)(_img(500, 375))
    assert out.shape[1] == 384 and out.shape[2] == round(500 * 384 / 375)


def test_pca_lighting_is_constant_shift_per_channel():
    x = torch.zeros(3, 8, 8)
    out = PCALighting(0.1)(x)
    assert torch.allclose(out, out[:, :1, :1].expand_as(out))
    assert torch.equal(PCALighting(0.0)(x), x)


def test_mean_subtraction():
    x = torch.ones(3, 2, 2)
    out = SubtractMean()(x)
    assert torch.allclose(out[:, 0, 0], 1 - torch.tensor([0.485, 0.456, 0.406]))
