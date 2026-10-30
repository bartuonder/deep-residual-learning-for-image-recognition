import torch

from resnet.cifar_data import BatchSampler, augment
from resnet.schedule import CifarSchedule


def test_augment_preserves_shape_and_pads_with_zeros():
    x = torch.ones(64, 3, 32, 32)
    out = augment(x)
    assert out.shape == x.shape
    # a crop of an all-ones image with zero padding contains only 0s and 1s
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert (out == 0).any()


def test_augment_crop_is_a_window_of_the_padded_image():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    out = augment(x)
    for xi, oi in zip(x, out):
        padded = torch.nn.functional.pad(xi, (4, 4, 4, 4))
        padded_flip = torch.nn.functional.pad(xi.flip(2), (4, 4, 4, 4))
        found = any(torch.equal(oi, p[:, dy:dy + 32, dx:dx + 32])
                    for p in (padded, padded_flip)
                    for dy in range(9) for dx in range(9))
        assert found


def test_sampler_covers_every_sample_once_per_epoch():
    s = BatchSampler(12, 4, torch.device("cpu"))
    seen = torch.cat([s.next() for _ in range(3)])
    assert sorted(seen.tolist()) == list(range(12))


def test_step_schedule():
    sched = CifarSchedule()
    assert sched(0) == 0.1
    assert sched(31_999) == 0.1
    assert abs(sched(32_000) - 0.01) < 1e-12
    assert abs(sched(48_000) - 0.001) < 1e-12


def test_warmup_for_110_layer_net():
    sched = CifarSchedule(warmup_lr=0.01)
    assert sched(0, train_error=0.9) == 0.01
    assert sched(300, train_error=0.85) == 0.01
    assert sched(400, train_error=0.79) == 0.1
    # warm-up never resumes once finished
    assert sched(401, train_error=0.95) == 0.1
