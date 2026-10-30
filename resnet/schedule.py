"""Learning-rate schedule for the CIFAR-10 experiments (Sec. 4.2).

"We start with a learning rate of 0.1, divide it by 10 at 32k and 48k
iterations, and terminate training at 64k iterations."

For the 110-layer network "0.1 is slightly too large to start converging. So we
use 0.01 to warm up the training until the training error is below 80%
(about 400 iterations), and then go back to 0.1 and continue training."
"""
from typing import Optional, Sequence


class CifarSchedule:
    def __init__(self, base_lr: float = 0.1,
                 milestones: Sequence[int] = (32_000, 48_000),
                 gamma: float = 0.1,
                 warmup_lr: Optional[float] = None,
                 warmup_error: float = 0.8):
        self.base_lr = base_lr
        self.milestones = tuple(milestones)
        self.gamma = gamma
        self.warmup_lr = warmup_lr
        self.warmup_error = warmup_error
        self.warming_up = warmup_lr is not None

    def __call__(self, iteration: int, train_error: Optional[float] = None) -> float:
        """`train_error` is a running estimate of the training error in [0, 1]."""
        if self.warming_up:
            if train_error is not None and train_error < self.warmup_error:
                self.warming_up = False
            else:
                return self.warmup_lr
        drops = sum(iteration >= m for m in self.milestones)
        return self.base_lr * self.gamma ** drops
