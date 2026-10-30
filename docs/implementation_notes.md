# Implementation notes

Where the paper is explicit the code follows it; where it is silent, the choice
made here is listed below so a reader can tell the two apart.

## Paper → code

| Paper | Where |
|---|---|
| Eq. (1) identity shortcut, Eq. (2) projection shortcut | `resnet/blocks.py` (`BasicBlock`, `Bottleneck`, `make_shortcut`) |
| Fig. 2 building block, Fig. 5 bottleneck | `resnet/blocks.py` |
| Shortcut options A / B / C (Sec. 4.1) | `resnet/blocks.py::make_shortcut` |
| Table 1 architectures | `resnet/imagenet.py::CONFIGS` |
| Plain networks (Fig. 3 middle) | `residual=False` on every model |
| Sec. 3.4 initialization [13] | `resnet/init.py` |
| Sec. 3.4 ImageNet augmentation | `resnet/imagenet_data.py::train_transform` |
| Sec. 3.4 ImageNet optimization | `train_imagenet.py` |
| Sec. 4.1 10-crop / multi-scale fully-conv testing | `eval_imagenet.py` |
| Sec. 4.2 CIFAR-10 6n+2 architecture | `resnet/cifar.py` |
| Sec. 4.2 CIFAR-10 schedule, warm-up for 110 layers | `resnet/schedule.py`, `train_cifar.py` |
| Sec. 4.2 CIFAR-10 augmentation | `resnet/cifar_data.py` |

## Sanity checks against the paper

`tests/test_models.py` asserts, among other things:

* ImageNet FLOPs (multiply-adds) of 18/34/50/101/152 layers are 1.8 / 3.6 / 3.8 /
  7.6 / 11.3 ×10⁹ (Table 1);
* CIFAR-10 parameter counts of 20/32/44/56/110/1202 layers are
  0.27 / 0.46 / 0.66 / 0.85 / 1.7 / 19.4 M (Sec. 4.2);
* with option A the residual and plain CIFAR nets have identical parameter counts.

## Things the paper does not specify

| Question | Choice | Why |
|---|---|---|
| Stride of a downsampling bottleneck (1×1 or 3×3?) | 1×1 (`stride_on_3x3=False`) | This is what the reference Caffe model does and it reproduces the 3.8 / 7.6 / 11.3 GFLOPs of Table 1 (putting the stride on the 3×3 gives ~4.1 for ResNet-50). Switchable. |
| Option-A padding layout | zeros appended after the existing channels | Literal reading of "extra zero entries padded". |
| Shortcut projection | 1×1 conv + BN | Eq. (2) is only `W_s x`; BN follows the "BN after every convolution" rule of Sec. 3.4. |
| Init of the final FC layer | PyTorch default | Not mentioned. |
| Conv init fan | fan-in, ReLU gain | [13] gives `std = sqrt(2 / n_l)` with `n_l = k² c`. |
| "Per-pixel mean" on ImageNet | per-channel mean | A mean image cannot be subtracted from variable-size crops of scale-augmented images. |
| Input scale | [0, 1], mean subtracted, no std division | The paper only mentions mean subtraction. |
| Weight decay on BN / bias | applied to all parameters | The paper just says "weight decay of 0.0001". |
| ImageNet "error plateaus" | divide lr by 10 after `--patience` validations without top-1 improvement | Needs a concrete rule. |
| CIFAR-10 training data | the full 50k, evaluated on the 10k test set | The 45k/5k split in the paper was only used to choose the schedule (`--val-split` reproduces it). |
| CIFAR-10 reported number | error at the last iteration | No checkpoint selection on the test set. |

## Deviations forced by hardware

* **Single GPU.** The paper trained on 2 (CIFAR) / 8 (ImageNet) GPUs. CIFAR runs
  use batch 128 on one GPU, which is the same effective batch. ImageNet training
  supports `--accum-steps` to keep the effective batch at 256.
* **ImageNet was not trained.** The full 60×10⁴-iteration ImageNet schedule is
  far outside the budget of a single consumer GPU; the ImageNet code is covered
  by unit tests and smoke tests only, and no ImageNet numbers are claimed.
