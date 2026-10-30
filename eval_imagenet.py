"""Evaluate a trained ImageNet ResNet with the test protocols of Sec. 4.1.

    python eval_imagenet.py --data /path/to/imagenet --depth 50 --weights final.pt --mode ten-crop
    python eval_imagenet.py ... --mode multi-scale

  single       resize to 256, one 224 center crop
  ten-crop     standard 10-crop testing (Table 3 / 4)
  multi-scale  fully-convolutional form on the whole image, scores averaged over
               shorter sides {224, 256, 384, 480, 640} (Table 5 / 6)

The network ends with global average pooling followed by a linear layer, so it
can be applied to images of any size: the linear layer on the pooled features
equals the spatial average of the per-location class scores, which is the
fully-convolutional test of the paper.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from resnet.imagenet import imagenet_resnet
from resnet.imagenet_data import (center_crop_transform, full_image_transform,
                                  imagenet_folder, ten_crop_transform)

SCALES = (224, 256, 384, 480, 640)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--depth", type=int, default=50, choices=[18, 34, 50, 101, 152])
    p.add_argument("--plain", action="store_true")
    p.add_argument("--shortcut", default="B", choices=["A", "B", "C"])
    p.add_argument("--stride-on-3x3", action="store_true")
    p.add_argument("--weights", required=True)
    p.add_argument("--mode", default="ten-crop", choices=["single", "ten-crop", "multi-scale"])
    p.add_argument("--scales", type=int, nargs="+", default=list(SCALES))
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def scores(model, x, mode):
    """Class probabilities averaged over the views of one batch."""
    if mode == "ten-crop":
        b, n = x.shape[:2]
        probs = model(x.flatten(0, 1)).softmax(1).view(b, n, -1)
        return probs.mean(1)
    return model(x).softmax(1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = imagenet_folder(args.data, "val", None)
    model = imagenet_resnet(args.depth, shortcut=args.shortcut, residual=not args.plain,
                            stride_on_3x3=args.stride_on_3x3,
                            num_classes=len(probe.classes)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    if args.mode == "multi-scale":
        # whole images differ in size, so they are processed one at a time
        loaders = [DataLoader(imagenet_folder(args.data, "val", full_image_transform(s)),
                              batch_size=1, num_workers=args.workers) for s in args.scales]
        batches = zip(*loaders)
        views = lambda b: (sum(model(x.to(device)).softmax(1) for x, _ in b) / len(b),
                           b[0][1].to(device))
    else:
        tf = ten_crop_transform() if args.mode == "ten-crop" else center_crop_transform()
        loader = DataLoader(imagenet_folder(args.data, "val", tf), args.batch_size,
                            num_workers=args.workers)
        batches = ([b] for b in loader)
        views = lambda b: (scores(model, b[0][0].to(device), args.mode), b[0][1].to(device))

    top1 = top5 = n = 0
    with torch.no_grad():
        for b in batches:
            probs, y = views(b)
            top5_idx = probs.topk(5, dim=1).indices
            top1 += (top5_idx[:, 0] == y).sum().item()
            top5 += (top5_idx == y[:, None]).any(dim=1).sum().item()
            n += len(y)
    print(f"{args.mode}: top-1 err {100 * (1 - top1 / n):.2f}%  "
          f"top-5 err {100 * (1 - top5 / n):.2f}%  ({n} images)")


if __name__ == "__main__":
    main()
