"""Train a ResNet on ImageNet following Sec. 3.4 of the paper.

    python train_imagenet.py --data /path/to/imagenet --depth 50

The dataset must follow the ImageFolder layout (train/<class>/*.JPEG,
val/<class>/*.JPEG).

"We use SGD with a mini-batch size of 256. The learning rate starts from 0.1 and
is divided by 10 when the error plateaus, and the models are trained for up to
60 x 10^4 iterations. We use a weight decay of 0.0001 and a momentum of 0.9. We
do not use dropout."

The paper used 8 GPUs; on a single smaller GPU use --accum-steps to keep the
effective batch size at 256 (batch norm statistics are then computed over the
smaller micro-batch).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resnet.complexity import count_parameters
from resnet.imagenet import imagenet_resnet
from resnet.imagenet_data import center_crop_transform, imagenet_folder, train_transform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--depth", type=int, default=50, choices=[18, 34, 50, 101, 152])
    p.add_argument("--plain", action="store_true")
    p.add_argument("--shortcut", default="B", choices=["A", "B", "C"])
    p.add_argument("--stride-on-3x3", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="split each batch into this many micro-batches")
    p.add_argument("--iters", type=int, default=600_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="iterations between validations; the lr is divided by 10 "
                        "when validation error has not improved for --patience evals")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (not in the paper)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    top1 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        top5_idx = logits.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == y).sum().item()
        top5 += (top5_idx == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    model.train()
    return 100.0 * (1 - top1 / n), 100.0 * (1 - top5 / n)


def main():
    args = parse_args()
    kind = "plain" if args.plain else "resnet"
    name = args.name or f"imagenet_{kind}{args.depth}_s{args.seed}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    micro_bs = args.batch_size // args.accum_steps
    train_set = imagenet_folder(args.data, "train", train_transform())
    val_set = imagenet_folder(args.data, "val", center_crop_transform())
    train_loader = DataLoader(train_set, micro_bs, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, 100, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    model = imagenet_resnet(args.depth, shortcut=args.shortcut, residual=not args.plain,
                            stride_on_3x3=args.stride_on_3x3,
                            num_classes=len(train_set.classes)).to(device)
    print(f"{name}: {count_parameters(model) / 1e6:.2f}M params")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_err, bad_evals, start_iter = float("inf"), 0, 0
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_err, bad_evals = ckpt["iteration"], ckpt["best_err"], ckpt["bad_evals"]
        print(f"resumed from iteration {start_iter}")

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    new_log = not (args.resume and (out_dir / "log.csv").exists())
    log_file = open(out_dir / "log.csv", "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["iteration", "lr", "train_loss", "val_top1_err", "val_top5_err"])

    batches = infinite(train_loader)
    model.train()
    loss_sum, count, t0 = 0.0, 0, time.time()
    for it in range(start_iter, args.iters):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accum_steps):
            x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                loss = F.cross_entropy(model(x), y)
            (loss / args.accum_steps).backward()
            loss_sum += loss.detach()
            count += 1
        optimizer.step()

        if (it + 1) % args.eval_every == 0:
            top1, top5 = evaluate(model, val_loader, device, args.amp)
            lr = optimizer.param_groups[0]["lr"]
            train_loss = float(loss_sum) / count
            log.writerow([it + 1, lr, f"{train_loss:.4f}", f"{top1:.2f}", f"{top5:.2f}"])
            log_file.flush()
            print(f"[{it + 1:>7}/{args.iters}] lr {lr:.5f} loss {train_loss:.3f} "
                  f"val top-1 err {top1:.2f}% top-5 err {top5:.2f}% "
                  f"({time.time() - t0:.0f}s)", flush=True)
            loss_sum, count = 0.0, 0

            # "divided by 10 when the error plateaus"
            if top1 < best_err:
                best_err, bad_evals = top1, 0
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    for g in optimizer.param_groups:
                        g["lr"] = max(g["lr"] / 10, args.min_lr)
                    bad_evals = 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": it + 1, "best_err": best_err,
                        "bad_evals": bad_evals}, ckpt_path)

    torch.save(model.state_dict(), out_dir / "final.pt")


if __name__ == "__main__":
    main()
