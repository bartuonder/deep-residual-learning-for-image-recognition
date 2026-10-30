"""Train plain / residual networks on CIFAR-10 following Sec. 4.2 of the paper.

    python train_cifar.py --depth 20
    python train_cifar.py --depth 56 --plain
    python train_cifar.py --depth 110          # lr warm-up is enabled automatically

Defaults are the paper's: SGD, momentum 0.9, weight decay 1e-4, batch size 128,
lr 0.1 divided by 10 at 32k and 48k iterations, 64k iterations in total.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet.cifar import cifar_resnet
from resnet.cifar_data import BatchSampler, augment, load_cifar10
from resnet.complexity import count_parameters
from resnet.schedule import CifarSchedule


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=20, help="6n+2")
    p.add_argument("--plain", action="store_true", help="remove the shortcuts")
    p.add_argument("--shortcut", default="A", choices=["A", "B", "C"])
    p.add_argument("--iters", type=int, default=64_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--milestones", type=int, nargs="+", default=[32_000, 48_000])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-lr", type=float, default=None,
                   help="defaults to 0.01 for depth >= 110, as in the paper")
    p.add_argument("--val-split", action="store_true",
                   help="45k/5k train/val split instead of 50k train / 10k test")
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    wrong, loss = 0, 0.0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        loss += F.cross_entropy(logits, y[i:i + batch_size], reduction="sum").item()
        wrong += (logits.argmax(1) != y[i:i + batch_size]).sum().item()
    model.train()
    return 100.0 * wrong / len(x), loss / len(x)


def main():
    args = parse_args()
    kind = "plain" if args.plain else "resnet"
    name = args.name or f"cifar10_{kind}{args.depth}_s{args.seed}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_x, train_y, test_x, test_y, _ = load_cifar10(args.data, device, args.val_split)
    model = cifar_resnet(args.depth, shortcut=args.shortcut,
                         residual=not args.plain).to(device)
    print(f"{name}: {count_parameters(model) / 1e6:.3f}M params, "
          f"{len(train_x)} train / {len(test_x)} eval images, device={device}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    warmup_lr = args.warmup_lr if args.warmup_lr is not None else (
        0.01 if args.depth >= 110 else None)
    schedule = CifarSchedule(args.lr, args.milestones, warmup_lr=warmup_lr)
    sampler = BatchSampler(len(train_x), args.batch_size, device)

    start_iter, running_err = 0, 1.0
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, running_err = ckpt["iteration"], ckpt["running_err"]
        schedule.warming_up = ckpt["warming_up"]
        print(f"resumed from iteration {start_iter}")

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_path = out_dir / "log.csv"
    new_log = not (args.resume and log_path.exists())
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["iteration", "lr", "train_loss", "train_err", "eval_err", "eval_loss"])

    model.train()
    loss_sum, err_sum, count = 0.0, 0.0, 0
    t0 = time.time()
    for it in range(start_iter, args.iters):
        lr = schedule(it, running_err)
        for g in optimizer.param_groups:
            g["lr"] = lr

        idx = sampler.next()
        x, y = augment(train_x[idx]), train_y[idx]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_err = (logits.argmax(1) != y).float().mean()
        running_err = 0.99 * running_err + 0.01 * batch_err.item() if it % 10 == 0 \
            else running_err
        loss_sum += loss.detach()
        err_sum += batch_err
        count += 1

        if (it + 1) % args.eval_every == 0:
            eval_err, eval_loss = evaluate(model, test_x, test_y)
            train_loss, train_err = float(loss_sum) / count, 100.0 * float(err_sum) / count
            log.writerow([it + 1, lr, f"{train_loss:.4f}", f"{train_err:.2f}",
                          f"{eval_err:.2f}", f"{eval_loss:.4f}"])
            log_file.flush()
            print(f"[{it + 1:>6}/{args.iters}] lr {lr:.4f}  train loss {train_loss:.3f} "
                  f"err {train_err:5.2f}%  eval err {eval_err:5.2f}%  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            loss_sum, err_sum, count = 0.0, 0.0, 0
            if not torch.isfinite(torch.as_tensor(train_loss)):
                raise RuntimeError("training diverged")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": it + 1, "running_err": running_err,
                        "warming_up": schedule.warming_up}, ckpt_path)

    eval_err, _ = evaluate(model, test_x, test_y)
    (out_dir / "result.json").write_text(json.dumps(
        {"name": name, "depth": args.depth, "plain": args.plain,
         "params": count_parameters(model), "final_eval_error": eval_err,
         "eval_set": "val(5k)" if args.val_split else "test(10k)"}, indent=2))
    torch.save(model.state_dict(), out_dir / "final.pt")
    print(f"final eval error: {eval_err:.2f}%")


if __name__ == "__main__":
    main()
