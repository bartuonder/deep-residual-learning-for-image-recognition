"""Plot training/test error curves in the style of Fig. 6 and print a results table.

    python scripts/plot_cifar.py --runs runs --out docs/cifar10_plain_vs_resnet.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {20: "tab:green", 32: "tab:blue", 44: "tab:orange", 56: "tab:red", 110: "tab:purple"}


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["iteration"]) for r in rows],
            [float(r["train_err"]) for r in rows],
            [float(r["eval_err"]) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="docs/cifar10_plain_vs_resnet.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    table = []
    for ax, kind in zip(axes, ("plain", "resnet")):
        for depth, color in COLORS.items():
            d = Path(args.runs) / f"cifar10_{kind}{depth}_s{args.seed}"
            if not (d / "log.csv").exists():
                continue
            it, train_err, eval_err = read_log(d / "log.csv")
            ax.plot(it, train_err, "--", color=color, linewidth=1)
            ax.plot(it, eval_err, "-", color=color, linewidth=1.5, label=f"{kind}-{depth}")
            if (d / "result.json").exists():
                table.append((kind, depth, json.loads((d / "result.json").read_text())))
        ax.set_title(f"CIFAR-10 {kind} nets")
        ax.set_xlabel("iter.")
        ax.set_ylim(0, 20)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("error (%)  (dashed: train, bold: test)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print("| model | depth | params | test error (%) |")
    print("|---|---|---|---|")
    for kind, depth, r in sorted(table, key=lambda t: (t[0], t[1])):
        print(f"| {kind} | {depth} | {r['params'] / 1e6:.2f}M | {r['final_eval_error']:.2f} |")


if __name__ == "__main__":
    main()
