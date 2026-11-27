"""Turn evaluate.py's output for the 50 v1 checkpoints into a table, a plot and a short report."""
import argparse
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT


def main():
    parser = argparse.ArgumentParser(description="Plot mIoU against epoch for the v1 re-score")
    parser.add_argument("--eval-dir", default=os.path.join(ROOT, "outputs/v1_rescore"),
                        help="where evaluate.py wrote one JSON per checkpoint and summary.csv")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results/v1_rescore"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.eval_dir, "summary.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["epoch"]]
    rows.sort(key=lambda r: int(r["epoch"]))
    epochs = [int(r["epoch"]) for r in rows]
    miou = [float(r["miou"]) for r in rows]
    old = [float(r["old_batch_miou"]) for r in rows]

    best = max(rows, key=lambda r: float(r["miou"]))
    last = rows[-1]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, miou, marker=".", label="19-class mIoU (confusion matrix, full size)")
    plt.plot(epochs, old, marker=".", label="old per-batch number from train.py")
    plt.axvline(int(best["epoch"]), color="gray", ls="--", lw=0.8)
    plt.xlabel("epoch")
    plt.ylabel("mIoU on the 40 val images")
    plt.title("v1 checkpoints scored again")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "miou_vs_epoch.png"), dpi=120)
    plt.close()

    shutil.copy(os.path.join(args.eval_dir, "summary.csv"), os.path.join(args.out_dir, "summary.csv"))
    for tag, row in [("epoch500", last), ("best", best)]:
        stem = os.path.splitext(os.path.basename(row["checkpoint"]))[0]
        shutil.copy(os.path.join(args.eval_dir, f"{stem}_classes.csv"),
                    os.path.join(args.out_dir, f"{tag}_classes.csv"))

    with open(os.path.join(args.eval_dir, os.path.splitext(os.path.basename(last["checkpoint"]))[0] + ".json")) as f:
        info = json.load(f)
    lines = [
        f"commit {info['commit']}, split {info['split']} ({info['n_images']} images), "
        f"input {info['input_size'][0]}x{info['input_size'][1]}",
        f"command: {info['command']}",
        "",
        "| checkpoint | epoch | mIoU | pixel acc | old per-batch | classes averaged |",
        "|---|---|---|---|---|---|",
    ]
    for tag, row in [("last (used by the old inference)", last), ("best on val", best)]:
        lines.append(f"| {row['checkpoint']} ({tag}) | {row['epoch']} | {row['miou']} | {row['pixel_acc']} | "
                     f"{row['old_batch_miou']} | {row['n_classes_averaged']} |")
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
