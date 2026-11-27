"""Write splits/train.txt and splits/val.txt.

This is the split the first train.py made with torch.manual_seed(42) and
random_split, which was never saved. random_split only looks at len(dataset),
so splitting the sorted file list gives the same indices.
"""
import argparse
import os

import torch
from torch.utils.data import random_split

from config import ROOT, add_common_args


def old_split(names, seed=42):
    train_size = int(0.8 * len(names))
    val_size = len(names) - train_size
    torch.manual_seed(seed)
    train_set, val_set = random_split(names, [train_size, val_size])
    return [names[i] for i in train_set.indices], [names[i] for i in val_set.indices]


def read_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser(description="Save the 80/20 train/val split"))
    parser.add_argument("--split-dir", default=os.path.join(ROOT, "splits"))
    args = parser.parse_args()

    names = sorted(os.listdir(os.path.join(args.data_root, "data_semantics/training/image_2")))
    train_names, val_names = old_split(names, args.seed)

    os.makedirs(args.split_dir, exist_ok=True)
    for split, split_names in [("train", train_names), ("val", val_names)]:
        # keep the random_split order, so line i is still index i of the old Subset
        with open(os.path.join(args.split_dir, f"{split}.txt"), "w") as f:
            f.write("\n".join(split_names) + "\n")
        print(f"{split}: {len(split_names)} images")
