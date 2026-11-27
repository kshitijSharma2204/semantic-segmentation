"""Time training steps for the speed options in train.py.

  python bench.py train      real data, 10 warm-up + 50 timed steps per setting
  python bench.py synthetic  random batches, for the memory/out-of-memory checks
  python bench.py confusion  confusion matrix on the GPU vs the CPU
"""
import argparse
import itertools
import os
import time

import numpy as np
import torch
from PIL import Image

from config import ROOT, load_labels
from metrics import IGNORE, id_to_trainid_lut, new_confusion, update_confusion
from model import UNet
from make_splits import read_split
import train

WARMUP, STEPS = 10, 50

# each setting adds one change to the one above it
TRAIN_SETTINGS = [
    ("decode PNGs every step (old behaviour)", ["--no-cache", "--no-pin-memory"]),
    ("+ images cached in RAM", ["--no-pin-memory"]),
    ("+ pin_memory and non_blocking copies", []),
    ("+ channels_last", ["--channels-last"]),
    ("+ fused AdamW", ["--channels-last", "--fused-adam"]),
    ("+ torch.compile", ["--channels-last", "--fused-adam", "--compile"]),
]


def fp32_setup(cudnn_benchmark=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = cudnn_benchmark


def time_steps(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / STEPS * 1000, torch.cuda.max_memory_allocated() / 2**30


def bench_train(extra):
    fp32_setup()
    rows = []
    labels = load_labels(os.path.join(ROOT, "data"))
    for name, flags in TRAIN_SETTINGS:
        args = train.parse_args(flags + extra)
        torch.manual_seed(0)
        dataset, loader = train.make_loader(args, labels)
        model = train.build_model(args, "cuda")
        net = torch.compile(model) if args.compile else model
        optimizer, _ = train.build_optimizer(args, model, len(loader))
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        batches = itertools.chain.from_iterable(itertools.repeat(loader))

        def step():
            imgs, masks = next(batches)
            imgs = imgs.to("cuda", non_blocking=args.pin_memory)
            masks = masks.to("cuda", non_blocking=args.pin_memory)
            if args.channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            loss = criterion(net(imgs).float(), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        ms, mem = time_steps(step)
        rows.append((name, f"{ms:.0f}", f"{mem:.2f} GiB"))
        print(f"{name}: {ms:.0f} ms/step, peak {mem:.2f} GiB", flush=True)
        del model, net, optimizer, loader, dataset, batches
        torch.cuda.empty_cache()
    return ["| setting | ms/step | peak GPU memory |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]


def bench_synthetic(n_classes, batch, h, w, bf16, cudnn_benchmark):
    fp32_setup(cudnn_benchmark)
    torch.manual_seed(0)
    model = UNet(n_classes=n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    imgs = torch.randn(batch, 3, h, w, device="cuda")
    masks = torch.randint(0, n_classes, (batch, h, w), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(imgs)
        loss = torch.nn.functional.cross_entropy(out.float(), masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    desc = (f"{n_classes} classes, batch {batch}, {h}x{w}, {'bf16' if bf16 else 'fp32'}, "
            f"cudnn.benchmark {cudnn_benchmark}")
    try:
        ms, mem = time_steps(step)
        line = f"| {desc} | {ms:.0f} | {mem:.2f} GiB |"
    except torch.OutOfMemoryError:
        line = f"| {desc} | out of memory | - |"
    print(line)
    return [line]


def bench_confusion():
    labels = load_labels(os.path.join(ROOT, "data"))
    lut = id_to_trainid_lut(labels)
    mask_dir = os.path.join(ROOT, "data/data_semantics/training/semantic")
    gts = [torch.from_numpy(lut[np.array(Image.open(os.path.join(mask_dir, n)))]).long()
           for n in read_split(os.path.join(ROOT, "splits/val.txt"))]
    # a fake prediction with the right shape is enough for timing
    preds = [torch.roll(g, 7, dims=1) for g in gts]
    lines = []
    for device in ["cpu", "cuda"]:
        g = [x.to(device) for x in gts]
        p = [x.to(device) for x in preds]
        for _ in range(2):
            conf = new_confusion(19, device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for a, b in zip(g, p):
                update_confusion(conf, a, b)
            conf.sum().item()
            ms = (time.time() - t0) * 1000
        lines.append(f"| confusion matrix for 40 val images on {device} | {ms:.1f} ms |")
        print(lines[-1])
    return lines


def main():
    parser = argparse.ArgumentParser(description="Time the training speed options")
    parser.add_argument("mode", choices=["train", "synthetic", "confusion"])
    parser.add_argument("--classes", type=int, default=34)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--out", default=None, help="also write the table to this file")
    args, extra = parser.parse_known_args()

    if args.mode == "train":
        lines = bench_train(extra)
    elif args.mode == "synthetic":
        lines = bench_synthetic(args.classes, args.batch, args.height, args.width, args.bf16,
                                not args.no_cudnn_benchmark)
    else:
        lines = bench_confusion()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
