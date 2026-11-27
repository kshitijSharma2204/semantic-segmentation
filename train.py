import argparse
import csv
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import ROOT, add_common_args, load_labels
from evaluate import evaluate_split
from kitti_dataset import KittiSegmentationDataset
from make_splits import read_split
from metrics import IGNORE, id_to_trainid_lut
from model import UNet

N_CLASSES = 19
# settings that may change when resuming; everything else comes from the checkpoint
RUNTIME_KEYS = {"resume", "stop_after", "num_workers"}


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.reseed(info.seed % 2**32)


def make_loader(args, labels):
    img_dir = os.path.join(args.data_root, "data_semantics/training/image_2")
    mask_dir = os.path.join(args.data_root, "data_semantics/training/semantic")
    dataset = KittiSegmentationDataset(
        img_dir, mask_dir, read_split(args.train_split), id_to_trainid_lut(labels),
        size=(args.height, args.width), crop=(args.crop_height, args.crop_width),
        scale=(args.scale_min, args.scale_max), augment=True, cache=args.cache)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, worker_init_fn=seed_worker)
    return dataset, loader


def build_model(args, device):
    model = UNet(n_classes=N_CLASSES).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def build_optimizer(args, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  fused=args.fused_adam)
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        if args.schedule == "poly":
            return (1 - t) ** 0.9
        return 0.5 * (1 + math.cos(math.pi * t))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def class_weights(dataset):
    # log-inverse frequency, 1 / ln(1.02 + p) as in ENet, counted on the training split only
    counts = np.zeros(N_CLASSES)
    for i in range(len(dataset)):
        mask = dataset.cache[i][1] if dataset.cache is not None else dataset.load(dataset.images[i])[1]
        counts += np.bincount(mask[mask != IGNORE], minlength=N_CLASSES)[:N_CLASSES]
    freq = counts / counts.sum()
    return torch.tensor(1 / np.log(1.02 + freq), dtype=torch.float32)


def rng_state():
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # stored as a tensor so the checkpoint loads with weights_only=True
        "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64))] + list(np_state[2:]),
        "python": random.getstate(),
    }


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    random.setstate(state["python"])


def save_curves(history, path):
    epochs = [r["epoch"] for r in history]
    val = [r for r in history if r.get("val_miou") is not None]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(epochs, [r["train_loss"] for r in history], label="train loss")
    ax1.plot([r["epoch"] for r in val], [r["val_loss"] for r in val], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax2 = ax1.twinx()
    ax2.plot([r["epoch"] for r in val], [r["val_miou"] for r in val], color="tab:green", label="val mIoU")
    ax2.set_ylabel("val mIoU (19 classes)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.grid(alpha=0.3)
    plt.title("v2 training")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_log(history, path):
    keys = ["epoch", "train_loss", "lr", "epoch_s", "val_loss", "val_miou", "val_pixel_acc", "val_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})


def train_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    ckpt = None
    if args.resume:
        # map to cpu so the RNG states stay CPU tensors; the model and optimizer move themselves
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        for k, v in ckpt["config"].items():
            if k not in RUNTIME_KEYS:
                setattr(args, k, v)

    run_dir = os.path.join(args.out_root, args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    labels = load_labels(args.data_root)
    dataset, train_loader = make_loader(args, labels)
    val_names = read_split(args.val_split)
    print(f"Training samples: {len(dataset)}, validation samples: {len(val_names)}, "
          f"{len(train_loader)} steps per epoch")

    model = build_model(args, device)
    net = torch.compile(model) if args.compile else model
    optimizer, scheduler = build_optimizer(args, model, len(train_loader))
    weight = class_weights(dataset).to(device) if args.loss == "weighted_ce" else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE)

    start_epoch, best_miou, best_epoch, history = 1, -1.0, None, []
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou, best_epoch, history = ckpt["best_miou"], ckpt["best_epoch"], ckpt["history"]
        set_rng_state(ckpt["rng"])
        print(f"Resuming from {args.resume} at epoch {start_epoch}")

    def save(path, epoch):
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_miou": best_miou,
            "best_epoch": best_epoch, "config": vars(args), "history": history, "rng": rng_state(),
        }, path)

    end_epoch = args.epochs
    if args.stop_after:
        end_epoch = min(end_epoch, start_epoch + args.stop_after - 1)
    bar = tqdm(range(start_epoch, end_epoch + 1), desc="epochs", initial=start_epoch - 1, total=args.epochs)
    for epoch in bar:
        t0 = time.time()
        if args.num_workers == 0:
            dataset.reseed(int(torch.randint(2**31, ()).item()))
        model.train()
        loss_sum = torch.zeros((), device=device)
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = net(imgs)
            loss = criterion(out.float(), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()
        # loss is summed on the GPU and read once per epoch to avoid a sync every step
        row = {"epoch": epoch, "train_loss": round(loss_sum.item() / len(train_loader), 5),
               "lr": f"{scheduler.get_last_lr()[0]:.3e}", "epoch_s": round(time.time() - t0, 2)}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            model.eval()
            res = evaluate_split(model, "trainid19", (args.height, args.width), val_names,
                                 args.data_root, device, labels, progress=False)
            row.update({"val_loss": round(res["val_loss"], 5), "val_miou": round(res["miou"], 5),
                        "val_pixel_acc": round(res["pixel_acc"], 5), "val_s": round(time.time() - t1, 2)})
            if res["miou"] > best_miou:
                best_miou, best_epoch = res["miou"], epoch
                history.append(row)
                save(os.path.join(ckpt_dir, "best.pt"), epoch)
                history.pop()

        history.append(row)
        save(os.path.join(ckpt_dir, "last.pt"), epoch)
        write_log(history, os.path.join(run_dir, "log.csv"))
        save_curves(history, os.path.join(run_dir, "curves.png"))

        msg = f"epoch {epoch}: train loss {row['train_loss']:.4f}, lr {row['lr']}, {row['epoch_s']:.1f}s"
        if "val_miou" in row:
            msg += f" | val loss {row['val_loss']:.4f}, val mIoU {row['val_miou']:.4f} (best {best_miou:.4f} at {best_epoch})"
        tqdm.write(msg)
        bar.set_postfix(loss=row["train_loss"], best_miou=f"{best_miou:.4f}")

    print(f"Done. Best val mIoU {best_miou:.4f} at epoch {best_epoch}. Output in {run_dir}")


def parse_args(argv=None):
    on_off = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description="Train the UNet on the 19 KITTI/Cityscapes trainIds")
    add_common_args(parser)
    parser.add_argument("--run-name", default="train", help="output goes to <out-root>/<run-name>/")
    parser.add_argument("--train-split", default=os.path.join(ROOT, "splits/train.txt"))
    parser.add_argument("--val-split", default=os.path.join(ROOT, "splits/val.txt"))
    parser.add_argument("--height", type=int, default=352, help="every image is resized to this first")
    parser.add_argument("--width", type=int, default=1184)
    parser.add_argument("--crop-height", type=int, default=352, help="random training crop")
    parser.add_argument("--crop-width", type=int, default=592)
    parser.add_argument("--scale-min", type=float, default=0.75, help="random scale before the crop")
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate after warm-up")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--schedule", choices=["poly", "cosine"], default="poly")
    parser.add_argument("--loss", choices=["ce", "weighted_ce"], default="ce")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--resume", default=None, help="a last.pt to continue from")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after this many epochs in this run (the schedule still uses --epochs)")
    parser.add_argument("--cache", action=on_off, default=True, help="keep decoded images in RAM")
    parser.add_argument("--pin-memory", action=on_off, default=True)
    parser.add_argument("--channels-last", action=on_off, default=False)
    parser.add_argument("--compile", action=on_off, default=False, help="torch.compile the model")
    parser.add_argument("--fused-adam", action=on_off, default=False)
    parser.add_argument("--bf16", action=on_off, default=False, help="bf16 autocast (default is fp32)")
    args = parser.parse_args(argv)
    args.model_type = "trainid19"
    return args


if __name__ == '__main__':
    train_model(parse_args())
